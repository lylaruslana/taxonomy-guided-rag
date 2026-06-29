"""
analyze_intent_categories.py — IC3INA 2026
Klasifikasi intent otomatis (rule-based) per pertanyaan, lalu hitung
mapping intent → taxonomy category yang dominan.

Jalankan dari root project:
    python analyze_intent_categories.py

Output:
    intent_annotation.csv    — pertanyaan + intent + detected categories
    intent_category_map.json — mapping intent → allowed categories (threshold 20%)
"""

import sys, re, json, csv
from pathlib import Path
from collections import defaultdict, Counter

import openpyxl

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy_runtime import (
    load_taxonomy_concepts,
    load_synonym_index,
    detect_query_concepts,
)

ROOT          = Path(__file__).parent
TESTFILE      = ROOT / "data/data_test/qa-dokter-eg.xlsx"
CONCEPTS_PATH = ROOT / "data/taxonomy_concepts_v4.json"
SYNONYM_PATH  = ROOT / "data/synonym_index_v4.json"
OUT_CSV       = ROOT / "intent_annotation.csv"
OUT_MAP       = ROOT / "intent_category_map.json"

THRESHOLD = 0.20  # category masuk mapping jika muncul >= 20% dalam intent


# ---------------------------------------------------------------------------
# Aturan klasifikasi intent — urutan: LEBIH SPESIFIK dulu, LAIN paling akhir
# Setiap rule: (intent_label, [keyword/frasa], bobot)
# Skor tertinggi menang. Kata kunci diperiksa pada teks query yang dinormalisasi.
# ---------------------------------------------------------------------------
INTENT_RULES: list[tuple[str, list[str], float]] = [
    ("FASILITAS", [
        "fasilitas", "layanan", "rumah sakit", "puskesmas", "fktp", "klinik",
        "bpjs", "pembiayaan", "pendaftaran", "telemedicine", "rujukan",
        "unit hemodialisis", "iuran", "di mana saja", "tersedia",
    ], 1.0),

    ("NUTRISI", [
        "makanan", "diet", "asupan", "konsumsi", "makan", "minum", "nutrisi",
        "gizi", "protein", "cairan", "kalium", "fosfor", "natrium", "garam",
        "vitamin", "suplemen", "buah", "daging", "sayuran", "susu", "kopi",
        "teh", "alkohol", "air", "energi", "pantangan", "aman dikonsumsi",
        "sebaiknya dihindari", "makanan cepat saji",
    ], 1.0),

    ("KOMPLIKASI", [
        "komplikasi", "efek samping", "dampak", "risiko", "memengaruhi",
        "mempengaruhi", "komorbid", "penyakit penyerta", "lemas", "kelelahan",
        "pingsan", "infeksi", "anemia", "memperburuk", "mempersulit",
        "memperberat", "keluhan", "efektivitas", "kualitas hidup",
        "kesehatan mental", "depresi", "gangguan tulang", "gangguan jantung",
        "nyeri dada", "sesak napas", "bengkak",
    ], 1.0),

    ("MONITORING", [
        "frekuensi", "durasi", "kepatuhan", "jadwal", "seberapa sering",
        "berapa kali", "memantau", "pemantauan", "target", "nilai normal",
        "kadar normal", "angka", "seberapa lama",
    ], 1.0),

    ("TERAPI", [
        "terapi", "pengobatan", "penanganan", "tatalaksana", "inisiasi",
        "kapan mulai", "pilihan terapi", "modalitas", "manajemen",
        "rekomendasi terapi", "alternatif",
    ], 1.0),

    ("DIAGNOSIS", [
        "diagnos", "mendeteksi", "skrining", "cara mendiagnosis",
        "kriteria", "pemeriksaan", "stadium", "klasifikasi",
    ], 1.0),

    ("DEFINISI", [
        "apa itu", "pengertian", "definisi", "jelaskan", "apa yang dimaksud",
        "sebutkan jenis", "perbedaan", "apa perbedaan",
    ], 1.0),
]

# Mapping langsung dari Kategori_asal → intent (jika tidak ada ambiguitas)
KATEGORI_HINT: dict[str, str] = {
    "Faskes":   "FASILITAS",
    "Konsumsi": "NUTRISI",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def classify_intent(question: str, kategori_asal: str) -> tuple[str, dict[str, float]]:
    """Return (intent_label, {intent: score}) berdasarkan keyword rules."""
    # Hint langsung dari Kategori_asal — masih dikonfirmasi oleh keyword score
    hint = KATEGORI_HINT.get(str(kategori_asal).strip(), None)

    norm_q = _normalize(question)
    scores: dict[str, float] = defaultdict(float)

    for intent, keywords, weight in INTENT_RULES:
        for kw in keywords:
            if kw in norm_q:
                # Kata lebih panjang → sinyal lebih kuat
                scores[intent] += weight * (1 + len(kw.split()) * 0.2)

    if not scores:
        return hint or "LAIN", {}

    best_intent = max(scores, key=lambda k: scores[k])

    # Jika hint sangat kuat (Faskes/Konsumsi) dan skor-nya konsisten, pakai hint
    if hint and scores.get(hint, 0) >= scores[best_intent] * 0.5:
        return hint, dict(scores)

    return best_intent, dict(scores)


def _read_xlsx(path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) if h is not None else "" for h in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:] if any(v is not None for v in row)]


def main():
    print("Memuat taxonomy...")
    taxonomy_by_id = load_taxonomy_concepts(str(CONCEPTS_PATH))
    synonym_index  = load_synonym_index(str(SYNONYM_PATH))

    questions = [r for r in _read_xlsx(TESTFILE) if r.get("Pertanyaan")]
    print(f"Total pertanyaan: {len(questions)}\n")

    # ------------------------------------------------------------------
    # STEP 1 — Deteksi concept + klasifikasi intent
    # ------------------------------------------------------------------
    records = []
    intent_counter: Counter = Counter()
    cat_counter: Counter = Counter()
    no_concept_count = 0

    for row in questions:
        q   = str(row["Pertanyaan"]).strip()
        kat = str(row.get("Kategori", "")).strip()

        detected      = detect_query_concepts(q, synonym_index, taxonomy_by_id)
        categories    = sorted({d["category"] for d in detected})
        subcategories = sorted({d["subcategory"] for d in detected})
        concepts      = [d["preferred_term_id"] for d in detected]

        intent, scores = classify_intent(q, kat)
        intent_counter[intent] += 1
        if not detected:
            no_concept_count += 1
        for cat in categories:
            cat_counter[cat] += 1

        top_scores = ", ".join(
            f"{k}:{v:.2f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3]
        )

        records.append({
            "No":                    row.get("No", ""),
            "Kategori_asal":         kat,
            "Intent":                intent,
            "Pertanyaan":            q,
            "Detected_concepts":     "; ".join(concepts) if concepts else "(tidak ada)",
            "Detected_categories":   "; ".join(categories) if categories else "(tidak ada)",
            "Detected_subcategories":"; ".join(subcategories) if subcategories else "(tidak ada)",
            "N_concepts":            len(detected),
            "Intent_scores":         top_scores,
        })

    # Tulis CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV  → {OUT_CSV}")

    # Ringkasan distribusi intent
    print("\nDistribusi intent:")
    for intent, cnt in intent_counter.most_common():
        print(f"  {cnt:3d}x  {intent}")
    print(f"\nTanpa konsep terdeteksi: {no_concept_count} ({no_concept_count/len(records)*100:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 2 — Hitung mapping intent → category (threshold >= 20%)
    # ------------------------------------------------------------------
    intent_cat_counts: dict[str, Counter] = defaultdict(Counter)
    intent_total: Counter = Counter()

    for rec in records:
        intent = rec["Intent"]
        cats_raw = rec["Detected_categories"]
        intent_total[intent] += 1
        if cats_raw and cats_raw != "(tidak ada)":
            for cat in cats_raw.split(";"):
                cat = cat.strip()
                if cat:
                    intent_cat_counts[intent][cat] += 1

    print(f"\nMapping intent → category (threshold ≥ {THRESHOLD*100:.0f}%):\n")
    result = {}

    for intent in sorted(intent_total.keys()):
        total = intent_total[intent]
        cat_counts = intent_cat_counts[intent]
        allowed = []

        print(f"[{intent}] — {total} pertanyaan")
        if not cat_counts:
            print("  (tidak ada konsep terdeteksi)")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total
            flag = "✓" if pct >= THRESHOLD else " "
            print(f"  {flag} {pct*100:5.1f}%  ({cnt:2d}x)  {cat}")
            if pct >= THRESHOLD:
                allowed.append(cat)

        result[intent] = {
            "n_questions":       total,
            "allowed_categories": allowed,
            "category_counts":   dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        }
        print()

    # Tulis JSON
    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"JSON → {OUT_MAP}")

    # Ringkasan akhir
    print("\nRingkasan mapping:")
    for intent, data in sorted(result.items()):
        cats = ", ".join(data["allowed_categories"]) or "(semua — tidak ada dominan)"
        print(f"  {intent:15s} → {cats}")


if __name__ == "__main__":
    main()
