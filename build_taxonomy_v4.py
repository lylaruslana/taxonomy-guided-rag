#!/usr/bin/env python3
"""
build_taxonomy_v4.py — Konversi KDIGO2024_Taksonomi_Medis_v3.docx ke taxonomy v4.

Perbedaan dari v3:
  - Sinonim HANYA dari kolom "Nama Lain / Sinonim" (split by "·")
  - Tidak ada pemecahan / ekpansi dari teks dalam tanda kurung di preferred_term
  - Preferred_term ID dan EN tetap disertakan di synonym_index
  - Hasil bersih: tidak ada false-positive dari qualifier konteks seperti "(Ginjal / Dialisis)"

Output:
  data/taxonomy_concepts_v4.json   — list of 350 concepts
  data/synonym_index_v4.json       — {term: [concept_ids]} untuk runtime

Jalankan dari root project:
    python build_taxonomy_v4.py
"""

import re, json, sys, os
from pathlib import Path
import docx

_ROOT = Path(__file__).parent
DOCX_PATH    = str(_ROOT / "data/KDIGO2024_Taksonomi_Medis_v3.docx")
OUT_CONCEPTS = str(_ROOT / "data/taxonomy_concepts_v4.json")
OUT_INDEX    = str(_ROOT / "data/synonym_index_v4.json")

# ── Normalisasi (sama persis dengan taxonomy_runtime._normalize_text) ──────────
_TOKEN_RE   = re.compile(r"[^a-z0-9\s]+")
_PAREN_RE   = re.compile(r"\(([^)]+)\)")   # semua (...)
_ABBREV_RE  = re.compile(r"^[A-Z0-9/\-\.]{1,10}$")  # isi kurung = abbreviation

def normalize(text: str) -> str:
    t = str(text or "").lower().strip()
    t = _TOKEN_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_paren_terms(term: str) -> list[str]:
    """
    Dari preferred_term seperti "Hemodialisis (HD)" atau "PGK Stadium 5 / Gagal Ginjal Terminal":
    - Jika ada parenthetical yang berisi ABBREVIATION (≤10 char, semua uppercase/digit/slash):
        → tambahkan abbreviation sebagai sinonim terpisah
        → tambahkan base_term (tanpa parenthetical) sebagai sinonim terpisah
    - Jika parenthetical berisi QUALIFIER (ada lowercase / panjang):
        → abaikan, tidak dipecah

    Return: list of additional terms to add to synonym_index (di luar full term itu sendiri).
    """
    extras: list[str] = []

    # 1. Base term: hapus semua parenthetical
    base = _PAREN_RE.sub("", term).strip()
    base = re.sub(r"\s+", " ", base).strip()
    if base and base != term:
        extras.append(base)

    # 2. Split " / " pada base (atau term jika tidak ada kurung)
    # Menangkap: "Kelelahan / Fatigue", "Dialisis Peritoneal / CAPD / DPMB",
    #            "PGK Stadium 5 / Gagal Ginjal Terminal", dll.
    slash_source = base if base else term
    if " / " in slash_source:
        for part in slash_source.split(" / "):
            part = re.sub(r"\s+", " ", part).strip()
            if part and part not in extras and part != term:
                extras.append(part)

    # 3. Isi kurung: hanya jika abbreviation (HD, PGK, GGA, FSGS, dll.)
    for paren_content in _PAREN_RE.findall(term):
        content = paren_content.strip()
        if _ABBREV_RE.match(content) and content not in extras:
            extras.append(content)

    return extras

# ── Parse DOCX ────────────────────────────────────────────────────────────────
doc = docx.Document(DOCX_PATH)
table = doc.tables[0]

# Regex untuk deteksi nomor kategori dan subkategori dari header
CAT_RE    = re.compile(r"^\s*\S+\s+(\d+)\.\s+")   # "🫘  1.  KONDISI ..."
SUBCAT_RE = re.compile(r"^\s*(\d+)\.(\d+)\s+")     # "1.1  Penyakit ..."

current_cat_id    = 0
current_subcat_id = 0
current_cat_name  = ""
current_subcat_name = ""

concepts = []

for row in table.rows:
    vals = [cell.text.strip() for cell in row.cells]

    # Deteksi tipe baris
    unique_vals = set(vals)

    # ── Header merged (semua sel sama) ────────────────────────────────────────
    if len(unique_vals) == 1:
        text = vals[0]
        m_sub = SUBCAT_RE.match(text)
        m_cat = CAT_RE.match(text)
        if m_sub:
            # Subkategori: "1.1  Nama Subkategori"
            current_subcat_id = int(m_sub.group(2))
            # Nama subkategori: teks setelah "N.M  "
            current_subcat_name = SUBCAT_RE.sub("", text).strip()
        elif m_cat:
            # Kategori: "🫘  1.  NAMA KATEGORI"
            current_cat_id = int(m_cat.group(1))
            current_subcat_id = 0
            current_cat_name = CAT_RE.sub("", text).strip()
        continue

    # ── Header kolom (No, EN, ID, ...) ───────────────────────────────────────
    if vals[0] == "No":
        continue

    # ── Data concept ──────────────────────────────────────────────────────────
    item_no_raw = vals[0]
    if not item_no_raw or not item_no_raw[0].isdigit():
        continue

    try:
        item_no = int(re.match(r"\d+", item_no_raw).group())
    except (ValueError, AttributeError):
        continue

    en_term  = vals[1].strip() if len(vals) > 1 else ""
    id_term  = vals[2].strip() if len(vals) > 2 else ""
    sinonim  = vals[3].strip() if len(vals) > 3 else ""
    icd11    = vals[4].strip() if len(vals) > 4 else ""
    icd10    = vals[5].strip() if len(vals) > 5 else ""
    snomed   = vals[6].strip() if len(vals) > 6 else ""

    if not id_term and not en_term:
        continue

    # Concept ID
    concept_id = f"TAX_{current_cat_id:02d}_{current_subcat_id:02d}_{item_no:03d}"

    # ── Sinonim: hanya dari kolom "Nama Lain / Sinonim" ─────────────────────
    # Separator: · (U+00B7). Hapus "–", "-", string kosong, dan terlalu pendek.
    raw_syns = []
    if sinonim and sinonim not in ("–", "-", "—"):
        for s in sinonim.split("·"):
            s = s.strip()
            if s and s not in ("–", "-", "—") and len(s) > 1:
                raw_syns.append(s)

    # Daftar lengkap untuk synonym_index:
    # preferred_term ID + abbreviation dari kurung ID + base ID (tanpa kurung)
    # preferred_term EN + abbreviation dari kurung EN
    # sinonim dari kolom "Nama Lain / Sinonim"
    all_terms = []
    if id_term:
        all_terms.append(id_term)
        all_terms.extend(extract_paren_terms(id_term))
    if en_term and en_term != id_term:
        all_terms.append(en_term)
        all_terms.extend(extract_paren_terms(en_term))
    all_terms.extend(raw_syns)
    # Deduplicate sambil jaga urutan
    seen_terms: set[str] = set()
    deduped: list[str] = []
    for t in all_terms:
        if t not in seen_terms:
            seen_terms.add(t)
            deduped.append(t)
    all_terms = deduped

    concept = {
        "concept_id":       concept_id,
        "category_id":      str(current_cat_id),
        "category_name":    current_cat_name,
        "subcategory_id":   f"{current_cat_id}.{current_subcat_id}",
        "subcategory_name": current_subcat_name,
        "item_no":          item_no,
        "preferred_term_id": id_term,
        "preferred_term_en": en_term,
        "synonyms":         raw_syns,   # sinonim murni dari kolom, tanpa ID/EN term
        "all_index_terms":  all_terms,  # semua term yang masuk synonym_index
        "codes": {
            "icd_11":   icd11 or None,
            "icd_10":   icd10 or None,
            "snomed_ct": snomed or None,
        },
        "source": "KDIGO 2024 + PERNEFRI — KDIGO2024_Taksonomi_Medis_v3.docx",
    }
    concepts.append(concept)

print(f"Parsed {len(concepts)} concepts")

# ── Build synonym_index ───────────────────────────────────────────────────────
# Format: {normalized_term: [concept_id, ...]}
index: dict[str, list[str]] = {}

for c in concepts:
    cid = c["concept_id"]
    for term in c["all_index_terms"]:
        norm = normalize(term)
        if not norm or len(norm) < 2:
            continue
        if cid not in index.setdefault(norm, []):
            index[norm].append(cid)

print(f"Synonym index: {len(index)} entries")

# ── Statistik perbandingan ────────────────────────────────────────────────────
# Cek kasus problematik dari v3
for check in ["dialisis", "ginjal", "dialisis)", "ginjal / dialisis", "kaheksia"]:
    hits = index.get(check, [])
    print(f"  '{check}' → {hits if hits else '(tidak ada entry)'}")

# ── Simpan output ─────────────────────────────────────────────────────────────
meta = {
    "title":           "Taksonomi Istilah Medis KDIGO 2024 CKD Guideline + PERNEFRI — v4",
    "source_file":     os.path.basename(DOCX_PATH),
    "version":         "v4",
    "build_note":      "Sinonim hanya dari kolom 'Nama Lain / Sinonim'. Tidak ada pemecahan parenthetical qualifier.",
    "concept_count":   len(concepts),
    "synonym_count":   sum(len(c["synonyms"]) for c in concepts),
    "index_entry_count": len(index),
}

with open(OUT_CONCEPTS, "w", encoding="utf-8") as f:
    json.dump({"metadata": meta, "concepts": concepts}, f, ensure_ascii=False, indent=2)

with open(OUT_INDEX, "w", encoding="utf-8") as f:
    json.dump({"metadata": meta, "index": index}, f, ensure_ascii=False, indent=2)

print(f"\n[SAVED] {OUT_CONCEPTS}")
print(f"[SAVED] {OUT_INDEX}")
print(f"\nMetadata: {meta}")
