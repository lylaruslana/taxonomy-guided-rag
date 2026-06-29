"""
analyze_node_quality.py — IC3INA 2026
Analisis kualitas node retrieval dari taxonomy_trace.jsonl per skenario.
Mengukur: n_candidates, vector_score, ce_score (derived), taxonomy_score, rank_delta.

Jalankan: python analyze_node_quality.py
"""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path("/home/lyla001/CHATBOT/results")

# Skenario paper IC3INA — label baru → folder
SCENARIOS = [
    ("BASE",       "D003D-bgem3-sahabatai-semantic-vsi",              False, 0.0, 0.0),
    ("CE",         "D003D-CE-NOQEXP-bgem3-sahabatai-semantic",        True,  0.80, 0.20),
    ("QEXP-CE",    "D003D-QEXP-CE-CLEAN-V4-bgem3-sahabatai-semantic", True,  0.80, 0.20),
    ("QRW-CE",     "D003D-QREWRITE-CE-CLEAN-V4-bgem3-sahabatai-semantic", True, 0.80, 0.20),
    ("QEXP-CE-T-O",  "D003D-TAX-CE-CLEAN-V4-bgem3-sahabatai-semantic",  True,  0.70, 0.20),
    ("QEXP-BMT",   "D003D-TAX-QEXP-CLEAN-V4-bgem3-sahabatai-semantic", False, 0.0, 0.0),
    ("QEXP-BMT-O", "D003D-TAX-QEXP-OPT-CLEAN-bgem3-sahabatai-semantic", False, 0.0, 0.0),
    ("QRW-BMT",    "D003D-QREWRITE-TAX-CLEAN-V4-bgem3-sahabatai-semantic", False, 0.0, 0.0),
]
# (label, folder, has_ce, w_v, w_ce)

# RAGAS mean results (dari Table 2 paper)
RAGAS = {
    "BASE":       (0.8994, 0.8937, 0.8026),
    "CE":         (0.8960, 0.9089, 0.8699),
    "QEXP-CE":    (0.8859, 0.9026, 0.8573),
    "QRW-CE":     (0.9019, 0.9160, 0.8784),
    "QEXP-CE-T-O":  (0.8512, 0.8952, 0.8428),
    "QEXP-BMT":   (0.7747, 0.8532, 0.6737),
    "QEXP-BMT-O": (0.7864, 0.8621, 0.7321),
    "QRW-BMT":    (0.8132, 0.8857, 0.6846),
}


def load_trace(folder):
    for ext in ("taxonomy_trace.jsonl", "taxonomy_trace.json"):
        p = RESULTS_DIR / folder / ext
        if p.exists():
            with open(p) as f:
                lines = f.readlines()
            # try JSONL first
            try:
                return [json.loads(l) for l in lines if l.strip()]
            except json.JSONDecodeError:
                pass
    return None


def derive_ce_score(final_score, vector_score, w_v, w_ce, w_tau, taxonomy_score):
    # final = w_v*v + w_ce*ce + w_tau*tau  →  ce = (final - w_v*v - w_tau*tau) / w_ce
    if w_ce <= 0:
        return None
    ce = (final_score - w_v * vector_score - w_tau * taxonomy_score) / w_ce
    return max(0.0, min(1.0, ce))


print("=" * 95)
print(f"{'Scenario':<12} {'n_cand':>6} {'vec_score':>9} {'ce_score':>9} {'tax_score':>10} "
      f"{'rank_Δ':>7} {'Faith':>7} {'AnsRel':>7} {'CtxPre':>7}")
print("-" * 95)

for label, folder, has_ce, w_v, w_ce in SCENARIOS:
    trace = load_trace(folder)
    if trace is None:
        print(f"{label:<12}  [no trace]")
        continue

    n_cands, vec_scores, ce_scores, tax_scores, rank_deltas = [], [], [], [], []

    for entry in trace:
        n_cands.append(entry.get("n_candidates_before_rerank", 0))

        chunks = entry.get("chunk_scores_after_rerank", [])
        w_tau = 0.10 if label == "QEXP-CE-T-O" else (0.30 if "BMT" in label else 0.0)

        for c in chunks:
            v = c.get("vector_score", 0.0)
            t = c.get("taxonomy_score", 0.0)
            fs = c.get("final_score", 0.0)
            vec_scores.append(v)
            tax_scores.append(t)
            if has_ce and w_ce > 0:
                ce = derive_ce_score(fs, v, w_v, w_ce, w_tau, t)
                if ce is not None:
                    ce_scores.append(ce)

        for r in entry.get("rank_changes", []):
            rank_deltas.append(abs(r.get("delta", 0)))

    mean_n    = statistics.mean(n_cands) if n_cands else 0
    mean_vec  = statistics.mean(vec_scores) if vec_scores else 0
    mean_ce   = statistics.mean(ce_scores) if ce_scores else float("nan")
    mean_tax  = statistics.mean(tax_scores) if tax_scores else 0
    mean_rdel = statistics.mean(rank_deltas) if rank_deltas else 0

    faith, ansrel, ctxpre = RAGAS.get(label, (0, 0, 0))

    ce_str = f"{mean_ce:>9.4f}" if ce_scores else f"{'—':>9}"
    print(f"{label:<12} {mean_n:>6.1f} {mean_vec:>9.4f} {ce_str} {mean_tax:>10.4f} "
          f"{mean_rdel:>7.2f} {faith:>7.4f} {ansrel:>7.4f} {ctxpre:>7.4f}")

print("=" * 95)
print()

# --- Per-query breakdown: query mana yang PALING BERBEDA antara QRW-CE vs QEXP-CE ---
print("=" * 70)
print("Per-query: QRW-CE vs QEXP-CE — 10 query dengan gap Context Precision terbesar")
print("(menggunakan ce_score rata-rata top-5 nodes sebagai proxy kualitas node)")
print("-" * 70)

trace_qrw  = load_trace("D003D-QREWRITE-CE-CLEAN-V4-bgem3-sahabatai-semantic")
trace_qexp = load_trace("D003D-QEXP-CE-CLEAN-V4-bgem3-sahabatai-semantic")

if trace_qrw and trace_qexp:
    # Group by question_idx, average ce_score across iterations
    from collections import defaultdict

    def mean_ce_per_query(trace, w_v=0.80, w_ce=0.20, w_tau=0.0):
        per_q = defaultdict(list)
        for entry in trace:
            qidx = entry.get("question_idx")
            chunks = entry.get("chunk_scores_after_rerank", [])
            ces = []
            for c in chunks:
                v  = c.get("vector_score", 0.0)
                t  = c.get("taxonomy_score", 0.0)
                fs = c.get("final_score", 0.0)
                ce = derive_ce_score(fs, v, w_v, w_ce, w_tau, t)
                if ce is not None:
                    ces.append(ce)
            if ces:
                per_q[qidx].append(statistics.mean(ces))
        return {q: statistics.mean(vs) for q, vs in per_q.items()}

    def mean_ncand_per_query(trace):
        per_q = defaultdict(list)
        for entry in trace:
            qidx = entry.get("question_idx")
            per_q[qidx].append(entry.get("n_candidates_before_rerank", 0))
        return {q: statistics.mean(vs) for q, vs in per_q.items()}

    def first_question(trace, qidx):
        for e in trace:
            if e.get("question_idx") == qidx:
                return e.get("question", "")
        return ""

    ce_qrw  = mean_ce_per_query(trace_qrw)
    ce_qexp = mean_ce_per_query(trace_qexp)
    nc_qrw  = mean_ncand_per_query(trace_qrw)
    nc_qexp = mean_ncand_per_query(trace_qexp)

    common = sorted(set(ce_qrw) & set(ce_qexp))
    gaps = [(q, ce_qrw[q] - ce_qexp[q]) for q in common]
    gaps.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Q#':>4} {'gap_ce':>8} {'ce_QRW':>8} {'ce_QEXP':>8} {'nc_QRW':>8} {'nc_QEXP':>8}  Question[:60]")
    print("-" * 70)
    for qidx, gap in gaps[:10]:
        q_text = first_question(trace_qrw, qidx)[:60]
        print(f"{qidx:>4} {gap:>+8.4f} {ce_qrw[qidx]:>8.4f} {ce_qexp[qidx]:>8.4f} "
              f"{nc_qrw.get(qidx,0):>8.1f} {nc_qexp.get(qidx,0):>8.1f}  {q_text}")

print("=" * 70)
print()

# --- Distribusi rank_changes: CE mengubah urutan berapa banyak? ---
print("=" * 55)
print("Distribusi |rank_delta| top-5 nodes: CE scenarios")
print(f"{'Scenario':<12} {'delta=0':>8} {'delta 1-2':>10} {'delta 3+':>9} {'mean':>7}")
print("-" * 55)

for label, folder, has_ce, w_v, w_ce in SCENARIOS:
    if not has_ce:
        continue
    trace = load_trace(folder)
    if not trace:
        continue
    deltas = []
    for entry in trace:
        for r in entry.get("rank_changes", []):
            deltas.append(abs(r.get("delta", 0)))
    if not deltas:
        continue
    n = len(deltas)
    d0   = sum(1 for d in deltas if d == 0) / n * 100
    d12  = sum(1 for d in deltas if 1 <= d <= 2) / n * 100
    d3p  = sum(1 for d in deltas if d >= 3) / n * 100
    mean = statistics.mean(deltas)
    print(f"{label:<12} {d0:>7.1f}% {d12:>9.1f}% {d3p:>8.1f}% {mean:>7.2f}")

print("=" * 55)
