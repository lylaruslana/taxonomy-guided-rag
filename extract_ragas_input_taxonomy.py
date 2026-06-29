"""
Ekstrak data input RAGAS dari hasil taxonomy query-eval.
Output rag_llamaindex_taxonomy.py ada di: results/{nama_model}/responses.txt + nodes.txt
Output script ini: ragas_input_ic3ina/{nama_model}.json  (format identik baseline)

Jalankan SETELAH rag_llamaindex_taxonomy.py selesai, dari root project:
    python3 extract_ragas_input_taxonomy.py
"""

import os
import re
import json
import csv
from collections import defaultdict
import pandas as pd

QA_FILE    = 'data/data_test/qa-dokter-eg.xlsx'
RESULTS_BASE = 'results'
OUT_DIR    = 'ragas_input_ic3ina'
MAX_REPEAT = 10

TAX_CONFIGS = [
    'D003D-TAX-bgem3-sahabatai-semantic',
    'D003D-QEXP-bgem3-sahabatai-semantic',
    'D003D-TAX-QEXP-bgem3-sahabatai-semantic',
    'D003D-TAGONLY-bgem3-sahabatai-semantic',
    'D001D-TAX-bgem3-llama-semantic',
    'D001D-TAGONLY-bgem3-llama-semantic',
]
# Override via env var (diset oleh run_evaluation.py)
if os.environ.get('EVAL_CONFIGS'):
    import json as _json
    TAX_CONFIGS = _json.loads(os.environ['EVAL_CONFIGS'])

os.makedirs(OUT_DIR, exist_ok=True)

# Load QA reference
qa_df  = pd.read_excel(QA_FILE, engine='openpyxl')
no_col  = next(c for c in qa_df.columns if str(c).strip().lower() == 'no')
q_col   = next(c for c in qa_df.columns if 'pertanyaan' in str(c).lower())
ans_col = next(c for c in qa_df.columns if 'jawaban' in str(c).lower())

qa_map = {}
for _, row in qa_df.iterrows():
    no = int(pd.to_numeric(row[no_col], errors='coerce'))
    qa_map[no] = (str(row[q_col]), str(row[ans_col]))

print(f"[INFO] Loaded {len(qa_map)} Q&A references")


def parse_nodes(nodes_path):
    uuid_to_ctx = defaultdict(list)
    with open(nodes_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 3)
            if len(parts) < 4:
                continue
            uuid = parts[0].strip()
            texts = re.findall(
                r"text='(.*?)'(?:,\s*start_char_idx|,\s*end_char_idx|\))",
                parts[3], re.DOTALL
            )
            if not texts:
                texts = re.findall(r"text='(.*?)'", parts[3], re.DOTALL)
            for t in texts:
                cleaned = t.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()
                if cleaned:
                    uuid_to_ctx[uuid].append(cleaned)
    return uuid_to_ctx


summary = []

for config in TAX_CONFIGS:
    resp_path  = os.path.join(RESULTS_BASE, config, 'responses.txt')
    nodes_path = os.path.join(RESULTS_BASE, config, 'nodes.txt')

    if not os.path.exists(resp_path) or not os.path.exists(nodes_path):
        print(f"[SKIP] {config}: file belum ada (query-eval belum selesai?)")
        continue

    if os.path.getsize(resp_path) == 0:
        print(f"[SKIP] {config}: responses.txt kosong")
        continue

    print(f"\n[PROCESS] {config}")
    uuid_to_ctx = parse_nodes(nodes_path)
    print(f"  contexts: {len(uuid_to_ctx)} uuids")

    records = []
    skipped_ref = 0
    with open(resp_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            uuid      = parts[0].strip()
            q_no      = int(parts[1])
            rep_no    = int(parts[2])
            # parts[3] = question text (redundant, ambil dari qa_map untuk konsistensi)
            response  = parts[4].encode().decode('unicode_escape') if '\\u' in parts[4] or '\\n' in parts[4] else parts[4]

            if rep_no > MAX_REPEAT:
                continue

            ref_data = qa_map.get(q_no)
            if ref_data is None:
                skipped_ref += 1
                continue

            question, reference = ref_data
            contexts = uuid_to_ctx.get(uuid, [])

            records.append({
                'uuid':               uuid,
                'question_no':        q_no,
                'repeat_no':          rep_no,
                'user_input':         question,
                'response':           response,
                'retrieved_contexts': contexts,
                'reference':          reference,
            })

    print(f"  records: {len(records)}, skipped (no ref): {skipped_ref}")
    q_set = set(r['question_no'] for r in records)
    r_set = set(r['repeat_no'] for r in records)
    print(f"  questions: {len(q_set)}, repeats: {sorted(r_set)}")

    json_path = os.path.join(OUT_DIR, f"{config}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {json_path}")

    csv_path = os.path.join(OUT_DIR, f"{config}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'uuid','question_no','repeat_no',
            'user_input','response','retrieved_contexts','reference'
        ])
        writer.writeheader()
        for r in records:
            row = dict(r)
            row['retrieved_contexts'] = ' <CTX> '.join(r['retrieved_contexts'])
            writer.writerow(row)
    print(f"  [SAVED] {csv_path}")

    summary.append({'config': config, 'n': len(records), 'questions': len(q_set)})

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if summary:
    for s in summary:
        print(f"  {s['config']}: {s['n']} rows, {s['questions']} pertanyaan")
else:
    print("  (tidak ada config yang diproses — query-eval belum selesai?)")
print(f"\nOutput: {OUT_DIR}/")
