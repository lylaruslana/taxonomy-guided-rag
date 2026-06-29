"""
ragas_evaluate.py — IC3INA 2026
Evaluasi RAGAS dengan pilihan metrik: faithfulness, answer_relevancy,
context_precision, context_recall.

Output:
  ragas_results/<config>.csv      → faithfulness, answer_relevancy, context_precision
  ragas_results/<config>_gt.csv   → context_recall (membutuhkan ground truth)

Jika keduanya dipilih dalam satu run, kedua file disimpan sekaligus.

Jalankan:
  # Semua 4 metrik:
  python ragas_evaluate.py --metrics faithfulness answer_relevancy context_precision context_recall

  # Hanya reference-free:
  python ragas_evaluate.py --metrics faithfulness answer_relevancy context_precision

  # Hanya context_recall:
  python ragas_evaluate.py --metrics context_recall

  # Skenario tertentu saja:
  python ragas_evaluate.py --metrics faithfulness context_recall \\
      --configs D003D-CE-NOQEXP-bgem3-sahabatai-semantic

Konfigurasi (env var atau argumen CLI):
  --api-key      OpenRouter API key  [env: OPENROUTER_API_KEY]
  --api-base     API base URL        [env: OPENROUTER_API_BASE]
  --model        Nama LLM model      [env: OPENROUTER_MODEL]
  --embed-model  Path embedding model lokal
  --embed-device cpu / cuda          (default: cpu)
  --input-dir    Direktori file JSON input
  --output-dir   Direktori output CSV
  --configs      Satu atau lebih nama config (default: semua file di input-dir)
  --batch-size   Batch size evaluasi (default: 20)
  --max-samples  Batasi jumlah sampel per config (0 = semua)
"""

import argparse
import ast
import csv
import json
import os
import statistics
import sys
import time
import traceback

csv.field_size_limit(sys.maxsize)
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
VALID_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
REF_FREE      = {"faithfulness", "answer_relevancy", "context_precision"}
NEEDS_GT      = {"context_recall"}

parser = argparse.ArgumentParser(description="Evaluasi RAGAS 4 metrik")
parser.add_argument("--metrics", nargs="+", choices=VALID_METRICS,
                    default=["faithfulness", "answer_relevancy", "context_precision"],
                    metavar="METRIC",
                    help=f"Metrik yang dievaluasi. Pilihan: {VALID_METRICS}")
parser.add_argument("--api-key",    default=os.environ.get("OPENROUTER_API_KEY", ""),
                    help="OpenRouter API key")
parser.add_argument("--api-base",   default=os.environ.get("OPENROUTER_API_BASE",
                                    "https://openrouter.ai/api/v1"))
parser.add_argument("--model",      default=os.environ.get("OPENROUTER_MODEL",
                                    "deepseek/deepseek-v4-flash"))
parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL_NAME",
                                    "BAAI/bge-m3"),
                    help="Path lokal atau nama HuggingFace embedding model")
parser.add_argument("--embed-device", default="cpu")
parser.add_argument("--input-dir",  default=os.environ.get("RAGAS_INPUT_DIR",
                                    "ragas_input"),
                    help="Direktori berisi file JSON input (satu file per config)")
parser.add_argument("--output-dir", default=os.environ.get("RAGAS_OUTPUT_DIR",
                                    "ragas_results"))
parser.add_argument("--configs",    nargs="*", default=None,
                    help="Nama config yang diproses (tanpa .json). Default: semua.")
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--max-samples", type=int, default=0,
                    help="Batasi sampel per config (0 = semua)")
args = parser.parse_args()

if not args.api_key:
    sys.exit("[ERROR] API key tidak ditemukan. Gunakan --api-key atau env OPENROUTER_API_KEY")

selected_metrics = set(args.metrics)
run_ref_free = bool(selected_metrics & REF_FREE)
run_gt       = bool(selected_metrics & NEEDS_GT)

# ---------------------------------------------------------------------------
# Import RAGAS metric objects
# ---------------------------------------------------------------------------
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness     as _faithfulness,
    answer_relevancy as _answer_relevancy,
    context_precision as _context_precision,
    context_recall   as _context_recall,
)

METRIC_OBJ = {
    "faithfulness":      _faithfulness,
    "answer_relevancy":  _answer_relevancy,
    "context_precision": _context_precision,
    "context_recall":    _context_recall,
}

# ---------------------------------------------------------------------------
# Setup LLM & embedding
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model=args.model,
    openai_api_key=args.api_key,
    openai_api_base=args.api_base,
    temperature=0.0,
    max_retries=3,
    timeout=300,
    max_tokens=8192,
    default_headers={
        "HTTP-Referer": "https://github.com/lylaruslana/taxonomy-guided-rag",
        "X-Title": "RAG RAGAS Eval",
    },
)
print(f"[INFO] LLM      : {args.model}")

embed_model = HuggingFaceEmbeddings(
    model_name=args.embed_model,
    model_kwargs={"device": args.embed_device},
    encode_kwargs={"normalize_embeddings": True},
)
print(f"[INFO] Embedding: {args.embed_model} ({args.embed_device})")
print(f"[INFO] Metrik   : {sorted(selected_metrics)}")

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def parse_contexts(val):
    if not val or (isinstance(val, str) and val.strip() in ("", "[]")):
        return [""]
    if isinstance(val, list):
        return [str(c) for c in val] or [""]
    for parser_fn in (json.loads, ast.literal_eval):
        try:
            parsed = parser_fn(val)
            if isinstance(parsed, list):
                return [str(c) for c in parsed] or [""]
        except Exception:
            pass
    return [str(val)]


def evaluate_config(config_name: str, json_path: str):
    print(f"\n{'='*60}")
    print(f"[CONFIG] {config_name}")
    print(f"{'='*60}")

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    if args.max_samples > 0:
        records = records[:args.max_samples]
    print(f"[INFO] {len(records)} records")

    questions, answers, contexts_list, references, metas = [], [], [], [], []
    for r in records:
        questions.append(r["user_input"])
        answers.append(r["response"])
        ctxs = r.get("retrieved_contexts", [])
        contexts_list.append(parse_contexts(ctxs) if isinstance(ctxs, str) else (ctxs or [""]))
        references.append(r.get("reference", ""))
        metas.append({
            "uuid":        r.get("uuid", ""),
            "question_no": r.get("question_no", ""),
            "repeat_no":   r.get("repeat_no", ""),
        })

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": references,
    })

    metric_objs = [METRIC_OBJ[m] for m in args.metrics]

    t0 = time.time()
    try:
        result = evaluate(
            dataset,
            metrics=metric_objs,
            llm=llm,
            embeddings=embed_model,
            raise_exceptions=False,
            batch_size=args.batch_size,
        )
        print(f"[INFO] Selesai dalam {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return None

    df = result.to_pandas()
    for key in ["uuid", "question_no", "repeat_no"]:
        df[key] = [m[key] for m in metas]
    df["retrieved_contexts_raw"] = [json.dumps(c, ensure_ascii=False) for c in contexts_list]
    df["reference"] = references

    summary = {}

    # ---- Simpan reference-free metrics → <config>.csv ----
    ref_free_cols = [m for m in ["faithfulness", "answer_relevancy", "context_precision"]
                     if m in df.columns]
    if ref_free_cols:
        front = ["uuid", "question_no", "repeat_no", "question", "answer",
                 "retrieved_contexts_raw", "reference"] + ref_free_cols
        front = [c for c in front if c in df.columns]
        other = [c for c in df.columns if c not in front and c not in
                 ["context_recall", "retrieved_contexts_raw"] + front]
        out_df = df[front + other]
        csv_out = os.path.join(args.output_dir, f"{config_name}.csv")
        out_df.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"[SAVED] {csv_out}")
        for col in ref_free_cols:
            vals = df[col].dropna().tolist()
            if vals:
                mean_val = statistics.mean(vals)
                summary[col] = mean_val
                print(f"  {col}: mean={mean_val:.4f}  n={len(vals)}  nan={len(df)-len(vals)}")

    # ---- Simpan context_recall → <config>_gt.csv ----
    if "context_recall" in df.columns:
        gt_cols = ["question_no", "repeat_no", "question", "answer",
                   "retrieved_contexts_raw", "reference", "context_recall",
                   "uuid"]
        gt_cols = [c for c in gt_cols if c in df.columns]
        gt_df = df[gt_cols].rename(columns={
            "question": "user_input",
            "answer": "response",
            "retrieved_contexts_raw": "retrieved_contexts",
        })
        gt_out = os.path.join(args.output_dir, f"{config_name}_gt.csv")
        gt_df.to_csv(gt_out, index=False, encoding="utf-8")
        print(f"[SAVED] {gt_out}")
        vals = df["context_recall"].dropna().tolist()
        if vals:
            mean_val = statistics.mean(vals)
            summary["context_recall"] = mean_val
            print(f"  context_recall: mean={mean_val:.4f}  n={len(vals)}  nan={len(df)-len(vals)}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
json_files = sorted(f for f in os.listdir(args.input_dir)
                    if f.endswith(".json") and "_ragas_input" not in f)

if args.configs:
    json_files = [f for f in json_files
                  if any(f.startswith(c) for c in args.configs)]

if not json_files:
    sys.exit(f"[ERROR] Tidak ada file JSON di {args.input_dir} yang cocok.")

print(f"\n[INFO] {len(json_files)} config akan diproses:")
for f in json_files:
    print(f"  {f}")

all_summary = {}
for fname in json_files:
    config_name = fname.replace(".json", "")
    result = evaluate_config(config_name, os.path.join(args.input_dir, fname))
    if result:
        all_summary[config_name] = result

# ---- Summary ----
if all_summary:
    metric_cols = sorted({m for v in all_summary.values() for m in v})
    header = f"{'Config':<52}" + "".join(f"  {m[:10]:>10}" for m in metric_cols)
    print(f"\n{'='*len(header)}")
    print("SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for cfg, vals in all_summary.items():
        row = f"{cfg[:52]:<52}"
        for m in metric_cols:
            v = vals.get(m, float("nan"))
            row += f"  {v:>10.4f}"
        print(row)
    print(f"{'='*len(header)}")
    print(f"\n[DONE] Hasil tersimpan di: {args.output_dir}")
