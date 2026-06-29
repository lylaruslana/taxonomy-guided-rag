"""
run_evaluation.py — IC3INA 2026
Jalankan extract + RAGAS evaluation dalam satu perintah.

Cara pakai:
    # Foreground (lihat output langsung)
    python run_evaluation.py

    # Background + log
    nohup python run_evaluation.py > logs/ragas_run.log 2>&1 &
    tail -f logs/ragas_run.log

Konfigurasi API key via environment variable:
    export OPENROUTER_API_KEY=sk-or-v1-...
    export OPENROUTER_MODEL=deepseek/deepseek-v4-flash   # opsional
    export EMBED_MODEL_NAME=/path/to/local/bge-m3        # opsional

Hanya edit bagian KONFIGURASI di bawah, tidak perlu sentuh script lain.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ===========================================================================
# KONFIGURASI — edit di sini saja
# ===========================================================================

SELECTED_CONFIGS = [
    "D003D-bgem3-sahabatai-semantic-vsi",
    "D003D-CE-NOQEXP-bgem3-sahabatai-semantic",
    "D003D-QEXP-CE-CLEAN-V4-bgem3-sahabatai-semantic",
    "D003D-QREWRITE-CE-CLEAN-V4-bgem3-sahabatai-semantic",
    "D003D-TAX-CE-CLEAN-V4-bgem3-sahabatai-semantic",
    "D003D-TAX-QEXP-CLEAN-V4-bgem3-sahabatai-semantic",
    "D003D-TAX-QEXP-OPT-CLEAN-bgem3-sahabatai-semantic",
    "D003D-QREWRITE-TAX-CLEAN-V4-bgem3-sahabatai-semantic",
]

# Metrik yang dievaluasi — pilih sesuai kebutuhan:
#   "faithfulness", "answer_relevancy", "context_precision", "context_recall"
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

INPUT_DIR  = "ragas_input"    # direktori JSON input (output extract_ragas_input_taxonomy.py)
OUTPUT_DIR = "ragas_results"  # direktori output CSV

# ===========================================================================
# JALANKAN
# ===========================================================================

ROOT      = Path(__file__).parent
PYTHON    = sys.executable
RAGAS_DIR = ROOT / OUTPUT_DIR


def _header(title: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}", flush=True)
    print(f"[{ts}] {title}", flush=True)
    print(f"{'='*60}\n", flush=True)


def already_evaluated(config: str) -> bool:
    result_file = RAGAS_DIR / f"{config}.csv"
    return result_file.exists() and result_file.stat().st_size > 0


def check_configs() -> tuple[list[str], list[str]]:
    to_run, done = [], []
    for c in SELECTED_CONFIGS:
        if already_evaluated(c):
            done.append(c)
        else:
            to_run.append(c)
    return to_run, done


def run_extract(configs: list[str]) -> bool:
    _header("STEP: EXTRACT")
    env = os.environ.copy()
    result = subprocess.run(
        [PYTHON, "extract_ragas_input_taxonomy.py"] + configs,
        cwd=str(ROOT),
        env=env,
    )
    if result.returncode != 0:
        print(f"\n[ABORT] EXTRACT gagal (exit code {result.returncode})", flush=True)
    return result.returncode == 0


def run_ragas(configs: list[str]) -> bool:
    _header("STEP: RAGAS EVALUATION")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY tidak di-set. Export dulu sebelum menjalankan.", flush=True)
        return False

    cmd = [
        PYTHON, "ragas_evaluate.py",
        "--metrics", *METRICS,
        "--configs", *configs,
        "--input-dir", INPUT_DIR,
        "--output-dir", OUTPUT_DIR,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    if result.returncode != 0:
        print(f"\n[ABORT] RAGAS gagal (exit code {result.returncode})", flush=True)
    return result.returncode == 0


def main():
    _header(f"run_evaluation.py — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    to_run, done = check_configs()

    if done:
        print("[SKIP] Config berikut sudah pernah dievaluasi — dilewati:", flush=True)
        for c in done:
            result_file = RAGAS_DIR / f"{c}.csv"
            mtime = datetime.fromtimestamp(result_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  ✓ {c}  ({mtime})", flush=True)

    if not to_run:
        print(f"\n[INFO] Semua config sudah dievaluasi.", flush=True)
        print(f"       Hapus file di {OUTPUT_DIR}/ jika ingin re-evaluasi.", flush=True)
        return

    print(f"\nConfig yang akan dijalankan ({len(to_run)}):", flush=True)
    for c in to_run:
        print(f"  - {c}", flush=True)

    if not run_extract(to_run):
        sys.exit(1)

    if not run_ragas(to_run):
        sys.exit(1)

    _header("SELESAI")
    print(f"Selesai: {len(to_run)} config dievaluasi.", flush=True)
    if done:
        print(f"Dilewati: {len(done)} config (sudah ada hasilnya).", flush=True)


if __name__ == "__main__":
    main()
