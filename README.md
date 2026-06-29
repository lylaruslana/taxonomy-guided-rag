# Taxonomy-Guided RAG for Indonesian Hemodialysis QA

Source code for the paper:  
**"Taxonomy-Guided Query Rewriting for Indonesian Hemodialysis RAG"**  
IC3INA 2026

🔗 https://github.com/lylaruslana/taxonomy-guided-rag

---

## Overview

This repository implements a taxonomy-guided Retrieval-Augmented Generation (RAG) pipeline for Indonesian hemodialysis patient education. The system augments standard dense retrieval with a custom nephrology taxonomy to support query expansion, query rewriting, and concept-aware reranking.

**Best result:** Query rewriting + cross-encoder reranking (QRW-CE) achieves Faithfulness 0.9019, Answer Relevancy 0.9160, Context Precision 0.8784.

---

## Repository Structure

```
├── taxonomy_runtime.py          # Core: concept detection, tagging, reranking
├── rag_llamaindex_taxonomy.py   # Main RAG pipeline (LlamaIndex + ChromaDB)
├── build_taxonomy_v4.py         # Taxonomy construction from source documents
├── run_evaluation.py            # Experiment runner
├── grid_search_weights.py       # Grid search: BM25+Taxonomy weights (QEXP-BMT-O)
├── grid_search_ce_weights.py    # Grid search: CE weights (QEXP-CE-T-O)
├── analyze_node_quality.py      # Retrieval trace analysis (n_candidates, vec/CE score)
├── analyze_intent_categories.py # Query intent and concept detection analysis
├── plot_architecture.py         # Figure 1: system architecture
├── plot_line_iterations.py      # Figure 2: RAGAS metrics across iterations
├── config_paper.yaml            # Configurations for all 8 ablation scenarios
├── data/
│   ├── taxonomy_concepts_v4.json  # 350 nephrology concepts (not included — see Taxonomy section)
│   └── synonym_index_v4.json      # 1,841 surface forms (not included — see Taxonomy section)
└── figures/
    ├── architecture.png/pdf
    ├── bar_context_precision.png/pdf
    └── line_iterations.png/pdf
```

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally with `csalab/sahabatai1:latest`.

```bash
pip install -r requirements.txt
```

Set your HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here
```

Update `config_paper.yaml`:
- Replace `hf_YOUR_TOKEN_HERE` with your HF token (or rely on `HF_TOKEN` env var)
- Set `ollama_bin` to your Ollama binary path
- Set `datapath` to your document directory

---

## Ablation Scenarios

| Scenario | Query Processing | Reranking | $(w_v, w_{\text{ce}/b}, w_\tau)$ |
|---|---|---|---|
| BASE | Original query | None | — |
| CE | Original query | Cross-encoder | (0.80, 0.20, 0.00) |
| QEXP-CE | Taxonomy expansion | Cross-encoder | (0.80, 0.20, 0.00) |
| **QRW-CE** | **Query rewriting** | **Cross-encoder** | **(0.80, 0.20, 0.00)** |
| QEXP-CE-T-O | Taxonomy expansion | Cross-encoder + Tax | (0.70, 0.20, 0.10) |
| QEXP-BMT | Taxonomy expansion | BM25 + Taxonomy | (0.45, 0.25, 0.30) |
| QEXP-BMT-O | Taxonomy expansion | BM25 + Taxonomy | (0.50, 0.40, 0.10) |
| QRW-BMT | Query rewriting | BM25 + Taxonomy | (0.45, 0.25, 0.30) |

Weights: $w_v$ = vector, $w_{\text{ce}}$ = cross-encoder, $w_b$ = BM25, $w_\tau$ = taxonomy.

---

## Running Experiments

```bash
# Run all 8 scenarios defined in config_paper.yaml
python run_evaluation.py --config config_paper.yaml

# Reproduce weight grid search (QEXP-BMT-O)
python grid_search_weights.py

# Reproduce CE weight grid search (QEXP-CE-T-O)
python grid_search_ce_weights.py

# Analyze retrieval quality from taxonomy traces
python analyze_node_quality.py

# Reproduce figures
python plot_architecture.py
python plot_line_iterations.py
```

---

## Taxonomy

The nephrology taxonomy covers 350 concepts across 10 categories and 71 subcategories, grounded in KDIGO 2024 and PERNEFRI guidelines, with ICD-10, ICD-11, and SNOMED CT mappings. Build it from source with:

```bash
python build_taxonomy_v4.py
```

> **Note:** The compiled taxonomy files (`data/taxonomy_concepts_v4.json` and `data/synonym_index_v4.json`) are not included in this repository. To obtain them, please contact the authors at lyla.ilkomp@gmail.com.

---

## Citation

```bibtex
@inproceedings{ic3ina2026taxrag,
  title     = {Taxonomy-Guided Query Rewriting for Indonesian Hemodialysis RAG},
  booktitle = {Proceedings of IC3INA 2026},
  year      = {2026}
}
```
