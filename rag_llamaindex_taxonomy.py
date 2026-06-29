"""
rag_llamaindex_taxonomy.py
RAG pipeline identik dengan IC3INA baseline + taxonomy reranking.

Perubahan dari baseline (rag_llamaindex.py):
  1. create_chromadb_taxonomy() — setelah SemanticSplitter, tiap node di-tag
     dengan taxonomy concept dari taxonomy_concepts.json
  2. query_taxonomy() — sebelum synthesis, 20 kandidat di-rerank menggunakan
     3 skor: vector (0.65) + taxonomy concept overlap (0.25) + keyword (0.10)
  3. Output tambahan: taxonomy_trace.jsonl per query × iterasi
  4. n_iterations configurable via YAML (default 10)

Semua parameter lain identik dengan baseline:
  SemanticSplitter buffer=1 pct=95, top_k=5 (setelah rerank), compact mode,
  num_ctx=16384, prompt template_str_id, create_documents_v2 tidak berubah.

Cara pakai (jalankan dari root project):
  python3 rag_llamaindex_taxonomy.py
  python3 rag_llamaindex_taxonomy.py --config config_taxonomy_paper.yaml

Last Modified: 2026-06-10
"""

import time
import uuid
import sys
import os
import json
import argparse
import mimetypes
import socket
import subprocess
import logging
import gc

import chromadb
import nest_asyncio
import pandas as pd
import torch
import tiktoken
import yaml
from pathlib import Path
from ollama import Client

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Document, get_response_synthesizer
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Taxonomy — taxonomy_runtime.py ada di folder yang sama dengan script ini
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from taxonomy_runtime import (
    load_taxonomy_concepts,
    load_synonym_index,
    tag_chunk_with_taxonomy,
    detect_query_concepts,
    expand_query,
    rewrite_query,
    taxonomy_rerank,
    compute_cross_encoder_scores,
)

start_time = time.time()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Helpers (identik dengan baseline)
# ---------------------------------------------------------------------------

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def tiktoken_(teks: str):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(teks)
    return tokens, len(tokens)


def safe_parser(eval_response: str):
    if not eval_response.strip():
        return None, "No response"
    lines = eval_response.strip().split("\n", 1)
    score_str = lines[0]
    reasoning_str = lines[1] if len(lines) > 1 else ""
    try:
        score = float(score_str)
    except ValueError:
        score = 0.0
    return score, reasoning_str.strip()


def eval_(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    meteor = meteor_score([ref_tokens], hyp_tokens)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(ref, hyp)
    rouge_list = []
    for key, value in rouge.items():
        rouge_list.append((key, f"{value.precision:.4f}", f"{value.recall:.4f}", f"{value.fmeasure:.4f}"))
    return f"{bleu:.4f}", f"{meteor:.4f}", rouge_list


def mean_(list_):
    return f"{sum(list_) / len(list_):.4f}"


def mean__(list_):
    r1 = sum(i[0] for i in list_) / len(list_)
    r2 = sum(i[1] for i in list_) / len(list_)
    r3 = sum(i[2] for i in list_) / len(list_)
    return f"{r1:.4f}", f"{r2:.4f}", f"{r3:.4f}"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt
    return prompt + "<|assistant|>\n"


def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# create_documents_v2 — IDENTIK dengan baseline, tidak ada perubahan
# ---------------------------------------------------------------------------

def create_documents_v2(info_medis: bool, domain: str, datapath: str, chunking: str, client) -> list:
    documents = []
    file_list = [
        os.path.join(r, file)
        for r, d, f in os.walk(f"{datapath}")
        for file in f
        if file[-14:-4] != "checkpoint"
    ]

    for i in file_list:
        file_name = os.path.basename(i)
        mime_type, encoding = mimetypes.guess_type(i)
        file_type = mime_type
        file_size = os.path.getsize(i)
        creation_date = time.ctime(os.path.getctime(i))
        last_modified_date = time.ctime(os.path.getmtime(i))
        jenis_data = file_name.split("-")[0]

        if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            sumber = Path(i).stem
            if sumber == "qa-llm":
                df = pd.read_excel(i)
                df = pd.melt(
                    df,
                    id_vars=[col for col in df.columns if col not in ["ChatGPT", "Gemini", "CoPilot"]],
                    value_vars=["ChatGPT", "Gemini", "CoPilot"],
                    var_name="Referensi",
                    value_name="JAWABAN/ARTIKEL",
                )
            else:
                df = pd.read_excel(i)

            for _, row in df.iterrows():
                pertanyaan = (
                    row["PERTANYAAN"]
                    if sumber in ["qa-dokter", "qa-llm", "qa-alodokter"]
                    else row["TITLE"]
                )
                if chunking.startswith("token") and tiktoken_(pertanyaan)[1] > int(chunking.split(":")[1]) - 100:
                    msg_ = (
                        f"Tolong parafrase atau sederhanakan pertanyaan ini {pertanyaan} "
                        f"agar panjang tokennya tidak lebih dari {int(chunking.split(':')[1]) - 100}"
                    )
                    hasil_chat = client.chat(
                        model="llama3.1",
                        messages=[{"role": "user", "content": msg_}],
                        options={"num_ctx": 8192},
                    )
                    pertanyaan = hasil_chat["message"]["content"]

                document = Document(
                    metadata={
                        "filepath": i,
                        "file_name": file_name,
                        "file_type": file_type,
                        "file_size": file_size,
                        "creation_date": creation_date,
                        "last_modified_date": last_modified_date,
                        "domain": domain,
                        "info_medis": info_medis,
                        "references": row["REFERENSI"]
                        if sumber in ["qa-dokter", "qa-alodokter", "Artikel_alodokter", "Artikel_KPCDI", "artikelHalodoc"]
                        else "",
                        "question": pertanyaan,
                        "data_type": "qa" if jenis_data == "qa" else "web" if jenis_data == "web" else "pdf",
                    },
                    text=row["JAWABAN/ARTIKEL"]
                    if sumber in ["qa-dokter", "qa-alodokter", "qa-llm", "Artikel_alodokter"]
                    else row["PARAFRASE"],
                    excluded_embed_metadata_keys=[
                        "filepath", "file_name", "file_type", "file_size",
                        "creation_date", "last_modified_date", "data_type", "references",
                    ],
                    excluded_llm_metadata_keys=[
                        "filepath", "file_name", "file_type", "file_size",
                        "creation_date", "last_modified_date", "data_type", "references",
                    ],
                )
                documents.append(document)

        if file_type == "text/plain":
            sumber = Path(i).stem
            with open(i) as f1:
                listexception = [
                    "reference", "table", "references", "gambar", "tabel", "figure",
                    "daftar", "pustaka", "referensi", "daftar pustaka", "daftar rujukan", "kepustakaan",
                ]
                for line in f1:
                    if "\t" in line:
                        listtitle = [w for w in line.split("\t")[0].lower().split()]
                        if len([x for x in listexception if x in listtitle]) == 0:
                            chunk = line.split("\t")[1]

                    document = Document(
                        metadata={
                            "filepath": i,
                            "file_name": file_name,
                            "file_type": file_type,
                            "file_size": file_size,
                            "creation_date": creation_date,
                            "last_modified_date": last_modified_date,
                            "domain": domain,
                            "info_medis": info_medis,
                            "references": sumber,
                            "question": "",
                            "data_type": "qa" if jenis_data == "qa" else "web" if jenis_data == "web" else "",
                        },
                        text=chunk,
                        excluded_embed_metadata_keys=[
                            "filepath", "file_name", "file_type", "file_size",
                            "creation_date", "last_modified_date", "data_type", "references",
                        ],
                        excluded_llm_metadata_keys=[
                            "filepath", "file_name", "file_type", "file_size",
                            "creation_date", "last_modified_date", "data_type", "references",
                        ],
                    )
                    documents.append(document)

    return documents


# ---------------------------------------------------------------------------
# Taxonomy rerank helper
# ---------------------------------------------------------------------------

def _apply_taxonomy_rerank(
    orig_nodes: list,
    query: str,
    synonym_index: dict,
    taxonomy_by_id: dict,
    top_k: int = 5,
    w_vector: float = 0.45,
    w_bm25: float = 0.25,
    w_taxonomy: float = 0.30,
    ce_model=None,
    ce_tokenizer=None,
    ce_device: str = "cpu",
    w_cross: float = 0.0,
) -> tuple:
    """
    Rerank orig_nodes berdasarkan 3 skor lalu kembalikan top_k NodeWithScore
    beserta trace dict untuk logging.

    Skor:
      vector_score   — cosine similarity dari ChromaDB (BGE-M3)
      bm25_score     — BM25 lexical score query vs teks chunk (rank_bm25, normalized)
      taxonomy_score — overlap concept IDs chunk vs query (domain CKD)

    final_score = w_vector × vector + w_bm25 × BM25 + w_taxonomy × taxonomy
    """
    t_rerank_start = time.time()

    detected = detect_query_concepts(query, synonym_index, taxonomy_by_id)

    # Peta rank asli (sebelum rerank) untuk menghitung perubahan urutan
    orig_rank_map = {nws.node.node_id: i + 1 for i, nws in enumerate(orig_nodes)}

    # Convert NodeWithScore → dict format untuk taxonomy_rerank()
    candidates = []
    for nws in orig_nodes:
        metadata = dict(nws.node.metadata)
        # Deserialize taxonomy_tags dari JSON string (disimpan flat di ChromaDB)
        tags = metadata.get("taxonomy_tags", [])
        if isinstance(tags, str):
            try:
                metadata["taxonomy_tags"] = json.loads(tags)
            except (json.JSONDecodeError, ValueError):
                metadata["taxonomy_tags"] = []
        cand = {
            "text": nws.node.get_content(),
            "metadata": metadata,
            "_pernefri_score": float(nws.score or 0.0),
            "_node_id": nws.node.node_id,
        }
        candidates.append(cand)

    # Cross encoder scoring (opsional — hanya jika ce_model diisi dan w_cross > 0)
    t_ce = time.time()
    ce_scores = None
    if ce_model is not None and ce_tokenizer is not None and w_cross > 0:
        ce_scores = compute_cross_encoder_scores(
            query, candidates, ce_model, ce_tokenizer, device=ce_device
        )
    elapsed_ce = round(time.time() - t_ce, 4)

    reranked = taxonomy_rerank(
        candidates, detected, query_str=query,
        w_vector=w_vector, w_bm25=w_bm25, w_taxonomy=w_taxonomy,
        cross_encoder_scores=ce_scores, w_cross=w_cross,
    )

    # Match kembali ke NodeWithScore asli by node_id, set score=final_score
    node_map = {nws.node.node_id: nws for nws in orig_nodes}
    result_nodes = []
    for r in reranked[:top_k]:
        nid = r.get("_node_id", "")
        if nid in node_map:
            result_nodes.append(
                NodeWithScore(node=node_map[nid].node, score=r.get("final_score", 0.0))
            )

    elapsed_rerank = round(time.time() - t_rerank_start, 4)

    # Perubahan urutan chunk: orig_rank → new_rank, delta positif = naik
    rank_changes = []
    for new_rank, r in enumerate(reranked[:top_k], 1):
        nid = r.get("_node_id", "")
        orig_rank = orig_rank_map.get(nid, -1)
        rank_changes.append({
            "node_id": nid,
            "orig_rank": orig_rank,
            "new_rank": new_rank,
            "delta": orig_rank - new_rank,  # positif = naik rank
        })

    trace = {
        "use_rerank": True,
        "detected_concepts": [
            {
                "concept_id": c.get("concept_id"),
                "preferred_term": c.get("preferred_term_id"),
                "category": c.get("category"),
            }
            for c in detected
        ],
        "concept_detected": len(detected) > 0,
        "n_candidates_before_rerank": len(orig_nodes),
        "chunk_scores_after_rerank": [
            {
                "node_id": r.get("_node_id"),
                "vector_score": r.get("vector_score"),
                "bm25_score": r.get("bm25_score"),
                "taxonomy_score": r.get("taxonomy_score"),
                "final_score": r.get("final_score"),
            }
            for r in reranked[:top_k]
        ],
        "rank_changes": rank_changes,
        "elapsed_rerank_s": elapsed_rerank,
        "elapsed_ce_s": elapsed_ce,
        "use_cross_encoder": ce_model is not None and w_cross > 0,
    }

    return result_nodes, trace


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------

def load_index(nama_model: str, embed_model, chroma_model: str = "") -> VectorStoreIndex:
    """Load index dari ChromaDB. chroma_model: jika diisi, load dari nama DB lain
    (untuk ablation yang share index dengan config taxonomy penuh)."""
    db_name = chroma_model if chroma_model else nama_model
    db = chromadb.PersistentClient(path="chromaDB/" + db_name)
    chroma_collection = db.get_or_create_collection(db_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )


def require_cuda(component: str) -> str:
    """Fail fast: eksperimen ini tidak boleh jatuh diam-diam ke CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA tidak tersedia untuk {component}. "
            "Job dihentikan karena pipeline diset GPU-only."
        )
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[GPU-ONLY] {component} memakai {device}: {gpu_name}", flush=True)
    return device


# ---------------------------------------------------------------------------
# create_chromadb_taxonomy
# Sama dengan baseline kecuali mode 'semantic': tambah taxonomy tagging
# setelah SemanticSplitter dan sebelum insert ke ChromaDB.
# ---------------------------------------------------------------------------

def create_chromadb_taxonomy(
    nama_model: str,
    chunking: str,
    documents: list,
    append_db: bool,
    llm_model,
    embed_model,
    synonym_index: dict,
    taxonomy_by_id: dict,
    extra_excluded_embed_keys: list = None,
) -> VectorStoreIndex:
    print("... chroma")
    db = chromadb.PersistentClient(path="chromaDB/" + nama_model)
    chroma_collection = db.get_or_create_collection(nama_model)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chunking == "no":
        mode_arg = {"mode": "append"} if append_db else {}
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model, **mode_arg
        )

    elif ":" in chunking and chunking.split(":")[0] == "token":
        token = int(chunking.split(":")[1])
        overlap = int(chunking.split(":")[2])
        sentence_splitter = SentenceSplitter(separator=" ", chunk_size=token, chunk_overlap=overlap)
        mode_arg = {"mode": "append"} if append_db else {}
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[sentence_splitter],
            storage_context=storage_context,
            embed_model=embed_model,
            **mode_arg,
        )

    elif chunking == "semantic":
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        print("semantic chunking + taxonomy tagging ...")
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        print(f"  {len(nodes)} nodes setelah chunking")

        # Taxonomy tagging — tambahkan taxonomy_tags + taxonomy_concept_ids ke metadata tiap node
        tagged_count = 0
        category_concept_counts: dict = {}
        for node in nodes:
            result = tag_chunk_with_taxonomy(
                {"text": node.get_content(), "metadata": dict(node.metadata)},
                synonym_index,
                taxonomy_by_id,
            )
            # ChromaDB hanya terima flat types — serialize list/dict ke JSON string
            node.metadata["taxonomy_tags"] = json.dumps(result["taxonomy_tags"], ensure_ascii=False)
            node.metadata["taxonomy_concept_ids"] = json.dumps(result["metadata"]["taxonomy_concept_ids"], ensure_ascii=False)
            if result["taxonomy_tags"]:
                tagged_count += 1
                for tag in result["taxonomy_tags"]:
                    cat = tag.get("category") or "unknown"
                    category_concept_counts[cat] = category_concept_counts.get(cat, 0) + 1
        print(f"  taxonomy tagged: {tagged_count}/{len(nodes)} nodes ({100*tagged_count/max(len(nodes),1):.1f}%)")

        # Terapkan extra_excluded_embed_keys ke tiap node
        # (taxonomy_tags & taxonomy_concept_ids ditambahkan setelah excluded_embed_metadata_keys
        # di-set di Document, jadi perlu ditambahkan manual di sini)
        if extra_excluded_embed_keys:
            for node in nodes:
                for key in extra_excluded_embed_keys:
                    if key not in node.excluded_embed_metadata_keys:
                        node.excluded_embed_metadata_keys.append(key)
                    if key not in node.excluded_llm_metadata_keys:
                        node.excluded_llm_metadata_keys.append(key)
            print(f"  extra excluded from embed/llm: {extra_excluded_embed_keys}")

        # Simpan statistik indexing untuk laporan publikasi
        indexing_stats = {
            "total_nodes": len(nodes),
            "tagged_nodes": tagged_count,
            "tagged_pct": round(100 * tagged_count / max(len(nodes), 1), 2),
            "concept_counts_by_category": dict(sorted(
                category_concept_counts.items(), key=lambda x: x[1], reverse=True
            )),
        }
        stats_path = os.path.join("results", nama_model, "indexing_stats.json")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as _sf:
            json.dump(indexing_stats, _sf, ensure_ascii=False, indent=2)
        print(f"  indexing stats saved → {stats_path}")

        # ChromaDB hanya terima flat metadata — konversi semua nilai non-flat ke JSON string
        for node in nodes:
            for key, val in list(node.metadata.items()):
                if not isinstance(val, (str, int, float, type(None))):
                    node.metadata[key] = json.dumps(val, ensure_ascii=False, default=str)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

    return index


# ---------------------------------------------------------------------------
# query_taxonomy
# Sama dengan baseline query_() kecuali:
#   - retriever ambil 20 kandidat (bukan 5 langsung)
#   - sebelum synthesize: _apply_taxonomy_rerank() → top 5
#   - tulis tambahan taxonomy_trace.jsonl + coverage_summary.json
#   - n_iterations configurable (default 10)
#   - use_rerank=False untuk ablation study
# ---------------------------------------------------------------------------

def query_taxonomy(
    index,
    nama_model: str,
    llm,
    testfile: str,
    client,
    synonym_index: dict,
    taxonomy_by_id: dict,
    n_iterations: int = 10,
    retrieval_candidates_k: int = 20,
    top_k: int = 5,
    use_rerank: bool = True,
    use_query_expansion_for_retrieval: bool = False,
    use_query_rewriting: bool = False,
    max_query_variants: int = 3,
    frequency_boost_factor: float = 0.1,
    use_cross_substitution: bool = False,
    w_vector: float = 0.45,
    w_bm25: float = 0.25,
    w_taxonomy: float = 0.30,
    use_cross_encoder: bool = False,
    ce_model=None,
    ce_tokenizer=None,
    ce_device: str = "cpu",
    w_cross: float = 0.0,
) -> str:
    """
    use_rerank=True                         → full taxonomy reranking (BM25/CE + taxonomy)
    use_rerank=False                        → ambil top_k langsung tanpa rerank
    use_query_expansion_for_retrieval=True  → expand query dengan sinonim konsep sebelum
                                              kirim ke BGE-M3
    use_query_rewriting=True                → generate max_query_variants variasi query
                                              dengan substitusi sinonim, retrieve 4× total
                                              (asli + 3 variasi), merge+dedup+boost, rerank
    max_query_variants=3                    → max variasi sinonim (total 4 retrieve: asli + 3)
    frequency_boost_factor=0.1             → boost vector_score per tambahan variasi
    use_cross_substitution=True             → untuk query multi-concept: tambah 1 variasi
                                              cross (semua concept diganti sekaligus)
    use_cross_encoder=True                  → aktifkan bge-reranker-v2-m3 sebagai pengganti BM25
                                              (ce_model + ce_tokenizer harus diisi)
    """
    resultpath = os.path.join("results", nama_model)
    os.makedirs(resultpath, exist_ok=True)

    # Prompt identik dengan baseline
    template_str_id = """Berikut ini adalah informasi konteks:
    ---------------------
    {{ context_str }}
    ---------------------

    Contoh:
    Pertanyaan: Apa manfaat AI?
    Jawaban: Kecerdasan buatan (AI) memiliki banyak manfaat, seperti membantu pekerjaan manusia, meningkatkan efisiensi, dan mempercepat proses analisis data.

    Gunakan informasi konteks di atas untuk menjawab pertanyaan berikut **dalam Bahasa Indonesia saja dan jangan gunakan bahasa lain**: {query_str}
    """
    qa_template_id = RichPromptTemplate(template_str_id)

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        llm=llm,
        text_qa_template=qa_template_id,
    )

    # Retriever: ambil retrieval_candidates_k jika pakai rerank/rewriting, top_k jika ablation
    _retriever_k = retrieval_candidates_k if (use_rerank or use_query_rewriting) else top_k
    retriever = index.as_retriever(similarity_top_k=_retriever_k)

    mime_type, _ = mimetypes.guess_type(testfile)
    file_type = mime_type
    df = (
        pd.read_excel(testfile)
        if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else pd.read_csv(testfile, sep="\t")
    )

    file_hasil = open(os.path.join(resultpath, "responses.txt"), "w", encoding="utf-8")
    fs = open(os.path.join(resultpath, "nodes.txt"), "w", encoding="utf-8")
    ft = open(os.path.join(resultpath, "taxonomy_trace.jsonl"), "w", encoding="utf-8")

    # Akumulator untuk coverage summary
    n_queries_with_concept = 0
    total_concepts_detected = 0
    total_queries = 0
    latency_retrieve: list = []
    latency_rerank: list = []
    latency_ce: list = []
    latency_synthesize: list = []

    try:
        for _, row in df.iterrows():
            q = row["Pertanyaan"]
            print(q)

            for __ in range(n_iterations):
                unique_id = uuid.uuid4()
                total_queries += 1

                # 1. Retrieve dari ChromaDB — catat waktu
                t0 = time.time()

                if use_query_rewriting:
                    # Query rewriting: detect konsep → generate variasi sinonim →
                    # retrieve 4× (asli + 3 variasi) → merge+dedup+frequency_boost
                    _detected_pre = detect_query_concepts(q, synonym_index, taxonomy_by_id)
                    queries = rewrite_query(q, _detected_pre, taxonomy_by_id, max_variants=max_query_variants, use_cross_substitution=use_cross_substitution)
                    # retrieve per variasi, merge dengan frequency boost
                    freq: dict = {}
                    max_scores: dict = {}
                    best_node: dict = {}
                    for qv in queries:
                        for n in retriever.retrieve(QueryBundle(qv)):
                            nid = n.node.node_id
                            sc = float(n.score or 0.0)
                            freq[nid] = freq.get(nid, 0) + 1
                            if sc > max_scores.get(nid, -1.0):
                                max_scores[nid] = sc
                                best_node[nid] = n
                    # Terapkan frequency boost ke vector_score sebelum reranker
                    for nid, n in best_node.items():
                        boosted = max_scores[nid] * (1.0 + frequency_boost_factor * (freq[nid] - 1))
                        n.score = min(boosted, 1.0)
                    orig_nodes = list(best_node.values())

                elif use_query_expansion_for_retrieval:
                    _detected_pre = detect_query_concepts(q, synonym_index, taxonomy_by_id)
                    _expansion = expand_query(q, _detected_pre, taxonomy_by_id)
                    retrieval_query = _expansion.get("expanded_query", q)
                    orig_nodes = retriever.retrieve(QueryBundle(retrieval_query))

                else:
                    orig_nodes = retriever.retrieve(QueryBundle(q))

                elapsed_retrieve = round(time.time() - t0, 4)
                latency_retrieve.append(elapsed_retrieve)

                # 2. Taxonomy rerank (atau skip untuk ablation)
                if use_rerank:
                    top_nodes, trace = _apply_taxonomy_rerank(
                        orig_nodes, q, synonym_index, taxonomy_by_id, top_k=top_k,
                        w_vector=w_vector, w_bm25=w_bm25, w_taxonomy=w_taxonomy,
                        ce_model=ce_model if use_cross_encoder else None,
                        ce_tokenizer=ce_tokenizer if use_cross_encoder else None,
                        ce_device=ce_device,
                        w_cross=w_cross,
                    )
                    if trace["concept_detected"]:
                        n_queries_with_concept += 1
                    total_concepts_detected += len(trace["detected_concepts"])
                    latency_rerank.append(trace["elapsed_rerank_s"])
                    latency_ce.append(trace.get("elapsed_ce_s", 0.0))
                else:
                    top_nodes = orig_nodes[:top_k]
                    trace = {
                        "use_rerank": False,
                        "detected_concepts": [],
                        "concept_detected": False,
                        "n_candidates_before_rerank": len(orig_nodes),
                        "chunk_scores_after_rerank": [],
                        "rank_changes": [],
                        "elapsed_rerank_s": 0.0,
                        "elapsed_ce_s": 0.0,
                        "use_cross_encoder": False,
                    }
                    latency_rerank.append(0.0)
                    latency_ce.append(0.0)

                # 3. Synthesis — catat waktu
                t1 = time.time()
                response1 = response_synthesizer.synthesize(q, nodes=top_nodes)
                elapsed_synthesize = round(time.time() - t1, 4)
                latency_synthesize.append(elapsed_synthesize)

                elapsed_total = round(elapsed_retrieve + trace["elapsed_rerank_s"] + trace.get("elapsed_ce_s", 0.0) + elapsed_synthesize, 4)

                # responses.txt — format identik dengan baseline
                fs.write(f"{unique_id}\t{_}\t{__}\t{top_nodes}\n")
                hasil = (
                    f"{unique_id}\t{_+1}\t{__+1}\t{q}\t"
                    f"{str(response1).encode('unicode_escape').decode()}"
                )
                file_hasil.write(f"{hasil}\n")
                file_hasil.flush()

                # taxonomy_trace.jsonl — satu baris JSON per query × iterasi
                trace_rec = {
                    "run_id": str(unique_id),
                    "question_idx": _+1,
                    "iteration": __+1,
                    "question": q,
                    "elapsed_retrieve_s": elapsed_retrieve,
                    "elapsed_synthesize_s": elapsed_synthesize,
                    "elapsed_total_s": elapsed_total,
                    **trace,
                }
                ft.write(json.dumps(trace_rec, ensure_ascii=False) + "\n")
                ft.flush()
    finally:
        file_hasil.close()
        fs.close()
        ft.close()

    # Coverage summary — untuk tabel publikasi
    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    coverage_summary = {
        "nama_model": nama_model,
        "use_rerank": use_rerank,
        "total_query_iterations": total_queries,
        "n_unique_questions": len(df),
        "n_iterations_per_question": n_iterations,
        "taxonomy_coverage": {
            "queries_with_concept_pct": round(100 * n_queries_with_concept / max(total_queries, 1), 2),
            "avg_concepts_per_query": round(total_concepts_detected / max(total_queries, 1), 3),
        },
        "use_cross_encoder": use_cross_encoder,
        "latency_avg_s": {
            "retrieve": _avg(latency_retrieve),
            "rerank": _avg(latency_rerank),
            "cross_encoder": _avg(latency_ce),
            "synthesize": _avg(latency_synthesize),
            "total": _avg([r + k + c + s for r, k, c, s in zip(latency_retrieve, latency_rerank, latency_ce, latency_synthesize)]),
        },
    }
    summary_path = os.path.join(resultpath, "coverage_summary.json")
    with open(summary_path, "w", encoding="utf-8") as _sf:
        json.dump(coverage_summary, _sf, ensure_ascii=False, indent=2)
    print(f"\nCoverage summary saved → {summary_path}")
    print(json.dumps(coverage_summary, ensure_ascii=False, indent=2))

    return f"cek hasil query pada folder results/{nama_model}"


# ---------------------------------------------------------------------------
# run_taxonomy — wrapper utama
# ---------------------------------------------------------------------------

def run_taxonomy(
    nama_model: str,
    embed_model: str,
    llm_model: str,
    domain: str,
    datapath: str,
    chunking: str,
    append_db: bool,
    port: int,
    info_medis: bool,
    ollama_bin: str,
    token_hf: str,
    mode: str,
    testfile: str,
    taxonomy_concepts_path: str,
    synonym_index_path: str,
    n_iterations: int = 10,
    retrieval_candidates_k: int = 20,
    top_k: int = 5,
    use_rerank: bool = True,
    use_query_expansion_for_retrieval: bool = False,
    use_query_rewriting: bool = False,
    max_query_variants: int = 3,
    frequency_boost_factor: float = 0.1,
    use_cross_substitution: bool = False,
    w_vector: float = 0.45,
    w_bm25: float = 0.25,
    w_taxonomy: float = 0.30,
    use_cross_encoder: bool = False,
    cross_encoder_model: str = "",
    w_cross: float = 0.0,
    chroma_model: str = "",
    extra_excluded_embed_keys: list = None,
) -> str:

    # Resolve taxonomy file paths: jika tidak ketemu dari CWD,
    # coba cari di folder yang sama dengan script ini
    def _resolve(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute() or p.exists():
            return str(p)
        script_rel = _SCRIPT_DIR / p.name
        if script_rel.exists():
            return str(script_rel)
        return path_str

    taxonomy_concepts_path = _resolve(taxonomy_concepts_path)
    synonym_index_path = _resolve(synonym_index_path)
    cuda_device = require_cuda("PyTorch/HuggingFace embedding")

    # --- LLM setup (identik dengan baseline run_()) ---
    if llm_model.startswith("ollama"):
        llm_model_name = llm_model.split("|")[1]
        os.environ.pop("OLLAMA_LLM_LIBRARY", None)
        os.environ.pop("OLLAMA_NEW_ENGINE", None)
        os.environ.pop("OLLAMA_FLASH_ATTENTION", None)
        if not (is_port_in_use(port) or is_port_in_use(port + 1)):
            cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            ollama_proc_env = os.environ.copy()
            ollama_proc_env["CUDA_VISIBLE_DEVICES"] = cuda_dev
            ollama_proc_env.pop("ROCR_VISIBLE_DEVICES", None)
            ollama_proc_env.pop("HIP_VISIBLE_DEVICES", None)
            ollama_proc_env.pop("GPU_DEVICE_ORDINAL", None)
            cuda_lib_base = "/home/lyla001/CHATBOT/latest_venv/lib/python3.10/site-packages/nvidia"
            cuda_preload = [
                f"{cuda_lib_base}/cuda_runtime/lib/libcudart.so.12",
                f"{cuda_lib_base}/cublas/lib/libcublas.so.12",
                f"{cuda_lib_base}/cublas/lib/libcublasLt.so.12",
            ]
            if all(os.path.exists(lib) for lib in cuda_preload):
                old_preload = ollama_proc_env.get("LD_PRELOAD", "")
                ollama_proc_env["LD_PRELOAD"] = ":".join(cuda_preload + ([old_preload] if old_preload else []))
                old_ld_path = ollama_proc_env.get("LD_LIBRARY_PATH", "")
                ollama_proc_env["LD_LIBRARY_PATH"] = ":".join(
                    [
                        f"{cuda_lib_base}/cuda_runtime/lib",
                        f"{cuda_lib_base}/cublas/lib",
                        "/home/lyla001/ollama/.local/lib/ollama",
                    ] + ([old_ld_path] if old_ld_path else [])
                )

            ollama_main_env = ollama_proc_env.copy()
            ollama_main_env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
            ollama_embed_env = ollama_proc_env.copy()
            ollama_embed_env["OLLAMA_HOST"] = f"0.0.0.0:{port + 1}"
            subprocess.Popen([ollama_bin, "serve"], env=ollama_main_env)
            subprocess.Popen([ollama_bin, "serve"], env=ollama_embed_env)
            time.sleep(5)
        llm = Ollama(model=llm_model_name, request_timeout=3000, num_ctx=16384, base_url=f"http://localhost:{port}")
    elif llm_model.startswith("remote") or llm_model.startswith("local"):
        llm_model_name = llm_model.split("|")[1]
        require_cuda("HuggingFace LLM")
        llm = HuggingFaceLLM(
            context_window=8192,
            max_new_tokens=256,
            tokenizer_name=llm_model_name,
            model_name=llm_model_name,
            device_map={"": 0},
            tokenizer_kwargs={"max_length": 4096},
            model_kwargs={"torch_dtype": torch.bfloat16},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

    # --- Embedding setup (identik dengan baseline run_()) ---
    if embed_model.startswith("local|FT"):
        embed_model_obj = HuggingFaceEmbedding(
            model_name="./" + embed_model.split("|")[2],
            device=cuda_device,
            embed_batch_size=64,
        )
    elif embed_model.startswith("local|hf"):
        local_path = embed_model.split("|")[2]
        embed_model_obj = HuggingFaceEmbedding(
            model_name=local_path,
            device=cuda_device,
            embed_batch_size=64,
        )
    elif embed_model.startswith("remote|hf"):
        nama_embed = embed_model.split("|")[2]
        embed_model_obj = HuggingFaceEmbedding(
            model_name=nama_embed,
            trust_remote_code=True,
            device=cuda_device,
            embed_batch_size=64,
        )
    else:
        embed_model_obj = OllamaEmbedding(
            model_name=embed_model,
            base_url=f"0.0.0.0:{port+1}",
            ollama_additional_kwargs={"mirostat": 0},
        )

    client = Client(host=f"0.0.0.0:{port}")

    # --- Taxonomy loading ---
    print(f"Loading taxonomy dari {taxonomy_concepts_path} ...")
    taxonomy_by_id = load_taxonomy_concepts(taxonomy_concepts_path)
    synonym_index = load_synonym_index(synonym_index_path)
    print(f"  {len(taxonomy_by_id)} concepts, {len(synonym_index)} synonym terms")

    # --- Cross encoder loading (opsional) ---
    ce_model_obj = None
    ce_tokenizer_obj = None
    ce_device = cuda_device
    if use_cross_encoder:
        if not cross_encoder_model:
            raise ValueError("cross_encoder_model harus diisi jika use_cross_encoder=True")
        os.environ.setdefault("HF_TOKEN", token_hf)
        print(f"Loading cross encoder: {cross_encoder_model} → {ce_device} ...")
        ce_tokenizer_obj = AutoTokenizer.from_pretrained(cross_encoder_model, token=token_hf)
        ce_model_obj = AutoModelForSequenceClassification.from_pretrained(
            cross_encoder_model, token=token_hf
        ).to(ce_device)
        ce_model_obj.eval()
        if torch.cuda.is_available():
            print(f"  VRAM setelah load CE: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # --- Pipeline modes ---
    if mode == "createdb-query-eval":
        documents = create_documents_v2(info_medis, domain, datapath, chunking, client)
        index = create_chromadb_taxonomy(
            nama_model, chunking, documents, append_db, llm, embed_model_obj,
            synonym_index, taxonomy_by_id,
            extra_excluded_embed_keys=extra_excluded_embed_keys,
        )
        # Bebaskan GPU memory dari embed model setelah indexing selesai
        del embed_model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[GPU] VRAM setelah empty_cache: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
        query_taxonomy(
            index, nama_model, llm, testfile, client,
            synonym_index, taxonomy_by_id, n_iterations, retrieval_candidates_k, top_k,
            use_rerank=use_rerank,
            use_query_expansion_for_retrieval=use_query_expansion_for_retrieval,
            use_query_rewriting=use_query_rewriting,
            max_query_variants=max_query_variants,
            frequency_boost_factor=frequency_boost_factor,
            use_cross_substitution=use_cross_substitution,
            w_vector=w_vector, w_bm25=w_bm25, w_taxonomy=w_taxonomy,
            use_cross_encoder=use_cross_encoder,
            ce_model=ce_model_obj, ce_tokenizer=ce_tokenizer_obj, ce_device=ce_device,
            w_cross=w_cross,
        )

    elif mode == "createdb":
        documents = create_documents_v2(info_medis, domain, datapath, chunking, client)
        create_chromadb_taxonomy(
            nama_model, chunking, documents, append_db, llm, embed_model_obj,
            synonym_index, taxonomy_by_id,
            extra_excluded_embed_keys=extra_excluded_embed_keys,
        )

    elif mode == "query-eval":
        print(f"[{time.strftime('%H:%M:%S')}] Loading ChromaDB index ...", flush=True)
        index = load_index(nama_model, embed_model_obj, chroma_model=chroma_model)
        print(f"[{time.strftime('%H:%M:%S')}] Index loaded, mulai query-eval", flush=True)
        query_taxonomy(
            index, nama_model, llm, testfile, client,
            synonym_index, taxonomy_by_id, n_iterations, retrieval_candidates_k, top_k,
            use_rerank=use_rerank,
            use_query_expansion_for_retrieval=use_query_expansion_for_retrieval,
            use_query_rewriting=use_query_rewriting,
            max_query_variants=max_query_variants,
            frequency_boost_factor=frequency_boost_factor,
            use_cross_substitution=use_cross_substitution,
            w_vector=w_vector, w_bm25=w_bm25, w_taxonomy=w_taxonomy,
            use_cross_encoder=use_cross_encoder,
            ce_model=ce_model_obj, ce_tokenizer=ce_tokenizer_obj, ce_device=ce_device,
            w_cross=w_cross,
        )

    print("--- %s seconds ---" % (time.time() - start_time))
    return f"{nama_model} finished\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline dengan taxonomy reranking")
    parser.add_argument(
        "--config",
        default="config_taxonomy.yaml",
        help="Path ke config YAML (default: config_taxonomy.yaml di folder yang sama dengan script)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Jalankan hanya skenario dengan nama_model yang mengandung string ini",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Cek di CWD dulu, lalu di folder script
        if not config_path.exists():
            config_path = _SCRIPT_DIR / config_path.name

    with open(config_path, "r") as f:
        lst_model = yaml.safe_load(f)

    if args.filter:
        lst_model = [m for m in lst_model if args.filter in m.get("nama_model", "")]
        if not lst_model:
            print(f"[ERROR] Tidak ada skenario dengan nama_model mengandung '{args.filter}'")
            sys.exit(1)
        print(f"[INFO] Filter '{args.filter}': {len(lst_model)} skenario akan dijalankan")

    for i in lst_model:
        nama_model = i["nama_model"]
        print(f"+++++++ {nama_model} ++++++++\n")
        run_taxonomy(
            nama_model=nama_model,
            embed_model=i["embed_model"],
            llm_model=i["llm_model"],
            domain=i["domain"],
            datapath=i["datapath"],
            chunking=i["chunking"],
            append_db=i["append_db"],
            port=i["port"],
            info_medis=i["info_medis"],
            ollama_bin=i["ollama_bin"],
            token_hf=i["token_hf"],
            mode=i["mode"],
            testfile=i["testfile"],
            taxonomy_concepts_path=i.get("taxonomy_concepts_path", "taxonomy_concepts.json"),
            synonym_index_path=i.get("synonym_index_path", "synonym_index.json"),
            n_iterations=i.get("n_iterations", 10),
            retrieval_candidates_k=i.get("retrieval_candidates_k", 20),
            top_k=i.get("top_k", 5),
            use_rerank=i.get("use_rerank", True),
            use_query_expansion_for_retrieval=i.get("use_query_expansion_for_retrieval", False),
            use_query_rewriting=i.get("use_query_rewriting", False),
            max_query_variants=i.get("max_query_variants", 3),
            frequency_boost_factor=i.get("frequency_boost_factor", 0.1),
            use_cross_substitution=i.get("use_cross_substitution", False),
            w_vector=i.get("w_vector", 0.45),
            w_bm25=i.get("w_bm25", 0.0 if i.get("use_cross_encoder", False) else 0.25),
            w_taxonomy=i.get("w_taxonomy", 0.30),
            use_cross_encoder=i.get("use_cross_encoder", False),
            cross_encoder_model=i.get("cross_encoder_model", ""),
            w_cross=i.get("w_cross", 0.0),
            chroma_model=i.get("chroma_model", ""),
            extra_excluded_embed_keys=i.get("extra_excluded_embed_keys", None),
        )

    print("========== finish =============")
