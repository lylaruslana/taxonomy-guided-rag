from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

DEFAULT_CONCEPTS_PATH = Path("docs/taksonomi/taxonomy_concepts.json")
DEFAULT_SYNONYM_INDEX_PATH = Path("docs/taksonomi/synonym_index.json")
DEFAULT_SYNONYM_INDEX_DETAILED_PATH = Path("docs/taksonomi/synonym_index_detailed.json")

_TOKEN_RE = re.compile(r"[^a-z0-9\s]+")


def _normalize_text(value: str) -> str:
    text = str(value or "").lower().strip()
    text = _TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_boundary_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=2)
def load_taxonomy_concepts(path: str | Path = DEFAULT_CONCEPTS_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    concepts = payload.get("concepts") or []
    by_id: dict[str, dict[str, Any]] = {}
    for row in concepts:
        cid = str(row.get("concept_id") or "").strip()
        if not cid:
            continue
        clean = dict(row)
        clean.setdefault("synonyms", [])
        by_id[cid] = clean
    return by_id


@lru_cache(maxsize=2)
def load_synonym_index(path: str | Path = DEFAULT_SYNONYM_INDEX_PATH) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    index = payload.get("index") or {}
    normalized: dict[str, list[str]] = {}
    for term, concept_ids in index.items():
        key = _normalize_text(term)
        if not key:
            continue
        normalized[key] = [str(cid) for cid in (concept_ids or []) if str(cid).strip()]
    return normalized


def load_synonym_index_detailed(path: str | Path = DEFAULT_SYNONYM_INDEX_DETAILED_PATH) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    index = payload.get("index") or {}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for term, details in index.items():
        key = _normalize_text(term)
        if not key:
            continue
        normalized[key] = list(details or [])
    return normalized


def detect_query_concepts(query: str, synonym_index: dict[str, list[str]], taxonomy_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    text = _normalize_text(query)
    if not text:
        return []
    found_by_concept: dict[str, dict[str, Any]] = {}
    for term, concept_ids in synonym_index.items():
        if len(term) < 2:
            continue
        if not _word_boundary_pattern(term).search(text):
            continue
        # Highly ambiguous short aliases (for example "pgk") are useful, but
        # should not fan out to many unrelated concepts.
        if len(term) <= 3 and len(concept_ids) > 2:
            concept_ids = concept_ids[:2]
        ambiguity = max(len(concept_ids), 1)
        for concept_id in concept_ids:
            concept = taxonomy_by_id.get(concept_id) or {}
            preferred_term_id = str(concept.get("preferred_term_id") or "")
            category = str(concept.get("category_name") or concept.get("category") or "")
            subcategory = str(concept.get("subcategory_name") or concept.get("subcategory") or "")
            bucket = found_by_concept.setdefault(
                concept_id,
                {
                    "concept_id": concept_id,
                    "preferred_term_id": preferred_term_id,
                    "matched_term": "",
                    "matched_terms": [],
                    "category": category,
                    "subcategory": subcategory,
                    "_score": 0.0,
                },
            )
            if term not in bucket["matched_terms"]:
                bucket["matched_terms"].append(term)
            if len(term) > len(str(bucket.get("matched_term") or "")):
                bucket["matched_term"] = term
            # Prefer specific phrase matches and penalize ambiguous aliases.
            score_delta = (len(term) / max(ambiguity, 1)) * 0.1
            if preferred_term_id and term == _normalize_text(preferred_term_id):
                score_delta += 0.75
            if len(term.split()) >= 2:
                score_delta += 0.15
            bucket["_score"] = float(bucket.get("_score") or 0.0) + score_delta

    ranked = sorted(
        found_by_concept.values(),
        key=lambda row: (
            float(row.get("_score") or 0.0) * -1,
            len(str(row.get("matched_term") or "")) * -1,
            row.get("concept_id", ""),
        ),
    )
    # Keep only the most relevant concepts so query expansion stays focused.
    trimmed = ranked[:4]
    for row in trimmed:
        row.pop("_score", None)
    return trimmed


def expand_query(query: str, detected_concepts: list[dict[str, Any]], taxonomy_by_id: dict[str, dict[str, Any]], max_terms_per_concept: int = 3, max_total_terms: int = 10) -> dict[str, Any]:
    query_norm = _normalize_text(query)
    terms: list[str] = []
    seen = {query_norm}
    for item in detected_concepts:
        cid = str(item.get("concept_id") or "")
        concept = taxonomy_by_id.get(cid) or {}
        candidates = [concept.get("preferred_term_id")] + list(concept.get("synonyms") or [])
        added = 0
        for cand in candidates:
            term = _normalize_text(str(cand or ""))
            if not term or term in seen:
                continue
            if len(term) <= 1:
                continue
            # Skip very long noisy phrases in expansion.
            if len(term.split()) > 6:
                continue
            seen.add(term)
            terms.append(term)
            added += 1
            if len(terms) >= max_total_terms:
                break
            if added >= max_terms_per_concept:
                break
        if len(terms) >= max_total_terms:
            break
    expanded_query = " ".join([query.strip(), *terms]).strip()
    return {
        "original_query": query,
        "expanded_terms": terms,
        "expanded_query": expanded_query,
    }


def rewrite_query(
    query: str,
    detected_concepts: list[dict[str, Any]],
    taxonomy_by_id: dict[str, dict[str, Any]],
    max_variants: int = 3,
    use_cross_substitution: bool = False,
) -> list[str]:
    """
    Generate variasi query dengan mengganti istilah terdeteksi dengan sinonimnya.
    Setiap variasi adalah kalimat natural (bukan append keyword).

    use_cross_substitution=True (multi-concept):
      Selain variasi single-substitution, tambahkan variasi cross-substitution —
      semua concept diganti sinonimnya sekaligus dalam satu query.

    Contoh (2 concept, use_cross_substitution=True):
      query    = "apakah hipertensi memengaruhi hasil hemodialisis?"
      variasi  = ["apakah HTN memengaruhi hasil hemodialisis?",       ← single C1
                  "apakah hipertensi memengaruhi hasil HD?",          ← single C2
                  "apakah HTN memengaruhi hasil HD?"]                 ← cross C1+C2

    Return: [query_asli] + variasi (max max_variants+1 total, query asli selalu masuk)
    """
    import re as _re

    variants: list[str] = []
    seen_variants = {_normalize_text(query)}

    # --- Single-substitution (per concept) ---
    # Sisakan 1 slot untuk cross variant jika use_cross_substitution aktif dan ada 2+ concept
    n_cross = 1 if (use_cross_substitution and len(detected_concepts) >= 2) else 0
    max_single = max(max_variants - n_cross, 1)

    for concept in detected_concepts:
        if len(variants) >= max_single:
            break
        cid = str(concept.get("concept_id") or "")
        matched_term = str(concept.get("matched_term") or "")
        if not matched_term:
            continue

        concept_data = taxonomy_by_id.get(cid) or {}
        synonyms = list(concept_data.get("synonyms") or [])

        for syn in synonyms:
            if len(variants) >= max_single:
                break
            syn_clean = str(syn or "").strip()
            if not syn_clean or len(syn_clean) <= 1:
                continue
            if _normalize_text(syn_clean) in _normalize_text(query):
                continue
            if len(syn_clean.split()) > 5:
                continue
            pattern = _re.compile(_re.escape(matched_term), _re.IGNORECASE)
            variant = pattern.sub(syn_clean, query, count=1)
            if variant == query:
                continue
            norm_variant = _normalize_text(variant)
            if norm_variant in seen_variants:
                continue
            seen_variants.add(norm_variant)
            variants.append(variant)

    # --- Cross-substitution (semua concept diganti sekaligus) ---
    if use_cross_substitution and len(detected_concepts) >= 2 and len(variants) < max_variants:
        # Ambil sinonim terbaik (pertama yang valid) per concept
        best_syns: list[tuple[str, str]] = []  # [(matched_term, syn), ...]
        for concept in detected_concepts:
            cid = str(concept.get("concept_id") or "")
            matched_term = str(concept.get("matched_term") or "")
            if not matched_term:
                continue
            concept_data = taxonomy_by_id.get(cid) or {}
            synonyms = list(concept_data.get("synonyms") or [])
            for syn in synonyms:
                syn_clean = str(syn or "").strip()
                if not syn_clean or len(syn_clean) <= 1:
                    continue
                if _normalize_text(syn_clean) in _normalize_text(query):
                    continue
                if len(syn_clean.split()) > 5:
                    continue
                best_syns.append((matched_term, syn_clean))
                break  # satu sinonim terbaik per concept cukup

        if len(best_syns) >= 2:
            cross = query
            for matched_term, syn_clean in best_syns:
                pattern = _re.compile(_re.escape(matched_term), _re.IGNORECASE)
                cross = pattern.sub(syn_clean, cross, count=1)
            norm_cross = _normalize_text(cross)
            if cross != query and norm_cross not in seen_variants:
                seen_variants.add(norm_cross)
                variants.append(cross)

    return [query] + variants


def tag_chunk_with_taxonomy(chunk: dict[str, Any], synonym_index: dict[str, list[str]], taxonomy_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    text = _normalize_text(str(chunk.get("text") or ""))
    tags_by_concept: dict[str, dict[str, Any]] = {}
    if text:
        for term, concept_ids in synonym_index.items():
            if len(term) < 2:
                continue
            if not _word_boundary_pattern(term).search(text):
                continue
            for concept_id in concept_ids:
                concept = taxonomy_by_id.get(concept_id) or {}
                bucket = tags_by_concept.setdefault(
                    concept_id,
                    {
                        "concept_id": concept_id,
                        "preferred_term_id": str(concept.get("preferred_term_id") or ""),
                        "category": str(concept.get("category_name") or concept.get("category") or ""),
                        "subcategory": str(concept.get("subcategory_name") or concept.get("subcategory") or ""),
                        "matched_terms": [],
                    },
                )
                if term not in bucket["matched_terms"]:
                    bucket["matched_terms"].append(term)
    tagged = dict(chunk)
    tagged["taxonomy_tags"] = list(tags_by_concept.values())
    metadata = dict(tagged.get("metadata") or {})
    metadata["taxonomy_tags"] = tagged["taxonomy_tags"]
    metadata["taxonomy_concept_ids"] = [row["concept_id"] for row in tagged["taxonomy_tags"]]
    tagged["metadata"] = metadata
    return tagged


def tag_chunks_with_taxonomy(chunks: list[dict[str, Any]], synonym_index: dict[str, list[str]], taxonomy_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [tag_chunk_with_taxonomy(chunk, synonym_index, taxonomy_by_id) for chunk in chunks]


def compute_taxonomy_score(chunk_tags: list[dict[str, Any]], query_concepts: list[dict[str, Any]]) -> float:
    if not chunk_tags or not query_concepts:
        return 0.0
    tag_ids = {str(tag.get("concept_id") or "") for tag in chunk_tags}
    tag_subcats = {str(tag.get("subcategory") or "") for tag in chunk_tags}
    tag_cats = {str(tag.get("category") or "") for tag in chunk_tags}
    total = 0.0
    for concept in query_concepts:
        cid = str(concept.get("concept_id") or "")
        sub = str(concept.get("subcategory") or "")
        cat = str(concept.get("category") or "")
        if cid and cid in tag_ids:
            total += 1.0
        elif sub and sub in tag_subcats:
            total += 0.6
        elif cat and cat in tag_cats:
            total += 0.3
    return round(min(total / max(len(query_concepts), 1), 1.0), 4)


def taxonomy_rerank(
    candidates: list[dict[str, Any]],
    query_concepts: list[dict[str, Any]],
    query_str: str = "",
    w_vector: float = 0.45,
    w_bm25: float = 0.25,
    w_taxonomy: float = 0.30,
    cross_encoder_scores: list[float] | None = None,
    w_cross: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Scoring: w_vector×vector + w_bm25×BM25 + w_taxonomy×taxonomy + w_cross×cross_encoder
    BM25 dihitung di atas kandidat (bukan seluruh corpus) menggunakan query_str asli.
    cross_encoder_scores: list skor [0,1] per kandidat, urutan sama dengan candidates.
                          Jika None atau w_cross=0, CE diabaikan.
    """
    texts = [_normalize_text(str(row.get("text") or "")) for row in candidates]
    tokenized_corpus = [re.findall(r"\b\w+\b", t) for t in texts]
    query_tokens = re.findall(r"\b\w+\b", query_str.lower()) if query_str else []

    if w_bm25 > 0 and query_tokens and any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_raw = bm25.get_scores(query_tokens)
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_scores = [float(s) / max_bm25 for s in bm25_raw]
    else:
        bm25_scores = [0.0] * len(candidates)

    reranked: list[dict[str, Any]] = []
    for i, row in enumerate(candidates):
        metadata = dict(row.get("metadata") or {})
        tags = metadata.get("taxonomy_tags") or row.get("taxonomy_tags") or []
        taxonomy_score = compute_taxonomy_score(tags, query_concepts)
        vector_score = float(row.get("_pernefri_score") or row.get("score") or row.get("_semantic_score") or 0.0)
        bm25_score = bm25_scores[i]
        ce_score = float(cross_encoder_scores[i]) if (cross_encoder_scores and i < len(cross_encoder_scores)) else 0.0
        final_score = (
            (w_vector * vector_score)
            + (w_bm25 * bm25_score)
            + (w_taxonomy * taxonomy_score)
            + (w_cross * ce_score)
        )
        row2 = dict(row)
        row2["vector_score"]   = round(vector_score, 4)
        row2["bm25_score"]     = round(bm25_score, 4)
        row2["taxonomy_score"] = round(taxonomy_score, 4)
        row2["ce_score"]       = round(ce_score, 4)
        row2["final_score"]    = round(final_score, 4)
        md2 = dict(metadata)
        md2["taxonomy_score"] = row2["taxonomy_score"]
        md2["bm25_score"]     = row2["bm25_score"]
        md2["ce_score"]       = row2["ce_score"]
        md2["final_score"]    = row2["final_score"]
        row2["metadata"] = md2
        reranked.append(row2)
    reranked.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)
    return reranked


def compute_cross_encoder_scores(
    query: str,
    candidates: list[dict[str, Any]],
    ce_model: Any,
    ce_tokenizer: Any,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[float]:
    """
    Hitung cross encoder (bge-reranker-v2-m3) score untuk semua (query, chunk) pairs.
    Return list[float] panjang len(candidates), sigmoid(logit) → [0, 1].
    ce_model  : AutoModelForSequenceClassification, sudah .eval() dan .to(device)
    ce_tokenizer : AutoTokenizer yang sesuai
    """
    import torch
    pairs = [[query, str(c.get("text") or "")] for c in candidates]
    all_scores: list[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        enc = ce_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = ce_model(**enc, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(logits).cpu().tolist()
        all_scores.extend(scores)
    return all_scores
