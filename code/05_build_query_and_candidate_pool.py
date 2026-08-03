#!/usr/bin/env python3
"""Build EUR-LexRD queries, retriever pools, citation qrels, and candidate pool.

Edit the IDE configuration near the end of the file and run the complete
script. Outputs follow the canonical data/shared, data/qrels, and data/pooling
layout.
"""

from types import SimpleNamespace
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)
# from dotenv import load_dotenv
# load_dotenv()


# =========================
# ========= CONFIG ========
# =========================

ECLI_RE = re.compile(r"\b(?:ECLI:)?(EU:(?:C|T|F):\d{4}:\d{1,6})\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}")

DEFAULT_QUERY_SAMPLE_SIZE = 100
DEFAULT_RANDOM_NEGATIVES = 10
DEFAULT_RANDOM_SEED = 123
DEFAULT_BACKFILL_MIN_FRACTION = 0.50
DEFAULT_TOP_K_PER_POOL = 35
DEFAULT_RANDOM_TOP_K_PER_POOL = 5
DEFAULT_DENSE_AGGREGATION = "query_chunk_max_then_average"

SOURCE_PRIORITY = [
    "citation",
    "tfidf",
    "bm25",
    "minilm",
    "qwen",
    "gemini",
    "random",
]

DEFAULT_RETRIEVER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tfidf": {
        "enabled": True,
        "type": "sparse_tfidf",
        "topk": 250,
    },
    "bm25": {
        "enabled": True,
        "type": "sparse_bm25",
        "topk": 250,
    },
    "minilm": {
        "enabled": True,
        "type": "dense_sentence_transformers",
        "topk": 250,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "query_prefix": "",
        "document_prefix": "",
        "normalize": True,
        "batch_size": 4,
        "max_input_tokens": 256,
        "chars_per_token": 4.0,
        "headroom": 0.05,
        "overlap_fraction": 0.10,
    },
    "gemini": {
        "enabled": True,
        "type": "dense_gemini",
        "topk": 250,
        "model_name": "gemini-embedding-001",
        "embedding_dim": 3072,
        "query_prefix": "",
        "document_prefix": "",
        "normalize": True,
        "batch_size": 1,
        "max_input_tokens": 2048,
        "chars_per_token": 4.0,
        "headroom": 0.05,
        "overlap_fraction": 0.10,
    },
    "qwen": {
        "enabled": True,
        "type": "dense_sentence_transformers",
        "topk": 250,
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "embedding_dim": 4096,
        "query_prefix": "",
        "document_prefix": "",
        "normalize": True,
        "batch_size": 1,
        "max_input_tokens": 8192,
        "chars_per_token": 4.0,
        "headroom": 0.05,
        "overlap_fraction": 0.10,
    },
}

GEMINI_BATCH_SIZE = 1
GEMINI_REQUEST_DELAY_SECONDS = 0.5


# =========================
# ========= PATHS =========
# =========================

@dataclass(frozen=True)
class PipelinePaths:
    input_dir: Path
    eval_dir: Path
    query_pool_path: Path
    citation_candidates_path: Path
    qrels_citations_path: Path
    pooling_dir: Path
    citation_pool_path: Path
    random_negatives_path: Path
    pool_config_path: Path
    candidate_pool_path: Path
    embedding_cache_dir: Path


def build_paths(input_dir: Path, eval_dir: Path) -> PipelinePaths:
    """Build paths using the repository's canonical data layout."""
    input_dir = Path(input_dir)
    eval_dir = Path(eval_dir)
    shared_dir = eval_dir / "shared"
    qrels_dir = eval_dir / "qrels"
    pooling_dir = eval_dir / "pooling"
    cache_dir = eval_dir / "cache" / "embeddings"
    return PipelinePaths(
        input_dir=input_dir,
        eval_dir=eval_dir,
        query_pool_path=shared_dir / "query_pool.jsonl",
        citation_candidates_path=shared_dir / "citation_candidates.tsv",
        qrels_citations_path=qrels_dir / "qrels_citations.tsv",
        pooling_dir=pooling_dir,
        citation_pool_path=pooling_dir / "citation_pool.tsv",
        random_negatives_path=pooling_dir / "random_negatives.tsv",
        pool_config_path=pooling_dir / "pool_config.json",
        candidate_pool_path=shared_dir / "candidate_pool.tsv",
        embedding_cache_dir=cache_dir,
    )


# =========================
# ======== HELPERS ========
# =========================


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.json")


def canonical_ecli(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    match = ECLI_RE.search(value)
    return f"ECLI:{match.group(1).upper()}" if match else None


def choose_doc_id(meta: Dict[str, Any]) -> Optional[str]:
    ecli = canonical_ecli(meta.get("ecli")) if isinstance(meta, dict) else None
    if ecli:
        return ecli
    case_number = (meta.get("case_number") or "").strip() if isinstance(meta, dict) else ""
    return case_number or None


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_citation_candidates(path: Path) -> Dict[str, List[str]]:
    out: DefaultDict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        _ = handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            query_id, candidate_id, source = parts
            if source == "citation":
                out[query_id].append(candidate_id)
    return out


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def pool_output_path(paths: PipelinePaths, retriever_name: str) -> Path:
    return paths.pooling_dir / f"{retriever_name}_pool.tsv"


def stage1_outputs_exist(paths: PipelinePaths) -> bool:
    return (
        paths.query_pool_path.exists()
        and paths.citation_candidates_path.exists()
        and paths.qrels_citations_path.exists()
    )


def stable_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def safe_cache_name(doc_id: str) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:12]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("_")
    cleaned = cleaned[:120] if cleaned else "item"
    return f"{cleaned}__{digest}"


def get_effective_chunk_params(cfg: Dict[str, Any]) -> Tuple[int, int]:
    max_input_tokens = int(cfg["max_input_tokens"])
    headroom = float(cfg.get("headroom", 0.0))
    overlap_fraction = float(cfg.get("overlap_fraction", 0.0))

    effective_tokens = max(1, int(math.floor(max_input_tokens * (1.0 - headroom))))
    overlap_tokens = int(round(effective_tokens * overlap_fraction))
    overlap_tokens = max(0, min(overlap_tokens, max(0, effective_tokens - 1)))
    return effective_tokens, overlap_tokens


def get_embedding_cache_paths(paths: PipelinePaths, retriever_name: str, is_query: bool) -> Tuple[Path, Path]:
    side = "query" if is_query else "doc"
    cache_dir = paths.embedding_cache_dir / retriever_name / side
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, cache_dir / "manifest.json"


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"items": {}}


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cached_embedding(
    cache_dir: Path,
    manifest: Dict[str, Any],
    doc_id: str,
    expected_text_hash: str,
    expected_chunk_count: int,
) -> Optional[np.ndarray]:
    record = manifest.get("items", {}).get(doc_id)
    if not record:
        return None
    if record.get("text_hash") != expected_text_hash:
        return None
    if int(record.get("chunk_count", -1)) != int(expected_chunk_count):
        return None

    file_name = record.get("file")
    if not file_name:
        return None

    file_path = cache_dir / file_name
    if not file_path.exists():
        return None

    try:
        arr = np.load(file_path)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[0] != expected_chunk_count:
        return None

    return arr.astype(np.float32, copy=False)


def save_cached_embedding(
    cache_dir: Path,
    manifest: Dict[str, Any],
    doc_id: str,
    text_hash: str,
    arr: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    file_name = safe_cache_name(doc_id) + ".npy"
    np.save(cache_dir / file_name, arr)

    manifest.setdefault("items", {})
    manifest["items"][doc_id] = {
        "file": file_name,
        "text_hash": text_hash,
        "chunk_count": int(arr.shape[0]),
        "dim": int(arr.shape[1]) if arr.ndim == 2 and arr.shape[0] > 0 else 0,
        **metadata,
    }


def write_pool_tsv(path: Path, rows: List[Tuple[str, str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "candidate_id", "source", "score"])
        for query_id, candidate_id, source, score in rows:
            writer.writerow([query_id, candidate_id, source, score])


def write_pool_tsv_no_score(path: Path, rows: List[Tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "candidate_id", "source"])
        for query_id, candidate_id, source in rows:
            writer.writerow([query_id, candidate_id, source])


def choose_primary_source(sources: Set[str]) -> str:
    for source in SOURCE_PRIORITY:
        if source in sources:
            return source
    return "unknown"


# =========================
# ===== TEXT EXTRACT ======
# =========================


def _collect_text(node: Any) -> List[str]:
    parts: List[str] = []
    if node is None:
        return parts
    if isinstance(node, str):
        if node.strip():
            parts.append(node.strip())
    elif isinstance(node, list):
        for item in node:
            parts.extend(_collect_text(item))
    elif isinstance(node, dict):
        for value in node.values():
            parts.extend(_collect_text(value))
    return parts


def get_summary_text(data: Dict[str, Any]) -> str:
    case_summary = data.get("case_summary") or {}
    summary = case_summary.get("summary_text")
    return summary.strip() if isinstance(summary, str) and summary.strip() else ""


def get_segment_text(data: Dict[str, Any], segment_name: str) -> str:
    segment = (data.get("case_segments") or {}).get(segment_name)
    parts = _collect_text(segment)
    return "\n\n".join([part for part in parts if part]).strip()


def get_full_text(data: Dict[str, Any]) -> str:
    ordered_segments = [
        "document_start",
        "procedural_introduction",
        "legal_context",
        "facts_and_background",
        "reasoning",
        "decision_on_costs",
        "operative_part",
        "document_end",
    ]
    parts: List[str] = []
    for segment_name in ordered_segments:
        text = get_segment_text(data, segment_name)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


# =========================
# ===== ELIGIBILITY =======
# =========================


def is_query_candidate(data: Dict[str, Any]) -> bool:
    return (
        get_summary_text(data) != ""
        and get_segment_text(data, "facts_and_background") != ""
        and get_segment_text(data, "reasoning") != ""
    )


def is_doc_candidate(data: Dict[str, Any]) -> bool:
    return (
        get_summary_text(data) != ""
        and get_segment_text(data, "facts_and_background") != ""
        and get_segment_text(data, "reasoning") != ""
    )


# =========================
# ===== LOAD CORPUS =======
# =========================


def load_corpus(input_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    docs: Dict[str, Dict[str, Any]] = {}
    ecli_to_id: Dict[str, str] = {}

    for file_path in iter_json_files(input_dir):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = data.get("case_metadata") or {}
        doc_id = choose_doc_id(meta)
        if not doc_id:
            continue

        docs[doc_id] = {
            "_raw": data,
            "title": meta.get("case_name") or meta.get("case_number") or doc_id,
            "date": meta.get("date_of_judgment") or "",
            "court": meta.get("court") or "",
            "link": meta.get("document_link") or "",
            "summary": get_summary_text(data),
            "facts": get_segment_text(data, "facts_and_background"),
            "reasoning": get_segment_text(data, "reasoning"),
            "full": get_full_text(data),
        }

        ecli = canonical_ecli(meta.get("ecli"))
        if ecli:
            ecli_to_id[ecli] = doc_id

    return docs, ecli_to_id


# =========================
# ===== CITATIONS =========
# =========================


def positives_from_citations(
    raw: Dict[str, Any],
    ecli_to_id: Dict[str, str],
    docs: Dict[str, Dict[str, Any]],
    require_doc_candidate: bool = True,
) -> List[str]:
    citations = (raw.get("citations") or {}).get("list") or []
    seen: Set[str] = set()
    out: List[str] = []

    for citation in citations:
        ecli = canonical_ecli(citation)
        if not ecli:
            continue

        doc_id = ecli_to_id.get(ecli)
        if not doc_id:
            continue

        if require_doc_candidate and not is_doc_candidate(docs[doc_id]["_raw"]):
            continue

        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)

    return out


# =========================
# ===== STRATIFICATION ====
# =========================


def stratified_sample(
    docs: Dict[str, Dict[str, Any]],
    ecli_to_id: Dict[str, str],
    sample_size: int,
    rng: random.Random,
) -> List[str]:
    buckets: Dict[int, List[str]] = {1: [], 3: [], 6: [], 10: []}

    for doc_id, record in docs.items():
        raw = record["_raw"]
        if not is_query_candidate(raw):
            continue

        positives = [
            candidate
            for candidate in positives_from_citations(raw, ecli_to_id, docs, require_doc_candidate=True)
            if candidate != doc_id
        ]
        count = len(positives)
        if count == 0:
            continue

        bucket = 10 if count >= 10 else 6 if count >= 6 else 3 if count >= 3 else 1
        buckets[bucket].append(doc_id)

    chosen: List[str] = []
    per_bucket = max(1, sample_size // 4)

    for bucket in [1, 3, 6, 10]:
        rng.shuffle(buckets[bucket])
        chosen.extend(buckets[bucket][:per_bucket])

    if len(chosen) < sample_size:
        rest: List[str] = []
        for bucket in [1, 3, 6, 10]:
            rest.extend([doc_id for doc_id in buckets[bucket] if doc_id not in chosen])
        rng.shuffle(rest)
        chosen.extend(rest[: sample_size - len(chosen)])

    return chosen[:sample_size]


# =========================
# ===== SPARSE RETRIEVAL ==
# =========================


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


def build_text_for_retrieval(doc: Dict[str, Any]) -> str:
    return doc.get("full") or ""


def tfidf_build_index(corpus_texts: Dict[str, str]) -> Tuple[Dict[str, Counter], Counter, int]:
    tf_by_doc: Dict[str, Counter] = {}
    df: Counter = Counter()
    total_docs = len(corpus_texts)

    for doc_id, text in corpus_texts.items():
        tokens = tokenize(text)
        tf = Counter(tokens)
        tf_by_doc[doc_id] = tf
        for term in tf:
            df[term] += 1

    return tf_by_doc, df, total_docs


def tfidf_rank(
    query_text: str,
    tf_by_doc: Dict[str, Counter],
    df: Counter,
    total_docs: int,
    exclude: Set[str],
    topk: int,
) -> List[Tuple[str, float]]:
    query_terms = tokenize(query_text)
    if not query_terms:
        return []

    q_tf = Counter(query_terms)
    scores: DefaultDict[str, float] = defaultdict(float)

    for term, _ in q_tf.items():
        df_t = df.get(term, 0)
        if df_t == 0:
            continue
        idf = math.log((total_docs + 1) / (df_t + 1)) + 1.0

        for doc_id, tf in tf_by_doc.items():
            if doc_id in exclude:
                continue
            tf_td = tf.get(term, 0)
            if tf_td > 0:
                scores[doc_id] += (1.0 + math.log(tf_td)) * idf

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topk]


def bm25_build_index(corpus_texts: Dict[str, str]) -> Tuple[DefaultDict[str, List[Tuple[str, int]]], Dict[str, int], int, Dict[str, int], float]:
    inverted: DefaultDict[str, List[Tuple[str, int]]] = defaultdict(list)
    df: Dict[str, int] = defaultdict(int)
    doc_lengths: Dict[str, int] = {}
    total_docs = len(corpus_texts)

    for doc_id, text in corpus_texts.items():
        tokens = tokenize(text)
        doc_lengths[doc_id] = len(tokens)
        tf: DefaultDict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        for token, freq in tf.items():
            inverted[token].append((doc_id, freq))
            df[token] += 1

    avgdl = sum(doc_lengths.values()) / max(1, total_docs)
    return inverted, df, total_docs, doc_lengths, avgdl


def bm25_idf(total_docs: int, df_t: int) -> float:
    return math.log((total_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)


def bm25_rank(
    query_text: str,
    inv: DefaultDict[str, List[Tuple[str, int]]],
    df: Dict[str, int],
    total_docs: int,
    doc_lengths: Dict[str, int],
    avgdl: float,
    exclude: Set[str],
    topk: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    query_terms = set(tokenize(query_text))
    if not query_terms:
        return []

    scores: DefaultDict[str, float] = defaultdict(float)
    idf = {term: bm25_idf(total_docs, df.get(term, 0)) for term in query_terms if df.get(term, 0) > 0}

    for term, idf_t in idf.items():
        for doc_id, f_td in inv.get(term, []):
            if doc_id in exclude:
                continue
            denom = f_td + k1 * (1 - b + b * (doc_lengths.get(doc_id, 0) / max(1.0, avgdl)))
            scores[doc_id] += idf_t * (f_td * (k1 + 1)) / max(1e-9, denom)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topk]


# =========================
# ===== DENSE CHUNKING ====
# =========================


def format_query_text(text: str, cfg: Dict[str, Any]) -> str:
    fmt = cfg.get("query_prefix", "{}")
    return fmt.format(text) if "{}" in fmt else (fmt + text)


def format_document_text(text: str, cfg: Dict[str, Any]) -> str:
    fmt = cfg.get("document_prefix", "{}")
    return fmt.format(text) if "{}" in fmt else (fmt + text)


def _compute_chunk_starts_with_refined_backfill(
    total_len: int,
    chunk_size: int,
    overlap_size: int,
    min_backfill_fraction: float,
) -> List[int]:
    if total_len <= 0:
        return []
    if total_len <= chunk_size:
        return [0]

    stride = max(1, chunk_size - overlap_size)
    starts: List[int] = []
    start = 0

    while start < total_len:
        starts.append(start)
        end = min(start + chunk_size, total_len)
        if end >= total_len:
            break
        start += stride

    if not starts:
        return [0]

    last_start = starts[-1]
    last_len = total_len - last_start
    min_backfill_len = max(1, int(math.ceil(chunk_size * min_backfill_fraction)))

    if last_len < min_backfill_len:
        tail_start = max(0, total_len - chunk_size)
        if tail_start != last_start:
            starts[-1] = tail_start

    deduped: List[int] = []
    seen: Set[int] = set()
    for chunk_start in starts:
        if chunk_start not in seen:
            seen.add(chunk_start)
            deduped.append(chunk_start)
    return deduped


def raw_token_slice_hf(
    text: str,
    tokenizer: Any,
    max_chunk_tokens: int,
    overlap_tokens: int,
    min_backfill_fraction: float,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return []

    starts = _compute_chunk_starts_with_refined_backfill(
        total_len=len(token_ids),
        chunk_size=max_chunk_tokens,
        overlap_size=overlap_tokens,
        min_backfill_fraction=min_backfill_fraction,
    )

    chunks: List[str] = []
    for start in starts:
        end = min(start + max_chunk_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def raw_token_slice_approx_chars(
    text: str,
    max_chunk_tokens: int,
    overlap_tokens: int,
    chars_per_token: float,
    min_backfill_fraction: float,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    max_chunk_chars = max(1, int(round(max_chunk_tokens * chars_per_token)))
    overlap_chars = max(0, int(round(overlap_tokens * chars_per_token)))

    starts = _compute_chunk_starts_with_refined_backfill(
        total_len=len(text),
        chunk_size=max_chunk_chars,
        overlap_size=overlap_chars,
        min_backfill_fraction=min_backfill_fraction,
    )

    chunks: List[str] = []
    for start in starts:
        end = min(start + max_chunk_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


# =========================
# ===== DENSE ENCODING ====
# =========================


def encode_sentence_transformer(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on device: {device}")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return l2_normalize(np.asarray(vectors, dtype=np.float32))


def encode_gemini_documents(texts: List[str], model_name: str) -> np.ndarray:
    import time
    from google import genai
    from google.genai import types

    client = genai.Client()
    vectors: List[np.ndarray] = []
    total_batches = math.ceil(len(texts) / GEMINI_BATCH_SIZE)

    for batch_index, start in enumerate(range(0, len(texts), GEMINI_BATCH_SIZE), start=1):
        batch = texts[start:start + GEMINI_BATCH_SIZE]
        print(f"Gemini embedding (documents) batch {batch_index}/{total_batches} with {len(batch)} items")
        response = client.models.embed_content(
            model=model_name,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        for emb in response.embeddings:
            vectors.append(np.array(emb.values, dtype=np.float32))
        if GEMINI_REQUEST_DELAY_SECONDS > 0:
            time.sleep(GEMINI_REQUEST_DELAY_SECONDS)

    return l2_normalize(np.vstack(vectors))


def encode_gemini_queries(texts: List[str], model_name: str) -> np.ndarray:
    import time
    from google import genai
    from google.genai import types

    client = genai.Client()
    vectors: List[np.ndarray] = []
    total_batches = math.ceil(len(texts) / GEMINI_BATCH_SIZE)

    for batch_index, start in enumerate(range(0, len(texts), GEMINI_BATCH_SIZE), start=1):
        batch = texts[start:start + GEMINI_BATCH_SIZE]
        print(f"Gemini embedding (queries) batch {batch_index}/{total_batches} with {len(batch)} items")
        response = client.models.embed_content(
            model=model_name,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        for emb in response.embeddings:
            vectors.append(np.array(emb.values, dtype=np.float32))
        if GEMINI_REQUEST_DELAY_SECONDS > 0:
            time.sleep(GEMINI_REQUEST_DELAY_SECONDS)

    return l2_normalize(np.vstack(vectors))


def build_dense_hf_chunk_embeddings(
    texts_by_id: Dict[str, str],
    cfg: Dict[str, Any],
    is_query: bool,
    retriever_name: str,
    paths: PipelinePaths,
    min_backfill_fraction: float,
) -> Dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    model_name = cfg["model_name"]
    batch_size = int(cfg.get("batch_size", 8))
    expected_dim = int(cfg.get("embedding_dim", 0))
    max_chunk_tokens, overlap_tokens = get_effective_chunk_params(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    cache_dir, manifest_path = get_embedding_cache_paths(paths, retriever_name, is_query)
    manifest = load_manifest(manifest_path)

    out: Dict[str, np.ndarray] = {}
    missing_items: List[Tuple[str, List[str], str]] = []
    total_items = len(texts_by_id)

    for idx, (doc_id, text) in enumerate(texts_by_id.items(), start=1):
        raw_chunks = raw_token_slice_hf(
            text=text,
            tokenizer=tokenizer,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
            min_backfill_fraction=min_backfill_fraction,
        )
        formatted_chunks = [format_query_text(ch, cfg) for ch in raw_chunks] if is_query else [format_document_text(ch, cfg) for ch in raw_chunks]
        if not formatted_chunks:
            formatted_chunks = [format_query_text("", cfg) if is_query else format_document_text("", cfg)]

        text_hash = stable_text_hash("\n<CHUNK_BOUNDARY>\n".join(formatted_chunks))
        cached = load_cached_embedding(cache_dir, manifest, doc_id, text_hash, len(formatted_chunks))

        if cached is not None:
            if expected_dim > 0 and cached.shape[1] != expected_dim:
                raise ValueError(
                    f"Cached embedding dimension mismatch for {retriever_name} ({'query' if is_query else 'doc'}) {doc_id}: expected {expected_dim}, got {cached.shape[1]}"
                )
            out[doc_id] = cached
        else:
            missing_items.append((doc_id, formatted_chunks, text_hash))

        if idx % 100 == 0 or idx == total_items:
            print(f"Prepared chunks {idx}/{total_items} for {retriever_name} ({'query' if is_query else 'doc'})")

    if missing_items:
        total_missing = len(missing_items)
        for start in range(0, total_missing, batch_size):
            batch_items = missing_items[start:start + batch_size]
            batch_chunk_texts: List[str] = []
            batch_meta: List[Tuple[str, int, str]] = []

            for doc_id, formatted_chunks, text_hash in batch_items:
                batch_chunk_texts.extend(formatted_chunks)
                batch_meta.append((doc_id, len(formatted_chunks), text_hash))

            encoded = encode_sentence_transformer(
                texts=batch_chunk_texts,
                model_name=model_name,
                batch_size=batch_size,
            )
            if expected_dim > 0 and encoded.shape[1] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for {retriever_name} ({'query' if is_query else 'doc'}): expected {expected_dim}, got {encoded.shape[1]}"
                )

            cursor = 0
            for doc_id, n_chunks, text_hash in batch_meta:
                arr = encoded[cursor:cursor + n_chunks]
                cursor += n_chunks
                out[doc_id] = arr
                save_cached_embedding(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    doc_id=doc_id,
                    text_hash=text_hash,
                    arr=arr,
                    metadata={
                        "model_name": model_name,
                        "is_query": bool(is_query),
                        "retriever_name": retriever_name,
                        "max_chunk_tokens": int(max_chunk_tokens),
                        "overlap_tokens": int(overlap_tokens),
                        "embedding_dim": int(arr.shape[1]),
                    },
                )

            save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)
    return out


def build_dense_gemini_chunk_embeddings(
    texts_by_id: Dict[str, str],
    cfg: Dict[str, Any],
    is_query: bool,
    retriever_name: str,
    paths: PipelinePaths,
    min_backfill_fraction: float,
) -> Dict[str, np.ndarray]:
    model_name = cfg["model_name"]
    chars_per_token = float(cfg.get("chars_per_token", 4.0))
    batch_size = int(cfg.get("batch_size", 1))
    expected_dim = int(cfg.get("embedding_dim", 0))
    max_chunk_tokens, overlap_tokens = get_effective_chunk_params(cfg)

    cache_dir, manifest_path = get_embedding_cache_paths(paths, retriever_name, is_query)
    manifest = load_manifest(manifest_path)

    out: Dict[str, np.ndarray] = {}
    missing_items: List[Tuple[str, List[str], str]] = []
    total_items = len(texts_by_id)

    for idx, (doc_id, text) in enumerate(texts_by_id.items(), start=1):
        raw_chunks = raw_token_slice_approx_chars(
            text=text,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
            chars_per_token=chars_per_token,
            min_backfill_fraction=min_backfill_fraction,
        )
        formatted_chunks = [format_query_text(ch, cfg) for ch in raw_chunks] if is_query else [format_document_text(ch, cfg) for ch in raw_chunks]
        if not formatted_chunks:
            formatted_chunks = [format_query_text("", cfg) if is_query else format_document_text("", cfg)]

        text_hash = stable_text_hash("\n<CHUNK_BOUNDARY>\n".join(formatted_chunks))
        cached = load_cached_embedding(cache_dir, manifest, doc_id, text_hash, len(formatted_chunks))

        if cached is not None:
            if expected_dim > 0 and cached.shape[1] != expected_dim:
                raise ValueError(
                    f"Cached embedding dimension mismatch for {retriever_name} ({'query' if is_query else 'doc'}) {doc_id}: expected {expected_dim}, got {cached.shape[1]}"
                )
            out[doc_id] = cached
        else:
            missing_items.append((doc_id, formatted_chunks, text_hash))

        if idx % 100 == 0 or idx == total_items:
            print(f"Prepared chunks {idx}/{total_items} for {retriever_name} ({'query' if is_query else 'doc'})")

    if missing_items:
        total_missing = len(missing_items)
        for start in range(0, total_missing, batch_size):
            batch_items = missing_items[start:start + batch_size]
            batch_chunk_texts: List[str] = []
            batch_meta: List[Tuple[str, int, str]] = []

            for doc_id, formatted_chunks, text_hash in batch_items:
                batch_chunk_texts.extend(formatted_chunks)
                batch_meta.append((doc_id, len(formatted_chunks), text_hash))

            encoded = encode_gemini_queries(batch_chunk_texts, model_name) if is_query else encode_gemini_documents(batch_chunk_texts, model_name)
            if expected_dim > 0 and encoded.shape[1] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for {retriever_name} ({'query' if is_query else 'doc'}): expected {expected_dim}, got {encoded.shape[1]}"
                )

            cursor = 0
            for doc_id, n_chunks, text_hash in batch_meta:
                arr = encoded[cursor:cursor + n_chunks]
                cursor += n_chunks
                out[doc_id] = arr
                save_cached_embedding(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    doc_id=doc_id,
                    text_hash=text_hash,
                    arr=arr,
                    metadata={
                        "model_name": model_name,
                        "is_query": bool(is_query),
                        "retriever_name": retriever_name,
                        "max_chunk_tokens": int(max_chunk_tokens),
                        "overlap_tokens": int(overlap_tokens),
                        "chars_per_token": float(chars_per_token),
                        "embedding_dim": int(arr.shape[1]),
                    },
                )

            save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)
    return out


# =========================
# ===== DENSE SCORING =====
# =========================


def score_query_chunk_max_then_average(query_chunk_matrix: np.ndarray, candidate_chunk_matrix: np.ndarray) -> float:
    if query_chunk_matrix.size == 0 or candidate_chunk_matrix.size == 0:
        return float("-inf")
    similarities = query_chunk_matrix @ candidate_chunk_matrix.T
    best_per_query_chunk = np.max(similarities, axis=1)
    return float(np.mean(best_per_query_chunk))


def dense_rank_chunked_strategy3(
    query_chunk_matrix: np.ndarray,
    candidate_chunk_matrices: Dict[str, np.ndarray],
    exclude: Set[str],
    topk: int,
) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for doc_id, candidate_matrix in candidate_chunk_matrices.items():
        if doc_id in exclude:
            continue
        score = score_query_chunk_max_then_average(query_chunk_matrix, candidate_matrix)
        items.append((doc_id, score))
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:topk]


# =========================
# ===== STAGE 1 BUILD =====
# =========================


def build_query_pool_and_citation_candidates(
    docs: Dict[str, Dict[str, Any]],
    ecli_to_id: Dict[str, str],
    paths: PipelinePaths,
    query_sample_size: int,
    rng: random.Random,
) -> List[str]:
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    paths.query_pool_path.parent.mkdir(parents=True, exist_ok=True)
    paths.qrels_citations_path.parent.mkdir(parents=True, exist_ok=True)

    positives: Dict[str, List[str]] = {}
    for doc_id, record in docs.items():
        pos = [
            candidate
            for candidate in positives_from_citations(record["_raw"], ecli_to_id, docs, require_doc_candidate=True)
            if candidate != doc_id
        ]
        if pos:
            positives[doc_id] = pos

    candidates = [doc_id for doc_id in positives if is_query_candidate(docs[doc_id]["_raw"])]
    if not candidates:
        raise SystemExit("No documents meet the eligibility rules.")

    queries = stratified_sample(docs, ecli_to_id, min(query_sample_size, len(candidates)), rng)

    write_jsonl(
        paths.query_pool_path,
        (
            {
                "query_id": query_id,
                "summary": docs[query_id]["summary"],
                "facts": docs[query_id]["facts"],
                "reasoning": docs[query_id]["reasoning"],
                "full": docs[query_id]["full"],
            }
            for query_id in queries
        ),
    )

    with paths.citation_candidates_path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tcandidate_id\tsource\n")
        for query_id in queries:
            for positive_id in positives.get(query_id, []):
                handle.write(f"{query_id}\t{positive_id}\tcitation\n")

    with paths.qrels_citations_path.open("w", encoding="utf-8") as handle:
        for query_id in queries:
            for positive_id in positives.get(query_id, []):
                handle.write(f"{query_id}\t0\t{positive_id}\t1\n")

    return queries


# =========================
# ===== STAGE 2 BUILD =====
# =========================


def build_candidate_pool_files(
    docs: Dict[str, Dict[str, Any]],
    paths: PipelinePaths,
    retriever_configs: Dict[str, Dict[str, Any]],
    random_negatives: int,
    rng: random.Random,
    min_backfill_fraction: float,
    dense_aggregation: str,
) -> None:
    paths.pooling_dir.mkdir(parents=True, exist_ok=True)
    paths.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    query_recs = read_jsonl(paths.query_pool_path)
    query_ids = [row["query_id"] for row in query_recs if row.get("query_id") in docs]
    citation_map = read_citation_candidates(paths.citation_candidates_path)

    corpus_texts = {doc_id: build_text_for_retrieval(record) for doc_id, record in docs.items()}
    query_texts = {query_id: build_text_for_retrieval(docs[query_id]) for query_id in query_ids}

    enabled_retrievers = {name: cfg for name, cfg in retriever_configs.items() if cfg.get("enabled", False)}
    sparse_indexes: Dict[str, Any] = {}
    dense_doc_chunk_matrices: Dict[str, Dict[str, np.ndarray]] = {}
    dense_query_chunk_matrices: Dict[str, Dict[str, np.ndarray]] = {}

    if any(cfg["type"] == "sparse_tfidf" for cfg in enabled_retrievers.values()):
        sparse_indexes["tfidf"] = tfidf_build_index(corpus_texts)
    if any(cfg["type"] == "sparse_bm25" for cfg in enabled_retrievers.values()):
        sparse_indexes["bm25"] = bm25_build_index(corpus_texts)

    for name, cfg in enabled_retrievers.items():
        if cfg["type"] == "dense_sentence_transformers":
            dense_doc_chunk_matrices[name] = build_dense_hf_chunk_embeddings(
                texts_by_id=corpus_texts,
                cfg=cfg,
                is_query=False,
                retriever_name=name,
                paths=paths,
                min_backfill_fraction=min_backfill_fraction,
            )
            dense_query_chunk_matrices[name] = build_dense_hf_chunk_embeddings(
                texts_by_id=query_texts,
                cfg=cfg,
                is_query=True,
                retriever_name=name,
                paths=paths,
                min_backfill_fraction=min_backfill_fraction,
            )
        elif cfg["type"] == "dense_gemini":
            dense_doc_chunk_matrices[name] = build_dense_gemini_chunk_embeddings(
                texts_by_id=corpus_texts,
                cfg=cfg,
                is_query=False,
                retriever_name=name,
                paths=paths,
                min_backfill_fraction=min_backfill_fraction,
            )
            dense_query_chunk_matrices[name] = build_dense_gemini_chunk_embeddings(
                texts_by_id=query_texts,
                cfg=cfg,
                is_query=True,
                retriever_name=name,
                paths=paths,
                min_backfill_fraction=min_backfill_fraction,
            )

    citation_rows: List[Tuple[str, str, str]] = []
    retriever_rows: Dict[str, List[Tuple[str, str, str, float]]] = {name: [] for name in enabled_retrievers}
    random_rows: List[Tuple[str, str, str]] = []

    retriever_diagnostics: Dict[str, Any] = {}
    for name, cfg in enabled_retrievers.items():
        cfg_copy = dict(cfg)
        if cfg["type"] in {"dense_sentence_transformers", "dense_gemini"}:
            effective_chunk_tokens, overlap_tokens = get_effective_chunk_params(cfg)
            cfg_copy["effective_chunk_tokens"] = effective_chunk_tokens
            cfg_copy["overlap_tokens"] = overlap_tokens
            cfg_copy["chunking"] = "raw_slicing"
            cfg_copy["refined_backfill_threshold_fraction"] = min_backfill_fraction
        retriever_diagnostics[name] = cfg_copy

    diagnostics: Dict[str, Any] = {
        "queries_total": len(query_ids),
        "random_negatives": random_negatives,
        "random_seed": DEFAULT_RANDOM_SEED,
        "dense_aggregation": dense_aggregation,
        "dense_representation": {
            "query_text": "full",
            "candidate_text": "full",
            "chunking": "raw_slicing",
            "refined_backfill": {
                "enabled": True,
                "threshold_fraction": min_backfill_fraction,
                "behavior": "replace_final_chunk_with_tail_aligned_full_chunk_when_final_chunk_is_too_short",
            },
        },
        "retrievers": retriever_diagnostics,
        "per_query_counts": {},
        "embedding_cache_dir": str(paths.embedding_cache_dir),
    }

    doc_ids = list(corpus_texts.keys())

    for index, query_id in enumerate(query_ids, start=1):
        query_text = query_texts[query_id]
        citation_ids = [candidate for candidate in citation_map.get(query_id, []) if candidate != query_id and candidate in docs]
        citation_ids = dedupe_preserve_order(citation_ids)

        for candidate_id in citation_ids:
            citation_rows.append((query_id, candidate_id, "citation"))

        exclude_for_retrieval = set(citation_ids) | {query_id}
        current_query_retrieved_ids: Dict[str, List[str]] = {}
        per_query_counts: Dict[str, int] = {"citation": len(citation_ids), "random": 0}

        for name, cfg in enabled_retrievers.items():
            topk = int(cfg["topk"])
            hits: List[Tuple[str, float]]

            if cfg["type"] == "sparse_tfidf":
                tf_by_doc, df, total_docs = sparse_indexes["tfidf"]
                hits = tfidf_rank(query_text, tf_by_doc, df, total_docs, exclude_for_retrieval, topk)
            elif cfg["type"] == "sparse_bm25":
                inv, df, total_docs, doc_lengths, avgdl = sparse_indexes["bm25"]
                hits = bm25_rank(query_text, inv, df, total_docs, doc_lengths, avgdl, exclude_for_retrieval, topk)
            elif cfg["type"] in {"dense_sentence_transformers", "dense_gemini"}:
                hits = dense_rank_chunked_strategy3(
                    query_chunk_matrix=dense_query_chunk_matrices[name][query_id],
                    candidate_chunk_matrices=dense_doc_chunk_matrices[name],
                    exclude=exclude_for_retrieval,
                    topk=topk,
                )
            else:
                raise ValueError(f"Unsupported retriever type: {cfg['type']}")

            current_query_retrieved_ids[name] = [candidate_id for candidate_id, _ in hits]
            for candidate_id, score in hits:
                retriever_rows[name].append((query_id, candidate_id, name, float(score)))
            per_query_counts[name] = len(hits)

        used: Set[str] = set(citation_ids)
        used.add(query_id)
        for name in enabled_retrievers:
            used.update(current_query_retrieved_ids.get(name, []))

        random_pool = [doc_id for doc_id in doc_ids if doc_id not in used]
        rng.shuffle(random_pool)
        sampled_randoms = random_pool[:random_negatives]
        for candidate_id in sampled_randoms:
            random_rows.append((query_id, candidate_id, "random"))

        per_query_counts["random"] = len(sampled_randoms)
        diagnostics["per_query_counts"][query_id] = per_query_counts

        if index % 10 == 0 or index == len(query_ids):
            print(f"Scored query {index}/{len(query_ids)}")

    write_pool_tsv_no_score(paths.citation_pool_path, citation_rows)
    for name, rows in retriever_rows.items():
        write_pool_tsv(pool_output_path(paths, name), rows)
    write_pool_tsv_no_score(paths.random_negatives_path, random_rows)
    paths.pool_config_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# ===== STAGE 3 UNION =====
# =========================


def load_query_ids(query_pool_path: Path) -> List[str]:
    return [row["query_id"] for row in read_jsonl(query_pool_path)]


def get_pool_files(paths: PipelinePaths) -> Dict[str, Path]:
    return {
        "citation": paths.pooling_dir / "citation_pool.tsv",
        "tfidf": paths.pooling_dir / "tfidf_pool.tsv",
        "bm25": paths.pooling_dir / "bm25_pool.tsv",
        "minilm": paths.pooling_dir / "minilm_pool.tsv",
        "qwen": paths.pooling_dir / "qwen_pool.tsv",
        "gemini": paths.pooling_dir / "gemini_pool.tsv",
        "random": paths.pooling_dir / "random_negatives.tsv",
    }


def read_topk_per_query(path: Path, source_name: str, top_k_per_pool: int, random_top_k_per_pool: int) -> Dict[str, List[Dict[str, str]]]:
    per_query: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    if not path.exists():
        print(f"Skipping missing: {path.name}")
        return per_query

    limit = random_top_k_per_pool if source_name == "random" else top_k_per_pool
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = row["query_id"]
            if len(per_query[query_id]) >= limit:
                continue
            per_query[query_id].append(
                {
                    "candidate_id": row["candidate_id"],
                    "source": source_name,
                    "score": row.get("score", ""),
                }
            )
    return per_query


def build_unified_candidate_pool(
    paths: PipelinePaths,
    top_k_per_pool: int,
    random_top_k_per_pool: int,
) -> None:
    query_ids = load_query_ids(paths.query_pool_path)
    pool_files = get_pool_files(paths)

    union: DefaultDict[str, DefaultDict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"sources": set(), "scores": {}})
    )

    for source_name, path in pool_files.items():
        per_query = read_topk_per_query(path, source_name, top_k_per_pool, random_top_k_per_pool)
        for query_id, rows in per_query.items():
            if query_id not in query_ids:
                continue
            for row in rows:
                candidate_id = row["candidate_id"]
                entry = union[query_id][candidate_id]
                entry["sources"].add(source_name)
                if row["score"]:
                    entry["scores"][source_name] = row["score"]

    all_score_sources: Set[str] = set()
    for query_candidates in union.values():
        for candidate_meta in query_candidates.values():
            all_score_sources.update(candidate_meta["scores"].keys())
    score_fields = [f"score_{name}" for name in sorted(all_score_sources)]

    paths.candidate_pool_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.candidate_pool_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "candidate_id", "source_primary", "source_all", *score_fields],
            delimiter="\t",
        )
        writer.writeheader()

        for query_id in query_ids:
            for candidate_id, meta in union.get(query_id, {}).items():
                sources = meta["sources"]
                row = {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "source_primary": choose_primary_source(sources),
                    "source_all": "|".join(sorted(sources)),
                }
                for score_field in score_fields:
                    source_key = score_field.replace("score_", "")
                    row[score_field] = meta["scores"].get(source_key, "")
                writer.writerow(row)


# =========================
# ====== ORCHESTRATION ====
# =========================


def load_retriever_configs(config_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if config_path is None:
        return json.loads(json.dumps(DEFAULT_RETRIEVER_CONFIGS))
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def run_stage1(args: SimpleNamespace) -> None:
    rng = random.Random(args.random_seed)
    paths = build_paths(args.input_dir, args.eval_dir)
    docs, ecli_to_id = load_corpus(paths.input_dir)
    build_query_pool_and_citation_candidates(
        docs=docs,
        ecli_to_id=ecli_to_id,
        paths=paths,
        query_sample_size=args.query_sample_size,
        rng=rng,
    )
    print(f"Wrote {paths.query_pool_path}")
    print(f"Wrote {paths.citation_candidates_path}")
    print(f"Wrote {paths.qrels_citations_path}")


def run_stage2(args: SimpleNamespace) -> None:
    rng = random.Random(args.random_seed)
    paths = build_paths(args.input_dir, args.eval_dir)
    retriever_configs = load_retriever_configs(args.retriever_config)
    docs, _ = load_corpus(paths.input_dir)
    build_candidate_pool_files(
        docs=docs,
        paths=paths,
        retriever_configs=retriever_configs,
        random_negatives=args.random_negatives,
        rng=rng,
        min_backfill_fraction=args.backfill_min_fraction,
        dense_aggregation=DEFAULT_DENSE_AGGREGATION,
    )
    print(f"Wrote pooling files under {paths.pooling_dir}")


def run_stage3(args: SimpleNamespace) -> None:
    paths = build_paths(args.input_dir, args.eval_dir)
    build_unified_candidate_pool(
        paths=paths,
        top_k_per_pool=args.top_k_per_pool,
        random_top_k_per_pool=args.random_top_k_per_pool,
    )
    print(f"Wrote {paths.candidate_pool_path}")


def run_pipeline(args: SimpleNamespace) -> None:
    rng = random.Random(args.random_seed)
    paths = build_paths(args.input_dir, args.eval_dir)
    retriever_configs = load_retriever_configs(args.retriever_config)

    docs, ecli_to_id = load_corpus(paths.input_dir)
    if stage1_outputs_exist(paths) and not args.force_stage1:
        print("Stage 1 outputs already exist. Reusing them.")
    else:
        build_query_pool_and_citation_candidates(
            docs=docs,
            ecli_to_id=ecli_to_id,
            paths=paths,
            query_sample_size=args.query_sample_size,
            rng=rng,
        )

    build_candidate_pool_files(
        docs=docs,
        paths=paths,
        retriever_configs=retriever_configs,
        random_negatives=args.random_negatives,
        rng=rng,
        min_backfill_fraction=args.backfill_min_fraction,
        dense_aggregation=DEFAULT_DENSE_AGGREGATION,
    )
    build_unified_candidate_pool(
        paths=paths,
        top_k_per_pool=args.top_k_per_pool,
        random_top_k_per_pool=args.random_top_k_per_pool,
    )

    print("Done.")
    print(f"- {paths.query_pool_path}")
    print(f"- {paths.citation_candidates_path}")
    print(f"- {paths.qrels_citations_path}")
    print(f"- pooling/* under {paths.pooling_dir}")
    print(f"- {paths.candidate_pool_path}")
    print(f"- embedding cache under {paths.embedding_cache_dir}")


# =========================
# ========= CLI ===========
# =========================




# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
RUN_MODE = "pipeline"  # stage1, stage2, stage3, or pipeline

INPUT_JSON_DIRECTORY = Path("data/enriched_json")
DATA_DIRECTORY = Path("data")
RETRIEVER_CONFIG_PATH: Path | None = None

QUERY_SAMPLE_SIZE = DEFAULT_QUERY_SAMPLE_SIZE
RANDOM_NEGATIVES = DEFAULT_RANDOM_NEGATIVES
BACKFILL_MIN_FRACTION = DEFAULT_BACKFILL_MIN_FRACTION
TOP_K_PER_POOL = DEFAULT_TOP_K_PER_POOL
RANDOM_TOP_K_PER_POOL = DEFAULT_RANDOM_TOP_K_PER_POOL
FORCE_STAGE1 = False
RANDOM_SEED = 123


def build_ide_arguments(project_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_dir=resolve_project_path(project_root, INPUT_JSON_DIRECTORY),
        eval_dir=resolve_project_path(project_root, DATA_DIRECTORY),
        retriever_config=(
            resolve_project_path(project_root, RETRIEVER_CONFIG_PATH)
            if RETRIEVER_CONFIG_PATH is not None
            else None
        ),
        query_sample_size=QUERY_SAMPLE_SIZE,
        random_negatives=RANDOM_NEGATIVES,
        backfill_min_fraction=BACKFILL_MIN_FRACTION,
        top_k_per_pool=TOP_K_PER_POOL,
        random_top_k_per_pool=RANDOM_TOP_K_PER_POOL,
        force_stage1=FORCE_STAGE1,
        random_seed=RANDOM_SEED,
    )


def main() -> int:
    set_reproducible_seed(RANDOM_SEED)
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    args = build_ide_arguments(project_root)
    mode = RUN_MODE.strip().lower()
    commands = {
        "stage1": run_stage1,
        "stage2": run_stage2,
        "stage3": run_stage3,
        "pipeline": run_pipeline,
    }

    try:
        if mode not in commands:
            raise ValueError("RUN_MODE must be stage1, stage2, stage3, or pipeline.")
        commands[mode](args)
        return 0
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    main()
