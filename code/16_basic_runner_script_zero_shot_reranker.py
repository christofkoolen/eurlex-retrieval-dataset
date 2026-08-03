#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot reranker runner for EUR-LexRD.

This script reranks a first-stage TREC run, such as the output of:

    code/zero_shot_retrieval_run.py


The default adapter uses a Sentence Transformers CrossEncoder. To plug in a
different compatible reranker, change RERANKER_MODEL_NAME. To use another
model family or an external API, replace only:

    load_reranker()
    score_pairs()

The qrels are used only after reranking to calculate metrics. They are never
used for model training, model selection, or scoring.

Expected first-stage run format
-------------------------------
query_id Q0 document_id rank score run_name

Outputs
-------
results/zero_shot_reranking/<run name>/
    run.trec
    candidate_scores.tsv
    metrics.json
    per_query_metrics.tsv
    run_config.json

Dependencies
------------
numpy
sentence-transformers

Install when needed:
    python -m pip install numpy sentence-transformers
"""


import csv
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 123

PROJECT_ROOT_OVERRIDE: Path | None = None
EVAL_VIEWS_ROOT_OVERRIDE: Path | None = None

# First-stage run from script 15, relative to the repository root.
FIRST_STAGE_RUN = Path(
    "results/zero_shot/"
    "all-MiniLM-L6-v2_human_graded_summary_summary/"
    "run.trec"
)

# These settings must match the first-stage run.
QRELS_SOURCE = "human"          # "human" or "citations"
QUERY_VIEW = "summary"          # "summary", "facts", or "full"
CORPUS_VIEW = "summary"            # "summary", "facts", "reasoning", or "full"
USE_GRADED_HUMAN_QRELS = True

# Default zero-shot CrossEncoder reranker.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_LABEL = "ms-marco-MiniLM-L6-v2"

# None lets Sentence Transformers choose the device.
# Examples: "cpu", "cuda", "cuda:0", "mps"
DEVICE: str | None = None
TRUST_REMOTE_CODE = False
BATCH_SIZE = 8
SHOW_PROGRESS_BAR = True

# None uses the model's configured maximum sequence length.
MAX_LENGTH: int | None = None

# Standard rerankers return one score per pair. For a multi-class model, set
# this to the column containing the relevance score.
RERANKER_SCORE_COLUMN: int | None = None

# Optional prefixes or instructions required by some rerankers.
QUERY_PREFIX = ""
DOCUMENT_PREFIX = ""

# Rerank this many first-stage documents per query.
RERANK_TOP_K = 100

# Write this many final documents per query. Documents below RERANK_TOP_K are
# appended in their original first-stage order.
OUTPUT_TOP_K = 100

METRIC_CUTOFFS: tuple[int, ...] = (5, 10, 20, 30)

# None means relevance >= 2 for graded human qrels and >= 1 otherwise.
BINARY_RELEVANCE_THRESHOLD: int | None = None

EXCLUDE_QUERY_DOCUMENT = True
OUTPUT_ROOT = Path("results/zero_shot_reranking")
OVERWRITE_OUTPUT = True


# =============================================================================
# Model adapter
# =============================================================================

def load_reranker() -> Any:
    """Load the reranker. Replace for another model family or API."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "Install the default adapter with:\n"
            "python -m pip install sentence-transformers"
        ) from exc

    arguments: dict[str, Any] = {
        "device": DEVICE,
        "trust_remote_code": TRUST_REMOTE_CODE,
    }

    if MAX_LENGTH is not None:
        arguments["max_length"] = MAX_LENGTH

    return CrossEncoder(RERANKER_MODEL_NAME, **arguments)


def score_pairs(
    model: Any,
    pairs: Sequence[tuple[str, str]],
) -> np.ndarray:
    """
    Return one relevance score per (query, document) pair.

    Replace this function for another model family or API.
    """
    prepared_pairs = [
        (
            QUERY_PREFIX + query_text,
            DOCUMENT_PREFIX + document_text,
        )
        for query_text, document_text in pairs
    ]

    scores = model.predict(
        prepared_pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=SHOW_PROGRESS_BAR,
        convert_to_numpy=True,
    )
    array = np.asarray(scores)

    if array.ndim == 2:
        if RERANKER_SCORE_COLUMN is None:
            if array.shape[1] != 1:
                raise ValueError(
                    "The reranker returned multiple scores per pair. Set "
                    "RERANKER_SCORE_COLUMN to the relevance-score column."
                )
            array = array[:, 0]
        else:
            if not 0 <= RERANKER_SCORE_COLUMN < array.shape[1]:
                raise ValueError(
                    "RERANKER_SCORE_COLUMN is outside the returned score "
                    f"matrix with shape {array.shape}."
                )
            array = array[:, RERANKER_SCORE_COLUMN]

    array = np.asarray(array, dtype=np.float64).reshape(-1)

    if len(array) != len(pairs):
        raise ValueError(
            "score_pairs() must return exactly one score per input pair."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("The reranker returned a non-finite score.")

    return array


# =============================================================================
# Paths and validation
# =============================================================================

VALID_QRELS_SOURCES = {"human", "citations"}
VALID_QUERY_VIEWS = {"summary", "facts", "full"}
VALID_CORPUS_VIEWS = {"summary", "facts", "reasoning", "full"}


def validate_configuration() -> None:
    if QRELS_SOURCE not in VALID_QRELS_SOURCES:
        raise ValueError(
            f"QRELS_SOURCE must be one of {sorted(VALID_QRELS_SOURCES)}."
        )
    if QUERY_VIEW not in VALID_QUERY_VIEWS:
        raise ValueError(
            f"QUERY_VIEW must be one of {sorted(VALID_QUERY_VIEWS)}."
        )
    if CORPUS_VIEW not in VALID_CORPUS_VIEWS:
        raise ValueError(
            f"CORPUS_VIEW must be one of {sorted(VALID_CORPUS_VIEWS)}."
        )
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")
    if RERANK_TOP_K <= 0 or OUTPUT_TOP_K <= 0:
        raise ValueError("RERANK_TOP_K and OUTPUT_TOP_K must be positive.")
    if MAX_LENGTH is not None and MAX_LENGTH <= 0:
        raise ValueError("MAX_LENGTH must be None or positive.")
    if not METRIC_CUTOFFS or any(k <= 0 for k in METRIC_CUTOFFS):
        raise ValueError("METRIC_CUTOFFS must contain positive integers.")


def find_project_root() -> Path:
    if PROJECT_ROOT_OVERRIDE is not None:
        root = PROJECT_ROOT_OVERRIDE.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root

    candidates: list[Path] = []

    if "__file__" in globals():
        script_path = Path(__file__).expanduser().resolve()
        candidates.extend([script_path.parent, *script_path.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for candidate in dict.fromkeys(candidates):
        if (
            (candidate / "data" / "eval_views").is_dir()
            or (candidate / "eval_views").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Repository root not found. Set PROJECT_ROOT_OVERRIDE."
    )


def find_eval_views_root(project_root: Path) -> Path:
    if EVAL_VIEWS_ROOT_OVERRIDE is not None:
        path = EVAL_VIEWS_ROOT_OVERRIDE.expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    for path in (
        project_root / "data" / "eval_views",
        project_root / "eval_views",
    ):
        if path.is_dir():
            return path.resolve()

    raise FileNotFoundError(
        "No data/eval_views or eval_views directory was found."
    )


def resolve_project_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "reranker"


# =============================================================================
# Readers
# =============================================================================

def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            yield record


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}

    for record in iter_jsonl(path):
        query_id = str(record.get("query_id", "")).strip()
        text = str(record.get("text", "")).strip()

        if not query_id or not text:
            raise ValueError(
                f"A query in {path} has a blank query_id or text."
            )
        if query_id in queries:
            raise ValueError(f"Duplicate query_id in {path}: {query_id}")

        queries[query_id] = text

    if not queries:
        raise ValueError(f"No queries found in {path}.")

    return queries


def load_first_stage_run(
    path: Path,
) -> tuple[dict[str, list[tuple[str, float]]], str]:
    if not path.is_file():
        raise FileNotFoundError(
            "First-stage run not found. Run script 15 first or update "
            f"FIRST_STAGE_RUN:\n{path}"
        )

    rows_by_query: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    run_names: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) != 6:
                raise ValueError(
                    f"Expected six TREC run columns in {path} at line "
                    f"{line_number}; found {len(columns)}."
                )

            query_id, _q0, document_id, rank_text, score_text, run_name = columns

            try:
                rank = int(rank_text)
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid rank or score in {path} at line {line_number}."
                ) from exc

            pair = (query_id, document_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate run pair: {query_id} -> {document_id}"
                )

            seen_pairs.add(pair)
            run_names.add(run_name)
            rows_by_query[query_id].append((rank, document_id, score))

    if not rows_by_query:
        raise ValueError(f"No rankings found in {path}.")

    rankings: dict[str, list[tuple[str, float]]] = {}

    for query_id, rows in rows_by_query.items():
        rows.sort(key=lambda row: (row[0], -row[2], row[1]))
        rankings[query_id] = [
            (document_id, score)
            for _rank, document_id, score in rows
        ]

    first_stage_name = (
        next(iter(run_names)) if len(run_names) == 1 else path.stem
    )
    return rankings, first_stage_name


def load_selected_documents(
    corpus_path: Path,
    required_document_ids: set[str],
) -> dict[str, str]:
    documents: dict[str, str] = {}

    for record in iter_jsonl(corpus_path):
        document_id = str(record.get("doc_id", "")).strip()
        if document_id not in required_document_ids:
            continue

        text = str(record.get("text", "")).strip()
        if not text:
            raise ValueError(
                f"Required reranking document has empty text: {document_id}"
            )
        if document_id in documents:
            raise ValueError(f"Duplicate corpus document: {document_id}")

        documents[document_id] = text
        if len(documents) == len(required_document_ids):
            break

    missing = sorted(required_document_ids - set(documents))
    if missing:
        raise ValueError(
            f"{len(missing)} first-stage document(s) were not found in "
            f"{corpus_path}. Examples: {missing[:20]}"
        )

    return documents


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    qrels: dict[str, dict[str, int]] = defaultdict(dict)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")

        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(
                    f"Expected four qrels columns in {path} at line "
                    f"{line_number}."
                )

            query_id = str(row[0]).strip()
            document_id = str(row[2]).strip()

            try:
                relevance = int(str(row[3]).strip())
            except ValueError as exc:
                if line_number == 1 and str(row[3]).strip().lower() == "relevance":
                    continue
                raise ValueError(
                    f"Invalid relevance in {path} at line {line_number}."
                ) from exc

            if document_id in qrels[query_id]:
                raise ValueError(
                    f"Duplicate qrel pair: {query_id} -> {document_id}"
                )
            qrels[query_id][document_id] = relevance

    if not qrels:
        raise ValueError(f"No qrels found in {path}.")

    return dict(qrels)


# =============================================================================
# Reranking
# =============================================================================

def select_reranking_documents(
    queries: Mapping[str, str],
    first_stage: Mapping[str, Sequence[tuple[str, float]]],
) -> set[str]:
    required: set[str] = set()

    for query_id in queries:
        count = 0
        for document_id, _score in first_stage[query_id]:
            if EXCLUDE_QUERY_DOCUMENT and document_id == query_id:
                continue
            required.add(document_id)
            count += 1
            if count >= RERANK_TOP_K:
                break

    return required


def rerank(
    model: Any,
    queries: Mapping[str, str],
    documents: Mapping[str, str],
    first_stage: Mapping[str, Sequence[tuple[str, float]]],
) -> tuple[
    dict[str, list[tuple[str, float]]],
    list[dict[str, object]],
]:
    reranked: dict[str, list[tuple[str, float]]] = {}
    score_rows: list[dict[str, object]] = []

    for query_number, (query_id, query_text) in enumerate(
        queries.items(), start=1
    ):
        original = [
            (document_id, score)
            for document_id, score in first_stage[query_id]
            if not (EXCLUDE_QUERY_DOCUMENT and document_id == query_id)
        ]
        head = original[:RERANK_TOP_K]
        tail = original[RERANK_TOP_K:OUTPUT_TOP_K]

        pairs = [
            (query_text, documents[document_id])
            for document_id, _score in head
        ]
        reranker_scores = score_pairs(model, pairs)

        scored_head = [
            (
                document_id,
                first_stage_score,
                float(reranker_score),
                first_stage_rank,
            )
            for first_stage_rank, (
                (document_id, first_stage_score),
                reranker_score,
            ) in enumerate(zip(head, reranker_scores), start=1)
        ]
        scored_head.sort(key=lambda item: (-item[2], item[3], item[0]))

        final = [
            (document_id, reranker_score)
            for document_id, _first_score, reranker_score, _first_rank
            in scored_head
        ]
        final.extend(
            (document_id, float("nan"))
            for document_id, _score in tail
        )
        final = final[:OUTPUT_TOP_K]
        reranked[query_id] = final

        final_rank = {
            document_id: rank
            for rank, (document_id, _score) in enumerate(final, start=1)
        }

        for document_id, first_score, reranker_score, first_rank in scored_head:
            score_rows.append(
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "first_stage_rank": first_rank,
                    "first_stage_score": first_score,
                    "reranker_score": reranker_score,
                    "final_rank": final_rank[document_id],
                    "reranked": 1,
                }
            )

        for first_rank, (document_id, first_score) in enumerate(
            tail, start=RERANK_TOP_K + 1
        ):
            if document_id in final_rank:
                score_rows.append(
                    {
                        "query_id": query_id,
                        "document_id": document_id,
                        "first_stage_rank": first_rank,
                        "first_stage_score": first_score,
                        "reranker_score": "",
                        "final_rank": final_rank[document_id],
                        "reranked": 0,
                    }
                )

        if query_number % 10 == 0 or query_number == len(queries):
            print(f"Reranked {query_number:,}/{len(queries):,} queries.")

    return reranked, score_rows


# =============================================================================
# Metrics
# =============================================================================

def precision_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    threshold: int,
    k: int,
) -> float:
    return sum(
        qrels.get(document_id, 0) >= threshold
        for document_id in ranked[:k]
    ) / k


def average_precision_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    threshold: int,
    k: int,
) -> float:
    total_relevant = sum(relevance >= threshold for relevance in qrels.values())
    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    score = 0.0

    for rank, document_id in enumerate(ranked[:k], start=1):
        if qrels.get(document_id, 0) >= threshold:
            relevant_seen += 1
            score += relevant_seen / rank

    return score / min(total_relevant, k)


def dcg(relevances: Sequence[int]) -> float:
    return sum(
        ((2.0 ** relevance) - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def ndcg_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    actual = [qrels.get(document_id, 0) for document_id in ranked[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 0.0


def judged_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    return sum(document_id in qrels for document_id in ranked[:k]) / k


def evaluate_comparison(
    query_ids: Sequence[str],
    first_stage: Mapping[str, Sequence[tuple[str, float]]],
    reranked: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    threshold: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    rows: list[dict[str, object]] = []

    for query_id in query_ids:
        query_qrels = qrels[query_id]
        first_ids = [document_id for document_id, _ in first_stage[query_id]]
        reranked_ids = [document_id for document_id, _ in reranked[query_id]]
        row: dict[str, object] = {"query_id": query_id}

        for k in METRIC_CUTOFFS:
            first_metrics = {
                f"P@{k}": precision_at_k(first_ids, query_qrels, threshold, k),
                f"MAP@{k}": average_precision_at_k(
                    first_ids, query_qrels, threshold, k
                ),
                f"nDCG@{k}": ndcg_at_k(first_ids, query_qrels, k),
                f"Judged@{k}": judged_at_k(first_ids, query_qrels, k),
            }
            reranked_metrics = {
                f"P@{k}": precision_at_k(reranked_ids, query_qrels, threshold, k),
                f"MAP@{k}": average_precision_at_k(
                    reranked_ids, query_qrels, threshold, k
                ),
                f"nDCG@{k}": ndcg_at_k(reranked_ids, query_qrels, k),
                f"Judged@{k}": judged_at_k(reranked_ids, query_qrels, k),
            }

            for metric_name, first_value in first_metrics.items():
                reranked_value = reranked_metrics[metric_name]
                row[f"first_{metric_name}"] = first_value
                row[f"reranked_{metric_name}"] = reranked_value
                row[f"delta_{metric_name}"] = reranked_value - first_value

        rows.append(row)

    metric_names = [
        key.removeprefix("first_")
        for key in rows[0]
        if key.startswith("first_")
    ]
    aggregate = {"first_stage": {}, "reranked": {}, "delta": {}}

    for metric_name in metric_names:
        first_value = float(np.mean([
            float(row[f"first_{metric_name}"]) for row in rows
        ]))
        reranked_value = float(np.mean([
            float(row[f"reranked_{metric_name}"]) for row in rows
        ]))
        aggregate["first_stage"][metric_name] = first_value
        aggregate["reranked"][metric_name] = reranked_value
        aggregate["delta"][metric_name] = reranked_value - first_value

    return rows, aggregate


# =============================================================================
# Outputs
# =============================================================================

def prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_trec_run(
    path: Path,
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    run_name: str,
) -> None:
    """
    Write rank-consistent synthetic scores.

    Raw reranker scores are retained in candidate_scores.tsv. Synthetic
    descending scores prevent appended, unscored documents from being sorted
    above reranked documents by evaluation software.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")

        for query_id, documents in rankings.items():
            document_count = len(documents)
            for rank, (document_id, _raw_score) in enumerate(
                documents, start=1
            ):
                writer.writerow(
                    [
                        query_id,
                        "Q0",
                        document_id,
                        rank,
                        f"{float(document_count - rank + 1):.8f}",
                        run_name,
                    ]
                )


def write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}.")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def set_random_seeds() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    validate_configuration()
    set_random_seeds()

    project_root = find_project_root()
    eval_views_root = find_eval_views_root(project_root)
    first_stage_path = resolve_project_path(project_root, FIRST_STAGE_RUN)

    view_name = f"q_{QUERY_VIEW}__c_{CORPUS_VIEW}"
    view_directory = eval_views_root / QRELS_SOURCE / view_name
    corpus_path = view_directory / "corpus.jsonl"
    queries_path = view_directory / "queries.jsonl"

    if QRELS_SOURCE == "human" and USE_GRADED_HUMAN_QRELS:
        qrels_path = view_directory / "qrels_selected_graded.tsv"
        qrels_label = "human_graded"
        default_threshold = 2
    else:
        qrels_path = view_directory / "qrels_selected.tsv"
        qrels_label = "human_binary" if QRELS_SOURCE == "human" else "citations"
        default_threshold = 1

    for path in (view_directory, corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(path)

    threshold = (
        BINARY_RELEVANCE_THRESHOLD
        if BINARY_RELEVANCE_THRESHOLD is not None
        else default_threshold
    )

    queries = load_queries(queries_path)
    first_stage, first_stage_name = load_first_stage_run(first_stage_path)
    qrels = load_qrels(qrels_path)

    expected_queries = set(queries)
    run_queries = set(first_stage)

    if expected_queries != run_queries:
        raise ValueError(
            "The first-stage run query IDs do not match the selected view.\n"
            f"Missing: {sorted(expected_queries - run_queries)[:20]}\n"
            f"Unexpected: {sorted(run_queries - expected_queries)[:20]}"
        )

    missing_qrels = [query_id for query_id in queries if query_id not in qrels]
    if missing_qrels:
        raise ValueError(f"Queries without qrels: {missing_qrels[:20]}")

    required_document_ids = select_reranking_documents(queries, first_stage)
    documents = load_selected_documents(corpus_path, required_document_ids)

    run_name = safe_name(
        f"{first_stage_name}__rerank_{RERANKER_LABEL}__"
        f"{qrels_label}__{QUERY_VIEW}_{CORPUS_VIEW}"
    )
    output_directory = project_root / OUTPUT_ROOT / run_name
    prepare_output_directory(output_directory)

    print(f"Evaluation view: {view_directory}")
    print(f"First-stage run: {first_stage_path}")
    print(f"Reranker: {RERANKER_MODEL_NAME}")
    print(f"Rerank depth: {RERANK_TOP_K}")
    print("Protocol: zero-shot reranking over all queries")
    print()

    model = load_reranker()
    reranked, candidate_rows = rerank(
        model, queries, documents, first_stage
    )
    per_query_rows, metrics = evaluate_comparison(
        list(queries), first_stage, reranked, qrels, threshold
    )

    write_trec_run(output_directory / "run.trec", reranked, run_name)
    write_table(output_directory / "candidate_scores.tsv", candidate_rows)
    write_table(output_directory / "per_query_metrics.tsv", per_query_rows)
    write_json(
        output_directory / "metrics.json",
        {
            "run_name": run_name,
            "query_count": len(queries),
            "binary_relevance_threshold": threshold,
            **metrics,
        },
    )
    write_json(
        output_directory / "run_config.json",
        {
            "protocol": "zero_shot_reranking_all_queries",
            "random_seed": RANDOM_SEED,
            "first_stage_run": str(first_stage_path.relative_to(project_root)),
            "first_stage_name": first_stage_name,
            "reranker_model_name": RERANKER_MODEL_NAME,
            "reranker_label": RERANKER_LABEL,
            "device": DEVICE,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "reranker_score_column": RERANKER_SCORE_COLUMN,
            "query_prefix": QUERY_PREFIX,
            "document_prefix": DOCUMENT_PREFIX,
            "qrels_source": qrels_label,
            "query_view": QUERY_VIEW,
            "corpus_view": CORPUS_VIEW,
            "rerank_top_k": RERANK_TOP_K,
            "output_top_k": OUTPUT_TOP_K,
            "metric_cutoffs": list(METRIC_CUTOFFS),
            "binary_relevance_threshold": threshold,
            "exclude_query_document": EXCLUDE_QUERY_DOCUMENT,
        },
    )

    print()
    print("Aggregate comparison")
    print("--------------------")

    for metric_name in sorted(
        metrics["reranked"],
        key=lambda name: (int(name.split("@")[1]), name.split("@")[0]),
    ):
        first_value = metrics["first_stage"][metric_name]
        reranked_value = metrics["reranked"][metric_name]
        delta = metrics["delta"][metric_name]
        print(
            f"{metric_name:>12}: {first_value:.6f} -> "
            f"{reranked_value:.6f} ({delta:+.6f})"
        )

    print()
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    main()
