#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised holdout bi-encoder runner for EUR-LexRD.

This script fine-tunes a Sentence Transformers bi-encoder using the released
70/10/20 query-level holdout:

    train: 70 queries
    dev:   10 queries
    test:  20 queries

The train qrels are used to create positive query-document pairs. The dev
partition is used for checkpoint selection with nDCG@10. The test partition is
kept untouched unless FINAL_TEST_RUN is explicitly set to True.

The script is designed to be opened and run directly in Spyder. It has no
command-line interface.

Canonical inputs
----------------
data/splits/holdout_70_10_20_v1/query_assignments.tsv

data/eval_views/<qrels source>/<view>/
    corpus.jsonl
    queries.jsonl
    qrels_selected.tsv
    qrels_selected_graded.tsv  # human views only

Outputs
-------
results/holdout_biencoder/<run name>/
    best_model/
    checkpoints/
    dev_metrics.json
    dev_run.trec
    dev_per_query_metrics.tsv
    test_metrics.json              # only when FINAL_TEST_RUN=True
    test_run.trec                  # only when FINAL_TEST_RUN=True
    test_per_query_metrics.tsv     # only when FINAL_TEST_RUN=True
    run_config.json
    training_pairs.tsv

Default training objective
--------------------------
MultipleNegativesRankingLoss over positive query-document pairs. Other
positives in a batch act as in-batch negatives.

Dependencies
------------
numpy
datasets
sentence-transformers
transformers
accelerate>=1.1.0

Install when needed:
    python -m pip install numpy datasets sentence-transformers transformers accelerate>=1.1.0
"""

from __future__ import annotations

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

SPLIT_ASSIGNMENTS = Path("data/splits/holdout_70train_10dev_20test/query_assignments.tsv")

QRELS_SOURCE = "human"       # "human" or "citations"
QUERY_VIEW = "summary"       # "summary", "facts", or "full"
CORPUS_VIEW = "summary"         # "summary", "facts", "reasoning", or "full"
USE_GRADED_HUMAN_QRELS = True

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LABEL = "all-MiniLM-L6-v2-holdout"

DEVICE: str | None = None
TRUST_REMOTE_CODE = False
MAX_SEQUENCE_LENGTH: int | None = None

QUERY_PREFIX = ""
DOCUMENT_PREFIX = ""

# Training positives:
# - graded human qrels: relevance >= 2 by default
# - binary human or citation qrels: relevance >= 1 by default
TRAIN_RELEVANCE_THRESHOLD: int | None = None

NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

USE_FP16 = False
USE_BF16 = False
GRADIENT_CHECKPOINTING = False

# Dev evaluation and checkpointing happen after every epoch.
SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 0

# InformationRetrievalEvaluator settings used for checkpoint selection.
DEV_SELECTION_CUTOFF = 10
CORPUS_CHUNK_SIZE = 5000
SHOW_PROGRESS_BAR = True

# Final metrics and TREC run depth.
RETRIEVE_TOP_K = 100
METRIC_CUTOFFS: tuple[int, ...] = (5, 10, 20, 30)

# Keep False while choosing models or hyperparameters. Change to True only
# after the full configuration has been frozen.
FINAL_TEST_RUN = False

EXCLUDE_QUERY_DOCUMENT = True

OUTPUT_ROOT = Path("results/holdout_biencoder")
OVERWRITE_OUTPUT = True


# =============================================================================
# Validation and paths
# =============================================================================

VALID_QRELS_SOURCES = {"human", "citations"}
VALID_QUERY_VIEWS = {"summary", "facts", "full"}
VALID_CORPUS_VIEWS = {"summary", "facts", "reasoning", "full"}
EXPECTED_SPLIT_COUNTS = {
    "train": 70,
    "dev": 10,
    "test": 20,
}


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

    if NUM_TRAIN_EPOCHS <= 0:
        raise ValueError("NUM_TRAIN_EPOCHS must be positive.")

    if TRAIN_BATCH_SIZE <= 1:
        raise ValueError(
            "TRAIN_BATCH_SIZE must be greater than 1 for in-batch negatives."
        )

    if EVAL_BATCH_SIZE <= 0:
        raise ValueError("EVAL_BATCH_SIZE must be positive.")

    if GRADIENT_ACCUMULATION_STEPS <= 0:
        raise ValueError(
            "GRADIENT_ACCUMULATION_STEPS must be positive."
        )

    if LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive.")

    if not 0.0 <= WARMUP_RATIO < 1.0:
        raise ValueError("WARMUP_RATIO must be in [0, 1).")

    if MAX_SEQUENCE_LENGTH is not None and MAX_SEQUENCE_LENGTH <= 0:
        raise ValueError("MAX_SEQUENCE_LENGTH must be None or positive.")

    if RETRIEVE_TOP_K <= 0:
        raise ValueError("RETRIEVE_TOP_K must be positive.")

    if not METRIC_CUTOFFS or any(k <= 0 for k in METRIC_CUTOFFS):
        raise ValueError("METRIC_CUTOFFS must contain positive integers.")

    if DEV_SELECTION_CUTOFF <= 0:
        raise ValueError("DEV_SELECTION_CUTOFF must be positive.")


def find_project_root() -> Path:
    if PROJECT_ROOT_OVERRIDE is not None:
        root = PROJECT_ROOT_OVERRIDE.expanduser().resolve()

        if not root.is_dir():
            raise FileNotFoundError(
                f"PROJECT_ROOT_OVERRIDE is not a directory: {root}"
            )

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
        "Could not locate the repository root. Set PROJECT_ROOT_OVERRIDE."
    )


def find_eval_views_root(project_root: Path) -> Path:
    if EVAL_VIEWS_ROOT_OVERRIDE is not None:
        path = EVAL_VIEWS_ROOT_OVERRIDE.expanduser()

        if not path.is_absolute():
            path = project_root / path

        path = path.resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"EVAL_VIEWS_ROOT_OVERRIDE is not a directory: {path}"
            )

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
    expanded = path.expanduser()

    if not expanded.is_absolute():
        expanded = project_root / expanded

    return expanded.resolve()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "holdout-model"



def create_training_arguments(
    arguments_class: Any,
    **requested_arguments: Any,
) -> Any:
    """
    Construct SentenceTransformerTrainingArguments across compatible releases.

    Some Sentence Transformers / Transformers combinations use
    ``evaluation_strategy`` while newer releases use ``eval_strategy``.
    The runner maps that one known rename and otherwise fails with a clear
    package-compatibility message rather than silently ignoring settings.

    ``overwrite_output_dir`` is intentionally not passed: this runner already
    deletes and recreates its output directory before training starts.
    """
    import inspect

    signature = inspect.signature(arguments_class.__init__)
    parameters = signature.parameters
    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    arguments = dict(requested_arguments)

    if (
        "eval_strategy" in arguments
        and "eval_strategy" not in parameters
        and "evaluation_strategy" in parameters
    ):
        arguments["evaluation_strategy"] = arguments.pop(
            "eval_strategy"
        )

    if not accepts_arbitrary_keywords:
        unsupported = sorted(
            key
            for key in arguments
            if key not in parameters
        )

        if unsupported:
            raise TypeError(
                "The installed Sentence Transformers / Transformers "
                "versions do not support these training arguments: "
                f"{unsupported}. Upgrade the packages together with:\n"
                "python -m pip install --upgrade "
                "sentence-transformers transformers accelerate datasets"
            )

    return arguments_class(**arguments)

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


def load_corpus(path: Path) -> dict[str, str]:
    corpus: dict[str, str] = {}

    for record in iter_jsonl(path):
        document_id = str(record.get("doc_id", "")).strip()
        text = str(record.get("text", "")).strip()

        if not document_id:
            raise ValueError(f"A corpus row in {path} has no doc_id.")

        if document_id in corpus:
            raise ValueError(f"Duplicate doc_id in {path}: {document_id}")

        if not text:
            continue

        corpus[document_id] = text

    if not corpus:
        raise ValueError(f"No non-empty corpus documents found in {path}.")

    return corpus


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    qrels: dict[str, dict[str, int]] = defaultdict(dict)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
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
            relevance_text = str(row[3]).strip()

            try:
                relevance = int(relevance_text)
            except ValueError as exc:
                if (
                    line_number == 1
                    and relevance_text.lower() == "relevance"
                ):
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


def load_split_assignments(
    path: Path,
) -> dict[str, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Holdout assignment file not found. Run script 13 first or "
            f"update SPLIT_ASSIGNMENTS:\n{path}"
        )

    assignments = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}.")

        normalized = {
            str(field).strip().lower(): field
            for field in reader.fieldnames
        }

        query_column = normalized.get("query_id")
        split_column = (
            normalized.get("split")
            or normalized.get("partition")
        )

        if query_column is None or split_column is None:
            raise ValueError(
                f"{path} must contain query_id and split columns."
            )

        seen: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            query_id = str(row.get(query_column, "")).strip()
            split = str(row.get(split_column, "")).strip().lower()

            if not query_id:
                raise ValueError(
                    f"Blank query_id in {path} at line {row_number}."
                )

            if split not in assignments:
                raise ValueError(
                    f"Invalid split {split!r} in {path} at line "
                    f"{row_number}."
                )

            if query_id in seen:
                raise ValueError(
                    f"Duplicate query assignment in {path}: {query_id}"
                )

            seen.add(query_id)
            assignments[split].add(query_id)

    actual_counts = {
        split: len(query_ids)
        for split, query_ids in assignments.items()
    }

    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected holdout counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Actual:   {actual_counts}"
        )

    return assignments


# =============================================================================
# Training data and evaluator inputs
# =============================================================================

def resolve_relevance_threshold() -> int:
    if TRAIN_RELEVANCE_THRESHOLD is not None:
        return TRAIN_RELEVANCE_THRESHOLD

    if QRELS_SOURCE == "human" and USE_GRADED_HUMAN_QRELS:
        return 2

    return 1


def subset_queries(
    all_queries: Mapping[str, str],
    query_ids: set[str],
) -> dict[str, str]:
    missing = sorted(query_ids - set(all_queries))

    if missing:
        raise ValueError(
            f"{len(missing)} split query or queries are absent from "
            f"queries.jsonl. Examples: {missing[:20]}"
        )

    return {
        query_id: all_queries[query_id]
        for query_id in all_queries
        if query_id in query_ids
    }


def subset_qrels(
    all_qrels: Mapping[str, Mapping[str, int]],
    query_ids: set[str],
) -> dict[str, dict[str, int]]:
    missing = sorted(query_ids - set(all_qrels))

    if missing:
        raise ValueError(
            f"{len(missing)} split query or queries have no qrels. "
            f"Examples: {missing[:20]}"
        )

    return {
        query_id: dict(all_qrels[query_id])
        for query_id in all_qrels
        if query_id in query_ids
    }


def build_relevant_documents(
    qrels: Mapping[str, Mapping[str, int]],
    threshold: int,
) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = {}

    for query_id, query_qrels in qrels.items():
        documents = {
            document_id
            for document_id, relevance in query_qrels.items()
            if relevance >= threshold
        }

        if not documents:
            raise ValueError(
                f"Query {query_id} has no relevance >= {threshold}."
            )

        relevant[query_id] = documents

    return relevant


def build_training_pairs(
    train_queries: Mapping[str, str],
    train_qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, str],
    threshold: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing_documents: set[str] = set()

    for query_id, query_text in train_queries.items():
        positives = [
            document_id
            for document_id, relevance
            in train_qrels[query_id].items()
            if relevance >= threshold
        ]

        if not positives:
            raise ValueError(
                f"Training query {query_id} has no positive document."
            )

        for document_id in positives:
            document_text = corpus.get(document_id)

            if document_text is None:
                missing_documents.add(document_id)
                continue

            rows.append(
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "query": QUERY_PREFIX + query_text,
                    "positive": DOCUMENT_PREFIX + document_text,
                }
            )

    if missing_documents:
        raise ValueError(
            f"{len(missing_documents)} positive training documents are "
            "absent from the corpus. Examples: "
            f"{sorted(missing_documents)[:20]}"
        )

    random.Random(RANDOM_SEED).shuffle(rows)

    if not rows:
        raise ValueError("No training pairs were created.")

    return rows


def validate_qrel_documents(
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, str],
) -> None:
    missing = {
        document_id
        for query_qrels in qrels.values()
        for document_id in query_qrels
        if document_id not in corpus
    }

    if missing:
        raise ValueError(
            f"{len(missing)} qrel documents are absent from the corpus. "
            f"Examples: {sorted(missing)[:20]}"
        )


# =============================================================================
# Retrieval and graded evaluation
# =============================================================================

def encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    role: str,
) -> np.ndarray:
    prefix = QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX

    vectors = model.encode(
        [prefix + text for text in texts],
        batch_size=EVAL_BATCH_SIZE,
        show_progress_bar=SHOW_PROGRESS_BAR,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    array = np.asarray(vectors, dtype=np.float32)

    if array.ndim != 2 or array.shape[0] != len(texts):
        raise ValueError("Model returned an invalid embedding matrix.")

    return array


def retrieve_partition(
    model: Any,
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
) -> dict[str, list[tuple[str, float]]]:
    query_ids = list(queries)
    document_ids = list(corpus)

    query_vectors = encode_texts(
        model,
        [queries[query_id] for query_id in query_ids],
        role="query",
    )
    document_vectors = encode_texts(
        model,
        [corpus[document_id] for document_id in document_ids],
        role="document",
    )

    document_index = {
        document_id: index
        for index, document_id in enumerate(document_ids)
    }

    top_k = min(RETRIEVE_TOP_K, len(document_ids))
    rankings: dict[str, list[tuple[str, float]]] = {}

    for query_id, query_vector in zip(query_ids, query_vectors):
        scores = document_vectors @ query_vector

        if EXCLUDE_QUERY_DOCUMENT and query_id in document_index:
            scores[document_index[query_id]] = -np.inf

        if top_k == len(scores):
            selected = np.argsort(-scores)
        else:
            selected = np.argpartition(
                -scores,
                top_k - 1,
            )[:top_k]
            selected = selected[np.argsort(-scores[selected])]

        rankings[query_id] = [
            (document_ids[int(index)], float(scores[int(index)]))
            for index in selected
            if np.isfinite(scores[int(index)])
        ]

    return rankings


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


def recall_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    threshold: int,
    k: int,
) -> float:
    relevant = {
        document_id
        for document_id, relevance in qrels.items()
        if relevance >= threshold
    }

    if not relevant:
        return 0.0

    return sum(
        document_id in relevant
        for document_id in ranked[:k]
    ) / len(relevant)


def average_precision_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    threshold: int,
    k: int,
) -> float:
    total_relevant = sum(
        relevance >= threshold
        for relevance in qrels.values()
    )

    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    score = 0.0

    for rank, document_id in enumerate(ranked[:k], start=1):
        if qrels.get(document_id, 0) >= threshold:
            relevant_seen += 1
            score += relevant_seen / rank

    return score / min(total_relevant, k)


def discounted_cumulative_gain(
    relevances: Sequence[int],
) -> float:
    return sum(
        ((2.0 ** relevance) - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def ndcg_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    actual = [
        qrels.get(document_id, 0)
        for document_id in ranked[:k]
    ]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_score = discounted_cumulative_gain(ideal)

    if ideal_score == 0.0:
        return 0.0

    return discounted_cumulative_gain(actual) / ideal_score


def judged_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    return sum(
        document_id in qrels
        for document_id in ranked[:k]
    ) / k


def evaluate_rankings(
    queries: Mapping[str, str],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    threshold: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []

    for query_id in queries:
        ranked = [
            document_id
            for document_id, _score in rankings[query_id]
        ]
        query_qrels = qrels[query_id]

        row: dict[str, object] = {
            "query_id": query_id,
            "judged_documents": len(query_qrels),
            "relevant_documents": sum(
                relevance >= threshold
                for relevance in query_qrels.values()
            ),
        }

        for k in METRIC_CUTOFFS:
            row[f"P@{k}"] = precision_at_k(
                ranked, query_qrels, threshold, k
            )
            row[f"Recall@{k}"] = recall_at_k(
                ranked, query_qrels, threshold, k
            )
            row[f"MAP@{k}"] = average_precision_at_k(
                ranked, query_qrels, threshold, k
            )
            row[f"nDCG@{k}"] = ndcg_at_k(
                ranked, query_qrels, k
            )
            row[f"Judged@{k}"] = judged_at_k(
                ranked, query_qrels, k
            )

        rows.append(row)

    metrics = {
        field: float(np.mean([float(row[field]) for row in rows]))
        for field in rows[0]
        if field not in {
            "query_id",
            "judged_documents",
            "relevant_documents",
        }
    }

    return rows, metrics


# =============================================================================
# Writers
# =============================================================================

def prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(path)

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_table(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}.")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_trec_run(
    path: Path,
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    run_name: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )

        for query_id, documents in rankings.items():
            for rank, (document_id, score) in enumerate(
                documents,
                start=1,
            ):
                writer.writerow(
                    [
                        query_id,
                        "Q0",
                        document_id,
                        rank,
                        f"{score:.8f}",
                        run_name,
                    ]
                )


def write_training_pairs(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> None:
    audit_rows = [
        {
            "query_id": row["query_id"],
            "document_id": row["document_id"],
        }
        for row in rows
    ]
    write_table(path, audit_rows)


# =============================================================================
# Reproducibility
# =============================================================================

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

    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.evaluation import (
            InformationRetrievalEvaluator,
        )
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as exc:
        raise ImportError(
            "Install the training dependencies with:\n"
            "python -m pip install numpy datasets sentence-transformers "
            "transformers accelerate"
        ) from exc

    project_root = find_project_root()
    eval_views_root = find_eval_views_root(project_root)
    split_path = resolve_project_path(
        project_root,
        SPLIT_ASSIGNMENTS,
    )

    view_name = f"q_{QUERY_VIEW}__c_{CORPUS_VIEW}"
    view_directory = (
        eval_views_root
        / QRELS_SOURCE
        / view_name
    )

    corpus_path = view_directory / "corpus.jsonl"
    queries_path = view_directory / "queries.jsonl"

    if QRELS_SOURCE == "human" and USE_GRADED_HUMAN_QRELS:
        qrels_path = (
            view_directory
            / "qrels_selected_graded.tsv"
        )
        qrels_label = "human_graded"
    else:
        qrels_path = (
            view_directory
            / "qrels_selected.tsv"
        )
        qrels_label = (
            "human_binary"
            if QRELS_SOURCE == "human"
            else "citations"
        )

    for path in (
        split_path,
        view_directory,
        corpus_path,
        queries_path,
        qrels_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    assignments = load_split_assignments(split_path)
    all_queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)
    all_qrels = load_qrels(qrels_path)
    threshold = resolve_relevance_threshold()

    assigned_queries = set().union(*assignments.values())
    query_ids = set(all_queries)

    if assigned_queries != query_ids:
        raise ValueError(
            "Split assignments and queries.jsonl do not contain exactly "
            "the same query IDs.\n"
            f"Only in assignments: "
            f"{sorted(assigned_queries - query_ids)[:20]}\n"
            f"Only in queries: "
            f"{sorted(query_ids - assigned_queries)[:20]}"
        )

    train_queries = subset_queries(
        all_queries,
        assignments["train"],
    )
    dev_queries = subset_queries(
        all_queries,
        assignments["dev"],
    )
    test_queries = subset_queries(
        all_queries,
        assignments["test"],
    )

    train_qrels = subset_qrels(
        all_qrels,
        assignments["train"],
    )
    dev_qrels = subset_qrels(
        all_qrels,
        assignments["dev"],
    )
    test_qrels = subset_qrels(
        all_qrels,
        assignments["test"],
    )

    validate_qrel_documents(train_qrels, corpus)
    validate_qrel_documents(dev_qrels, corpus)
    validate_qrel_documents(test_qrels, corpus)

    train_rows = build_training_pairs(
        train_queries,
        train_qrels,
        corpus,
        threshold,
    )

    run_name = safe_name(
        f"{MODEL_LABEL}"
        f"__{qrels_label}"
        f"__{QUERY_VIEW}_{CORPUS_VIEW}"
    )
    output_directory = (
        project_root
        / OUTPUT_ROOT
        / run_name
    )
    prepare_output_directory(output_directory)

    checkpoints_directory = output_directory / "checkpoints"
    best_model_directory = output_directory / "best_model"

    write_training_pairs(
        output_directory / "training_pairs.tsv",
        train_rows,
    )

    print(f"Project root: {project_root}")
    print(f"Evaluation view: {view_directory}")
    print(f"Split assignments: {split_path}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Training queries: {len(train_queries)}")
    print(f"Development queries: {len(dev_queries)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Training pairs: {len(train_rows):,}")
    print(f"Positive threshold: {threshold}")
    print(f"Final test enabled: {FINAL_TEST_RUN}")
    print()

    model = SentenceTransformer(
        BASE_MODEL_NAME,
        device=DEVICE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if MAX_SEQUENCE_LENGTH is not None:
        model.max_seq_length = MAX_SEQUENCE_LENGTH

    train_dataset = Dataset.from_dict(
        {
            "query": [row["query"] for row in train_rows],
            "positive": [row["positive"] for row in train_rows],
        }
    )

    dev_relevant_documents = build_relevant_documents(
        dev_qrels,
        threshold,
    )

    dev_evaluator = InformationRetrievalEvaluator(
        queries={
            query_id: QUERY_PREFIX + text
            for query_id, text in dev_queries.items()
        },
        corpus={
            document_id: DOCUMENT_PREFIX + text
            for document_id, text in corpus.items()
        },
        relevant_docs=dev_relevant_documents,
        corpus_chunk_size=CORPUS_CHUNK_SIZE,
        mrr_at_k=[DEV_SELECTION_CUTOFF],
        ndcg_at_k=[DEV_SELECTION_CUTOFF],
        accuracy_at_k=[1, 5, DEV_SELECTION_CUTOFF],
        precision_recall_at_k=[5, DEV_SELECTION_CUTOFF],
        map_at_k=[DEV_SELECTION_CUTOFF],
        show_progress_bar=SHOW_PROGRESS_BAR,
        batch_size=EVAL_BATCH_SIZE,
        name="eurlexrd_dev",
        write_csv=True,
    )

    # Some Sentence Transformers releases populate primary_metric only after
    # the evaluator has run once. Passing its initial None value makes
    # Transformers fall back to "eval_loss", which this IR-only evaluation
    # does not produce. Use the exact metric key returned by Trainer.
    dev_selection_metric = (
        f"eval_{dev_evaluator.name}_cosine_ndcg@"
        f"{DEV_SELECTION_CUTOFF}"
    )

    loss = losses.MultipleNegativesRankingLoss(model)

    training_arguments = create_training_arguments(
        SentenceTransformerTrainingArguments,
        output_dir=str(checkpoints_directory),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=USE_FP16,
        bf16=USE_BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=max(1, len(train_dataset) // TRAIN_BATCH_SIZE),
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=dev_selection_metric,
        greater_is_better=True,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        report_to="none",
        run_name=run_name,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    trainer.train()
    model.save_pretrained(str(best_model_directory))

    # Re-evaluate the selected model with the full graded qrels-aware metrics.
    dev_rankings = retrieve_partition(
        model,
        dev_queries,
        corpus,
    )
    dev_rows, dev_metrics = evaluate_rankings(
        dev_queries,
        dev_rankings,
        dev_qrels,
        threshold,
    )

    write_trec_run(
        output_directory / "dev_run.trec",
        dev_rankings,
        run_name + "__dev",
    )
    write_table(
        output_directory / "dev_per_query_metrics.tsv",
        dev_rows,
    )
    write_json(
        output_directory / "dev_metrics.json",
        {
            "partition": "dev",
            "query_count": len(dev_queries),
            "relevance_threshold": threshold,
            "metrics": dev_metrics,
        },
    )

    test_metrics: dict[str, float] | None = None

    if FINAL_TEST_RUN:
        test_rankings = retrieve_partition(
            model,
            test_queries,
            corpus,
        )
        test_rows, test_metrics = evaluate_rankings(
            test_queries,
            test_rankings,
            test_qrels,
            threshold,
        )

        write_trec_run(
            output_directory / "test_run.trec",
            test_rankings,
            run_name + "__test",
        )
        write_table(
            output_directory / "test_per_query_metrics.tsv",
            test_rows,
        )
        write_json(
            output_directory / "test_metrics.json",
            {
                "partition": "test",
                "query_count": len(test_queries),
                "relevance_threshold": threshold,
                "metrics": test_metrics,
            },
        )

    write_json(
        output_directory / "run_config.json",
        {
            "protocol": "supervised_holdout_70_10_20",
            "random_seed": RANDOM_SEED,
            "base_model_name": BASE_MODEL_NAME,
            "model_label": MODEL_LABEL,
            "qrels_source": qrels_label,
            "query_view": QUERY_VIEW,
            "corpus_view": CORPUS_VIEW,
            "split_assignments": str(
                split_path.relative_to(project_root)
            ),
            "train_query_count": len(train_queries),
            "dev_query_count": len(dev_queries),
            "test_query_count": len(test_queries),
            "training_pair_count": len(train_rows),
            "train_relevance_threshold": threshold,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "query_prefix": QUERY_PREFIX,
            "document_prefix": DOCUMENT_PREFIX,
            "dev_selection_metric": dev_selection_metric,
            "dev_selection_cutoff": DEV_SELECTION_CUTOFF,
            "retrieve_top_k": RETRIEVE_TOP_K,
            "metric_cutoffs": list(METRIC_CUTOFFS),
            "final_test_run": FINAL_TEST_RUN,
            "best_model_directory": str(
                best_model_directory.relative_to(project_root)
            ),
        },
    )

    print()
    print("Development metrics")
    print("-------------------")

    for metric_name in sorted(
        dev_metrics,
        key=lambda name: (
            int(name.split("@")[1]),
            name.split("@")[0],
        ),
    ):
        print(f"{metric_name:>12}: {dev_metrics[metric_name]:.6f}")

    if test_metrics is not None:
        print()
        print("Final test metrics")
        print("------------------")

        for metric_name in sorted(
            test_metrics,
            key=lambda name: (
                int(name.split("@")[1]),
                name.split("@")[0],
            ),
        ):
            print(
                f"{metric_name:>12}: "
                f"{test_metrics[metric_name]:.6f}"
            )
    else:
        print()
        print(
            "The test partition was not evaluated. After all model and "
            "hyperparameter choices are frozen, set FINAL_TEST_RUN = True "
            "and run the final configuration once."
        )

    print()
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    main()
