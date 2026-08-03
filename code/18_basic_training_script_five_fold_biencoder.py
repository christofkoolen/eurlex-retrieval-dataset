#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised five-fold cross-validation runner for EUR-LexRD.

This script fine-tunes five independent Sentence Transformers bi-encoders
using the released query-level fold assignments.

For each outer fold:

    70 queries -> model training
    10 queries -> inner development and checkpoint selection
    20 queries -> outer test evaluation

The 10 inner-development queries are selected deterministically from the
80 non-test queries without consulting their relevance labels. Consequently,
the outer test fold remains isolated from training and model selection.

The script is designed to be opened and run directly in Spyder. It has no
command-line interface.

Important evaluation rule
-------------------------
Keep FINAL_CV_RUN = False while changing models, learning rates, epochs,
prefixes, chunking, or other settings. Set FINAL_CV_RUN = True only after the
complete configuration has been frozen. In final mode, each of the 100 queries
is evaluated exactly once as outer test data.

Canonical inputs
----------------
data/splits/five_fold_cv_v1/fold_assignments.tsv

data/eval_views/<qrels source>/<view>/
    corpus.jsonl
    queries.jsonl
    qrels_selected.tsv
    qrels_selected_graded.tsv  # human views only

Outputs
-------
results/five_fold_biencoder/<run name>/
    inner_split_assignments.tsv
    run_config.json
    fold_summary.tsv
    cross_validation_metrics.json
    all_test_run.trec
    all_test_per_query_metrics.tsv

    fold_1/
        best_model/
        checkpoints/
        training_pairs.tsv
        dev_run.trec
        dev_metrics.json
        dev_per_query_metrics.tsv
        test_run.trec                  # final mode only
        test_metrics.json              # final mode only
        test_per_query_metrics.tsv     # final mode only
        fold_config.json

    ...
    fold_5/

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
import gc
import hashlib
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
INNER_DEV_SEED = 123

PROJECT_ROOT_OVERRIDE: Path | None = None
EVAL_VIEWS_ROOT_OVERRIDE: Path | None = None

FOLD_ASSIGNMENTS = Path("data/splits/five_fold_cross_validation/fold_assignments.tsv")

QRELS_SOURCE = "human"       # "human" or "citations"
QUERY_VIEW = "summary"       # "summary", "facts", or "full"
CORPUS_VIEW = "summary"         # "summary", "facts", "reasoning", or "full"
USE_GRADED_HUMAN_QRELS = True

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LABEL = "all-MiniLM-L6-v2-five-fold"

DEVICE: str | None = None
TRUST_REMOTE_CODE = False
MAX_SEQUENCE_LENGTH: int | None = None

QUERY_PREFIX = ""
DOCUMENT_PREFIX = ""

# Run a subset such as (1,) for a smoke test. Use all five for the final run.
FOLDS_TO_RUN: tuple[int, ...] = (1, 2, 3, 4, 5)

# Ten of the 80 non-test queries become an inner development set.
INNER_DEV_QUERY_COUNT = 10

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

SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 0

# The inner-development evaluator selects the best epoch.
DEV_SELECTION_CUTOFF = 10
CORPUS_CHUNK_SIZE = 5000
SHOW_PROGRESS_BAR = True

# Final retrieval and reporting.
RETRIEVE_TOP_K = 100
METRIC_CUTOFFS: tuple[int, ...] = (5, 10, 20, 30)

# Keep False while configuring the experiment. Set True only when all
# hyperparameters and modelling decisions have been frozen.
FINAL_CV_RUN = False

EXCLUDE_QUERY_DOCUMENT = True

OUTPUT_ROOT = Path("results/five_fold_biencoder")
OVERWRITE_OUTPUT = True


# =============================================================================
# Constants and validation
# =============================================================================

VALID_QRELS_SOURCES = {"human", "citations"}
VALID_QUERY_VIEWS = {"summary", "facts", "full"}
VALID_CORPUS_VIEWS = {"summary", "facts", "reasoning", "full"}

EXPECTED_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_QUERIES_PER_OUTER_FOLD = 20
EXPECTED_TOTAL_QUERY_COUNT = 100


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

    requested_folds = set(FOLDS_TO_RUN)

    if not requested_folds:
        raise ValueError("FOLDS_TO_RUN cannot be empty.")

    if not requested_folds <= EXPECTED_FOLDS:
        raise ValueError(
            f"FOLDS_TO_RUN must contain only {sorted(EXPECTED_FOLDS)}."
        )

    if len(FOLDS_TO_RUN) != len(requested_folds):
        raise ValueError("FOLDS_TO_RUN contains duplicate folds.")

    if INNER_DEV_QUERY_COUNT <= 0:
        raise ValueError("INNER_DEV_QUERY_COUNT must be positive.")

    if INNER_DEV_QUERY_COUNT >= 80:
        raise ValueError(
            "INNER_DEV_QUERY_COUNT must be smaller than the 80 outer "
            "training queries."
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

    if FINAL_CV_RUN and requested_folds != EXPECTED_FOLDS:
        raise ValueError(
            "FINAL_CV_RUN requires FOLDS_TO_RUN = (1, 2, 3, 4, 5) so "
            "that every query is evaluated exactly once."
        )


# =============================================================================
# Paths
# =============================================================================

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

    working_directory = Path.cwd().resolve()
    candidates.extend([working_directory, *working_directory.parents])

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
    return cleaned or "five-fold-model"



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
# Input readers
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


def load_fold_assignments(
    path: Path,
) -> dict[int, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Fold-assignment file not found. Run script 14 first or "
            f"update FOLD_ASSIGNMENTS:\n{path}"
        )

    assignments: dict[int, set[str]] = {
        fold: set()
        for fold in sorted(EXPECTED_FOLDS)
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
        fold_column = normalized.get("fold")

        if query_column is None or fold_column is None:
            raise ValueError(
                f"{path} must contain query_id and fold columns."
            )

        seen: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            query_id = str(row.get(query_column, "")).strip()
            fold_text = str(row.get(fold_column, "")).strip()

            if not query_id:
                raise ValueError(
                    f"Blank query_id in {path} at line {row_number}."
                )

            try:
                fold = int(fold_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid fold {fold_text!r} in {path} at line "
                    f"{row_number}."
                ) from exc

            if fold not in assignments:
                raise ValueError(
                    f"Fold {fold} is outside {sorted(EXPECTED_FOLDS)}."
                )

            if query_id in seen:
                raise ValueError(
                    f"Duplicate query assignment in {path}: {query_id}"
                )

            seen.add(query_id)
            assignments[fold].add(query_id)

    counts = {
        fold: len(query_ids)
        for fold, query_ids in assignments.items()
    }

    expected_counts = {
        fold: EXPECTED_QUERIES_PER_OUTER_FOLD
        for fold in EXPECTED_FOLDS
    }

    if counts != expected_counts:
        raise ValueError(
            "Unexpected fold counts.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {counts}"
        )

    if len(seen) != EXPECTED_TOTAL_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_QUERY_COUNT} assigned queries; "
            f"found {len(seen)}."
        )

    return assignments


# =============================================================================
# Fold construction
# =============================================================================

def deterministic_inner_order(
    query_id: str,
    outer_fold: int,
) -> str:
    material = (
        f"eurlexrd|inner-dev|{INNER_DEV_SEED}|"
        f"outer-fold-{outer_fold}|{query_id}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def make_nested_fold_split(
    fold_assignments: Mapping[int, set[str]],
    outer_fold: int,
) -> dict[str, set[str]]:
    outer_test = set(fold_assignments[outer_fold])
    outer_training = set().union(
        *[
            query_ids
            for fold, query_ids in fold_assignments.items()
            if fold != outer_fold
        ]
    )

    ordered_outer_training = sorted(
        outer_training,
        key=lambda query_id: (
            deterministic_inner_order(query_id, outer_fold),
            query_id,
        ),
    )

    inner_dev = set(
        ordered_outer_training[:INNER_DEV_QUERY_COUNT]
    )
    inner_train = outer_training - inner_dev

    split = {
        "train": inner_train,
        "dev": inner_dev,
        "test": outer_test,
    }

    if set().union(*split.values()) != set().union(
        *fold_assignments.values()
    ):
        raise ValueError(
            f"Nested split for fold {outer_fold} does not cover all queries."
        )

    if any(
        split[left] & split[right]
        for left, right in (
            ("train", "dev"),
            ("train", "test"),
            ("dev", "test"),
        )
    ):
        raise ValueError(
            f"Nested split for fold {outer_fold} contains overlap."
        )

    expected_train_count = (
        EXPECTED_TOTAL_QUERY_COUNT
        - EXPECTED_QUERIES_PER_OUTER_FOLD
        - INNER_DEV_QUERY_COUNT
    )

    actual_counts = {
        partition: len(query_ids)
        for partition, query_ids in split.items()
    }
    expected_counts = {
        "train": expected_train_count,
        "dev": INNER_DEV_QUERY_COUNT,
        "test": EXPECTED_QUERIES_PER_OUTER_FOLD,
    }

    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected nested split counts for fold {outer_fold}.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {actual_counts}"
        )

    return split


def build_all_nested_splits(
    fold_assignments: Mapping[int, set[str]],
) -> dict[int, dict[str, set[str]]]:
    return {
        fold: make_nested_fold_split(fold_assignments, fold)
        for fold in sorted(EXPECTED_FOLDS)
    }


# =============================================================================
# Data preparation
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
    fold_seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing_documents: set[str] = set()

    for query_id, query_text in train_queries.items():
        positive_documents = [
            document_id
            for document_id, relevance
            in train_qrels[query_id].items()
            if relevance >= threshold
        ]

        if not positive_documents:
            raise ValueError(
                f"Training query {query_id} has no positive document."
            )

        for document_id in positive_documents:
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

    random.Random(fold_seed).shuffle(rows)

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
# Retrieval
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


def encode_corpus_in_batches(
    model: Any,
    corpus: Mapping[str, str],
) -> tuple[list[str], np.ndarray]:
    document_ids = list(corpus)
    batches: list[np.ndarray] = []

    for start in range(0, len(document_ids), EVAL_BATCH_SIZE):
        batch_ids = document_ids[start:start + EVAL_BATCH_SIZE]
        batches.append(
            encode_texts(
                model,
                [corpus[document_id] for document_id in batch_ids],
                role="document",
            )
        )

    return document_ids, np.vstack(batches)


def retrieve_partition(
    model: Any,
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
) -> dict[str, list[tuple[str, float]]]:
    query_ids = list(queries)

    query_vectors = encode_texts(
        model,
        [queries[query_id] for query_id in query_ids],
        role="query",
    )
    document_ids, document_vectors = encode_corpus_in_batches(
        model,
        corpus,
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

    metrics = aggregate_per_query_rows(rows)
    return rows, metrics


def metric_field_names(
    row: Mapping[str, object],
) -> list[str]:
    return [
        field
        for field in row
        if field not in {
            "query_id",
            "outer_fold",
            "judged_documents",
            "relevant_documents",
        }
    ]


def aggregate_per_query_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric table.")

    return {
        field: float(
            np.mean([float(row[field]) for row in rows])
        )
        for field in metric_field_names(rows[0])
    }


def summarize_fold_metrics(
    fold_metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    if not fold_metric_rows:
        raise ValueError("No fold metrics supplied.")

    metric_names = [
        field
        for field in fold_metric_rows[0]
        if field not in {
            "outer_fold",
            "train_queries",
            "dev_queries",
            "test_queries",
            "training_pairs",
        }
    ]

    summary: dict[str, dict[str, float]] = {}

    for metric_name in metric_names:
        values = np.asarray(
            [
                float(row[metric_name])
                for row in fold_metric_rows
            ],
            dtype=np.float64,
        )

        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            ),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return summary


# =============================================================================
# Output writers
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
    write_table(
        path,
        [
            {
                "query_id": row["query_id"],
                "document_id": row["document_id"],
            }
            for row in rows
        ],
    )


def write_inner_split_assignments(
    path: Path,
    nested_splits: Mapping[
        int,
        Mapping[str, set[str]],
    ],
) -> None:
    rows: list[dict[str, object]] = []

    for outer_fold in sorted(nested_splits):
        for partition in ("train", "dev", "test"):
            for query_id in sorted(
                nested_splits[outer_fold][partition]
            ):
                rows.append(
                    {
                        "outer_fold": outer_fold,
                        "query_id": query_id,
                        "partition": partition,
                    }
                )

    write_table(path, rows)


# =============================================================================
# Reproducibility and cleanup
# =============================================================================

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_model_memory(*objects: object) -> None:
    for object_value in objects:
        del object_value

    gc.collect()

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# One-fold training
# =============================================================================

def run_one_fold(
    *,
    outer_fold: int,
    nested_split: Mapping[str, set[str]],
    all_queries: Mapping[str, str],
    all_qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, str],
    threshold: int,
    run_directory: Path,
    run_name: str,
    dataset_class: Any,
    sentence_transformer_class: Any,
    trainer_class: Any,
    training_arguments_class: Any,
    losses_module: Any,
    evaluator_class: Any,
    batch_samplers: Any,
) -> dict[str, object]:
    fold_seed = RANDOM_SEED + outer_fold
    set_random_seeds(fold_seed)

    fold_directory = run_directory / f"fold_{outer_fold}"
    fold_directory.mkdir(parents=True, exist_ok=False)

    checkpoints_directory = fold_directory / "checkpoints"
    best_model_directory = fold_directory / "best_model"

    train_queries = subset_queries(
        all_queries,
        nested_split["train"],
    )
    dev_queries = subset_queries(
        all_queries,
        nested_split["dev"],
    )
    test_queries = subset_queries(
        all_queries,
        nested_split["test"],
    )

    train_qrels = subset_qrels(
        all_qrels,
        nested_split["train"],
    )
    dev_qrels = subset_qrels(
        all_qrels,
        nested_split["dev"],
    )
    test_qrels = subset_qrels(
        all_qrels,
        nested_split["test"],
    )

    validate_qrel_documents(train_qrels, corpus)
    validate_qrel_documents(dev_qrels, corpus)
    validate_qrel_documents(test_qrels, corpus)

    training_rows = build_training_pairs(
        train_queries,
        train_qrels,
        corpus,
        threshold,
        fold_seed,
    )

    write_training_pairs(
        fold_directory / "training_pairs.tsv",
        training_rows,
    )

    print()
    print("=" * 72)
    print(f"Outer fold {outer_fold}")
    print("=" * 72)
    print(f"Training queries: {len(train_queries)}")
    print(f"Inner-development queries: {len(dev_queries)}")
    print(f"Outer-test queries: {len(test_queries)}")
    print(f"Training pairs: {len(training_rows):,}")
    print(f"Fold seed: {fold_seed}")
    print()

    model = sentence_transformer_class(
        BASE_MODEL_NAME,
        device=DEVICE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if MAX_SEQUENCE_LENGTH is not None:
        model.max_seq_length = MAX_SEQUENCE_LENGTH

    train_dataset = dataset_class.from_dict(
        {
            "query": [
                row["query"]
                for row in training_rows
            ],
            "positive": [
                row["positive"]
                for row in training_rows
            ],
        }
    )

    dev_relevant_documents = build_relevant_documents(
        dev_qrels,
        threshold,
    )

    dev_evaluator = evaluator_class(
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
        name=f"eurlexrd_fold_{outer_fold}_dev",
        write_csv=True,
    )

    # Some releases populate primary_metric only after the evaluator runs.
    # Use the exact Trainer metric key so checkpoint selection cannot fall
    # back to the unavailable "eval_loss".
    dev_selection_metric = (
        f"eval_{dev_evaluator.name}_cosine_ndcg@"
        f"{DEV_SELECTION_CUTOFF}"
    )

    loss = losses_module.MultipleNegativesRankingLoss(model)

    training_arguments = training_arguments_class(
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
        batch_sampler=batch_samplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=max(
            1,
            len(train_dataset) // TRAIN_BATCH_SIZE,
        ),
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=dev_selection_metric,
        greater_is_better=True,
        seed=fold_seed,
        data_seed=fold_seed,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        report_to="none",
        run_name=f"{run_name}__fold_{outer_fold}",
    )

    trainer = trainer_class(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    trainer.train()
    model.save_pretrained(str(best_model_directory))

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

    for row in dev_rows:
        row["outer_fold"] = outer_fold

    dev_rows = [
        {
            "outer_fold": row.pop("outer_fold"),
            **row,
        }
        for row in dev_rows
    ]

    write_trec_run(
        fold_directory / "dev_run.trec",
        dev_rankings,
        f"{run_name}__fold_{outer_fold}__dev",
    )
    write_table(
        fold_directory / "dev_per_query_metrics.tsv",
        dev_rows,
    )
    write_json(
        fold_directory / "dev_metrics.json",
        {
            "partition": "inner_dev",
            "outer_fold": outer_fold,
            "query_count": len(dev_queries),
            "relevance_threshold": threshold,
            "metrics": dev_metrics,
        },
    )

    fold_result: dict[str, object] = {
        "outer_fold": outer_fold,
        "train_query_count": len(train_queries),
        "dev_query_count": len(dev_queries),
        "test_query_count": len(test_queries),
        "training_pair_count": len(training_rows),
        "dev_metrics": dev_metrics,
        "test_metrics": None,
        "test_rows": [],
        "test_rankings": {},
    }

    if FINAL_CV_RUN:
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

        for row in test_rows:
            row["outer_fold"] = outer_fold

        test_rows = [
            {
                "outer_fold": row.pop("outer_fold"),
                **row,
            }
            for row in test_rows
        ]

        write_trec_run(
            fold_directory / "test_run.trec",
            test_rankings,
            f"{run_name}__fold_{outer_fold}__test",
        )
        write_table(
            fold_directory / "test_per_query_metrics.tsv",
            test_rows,
        )
        write_json(
            fold_directory / "test_metrics.json",
            {
                "partition": "outer_test",
                "outer_fold": outer_fold,
                "query_count": len(test_queries),
                "relevance_threshold": threshold,
                "metrics": test_metrics,
            },
        )

        fold_result["test_metrics"] = test_metrics
        fold_result["test_rows"] = test_rows
        fold_result["test_rankings"] = test_rankings

    write_json(
        fold_directory / "fold_config.json",
        {
            "outer_fold": outer_fold,
            "fold_seed": fold_seed,
            "protocol": "nested_five_fold_cross_validation",
            "base_model_name": BASE_MODEL_NAME,
            "train_query_count": len(train_queries),
            "dev_query_count": len(dev_queries),
            "test_query_count": len(test_queries),
            "training_pair_count": len(training_rows),
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
            "final_cv_run": FINAL_CV_RUN,
        },
    )

    print()
    print(f"Fold {outer_fold} development metrics")
    print("-" * 42)

    for metric_name in sorted(
        dev_metrics,
        key=lambda name: (
            int(name.split("@")[1]),
            name.split("@")[0],
        ),
    ):
        print(f"{metric_name:>12}: {dev_metrics[metric_name]:.6f}")

    if FINAL_CV_RUN:
        test_metrics = fold_result["test_metrics"]
        assert isinstance(test_metrics, Mapping)

        print()
        print(f"Fold {outer_fold} outer-test metrics")
        print("-" * 42)

        for metric_name in sorted(
            test_metrics,
            key=lambda name: (
                int(name.split("@")[1]),
                name.split("@")[0],
            ),
        ):
            print(
                f"{metric_name:>12}: "
                f"{float(test_metrics[metric_name]):.6f}"
            )

    release_model_memory(
        trainer,
        model,
        loss,
        train_dataset,
        dev_evaluator,
    )

    return fold_result


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    validate_configuration()
    set_random_seeds(RANDOM_SEED)

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
    fold_assignment_path = resolve_project_path(
        project_root,
        FOLD_ASSIGNMENTS,
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
        fold_assignment_path,
        view_directory,
        corpus_path,
        queries_path,
        qrels_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    fold_assignments = load_fold_assignments(
        fold_assignment_path
    )
    nested_splits = build_all_nested_splits(
        fold_assignments
    )
    all_queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)
    all_qrels = load_qrels(qrels_path)
    threshold = resolve_relevance_threshold()

    assigned_query_ids = set().union(*fold_assignments.values())
    query_ids = set(all_queries)

    if assigned_query_ids != query_ids:
        raise ValueError(
            "Fold assignments and queries.jsonl do not contain exactly "
            "the same query IDs.\n"
            f"Only in assignments: "
            f"{sorted(assigned_query_ids - query_ids)[:20]}\n"
            f"Only in queries: "
            f"{sorted(query_ids - assigned_query_ids)[:20]}"
        )

    missing_qrel_queries = sorted(query_ids - set(all_qrels))

    if missing_qrel_queries:
        raise ValueError(
            f"{len(missing_qrel_queries)} query or queries have no qrels. "
            f"Examples: {missing_qrel_queries[:20]}"
        )

    validate_qrel_documents(all_qrels, corpus)

    run_name = safe_name(
        f"{MODEL_LABEL}"
        f"__{qrels_label}"
        f"__{QUERY_VIEW}_{CORPUS_VIEW}"
    )
    run_directory = (
        project_root
        / OUTPUT_ROOT
        / run_name
    )
    prepare_output_directory(run_directory)

    write_inner_split_assignments(
        run_directory / "inner_split_assignments.tsv",
        nested_splits,
    )

    write_json(
        run_directory / "run_config.json",
        {
            "protocol": "nested_five_fold_cross_validation",
            "random_seed": RANDOM_SEED,
            "inner_dev_seed": INNER_DEV_SEED,
            "base_model_name": BASE_MODEL_NAME,
            "model_label": MODEL_LABEL,
            "qrels_source": qrels_label,
            "query_view": QUERY_VIEW,
            "corpus_view": CORPUS_VIEW,
            "fold_assignments": str(
                fold_assignment_path.relative_to(project_root)
            ),
            "folds_to_run": list(FOLDS_TO_RUN),
            "outer_test_query_count": (
                EXPECTED_QUERIES_PER_OUTER_FOLD
            ),
            "inner_dev_query_count": INNER_DEV_QUERY_COUNT,
            "inner_train_query_count": (
                EXPECTED_TOTAL_QUERY_COUNT
                - EXPECTED_QUERIES_PER_OUTER_FOLD
                - INNER_DEV_QUERY_COUNT
            ),
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
            "dev_selection_cutoff": DEV_SELECTION_CUTOFF,
            "retrieve_top_k": RETRIEVE_TOP_K,
            "metric_cutoffs": list(METRIC_CUTOFFS),
            "final_cv_run": FINAL_CV_RUN,
            "exclude_query_document": EXCLUDE_QUERY_DOCUMENT,
        },
    )

    print(f"Project root: {project_root}")
    print(f"Evaluation view: {view_directory}")
    print(f"Fold assignments: {fold_assignment_path}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Folds to run: {FOLDS_TO_RUN}")
    print(
        "Per fold: "
        f"{EXPECTED_TOTAL_QUERY_COUNT - EXPECTED_QUERIES_PER_OUTER_FOLD - INNER_DEV_QUERY_COUNT} train / "
        f"{INNER_DEV_QUERY_COUNT} dev / "
        f"{EXPECTED_QUERIES_PER_OUTER_FOLD} test"
    )
    print(f"Positive threshold: {threshold}")
    print(f"Final cross-validation run: {FINAL_CV_RUN}")

    fold_results: list[dict[str, object]] = []

    for outer_fold in FOLDS_TO_RUN:
        fold_results.append(
            run_one_fold(
                outer_fold=outer_fold,
                nested_split=nested_splits[outer_fold],
                all_queries=all_queries,
                all_qrels=all_qrels,
                corpus=corpus,
                threshold=threshold,
                run_directory=run_directory,
                run_name=run_name,
                dataset_class=Dataset,
                sentence_transformer_class=SentenceTransformer,
                trainer_class=SentenceTransformerTrainer,
                training_arguments_class=(
                    SentenceTransformerTrainingArguments
                ),
                losses_module=losses,
                evaluator_class=InformationRetrievalEvaluator,
                batch_samplers=BatchSamplers,
            )
        )

    dev_fold_rows: list[dict[str, object]] = []

    for result in fold_results:
        dev_metrics = result["dev_metrics"]
        assert isinstance(dev_metrics, Mapping)

        dev_fold_rows.append(
            {
                "outer_fold": result["outer_fold"],
                "train_queries": result["train_query_count"],
                "dev_queries": result["dev_query_count"],
                "test_queries": result["test_query_count"],
                "training_pairs": result["training_pair_count"],
                **{
                    metric_name: float(metric_value)
                    for metric_name, metric_value
                    in dev_metrics.items()
                },
            }
        )

    write_table(
        run_directory / "dev_fold_summary.tsv",
        dev_fold_rows,
    )
    write_json(
        run_directory / "development_summary.json",
        {
            "folds_run": list(FOLDS_TO_RUN),
            "fold_metric_summary": summarize_fold_metrics(
                dev_fold_rows
            ),
        },
    )

    if FINAL_CV_RUN:
        all_test_rows: list[dict[str, object]] = []
        all_test_rankings: dict[
            str,
            Sequence[tuple[str, float]]
        ] = {}
        test_fold_rows: list[dict[str, object]] = []

        for result in fold_results:
            test_metrics = result["test_metrics"]
            test_rows = result["test_rows"]
            test_rankings = result["test_rankings"]

            assert isinstance(test_metrics, Mapping)
            assert isinstance(test_rows, list)
            assert isinstance(test_rankings, Mapping)

            all_test_rows.extend(test_rows)

            for query_id, ranking in test_rankings.items():
                if query_id in all_test_rankings:
                    raise ValueError(
                        f"Test query evaluated more than once: {query_id}"
                    )
                all_test_rankings[query_id] = ranking

            test_fold_rows.append(
                {
                    "outer_fold": result["outer_fold"],
                    "train_queries": result["train_query_count"],
                    "dev_queries": result["dev_query_count"],
                    "test_queries": result["test_query_count"],
                    "training_pairs": result["training_pair_count"],
                    **{
                        metric_name: float(metric_value)
                        for metric_name, metric_value
                        in test_metrics.items()
                    },
                }
            )

        if len(all_test_rows) != EXPECTED_TOTAL_QUERY_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_TOTAL_QUERY_COUNT} outer-test rows; "
                f"found {len(all_test_rows)}."
            )

        if set(all_test_rankings) != set(all_queries):
            raise ValueError(
                "The combined outer-test run does not contain every query "
                "exactly once."
            )

        all_test_rows.sort(
            key=lambda row: (
                int(row["outer_fold"]),
                str(row["query_id"]),
            )
        )
        pooled_metrics = aggregate_per_query_rows(
            all_test_rows
        )
        fold_metric_summary = summarize_fold_metrics(
            test_fold_rows
        )

        write_table(
            run_directory / "fold_summary.tsv",
            test_fold_rows,
        )
        write_table(
            run_directory
            / "all_test_per_query_metrics.tsv",
            all_test_rows,
        )
        write_trec_run(
            run_directory / "all_test_run.trec",
            all_test_rankings,
            run_name + "__all_outer_test",
        )
        write_json(
            run_directory
            / "cross_validation_metrics.json",
            {
                "protocol": (
                    "nested_five_fold_cross_validation"
                ),
                "query_count": EXPECTED_TOTAL_QUERY_COUNT,
                "fold_count": len(EXPECTED_FOLDS),
                "relevance_threshold": threshold,
                "pooled_query_macro_metrics": pooled_metrics,
                "fold_metric_summary": fold_metric_summary,
            },
        )

        print()
        print("=" * 72)
        print("Combined outer-test metrics over all 100 queries")
        print("=" * 72)

        for metric_name in sorted(
            pooled_metrics,
            key=lambda name: (
                int(name.split("@")[1]),
                name.split("@")[0],
            ),
        ):
            fold_stats = fold_metric_summary[metric_name]

            print(
                f"{metric_name:>12}: "
                f"{pooled_metrics[metric_name]:.6f} "
                f"(fold mean "
                f"{fold_stats['mean']:.6f}, "
                f"SD {fold_stats['standard_deviation']:.6f})"
            )
    else:
        print()
        print(
            "Outer-test evaluation was not performed. Keep this mode while "
            "developing the configuration. After every modelling and "
            "hyperparameter choice is frozen, set FINAL_CV_RUN = True and "
            "run all five folds once."
        )

    print()
    print(f"Output: {run_directory}")


if __name__ == "__main__":
    main()
