#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple zero-shot retrieval runner for EUR-LexRD.

Run this file directly in your IDE.

To use a different Sentence Transformers model, change MODEL_NAME.
To use a model from another family or an external API, replace
load_model() and encode().

Dependencies:
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

QRELS_SOURCE = "human"       # "human" or "citations"
QUERY_VIEW = "summary"       # "summary", "facts", or "full"
CORPUS_VIEW = "summary"         # "summary", "facts", "reasoning", or "full"
USE_GRADED_HUMAN_QRELS = True

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LABEL = "all-MiniLM-L6-v2"
DEVICE: str | None = None
TRUST_REMOTE_CODE = False

QUERY_PREFIX = ""
DOCUMENT_PREFIX = ""
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH: int | None = None

RETRIEVE_TOP_K = 100
METRIC_CUTOFFS = (5, 10, 20, 30)

# None means relevance >= 2 for graded human qrels and >= 1 otherwise.
BINARY_RELEVANCE_THRESHOLD: int | None = None

EXCLUDE_QUERY_DOCUMENT = True
OUTPUT_ROOT = Path("results/zero_shot")
OVERWRITE_OUTPUT = True


# =============================================================================
# Model adapter
# =============================================================================

def load_model() -> Any:
    """Replace this function when using another model family or API."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install the default adapter with:\n"
            "python -m pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if MAX_SEQUENCE_LENGTH is not None:
        model.max_seq_length = MAX_SEQUENCE_LENGTH

    return model


def encode(
    model: Any,
    texts: Sequence[str],
    *,
    role: str,
) -> np.ndarray:
    """
    Return one normalized embedding per text.

    Replace this function when using another model family or API.
    The returned array must have shape [len(texts), embedding_dimension].
    """
    if role not in {"query", "document"}:
        raise ValueError(
            f"role must be 'query' or 'document', received {role!r}."
        )

    prefix = QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX

    vectors = model.encode(
        [prefix + text for text in texts],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("encode() returned an invalid embedding array.")

    return vectors


# =============================================================================
# File handling
# =============================================================================

def find_project_root() -> Path:
    """Find the repository root from the script path or working directory."""
    if PROJECT_ROOT_OVERRIDE is not None:
        path = PROJECT_ROOT_OVERRIDE.expanduser().resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"PROJECT_ROOT_OVERRIDE is not a directory: {path}"
            )

        return path

    candidates: list[Path] = []

    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        candidates.extend([script_path.parent, *script_path.parents])

    current_directory = Path.cwd().resolve()
    candidates.extend(
        [current_directory, *current_directory.parents]
    )

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
    """Find data/eval_views or eval_views."""
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


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield JSON objects from a JSONL file."""
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
                    f"Expected a JSON object in {path} at line "
                    f"{line_number}."
                )

            yield record


def load_queries(path: Path) -> tuple[list[str], list[str]]:
    """Load query IDs and query texts."""
    query_ids: list[str] = []
    texts: list[str] = []

    for record in iter_jsonl(path):
        query_id = str(record.get("query_id", "")).strip()
        text = str(record.get("text", "")).strip()

        if not query_id or not text:
            raise ValueError(
                f"A query in {path} has a blank query_id or text."
            )

        query_ids.append(query_id)
        texts.append(text)

    if not query_ids:
        raise ValueError(f"No queries found in {path}.")

    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"Duplicate query IDs found in {path}.")

    return query_ids, texts


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load four-column TREC qrels."""
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
            if not row or all(not str(value).strip() for value in row):
                continue

            if len(row) != 4:
                raise ValueError(
                    f"Expected four columns in {path} at line "
                    f"{line_number}."
                )

            query_id = str(row[0]).strip()
            document_id = str(row[2]).strip()
            relevance_text = str(row[3]).strip()

            try:
                relevance_value = int(relevance_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid relevance in {path} at line "
                    f"{line_number}: {relevance_text!r}"
                ) from exc

            if document_id in qrels[query_id]:
                raise ValueError(
                    f"Duplicate qrel pair in {path}: "
                    f"{query_id} -> {document_id}"
                )

            qrels[query_id][document_id] = relevance_value

    if not qrels:
        raise ValueError(f"No qrels found in {path}.")

    return dict(qrels)


def encode_queries(
    model: Any,
    texts: Sequence[str],
) -> np.ndarray:
    """Encode queries in batches."""
    batches: list[np.ndarray] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        batches.append(
            encode(
                model,
                batch,
                role="query",
            )
        )

    return np.vstack(batches).astype(np.float32, copy=False)


def encode_corpus(
    model: Any,
    path: Path,
) -> tuple[list[str], np.ndarray]:
    """Load and encode the corpus in batches."""
    document_ids: list[str] = []
    embedding_batches: list[np.ndarray] = []
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    seen_document_ids: set[str] = set()

    def flush_batch() -> None:
        if not batch_ids:
            return

        document_ids.extend(batch_ids)
        embedding_batches.append(
            encode(
                model,
                batch_texts,
                role="document",
            )
        )
        batch_ids.clear()
        batch_texts.clear()

    for record in iter_jsonl(path):
        document_id = str(record.get("doc_id", "")).strip()
        text = str(record.get("text", "")).strip()

        if not document_id:
            raise ValueError(
                f"A corpus row in {path} has no doc_id."
            )

        if document_id in seen_document_ids:
            raise ValueError(
                f"Duplicate document ID found in {path}: {document_id}"
            )

        seen_document_ids.add(document_id)

        if not text:
            continue

        batch_ids.append(document_id)
        batch_texts.append(text)

        if len(batch_ids) >= BATCH_SIZE:
            flush_batch()

            if len(document_ids) % (BATCH_SIZE * 25) == 0:
                print(
                    f"Encoded {len(document_ids):,} documents..."
                )

    flush_batch()

    if not document_ids:
        raise ValueError(f"No non-empty documents found in {path}.")

    return (
        document_ids,
        np.vstack(embedding_batches).astype(
            np.float32,
            copy=False,
        ),
    )


# =============================================================================
# Retrieval
# =============================================================================

def retrieve(
    query_ids: Sequence[str],
    query_vectors: np.ndarray,
    document_ids: Sequence[str],
    document_vectors: np.ndarray,
) -> dict[str, list[tuple[str, float]]]:
    """Rank corpus documents by cosine similarity."""
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValueError(
            "Query and document embedding dimensions do not match."
        )

    document_index = {
        document_id: index
        for index, document_id in enumerate(document_ids)
    }

    top_k = min(RETRIEVE_TOP_K, len(document_ids))
    rankings: dict[str, list[tuple[str, float]]] = {}

    for number, (query_id, query_vector) in enumerate(
        zip(query_ids, query_vectors),
        start=1,
    ):
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
            (
                document_ids[int(index)],
                float(scores[int(index)]),
            )
            for index in selected
            if np.isfinite(scores[int(index)])
        ]

        if number % 10 == 0 or number == len(query_ids):
            print(
                f"Retrieved {number}/{len(query_ids)} queries."
            )

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
    """Calculate precision at k."""
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
    """Calculate recall at k."""
    relevant_documents = {
        document_id
        for document_id, relevance in qrels.items()
        if relevance >= threshold
    }

    if not relevant_documents:
        return 0.0

    retrieved_relevant = sum(
        document_id in relevant_documents
        for document_id in ranked[:k]
    )

    return retrieved_relevant / len(relevant_documents)


def average_precision_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    threshold: int,
    k: int,
) -> float:
    """Calculate average precision at k."""
    total_relevant = sum(
        relevance >= threshold
        for relevance in qrels.values()
    )

    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    score = 0.0

    for rank, document_id in enumerate(
        ranked[:k],
        start=1,
    ):
        if qrels.get(document_id, 0) >= threshold:
            relevant_seen += 1
            score += relevant_seen / rank

    return score / min(total_relevant, k)


def dcg(relevances: Sequence[int]) -> float:
    """Calculate discounted cumulative gain."""
    return sum(
        ((2 ** relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(
            relevances,
            start=1,
        )
    )


def ndcg_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain at k."""
    actual = [
        qrels.get(document_id, 0)
        for document_id in ranked[:k]
    ]
    ideal = sorted(
        qrels.values(),
        reverse=True,
    )[:k]
    ideal_score = dcg(ideal)

    return dcg(actual) / ideal_score if ideal_score else 0.0


def judged_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    """
    Calculate the proportion of the top-k ranking covered by the qrels.

    Judged@k is a judgment-coverage diagnostic, not a retrieval-effectiveness
    metric. A retrieved document counts as judged when its document ID occurs
    in the query's qrels, irrespective of its relevance grade.
    """
    returned_count = min(k, len(ranked))

    if returned_count == 0:
        return 0.0

    judged_count = sum(
        document_id in qrels
        for document_id in ranked[:k]
    )

    return judged_count / returned_count


def evaluate(
    query_ids: Sequence[str],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    threshold: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Calculate per-query and aggregate retrieval metrics."""
    rows: list[dict[str, object]] = []

    for query_id in query_ids:
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
                ranked,
                query_qrels,
                threshold,
                k,
            )
            row[f"Recall@{k}"] = recall_at_k(
                ranked,
                query_qrels,
                threshold,
                k,
            )
            row[f"MAP@{k}"] = average_precision_at_k(
                ranked,
                query_qrels,
                threshold,
                k,
            )
            row[f"nDCG@{k}"] = ndcg_at_k(
                ranked,
                query_qrels,
                k,
            )
            row[f"Judged@{k}"] = judged_at_k(
                ranked,
                query_qrels,
                k,
            )

        rows.append(row)

    non_metric_fields = {
        "query_id",
        "judged_documents",
        "relevant_documents",
    }

    metric_names = [
        field_name
        for field_name in rows[0]
        if field_name not in non_metric_fields
    ]

    aggregate = {
        metric_name: float(
            np.mean(
                [
                    float(row[metric_name])
                    for row in rows
                ]
            )
        )
        for metric_name in metric_names
    }

    aggregate["mean_judged_documents_per_query"] = float(
        np.mean(
            [
                int(row["judged_documents"])
                for row in rows
            ]
        )
    )
    aggregate["mean_relevant_documents_per_query"] = float(
        np.mean(
            [
                int(row["relevant_documents"])
                for row in rows
            ]
        )
    )

    return rows, aggregate


# =============================================================================
# Outputs
# =============================================================================

def safe_name(value: str) -> str:
    """Return a filesystem-safe name."""
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    ).strip("._")

    return cleaned or "run"


def write_outputs(
    output_directory: Path,
    run_name: str,
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    rows: Sequence[Mapping[str, object]],
    metrics: Mapping[str, float],
    config: Mapping[str, object],
) -> None:
    """Write ranking, metrics, and configuration files."""
    if output_directory.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(output_directory)

        shutil.rmtree(output_directory)

    output_directory.mkdir(parents=True)

    with (output_directory / "run.trec").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
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

    with (output_directory / "per_query_metrics.tsv").open(
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

    (output_directory / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (output_directory / "run_config.json").write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Reproducibility
# =============================================================================

def set_random_seeds() -> None:
    """Set the random seeds used by the runner."""
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
    """Run zero-shot retrieval and evaluation."""
    set_random_seeds()

    project_root = find_project_root()
    eval_views_root = find_eval_views_root(project_root)

    view_name = f"q_{QUERY_VIEW}__c_{CORPUS_VIEW}"
    view_directory = (
        eval_views_root
        / QRELS_SOURCE
        / view_name
    )

    if QRELS_SOURCE == "human" and USE_GRADED_HUMAN_QRELS:
        qrels_filename = "qrels_selected_graded.tsv"
        qrels_label = "human_graded"
        default_threshold = 2
    else:
        qrels_filename = "qrels_selected.tsv"
        qrels_label = (
            "human_binary"
            if QRELS_SOURCE == "human"
            else "citations"
        )
        default_threshold = 1

    threshold = (
        BINARY_RELEVANCE_THRESHOLD
        if BINARY_RELEVANCE_THRESHOLD is not None
        else default_threshold
    )

    corpus_path = view_directory / "corpus.jsonl"
    queries_path = view_directory / "queries.jsonl"
    qrels_path = view_directory / qrels_filename

    for path in (
        corpus_path,
        queries_path,
        qrels_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"View: {view_directory}")
    print(f"Model: {MODEL_NAME}")
    print("Protocol: zero-shot evaluation over all queries")
    print()

    query_ids, query_texts = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    missing_queries = [
        query_id
        for query_id in query_ids
        if query_id not in qrels
    ]

    if missing_queries:
        raise ValueError(
            "Queries without qrels: "
            + ", ".join(missing_queries[:10])
        )

    model = load_model()

    print(f"Encoding {len(query_ids)} queries...")
    query_vectors = encode_queries(
        model,
        query_texts,
    )

    print("Encoding corpus...")
    document_ids, document_vectors = encode_corpus(
        model,
        corpus_path,
    )

    print(f"Encoded {len(document_ids):,} documents.")

    rankings = retrieve(
        query_ids,
        query_vectors,
        document_ids,
        document_vectors,
    )

    per_query, aggregate = evaluate(
        query_ids,
        rankings,
        qrels,
        threshold,
    )

    run_name = safe_name(
        f"{MODEL_LABEL}_{qrels_label}_{QUERY_VIEW}_{CORPUS_VIEW}"
    )
    output_directory = (
        project_root
        / OUTPUT_ROOT
        / run_name
    )

    write_outputs(
        output_directory,
        run_name,
        rankings,
        per_query,
        aggregate,
        {
            "protocol": "zero_shot_all_queries",
            "random_seed": RANDOM_SEED,
            "model_name": MODEL_NAME,
            "model_label": MODEL_LABEL,
            "qrels_source": qrels_label,
            "query_view": QUERY_VIEW,
            "corpus_view": CORPUS_VIEW,
            "binary_relevance_threshold": threshold,
            "retrieve_top_k": RETRIEVE_TOP_K,
            "metric_cutoffs": list(METRIC_CUTOFFS),
            "query_prefix": QUERY_PREFIX,
            "document_prefix": DOCUMENT_PREFIX,
            "batch_size": BATCH_SIZE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "exclude_query_document": EXCLUDE_QUERY_DOCUMENT,
        },
    )

    print()
    print("Aggregate metrics")
    print("-----------------")

    for metric, value in sorted(aggregate.items()):
        print(f"{metric:>34}: {value:.6f}")

    print()
    print(
        "Judged@k measures qrel coverage of the retrieved ranking; "
        "it is not an effectiveness score."
    )
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    main()
