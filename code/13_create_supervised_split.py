#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_create_supervised_split.py

Create the canonical 70/10/20 query-level supervised split for EUR-LexRD.

Run this file directly in your IDE.
Edit only the configuration constants below when needed.

Canonical input
---------------
data/qrels/qrels_human_graded.tsv

Canonical outputs
-----------------
data/splits/holdout_70train_10dev_20test/
    query_assignments.tsv
    split_config.json
    split_summary.txt
    train/
        query_ids.txt
        qrels/
        shared/
    dev/
        query_ids.txt
        qrels/
        shared/
    test/
        query_ids.txt
        qrels/
        shared/
    eval_views/
        human/<view>/train|dev|test/
        citations/<view>/train|dev|test/

The split is made at query level. Every qrel and candidate-pool row associated
with a query stays in that query's partition. Candidate corpora remain shared.

Script 17_basic_training_script_holdout_biencoder.py consumes:
data/splits/holdout_70train_10dev_20test/query_assignments.tsv

Dependencies
------------
Python standard library only, plus pipeline_common.py from this repository.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from pipeline_common import (
        find_project_root,
        resolve_project_path,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import pipeline_common.py. Place this script and "
        "pipeline_common.py together in the repository's scripts/ or code/ "
        "directory."
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 123

PROJECT_ROOT_OVERRIDE: Path | None = None

EXPECTED_QUERY_COUNT: int | None = 100
TRAIN_QUERY_COUNT = 70
DEV_QUERY_COUNT = 10
TEST_QUERY_COUNT = 20

QRELS_DIRECTORY = Path("data/qrels")
MASTER_QRELS_FILENAME = "qrels_human_graded.tsv"
QREL_FILENAMES: tuple[str, ...] = (
    "qrels_human_binary.tsv",
    "qrels_human_graded.tsv",
    "qrels_citations.tsv",
)

SHARED_DIRECTORY = Path("data/shared")
QUERY_POOL_FILENAME = "query_pool.jsonl"
CANDIDATE_POOL_FILENAME = "candidate_pool.tsv"

EVAL_VIEWS_DIRECTORY = Path("data/eval_views")
MATERIALIZE_EVAL_VIEW_SPLITS = True

# False avoids storing the same large candidate corpus three times per view.
# A relative corpus_source.txt pointer is written instead.
COPY_EVAL_VIEW_CORPORA = False

OUTPUT_DIRECTORY = Path(
    "data/splits/holdout_70train_10dev_20test"
)
OVERWRITE_OUTPUT_DIRECTORY = True

RELEVANT_GRADE_THRESHOLD = 2

QUERY_VIEWS: tuple[str, ...] = (
    "summary",
    "facts",
    "full",
)
CORPUS_VIEWS: tuple[str, ...] = (
    "summary",
    "facts",
    "reasoning",
    "full",
)
QRELS_SOURCES: tuple[str, ...] = (
    "human",
    "citations",
)

FAIL_ON_UNEXPECTED_EVAL_VIEW_FOLDERS = True


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class QrelRow:
    query_id: str
    iteration: str
    document_id: str
    relevance: int


@dataclass(frozen=True)
class QueryStats:
    query_id: str
    judged_pairs: int
    relevant_pairs: int
    directly_relevant_pairs: int
    relevance_sum: int

    @property
    def mean_relevance(self) -> float:
        if self.judged_pairs == 0:
            return 0.0
        return self.relevance_sum / self.judged_pairs


# =============================================================================
# General helpers
# =============================================================================

def normalize_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    write_text_atomic(
        path,
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_rows_atomic(
    path: Path,
    rows: Iterable[Sequence[object]],
    *,
    delimiter: str = "\t",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(
                handle,
                delimiter=delimiter,
                lineterminator="\n",
            )
            writer.writerows(rows)

        temporary_path.replace(path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_output_directory(
    path: Path,
    *,
    protected_paths: Sequence[Path],
) -> None:
    resolved_output = path.resolve()
    protected = {
        protected_path.resolve()
        for protected_path in protected_paths
    }

    if resolved_output in protected:
        raise ValueError(
            f"Refusing to replace protected directory: {resolved_output}"
        )

    if path.exists():
        if not OVERWRITE_OUTPUT_DIRECTORY:
            raise FileExistsError(
                f"Output directory already exists: {path}"
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Qrels
# =============================================================================

def read_qrels(path: Path) -> list[QrelRow]:
    """Read a TREC-style four-column qrels file."""
    if not path.is_file():
        raise FileNotFoundError(f"Qrels file does not exist: {path}")

    rows: list[QrelRow] = []
    seen_pairs: set[tuple[str, str]] = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(handle, delimiter="\t")

        for line_number, row in enumerate(reader, start=1):
            if not row or all(not normalize_text(value) for value in row):
                continue

            if len(row) != 4:
                raise ValueError(
                    f"Expected four tab-separated columns in {path} at "
                    f"line {line_number}; found {len(row)} columns."
                )

            query_id = normalize_text(row[0])
            iteration = normalize_text(row[1]) or "0"
            document_id = normalize_text(row[2])
            relevance_text = normalize_text(row[3])

            if (
                line_number == 1
                and query_id.lower() == "query_id"
                and relevance_text.lower() == "relevance"
            ):
                continue

            if not query_id or not document_id:
                raise ValueError(
                    f"Blank query or document ID in {path} at "
                    f"line {line_number}."
                )

            try:
                relevance = int(relevance_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid relevance value {relevance_text!r} in {path} "
                    f"at line {line_number}."
                ) from exc

            pair = (query_id, document_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate qrel pair in {path}: "
                    f"{query_id} -> {document_id}"
                )
            seen_pairs.add(pair)

            rows.append(
                QrelRow(
                    query_id=query_id,
                    iteration=iteration,
                    document_id=document_id,
                    relevance=relevance,
                )
            )

    if not rows:
        raise ValueError(f"Qrels file contains no rows: {path}")

    return rows


def write_qrels(path: Path, rows: Iterable[QrelRow]) -> int:
    materialized_rows = list(rows)

    write_rows_atomic(
        path,
        (
            (
                row.query_id,
                row.iteration,
                row.document_id,
                row.relevance,
            )
            for row in materialized_rows
        ),
    )

    return len(materialized_rows)


def build_query_stats(
    qrels: Sequence[QrelRow],
) -> dict[str, QueryStats]:
    judged_pairs: Counter[str] = Counter()
    relevant_pairs: Counter[str] = Counter()
    directly_relevant_pairs: Counter[str] = Counter()
    relevance_sum: Counter[str] = Counter()

    for row in qrels:
        judged_pairs[row.query_id] += 1
        relevance_sum[row.query_id] += row.relevance

        if row.relevance >= RELEVANT_GRADE_THRESHOLD:
            relevant_pairs[row.query_id] += 1

        if row.relevance == 3:
            directly_relevant_pairs[row.query_id] += 1

    return {
        query_id: QueryStats(
            query_id=query_id,
            judged_pairs=judged_pairs[query_id],
            relevant_pairs=relevant_pairs[query_id],
            directly_relevant_pairs=directly_relevant_pairs[query_id],
            relevance_sum=relevance_sum[query_id],
        )
        for query_id in judged_pairs
    }


# =============================================================================
# Split construction
# =============================================================================

def create_holdout_assignments(
    stats_by_query: Mapping[str, QueryStats],
) -> dict[str, str]:
    """
    Create a deterministic, approximately relevance-balanced 70/10/20 split.

    Queries are ordered by their aggregate relevance profile and divided into
    blocks of ten. Each block contributes seven training, one development, and
    two test queries. Labels within each block are shuffled with RANDOM_SEED.
    """
    query_ids = list(stats_by_query)
    expected_total = (
        TRAIN_QUERY_COUNT
        + DEV_QUERY_COUNT
        + TEST_QUERY_COUNT
    )

    if len(query_ids) != expected_total:
        raise ValueError(
            f"The configured split requires {expected_total} queries, but "
            f"{len(query_ids)} unique queries were found in the master qrels."
        )

    if (
        EXPECTED_QUERY_COUNT is not None
        and len(query_ids) != EXPECTED_QUERY_COUNT
    ):
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} queries, but found "
            f"{len(query_ids)}."
        )

    block_size = 10
    if len(query_ids) % block_size != 0:
        raise ValueError(
            f"The query count must be divisible by {block_size} for the "
            "configured block-stratified split."
        )

    rng = random.Random(RANDOM_SEED)
    tie_breakers = {
        query_id: rng.random()
        for query_id in query_ids
    }

    ordered_queries = sorted(
        query_ids,
        key=lambda query_id: (
            -stats_by_query[query_id].relevant_pairs,
            -stats_by_query[query_id].directly_relevant_pairs,
            -stats_by_query[query_id].judged_pairs,
            -stats_by_query[query_id].mean_relevance,
            tie_breakers[query_id],
            query_id,
        ),
    )

    split_template = (
        ["train"] * 7
        + ["dev"]
        + ["test"] * 2
    )
    assignments: dict[str, str] = {}

    for block_start in range(
        0,
        len(ordered_queries),
        block_size,
    ):
        block = ordered_queries[
            block_start:block_start + block_size
        ]
        block_labels = split_template.copy()
        rng.shuffle(block_labels)

        for query_id, split_name in zip(
            block,
            block_labels,
            strict=True,
        ):
            assignments[query_id] = split_name

    expected_counts = {
        "train": TRAIN_QUERY_COUNT,
        "dev": DEV_QUERY_COUNT,
        "test": TEST_QUERY_COUNT,
    }
    actual_counts = Counter(assignments.values())

    if dict(actual_counts) != expected_counts:
        raise RuntimeError(
            f"Unexpected split counts: {dict(actual_counts)}; "
            f"expected {expected_counts}."
        )

    return assignments


def query_ids_for_split(
    assignments: Mapping[str, str],
    split_name: str,
) -> list[str]:
    return sorted(
        query_id
        for query_id, assigned_split in assignments.items()
        if assigned_split == split_name
    )


def write_assignment_file(
    path: Path,
    assignments: Mapping[str, str],
    stats_by_query: Mapping[str, QueryStats],
) -> None:
    split_order = {
        "train": 0,
        "dev": 1,
        "test": 2,
    }

    rows: list[Sequence[object]] = [
        (
            "query_id",
            "split",
            "judged_pairs",
            "relevant_pairs_grade_ge_2",
            "directly_relevant_pairs_grade_3",
            "mean_relevance",
        )
    ]

    for query_id in sorted(
        assignments,
        key=lambda value: (
            split_order[assignments[value]],
            value,
        ),
    ):
        stats = stats_by_query[query_id]
        rows.append(
            (
                query_id,
                assignments[query_id],
                stats.judged_pairs,
                stats.relevant_pairs,
                stats.directly_relevant_pairs,
                f"{stats.mean_relevance:.6f}",
            )
        )

    write_rows_atomic(path, rows)


# =============================================================================
# Query-dependent shared files
# =============================================================================

def extract_query_id_from_json(
    record: Mapping[str, Any],
) -> str | None:
    for key in (
        "query_id",
        "doc_id",
        "document_id",
        "id",
        "_id",
        "ecli",
    ):
        value = normalize_text(record.get(key))
        if value:
            return value

    metadata = record.get("case_metadata")
    if isinstance(metadata, Mapping):
        for key in (
            "query_id",
            "doc_id",
            "document_id",
            "id",
            "_id",
            "ecli",
        ):
            value = normalize_text(metadata.get(key))
            if value:
                return value

    return None


def filter_jsonl_by_query(
    source_path: Path,
    destination_path: Path,
    allowed_query_ids: set[str],
) -> int:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(
        destination_path.suffix + ".tmp"
    )

    written = 0

    try:
        with source_path.open(
            "r",
            encoding="utf-8-sig",
        ) as source_handle, temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as destination_handle:
            for line_number, line in enumerate(
                source_handle,
                start=1,
            ):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {source_path} at line "
                        f"{line_number}."
                    ) from exc

                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"Expected a JSON object in {source_path} at line "
                        f"{line_number}."
                    )

                query_id = extract_query_id_from_json(record)
                if query_id is None:
                    raise ValueError(
                        f"Could not identify the query ID in {source_path} "
                        f"at line {line_number}."
                    )

                if query_id in allowed_query_ids:
                    destination_handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    written += 1

        temporary_path.replace(destination_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if written != len(allowed_query_ids):
        raise ValueError(
            f"Expected {len(allowed_query_ids)} query rows from "
            f"{source_path}, but wrote {written}."
        )

    return written


def filter_headered_tsv_by_query(
    source_path: Path,
    destination_path: Path,
    allowed_query_ids: set[str],
) -> int:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(
        destination_path.suffix + ".tmp"
    )

    written = 0

    try:
        with source_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source_handle:
            reader = csv.DictReader(
                source_handle,
                delimiter="\t",
            )
            fieldnames = reader.fieldnames or []

            if "query_id" not in fieldnames:
                raise ValueError(
                    f"Expected a query_id column in {source_path}."
                )

            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as destination_handle:
                writer = csv.DictWriter(
                    destination_handle,
                    fieldnames=fieldnames,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()

                for row in reader:
                    query_id = normalize_text(row.get("query_id"))
                    if query_id in allowed_query_ids:
                        writer.writerow(row)
                        written += 1

        temporary_path.replace(destination_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return written


# =============================================================================
# Evaluation views
# =============================================================================

def expected_eval_view_paths() -> tuple[Path, ...]:
    return tuple(
        Path(source)
        / f"q_{query_view}__c_{corpus_view}"
        for source in QRELS_SOURCES
        for query_view in QUERY_VIEWS
        for corpus_view in CORPUS_VIEWS
    )


def read_query_ids_from_jsonl(path: Path) -> list[str]:
    query_ids: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc

            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Expected a JSON object in {path} at line "
                    f"{line_number}."
                )

            query_id = normalize_text(record.get("query_id"))
            if not query_id:
                raise ValueError(
                    f"Blank query_id in {path} at line {line_number}."
                )

            query_ids.append(query_id)

    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"Duplicate query IDs found in {path}.")

    return query_ids


def validate_eval_view_folders(
    eval_views_root: Path,
    master_query_ids: set[str],
) -> list[tuple[Path, Path]]:
    if not eval_views_root.is_dir():
        raise FileNotFoundError(
            f"Evaluation views directory does not exist: "
            f"{eval_views_root}"
        )

    expected_paths = expected_eval_view_paths()
    expected_set = set(expected_paths)

    discovered_set = {
        path.parent.relative_to(eval_views_root)
        for path in eval_views_root.rglob("queries.jsonl")
        if path.parent.parent.name in QRELS_SOURCES
    }

    missing = sorted(expected_set - discovered_set)
    unexpected = sorted(discovered_set - expected_set)

    if missing:
        raise FileNotFoundError(
            "Missing evaluation-view folder(s): "
            + ", ".join(str(path) for path in missing)
        )

    if unexpected and FAIL_ON_UNEXPECTED_EVAL_VIEW_FOLDERS:
        raise ValueError(
            "Unexpected evaluation-view folder(s): "
            + ", ".join(str(path) for path in unexpected)
        )

    validated: list[tuple[Path, Path]] = []

    for relative_path in expected_paths:
        view_folder = eval_views_root / relative_path
        required_files = [
            view_folder / "corpus.jsonl",
            view_folder / "queries.jsonl",
            view_folder / "qrels_selected.tsv",
        ]

        if relative_path.parts[0] == "human":
            required_files.append(
                view_folder / "qrels_selected_graded.tsv"
            )

        missing_files = [
            path
            for path in required_files
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                "Missing evaluation-view file(s): "
                + ", ".join(str(path) for path in missing_files)
            )

        query_ids = set(
            read_query_ids_from_jsonl(
                view_folder / "queries.jsonl"
            )
        )
        if query_ids != master_query_ids:
            missing_queries = sorted(master_query_ids - query_ids)
            extra_queries = sorted(query_ids - master_query_ids)
            raise ValueError(
                f"Query-ID mismatch in {view_folder}. "
                f"Missing examples: {missing_queries[:10]}; "
                f"extra examples: {extra_queries[:10]}."
            )

        for qrels_name in (
            "qrels_selected.tsv",
            "qrels_selected_graded.tsv",
        ):
            qrels_path = view_folder / qrels_name
            if not qrels_path.is_file():
                continue

            qrel_query_ids = {
                row.query_id
                for row in read_qrels(qrels_path)
            }
            unknown_query_ids = sorted(
                qrel_query_ids - master_query_ids
            )
            if unknown_query_ids:
                raise ValueError(
                    f"Unknown query IDs in {qrels_path}: "
                    f"{unknown_query_ids[:20]}"
                )

        validated.append((relative_path, view_folder))

    return validated


def materialize_eval_view_splits(
    eval_views_root: Path,
    output_root: Path,
    assignments: Mapping[str, str],
) -> dict[str, int]:
    master_query_ids = set(assignments)
    view_folders = validate_eval_view_folders(
        eval_views_root,
        master_query_ids,
    )

    counters: Counter[str] = Counter()
    counters["view_folders"] = len(view_folders)

    for relative_path, view_folder in view_folders:
        destination_folder = output_root / relative_path
        destination_folder.mkdir(parents=True, exist_ok=True)

        corpus_path = view_folder / "corpus.jsonl"

        if COPY_EVAL_VIEW_CORPORA:
            shutil.copy2(
                corpus_path,
                destination_folder / "corpus.jsonl",
            )
        else:
            relative_corpus_path = os.path.relpath(
                corpus_path.resolve(),
                start=destination_folder.resolve(),
            )
            write_text_atomic(
                destination_folder / "corpus_source.txt",
                relative_corpus_path + "\n",
            )

        for split_name in ("train", "dev", "test"):
            allowed_ids = set(
                query_ids_for_split(
                    assignments,
                    split_name,
                )
            )
            split_folder = destination_folder / split_name

            query_count = filter_jsonl_by_query(
                view_folder / "queries.jsonl",
                split_folder / "queries.jsonl",
                allowed_ids,
            )
            counters[f"queries_{split_name}"] += query_count

            for qrels_name in (
                "qrels_selected.tsv",
                "qrels_selected_graded.tsv",
            ):
                source_qrels = view_folder / qrels_name
                if not source_qrels.is_file():
                    continue

                selected_rows = [
                    row
                    for row in read_qrels(source_qrels)
                    if row.query_id in allowed_ids
                ]
                row_count = write_qrels(
                    split_folder / qrels_name,
                    selected_rows,
                )
                counters[
                    f"{Path(qrels_name).stem}_{split_name}"
                ] += row_count

    return dict(counters)


# =============================================================================
# Reporting
# =============================================================================

def relevance_summary(
    split_name: str,
    query_ids: Sequence[str],
    stats_by_query: Mapping[str, QueryStats],
) -> list[str]:
    relevant_counts = [
        stats_by_query[query_id].relevant_pairs
        for query_id in query_ids
    ]
    direct_counts = [
        stats_by_query[query_id].directly_relevant_pairs
        for query_id in query_ids
    ]
    judged_counts = [
        stats_by_query[query_id].judged_pairs
        for query_id in query_ids
    ]

    def mean(values: Sequence[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return [
        split_name,
        f"  queries: {len(query_ids)}",
        f"  mean judged pairs per query: {mean(judged_counts):.3f}",
        f"  mean relevant pairs per query: {mean(relevant_counts):.3f}",
        f"  mean directly relevant pairs per query: {mean(direct_counts):.3f}",
    ]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )

    qrels_root = resolve_project_path(
        project_root,
        QRELS_DIRECTORY,
    )
    shared_root = resolve_project_path(
        project_root,
        SHARED_DIRECTORY,
    )
    eval_views_root = resolve_project_path(
        project_root,
        EVAL_VIEWS_DIRECTORY,
    )
    output_root = resolve_project_path(
        project_root,
        OUTPUT_DIRECTORY,
    )

    master_qrels_path = (
        qrels_root / MASTER_QRELS_FILENAME
    )
    master_qrels = read_qrels(master_qrels_path)
    stats_by_query = build_query_stats(master_qrels)
    assignments = create_holdout_assignments(
        stats_by_query
    )

    prepare_output_directory(
        output_root,
        protected_paths=(
            project_root,
            qrels_root,
            shared_root,
            eval_views_root,
        ),
    )

    write_assignment_file(
        output_root / "query_assignments.tsv",
        assignments,
        stats_by_query,
    )

    write_json_atomic(
        output_root / "split_config.json",
        {
            "split_name": "holdout_70train_10dev_20test",
            "random_seed": RANDOM_SEED,
            "query_level_split": True,
            "expected_query_count": EXPECTED_QUERY_COUNT,
            "train_queries": TRAIN_QUERY_COUNT,
            "dev_queries": DEV_QUERY_COUNT,
            "test_queries": TEST_QUERY_COUNT,
            "master_qrels": str(
                master_qrels_path.relative_to(project_root)
            ),
            "eval_views_materialized": (
                MATERIALIZE_EVAL_VIEW_SPLITS
            ),
            "eval_view_corpora_copied": (
                COPY_EVAL_VIEW_CORPORA
            ),
        },
    )

    report_lines = [
        "EUR-LexRD fixed 70/10/20 query split",
        "====================================",
        f"Project root: {project_root}",
        f"Random seed: {RANDOM_SEED}",
        f"Master qrels: {master_qrels_path}",
        "",
    ]

    qrels_cache: dict[str, list[QrelRow]] = {}
    for qrels_filename in QREL_FILENAMES:
        qrels_path = qrels_root / qrels_filename
        if not qrels_path.is_file():
            raise FileNotFoundError(
                f"Required qrels file not found: {qrels_path}"
            )
        qrels_cache[qrels_filename] = read_qrels(
            qrels_path
        )

    for split_name in ("train", "dev", "test"):
        split_query_ids = query_ids_for_split(
            assignments,
            split_name,
        )
        split_query_set = set(split_query_ids)
        split_root = output_root / split_name

        write_text_atomic(
            split_root / "query_ids.txt",
            "\n".join(split_query_ids) + "\n",
        )

        for qrels_filename, qrels_rows in qrels_cache.items():
            selected_rows = [
                row
                for row in qrels_rows
                if row.query_id in split_query_set
            ]
            write_qrels(
                split_root / "qrels" / qrels_filename,
                selected_rows,
            )

        query_pool_count = filter_jsonl_by_query(
            shared_root / QUERY_POOL_FILENAME,
            split_root / "shared" / QUERY_POOL_FILENAME,
            split_query_set,
        )
        candidate_pool_count = (
            filter_headered_tsv_by_query(
                shared_root / CANDIDATE_POOL_FILENAME,
                split_root
                / "shared"
                / CANDIDATE_POOL_FILENAME,
                split_query_set,
            )
        )

        report_lines.extend(
            relevance_summary(
                split_name,
                split_query_ids,
                stats_by_query,
            )
        )
        report_lines.append(
            f"  query-pool JSONL rows: {query_pool_count}"
        )
        report_lines.append(
            f"  candidate-pool TSV rows: {candidate_pool_count}"
        )
        report_lines.append("")

    if MATERIALIZE_EVAL_VIEW_SPLITS:
        eval_counters = materialize_eval_view_splits(
            eval_views_root,
            output_root / "eval_views",
            assignments,
        )

        report_lines.extend(
            [
                "Evaluation-view outputs",
                "-----------------------",
            ]
        )
        for key in sorted(eval_counters):
            report_lines.append(
                f"{key}: {eval_counters[key]:,}"
            )
        report_lines.append("")

    report_lines.extend(
        [
            "Important",
            "---------",
            "The split is query-level.",
            "Candidate corpora remain shared across partitions.",
            "Use all 100 queries for fixed zero-shot evaluation.",
            "Use this split when EUR-LexRD qrels influence training or tuning.",
            "",
        ]
    )

    summary_text = "\n".join(report_lines)
    write_text_atomic(
        output_root / "split_summary.txt",
        summary_text,
    )

    print(summary_text)
    print(f"Output written to: {output_root}")


if __name__ == "__main__":
    main()
