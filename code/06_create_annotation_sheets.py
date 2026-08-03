#!/usr/bin/env python3
"""Create anonymized annotation sheets and balanced annotator assignments.

Edit the IDE configuration near the end of the file and run the complete
script. Annotators are represented only by neutral IDs such as annotator_01.
"""

from types import SimpleNamespace
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


ECLI_RE = re.compile(
    r"\b(?:ECLI:)?(EU:(?:C|T|F):\d{4}:\d{1,6})\b",
    re.IGNORECASE,
)

DEFAULT_ANNOTATORS: Tuple[str, ...] = tuple(
    f"annotator_{index:02d}" for index in range(1, 10)
)
DEFAULT_RANDOM_SEED = 123


@dataclass(frozen=True)
class AssignmentConfig:
    annotators: Tuple[str, ...]
    shared_rows: int = 150
    double_rows_per_pair: int = 10
    warmup_single_rows: int = 150
    max_rows_per_annotator_file: int = 100
    random_seed: int = DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class OutputPaths:
    candidate_pool_path: Path
    annotation_sheet_path: Path
    annotator_dir: Path
    annotator_summary_path: Path




def build_output_paths(args: SimpleNamespace) -> OutputPaths:
    eval_dir: Path = args.eval_dir
    return OutputPaths(
        candidate_pool_path=args.candidate_pool or eval_dir / "candidate_pool.tsv",
        annotation_sheet_path=(
            args.annotation_sheet or eval_dir / "annotation_sheet_empty.csv"
        ),
        annotator_dir=args.annotator_dir or eval_dir / "annotator_assignments",
        annotator_summary_path=(
            args.summary_path or eval_dir / "annotator_assignment_summary.tsv"
        ),
    )


def validate_config(config: AssignmentConfig) -> None:
    if len(config.annotators) < 2:
        raise ValueError("At least two annotators are required.")

    normalized = [annotator.strip() for annotator in config.annotators]
    if any(not annotator for annotator in normalized):
        raise ValueError("Annotator IDs must be non-empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Annotator IDs must be unique.")

    if config.shared_rows < 0:
        raise ValueError("SHARED_ROWS must be zero or greater.")
    if config.double_rows_per_pair < 0:
        raise ValueError("DOUBLE_ROWS_PER_PAIR must be zero or greater.")
    if config.warmup_single_rows < 0:
        raise ValueError("WARMUP_SINGLE_ROWS must be zero or greater.")
    if config.max_rows_per_annotator_file <= 0:
        raise ValueError("MAX_ROWS_PER_ANNOTATOR_FILE must be greater than zero.")


def validate_paths(input_path: Path, outputs: OutputPaths) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"JSON input does not exist: {input_path}")
    if input_path.is_file() and input_path.suffix.lower() != ".json":
        raise ValueError(f"JSON input file must end in .json: {input_path}")
    if not outputs.candidate_pool_path.exists():
        raise FileNotFoundError(
            f"Candidate pool TSV does not exist: {outputs.candidate_pool_path}"
        )
    if not outputs.candidate_pool_path.is_file():
        raise ValueError(
            f"Candidate pool path is not a file: {outputs.candidate_pool_path}"
        )


def iter_json_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        if root.suffix.lower() == ".json":
            yield root
        return

    yield from sorted(root.rglob("*.json"))


def canonical_ecli(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    match = ECLI_RE.search(value)
    return f"ECLI:{match.group(1).upper()}" if match else None


def choose_doc_id(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None

    ecli = canonical_ecli(metadata.get("ecli"))
    if ecli:
        return ecli

    case_number = metadata.get("case_number")
    if isinstance(case_number, str) and case_number.strip():
        return case_number.strip()
    return None


def collect_text(node: Any) -> List[str]:
    parts: List[str] = []

    if isinstance(node, str):
        text = node.strip()
        if text:
            parts.append(text)
    elif isinstance(node, list):
        for item in node:
            parts.extend(collect_text(item))
    elif isinstance(node, dict):
        for value in node.values():
            parts.extend(collect_text(value))

    return parts


def get_summary_text(data: Dict[str, Any]) -> str:
    case_summary = data.get("case_summary")
    if not isinstance(case_summary, dict):
        return ""

    summary_text = case_summary.get("summary_text")
    if isinstance(summary_text, str):
        return summary_text.strip()
    return ""


def get_full_text(data: Dict[str, Any]) -> str:
    case_segments = data.get("case_segments")
    if not isinstance(case_segments, dict):
        return ""

    parts: List[str] = []
    for segment_value in case_segments.values():
        segment_text = "\n\n".join(collect_text(segment_value)).strip()
        if segment_text:
            parts.append(segment_text)
    return "\n\n".join(parts).strip()


def load_corpus(input_path: Path) -> Dict[str, Dict[str, Any]]:
    documents: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: Counter[str] = Counter()
    failed_files = 0

    for file_path in iter_json_files(input_path):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failed_files += 1
            print(f"[skip] Could not read {file_path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            failed_files += 1
            print(f"[skip] JSON root is not an object: {file_path}", file=sys.stderr)
            continue

        metadata = data.get("case_metadata")
        if not isinstance(metadata, dict):
            continue

        document_id = choose_doc_id(metadata)
        if not document_id:
            continue

        if document_id in documents:
            duplicate_ids[document_id] += 1

        documents[document_id] = {
            "title": (
                metadata.get("case_name")
                or metadata.get("case_number")
                or document_id
            ),
            "date": metadata.get("date_of_judgment") or "",
            "link": metadata.get("document_link") or "",
            "summary": get_summary_text(data),
            "full": get_full_text(data),
        }

    if failed_files:
        print(f"Skipped unreadable or invalid JSON files: {failed_files}")
    if duplicate_ids:
        print(
            "Warning: duplicate document IDs were found; the last file for each ID was used: "
            + ", ".join(sorted(duplicate_ids))
        )

    return documents


def read_candidate_pool(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Candidate pool has no header: {path}")

        required_columns = {"query_id", "candidate_id"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Candidate pool is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        rows: List[Dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            candidate_id = (row.get("candidate_id") or "").strip()
            if not query_id or not candidate_id:
                raise ValueError(
                    f"Candidate pool row {line_number} has an empty query_id or candidate_id."
                )
            cleaned = {key: (value or "") for key, value in row.items() if key is not None}
            cleaned["query_id"] = query_id
            cleaned["candidate_id"] = candidate_id
            rows.append(cleaned)

    if not rows:
        raise ValueError(f"Candidate pool contains no data rows: {path}")
    return rows


def chunk_list(
    items: Sequence[Dict[str, Any]],
    chunk_size: int,
) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    return [
        list(items[index : index + chunk_size])
        for index in range(0, len(items), chunk_size)
    ]


def pair_name(annotator_a: str, annotator_b: str) -> str:
    return f"{annotator_a}__{annotator_b}"


def is_random_row(row: Dict[str, Any]) -> bool:
    source_primary = str(row.get("source_primary") or "").strip().lower()
    source_all = {
        item.strip().lower()
        for item in str(row.get("source_all") or "").split("|")
        if item.strip()
    }
    return source_primary == "random" or "random" in source_all


def assign_annotation_roles(
    rows: List[Dict[str, Any]],
    config: AssignmentConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Assign each candidate row to shared, double, or single annotation."""
    pair_list = list(combinations(config.annotators, 2))
    expected_pairs = math.comb(len(config.annotators), 2)
    if len(pair_list) != expected_pairs:
        raise ValueError(
            f"Unexpected annotator-pair count: {len(pair_list)} instead of {expected_pairs}."
        )

    unique_double_rows = len(pair_list) * config.double_rows_per_pair
    required_nonrandom_rows = config.shared_rows + unique_double_rows

    nonrandom_rows = [row.copy() for row in rows if not is_random_row(row)]
    random_rows = [row.copy() for row in rows if is_random_row(row)]

    if len(nonrandom_rows) < required_nonrandom_rows:
        raise ValueError(
            "Not enough non-random rows for shared and double allocation. "
            f"Need {required_nonrandom_rows}, found {len(nonrandom_rows)}."
        )

    rng.shuffle(nonrandom_rows)
    rng.shuffle(random_rows)

    shared_block = nonrandom_rows[: config.shared_rows]
    double_start = config.shared_rows
    double_end = double_start + unique_double_rows
    double_block = nonrandom_rows[double_start:double_end]
    single_block = nonrandom_rows[double_end:] + random_rows
    rng.shuffle(single_block)

    required_single_rows = len(config.annotators) * config.warmup_single_rows
    if len(single_block) < required_single_rows:
        raise ValueError(
            "Not enough rows remain for the annotator warm-up requirement. "
            f"Need at least {required_single_rows} single rows, found {len(single_block)}."
        )

    for row in shared_block:
        row["assignment_type"] = "shared"
        row["assigned_annotators"] = "|".join(config.annotators)
        row["pair_id"] = ""

    if config.double_rows_per_pair > 0:
        for pair_index, (annotator_a, annotator_b) in enumerate(pair_list):
            start = pair_index * config.double_rows_per_pair
            end = start + config.double_rows_per_pair
            block = double_block[start:end]
            if len(block) != config.double_rows_per_pair:
                raise ValueError(
                    f"Double-assignment block for {annotator_a} and {annotator_b} "
                    f"contains {len(block)} rows instead of {config.double_rows_per_pair}."
                )

            pair_id = pair_name(annotator_a, annotator_b)
            for row in block:
                row["assignment_type"] = "double"
                row["assigned_annotators"] = f"{annotator_a}|{annotator_b}"
                row["pair_id"] = pair_id

    single_counts: Counter[str] = Counter()
    for row in single_block:
        chosen = min(
            config.annotators,
            key=lambda annotator: (single_counts[annotator], annotator),
        )
        row["assignment_type"] = "single"
        row["assigned_annotators"] = chosen
        row["pair_id"] = ""
        single_counts[chosen] += 1

    assigned_rows = shared_block + double_block + single_block
    validate_assignments(assigned_rows, config)
    return assigned_rows


def validate_assignments(
    rows: Sequence[Dict[str, Any]],
    config: AssignmentConfig,
) -> None:
    valid_types = {"shared", "double", "single"}
    known_annotators = set(config.annotators)

    for index, row in enumerate(rows, start=1):
        assignment_type = row.get("assignment_type")
        if assignment_type not in valid_types:
            raise ValueError(f"Assigned row {index} has invalid assignment_type.")

        assigned = [
            annotator
            for annotator in str(row.get("assigned_annotators") or "").split("|")
            if annotator
        ]
        if not assigned or any(annotator not in known_annotators for annotator in assigned):
            raise ValueError(f"Assigned row {index} has invalid annotator IDs.")

        if assignment_type == "shared" and set(assigned) != known_annotators:
            raise ValueError(f"Shared row {index} is not assigned to every annotator.")
        if assignment_type == "double" and len(assigned) != 2:
            raise ValueError(f"Double row {index} is not assigned to exactly two annotators.")
        if assignment_type == "single" and len(assigned) != 1:
            raise ValueError(f"Single row {index} is not assigned to exactly one annotator.")
        if is_random_row(row) and assignment_type != "single":
            raise ValueError(
                f"Random-negative row {index} was assigned as {assignment_type}; "
                "random rows must be single-assigned."
            )


def enrich_rows_with_case_metadata(
    rows: Iterable[Dict[str, Any]],
    documents: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched_rows: List[Dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id") or "").strip()
        candidate_id = str(row.get("candidate_id") or "").strip()

        if query_id not in documents:
            raise ValueError(
                f"Query ID on assigned row {row_number} was not found in the corpus: {query_id}"
            )
        if candidate_id not in documents:
            raise ValueError(
                f"Candidate ID on assigned row {row_number} was not found in the corpus: {candidate_id}"
            )

        query_record = documents[query_id]
        candidate_record = documents[candidate_id]
        output_row = row.copy()

        output_row["query_title"] = query_record.get("title", "")
        output_row["query_summary"] = query_record.get("summary", "")
        output_row["query_full"] = query_record.get("full", "")
        output_row["query_link"] = query_record.get("link", "")
        output_row["query_date"] = query_record.get("date", "")

        output_row["candidate_title"] = candidate_record.get("title", "")
        output_row["candidate_summary"] = candidate_record.get("summary", "")
        output_row["candidate_full"] = candidate_record.get("full", "")
        output_row["candidate_link"] = candidate_record.get("link", "")
        output_row["candidate_date"] = candidate_record.get("date", "")

        enriched_rows.append(output_row)

    return enriched_rows


def write_annotation_sheet(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("No annotation rows are available to write.")

    score_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if isinstance(key, str) and key.startswith("score_")
        }
    )

    fieldnames = [
        "row_id",
        "query_id",
        "query_title",
        "query_link",
        "query_date",
        "candidate_id",
        "candidate_title",
        "candidate_link",
        "candidate_date",
        "source_primary",
        "source_all",
        *score_fields,
        "assignment_type",
        "assigned_annotators",
        "pair_id",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row_id, row in enumerate(rows, start=1):
            output_row: Dict[str, Any] = {
                "row_id": row_id,
                "query_id": row["query_id"],
                "query_title": row.get("query_title", ""),
                "query_link": row.get("query_link", ""),
                "query_date": row.get("query_date", ""),
                "candidate_id": row["candidate_id"],
                "candidate_title": row.get("candidate_title", ""),
                "candidate_link": row.get("candidate_link", ""),
                "candidate_date": row.get("candidate_date", ""),
                "source_primary": row.get("source_primary", ""),
                "source_all": row.get("source_all", ""),
                "assignment_type": row["assignment_type"],
                "assigned_annotators": row["assigned_annotators"],
                "pair_id": row["pair_id"],
            }
            for score_field in score_fields:
                output_row[score_field] = row.get(score_field, "")
            writer.writerow(output_row)

    print(f"Wrote: {output_path}")


def remove_stale_annotator_parts(
    output_dir: Path,
    annotators: Sequence[str],
) -> None:
    if not output_dir.exists():
        return

    prefixes = tuple(f"{annotator}_part" for annotator in annotators)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".csv" and path.name.startswith(prefixes):
            path.unlink()


def write_annotator_files(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    config: AssignmentConfig,
    rng: random.Random,
) -> Dict[str, int]:
    """Write warm-up-first annotator files and return row counts per annotator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_annotator_parts(output_dir, config.annotators)

    annotator_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row_id, row in enumerate(rows, start=1):
        assigned = str(row["assigned_annotators"]).split("|")
        for annotator in assigned:
            annotator_rows[annotator].append(
                {
                    "row_id": row_id,
                    "query_id": row["query_id"],
                    "query_title": row.get("query_title", ""),
                    "query_summary": row.get("query_summary", ""),
                    "query_full": row.get("query_full", ""),
                    "query_link": row.get("query_link", ""),
                    "query_date": row.get("query_date", ""),
                    "candidate_id": row["candidate_id"],
                    "candidate_title": row.get("candidate_title", ""),
                    "candidate_summary": row.get("candidate_summary", ""),
                    "candidate_full": row.get("candidate_full", ""),
                    "candidate_link": row.get("candidate_link", ""),
                    "candidate_date": row.get("candidate_date", ""),
                    "label_grade": "",
                    "notes": "",
                    "_assignment_type_hidden": row["assignment_type"],
                }
            )

    fieldnames = [
        "row_id",
        "query_id",
        "query_title",
        "query_summary",
        "query_full",
        "query_link",
        "query_date",
        "candidate_id",
        "candidate_title",
        "candidate_summary",
        "candidate_full",
        "candidate_link",
        "candidate_date",
        "label_grade",
        "notes",
    ]

    row_counts: Dict[str, int] = {}
    for annotator in config.annotators:
        rows_for_annotator = annotator_rows.get(annotator, [])
        single_rows = [
            row
            for row in rows_for_annotator
            if row["_assignment_type_hidden"] == "single"
        ]
        other_rows = [
            row
            for row in rows_for_annotator
            if row["_assignment_type_hidden"] != "single"
        ]

        if len(single_rows) < config.warmup_single_rows:
            raise ValueError(
                f"{annotator} has only {len(single_rows)} single rows, fewer than "
                f"the warm-up requirement of {config.warmup_single_rows}."
            )

        rng.shuffle(single_rows)
        warmup_rows = single_rows[: config.warmup_single_rows]
        remaining_single_rows = single_rows[config.warmup_single_rows :]

        mixed_rest = remaining_single_rows + other_rows
        rng.shuffle(mixed_rest)
        final_rows = warmup_rows + mixed_rest

        cleaned_rows = [
            {
                key: value
                for key, value in row.items()
                if key != "_assignment_type_hidden"
            }
            for row in final_rows
        ]
        row_counts[annotator] = len(cleaned_rows)

        for part_index, block in enumerate(
            chunk_list(cleaned_rows, config.max_rows_per_annotator_file),
            start=1,
        ):
            output_path = output_dir / f"{annotator}_part{part_index:02d}.csv"
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(block)
            print(f"Wrote: {output_path}")

    return row_counts


def write_assignment_summary(
    rows: List[Dict[str, Any]],
    output_path: Path,
    config: AssignmentConfig,
) -> Dict[str, Dict[str, int]]:
    summary = {
        annotator: {"shared": 0, "double": 0, "single": 0, "total": 0}
        for annotator in config.annotators
    }

    for row in rows:
        assigned = str(row["assigned_annotators"]).split("|")
        assignment_type = str(row["assignment_type"])
        for annotator in assigned:
            summary[annotator][assignment_type] += 1
            summary[annotator]["total"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["annotator", "shared", "double", "single", "total"],
            delimiter="\t",
        )
        writer.writeheader()
        for annotator in config.annotators:
            writer.writerow({"annotator": annotator, **summary[annotator]})

    print(f"Wrote: {output_path}")
    return summary


def verify_outputs(
    rows: Sequence[Dict[str, Any]],
    outputs: OutputPaths,
    config: AssignmentConfig,
    annotator_row_counts: Dict[str, int],
    summary: Dict[str, Dict[str, int]],
) -> None:
    required_files = [
        outputs.annotation_sheet_path,
        outputs.annotator_summary_path,
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required output is missing or empty: {path}")

    with outputs.annotation_sheet_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        master_row_count = sum(1 for _ in csv.DictReader(handle))
    if master_row_count != len(rows):
        raise RuntimeError(
            "Master annotation-sheet row count does not match the assigned candidate count: "
            f"{master_row_count} versus {len(rows)}."
        )

    for annotator in config.annotators:
        expected_count = summary[annotator]["total"]
        written_count = annotator_row_counts.get(annotator, 0)
        if written_count != expected_count:
            raise RuntimeError(
                f"Annotator row-count mismatch for {annotator}: "
                f"expected {expected_count}, wrote {written_count}."
            )

        part_files = sorted(outputs.annotator_dir.glob(f"{annotator}_part*.csv"))
        if expected_count > 0 and not part_files:
            raise RuntimeError(f"No assignment files were written for {annotator}.")

        counted_rows = 0
        for part_file in part_files:
            with part_file.open("r", encoding="utf-8-sig", newline="") as handle:
                part_count = sum(1 for _ in csv.DictReader(handle))
            if part_count > config.max_rows_per_annotator_file:
                raise RuntimeError(
                    f"{part_file} contains {part_count} rows, exceeding the limit of "
                    f"{config.max_rows_per_annotator_file}."
                )
            counted_rows += part_count

        if counted_rows != expected_count:
            raise RuntimeError(
                f"Assignment files for {annotator} contain {counted_rows} rows, "
                f"but the summary reports {expected_count}."
            )


def run(args: SimpleNamespace) -> None:
    config = AssignmentConfig(
        annotators=tuple(annotator.strip() for annotator in args.annotators),
        shared_rows=args.shared_rows,
        double_rows_per_pair=args.double_rows_per_pair,
        warmup_single_rows=args.warmup_single_rows,
        max_rows_per_annotator_file=args.max_rows_per_annotator_file,
        random_seed=args.random_seed,
    )
    outputs = build_output_paths(args)

    validate_config(config)
    validate_paths(args.input_dir, outputs)

    rng = random.Random(config.random_seed)

    print(f"Loading corpus metadata from: {args.input_dir}")
    documents = load_corpus(args.input_dir)
    if not documents:
        raise ValueError("No usable case documents were loaded from the JSON input.")
    print(f"Loaded corpus documents: {len(documents):,}")

    print(f"\nReading candidate pool: {outputs.candidate_pool_path}")
    candidate_rows = read_candidate_pool(outputs.candidate_pool_path)
    print(f"Loaded candidate rows: {len(candidate_rows):,}")

    print("\nAssigning annotation roles...")
    assigned_rows = assign_annotation_roles(candidate_rows, config, rng)

    print("\nEnriching rows with query and candidate metadata...")
    enriched_rows = enrich_rows_with_case_metadata(assigned_rows, documents)

    print("\nWriting annotation outputs...")
    write_annotation_sheet(enriched_rows, outputs.annotation_sheet_path)
    annotator_row_counts = write_annotator_files(
        enriched_rows,
        outputs.annotator_dir,
        config,
        rng,
    )
    summary = write_assignment_summary(
        enriched_rows,
        outputs.annotator_summary_path,
        config,
    )

    verify_outputs(
        enriched_rows,
        outputs,
        config,
        annotator_row_counts,
        summary,
    )

    annotator_pair_count = math.comb(len(config.annotators), 2)
    unique_double_rows = annotator_pair_count * config.double_rows_per_pair
    single_rows = len(candidate_rows) - config.shared_rows - unique_double_rows

    print("\nAnnotator workload summary:")
    for annotator in config.annotators:
        stats = summary[annotator]
        print(
            f"{annotator}: shared={stats['shared']}, double={stats['double']}, "
            f"single={stats['single']}, total={stats['total']}"
        )

    random_breakdown = Counter(
        row["assignment_type"] for row in enriched_rows if is_random_row(row)
    )
    print("\nRandom-row allocation check:")
    print(
        {
            "shared_random_rows": random_breakdown["shared"],
            "double_random_rows": random_breakdown["double"],
            "single_random_rows": random_breakdown["single"],
        }
    )

    print("\nDone.")
    print(f"Master annotation sheet: {outputs.annotation_sheet_path}")
    print(f"Candidate pool rows: {len(candidate_rows):,}")
    print(f"Shared rows: {config.shared_rows:,}")
    print(f"Double rows per pair: {config.double_rows_per_pair:,}")
    print(f"Annotator pairs: {annotator_pair_count:,}")
    print(f"Unique double rows: {unique_double_rows:,}")
    print(f"Single rows: {single_rows:,}")
    print("Output verification: passed")




# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
INPUT_JSON_DIRECTORY = Path("data/enriched_json")
ANNOTATION_DIRECTORY = Path("data/annotations")
CANDIDATE_POOL_PATH = Path("data/shared/candidate_pool.tsv")

ANNOTATORS: tuple[str, ...] = tuple(
    f"annotator_{index:02d}" for index in range(1, 10)
)
SHARED_ROWS = 150
DOUBLE_ROWS_PER_PAIR = 10
WARMUP_SINGLE_ROWS = 150
MAX_ROWS_PER_ANNOTATOR_FILE = 100
RANDOM_SEED = 123


def build_ide_arguments(project_root: Path) -> SimpleNamespace:
    eval_dir = resolve_project_path(project_root, ANNOTATION_DIRECTORY)
    return SimpleNamespace(
        input_dir=resolve_project_path(project_root, INPUT_JSON_DIRECTORY),
        eval_dir=eval_dir,
        candidate_pool=resolve_project_path(project_root, CANDIDATE_POOL_PATH),
        annotation_sheet=eval_dir / "annotation_sheet_empty.csv",
        annotator_dir=eval_dir / "annotator_assignments",
        summary_path=eval_dir / "annotator_assignment_summary.tsv",
        annotators=list(ANNOTATORS),
        shared_rows=SHARED_ROWS,
        double_rows_per_pair=DOUBLE_ROWS_PER_PAIR,
        warmup_single_rows=WARMUP_SINGLE_ROWS,
        max_rows_per_annotator_file=MAX_ROWS_PER_ANNOTATOR_FILE,
        random_seed=RANDOM_SEED,
    )


def main() -> int:
    set_reproducible_seed(RANDOM_SEED)
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    args = build_ide_arguments(project_root)

    try:
        run(args)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
