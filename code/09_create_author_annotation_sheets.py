#!/usr/bin/env python3
"""Create adjudication sheets for unresolved annotation disagreements.

Edit the IDE configuration near the end of the file and run the complete
script. Output files contain no names or direct personal identifiers.
"""


from types import SimpleNamespace
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


ECLI_RE = re.compile(
    r"\b(?:ECLI:)?(EU:(?:C|T|F):\d{4}:\d{1,6})\b",
    re.IGNORECASE,
)
ANNOTATOR_GRADE_RE = re.compile(r"^annotator_\d+_label_grade$", re.IGNORECASE)
DEFAULT_STATUSES: Tuple[str, ...] = ("needs_author_judgment",)
AUTHOR_FIELDS: Tuple[str, ...] = ("author_label_grade", "author_notes")
REQUIRED_RECOMPILED_COLUMNS: Tuple[str, ...] = (
    "row_id",
    "query_id",
    "candidate_id",
    "assignment_type",
    "assigned_annotators",
    "human_resolution_status",
    "human_resolution_note",
)


@dataclass(frozen=True)
class AuthorPaths:
    recompiled_sheet: Path
    master_output: Path
    author_dir: Path
    summary_output: Path


@dataclass(frozen=True)
class AuthorConfig:
    statuses: Tuple[str, ...]
    max_rows_per_file: int = 100
    allow_missing_documents: bool = False


@dataclass
class AuthorStats:
    input_rows: int = 0
    selected_rows: int = 0
    part_files: int = 0
    missing_query_documents: int = 0
    missing_candidate_documents: int = 0


# -----------------------------------------------------------------------------
# CLI and path handling
# -----------------------------------------------------------------------------




def resolve_paths(args: SimpleNamespace) -> AuthorPaths:
    eval_dir: Path = args.eval_dir
    return AuthorPaths(
        recompiled_sheet=(
            args.recompiled_sheet or eval_dir / "annotation_sheet_recompiled.csv"
        ),
        master_output=(
            args.master_output or eval_dir / "author_annotation_sheet_empty.csv"
        ),
        author_dir=args.author_dir or eval_dir / "author_assignments",
        summary_output=(
            args.summary_output or eval_dir / "author_annotation_summary.tsv"
        ),
    )


def normalize_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_statuses(values: Sequence[str]) -> Tuple[str, ...]:
    statuses = tuple(normalize_value(value).lower() for value in values)
    if not statuses or any(not status for status in statuses):
        raise ValueError("At least one non-empty resolution status must be supplied.")
    if len(set(statuses)) != len(statuses):
        raise ValueError("Resolution statuses must be unique, ignoring case.")
    return statuses


def validate_config(config: AuthorConfig) -> None:
    if config.max_rows_per_file <= 0:
        raise ValueError("MAX_ROWS_PER_FILE must be greater than zero.")


def validate_paths(input_path: Path, paths: AuthorPaths) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"JSON input does not exist: {input_path}")
    if input_path.is_file() and input_path.suffix.lower() != ".json":
        raise ValueError(f"JSON input file must end in .json: {input_path}")
    if not paths.recompiled_sheet.exists():
        raise FileNotFoundError(
            f"Recompiled annotation sheet not found: {paths.recompiled_sheet}"
        )
    if not paths.recompiled_sheet.is_file():
        raise ValueError(
            f"Recompiled annotation sheet is not a file: {paths.recompiled_sheet}"
        )


# -----------------------------------------------------------------------------
# JSON corpus loading
# -----------------------------------------------------------------------------


def iter_json_files(root: Path) -> Iterator[Path]:
    if root.is_file():
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


def get_summary_text(data: Mapping[str, Any]) -> str:
    case_summary = data.get("case_summary")
    if not isinstance(case_summary, dict):
        return ""

    summary_text = case_summary.get("summary_text")
    return summary_text.strip() if isinstance(summary_text, str) else ""


def get_full_text(data: Mapping[str, Any]) -> str:
    case_segments = data.get("case_segments")
    if not isinstance(case_segments, dict):
        return ""

    segments: List[str] = []
    for segment_value in case_segments.values():
        segment_text = "\n\n".join(collect_text(segment_value)).strip()
        if segment_text:
            segments.append(segment_text)
    return "\n\n".join(segments).strip()


def load_corpus(input_path: Path) -> Dict[str, Dict[str, str]]:
    documents: Dict[str, Dict[str, str]] = {}
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
        document_id = choose_doc_id(metadata)
        if not document_id or not isinstance(metadata, dict):
            continue

        if document_id in documents:
            duplicate_ids[document_id] += 1

        documents[document_id] = {
            "title": normalize_value(
                metadata.get("case_name")
                or metadata.get("case_number")
                or document_id
            ),
            "date": normalize_value(metadata.get("date_of_judgment")),
            "link": normalize_value(metadata.get("document_link")),
            "summary": get_summary_text(data),
            "full": get_full_text(data),
        }

    if failed_files:
        print(f"Skipped unreadable or invalid JSON files: {failed_files}")
    if duplicate_ids:
        print(
            "Warning: duplicate document IDs were found; the last file for each "
            "ID was used: " + ", ".join(sorted(duplicate_ids))
        )

    return documents


# -----------------------------------------------------------------------------
# CSV reading, filtering, and enrichment
# -----------------------------------------------------------------------------


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ValueError(f"CSV file has no header: {path}")
    return rows, fieldnames


def validate_recompiled_sheet(
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> None:
    missing = [
        column for column in REQUIRED_RECOMPILED_COLUMNS if column not in fieldnames
    ]
    if missing:
        raise ValueError(
            "Recompiled sheet is missing required column(s): " + ", ".join(missing)
        )

    seen_row_ids = set()
    duplicates = set()
    for line_number, row in enumerate(rows, start=2):
        row_id = normalize_value(row.get("row_id"))
        if not row_id:
            raise ValueError(
                f"Recompiled sheet contains a blank row_id at CSV line {line_number}."
            )
        if row_id in seen_row_ids:
            duplicates.add(row_id)
        seen_row_ids.add(row_id)

    if duplicates:
        examples = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"Duplicate row_id values in recompiled sheet: {examples}")


def filter_rows(
    rows: Sequence[Mapping[str, str]],
    statuses: Sequence[str],
) -> List[Dict[str, str]]:
    allowed = set(statuses)
    return [
        dict(row)
        for row in rows
        if normalize_value(row.get("human_resolution_status")).lower() in allowed
    ]


def assigned_grade_summary(row: Mapping[str, str]) -> str:
    assigned = [
        normalize_value(item).lower()
        for item in normalize_value(row.get("assigned_annotators")).split("|")
        if normalize_value(item)
    ]

    pairs: List[str] = []
    for annotator in assigned:
        grade = normalize_value(row.get(f"{annotator}_label_grade"))
        pairs.append(f"{annotator}={grade or '[blank]'}")
    return " | ".join(pairs)


def corpus_value(
    document: Optional[Mapping[str, str]],
    key: str,
    fallback: object,
) -> str:
    if document is not None:
        value = normalize_value(document.get(key))
        if value:
            return value
    return normalize_value(fallback)


def enrich_selected_rows(
    selected_rows: Sequence[Mapping[str, str]],
    documents: Mapping[str, Mapping[str, str]],
    allow_missing_documents: bool,
) -> Tuple[List[Dict[str, str]], int, int]:
    enriched: List[Dict[str, str]] = []
    missing_queries: List[str] = []
    missing_candidates: List[str] = []

    for row in selected_rows:
        query_id = normalize_value(row.get("query_id"))
        candidate_id = normalize_value(row.get("candidate_id"))
        query_document = documents.get(query_id)
        candidate_document = documents.get(candidate_id)

        if query_document is None:
            missing_queries.append(query_id)
        if candidate_document is None:
            missing_candidates.append(candidate_id)

        output = {key: normalize_value(value) for key, value in row.items()}
        output["query_title"] = corpus_value(
            query_document, "title", row.get("query_title")
        )
        output["query_summary"] = corpus_value(query_document, "summary", "")
        output["query_full"] = corpus_value(query_document, "full", "")
        output["query_link"] = corpus_value(
            query_document, "link", row.get("query_link")
        )
        output["query_date"] = corpus_value(
            query_document, "date", row.get("query_date")
        )

        output["candidate_title"] = corpus_value(
            candidate_document, "title", row.get("candidate_title")
        )
        output["candidate_summary"] = corpus_value(
            candidate_document, "summary", ""
        )
        output["candidate_full"] = corpus_value(candidate_document, "full", "")
        output["candidate_link"] = corpus_value(
            candidate_document, "link", row.get("candidate_link")
        )
        output["candidate_date"] = corpus_value(
            candidate_document, "date", row.get("candidate_date")
        )

        output["assigned_grade_summary"] = assigned_grade_summary(row)
        output["author_label_grade"] = ""
        output["author_notes"] = ""
        enriched.append(output)

    if (missing_queries or missing_candidates) and not allow_missing_documents:
        messages: List[str] = []
        if missing_queries:
            examples = ", ".join(sorted(set(missing_queries))[:10])
            messages.append(
                f"{len(missing_queries)} selected row(s) have query IDs absent from "
                f"the JSON corpus. Examples: {examples}"
            )
        if missing_candidates:
            examples = ", ".join(sorted(set(missing_candidates))[:10])
            messages.append(
                f"{len(missing_candidates)} selected row(s) have candidate IDs absent "
                f"from the JSON corpus. Examples: {examples}"
            )
        raise ValueError(
            "; ".join(messages)
            + ". Set ALLOW_MISSING_DOCUMENTS = True to continue with blank text fields."
        )

    return enriched, len(missing_queries), len(missing_candidates)


# -----------------------------------------------------------------------------
# Output construction
# -----------------------------------------------------------------------------


def deduplicate_preserving_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def score_fields(fieldnames: Sequence[str]) -> List[str]:
    return sorted(
        fieldname for fieldname in fieldnames if fieldname.startswith("score_")
    )


def annotator_grade_fields(fieldnames: Sequence[str]) -> List[str]:
    return sorted(
        fieldname for fieldname in fieldnames if ANNOTATOR_GRADE_RE.match(fieldname)
    )


def build_master_fieldnames(source_fieldnames: Sequence[str]) -> List[str]:
    preferred = [
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
    ]
    return deduplicate_preserving_order(
        [
            *preferred,
            *source_fieldnames,
            "assigned_grade_summary",
            *AUTHOR_FIELDS,
        ]
    )


def build_author_file_fieldnames(source_fieldnames: Sequence[str]) -> List[str]:
    context_fields = [
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
        "source_primary",
        "source_all",
        *score_fields(source_fieldnames),
        "assignment_type",
        "assigned_annotators",
        "pair_id",
        "assigned_grade_summary",
        *annotator_grade_fields(source_fieldnames),
        "n_answers",
        "answered_by",
        "human_graded",
        "human_binary",
        "human_resolution_status",
        "human_resolution_note",
        *AUTHOR_FIELDS,
    ]
    return deduplicate_preserving_order(context_fields)


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def remove_stale_author_parts(author_dir: Path) -> None:
    if not author_dir.exists():
        return
    for path in author_dir.glob("author_part*.csv"):
        if path.is_file():
            path.unlink()


def chunk_rows(
    rows: Sequence[Dict[str, str]],
    chunk_size: int,
) -> List[List[Dict[str, str]]]:
    return [
        list(rows[index : index + chunk_size])
        for index in range(0, len(rows), chunk_size)
    ]


def write_author_parts(
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
    author_dir: Path,
    max_rows_per_file: int,
) -> List[Tuple[Path, List[Dict[str, str]]]]:
    author_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_author_parts(author_dir)

    written: List[Tuple[Path, List[Dict[str, str]]]] = []
    for part_index, block in enumerate(
        chunk_rows(rows, max_rows_per_file),
        start=1,
    ):
        output_path = author_dir / f"author_part{part_index:02d}.csv"
        write_csv_atomic(output_path, fieldnames, block)
        written.append((output_path, block))
        print(f"Wrote: {output_path}")
    return written


def disagreement_key(row: Mapping[str, str]) -> str:
    assigned = [
        normalize_value(item).lower()
        for item in normalize_value(row.get("assigned_annotators")).split("|")
        if normalize_value(item)
    ]
    grades = [
        normalize_value(row.get(f"{annotator}_label_grade")) or "[blank]"
        for annotator in assigned
    ]
    return " vs ".join(grades) if grades else "[no assigned grades]"


def write_summary(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    written_parts: Sequence[Tuple[Path, Sequence[Mapping[str, str]]]],
) -> None:
    status_counts = Counter(
        normalize_value(row.get("human_resolution_status")) for row in rows
    )
    pair_counts = Counter(
        normalize_value(row.get("assigned_annotators")) or "[blank]" for row in rows
    )
    disagreement_counts = Counter(disagreement_key(row) for row in rows)

    summary_rows: List[Dict[str, object]] = [
        {"section": "overview", "key": "total_rows", "count": len(rows)}
    ]
    summary_rows.extend(
        {
            "section": "resolution_status",
            "key": key,
            "count": count,
        }
        for key, count in sorted(status_counts.items())
    )
    summary_rows.extend(
        {
            "section": "annotator_pair",
            "key": key,
            "count": count,
        }
        for key, count in sorted(pair_counts.items())
    )
    summary_rows.extend(
        {
            "section": "grade_disagreement",
            "key": key,
            "count": count,
        }
        for key, count in sorted(disagreement_counts.items())
    )
    summary_rows.extend(
        {
            "section": "output_part",
            "key": output_path.name,
            "count": len(block),
        }
        for output_path, block in written_parts
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["section", "key", "count"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Wrote: {path}")


def verify_outputs(
    paths: AuthorPaths,
    selected_rows: Sequence[Mapping[str, str]],
    written_parts: Sequence[Tuple[Path, Sequence[Mapping[str, str]]]],
    max_rows_per_file: int,
) -> None:
    for required_path in (paths.master_output, paths.summary_output):
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise RuntimeError(f"Required output is missing or empty: {required_path}")

    with paths.master_output.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        master_count = sum(1 for _ in csv.DictReader(handle))
    if master_count != len(selected_rows):
        raise RuntimeError(
            "Author master-sheet row count mismatch: "
            f"expected {len(selected_rows)}, found {master_count}."
        )

    part_count = 0
    for part_path, block in written_parts:
        if len(block) > max_rows_per_file:
            raise RuntimeError(
                f"{part_path} contains {len(block)} rows, exceeding the limit of "
                f"{max_rows_per_file}."
            )
        with part_path.open("r", encoding="utf-8-sig", newline="") as handle:
            actual_count = sum(1 for _ in csv.DictReader(handle))
        if actual_count != len(block):
            raise RuntimeError(
                f"Row-count mismatch for {part_path}: expected {len(block)}, "
                f"found {actual_count}."
            )
        part_count += actual_count

    if part_count != len(selected_rows):
        raise RuntimeError(
            "Combined author-part row count mismatch: "
            f"expected {len(selected_rows)}, found {part_count}."
        )


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------


def print_summary(paths: AuthorPaths, stats: AuthorStats) -> None:
    print("\nAdjudication sheet summary:")
    print(f"Rows in recompiled sheet: {stats.input_rows:,}")
    print(f"Rows selected for adjudication: {stats.selected_rows:,}")
    print(f"Author part files written: {stats.part_files:,}")
    if stats.missing_query_documents:
        print(f"Missing query documents: {stats.missing_query_documents:,}")
    if stats.missing_candidate_documents:
        print(f"Missing candidate documents: {stats.missing_candidate_documents:,}")
    print(f"Combined author sheet: {paths.master_output}")
    print(f"Adjudication files: {paths.author_dir}")
    print(f"Summary: {paths.summary_output}")


def run(
    input_path: Path,
    paths: AuthorPaths,
    config: AuthorConfig,
) -> AuthorStats:
    set_csv_field_size_limit()
    validate_config(config)
    validate_paths(input_path, paths)

    print(f"Reading recompiled annotation sheet: {paths.recompiled_sheet}")
    source_rows, source_fieldnames = read_csv_rows(paths.recompiled_sheet)
    validate_recompiled_sheet(source_rows, source_fieldnames)

    selected_rows = filter_rows(source_rows, config.statuses)
    if not selected_rows:
        raise ValueError(
            "No rows matched the requested human_resolution_status value(s): "
            + ", ".join(config.statuses)
        )
    print(f"Selected unresolved rows: {len(selected_rows):,}")

    print(f"Loading case summaries and full text from: {input_path}")
    documents = load_corpus(input_path)
    if not documents:
        raise ValueError("No usable case documents were loaded from the JSON input.")
    print(f"Loaded corpus documents: {len(documents):,}")

    enriched_rows, missing_queries, missing_candidates = enrich_selected_rows(
        selected_rows=selected_rows,
        documents=documents,
        allow_missing_documents=config.allow_missing_documents,
    )

    master_fieldnames = build_master_fieldnames(source_fieldnames)
    author_fieldnames = build_author_file_fieldnames(source_fieldnames)

    write_csv_atomic(paths.master_output, master_fieldnames, enriched_rows)
    print(f"Wrote: {paths.master_output}")

    written_parts = write_author_parts(
        rows=enriched_rows,
        fieldnames=author_fieldnames,
        author_dir=paths.author_dir,
        max_rows_per_file=config.max_rows_per_file,
    )
    write_summary(paths.summary_output, enriched_rows, written_parts)
    verify_outputs(
        paths=paths,
        selected_rows=enriched_rows,
        written_parts=written_parts,
        max_rows_per_file=config.max_rows_per_file,
    )

    stats = AuthorStats(
        input_rows=len(source_rows),
        selected_rows=len(enriched_rows),
        part_files=len(written_parts),
        missing_query_documents=missing_queries,
        missing_candidate_documents=missing_candidates,
    )
    print_summary(paths, stats)
    return stats

# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
INPUT_JSON_DIRECTORY = Path("data/enriched_json")
ANNOTATION_DIRECTORY = Path("data/annotations")
STATUSES: tuple[str, ...] = ("needs_author_judgment",)
MAX_ROWS_PER_FILE = 100
ALLOW_MISSING_DOCUMENTS = False


def main() -> int:
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    input_dir = resolve_project_path(project_root, INPUT_JSON_DIRECTORY)
    eval_dir = resolve_project_path(project_root, ANNOTATION_DIRECTORY)
    args = SimpleNamespace(
        eval_dir=eval_dir,
        recompiled_sheet=eval_dir / "annotation_sheet_recompiled.csv",
        master_output=eval_dir / "adjudication_sheet_empty.csv",
        author_dir=eval_dir / "adjudication_assignments",
        summary_output=eval_dir / "adjudication_summary.tsv",
    )
    paths = resolve_paths(args)
    config = AuthorConfig(
        statuses=normalize_statuses(STATUSES),
        max_rows_per_file=MAX_ROWS_PER_FILE,
        allow_missing_documents=ALLOW_MISSING_DOCUMENTS,
    )

    try:
        run(input_dir, paths, config)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
