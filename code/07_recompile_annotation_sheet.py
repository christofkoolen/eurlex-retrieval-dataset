#!/usr/bin/env python3
"""Recompile labeled annotator files into the master annotation sheet.

Edit the IDE configuration near the end of the file and run the complete
script. The accepted ordinal grades are 0, 1, 2, and 3.
"""

from __future__ import annotations


from types import SimpleNamespace
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


DEFAULT_ANNOTATORS: Tuple[str, ...] = tuple(
    f"annotator_{index:02d}" for index in range(1, 10)
)
ANSWER_FIELDS: Tuple[str, ...] = ("label_grade",)
REQUIRED_MASTER_COLUMNS: Tuple[str, ...] = (
    "row_id",
    "assignment_type",
    "assigned_annotators",
)
VALID_ASSIGNMENT_TYPES = {"single", "double", "shared"}
VALID_GRADES = {0, 1, 2, 3}

ANNOTATOR_FILE_RE = re.compile(
    r"^(annotator_\d{2})_part\d{2}.*_labeled(?:\.csv)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecompilePaths:
    master_sheet: Path
    annotator_dir: Path
    output_sheet: Path


@dataclass(frozen=True)
class RecompileConfig:
    annotators: Tuple[str, ...]
    allow_unknown_row_ids: bool = False


@dataclass
class RecompileStats:
    master_rows: int = 0
    annotator_files: int = 0
    rows_with_any_answer: int = 0
    unknown_answer_rows: int = 0
    answer_count_distribution: Counter[int] = None  # type: ignore[assignment]
    resolution_distribution: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.answer_count_distribution is None:
            self.answer_count_distribution = Counter()
        if self.resolution_distribution is None:
            self.resolution_distribution = Counter()


AnswerPayload = Dict[str, str]
AnswerBlock = Dict[str, AnswerPayload]
AnswerMap = Dict[str, AnswerBlock]


# -----------------------------------------------------------------------------
# CLI and configuration
# -----------------------------------------------------------------------------




def resolve_paths(args: SimpleNamespace) -> RecompilePaths:
    eval_dir: Path = args.eval_dir
    return RecompilePaths(
        master_sheet=args.master_sheet or eval_dir / "annotation_sheet_empty.csv",
        annotator_dir=args.annotator_dir or eval_dir / "annotator_assignments" / "labeled",
        output_sheet=args.output_sheet or eval_dir / "annotation_sheet_recompiled.csv",
    )


def validate_config(config: RecompileConfig) -> None:
    if not config.annotators:
        raise ValueError("At least one annotator must be configured.")

    normalized = [normalize_value(item).lower() for item in config.annotators]
    if any(not item for item in normalized):
        raise ValueError("Annotator IDs must not be blank.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Annotator IDs must be unique, ignoring case.")


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------


def set_csv_field_size_limit() -> None:
    """Raise the CSV field-size limit for files containing full case text."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def deduplicate_preserving_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def read_csv_rows(path: Path, encoding: str = "utf-8-sig") -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a CSV file, found: {path}")

    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ValueError(f"CSV file has no header: {path}")
    return rows, list(fieldnames)


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
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


# -----------------------------------------------------------------------------
# Grade handling
# -----------------------------------------------------------------------------


def parse_grade(value: object) -> Optional[int]:
    """Accept exact integral grades 0, 1, 2, or 3; return None otherwise."""
    text = normalize_value(value)
    if not text:
        return None

    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return None

    if decimal_value != decimal_value.to_integral_value():
        return None

    integer_value = int(decimal_value)
    return integer_value if integer_value in VALID_GRADES else None


def rounded_mean(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("Cannot calculate a mean from an empty grade list.")
    mean_value = Decimal(sum(values)) / Decimal(len(values))
    return int(mean_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def human_binary_from_grade(grade: Optional[int]) -> str:
    if grade in {0, 1}:
        return "0"
    if grade in {2, 3}:
        return "1"
    return ""


# -----------------------------------------------------------------------------
# Input discovery and validation
# -----------------------------------------------------------------------------


def normalize_annotators(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(normalize_value(value).lower() for value in values)


def read_master_annotation_sheet(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    rows, fieldnames = read_csv_rows(path)
    missing = [column for column in REQUIRED_MASTER_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(
            f"Master sheet is missing required column(s): {', '.join(missing)}"
        )
    if not rows:
        raise ValueError(f"Master sheet contains no rows: {path}")

    seen_row_ids = set()
    duplicate_row_ids = set()

    for line_number, row in enumerate(rows, start=2):
        row_id = normalize_value(row.get("row_id"))
        if not row_id:
            raise ValueError(f"Blank row_id in master sheet at CSV line {line_number}.")
        if row_id in seen_row_ids:
            duplicate_row_ids.add(row_id)
        seen_row_ids.add(row_id)

        assignment_type = normalize_value(row.get("assignment_type")).lower()
        if assignment_type not in VALID_ASSIGNMENT_TYPES:
            raise ValueError(
                f"Invalid assignment_type={assignment_type!r} for row_id={row_id}."
            )

    if duplicate_row_ids:
        examples = ", ".join(sorted(duplicate_row_ids)[:10])
        raise ValueError(f"Duplicate row_id values in master sheet: {examples}")

    return rows, fieldnames


def validate_master_assignments(
    rows: Sequence[Mapping[str, str]],
    annotators: Sequence[str],
) -> None:
    allowed = set(annotators)

    for row in rows:
        row_id = normalize_value(row.get("row_id"))
        assignment_type = normalize_value(row.get("assignment_type")).lower()
        assigned = [
            normalize_value(item).lower()
            for item in normalize_value(row.get("assigned_annotators")).split("|")
            if normalize_value(item)
        ]

        if len(set(assigned)) != len(assigned):
            raise ValueError(f"Duplicate assigned annotator for row_id={row_id}.")

        unknown = [annotator for annotator in assigned if annotator not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown assigned annotator(s) for row_id={row_id}: {', '.join(unknown)}"
            )

        expected_count = {"single": 1, "double": 2}.get(assignment_type)
        if expected_count is not None and len(assigned) != expected_count:
            raise ValueError(
                f"row_id={row_id} has assignment_type={assignment_type!r} but "
                f"{len(assigned)} assigned annotator(s); expected {expected_count}."
            )
        if assignment_type == "shared" and not assigned:
            raise ValueError(f"Shared row_id={row_id} has no assigned annotators.")


def find_annotator_files(
    folder: Path,
    annotators: Sequence[str],
) -> List[Tuple[str, Path]]:
    if not folder.exists():
        raise FileNotFoundError(f"Annotator directory not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected an annotator directory, found: {folder}")

    allowed = set(annotators)
    matches: List[Tuple[str, Path]] = []

    for path in sorted(folder.rglob("*.csv")):
        match = ANNOTATOR_FILE_RE.match(path.name)
        if not match:
            continue
        annotator = match.group(1).lower()
        if annotator in allowed:
            matches.append((annotator, path))

    if not matches:
        raise FileNotFoundError(
            f"No labeled annotator CSV files matching the expected naming pattern were found in {folder}"
        )
    return matches


def read_annotator_csv(path: Path) -> List[Dict[str, str]]:
    rows, fieldnames = read_csv_rows(path)
    missing = [column for column in ("row_id", "label_grade") if column not in fieldnames]
    if missing:
        raise ValueError(
            f"Annotator file {path.name} is missing column(s): {', '.join(missing)}"
        )

    return [
        {
            "row_id": normalize_value(row.get("row_id")),
            "label_grade": normalize_value(row.get("label_grade")),
        }
        for row in rows
    ]


# -----------------------------------------------------------------------------
# Answer-map construction and aggregation
# -----------------------------------------------------------------------------


def has_any_answer(payload: Mapping[str, str]) -> bool:
    return any(normalize_value(payload.get(field)) for field in ANSWER_FIELDS)


def build_answer_map(
    annotator_files: Sequence[Tuple[str, Path]],
    master_row_ids: set[str],
    allow_unknown_row_ids: bool,
) -> Tuple[AnswerMap, set[str]]:
    answers_by_row: MutableMapping[str, AnswerBlock] = defaultdict(dict)
    unknown_row_ids: set[str] = set()

    for annotator, path in annotator_files:
        print(f"Reading: {path}")
        for record in read_annotator_csv(path):
            row_id = normalize_value(record.get("row_id"))
            if not row_id:
                continue

            if row_id not in master_row_ids:
                unknown_row_ids.add(row_id)
                if allow_unknown_row_ids:
                    continue

            payload = {
                field: normalize_value(record.get(field))
                for field in ANSWER_FIELDS
            }

            existing = answers_by_row[row_id].get(annotator)
            if existing is None:
                answers_by_row[row_id][annotator] = payload
                continue

            existing_nonempty = has_any_answer(existing)
            incoming_nonempty = has_any_answer(payload)

            if not existing_nonempty and incoming_nonempty:
                answers_by_row[row_id][annotator] = payload
            elif existing_nonempty and not incoming_nonempty:
                continue
            elif existing != payload:
                raise ValueError(
                    "Conflicting duplicate answers for "
                    f"row_id={row_id}, annotator={annotator}."
                )

    if unknown_row_ids and not allow_unknown_row_ids:
        examples = ", ".join(sorted(unknown_row_ids)[:10])
        raise ValueError(
            f"Labeled files contain {len(unknown_row_ids)} row_id value(s) absent "
            f"from the master sheet. Examples: {examples}. Use "
            "Set ALLOW_UNKNOWN_ROW_IDS = True to ignore them."
        )

    return dict(answers_by_row), unknown_row_ids


def aggregate_human_grade(
    assignment_type: str,
    assigned_annotators: Sequence[str],
    answer_block: Mapping[str, Mapping[str, str]],
) -> Tuple[str, str, str]:
    """Return human_graded, resolution status, and resolution note."""
    valid_grade_pairs: List[Tuple[str, int]] = []
    invalid_grade_pairs: List[Tuple[str, str]] = []

    for annotator in assigned_annotators:
        raw_grade = normalize_value(
            answer_block.get(annotator, {}).get("label_grade", "")
        )
        if not raw_grade:
            continue

        parsed_grade = parse_grade(raw_grade)
        if parsed_grade is None:
            invalid_grade_pairs.append((annotator, raw_grade))
        else:
            valid_grade_pairs.append((annotator, parsed_grade))

    if invalid_grade_pairs:
        detail = "; ".join(
            f"{annotator}={grade!r}" for annotator, grade in invalid_grade_pairs
        )
        return "", "invalid_grade", f"Invalid assigned grade(s): {detail}."

    if not valid_grade_pairs:
        return "", "no_answer", "No valid human grade was provided."

    grades = [grade for _, grade in valid_grade_pairs]

    if assignment_type == "single":
        if len(valid_grade_pairs) != 1:
            return (
                "",
                "incomplete_single",
                f"Expected 1 assigned grade, found {len(valid_grade_pairs)}.",
            )
        return str(grades[0]), "resolved_single", ""

    if assignment_type == "double":
        if len(valid_grade_pairs) != 2:
            return (
                "",
                "incomplete_double",
                f"Expected 2 assigned grades, found {len(valid_grade_pairs)}.",
            )
        if grades[0] == grades[1]:
            return str(grades[0]), "resolved_double_agreement", ""

        disagreement = " and ".join(
            f"{annotator}={grade}" for annotator, grade in valid_grade_pairs
        )
        return (
            "",
            "needs_author_judgment",
            f"Double disagreement: {disagreement}.",
        )

    if assignment_type == "shared":
        average = rounded_mean(grades)
        supplied = len(valid_grade_pairs)
        expected = len(assigned_annotators)
        status = (
            "resolved_shared_average_complete"
            if supplied == expected
            else "resolved_shared_average_partial"
        )
        return (
            str(average),
            status,
            f"Rounded average over {supplied} of {expected} assigned grades.",
        )

    return "", "error", f"Unknown assignment_type: {assignment_type!r}."


def compile_rows(
    master_rows: Sequence[Mapping[str, str]],
    master_fieldnames: Sequence[str],
    answers_by_row: AnswerMap,
    config: RecompileConfig,
) -> Tuple[List[Dict[str, object]], List[str], RecompileStats]:
    annotator_answer_fields = [
        f"{annotator}_{field}"
        for annotator in config.annotators
        for field in ANSWER_FIELDS
    ]
    derived_fields = [
        "n_answers",
        "answered_by",
        "human_graded",
        "human_binary",
        "human_resolution_status",
        "human_resolution_note",
    ]
    output_fieldnames = deduplicate_preserving_order(
        [*master_fieldnames, *annotator_answer_fields, *derived_fields]
    )

    compiled_rows: List[Dict[str, object]] = []
    stats = RecompileStats(master_rows=len(master_rows))

    for master_row in master_rows:
        row_id = normalize_value(master_row.get("row_id"))
        assignment_type = normalize_value(master_row.get("assignment_type")).lower()
        assigned_annotators = [
            normalize_value(item).lower()
            for item in normalize_value(master_row.get("assigned_annotators")).split("|")
            if normalize_value(item)
        ]

        answer_block = answers_by_row.get(row_id, {})
        output_row: Dict[str, object] = dict(master_row)
        answered_by: List[str] = []

        for annotator in config.annotators:
            payload = answer_block.get(annotator, {})
            if has_any_answer(payload):
                answered_by.append(annotator)

            for field in ANSWER_FIELDS:
                output_row[f"{annotator}_{field}"] = normalize_value(
                    payload.get(field, "")
                )

        human_graded, resolution_status, resolution_note = aggregate_human_grade(
            assignment_type=assignment_type,
            assigned_annotators=assigned_annotators,
            answer_block=answer_block,
        )
        human_binary = human_binary_from_grade(parse_grade(human_graded))

        output_row["n_answers"] = len(answered_by)
        output_row["answered_by"] = "|".join(answered_by)
        output_row["human_graded"] = human_graded
        output_row["human_binary"] = human_binary
        output_row["human_resolution_status"] = resolution_status
        output_row["human_resolution_note"] = resolution_note

        if answered_by:
            stats.rows_with_any_answer += 1
        stats.answer_count_distribution[len(answered_by)] += 1
        stats.resolution_distribution[resolution_status] += 1
        compiled_rows.append(output_row)

    return compiled_rows, output_fieldnames, stats


# -----------------------------------------------------------------------------
# Reporting and execution
# -----------------------------------------------------------------------------


def print_summary(paths: RecompilePaths, stats: RecompileStats) -> None:
    print(f"\nWrote: {paths.output_sheet}")
    print(f"Rows in master sheet: {stats.master_rows:,}")
    print(f"Labeled annotator files read: {stats.annotator_files:,}")
    print(f"Rows with at least one answer: {stats.rows_with_any_answer:,}")
    if stats.unknown_answer_rows:
        print(f"Unknown answer row IDs ignored: {stats.unknown_answer_rows:,}")

    print("\nDistribution of number of answers per row:")
    for answer_count in sorted(stats.answer_count_distribution):
        print(
            f"{answer_count} answer(s): "
            f"{stats.answer_count_distribution[answer_count]:,}"
        )

    print("\nDistribution of human resolution status:")
    for status in sorted(stats.resolution_distribution):
        print(f"{status}: {stats.resolution_distribution[status]:,}")


def run(paths: RecompilePaths, config: RecompileConfig) -> RecompileStats:
    set_csv_field_size_limit()
    validate_config(config)

    print(f"Reading master annotation sheet: {paths.master_sheet}")
    master_rows, master_fieldnames = read_master_annotation_sheet(paths.master_sheet)
    validate_master_assignments(master_rows, config.annotators)
    print(f"Master rows: {len(master_rows):,}")

    annotator_files = find_annotator_files(paths.annotator_dir, config.annotators)
    print(f"Found {len(annotator_files):,} labeled annotator file(s).")

    master_row_ids = {
        normalize_value(row.get("row_id"))
        for row in master_rows
    }
    answers_by_row, unknown_row_ids = build_answer_map(
        annotator_files=annotator_files,
        master_row_ids=master_row_ids,
        allow_unknown_row_ids=config.allow_unknown_row_ids,
    )

    compiled_rows, output_fieldnames, stats = compile_rows(
        master_rows=master_rows,
        master_fieldnames=master_fieldnames,
        answers_by_row=answers_by_row,
        config=config,
    )
    stats.annotator_files = len(annotator_files)
    stats.unknown_answer_rows = len(unknown_row_ids)

    write_csv_atomic(paths.output_sheet, output_fieldnames, compiled_rows)
    print_summary(paths, stats)
    return stats

# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
ANNOTATION_DIRECTORY = Path("data/annotations")
ANNOTATORS: tuple[str, ...] = tuple(
    f"annotator_{index:02d}" for index in range(1, 10)
)
ALLOW_UNKNOWN_ROW_IDS = False


def main() -> int:
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    eval_dir = resolve_project_path(project_root, ANNOTATION_DIRECTORY)
    args = SimpleNamespace(
        eval_dir=eval_dir,
        master_sheet=eval_dir / "annotation_sheet_empty.csv",
        annotator_dir=eval_dir / "annotator_assignments" / "labeled",
        output_sheet=eval_dir / "annotation_sheet_recompiled.csv",
    )
    paths = resolve_paths(args)
    config = RecompileConfig(
        annotators=normalize_annotators(ANNOTATORS),
        allow_unknown_row_ids=ALLOW_UNKNOWN_ROW_IDS,
    )

    try:
        run(paths, config)
        return 0
    except (FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
