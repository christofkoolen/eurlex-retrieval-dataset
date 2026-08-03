#!/usr/bin/env python3
"""Combine annotator labels and adjudication decisions into final labels.

Edit the IDE configuration near the end of the file and run the complete
script. Internal author_* column names are retained for compatibility with
the existing annotation schema.
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
AUTHOR_FIELDS: Tuple[str, ...] = (
    "author_label_grade",
    "author_notes",
    "author_source_file",
)
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
AUTHOR_FILE_RE = re.compile(
    r"^author_part\d+.*\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecompilePaths:
    master_sheet: Path
    annotator_dir: Path
    author_dir: Path
    explicit_author_sheets: Tuple[Path, ...]
    output_sheet: Path


@dataclass(frozen=True)
class RecompileConfig:
    annotators: Tuple[str, ...]
    allow_unknown_row_ids: bool = False
    allow_author_overrides: bool = False
    require_complete_author_judgments: bool = False


@dataclass
class RecompileStats:
    master_rows: int = 0
    annotator_files: int = 0
    author_files: int = 0
    rows_with_any_annotator_answer: int = 0
    rows_with_author_grade: int = 0
    rows_resolved_by_author: int = 0
    unknown_annotator_row_ids: int = 0
    unknown_author_row_ids: int = 0
    answer_count_distribution: Counter[int] = None  # type: ignore[assignment]
    annotator_resolution_distribution: Counter[str] = None  # type: ignore[assignment]
    final_resolution_distribution: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.answer_count_distribution is None:
            self.answer_count_distribution = Counter()
        if self.annotator_resolution_distribution is None:
            self.annotator_resolution_distribution = Counter()
        if self.final_resolution_distribution is None:
            self.final_resolution_distribution = Counter()


AnswerPayload = Dict[str, str]
AnswerBlock = Dict[str, AnswerPayload]
AnswerMap = Dict[str, AnswerBlock]
AuthorPayload = Dict[str, str]
AuthorMap = Dict[str, AuthorPayload]


# -----------------------------------------------------------------------------
# CLI and configuration
# -----------------------------------------------------------------------------




def resolve_paths(args: SimpleNamespace) -> RecompilePaths:
    eval_dir: Path = args.eval_dir
    default_author_root = eval_dir / "author_assignments"
    default_author_labeled = default_author_root / "labeled"
    default_author_dir = (
        default_author_labeled if default_author_labeled.is_dir() else default_author_root
    )

    return RecompilePaths(
        master_sheet=args.master_sheet or eval_dir / "annotation_sheet_empty.csv",
        annotator_dir=(
            args.annotator_dir or eval_dir / "annotator_assignments" / "labeled"
        ),
        author_dir=args.author_dir or default_author_dir,
        explicit_author_sheets=tuple(args.author_sheet or ()),
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


def read_csv_rows(
    path: Path,
    encoding: str = "utf-8-sig",
) -> Tuple[List[Dict[str, str]], List[str]]:
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


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
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
            "No labeled annotator CSV files matching the expected naming pattern "
            f"were found in {folder}"
        )
    return matches


def find_author_files(paths: RecompilePaths) -> List[Path]:
    if paths.explicit_author_sheets:
        files = list(paths.explicit_author_sheets)
        for path in files:
            if not path.exists():
                raise FileNotFoundError(f"Author sheet not found: {path}")
            if not path.is_file():
                raise ValueError(f"Expected an author CSV file, found: {path}")
        return files

    folder = paths.author_dir
    if not folder.exists():
        raise FileNotFoundError(f"Author directory not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected an author directory, found: {folder}")

    matches = [
        path
        for path in sorted(folder.rglob("*.csv"))
        if AUTHOR_FILE_RE.match(path.name)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No author_part*.csv files were found in {folder}. Supply explicit "
            "files through EXPLICIT_ADJUDICATION_SHEETS when using a different filename."
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


def read_author_csv(path: Path) -> List[Dict[str, str]]:
    rows, fieldnames = read_csv_rows(path)
    required = ("row_id", "author_label_grade")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(
            f"Author file {path.name} is missing column(s): {', '.join(missing)}"
        )

    return [
        {
            "row_id": normalize_value(row.get("row_id")),
            "author_label_grade": normalize_value(row.get("author_label_grade")),
            "author_notes": normalize_value(row.get("author_notes")),
            "author_source_file": path.name,
        }
        for row in rows
    ]


# -----------------------------------------------------------------------------
# Answer-map construction
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
        print(f"Reading annotator file: {path}")
        for record in read_annotator_csv(path):
            row_id = normalize_value(record.get("row_id"))
            if not row_id:
                continue

            if row_id not in master_row_ids:
                unknown_row_ids.add(row_id)
                if allow_unknown_row_ids:
                    continue

            payload = {
                field: normalize_value(record.get(field)) for field in ANSWER_FIELDS
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
                    "Conflicting duplicate annotator answers for "
                    f"row_id={row_id}, annotator={annotator}."
                )

    if unknown_row_ids and not allow_unknown_row_ids:
        examples = ", ".join(sorted(unknown_row_ids)[:10])
        raise ValueError(
            f"Annotator files contain {len(unknown_row_ids)} row_id value(s) absent "
            f"from the master sheet. Examples: {examples}. Use "
            "Set ALLOW_UNKNOWN_ROW_IDS = True to ignore them."
        )

    return dict(answers_by_row), unknown_row_ids


def merge_text_values(left: str, right: str) -> str:
    values = deduplicate_preserving_order(
        value for value in (normalize_value(left), normalize_value(right)) if value
    )
    return " || ".join(values)


def build_author_map(
    author_files: Sequence[Path],
    master_row_ids: set[str],
    allow_unknown_row_ids: bool,
) -> Tuple[AuthorMap, set[str]]:
    answers: AuthorMap = {}
    unknown_row_ids: set[str] = set()

    for path in author_files:
        print(f"Reading author file: {path}")
        for record in read_author_csv(path):
            row_id = normalize_value(record.get("row_id"))
            if not row_id:
                continue

            if row_id not in master_row_ids:
                unknown_row_ids.add(row_id)
                if allow_unknown_row_ids:
                    continue

            incoming = {
                field: normalize_value(record.get(field)) for field in AUTHOR_FIELDS
            }
            existing = answers.get(row_id)
            if existing is None:
                answers[row_id] = incoming
                continue

            existing_grade = normalize_value(existing.get("author_label_grade"))
            incoming_grade = normalize_value(incoming.get("author_label_grade"))
            if existing_grade and incoming_grade and existing_grade != incoming_grade:
                raise ValueError(
                    "Conflicting duplicate author grades for "
                    f"row_id={row_id}: {existing_grade!r} versus {incoming_grade!r}."
                )

            answers[row_id] = {
                "author_label_grade": existing_grade or incoming_grade,
                "author_notes": merge_text_values(
                    existing.get("author_notes", ""),
                    incoming.get("author_notes", ""),
                ),
                "author_source_file": merge_text_values(
                    existing.get("author_source_file", ""),
                    incoming.get("author_source_file", ""),
                ),
            }

    if unknown_row_ids and not allow_unknown_row_ids:
        examples = ", ".join(sorted(unknown_row_ids)[:10])
        raise ValueError(
            f"Author files contain {len(unknown_row_ids)} row_id value(s) absent "
            f"from the master sheet. Examples: {examples}. Use "
            "Set ALLOW_UNKNOWN_ROW_IDS = True to ignore them."
        )

    return answers, unknown_row_ids


# -----------------------------------------------------------------------------
# Annotator aggregation and author adjudication
# -----------------------------------------------------------------------------


def aggregate_annotator_grade(
    assignment_type: str,
    assigned_annotators: Sequence[str],
    answer_block: Mapping[str, Mapping[str, str]],
) -> Tuple[str, str, str]:
    """Return annotator grade, annotator resolution status, and note."""
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


def build_author_resolution_note(
    author_grade: int,
    annotator_status: str,
    annotator_note: str,
    author_notes: str,
    override: bool,
) -> str:
    action = "Author override" if override else "Author adjudication"
    components = [f"{action} selected grade {author_grade}."]
    if annotator_note:
        components.append(f"Annotator result: {annotator_note}")
    else:
        components.append(f"Annotator status: {annotator_status}.")
    if author_notes:
        components.append(f"Author note: {author_notes}")
    return " ".join(components)


def apply_author_adjudication(
    row_id: str,
    annotator_grade: str,
    annotator_status: str,
    annotator_note: str,
    author_payload: Mapping[str, str],
    allow_author_overrides: bool,
) -> Tuple[str, str, str, str, bool]:
    """Return final grade, binary, status, note, and author-resolution flag."""
    raw_author_grade = normalize_value(author_payload.get("author_label_grade"))
    author_notes = normalize_value(author_payload.get("author_notes"))

    if not raw_author_grade:
        final_grade = annotator_grade
        return (
            final_grade,
            human_binary_from_grade(parse_grade(final_grade)),
            annotator_status,
            annotator_note,
            False,
        )

    parsed_author_grade = parse_grade(raw_author_grade)
    if parsed_author_grade is None:
        note = f"Invalid author grade for row_id={row_id}: {raw_author_grade!r}."
        if author_notes:
            note += f" Author note: {author_notes}"
        return "", "", "invalid_author_grade", note, False

    if annotator_status == "needs_author_judgment":
        return (
            str(parsed_author_grade),
            human_binary_from_grade(parsed_author_grade),
            "resolved_author_judgment",
            build_author_resolution_note(
                author_grade=parsed_author_grade,
                annotator_status=annotator_status,
                annotator_note=annotator_note,
                author_notes=author_notes,
                override=False,
            ),
            True,
        )

    if not allow_author_overrides:
        raise ValueError(
            f"row_id={row_id} has author grade {parsed_author_grade}, but its "
            f"annotator status is {annotator_status!r}, not "
            "'needs_author_judgment'. Set ALLOW_ADJUDICATION_OVERRIDES = True to permit this."
        )

    return (
        str(parsed_author_grade),
        human_binary_from_grade(parsed_author_grade),
        "resolved_author_override",
        build_author_resolution_note(
            author_grade=parsed_author_grade,
            annotator_status=annotator_status,
            annotator_note=annotator_note,
            author_notes=author_notes,
            override=True,
        ),
        True,
    )


def compile_rows(
    master_rows: Sequence[Mapping[str, str]],
    master_fieldnames: Sequence[str],
    answers_by_row: AnswerMap,
    author_answers_by_row: AuthorMap,
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
        "annotator_human_graded",
        "annotator_human_binary",
        "annotator_resolution_status",
        "annotator_resolution_note",
        *AUTHOR_FIELDS,
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
    missing_author_judgments: List[str] = []

    for master_row in master_rows:
        row_id = normalize_value(master_row.get("row_id"))
        assignment_type = normalize_value(master_row.get("assignment_type")).lower()
        assigned_annotators = [
            normalize_value(item).lower()
            for item in normalize_value(master_row.get("assigned_annotators")).split("|")
            if normalize_value(item)
        ]

        answer_block = answers_by_row.get(row_id, {})
        author_payload = author_answers_by_row.get(row_id, {})
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

        (
            annotator_grade,
            annotator_resolution_status,
            annotator_resolution_note,
        ) = aggregate_annotator_grade(
            assignment_type=assignment_type,
            assigned_annotators=assigned_annotators,
            answer_block=answer_block,
        )
        annotator_binary = human_binary_from_grade(parse_grade(annotator_grade))

        for field in AUTHOR_FIELDS:
            output_row[field] = normalize_value(author_payload.get(field, ""))

        (
            final_grade,
            final_binary,
            final_resolution_status,
            final_resolution_note,
            resolved_by_author,
        ) = apply_author_adjudication(
            row_id=row_id,
            annotator_grade=annotator_grade,
            annotator_status=annotator_resolution_status,
            annotator_note=annotator_resolution_note,
            author_payload=author_payload,
            allow_author_overrides=config.allow_author_overrides,
        )

        if (
            annotator_resolution_status == "needs_author_judgment"
            and parse_grade(author_payload.get("author_label_grade")) is None
        ):
            missing_author_judgments.append(row_id)

        output_row["n_answers"] = len(answered_by)
        output_row["answered_by"] = "|".join(answered_by)
        output_row["annotator_human_graded"] = annotator_grade
        output_row["annotator_human_binary"] = annotator_binary
        output_row["annotator_resolution_status"] = annotator_resolution_status
        output_row["annotator_resolution_note"] = annotator_resolution_note
        output_row["human_graded"] = final_grade
        output_row["human_binary"] = final_binary
        output_row["human_resolution_status"] = final_resolution_status
        output_row["human_resolution_note"] = final_resolution_note

        if answered_by:
            stats.rows_with_any_annotator_answer += 1
        if normalize_value(author_payload.get("author_label_grade")):
            stats.rows_with_author_grade += 1
        if resolved_by_author:
            stats.rows_resolved_by_author += 1

        stats.answer_count_distribution[len(answered_by)] += 1
        stats.annotator_resolution_distribution[annotator_resolution_status] += 1
        stats.final_resolution_distribution[final_resolution_status] += 1
        compiled_rows.append(output_row)

    if missing_author_judgments and config.require_complete_author_judgments:
        examples = ", ".join(missing_author_judgments[:10])
        raise ValueError(
            f"{len(missing_author_judgments)} row(s) still require a valid author "
            f"grade. Examples: {examples}."
        )

    return compiled_rows, output_fieldnames, stats


# -----------------------------------------------------------------------------
# Reporting and execution
# -----------------------------------------------------------------------------


def print_summary(paths: RecompilePaths, stats: RecompileStats) -> None:
    print(f"\nWrote: {paths.output_sheet}")
    print(f"Rows in master sheet: {stats.master_rows:,}")
    print(f"Labeled annotator files read: {stats.annotator_files:,}")
    print(f"Author files read: {stats.author_files:,}")
    print(
        "Rows with at least one annotator answer: "
        f"{stats.rows_with_any_annotator_answer:,}"
    )
    print(f"Rows with a submitted author grade: {stats.rows_with_author_grade:,}")
    print(f"Rows resolved by author: {stats.rows_resolved_by_author:,}")
    if stats.unknown_annotator_row_ids:
        print(
            "Unknown annotator row IDs ignored: "
            f"{stats.unknown_annotator_row_ids:,}"
        )
    if stats.unknown_author_row_ids:
        print(f"Unknown author row IDs ignored: {stats.unknown_author_row_ids:,}")

    print("\nDistribution of number of annotator answers per row:")
    for answer_count in sorted(stats.answer_count_distribution):
        print(
            f"{answer_count} answer(s): "
            f"{stats.answer_count_distribution[answer_count]:,}"
        )

    print("\nAnnotator-only resolution status:")
    for status in sorted(stats.annotator_resolution_distribution):
        print(f"{status}: {stats.annotator_resolution_distribution[status]:,}")

    print("\nFinal resolution status after author adjudication:")
    for status in sorted(stats.final_resolution_distribution):
        print(f"{status}: {stats.final_resolution_distribution[status]:,}")


def run(paths: RecompilePaths, config: RecompileConfig) -> RecompileStats:
    set_csv_field_size_limit()
    validate_config(config)

    print(f"Reading master annotation sheet: {paths.master_sheet}")
    master_rows, master_fieldnames = read_master_annotation_sheet(paths.master_sheet)
    validate_master_assignments(master_rows, config.annotators)
    print(f"Master rows: {len(master_rows):,}")

    annotator_files = find_annotator_files(paths.annotator_dir, config.annotators)
    print(f"Found {len(annotator_files):,} labeled annotator file(s).")

    author_files = find_author_files(paths)
    print(f"Found {len(author_files):,} author file(s).")

    master_row_ids = {normalize_value(row.get("row_id")) for row in master_rows}
    answers_by_row, unknown_annotator_row_ids = build_answer_map(
        annotator_files=annotator_files,
        master_row_ids=master_row_ids,
        allow_unknown_row_ids=config.allow_unknown_row_ids,
    )
    author_answers_by_row, unknown_author_row_ids = build_author_map(
        author_files=author_files,
        master_row_ids=master_row_ids,
        allow_unknown_row_ids=config.allow_unknown_row_ids,
    )

    compiled_rows, output_fieldnames, stats = compile_rows(
        master_rows=master_rows,
        master_fieldnames=master_fieldnames,
        answers_by_row=answers_by_row,
        author_answers_by_row=author_answers_by_row,
        config=config,
    )
    stats.annotator_files = len(annotator_files)
    stats.author_files = len(author_files)
    stats.unknown_annotator_row_ids = len(unknown_annotator_row_ids)
    stats.unknown_author_row_ids = len(unknown_author_row_ids)

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
EXPLICIT_ADJUDICATION_SHEETS: tuple[Path, ...] = ()
ALLOW_UNKNOWN_ROW_IDS = False
ALLOW_ADJUDICATION_OVERRIDES = False
REQUIRE_COMPLETE_ADJUDICATION = True


def main() -> int:
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    eval_dir = resolve_project_path(project_root, ANNOTATION_DIRECTORY)
    adjudication_root = eval_dir / "adjudication_assignments"
    adjudication_labeled = adjudication_root / "labeled"
    adjudication_dir = (
        adjudication_labeled if adjudication_labeled.is_dir() else adjudication_root
    )
    args = SimpleNamespace(
        eval_dir=eval_dir,
        master_sheet=eval_dir / "annotation_sheet_empty.csv",
        annotator_dir=eval_dir / "annotator_assignments" / "labeled",
        author_dir=adjudication_dir,
        author_sheet=[
            resolve_project_path(project_root, path)
            for path in EXPLICIT_ADJUDICATION_SHEETS
        ],
        output_sheet=eval_dir / "annotation_sheet_final.csv",
    )
    paths = resolve_paths(args)
    config = RecompileConfig(
        annotators=normalize_annotators(ANNOTATORS),
        allow_unknown_row_ids=ALLOW_UNKNOWN_ROW_IDS,
        allow_author_overrides=ALLOW_ADJUDICATION_OVERRIDES,
        require_complete_author_judgments=REQUIRE_COMPLETE_ADJUDICATION,
    )

    try:
        run(paths, config)
        return 0
    except (FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
