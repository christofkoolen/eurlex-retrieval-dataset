#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create human binary and graded qrels from the final annotation sheet.

Edit the IDE configuration below and run the complete script.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pipeline_common import find_project_root, resolve_project_path


# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
INPUT_ANNOTATION_SHEET = Path("data/annotations/annotation_sheet_final.csv")
OUTPUT_QRELS_DIRECTORY = Path("data/qrels")
INCLUDE_ZERO_JUDGMENTS = True

VALID_BINARY_VALUES = {0, 1}
VALID_GRADED_VALUES = {0, 1, 2, 3}


def normalize_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_exact_integer(
    value: object,
    *,
    field_name: str,
    csv_line_number: int,
) -> int | None:
    text = normalize_text(value)
    if text == "":
        return None
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(
            f"Invalid {field_name} value {text!r} at CSV line "
            f"{csv_line_number}."
        ) from exc
    if decimal_value != decimal_value.to_integral_value():
        raise ValueError(
            f"{field_name} must be an integer at CSV line "
            f"{csv_line_number}; found {text!r}."
        )
    return int(decimal_value)


def binary_from_grade(grade: int) -> int:
    return 1 if grade >= 2 else 0


def write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for query_id, candidate_id, relevance in rows:
                writer.writerow([query_id, "0", candidate_id, relevance])
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def create_qrels(
    input_csv: Path,
    output_binary_tsv: Path,
    output_graded_tsv: Path,
    *,
    include_zero_judgments: bool,
) -> None:
    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Input annotation sheet does not exist: {input_csv}"
        )

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required_columns = {
            "query_id", "candidate_id", "human_binary", "human_graded"
        }
        missing_columns = required_columns.difference(fieldnames)
        if missing_columns:
            raise ValueError(
                "Input annotation sheet is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        judgments: dict[tuple[str, str], tuple[int, int, int]] = {}
        duplicate_identical_rows = 0
        skipped_blank_id_rows = 0

        for csv_line_number, row in enumerate(reader, start=2):
            query_id = normalize_text(row.get("query_id"))
            candidate_id = normalize_text(row.get("candidate_id"))
            if not query_id or not candidate_id:
                skipped_blank_id_rows += 1
                continue

            human_binary = parse_exact_integer(
                row.get("human_binary"),
                field_name="human_binary",
                csv_line_number=csv_line_number,
            )
            human_graded = parse_exact_integer(
                row.get("human_graded"),
                field_name="human_graded",
                csv_line_number=csv_line_number,
            )
            if human_graded is None or human_graded not in VALID_GRADED_VALUES:
                raise ValueError(
                    f"human_graded must be one of {sorted(VALID_GRADED_VALUES)} "
                    f"at CSV line {csv_line_number}; found {human_graded!r}."
                )

            expected_binary = binary_from_grade(human_graded)
            if human_binary is None:
                human_binary = expected_binary
            if human_binary not in VALID_BINARY_VALUES:
                raise ValueError(
                    f"human_binary must be 0 or 1 at CSV line "
                    f"{csv_line_number}; found {human_binary!r}."
                )
            if human_binary != expected_binary:
                raise ValueError(
                    f"Inconsistent judgments at CSV line {csv_line_number}: "
                    f"human_graded={human_graded} implies "
                    f"human_binary={expected_binary}, but found {human_binary}."
                )

            key = (query_id, candidate_id)
            existing = judgments.get(key)
            if existing is None:
                judgments[key] = (human_binary, human_graded, csv_line_number)
            elif existing[:2] == (human_binary, human_graded):
                duplicate_identical_rows += 1
            else:
                raise ValueError(
                    "Conflicting duplicate query-candidate judgment for "
                    f"query_id={query_id!r}, candidate_id={candidate_id!r}."
                )

    binary_qrels: list[tuple[str, str, int]] = []
    graded_qrels: list[tuple[str, str, int]] = []
    binary_distribution: Counter[int] = Counter()
    graded_distribution: Counter[int] = Counter()

    for (query_id, candidate_id), (human_binary, human_graded, _) in judgments.items():
        binary_distribution[human_binary] += 1
        graded_distribution[human_graded] += 1
        if include_zero_judgments or human_binary > 0:
            binary_qrels.append((query_id, candidate_id, human_binary))
        if include_zero_judgments or human_graded > 0:
            graded_qrels.append((query_id, candidate_id, human_graded))

    binary_qrels.sort(key=lambda row: (row[0], row[1]))
    graded_qrels.sort(key=lambda row: (row[0], row[1]))
    write_qrels(output_binary_tsv, binary_qrels)
    write_qrels(output_graded_tsv, graded_qrels)

    print("Qrels creation complete")
    print(f"Input: {input_csv}")
    print(f"Unique judged pairs: {len(judgments):,}")
    print(f"Skipped blank-ID rows: {skipped_blank_id_rows:,}")
    print(f"Identical duplicates ignored: {duplicate_identical_rows:,}")
    print(f"Binary distribution: {dict(sorted(binary_distribution.items()))}")
    print(f"Graded distribution: {dict(sorted(graded_distribution.items()))}")
    print(f"Binary qrels: {output_binary_tsv}")
    print(f"Graded qrels: {output_graded_tsv}")


def main() -> int:
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    input_csv = resolve_project_path(project_root, INPUT_ANNOTATION_SHEET)
    output_dir = resolve_project_path(project_root, OUTPUT_QRELS_DIRECTORY)

    try:
        create_qrels(
            input_csv,
            output_dir / "qrels_human_binary.tsv",
            output_dir / "qrels_human_graded.tsv",
            include_zero_judgments=INCLUDE_ZERO_JUDGMENTS,
        )
        return 0
    except (FileNotFoundError, ValueError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
