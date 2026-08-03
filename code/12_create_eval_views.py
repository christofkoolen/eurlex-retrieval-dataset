#!/usr/bin/env python3
"""Create all human and citation evaluation-view datasets.

The script creates twelve query/corpus view combinations under human and the
same twelve combinations under citations. Edit the IDE configuration near the
top of the file and run the complete script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
SOURCE_JSON_RELATIVE = Path("data/enriched_json")
QRELS_RELATIVE = Path("data/qrels")
EVAL_VIEWS_RELATIVE = Path("data/eval_views")

DATA_ROOT = Path()
SOURCE_JSON_PATH = Path()
QRELS_ROOT = Path()
EVAL_VIEWS_ROOT = Path()
HUMAN_BINARY_QRELS = Path()
HUMAN_GRADED_QRELS = Path()
CITATION_QRELS = Path()
QUERY_IDS_SOURCE = Path()
MANIFEST_PATH = Path()
SUMMARY_PATH = Path()

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

OVERWRITE_EXISTING_VIEW_FOLDERS = True
FILESYSTEM_RETRY_ATTEMPTS = 20
FILESYSTEM_RETRY_DELAY_SECONDS = 0.5
KEEP_ZERO_JUDGMENTS = True
REQUIRE_HUMAN_POSITIVE_PER_QUERY = True
REQUIRE_HUMAN_QRELS_PAIR_MATCH = True
REQUIRE_BINARY_GRADED_CONSISTENCY = True
FAIL_ON_MISSING_QREL_DOCUMENTS = True
FAIL_ON_EMPTY_QUERY_TEXT = True
FAIL_ON_EMPTY_CORPUS_TEXT = False
TEXT_MAX_CHARACTERS: int | None = None

VALID_BINARY_RELEVANCE = {0, 1}
VALID_GRADED_RELEVANCE = {0, 1, 2, 3}
VALID_QUERY_VIEWS = {"summary", "facts", "full"}
VALID_CORPUS_VIEWS = {"summary", "facts", "reasoning", "full"}


def configure_runtime_paths() -> Path:
    global DATA_ROOT
    global SOURCE_JSON_PATH
    global QRELS_ROOT
    global EVAL_VIEWS_ROOT
    global HUMAN_BINARY_QRELS
    global HUMAN_GRADED_QRELS
    global CITATION_QRELS
    global QUERY_IDS_SOURCE
    global MANIFEST_PATH
    global SUMMARY_PATH

    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    DATA_ROOT = resolve_project_path(project_root, Path("data"))
    SOURCE_JSON_PATH = resolve_project_path(project_root, SOURCE_JSON_RELATIVE)
    QRELS_ROOT = resolve_project_path(project_root, QRELS_RELATIVE)
    EVAL_VIEWS_ROOT = resolve_project_path(project_root, EVAL_VIEWS_RELATIVE)
    HUMAN_BINARY_QRELS = QRELS_ROOT / "qrels_human_binary.tsv"
    HUMAN_GRADED_QRELS = QRELS_ROOT / "qrels_human_graded.tsv"
    CITATION_QRELS = QRELS_ROOT / "qrels_citations.tsv"
    QUERY_IDS_SOURCE = HUMAN_BINARY_QRELS
    MANIFEST_PATH = EVAL_VIEWS_ROOT / "eval_views_manifest.tsv"
    SUMMARY_PATH = EVAL_VIEWS_ROOT / "eval_views_summary.txt"
    return project_root


# =============================================================================
# Data structures
# =============================================================================

ECLI_RE = re.compile(
    r"\b(?:ECLI:)?(EU:(?:C|T|F):\d{4}:\d{1,6})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Qrel:
    query_id: str
    document_id: str
    relevance: int


@dataclass(frozen=True)
class CaseDocument:
    document_id: str
    source_path: Path
    summary: str
    facts: str
    reasoning: str
    full: str

    def text_for_view(self, view: str) -> str:
        if view == "summary":
            return self.summary
        if view == "facts":
            return self.facts
        if view == "reasoning":
            return self.reasoning
        if view == "full":
            return self.full
        raise ValueError(f"Unknown text view: {view!r}")


@dataclass(frozen=True)
class ViewPair:
    query_view: str
    corpus_view: str

    @property
    def folder_name(self) -> str:
        return f"q_{self.query_view}__c_{self.corpus_view}"


# =============================================================================
# General helpers
# =============================================================================

def normalize_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_category(value: object) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def truncate_text(text: str) -> str:
    if TEXT_MAX_CHARACTERS is None:
        return text

    if TEXT_MAX_CHARACTERS <= 0:
        raise ValueError(
            "TEXT_MAX_CHARACTERS must be None or a positive integer."
        )

    return text[:TEXT_MAX_CHARACTERS]


def deduplicate_text_parts(parts: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for part in parts:
        cleaned = normalize_text(part)

        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)
        output.append(cleaned)

    return output


def join_text_parts(parts: Iterable[str]) -> str:
    return truncate_text(
        "\n\n".join(deduplicate_text_parts(parts)).strip()
    )


def collect_text(node: Any) -> list[str]:
    """
    Recursively collect textual content from nested JSON structures.

    Metadata-like identifier and link values are excluded where practical.
    """
    if isinstance(node, str):
        text = node.strip()
        return [text] if text else []

    if isinstance(node, list):
        parts: list[str] = []

        for item in node:
            parts.extend(collect_text(item))

        return parts

    if isinstance(node, dict):
        preferred_text_keys = (
            "content",
            "text",
            "paragraph",
            "summary_text",
            "value",
        )

        preferred_parts: list[str] = []

        for key in preferred_text_keys:
            if key in node:
                preferred_parts.extend(
                    collect_text(node.get(key))
                )

        if preferred_parts:
            return preferred_parts

        excluded_keys = {
            "id",
            "ecli",
            "case_number",
            "document_link",
            "url",
            "link",
            "category",
            "type",
            "label",
        }

        parts = []

        for key, value in node.items():
            if normalize_category(key) in excluded_keys:
                continue
            parts.extend(collect_text(value))

        return parts

    return []


def canonical_ecli(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    match = ECLI_RE.search(value)

    if not match:
        return None

    return f"ECLI:{match.group(1).upper()}"


def choose_document_id(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None

    ecli = canonical_ecli(metadata.get("ecli"))

    if ecli:
        return ecli

    case_number = normalize_text(metadata.get("case_number"))

    return case_number or None


def iter_json_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(
                f"Source file must have a .json extension: {path}"
            )
        yield path
        return

    yield from sorted(path.rglob("*.json"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def is_transient_filesystem_lock(exc: OSError) -> bool:
    """
    Return True for common Windows sharing and access-denied lock errors.

    Google Drive, OneDrive, antivirus software, and open editors can briefly
    lock a path while it is being synchronized or inspected.
    """
    winerror = getattr(exc, "winerror", None)

    return (
        isinstance(exc, PermissionError)
        or winerror in {5, 32, 33}
    )


def retry_filesystem_operation(
    operation,
    *,
    description: str,
):
    last_error: OSError | None = None

    for attempt in range(
        1,
        FILESYSTEM_RETRY_ATTEMPTS + 1,
    ):
        try:
            return operation()
        except OSError as exc:
            last_error = exc

            if (
                not is_transient_filesystem_lock(exc)
                or attempt == FILESYSTEM_RETRY_ATTEMPTS
            ):
                raise

            wait_seconds = (
                FILESYSTEM_RETRY_DELAY_SECONDS
                * attempt
            )

            print(
                f"[retry {attempt}/{FILESYSTEM_RETRY_ATTEMPTS}] "
                f"{description}: {exc}. "
                f"Waiting {wait_seconds:.1f} second(s)."
            )

            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        f"Filesystem operation did not run: {description}"
    )


def unlink_with_retry(
    path: Path,
    *,
    missing_ok: bool = True,
) -> None:
    if not path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(path)

    retry_filesystem_operation(
        lambda: path.unlink(),
        description=f"Removing file {path}",
    )


def remove_tree_with_retry(
    path: Path,
    *,
    ignore_failure: bool = False,
) -> None:
    if not path.exists():
        return

    try:
        retry_filesystem_operation(
            lambda: shutil.rmtree(path),
            description=f"Removing directory {path}",
        )
    except OSError:
        if not ignore_failure:
            raise

        print(
            "[warning] Could not remove temporary directory after retries: "
            f"{path}"
        )


def replace_file_with_retry(
    source: Path,
    destination: Path,
) -> None:
    retry_filesystem_operation(
        lambda: os.replace(source, destination),
        description=(
            f"Replacing {destination} with {source}"
        ),
    )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        unlink_with_retry(
            temporary_path,
            missing_ok=True,
        )

        temporary_path.write_text(
            text,
            encoding="utf-8",
        )

        replace_file_with_retry(
            temporary_path,
            path,
        )

    except Exception:
        try:
            unlink_with_retry(
                temporary_path,
                missing_ok=True,
            )
        except OSError:
            pass
        raise


def publish_prepared_directory(
    prepared_directory: Path,
    final_directory: Path,
) -> None:
    """
    Publish generated files without renaming the containing directory.

    Directory-level replacement can fail with WinError 5 in Google Drive,
    OneDrive, and other synchronized Windows folders. This function keeps the
    final directory in place and atomically replaces each generated file.
    """
    if not prepared_directory.is_dir():
        raise FileNotFoundError(
            f"Prepared directory does not exist: {prepared_directory}"
        )

    if final_directory.exists():
        if not final_directory.is_dir():
            raise ValueError(
                f"Expected an output directory, found a file: "
                f"{final_directory}"
            )

        if not OVERWRITE_EXISTING_VIEW_FOLDERS:
            raise FileExistsError(
                "Evaluation-view folder already exists and overwriting is "
                f"disabled: {final_directory}"
            )
    else:
        retry_filesystem_operation(
            lambda: final_directory.mkdir(
                parents=True,
                exist_ok=True,
            ),
            description=(
                f"Creating output directory {final_directory}"
            ),
        )

    prepared_entries = list(
        prepared_directory.iterdir()
    )

    unexpected_directories = [
        entry
        for entry in prepared_entries
        if entry.is_dir()
    ]

    if unexpected_directories:
        raise ValueError(
            "Prepared evaluation-view directory contains unexpected "
            "subdirectories: "
            + ", ".join(
                str(path)
                for path in unexpected_directories
            )
        )

    prepared_files = {
        entry.name: entry
        for entry in prepared_entries
        if entry.is_file()
    }

    expected_generated_names = {
        "corpus.jsonl",
        "queries.jsonl",
        "qrels_selected.tsv",
        "qrels_selected_graded.tsv",
    }

    # Remove a stale graded qrels file when publishing a citation view.
    stale_names = (
        expected_generated_names
        .difference(prepared_files)
    )

    for stale_name in sorted(stale_names):
        stale_path = final_directory / stale_name

        if stale_path.exists():
            unlink_with_retry(stale_path)

    for filename, prepared_file in sorted(
        prepared_files.items()
    ):
        final_file = final_directory / filename
        publishing_file = final_directory / (
            f".{filename}.publishing"
        )

        unlink_with_retry(
            publishing_file,
            missing_ok=True,
        )

        retry_filesystem_operation(
            lambda source=prepared_file, target=publishing_file: (
                shutil.copyfile(source, target)
            ),
            description=(
                f"Copying prepared file {prepared_file} "
                f"to {publishing_file}"
            ),
        )

        replace_file_with_retry(
            publishing_file,
            final_file,
        )

    remove_tree_with_retry(
        prepared_directory,
        ignore_failure=True,
    )


# =============================================================================
# Text extraction
# =============================================================================

FACT_CATEGORY_NAMES = {
    "facts",
    "fact",
    "background",
    "facts_and_background",
    "factual_background",
    "procedure_and_facts",
    "facts_background",
}

REASONING_CATEGORY_NAMES = {
    "reasoning",
    "legal_reasoning",
    "court_reasoning",
    "analysis",
    "assessment",
    "considerations",
    "grounds",
}

FACT_KEY_TERMS = (
    "fact",
    "background",
)

REASONING_KEY_TERMS = (
    "reason",
    "analysis",
    "assessment",
    "consideration",
    "ground",
)


def get_summary_text(record: Mapping[str, Any]) -> str:
    case_summary = record.get("case_summary")

    if not isinstance(case_summary, dict):
        return ""

    summary_text = case_summary.get("summary_text")

    return join_text_parts(
        [summary_text] if isinstance(summary_text, str) else []
    )


def body_paragraphs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    body = record.get("body")

    if not isinstance(body, list):
        return []

    return [
        paragraph
        for paragraph in body
        if isinstance(paragraph, dict)
    ]


def body_text_by_categories(
    record: Mapping[str, Any],
    accepted_categories: set[str],
) -> str:
    parts: list[str] = []

    for paragraph in body_paragraphs(record):
        category = normalize_category(
            paragraph.get("category")
            or paragraph.get("type")
            or paragraph.get("label")
        )

        if category not in accepted_categories:
            continue

        content = (
            paragraph.get("content")
            if "content" in paragraph
            else paragraph.get("text")
        )

        parts.extend(collect_text(content))

    return join_text_parts(parts)


def body_full_text(record: Mapping[str, Any]) -> str:
    parts: list[str] = []

    for paragraph in body_paragraphs(record):
        content = (
            paragraph.get("content")
            if "content" in paragraph
            else paragraph.get("text")
        )

        parts.extend(collect_text(content))

    return join_text_parts(parts)


def segment_text_by_key_terms(
    record: Mapping[str, Any],
    key_terms: Sequence[str],
) -> str:
    case_segments = record.get("case_segments")

    if not isinstance(case_segments, dict):
        return ""

    parts: list[str] = []

    for key, value in case_segments.items():
        normalized_key = normalize_category(key)

        if any(term in normalized_key for term in key_terms):
            parts.extend(collect_text(value))

    return join_text_parts(parts)


def segment_full_text(record: Mapping[str, Any]) -> str:
    case_segments = record.get("case_segments")

    if not isinstance(case_segments, dict):
        return ""

    parts: list[str] = []

    for value in case_segments.values():
        parts.extend(collect_text(value))

    return join_text_parts(parts)


def fallback_full_text(record: Mapping[str, Any]) -> str:
    for key in (
        "full_text",
        "case_text",
        "document_text",
        "text",
    ):
        value = record.get(key)
        parts = collect_text(value)

        if parts:
            return join_text_parts(parts)

    return ""


def extract_case_views(
    record: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    summary = get_summary_text(record)

    facts = body_text_by_categories(
        record,
        FACT_CATEGORY_NAMES,
    )

    if not facts:
        facts = segment_text_by_key_terms(
            record,
            FACT_KEY_TERMS,
        )

    reasoning = body_text_by_categories(
        record,
        REASONING_CATEGORY_NAMES,
    )

    if not reasoning:
        reasoning = segment_text_by_key_terms(
            record,
            REASONING_KEY_TERMS,
        )

    full = body_full_text(record)

    if not full:
        full = segment_full_text(record)

    if not full:
        full = fallback_full_text(record)

    return summary, facts, reasoning, full


# =============================================================================
# Corpus loading
# =============================================================================

def load_case_corpus(
    source_path: Path,
) -> dict[str, CaseDocument]:
    if not source_path.exists():
        raise FileNotFoundError(
            f"SOURCE_JSON_PATH does not exist: {source_path}"
        )

    documents: dict[str, CaseDocument] = {}
    json_files = list(iter_json_files(source_path))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files were found under: {source_path}"
        )

    for file_number, file_path in enumerate(
        json_files,
        start=1,
    ):
        try:
            record = json.loads(
                file_path.read_text(encoding="utf-8")
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                f"Could not read JSON file {file_path}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"JSON root must be an object: {file_path}"
            )

        metadata = record.get("case_metadata")
        document_id = choose_document_id(metadata)

        if not document_id:
            raise ValueError(
                f"Could not determine a document ID for: {file_path}"
            )

        if document_id in documents:
            first_path = documents[document_id].source_path
            raise ValueError(
                f"Duplicate document ID {document_id!r}: "
                f"{first_path} and {file_path}"
            )

        summary, facts, reasoning, full = extract_case_views(
            record
        )

        documents[document_id] = CaseDocument(
            document_id=document_id,
            source_path=file_path,
            summary=summary,
            facts=facts,
            reasoning=reasoning,
            full=full,
        )

        if file_number % 1000 == 0:
            print(
                f"  Loaded {file_number:,} of "
                f"{len(json_files):,} JSON files..."
            )

    return documents


# =============================================================================
# Qrels input and validation
# =============================================================================

def parse_integer_relevance(
    value: str,
    *,
    path: Path,
    line_number: int,
) -> int:
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            f"Invalid relevance value {value!r} in {path} "
            f"at line {line_number}."
        ) from exc

    if decimal_value != decimal_value.to_integral_value():
        raise ValueError(
            f"Relevance must be an integer in {path} at "
            f"line {line_number}; found {value!r}."
        )

    return int(decimal_value)


def read_qrels(
    path: Path,
    *,
    allowed_relevance: set[int],
) -> list[Qrel]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Qrels file does not exist: {path}"
        )

    qrels: list[Qrel] = []
    seen: dict[tuple[str, str], int] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            fields = line.split("\t")

            if len(fields) != 4:
                fields = line.split()

            if len(fields) != 4:
                raise ValueError(
                    f"Expected four qrels fields in {path} at line "
                    f"{line_number}; found {len(fields)}."
                )

            query_id = normalize_text(fields[0])
            document_id = normalize_text(fields[2])
            relevance_text = normalize_text(fields[3])

            # Permit a header row.
            if (
                line_number == 1
                and query_id.lower() in {"query_id", "qid"}
                and relevance_text.lower()
                in {"relevance", "rel", "label", "grade"}
            ):
                continue

            if not query_id or not document_id:
                raise ValueError(
                    f"Blank query or document ID in {path} at "
                    f"line {line_number}."
                )

            relevance = parse_integer_relevance(
                relevance_text,
                path=path,
                line_number=line_number,
            )

            if relevance not in allowed_relevance:
                raise ValueError(
                    f"Unexpected relevance {relevance} in {path} at "
                    f"line {line_number}. Allowed values: "
                    f"{sorted(allowed_relevance)}"
                )

            key = (query_id, document_id)
            existing = seen.get(key)

            if existing is not None:
                if existing != relevance:
                    raise ValueError(
                        "Conflicting duplicate qrels pair in "
                        f"{path}: query={query_id!r}, "
                        f"document={document_id!r}, "
                        f"relevance values {existing} and {relevance}."
                    )
                continue

            seen[key] = relevance
            qrels.append(
                Qrel(
                    query_id=query_id,
                    document_id=document_id,
                    relevance=relevance,
                )
            )

    if not qrels:
        raise ValueError(
            f"No qrels records were loaded from: {path}"
        )

    return qrels


def fixed_query_ids_from_qrels(
    qrels: Sequence[Qrel],
) -> list[str]:
    seen: set[str] = set()
    query_ids: list[str] = []

    for qrel in qrels:
        if qrel.query_id in seen:
            continue

        seen.add(qrel.query_id)
        query_ids.append(qrel.query_id)

    return query_ids


def validate_human_qrels(
    binary_qrels: Sequence[Qrel],
    graded_qrels: Sequence[Qrel],
) -> None:
    binary_map = {
        (qrel.query_id, qrel.document_id): qrel.relevance
        for qrel in binary_qrels
    }

    graded_map = {
        (qrel.query_id, qrel.document_id): qrel.relevance
        for qrel in graded_qrels
    }

    if REQUIRE_HUMAN_QRELS_PAIR_MATCH:
        binary_pairs = set(binary_map)
        graded_pairs = set(graded_map)

        if binary_pairs != graded_pairs:
            binary_only = sorted(
                binary_pairs.difference(graded_pairs)
            )
            graded_only = sorted(
                graded_pairs.difference(binary_pairs)
            )

            raise ValueError(
                "Human binary and graded qrels do not contain the same "
                "query-document pairs. "
                f"Binary-only examples: {binary_only[:10]}; "
                f"graded-only examples: {graded_only[:10]}."
            )

    if REQUIRE_BINARY_GRADED_CONSISTENCY:
        shared_pairs = set(binary_map).intersection(
            graded_map
        )

        inconsistent: list[
            tuple[str, str, int, int]
        ] = []

        for pair in shared_pairs:
            binary_value = binary_map[pair]
            graded_value = graded_map[pair]
            expected_binary = (
                1 if graded_value in {2, 3} else 0
            )

            if binary_value != expected_binary:
                inconsistent.append(
                    (
                        pair[0],
                        pair[1],
                        binary_value,
                        graded_value,
                    )
                )

        if inconsistent:
            raise ValueError(
                "Human binary qrels are inconsistent with the graded "
                "cutoff 0-1 versus 2-3. Examples: "
                f"{inconsistent[:10]}"
            )


def subset_qrels(
    qrels: Sequence[Qrel],
    *,
    allowed_query_ids: set[str],
    allowed_document_ids: set[str],
    label: str,
) -> tuple[list[Qrel], set[str]]:
    selected: list[Qrel] = []
    missing_document_ids: set[str] = set()

    for qrel in qrels:
        if qrel.query_id not in allowed_query_ids:
            continue

        if qrel.document_id not in allowed_document_ids:
            missing_document_ids.add(
                qrel.document_id
            )
            continue

        if (
            not KEEP_ZERO_JUDGMENTS
            and qrel.relevance == 0
        ):
            continue

        selected.append(qrel)

    if (
        missing_document_ids
        and FAIL_ON_MISSING_QREL_DOCUMENTS
    ):
        examples = sorted(missing_document_ids)[:20]

        raise ValueError(
            f"{label} qrels reference "
            f"{len(missing_document_ids):,} document(s) absent from "
            f"the JSON corpus. Examples: {examples}"
        )

    return selected, missing_document_ids


def validate_positive_human_queries(
    query_ids: Sequence[str],
    human_binary_qrels: Sequence[Qrel],
) -> None:
    positives_by_query: Counter[str] = Counter()

    for qrel in human_binary_qrels:
        if qrel.relevance == 1:
            positives_by_query[qrel.query_id] += 1

    missing = [
        query_id
        for query_id in query_ids
        if positives_by_query[query_id] == 0
    ]

    if missing and REQUIRE_HUMAN_POSITIVE_PER_QUERY:
        raise ValueError(
            f"{len(missing):,} fixed query or queries have no positive "
            "human candidate. Examples: "
            + ", ".join(missing[:20])
        )


# =============================================================================
# Output writers
# =============================================================================

def json_line(record: Mapping[str, object]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_corpus_jsonl(
    path: Path,
    documents: Mapping[str, CaseDocument],
    view: str,
) -> tuple[int, int]:
    row_count = 0
    empty_count = 0

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for document_id in sorted(documents):
            text = documents[
                document_id
            ].text_for_view(view)

            if not text:
                empty_count += 1

            handle.write(
                json_line(
                    {
                        "doc_id": document_id,
                        "text": text,
                    }
                )
                + "\n"
            )
            row_count += 1

    if empty_count and FAIL_ON_EMPTY_CORPUS_TEXT:
        raise ValueError(
            f"Corpus view {view!r} contains "
            f"{empty_count:,} empty document text(s)."
        )

    return row_count, empty_count


def write_queries_jsonl(
    path: Path,
    query_ids: Sequence[str],
    documents: Mapping[str, CaseDocument],
    view: str,
) -> tuple[int, int]:
    row_count = 0
    empty_query_ids: list[str] = []

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for query_id in query_ids:
            text = documents[
                query_id
            ].text_for_view(view)

            if not text:
                empty_query_ids.append(query_id)

            handle.write(
                json_line(
                    {
                        "query_id": query_id,
                        "text": text,
                    }
                )
                + "\n"
            )
            row_count += 1

    if empty_query_ids and FAIL_ON_EMPTY_QUERY_TEXT:
        raise ValueError(
            f"Query view {view!r} contains "
            f"{len(empty_query_ids):,} empty query text(s). "
            f"Examples: {empty_query_ids[:20]}"
        )

    return row_count, len(empty_query_ids)


def write_qrels_tsv(
    path: Path,
    qrels: Sequence[Qrel],
) -> int:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )

        for qrel in qrels:
            writer.writerow(
                [
                    qrel.query_id,
                    "0",
                    qrel.document_id,
                    qrel.relevance,
                ]
            )

    return len(qrels)


def write_view_folder(
    *,
    qrels_source: str,
    pair: ViewPair,
    documents: Mapping[str, CaseDocument],
    query_ids: Sequence[str],
    binary_qrels: Sequence[Qrel],
    graded_qrels: Sequence[Qrel] | None,
) -> dict[str, object]:
    source_root = EVAL_VIEWS_ROOT / qrels_source
    final_folder = source_root / pair.folder_name

    source_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clean up stale temporary folders left by an interrupted prior run.
    for stale_folder in source_root.glob(
        pair.folder_name + ".building.*"
    ):
        remove_tree_with_retry(
            stale_folder,
            ignore_failure=True,
        )

    temporary_folder = Path(
        tempfile.mkdtemp(
            prefix=pair.folder_name + ".building.",
            dir=source_root,
        )
    )

    try:
        corpus_path = temporary_folder / "corpus.jsonl"
        queries_path = temporary_folder / "queries.jsonl"
        binary_qrels_path = (
            temporary_folder / "qrels_selected.tsv"
        )

        corpus_count, empty_corpus_count = (
            write_corpus_jsonl(
                corpus_path,
                documents,
                pair.corpus_view,
            )
        )

        query_count, empty_query_count = (
            write_queries_jsonl(
                queries_path,
                query_ids,
                documents,
                pair.query_view,
            )
        )

        binary_qrels_count = write_qrels_tsv(
            binary_qrels_path,
            binary_qrels,
        )

        graded_qrels_count = 0
        graded_qrels_path: Path | None = None

        if graded_qrels is not None:
            graded_qrels_path = (
                temporary_folder
                / "qrels_selected_graded.tsv"
            )
            graded_qrels_count = write_qrels_tsv(
                graded_qrels_path,
                graded_qrels,
            )

        publish_prepared_directory(
            temporary_folder,
            final_folder,
        )

    except Exception:
        remove_tree_with_retry(
            temporary_folder,
            ignore_failure=True,
        )
        raise

    final_corpus_path = final_folder / "corpus.jsonl"
    final_queries_path = final_folder / "queries.jsonl"
    final_binary_path = (
        final_folder / "qrels_selected.tsv"
    )
    final_graded_path = (
        final_folder / "qrels_selected_graded.tsv"
        if graded_qrels is not None
        else None
    )

    return {
        "qrels_source": qrels_source,
        "query_view": pair.query_view,
        "corpus_view": pair.corpus_view,
        "view_folder": final_folder.relative_to(
            DATA_ROOT
        ).as_posix(),
        "document_count": corpus_count,
        "query_count": query_count,
        "empty_document_text_count": (
            empty_corpus_count
        ),
        "empty_query_text_count": empty_query_count,
        "binary_qrels_count": binary_qrels_count,
        "graded_qrels_count": graded_qrels_count,
        "corpus_sha256": sha256_file(
            final_corpus_path
        ),
        "queries_sha256": sha256_file(
            final_queries_path
        ),
        "binary_qrels_sha256": sha256_file(
            final_binary_path
        ),
        "graded_qrels_sha256": (
            sha256_file(final_graded_path)
            if final_graded_path is not None
            else ""
        ),
    }


# =============================================================================
# Manifest and summary
# =============================================================================

def write_manifest(
    rows: Sequence[Mapping[str, object]],
) -> None:
    fieldnames = [
        "qrels_source",
        "query_view",
        "corpus_view",
        "view_folder",
        "document_count",
        "query_count",
        "empty_document_text_count",
        "empty_query_text_count",
        "binary_qrels_count",
        "graded_qrels_count",
        "corpus_sha256",
        "queries_sha256",
        "binary_qrels_sha256",
        "graded_qrels_sha256",
    ]

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = MANIFEST_PATH.with_suffix(
        MANIFEST_PATH.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field: row.get(field, "")
                        for field in fieldnames
                    }
                )

        temporary_path.replace(MANIFEST_PATH)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def qrels_counts_by_query(
    qrels: Sequence[Qrel],
) -> dict[str, Counter[int]]:
    output: dict[str, Counter[int]] = defaultdict(
        Counter
    )

    for qrel in qrels:
        output[qrel.query_id][
            qrel.relevance
        ] += 1

    return dict(output)


def write_summary(
    *,
    documents: Mapping[str, CaseDocument],
    query_ids: Sequence[str],
    human_binary_qrels: Sequence[Qrel],
    human_graded_qrels: Sequence[Qrel],
    citation_qrels: Sequence[Qrel],
    manifest_rows: Sequence[Mapping[str, object]],
) -> None:
    human_binary_by_query = qrels_counts_by_query(
        human_binary_qrels
    )
    citation_by_query = qrels_counts_by_query(
        citation_qrels
    )

    lines: list[str] = []

    lines.append("EVALUATION VIEWS SUMMARY")
    lines.append("========================")
    lines.append("")
    lines.append(f"Source JSON: {SOURCE_JSON_PATH}")
    lines.append(f"Data root: {DATA_ROOT}")
    lines.append(
        f"Evaluation views root: {EVAL_VIEWS_ROOT}"
    )
    lines.append("")

    lines.append("Input counts")
    lines.append("------------")
    lines.append(
        f"Corpus documents: {len(documents):,}"
    )
    lines.append(
        f"Fixed evaluation queries: {len(query_ids):,}"
    )
    lines.append(
        "Selected human binary qrels: "
        f"{len(human_binary_qrels):,}"
    )
    lines.append(
        "Selected human graded qrels: "
        f"{len(human_graded_qrels):,}"
    )
    lines.append(
        "Selected citation qrels: "
        f"{len(citation_qrels):,}"
    )
    lines.append("")

    lines.append("Text-view coverage")
    lines.append("------------------")

    for view in (
        "summary",
        "facts",
        "reasoning",
        "full",
    ):
        empty_document_count = sum(
            not document.text_for_view(view)
            for document in documents.values()
        )
        empty_query_count = sum(
            not documents[query_id].text_for_view(view)
            for query_id in query_ids
        )

        lines.append(
            f"{view}: empty corpus texts="
            f"{empty_document_count:,}; "
            f"empty query texts={empty_query_count:,}"
        )

    lines.append("")
    lines.append("Human positive-candidate coverage")
    lines.append("---------------------------------")

    human_positive_counts = [
        human_binary_by_query.get(
            query_id,
            Counter(),
        )[1]
        for query_id in query_ids
    ]

    if human_positive_counts:
        mean_positive = (
            sum(human_positive_counts)
            / len(human_positive_counts)
        )

        sorted_positive = sorted(
            human_positive_counts
        )

        middle = len(sorted_positive) // 2

        if len(sorted_positive) % 2:
            median_positive = float(
                sorted_positive[middle]
            )
        else:
            median_positive = (
                sorted_positive[middle - 1]
                + sorted_positive[middle]
            ) / 2

        lines.append(
            "Queries with at least one positive: "
            f"{sum(count > 0 for count in human_positive_counts):,}"
        )
        lines.append(
            "Queries without a positive: "
            f"{sum(count == 0 for count in human_positive_counts):,}"
        )
        lines.append(
            "Positive candidates per query: "
            f"min={min(human_positive_counts):,}; "
            f"mean={mean_positive:.4f}; "
            f"median={median_positive:.2f}; "
            f"max={max(human_positive_counts):,}"
        )

    lines.append("")
    lines.append("Citation coverage")
    lines.append("-----------------")

    citation_positive_counts = [
        citation_by_query.get(
            query_id,
            Counter(),
        )[1]
        for query_id in query_ids
    ]

    lines.append(
        "Queries with at least one citation qrel: "
        f"{sum(count > 0 for count in citation_positive_counts):,}"
    )
    lines.append(
        "Queries without a citation qrel: "
        f"{sum(count == 0 for count in citation_positive_counts):,}"
    )

    lines.append("")
    lines.append("Generated view folders")
    lines.append("----------------------")

    for row in manifest_rows:
        lines.append(
            f"- {row['qrels_source']}/"
            f"q_{row['query_view']}__c_{row['corpus_view']}: "
            f"documents={int(row['document_count']):,}; "
            f"queries={int(row['query_count']):,}; "
            f"binary qrels={int(row['binary_qrels_count']):,}; "
            f"graded qrels={int(row['graded_qrels_count']):,}"
        )

    lines.append("")
    lines.append("Next step")
    lines.append("---------")
    lines.append(
        "Use the generated evaluation views with the zero-shot or training runner scripts."
    )

    write_text_atomic(
        SUMMARY_PATH,
        "\n".join(lines) + "\n",
    )


# =============================================================================
# Validation and execution
# =============================================================================

def validate_configuration() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {DATA_ROOT}"
        )

    unknown_query_views = set(
        QUERY_VIEWS
    ).difference(VALID_QUERY_VIEWS)

    if unknown_query_views:
        raise ValueError(
            "Unknown query view(s): "
            + ", ".join(
                sorted(unknown_query_views)
            )
        )

    unknown_corpus_views = set(
        CORPUS_VIEWS
    ).difference(VALID_CORPUS_VIEWS)

    if unknown_corpus_views:
        raise ValueError(
            "Unknown corpus view(s): "
            + ", ".join(
                sorted(unknown_corpus_views)
            )
        )

    if len(set(QUERY_VIEWS)) != len(QUERY_VIEWS):
        raise ValueError(
            "QUERY_VIEWS contains duplicates."
        )

    if len(set(CORPUS_VIEWS)) != len(
        CORPUS_VIEWS
    ):
        raise ValueError(
            "CORPUS_VIEWS contains duplicates."
        )

def main() -> int:
    set_reproducible_seed(123)
    configure_runtime_paths()

    try:
        validate_configuration()
        print(f"Loading JSON case corpus from: {SOURCE_JSON_PATH}")
        documents = load_case_corpus(SOURCE_JSON_PATH)
        print(f"Loaded corpus documents: {len(documents):,}")

        human_binary_all = read_qrels(
            HUMAN_BINARY_QRELS,
            allowed_relevance=VALID_BINARY_RELEVANCE,
        )
        human_graded_all = read_qrels(
            HUMAN_GRADED_QRELS,
            allowed_relevance=VALID_GRADED_RELEVANCE,
        )
        citation_all = read_qrels(
            CITATION_QRELS,
            allowed_relevance=VALID_BINARY_RELEVANCE,
        )
        query_source_qrels = read_qrels(
            QUERY_IDS_SOURCE,
            allowed_relevance=VALID_BINARY_RELEVANCE,
        )

        validate_human_qrels(human_binary_all, human_graded_all)
        query_ids = fixed_query_ids_from_qrels(query_source_qrels)
        missing_query_documents = [
            query_id for query_id in query_ids if query_id not in documents
        ]
        if missing_query_documents:
            raise ValueError(
                f"{len(missing_query_documents):,} fixed query ID(s) are absent "
                "from the JSON corpus. Examples: "
                + ", ".join(missing_query_documents[:20])
            )

        allowed_query_ids = set(query_ids)
        allowed_document_ids = set(documents)
        human_binary_selected, _ = subset_qrels(
            human_binary_all,
            allowed_query_ids=allowed_query_ids,
            allowed_document_ids=allowed_document_ids,
            label="Human binary",
        )
        human_graded_selected, _ = subset_qrels(
            human_graded_all,
            allowed_query_ids=allowed_query_ids,
            allowed_document_ids=allowed_document_ids,
            label="Human graded",
        )
        citation_selected, _ = subset_qrels(
            citation_all,
            allowed_query_ids=allowed_query_ids,
            allowed_document_ids=allowed_document_ids,
            label="Citation",
        )
        validate_positive_human_queries(query_ids, human_binary_selected)

        view_pairs = [
            ViewPair(query_view=query_view, corpus_view=corpus_view)
            for query_view in QUERY_VIEWS
            for corpus_view in CORPUS_VIEWS
        ]
        manifest_rows: list[dict[str, object]] = []
        total_folders = len(view_pairs) * 2
        completed_folders = 0

        for qrels_source in ("human", "citations"):
            for pair in view_pairs:
                completed_folders += 1
                print(
                    f"[{completed_folders}/{total_folders}] Writing "
                    f"{qrels_source}/{pair.folder_name}"
                )
                if qrels_source == "human":
                    row = write_view_folder(
                        qrels_source=qrels_source,
                        pair=pair,
                        documents=documents,
                        query_ids=query_ids,
                        binary_qrels=human_binary_selected,
                        graded_qrels=human_graded_selected,
                    )
                else:
                    row = write_view_folder(
                        qrels_source=qrels_source,
                        pair=pair,
                        documents=documents,
                        query_ids=query_ids,
                        binary_qrels=citation_selected,
                        graded_qrels=None,
                    )
                manifest_rows.append(row)

        write_manifest(manifest_rows)
        write_summary(
            documents=documents,
            query_ids=query_ids,
            human_binary_qrels=human_binary_selected,
            human_graded_qrels=human_graded_selected,
            citation_qrels=citation_selected,
            manifest_rows=manifest_rows,
        )
        print(f"Evaluation views written to: {EVAL_VIEWS_ROOT}")
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        OSError,
        csv.Error,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
