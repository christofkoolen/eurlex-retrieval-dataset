#!/usr/bin/env python3
"""Compute inter-annotator agreement metrics and diagnostics.

Edit the IDE configuration near the end of the file and run the complete
script. Agreement is calculated on repeated-assignment rows.
"""


import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


DEFAULT_DPI = 300
DEFAULT_GRADES: Tuple[int, ...] = (0, 1, 2, 3)
ANNOTATOR_COLUMN_RE = re.compile(r"^(annotator_\d{2})_label_grade$", re.IGNORECASE)
REPEATED_ASSIGNMENT_TYPES = {"double", "shared"}


@dataclass(frozen=True)
class AgreementConfig:
    input_path: Path
    output_dir: Path
    annotators: Tuple[str, ...]
    grades: Tuple[int, ...]
    dpi: int
    strict_grades: bool


@dataclass
class InputDiagnostics:
    total_rows: int = 0
    repeated_rows: int = 0
    single_rows: int = 0
    other_assignment_rows: int = 0
    invalid_grade_cells: int = 0
    unassigned_nonempty_grade_cells: int = 0
    repeated_rows_with_fewer_than_two_valid_grades: int = 0


# -----------------------------------------------------------------------------
# Input and validation
# -----------------------------------------------------------------------------


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_grade(value: object, valid_grades: Sequence[int]) -> Optional[int]:
    """Parse an exact integer grade from the configured grade set."""
    text = normalize_text(value)
    if text == "":
        return None

    try:
        numeric = float(text)
    except ValueError:
        return None

    if not math.isfinite(numeric) or not numeric.is_integer():
        return None

    grade = int(numeric)
    return grade if grade in set(valid_grades) else None


def detect_annotators(columns: Iterable[str]) -> Tuple[str, ...]:
    annotators = []
    for column in columns:
        match = ANNOTATOR_COLUMN_RE.match(str(column))
        if match:
            annotators.append(match.group(1).lower())
    return tuple(sorted(set(annotators)))


def load_annotation_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input annotation sheet not found: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_columns = {"row_id", "assignment_type", "assigned_annotators"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(
            "Input annotation sheet is missing required column(s): " + ", ".join(missing)
        )

    frame["row_id"] = frame["row_id"].map(normalize_text)
    if (frame["row_id"] == "").any():
        bad_count = int((frame["row_id"] == "").sum())
        raise ValueError(f"Found {bad_count} row(s) with a blank row_id.")

    duplicated = frame.loc[frame["row_id"].duplicated(keep=False), "row_id"].unique()
    if len(duplicated) > 0:
        sample = ", ".join(map(str, duplicated[:10]))
        raise ValueError(f"row_id must be unique. Duplicate row_id value(s): {sample}")

    return frame


def validate_annotators(
    frame: pd.DataFrame,
    requested_annotators: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    detected = detect_annotators(frame.columns)
    if not detected:
        raise ValueError(
            "No annotator grade columns were found. Expected columns such as "
            "annotator_01_label_grade."
        )

    if requested_annotators:
        requested = tuple(dict.fromkeys(item.strip().lower() for item in requested_annotators))
        missing_columns = [
            annotator
            for annotator in requested
            if f"{annotator}_label_grade" not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing grade column(s) for requested annotator(s): "
                + ", ".join(missing_columns)
            )
        if len(requested) < 2:
            raise ValueError("At least two annotators are required for agreement analysis.")
        return requested

    if len(detected) < 2:
        raise ValueError("At least two annotator grade columns are required for agreement analysis.")
    return detected


def parse_assigned_annotators(value: object) -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            part.strip().lower()
            for part in normalize_text(value).split("|")
            if part.strip()
        )
    )


def build_rating_matrix(
    frame: pd.DataFrame,
    config: AgreementConfig,
) -> Tuple[pd.DataFrame, InputDiagnostics, List[str]]:
    """
    Build a row-by-annotator matrix containing only assigned, valid grades.

    Returns
    -------
    matrix:
        Index is row_id; columns are annotators; values are grades or NaN.
    diagnostics:
        Input quality counts.
    warnings:
        Human-readable warnings for the overview and console.
    """
    diagnostics = InputDiagnostics(total_rows=len(frame))
    warnings: List[str] = []

    matrix = pd.DataFrame(
        np.nan,
        index=frame["row_id"].tolist(),
        columns=list(config.annotators),
        dtype=float,
    )

    known_annotators = set(config.annotators)
    invalid_examples: List[str] = []
    unknown_assignment_examples: List[str] = []

    for _, row in frame.iterrows():
        row_id = normalize_text(row["row_id"])
        assignment_type = normalize_text(row["assignment_type"]).lower()
        assigned = parse_assigned_annotators(row["assigned_annotators"])

        if assignment_type in REPEATED_ASSIGNMENT_TYPES:
            diagnostics.repeated_rows += 1
        elif assignment_type == "single":
            diagnostics.single_rows += 1
        else:
            diagnostics.other_assignment_rows += 1

        unknown = [annotator for annotator in assigned if annotator not in known_annotators]
        if unknown:
            unknown_assignment_examples.append(
                f"row_id={row_id}: {', '.join(unknown)}"
            )

        for annotator in config.annotators:
            column = f"{annotator}_label_grade"
            raw_value = row.get(column, "")
            raw_text = normalize_text(raw_value)
            is_assigned = annotator in assigned

            if not is_assigned:
                if raw_text != "":
                    diagnostics.unassigned_nonempty_grade_cells += 1
                continue

            grade = parse_grade(raw_value, config.grades)
            if grade is not None:
                matrix.at[row_id, annotator] = grade
            elif raw_text != "":
                diagnostics.invalid_grade_cells += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        f"row_id={row_id}, annotator={annotator}, value={raw_text!r}"
                    )

    if unknown_assignment_examples:
        sample = "; ".join(unknown_assignment_examples[:10])
        raise ValueError(
            "assigned_annotators contains annotators without matching grade columns. "
            f"Examples: {sample}"
        )

    repeated_mask = frame["assignment_type"].map(normalize_text).str.lower().isin(
        REPEATED_ASSIGNMENT_TYPES
    )
    repeated_ids = frame.loc[repeated_mask, "row_id"]
    repeated_valid_counts = matrix.loc[repeated_ids].notna().sum(axis=1)
    diagnostics.repeated_rows_with_fewer_than_two_valid_grades = int(
        (repeated_valid_counts < 2).sum()
    )

    if diagnostics.invalid_grade_cells:
        message = (
            f"Found {diagnostics.invalid_grade_cells} non-empty invalid grade cell(s). "
            "They were treated as missing."
        )
        if invalid_examples:
            message += " Examples: " + "; ".join(invalid_examples)
        if config.strict_grades:
            raise ValueError(message)
        warnings.append(message)

    if diagnostics.unassigned_nonempty_grade_cells:
        warnings.append(
            f"Ignored {diagnostics.unassigned_nonempty_grade_cells} non-empty grade cell(s) "
            "belonging to annotators who were not assigned to those rows."
        )

    if diagnostics.repeated_rows_with_fewer_than_two_valid_grades:
        warnings.append(
            f"{diagnostics.repeated_rows_with_fewer_than_two_valid_grades} repeated-assignment "
            "row(s) had fewer than two valid grades and could not contribute to agreement."
        )

    return matrix, diagnostics, warnings


# -----------------------------------------------------------------------------
# Agreement metrics
# -----------------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else float("nan")


def cohen_kappa(
    ratings_a: Sequence[int],
    ratings_b: Sequence[int],
    grades: Sequence[int],
    weighting: Optional[str] = None,
) -> float:
    """Calculate unweighted, linear-weighted, or quadratic-weighted kappa."""
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Cohen kappa inputs must have equal length.")
    if len(ratings_a) == 0:
        return float("nan")

    grade_to_index = {grade: index for index, grade in enumerate(grades)}
    size = len(grades)
    observed = np.zeros((size, size), dtype=float)

    for value_a, value_b in zip(ratings_a, ratings_b):
        observed[grade_to_index[int(value_a)], grade_to_index[int(value_b)]] += 1.0

    total = observed.sum()
    row_marginals = observed.sum(axis=1)
    column_marginals = observed.sum(axis=0)
    expected = np.outer(row_marginals, column_marginals) / total

    if weighting is None:
        disagreement = np.ones((size, size), dtype=float) - np.eye(size)
    else:
        if weighting not in {"linear", "quadratic"}:
            raise ValueError(f"Unknown kappa weighting: {weighting}")
        max_distance = max(size - 1, 1)
        disagreement = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                distance = abs(i - j) / max_distance
                disagreement[i, j] = distance if weighting == "linear" else distance**2

    observed_disagreement = float((observed * disagreement).sum())
    expected_disagreement = float((expected * disagreement).sum())

    if expected_disagreement == 0:
        return float("nan")

    return 1.0 - observed_disagreement / expected_disagreement


def pair_descriptive_metrics(
    ratings_a: Sequence[int],
    ratings_b: Sequence[int],
) -> Dict[str, float]:
    if len(ratings_a) == 0:
        return {
            "exact_agreement": float("nan"),
            "within_one_grade_agreement": float("nan"),
            "mean_absolute_difference": float("nan"),
        }

    array_a = np.asarray(ratings_a, dtype=float)
    array_b = np.asarray(ratings_b, dtype=float)
    differences = np.abs(array_a - array_b)
    return {
        "exact_agreement": float(np.mean(differences == 0)),
        "within_one_grade_agreement": float(np.mean(differences <= 1)),
        "mean_absolute_difference": float(np.mean(differences)),
    }


def build_pairwise_metrics(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    config: AgreementConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    repeated_ids = metadata.loc[
        metadata["assignment_type"].map(normalize_text).str.lower().isin(
            REPEATED_ASSIGNMENT_TYPES
        ),
        "row_id",
    ]
    repeated_matrix = matrix.loc[repeated_ids]

    metric_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []

    for annotator_a, annotator_b in itertools.combinations(config.annotators, 2):
        pair = repeated_matrix[[annotator_a, annotator_b]].dropna()
        n_overlap = len(pair)
        values_a = pair[annotator_a].astype(int).tolist()
        values_b = pair[annotator_b].astype(int).tolist()
        descriptive = pair_descriptive_metrics(values_a, values_b)

        metric_rows.append(
            {
                "annotator_a": annotator_a,
                "annotator_b": annotator_b,
                "n_overlap": n_overlap,
                **descriptive,
                "cohen_kappa": cohen_kappa(
                    values_a, values_b, config.grades, weighting=None
                ),
                "linear_weighted_kappa": cohen_kappa(
                    values_a, values_b, config.grades, weighting="linear"
                ),
                "quadratic_weighted_kappa": cohen_kappa(
                    values_a, values_b, config.grades, weighting="quadratic"
                ),
            }
        )

        for grade_a in config.grades:
            for grade_b in config.grades:
                count = int(
                    ((pair[annotator_a] == grade_a) & (pair[annotator_b] == grade_b)).sum()
                )
                confusion_rows.append(
                    {
                        "annotator_a": annotator_a,
                        "annotator_b": annotator_b,
                        "grade_a": grade_a,
                        "grade_b": grade_b,
                        "count": count,
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def coincidence_matrix(
    rating_matrix: np.ndarray,
    grades: Sequence[int],
) -> np.ndarray:
    grade_to_index = {grade: index for index, grade in enumerate(grades)}
    coincidence = np.zeros((len(grades), len(grades)), dtype=float)

    for row in rating_matrix:
        valid = [int(value) for value in row if not np.isnan(value)]
        n_ratings = len(valid)
        if n_ratings < 2:
            continue

        counts = np.zeros(len(grades), dtype=float)
        for value in valid:
            counts[grade_to_index[value]] += 1.0

        denominator = n_ratings - 1
        for i in range(len(grades)):
            for j in range(len(grades)):
                if i == j:
                    coincidence[i, j] += counts[i] * (counts[i] - 1) / denominator
                else:
                    coincidence[i, j] += counts[i] * counts[j] / denominator

    return coincidence


def krippendorff_alpha(
    rating_matrix: np.ndarray,
    grades: Sequence[int],
    level: str,
) -> float:
    """Calculate Krippendorff's alpha for nominal or ordinal data."""
    coincidence = coincidence_matrix(rating_matrix, grades)
    total = float(coincidence.sum())
    if total <= 1:
        return float("nan")

    marginals = coincidence.sum(axis=0)
    expected = np.zeros_like(coincidence)
    for i in range(len(grades)):
        for j in range(len(grades)):
            if i == j:
                expected[i, j] = marginals[i] * (marginals[i] - 1) / (total - 1)
            else:
                expected[i, j] = marginals[i] * marginals[j] / (total - 1)

    distances = np.zeros_like(coincidence)
    if level == "nominal":
        distances = np.ones_like(coincidence) - np.eye(len(grades))
    elif level == "ordinal":
        for i in range(len(grades)):
            for j in range(len(grades)):
                if i == j:
                    continue
                lower, upper = sorted((i, j))
                cumulative = marginals[lower : upper + 1].sum()
                interval_mass = cumulative - (marginals[lower] + marginals[upper]) / 2.0
                distances[i, j] = interval_mass**2
    else:
        raise ValueError(f"Unsupported Krippendorff level: {level}")

    observed_disagreement = float((coincidence * distances).sum())
    expected_disagreement = float((expected * distances).sum())
    if expected_disagreement == 0:
        return float("nan")

    return 1.0 - observed_disagreement / expected_disagreement


def pooled_pairwise_comparisons(rating_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    left: List[int] = []
    right: List[int] = []

    for _, row in rating_matrix.iterrows():
        values = [int(value) for value in row.dropna().tolist()]
        for value_a, value_b in itertools.combinations(values, 2):
            left.append(value_a)
            right.append(value_b)

    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def fleiss_kappa_complete(rating_matrix: pd.DataFrame, grades: Sequence[int]) -> float:
    complete = rating_matrix.dropna()
    n_items, n_raters = complete.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")

    category_counts = np.zeros((n_items, len(grades)), dtype=float)
    grade_to_index = {grade: index for index, grade in enumerate(grades)}

    for item_index, (_, row) in enumerate(complete.iterrows()):
        for value in row.astype(int):
            category_counts[item_index, grade_to_index[int(value)]] += 1.0

    item_agreement = (
        (category_counts**2).sum(axis=1) - n_raters
    ) / (n_raters * (n_raters - 1))
    observed = float(item_agreement.mean())
    category_proportions = category_counts.sum(axis=0) / (n_items * n_raters)
    expected = float((category_proportions**2).sum())

    return safe_divide(observed - expected, 1.0 - expected)


def icc_two_way_random_absolute(
    rating_matrix: pd.DataFrame,
) -> Tuple[float, float]:
    """Return ICC(2,1) and ICC(2,k) for a complete item-by-rater matrix."""
    complete = rating_matrix.dropna()
    n_items, n_raters = complete.shape
    if n_items < 2 or n_raters < 2:
        return float("nan"), float("nan")

    values = complete.to_numpy(dtype=float)
    grand_mean = values.mean()
    item_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)

    ss_items = n_raters * float(((item_means - grand_mean) ** 2).sum())
    ss_raters = n_items * float(((rater_means - grand_mean) ** 2).sum())
    residuals = values - item_means[:, None] - rater_means[None, :] + grand_mean
    ss_error = float((residuals**2).sum())

    ms_items = ss_items / (n_items - 1)
    ms_raters = ss_raters / (n_raters - 1)
    ms_error = ss_error / ((n_items - 1) * (n_raters - 1))

    denominator_single = (
        ms_items
        + (n_raters - 1) * ms_error
        + n_raters * (ms_raters - ms_error) / n_items
    )
    denominator_average = ms_items + (ms_raters - ms_error) / n_items

    icc_2_1 = safe_divide(ms_items - ms_error, denominator_single)
    icc_2_k = safe_divide(ms_items - ms_error, denominator_average)
    return icc_2_1, icc_2_k


def build_overall_metrics(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    config: AgreementConfig,
) -> pd.DataFrame:
    metadata_indexed = metadata.set_index("row_id", drop=False)
    output_rows: List[Dict[str, object]] = []

    scopes = [
        ("all_repeated_rows", REPEATED_ASSIGNMENT_TYPES),
        ("double_rows", {"double"}),
        ("shared_rows", {"shared"}),
    ]

    for scope_name, assignment_types in scopes:
        item_ids = metadata_indexed.loc[
            metadata_indexed["assignment_type"].map(normalize_text).str.lower().isin(
                assignment_types
            ),
            "row_id",
        ]
        scoped_matrix = matrix.loc[item_ids]
        valid_counts = scoped_matrix.notna().sum(axis=1)
        usable = scoped_matrix.loc[valid_counts >= 2]
        pooled_a, pooled_b = pooled_pairwise_comparisons(usable)
        descriptive = pair_descriptive_metrics(pooled_a, pooled_b)

        common = {
            "scope": scope_name,
            "n_items_total": len(scoped_matrix),
            "n_items_with_two_or_more_ratings": len(usable),
            "n_valid_ratings": int(scoped_matrix.notna().sum().sum()),
            "n_pairwise_comparisons": len(pooled_a),
        }

        metrics = {
            "pooled_exact_agreement": descriptive["exact_agreement"],
            "pooled_within_one_grade_agreement": descriptive[
                "within_one_grade_agreement"
            ],
            "pooled_mean_absolute_difference": descriptive[
                "mean_absolute_difference"
            ],
            "krippendorff_alpha_nominal": krippendorff_alpha(
                usable.to_numpy(dtype=float), config.grades, level="nominal"
            ),
            "krippendorff_alpha_ordinal": krippendorff_alpha(
                usable.to_numpy(dtype=float), config.grades, level="ordinal"
            ),
        }

        for metric_name, value in metrics.items():
            output_rows.append(
                {
                    **common,
                    "metric": metric_name,
                    "value": value,
                    "notes": "",
                }
            )

    shared_ids = metadata_indexed.loc[
        metadata_indexed["assignment_type"].map(normalize_text).str.lower() == "shared",
        "row_id",
    ]
    shared_matrix = matrix.loc[shared_ids]
    complete_shared = shared_matrix.dropna()
    icc_2_1, icc_2_k = icc_two_way_random_absolute(complete_shared)

    supplementary = [
        (
            "fleiss_kappa_nominal",
            fleiss_kappa_complete(complete_shared, config.grades),
            "Calculated on complete shared rows only.",
        ),
        (
            "icc_2_1_absolute_agreement",
            icc_2_1,
            "Supplementary; assumes the ordinal grades can be treated as approximately interval-scaled.",
        ),
        (
            "icc_2_k_absolute_agreement",
            icc_2_k,
            "Supplementary reliability of the mean of all annotators; complete shared rows only.",
        ),
    ]

    for metric_name, value, notes in supplementary:
        output_rows.append(
            {
                "scope": "complete_shared_rows",
                "n_items_total": len(shared_matrix),
                "n_items_with_two_or_more_ratings": len(complete_shared),
                "n_valid_ratings": int(complete_shared.notna().sum().sum()),
                "n_pairwise_comparisons": int(
                    len(complete_shared)
                    * math.comb(len(config.annotators), 2)
                    if len(config.annotators) >= 2
                    else 0
                ),
                "metric": metric_name,
                "value": value,
                "notes": notes,
            }
        )

    return pd.DataFrame(output_rows)


# -----------------------------------------------------------------------------
# Diagnostics and tables
# -----------------------------------------------------------------------------


def build_item_disagreement(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    config: AgreementConfig,
) -> pd.DataFrame:
    metadata = frame.set_index("row_id", drop=False)
    rows: List[Dict[str, object]] = []

    for row_id, ratings in matrix.iterrows():
        assignment_type = normalize_text(metadata.at[row_id, "assignment_type"]).lower()
        if assignment_type not in REPEATED_ASSIGNMENT_TYPES:
            continue

        valid = ratings.dropna().astype(int)
        if len(valid) < 2:
            continue

        values = valid.to_numpy(dtype=float)
        pair_a, pair_b = zip(*itertools.combinations(values, 2))
        descriptive = pair_descriptive_metrics(pair_a, pair_b)

        output_row: Dict[str, object] = {
            "row_id": row_id,
            "assignment_type": assignment_type,
            "assigned_annotators": normalize_text(
                metadata.at[row_id, "assigned_annotators"]
            ),
            "answered_by": "|".join(valid.index.tolist()),
            "n_valid_grades": len(valid),
            "mean_grade": float(values.mean()),
            "median_grade": float(np.median(values)),
            "minimum_grade": int(values.min()),
            "maximum_grade": int(values.max()),
            "grade_range": int(values.max() - values.min()),
            "grade_standard_deviation": float(values.std(ddof=0)),
            "unanimous": int(len(set(values.tolist())) == 1),
            **descriptive,
        }

        for optional_column in (
            "query_id",
            "query_title",
            "candidate_id",
            "candidate_title",
            "human_resolution_status",
            "human_resolution_note",
        ):
            if optional_column in metadata.columns:
                output_row[optional_column] = normalize_text(
                    metadata.at[row_id, optional_column]
                )

        rows.append(output_row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            [
                "grade_range",
                "grade_standard_deviation",
                "mean_absolute_difference",
                "row_id",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return result


def build_grade_distributions(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    config: AgreementConfig,
) -> pd.DataFrame:
    metadata = frame.set_index("row_id", drop=False)
    repeated_mask = metadata["assignment_type"].map(normalize_text).str.lower().isin(
        REPEATED_ASSIGNMENT_TYPES
    )
    repeated_ids = metadata.loc[repeated_mask, "row_id"]

    rows: List[Dict[str, object]] = []
    for annotator in config.annotators:
        for scope, series in (
            ("all_assigned_rows", matrix[annotator].dropna()),
            ("repeated_assignment_rows", matrix.loc[repeated_ids, annotator].dropna()),
        ):
            counts = series.astype(int).value_counts().to_dict()
            total = len(series)
            row: Dict[str, object] = {
                "annotator": annotator,
                "scope": scope,
                "n_valid_grades": total,
                "mean_grade": float(series.mean()) if total else float("nan"),
                "median_grade": float(series.median()) if total else float("nan"),
            }
            for grade in config.grades:
                count = int(counts.get(grade, 0))
                row[f"grade_{grade}_count"] = count
                row[f"grade_{grade}_proportion"] = (
                    count / total if total else float("nan")
                )
            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------


def configure_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 9,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def pairwise_metric_matrix(
    pairwise: pd.DataFrame,
    annotators: Sequence[str],
    metric: str,
    diagonal_value: Optional[float],
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=list(annotators),
        columns=list(annotators),
        dtype=float,
    )
    if diagonal_value is not None:
        np.fill_diagonal(result.values, diagonal_value)

    for row in pairwise.itertuples(index=False):
        value = getattr(row, metric)
        result.at[row.annotator_a, row.annotator_b] = value
        result.at[row.annotator_b, row.annotator_a] = value

    return result


def save_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    value_format: str,
    dpi: int,
) -> None:
    configure_plot_defaults()
    size = max(6.0, 0.75 * len(matrix.columns) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)

    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title, pad=12)

    values = matrix.to_numpy(dtype=float)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isnan(value):
                label = "NA"
            else:
                label = format(value, value_format)
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_heatmaps(
    pairwise: pd.DataFrame,
    config: AgreementConfig,
) -> None:
    exact_matrix = pairwise_metric_matrix(
        pairwise,
        config.annotators,
        metric="exact_agreement",
        diagonal_value=1.0,
    )
    weighted_kappa_matrix = pairwise_metric_matrix(
        pairwise,
        config.annotators,
        metric="linear_weighted_kappa",
        diagonal_value=1.0,
    )
    overlap_matrix = pairwise_metric_matrix(
        pairwise,
        config.annotators,
        metric="n_overlap",
        diagonal_value=None,
    )

    save_heatmap(
        exact_matrix,
        config.output_dir / "pairwise_exact_agreement_heatmap.png",
        title="Pairwise exact agreement",
        value_format=".2f",
        dpi=config.dpi,
    )
    save_heatmap(
        weighted_kappa_matrix,
        config.output_dir / "pairwise_linear_weighted_kappa_heatmap.png",
        title="Pairwise linear weighted Cohen's kappa",
        value_format=".2f",
        dpi=config.dpi,
    )
    save_heatmap(
        overlap_matrix,
        config.output_dir / "pairwise_overlap_heatmap.png",
        title="Number of overlapping repeated-assignment ratings",
        value_format=".0f",
        dpi=config.dpi,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def format_metric(value: object, decimals: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{decimals}f}"


def metric_lookup(
    overall: pd.DataFrame,
    scope: str,
    metric: str,
) -> str:
    matches = overall[(overall["scope"] == scope) & (overall["metric"] == metric)]
    if matches.empty:
        return "NA"
    return format_metric(matches.iloc[0]["value"])


def write_overview(
    output_path: Path,
    config: AgreementConfig,
    diagnostics: InputDiagnostics,
    warnings: Sequence[str],
    overall: pd.DataFrame,
    pairwise: pd.DataFrame,
    item_disagreement: pd.DataFrame,
) -> None:
    pairwise_with_overlap = pairwise[pairwise["n_overlap"] > 0]
    mean_pairwise_weighted = (
        pairwise_with_overlap["linear_weighted_kappa"].mean()
        if not pairwise_with_overlap.empty
        else float("nan")
    )
    mean_pairwise_exact = (
        pairwise_with_overlap["exact_agreement"].mean()
        if not pairwise_with_overlap.empty
        else float("nan")
    )

    lines = [
        "Inter-Annotator Agreement Overview",
        "==================================",
        "",
        "Input",
        "-----",
        f"Annotation sheet: {config.input_path}",
        f"Detected/selected annotators: {', '.join(config.annotators)}",
        f"Grade scale: {', '.join(map(str, config.grades))}",
        f"Total rows: {diagnostics.total_rows:,}",
        f"Repeated-assignment rows: {diagnostics.repeated_rows:,}",
        f"Single-assignment rows excluded from agreement: {diagnostics.single_rows:,}",
        f"Other assignment-type rows: {diagnostics.other_assignment_rows:,}",
        "",
        "Why these metrics are included",
        "------------------------------",
        "Exact agreement: directly reports how often annotators selected the same grade.",
        "Agreement within one grade: useful for a four-level ordinal scale where adjacent disagreements are less severe.",
        "Mean absolute difference: reports the average size of ordinal disagreements.",
        "Cohen's kappa: chance-corrected pairwise agreement without grade-distance weighting.",
        "Linear weighted kappa: primary pairwise chance-corrected metric because grades 0-3 are ordinal.",
        "Quadratic weighted kappa: supplementary pairwise metric that penalizes large disagreements more strongly.",
        "Krippendorff's alpha ordinal: primary overall metric because it supports multiple annotators and missing ratings.",
        "Krippendorff's alpha nominal: supplementary overall metric that ignores grade ordering.",
        "Fleiss' kappa: supplementary multi-rater nominal agreement on complete shared rows only.",
        "ICC(2,1) and ICC(2,k): supplementary absolute-agreement reliability metrics on complete shared rows; they assume an approximately interval scale.",
        "",
        "Headline results",
        "----------------",
        f"Krippendorff alpha, ordinal, all repeated rows: {metric_lookup(overall, 'all_repeated_rows', 'krippendorff_alpha_ordinal')}",
        f"Krippendorff alpha, nominal, all repeated rows: {metric_lookup(overall, 'all_repeated_rows', 'krippendorff_alpha_nominal')}",
        f"Pooled exact agreement, all repeated rows: {metric_lookup(overall, 'all_repeated_rows', 'pooled_exact_agreement')}",
        f"Pooled agreement within one grade: {metric_lookup(overall, 'all_repeated_rows', 'pooled_within_one_grade_agreement')}",
        f"Pooled mean absolute difference: {metric_lookup(overall, 'all_repeated_rows', 'pooled_mean_absolute_difference')}",
        f"Mean pairwise linear weighted kappa across pairs with overlap: {format_metric(mean_pairwise_weighted)}",
        f"Mean pairwise exact agreement across pairs with overlap: {format_metric(mean_pairwise_exact)}",
        f"Fleiss kappa, complete shared rows: {metric_lookup(overall, 'complete_shared_rows', 'fleiss_kappa_nominal')}",
        f"ICC(2,1), complete shared rows: {metric_lookup(overall, 'complete_shared_rows', 'icc_2_1_absolute_agreement')}",
        f"ICC(2,k), complete shared rows: {metric_lookup(overall, 'complete_shared_rows', 'icc_2_k_absolute_agreement')}",
        "",
        "Interpretation guidance",
        "-----------------------",
        "Report exact agreement and ordinal Krippendorff alpha as the main corpus-level results.",
        "Report linear weighted Cohen's kappa for named annotator pairs.",
        "Use quadratic weighted kappa, Fleiss' kappa, and ICC as sensitivity or supplementary analyses.",
        "Do not interpret a high correlation as agreement; association metrics are therefore intentionally omitted.",
        "Kappa-family metrics can be sensitive to category prevalence, so inspect grade distributions and raw agreement alongside them.",
        "",
        "Data-quality diagnostics",
        "------------------------",
        f"Invalid non-empty grade cells: {diagnostics.invalid_grade_cells:,}",
        f"Ignored unassigned non-empty grade cells: {diagnostics.unassigned_nonempty_grade_cells:,}",
        "Repeated rows with fewer than two valid grades: "
        f"{diagnostics.repeated_rows_with_fewer_than_two_valid_grades:,}",
        "",
        "Most disputed repeated-assignment rows",
        "--------------------------------------",
    ]

    if item_disagreement.empty:
        lines.append("No repeated-assignment row had at least two valid grades.")
    else:
        for position, row in enumerate(item_disagreement.head(20).itertuples(index=False), start=1):
            lines.append(
                f"{position}. row_id={row.row_id} | type={row.assignment_type} | "
                f"n={row.n_valid_grades} | range={row.grade_range} | "
                f"MAD={row.mean_absolute_difference:.3f} | answered_by={row.answered_by}"
            )

    if warnings:
        lines.extend(["", "Warnings", "--------"])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "Generated files",
            "---------------",
            "agreement_overview.txt",
            "overall_agreement_metrics.csv",
            "pairwise_agreement_metrics.csv",
            "pairwise_confusion_matrix.csv",
            "annotator_grade_distributions.csv",
            "item_disagreement.csv",
            "pairwise_exact_agreement_heatmap.png",
            "pairwise_linear_weighted_kappa_heatmap.png",
            "pairwise_overlap_heatmap.png",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    diagnostics: InputDiagnostics,
    warnings: Sequence[str],
    config: AgreementConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    pairwise, confusion = build_pairwise_metrics(matrix, frame, config)
    overall = build_overall_metrics(matrix, frame, config)
    distributions = build_grade_distributions(frame, matrix, config)
    item_disagreement = build_item_disagreement(frame, matrix, config)

    pairwise.to_csv(
        config.output_dir / "pairwise_agreement_metrics.csv",
        index=False,
    )
    confusion.to_csv(
        config.output_dir / "pairwise_confusion_matrix.csv",
        index=False,
    )
    overall.to_csv(
        config.output_dir / "overall_agreement_metrics.csv",
        index=False,
    )
    distributions.to_csv(
        config.output_dir / "annotator_grade_distributions.csv",
        index=False,
    )
    item_disagreement.to_csv(
        config.output_dir / "item_disagreement.csv",
        index=False,
    )

    save_pairwise_heatmaps(pairwise, config)
    write_overview(
        config.output_dir / "agreement_overview.txt",
        config,
        diagnostics,
        warnings,
        overall,
        pairwise,
        item_disagreement,
    )


def print_summary(config: AgreementConfig, diagnostics: InputDiagnostics) -> None:
    print("Inter-annotator agreement analysis completed.")
    print(f"Input rows: {diagnostics.total_rows:,}")
    print(f"Repeated-assignment rows: {diagnostics.repeated_rows:,}")
    print(f"Annotators: {len(config.annotators)}")
    print(f"Output directory: {config.output_dir}")
    for name in (
        "agreement_overview.txt",
        "overall_agreement_metrics.csv",
        "pairwise_agreement_metrics.csv",
        "pairwise_confusion_matrix.csv",
        "annotator_grade_distributions.csv",
        "item_disagreement.csv",
        "pairwise_exact_agreement_heatmap.png",
        "pairwise_linear_weighted_kappa_heatmap.png",
        "pairwise_overlap_heatmap.png",
    ):
        print(f"Wrote: {config.output_dir / name}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
ANNOTATION_DIRECTORY = Path("data/annotations")
INPUT_FILENAME = "annotation_sheet_recompiled.csv"
OUTPUT_SUBDIRECTORY = "inter_annotator_agreement"
ANNOTATORS: tuple[str, ...] | None = None
GRADES: tuple[int, ...] = (0, 1, 2, 3)
DPI = DEFAULT_DPI
STRICT_GRADES = True


def validate_ide_configuration() -> None:
    if DPI <= 0:
        raise ValueError("DPI must be greater than zero.")
    if len(GRADES) < 2 or len(set(GRADES)) != len(GRADES):
        raise ValueError("GRADES must contain at least two unique values.")
    if tuple(sorted(GRADES)) != GRADES:
        raise ValueError("GRADES must be in ascending ordinal order.")


def main() -> int:
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    eval_dir = resolve_project_path(project_root, ANNOTATION_DIRECTORY)
    input_path = eval_dir / INPUT_FILENAME
    output_dir = eval_dir / OUTPUT_SUBDIRECTORY

    try:
        validate_ide_configuration()
        frame = load_annotation_sheet(input_path)
        annotators = validate_annotators(frame, ANNOTATORS)
        config = AgreementConfig(
            input_path=input_path,
            output_dir=output_dir,
            annotators=annotators,
            grades=GRADES,
            dpi=DPI,
            strict_grades=STRICT_GRADES,
        )
        matrix, diagnostics, warnings = build_rating_matrix(frame, config)
        write_outputs(frame, matrix, diagnostics, warnings, config)
        print_summary(config, diagnostics)
        return 0
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
