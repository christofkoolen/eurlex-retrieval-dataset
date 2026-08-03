# -*- coding: utf-8 -*-
"""Generate the descriptive statistics and figures for EUR-LexRD.

Edit the IDE configuration near the end of the file and run the complete
script. All default paths are relative to the repository root.
"""

from types import SimpleNamespace
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import pandas as pd

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


RANDOM_SEED = 123
DEFAULT_DPI = 300
DEFAULT_TOP_K = 10
DEFAULT_KEYWORD_TOP_K = 25
DEFAULT_HIST_BINS = 50

REQUIRED_OUTPUT_FILES: Tuple[str, ...] = (
    "cases_per_year_vs_length.png",
    "cases_per_year_vs_length.txt",
    "citation_insights.txt",
    "costs.txt",
    "general_insights.txt",
    "incoming_citations_histogram.png",
    "keyword_counts.csv",
    "keyword_frequency_top25.png",
    "keyword_overview.txt",
    "outgoing_citations_histogram.png",
)

ECLI_RE = re.compile(
    r"\b(?:ECLI:)?(EU:(?:C|T|F):\d{4}:\d{1,6})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaseColumns:
    celex: str
    date: str
    raw_content: str


@dataclass
class CitationBuildStats:
    files_scanned: int = 0
    files_failed: int = 0
    files_without_source_id: int = 0
    raw_edges_found: int = 0
    unique_edges_kept: int = 0


@dataclass
class KeywordBuildStats:
    files_scanned: int = 0
    files_failed: int = 0
    files_with_keywords: int = 0
    total_keyword_instances: int = 0
    unique_keywords: int = 0


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def iter_json_files(root: Path) -> Iterable[Path]:
    """Yield JSON files from one file or recursively from one directory."""
    if root.is_file():
        if root.suffix.lower() == ".json":
            yield root
        return

    if root.is_dir():
        yield from sorted(root.rglob("*.json"))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load one JSON object, returning None when the file is invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[skip] Could not read JSON {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print(f"[skip] JSON root is not an object: {path}", file=sys.stderr)
        return None

    return data


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_number(value: float | int, decimals: int = 2) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# -----------------------------------------------------------------------------
# Citation extraction and enrichment
# -----------------------------------------------------------------------------


def canonicalize_ecli(value: Any) -> Optional[str]:
    """Return a canonical ECLI identifier when one is present."""
    if not isinstance(value, str):
        return None
    match = ECLI_RE.search(value)
    return f"ECLI:{match.group(1).upper()}" if match else None


def build_case_id(data: Dict[str, Any]) -> Optional[str]:
    """Prefer the case ECLI; fall back to the case number."""
    metadata = data.get("case_metadata")
    if not isinstance(metadata, dict):
        return None

    own_ecli = canonicalize_ecli(metadata.get("ecli"))
    if own_ecli:
        return own_ecli

    case_number = metadata.get("case_number")
    if isinstance(case_number, str) and case_number.strip():
        return case_number.strip()

    return None


def extract_eclis_with_locations(
    case_segments: Any,
    exclude: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, List[Dict[str, Optional[str]]]]]:
    """Extract ordered ECLIs and segment-level locations from case_segments."""
    excluded = exclude or set()
    ordered: List[str] = []
    seen: Set[str] = set()
    locations: Dict[str, List[Dict[str, Optional[str]]]] = {}

    if not isinstance(case_segments, dict):
        return ordered, locations

    for segment_name, segment_value in case_segments.items():
        segment_label = segment_name if isinstance(segment_name, str) else None

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
            elif isinstance(node, str):
                for match in ECLI_RE.finditer(node):
                    citation = f"ECLI:{match.group(1).upper()}"
                    if citation in excluded:
                        continue
                    if citation not in seen:
                        seen.add(citation)
                        ordered.append(citation)
                    location = {"segment": segment_label}
                    if location not in locations.setdefault(citation, []):
                        locations[citation].append(location)

        walk(segment_value)

    return ordered, locations


def add_citations_field(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with normalized citation list, details, and directed edges."""
    updated = dict(data)
    metadata = data.get("case_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    own_ecli = canonicalize_ecli(metadata.get("ecli"))
    excluded = {own_ecli} if own_ecli else set()

    ordered, locations = extract_eclis_with_locations(
        data.get("case_segments", {}),
        exclude=excluded,
    )

    existing = data.get("citations")
    existing_list = existing.get("list") if isinstance(existing, dict) else None

    final_list: List[str] = []
    seen: Set[str] = set()

    if isinstance(existing_list, list):
        for item in existing_list:
            citation = canonicalize_ecli(item)
            if citation and citation not in excluded and citation not in seen:
                seen.add(citation)
                final_list.append(citation)

    for citation in ordered:
        if citation not in seen:
            seen.add(citation)
            final_list.append(citation)

    source_id = build_case_id(data)
    edges = (
        [{"source": source_id, "target": target} for target in final_list]
        if source_id
        else []
    )

    updated["citations"] = {
        "list": final_list,
        "detailed": [
            {
                "type": "ECLI",
                "citation": citation,
                "locations": locations.get(citation, []),
            }
            for citation in final_list
        ],
        "edges": edges,
    }

    updated.pop("case_citations", None)
    updated.pop("case_citations_detailed", None)
    updated.pop("citation_edges", None)
    return updated


def write_enriched_json_corpus(input_path: Path, output_dir: Path) -> int:
    """Write normalized citation-enriched JSON files, preserving relative paths."""
    require_existing_path(input_path, "JSON input")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_path in iter_json_files(input_path):
        data = load_json(json_path)
        if data is None:
            continue

        enriched = add_citations_field(data)
        relative = (
            json_path.relative_to(input_path)
            if input_path.is_dir()
            else Path(json_path.name)
        )
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1

    return count


def extract_edges_from_data(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract normalized edges, deriving them from case segments when needed."""
    normalized = add_citations_field(data)
    citations = normalized.get("citations")
    if not isinstance(citations, dict):
        return []

    raw_edges = citations.get("edges")
    if not isinstance(raw_edges, list):
        return []

    edges: List[Tuple[str, str]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        source = source.strip()
        target = target.strip()
        if source and target and source != target:
            edges.append((source, target))

    return edges


def build_citation_edges(json_path: Path) -> Tuple[List[Tuple[str, str]], CitationBuildStats]:
    """Build a globally deduplicated citation edge list from the JSON corpus."""
    require_existing_path(json_path, "JSON input")

    stats = CitationBuildStats()
    unique_edges: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for path in iter_json_files(json_path):
        stats.files_scanned += 1
        data = load_json(path)
        if data is None:
            stats.files_failed += 1
            continue

        if build_case_id(data) is None:
            stats.files_without_source_id += 1

        edges = extract_edges_from_data(data)
        stats.raw_edges_found += len(edges)

        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                unique_edges.append(edge)

    stats.unique_edges_kept = len(unique_edges)
    return unique_edges, stats


def write_citation_graph_tsv(path: Path, edges: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "target"])
        writer.writerows(edges)


# -----------------------------------------------------------------------------
# Citation facts and figures
# -----------------------------------------------------------------------------


def citation_dataframe(edges: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(edges, columns=["source", "target"])
    if frame.empty:
        raise ValueError("No citation edges were found in the JSON corpus.")

    frame["source"] = frame["source"].astype(str).str.strip()
    frame["target"] = frame["target"].astype(str).str.strip()
    frame = frame[(frame["source"] != "") & (frame["target"] != "")]
    frame = frame[frame["source"] != frame["target"]]
    frame = frame.drop_duplicates().reset_index(drop=True)

    if frame.empty:
        raise ValueError("No valid citation edges remained after cleaning.")
    return frame


def configure_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 10,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def save_degree_histogram(
    values: Sequence[int],
    output_path: Path,
    xlabel: str,
    title: str,
    bins: int,
    dpi: int,
    log_y: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cases")
    ax.set_title(title, pad=10)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_optional_network_figure(
    graph: nx.DiGraph,
    output_path: Path,
    top_n: int,
    label_top_n: int,
    dpi: int,
) -> None:
    top_nodes = sorted(graph.in_degree, key=lambda item: item[1], reverse=True)[:top_n]
    node_ids = {node for node, _ in top_nodes}
    subgraph = graph.subgraph(node_ids).copy()
    subgraph.remove_nodes_from(list(nx.isolates(subgraph)))

    if subgraph.number_of_nodes() == 0:
        raise ValueError("The filtered citation network is empty.")

    node_sizes = [120 + graph.in_degree(node) * 18 for node in subgraph.nodes()]
    positions = nx.spring_layout(subgraph, seed=RANDOM_SEED, k=0.7)

    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_edges(
        subgraph,
        positions,
        ax=ax,
        alpha=0.22,
        arrows=True,
        arrowsize=10,
        width=0.7,
        connectionstyle="arc3,rad=0.04",
    )
    nx.draw_networkx_nodes(
        subgraph,
        positions,
        ax=ax,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.6,
        edgecolors="black",
    )

    label_candidates = sorted(
        subgraph.nodes(),
        key=lambda node: graph.in_degree(node),
        reverse=True,
    )[:label_top_n]
    labels = {
        node: node if len(node) <= 24 else f"{node[:23]}…"
        for node in label_candidates
    }
    nx.draw_networkx_labels(
        subgraph,
        positions,
        labels=labels,
        font_size=7,
        font_weight="bold",
        ax=ax,
    )
    ax.set_title(f"Core citation subgraph of the {top_n} most cited CJEU cases", pad=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_citation_insights(
    output_path: Path,
    frame: pd.DataFrame,
    build_stats: CitationBuildStats,
    top_k: int,
) -> None:
    graph = nx.from_pandas_edgelist(
        frame,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )

    in_degree = (
        frame["target"]
        .value_counts()
        .rename_axis("case_id")
        .reset_index(name="in_degree")
    )
    out_degree = (
        frame["source"]
        .value_counts()
        .rename_axis("case_id")
        .reset_index(name="out_degree")
    )

    in_values = [degree for _, degree in graph.in_degree()]
    out_values = [degree for _, degree in graph.out_degree()]
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0

    lines = [
        "CJEU Citation Insights",
        "======================",
        "",
        "Corpus processing",
        "-----------------",
        f"JSON files scanned: {build_stats.files_scanned:,}",
        f"JSON files skipped after read errors: {build_stats.files_failed:,}",
        f"Files without a usable source case identifier: {build_stats.files_without_source_id:,}",
        f"Raw citation edges found: {build_stats.raw_edges_found:,}",
        f"Unique citation edges retained: {len(frame):,}",
        "",
        "Citation graph",
        "--------------",
        f"Nodes (cases appearing in the graph): {graph.number_of_nodes():,}",
        f"Edges (unique directed citation relations): {graph.number_of_edges():,}",
        f"Graph density: {density:.8f}",
        f"Mean in-degree: {pd.Series(in_values).mean():.4f}",
        f"Median in-degree: {pd.Series(in_values).median():.4f}",
        f"Maximum in-degree: {max(in_values):,}",
        f"Mean out-degree: {pd.Series(out_values).mean():.4f}",
        f"Median out-degree: {pd.Series(out_values).median():.4f}",
        f"Maximum out-degree: {max(out_values):,}",
        "",
        f"Top {top_k} most cited cases",
        "-" * (len(f"Top {top_k} most cited cases")),
    ]

    for index, row in in_degree.head(top_k).iterrows():
        lines.append(f"{index + 1}. {row['case_id']} | incoming citations: {int(row['in_degree']):,}")

    lines.extend(
        [
            "",
            f"Top {top_k} most citing cases",
            "-" * (len(f"Top {top_k} most citing cases")),
        ]
    )
    for index, row in out_degree.head(top_k).iterrows():
        lines.append(f"{index + 1}. {row['case_id']} | outgoing citations: {int(row['out_degree']):,}")

    write_text(output_path, "\n".join(lines) + "\n")


def generate_citation_outputs(
    json_dir: Path,
    output_dir: Path,
    top_k: int,
    hist_bins: int,
    dpi: int,
    log_y: bool,
    citation_graph_tsv: Optional[Path] = None,
    network_output: Optional[Path] = None,
    network_top_n: int = 75,
    network_label_top_n: int = 20,
) -> None:
    edges, build_stats = build_citation_edges(json_dir)
    frame = citation_dataframe(edges)

    if citation_graph_tsv is not None:
        write_citation_graph_tsv(citation_graph_tsv, list(frame.itertuples(index=False, name=None)))

    graph = nx.from_pandas_edgelist(
        frame,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )

    configure_plot_defaults()
    save_degree_histogram(
        [degree for _, degree in graph.in_degree()],
        output_dir / "incoming_citations_histogram.png",
        xlabel="Incoming citations per case",
        title="Distribution of incoming citations in the CJEU corpus",
        bins=hist_bins,
        dpi=dpi,
        log_y=log_y,
    )
    save_degree_histogram(
        [degree for _, degree in graph.out_degree()],
        output_dir / "outgoing_citations_histogram.png",
        xlabel="Outgoing citations per case",
        title="Distribution of outgoing citations in the CJEU corpus",
        bins=hist_bins,
        dpi=dpi,
        log_y=log_y,
    )
    write_citation_insights(
        output_dir / "citation_insights.txt",
        frame,
        build_stats,
        top_k=top_k,
    )

    if network_output is not None:
        save_optional_network_figure(
            graph,
            output_path=network_output,
            top_n=network_top_n,
            label_top_n=network_label_top_n,
            dpi=dpi,
        )


# -----------------------------------------------------------------------------
# Case volume, length, and general insights
# -----------------------------------------------------------------------------


def resolve_case_columns(frame: pd.DataFrame) -> CaseColumns:
    lower_to_actual = {str(column).lower(): str(column) for column in frame.columns}
    required = ("celex", "date", "raw_content")
    missing = [name for name in required if name not in lower_to_actual]
    if missing:
        raise ValueError(
            "Case pickle is missing required column(s): " + ", ".join(missing)
        )

    return CaseColumns(
        celex=lower_to_actual["celex"],
        date=lower_to_actual["date"],
        raw_content=lower_to_actual["raw_content"],
    )


def build_case_analysis_frame(
    frame: pd.DataFrame,
    columns: CaseColumns,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_text = frame[columns.raw_content].fillna("").astype(str)
    dates = pd.to_datetime(frame[columns.date], errors="coerce")

    analysis = frame.copy()
    analysis["year_parsed"] = dates.dt.year
    analysis["word_count"] = raw_text.str.split().str.len().astype(int)
    analysis["character_count"] = raw_text.str.len().astype(int)

    yearly = analysis.dropna(subset=["year_parsed"]).copy()
    if yearly.empty:
        raise ValueError("No valid years remained after parsing the date column.")
    yearly["year_parsed"] = yearly["year_parsed"].astype(int)

    return analysis, yearly


def build_year_summary(yearly: pd.DataFrame) -> pd.DataFrame:
    return (
        yearly.groupby("year_parsed", as_index=False)
        .agg(
            cases_per_year=("year_parsed", "size"),
            avg_words_per_case=("word_count", "mean"),
            median_words_per_case=("word_count", "median"),
            total_words=("word_count", "sum"),
        )
        .rename(columns={"year_parsed": "year"})
        .sort_values("year")
        .reset_index(drop=True)
    )


def choose_year_tick_step(summary: pd.DataFrame) -> int:
    min_year = int(summary["year"].min())
    max_year = int(summary["year"].max())
    year_range = max_year - min_year
    if year_range <= 10:
        return 1
    if year_range <= 20:
        return 2
    if year_range <= 50:
        return 5
    return 10


def save_cases_per_year_figure(summary: pd.DataFrame, output_path: Path, dpi: int) -> None:
    configure_plot_defaults()
    axis_font_size = 8

    fig, axis_cases = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    axis_cases.bar(
        summary["year"].astype(int),
        summary["cases_per_year"],
        label="Cases per year",
    )
    axis_cases.set_xlabel("Year", fontsize=axis_font_size)
    axis_cases.set_ylabel("Number of cases", fontsize=axis_font_size)
    axis_cases.tick_params(axis="y", labelsize=axis_font_size)
    axis_cases.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    axis_cases.xaxis.set_major_locator(ticker.MultipleLocator(choose_year_tick_step(summary)))
    axis_cases.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plt.setp(axis_cases.get_xticklabels(), rotation=45, fontsize=axis_font_size)

    axis_words = axis_cases.twinx()
    axis_words.plot(
        summary["year"].astype(int),
        summary["avg_words_per_case"],
        linewidth=1.5,
        label="Average word count per case",
    )
    axis_words.set_ylabel("Average word count", fontsize=axis_font_size)
    axis_words.tick_params(axis="y", labelsize=axis_font_size)

    fig.suptitle(
        "CJEU cases per year and average case length",
        fontsize=10,
        fontweight="bold",
    )
    case_lines, case_labels = axis_cases.get_legend_handles_labels()
    word_lines, word_labels = axis_words.get_legend_handles_labels()
    axis_cases.legend(
        case_lines + word_lines,
        case_labels + word_labels,
        fontsize=axis_font_size,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_cases_per_year_table(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "CJEU Cases per Year and Case Length",
        "===================================",
        "",
        "Columns",
        "-------",
        "year: calendar year parsed from the case date",
        "cases_per_year: number of corpus rows in that year",
        "avg_words_per_case: arithmetic mean case length",
        "median_words_per_case: median case length",
        "total_words: total corpus words for that year",
        "",
        "year\tcases_per_year\tavg_words_per_case\tmedian_words_per_case\ttotal_words",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"{int(row.year)}\t{int(row.cases_per_year)}\t"
            f"{row.avg_words_per_case:.2f}\t{row.median_words_per_case:.2f}\t"
            f"{int(row.total_words)}"
        )

    write_text(output_path, "\n".join(lines) + "\n")


def write_general_insights(
    analysis: pd.DataFrame,
    yearly: pd.DataFrame,
    summary: pd.DataFrame,
    columns: CaseColumns,
    output_path: Path,
    top_k: int,
) -> None:
    total_rows = len(analysis)
    unique_celex = analysis[columns.celex].astype(str).str.strip().nunique()
    valid_date_rows = int(analysis["year_parsed"].notna().sum())
    empty_text_rows = int((analysis["word_count"] == 0).sum())

    most_published = summary.sort_values(
        ["cases_per_year", "year"],
        ascending=[False, True],
    ).iloc[0]
    least_published = summary.sort_values(
        ["cases_per_year", "year"],
        ascending=[True, True],
    ).iloc[0]
    most_words_year = summary.sort_values(
        ["total_words", "year"],
        ascending=[False, True],
    ).iloc[0]

    longest = analysis.sort_values(
        ["word_count", columns.celex],
        ascending=[False, True],
    ).head(top_k)
    shortest = analysis.sort_values(
        ["word_count", columns.celex],
        ascending=[True, True],
    ).head(top_k)

    lines = [
        "CJEU Dataset General Insights",
        "=============================",
        "",
        "General",
        "-------",
        f"Total rows: {total_rows:,}",
        f"Unique CELEX identifiers: {unique_celex:,}",
        f"Rows with a valid parsed date: {valid_date_rows:,}",
        f"Rows with empty raw_content: {empty_text_rows:,}",
        f"Earliest parsed year: {int(yearly['year_parsed'].min())}",
        f"Latest parsed year: {int(yearly['year_parsed'].max())}",
        f"Average word count per row: {analysis['word_count'].mean():,.2f}",
        f"Median word count per row: {analysis['word_count'].median():,.0f}",
        f"Minimum word count: {int(analysis['word_count'].min()):,}",
        f"Maximum word count: {int(analysis['word_count'].max()):,}",
        f"Average character count per row: {analysis['character_count'].mean():,.2f}",
        f"Median character count per row: {analysis['character_count'].median():,.0f}",
        "",
        "Publication volume by year",
        "--------------------------",
        f"Most published year: {int(most_published['year'])} "
        f"({int(most_published['cases_per_year']):,} cases)",
        f"Least published observed year: {int(least_published['year'])} "
        f"({int(least_published['cases_per_year']):,} cases)",
        f"Year with the most total words: {int(most_words_year['year'])} "
        f"({int(most_words_year['total_words']):,} words)",
        "",
        f"Top {top_k} longest cases by word count",
        "-" * len(f"Top {top_k} longest cases by word count"),
    ]

    for position, (_, row) in enumerate(longest.iterrows(), start=1):
        lines.append(
            f"{position}. CELEX: {row[columns.celex]} | "
            f"Date: {row[columns.date]} | Words: {int(row['word_count']):,}"
        )

    lines.extend(
        [
            "",
            f"Top {top_k} shortest cases by word count",
            "-" * len(f"Top {top_k} shortest cases by word count"),
        ]
    )
    for position, (_, row) in enumerate(shortest.iterrows(), start=1):
        lines.append(
            f"{position}. CELEX: {row[columns.celex]} | "
            f"Date: {row[columns.date]} | Words: {int(row['word_count']):,}"
        )

    write_text(output_path, "\n".join(lines) + "\n")


def generate_case_outputs(
    case_pickle: Path,
    output_dir: Path,
    dpi: int,
    top_k: int,
) -> None:
    require_existing_path(case_pickle, "Case pickle")
    frame = pd.read_pickle(case_pickle)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("The case pickle did not contain a pandas DataFrame.")

    columns = resolve_case_columns(frame)
    analysis, yearly = build_case_analysis_frame(frame, columns)
    summary = build_year_summary(yearly)

    save_cases_per_year_figure(
        summary,
        output_dir / "cases_per_year_vs_length.png",
        dpi=dpi,
    )
    write_cases_per_year_table(
        summary,
        output_dir / "cases_per_year_vs_length.txt",
    )
    write_general_insights(
        analysis,
        yearly,
        summary,
        columns,
        output_dir / "general_insights.txt",
        top_k=top_k,
    )


# -----------------------------------------------------------------------------
# Keyword facts and figures
# -----------------------------------------------------------------------------


def normalize_keyword(term: str) -> str:
    return " ".join(term.strip().lower().split())


def extract_keyword_counts(
    json_dir: Path,
) -> Tuple[pd.DataFrame, KeywordBuildStats]:
    require_existing_path(json_dir, "JSON input")

    counts: Counter[str] = Counter()
    display_labels: Dict[str, str] = {}
    stats = KeywordBuildStats()

    for path in iter_json_files(json_dir):
        stats.files_scanned += 1
        data = load_json(path)
        if data is None:
            stats.files_failed += 1
            continue

        case_summary = data.get("case_summary")
        if not isinstance(case_summary, dict):
            continue

        keywords = case_summary.get("keywords")
        if not isinstance(keywords, list):
            continue

        valid_in_file = False
        for keyword in keywords:
            if not isinstance(keyword, dict):
                continue
            term = keyword.get("term")
            if not isinstance(term, str) or not term.strip():
                continue

            normalized = normalize_keyword(term)
            if not normalized:
                continue

            valid_in_file = True
            counts[normalized] += 1
            display_labels.setdefault(normalized, " ".join(term.strip().split()))

        if valid_in_file:
            stats.files_with_keywords += 1

    stats.total_keyword_instances = sum(counts.values())
    stats.unique_keywords = len(counts)

    frame = pd.DataFrame(
        [(display_labels[key], count) for key, count in counts.items()],
        columns=["keyword", "count"],
    )
    if not frame.empty:
        frame = frame.sort_values(
            ["count", "keyword"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return frame, stats


def save_keyword_figure(
    frame: pd.DataFrame,
    output_path: Path,
    top_k: int,
    dpi: int,
) -> None:
    if frame.empty:
        raise ValueError("No valid keywords were found in the JSON corpus.")

    configure_plot_defaults()
    plot_frame = frame.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
    ax.barh(plot_frame["keyword"], plot_frame["count"])
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("Keyword")
    ax.set_title(f"Top {top_k} most frequent keywords in the CJEU corpus", pad=10)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_keyword_overview(
    frame: pd.DataFrame,
    stats: KeywordBuildStats,
    output_path: Path,
    top_k: int,
) -> None:
    lines = [
        "CJEU Keyword Overview",
        "=====================",
        "",
        "General statistics",
        "------------------",
        f"JSON files processed: {stats.files_scanned:,}",
        f"JSON files skipped after read errors: {stats.files_failed:,}",
        f"Cases with at least one valid keyword: {stats.files_with_keywords:,}",
        f"Total keyword instances: {stats.total_keyword_instances:,}",
        f"Unique normalized keywords: {stats.unique_keywords:,}",
        "",
        f"Top {top_k} most frequent keywords",
        "-" * len(f"Top {top_k} most frequent keywords"),
    ]

    for position, row in enumerate(frame.head(top_k).itertuples(index=False), start=1):
        percentage = (
            row.count / stats.total_keyword_instances * 100
            if stats.total_keyword_instances
            else 0.0
        )
        lines.append(f"{position}. {row.keyword} | {row.count:,} ({percentage:.2f}%)")

    write_text(output_path, "\n".join(lines) + "\n")


def generate_keyword_outputs(
    json_dir: Path,
    output_dir: Path,
    top_k: int,
    dpi: int,
) -> None:
    frame, stats = extract_keyword_counts(json_dir)
    if frame.empty:
        raise ValueError("No valid keyword terms were found in the JSON corpus.")

    frame.to_csv(output_dir / "keyword_counts.csv", index=False)
    write_keyword_overview(
        frame,
        stats,
        output_dir / "keyword_overview.txt",
        top_k=top_k,
    )
    save_keyword_figure(
        frame,
        output_dir / "keyword_frequency_top25.png",
        top_k=top_k,
        dpi=dpi,
    )


# -----------------------------------------------------------------------------
# Costs file and manifest verification
# -----------------------------------------------------------------------------


def prepare_costs_file(
    output_dir: Path,
    costs_source: Optional[Path],
    overwrite_template: bool = False,
) -> str:
    """Copy a supplied costs file or create a non-fabricated manual template."""
    target = output_dir / "costs.txt"

    if costs_source is not None:
        require_existing_path(costs_source, "Costs source")
        if costs_source.resolve() != target.resolve():
            shutil.copyfile(costs_source, target)
        return "copied from the supplied costs file"

    if target.exists() and not overwrite_template:
        return "preserved existing costs.txt"

    template = """CJEU Dataset Costs
==================

Status
------
This file requires manual completion. Cost information is not present in the
case pickle or JSON corpus and therefore cannot be calculated by this script.
No monetary values have been inferred or invented.

Suggested fields
----------------
Currency:
Gemini/API model(s):
Input-token cost:
Output-token cost:
Other API or infrastructure costs:
Manual annotation costs:
Total estimated cost:
Method and assumptions:
Date last updated:
"""
    write_text(target, template)
    return "created a manual completion template"


def verify_required_outputs(output_dir: Path) -> Tuple[List[Path], List[Path]]:
    present: List[Path] = []
    missing: List[Path] = []

    for filename in REQUIRED_OUTPUT_FILES:
        path = output_dir / filename
        if path.is_file() and path.stat().st_size > 0:
            present.append(path)
        else:
            missing.append(path)

    return present, missing


def print_manifest_report(output_dir: Path) -> bool:
    present, missing = verify_required_outputs(output_dir)

    print("\nOutput manifest")
    print("===============")
    for path in present:
        marker = "MANUAL CONTENT" if path.name == "costs.txt" else "OK"
        print(f"[{marker}] {path.name}")
    for path in missing:
        print(f"[MISSING] {path.name}")

    if missing:
        print(f"\nMissing {len(missing)} required output file(s).")
        return False

    print("\nAll required output files are present and non-empty.")
    return True


# -----------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------


def command_all(args: SimpleNamespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating case facts and figures...")
    generate_case_outputs(
        case_pickle=args.case_pickle,
        output_dir=output_dir,
        dpi=args.dpi,
        top_k=args.top_k,
    )

    print("Generating keyword facts and figures...")
    generate_keyword_outputs(
        json_dir=args.json_dir,
        output_dir=output_dir,
        top_k=args.keyword_top_k,
        dpi=args.dpi,
    )

    print("Generating citation facts and figures...")
    citation_graph_tsv = (
        output_dir / "citation_graph.tsv" if args.write_citation_graph else None
    )
    network_output = (
        output_dir / "citation_network_top_nodes.png"
        if args.include_network
        else None
    )
    generate_citation_outputs(
        json_dir=args.json_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        hist_bins=args.hist_bins,
        dpi=args.dpi,
        log_y=not args.linear_hist_y,
        citation_graph_tsv=citation_graph_tsv,
        network_output=network_output,
        network_top_n=args.network_top_n,
        network_label_top_n=args.network_label_top_n,
    )

    if args.enriched_json_dir is not None:
        print("Writing citation-enriched JSON corpus...")
        enriched_count = write_enriched_json_corpus(
            args.json_dir,
            args.enriched_json_dir,
        )
        print(f"Wrote {enriched_count:,} enriched JSON file(s).")

    costs_status = prepare_costs_file(
        output_dir=output_dir,
        costs_source=args.costs_source,
        overwrite_template=args.overwrite_costs_template,
    )
    print(f"costs.txt: {costs_status}.")

    return 0 if print_manifest_report(output_dir) else 1


def command_cases(args: SimpleNamespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_case_outputs(
        case_pickle=args.case_pickle,
        output_dir=args.output_dir,
        dpi=args.dpi,
        top_k=args.top_k,
    )
    return 0


def command_keywords(args: SimpleNamespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_keyword_outputs(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        top_k=args.keyword_top_k,
        dpi=args.dpi,
    )
    return 0


def command_citations(args: SimpleNamespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_citation_outputs(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        hist_bins=args.hist_bins,
        dpi=args.dpi,
        log_y=not args.linear_hist_y,
        citation_graph_tsv=(
            args.output_dir / "citation_graph.tsv"
            if args.write_citation_graph
            else None
        ),
        network_output=(
            args.output_dir / "citation_network_top_nodes.png"
            if args.include_network
            else None
        ),
        network_top_n=args.network_top_n,
        network_label_top_n=args.network_label_top_n,
    )
    return 0


def command_enrich(args: SimpleNamespace) -> int:
    count = write_enriched_json_corpus(args.json_dir, args.enriched_json_dir)
    print(f"Wrote {count:,} citation-enriched JSON file(s).")
    return 0


def command_costs(args: SimpleNamespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = prepare_costs_file(
        output_dir=args.output_dir,
        costs_source=args.costs_source,
        overwrite_template=args.overwrite_costs_template,
    )
    print(f"costs.txt: {status}.")
    return 0


def command_verify(args: SimpleNamespace) -> int:
    return 0 if print_manifest_report(args.output_dir) else 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


















# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
RUN_MODE = "all"  # all, cases, keywords, citations, enrich, costs, or verify

CASE_PICKLE = Path("data/raw/eurlex_raw.pkl")
JSON_CORPUS = Path("data/enriched_json")
OUTPUT_DIRECTORY = Path("facts_and_figures")
OPTIONAL_ENRICHED_JSON_DIRECTORY: Path | None = None
COSTS_SOURCE: Path | None = None

DPI = DEFAULT_DPI
TOP_K = DEFAULT_TOP_K
KEYWORD_TOP_K = DEFAULT_KEYWORD_TOP_K
HISTOGRAM_BINS = DEFAULT_HIST_BINS
LINEAR_HISTOGRAM_Y = False
WRITE_CITATION_GRAPH = True
INCLUDE_NETWORK = False
NETWORK_TOP_N = 75
NETWORK_LABEL_TOP_N = 20
OVERWRITE_COSTS_TEMPLATE = False


def build_ide_arguments(project_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        case_pickle=resolve_project_path(project_root, CASE_PICKLE),
        json_dir=resolve_project_path(project_root, JSON_CORPUS),
        output_dir=resolve_project_path(project_root, OUTPUT_DIRECTORY),
        enriched_json_dir=(
            resolve_project_path(project_root, OPTIONAL_ENRICHED_JSON_DIRECTORY)
            if OPTIONAL_ENRICHED_JSON_DIRECTORY is not None
            else None
        ),
        costs_source=(
            resolve_project_path(project_root, COSTS_SOURCE)
            if COSTS_SOURCE is not None
            else None
        ),
        dpi=DPI,
        top_k=TOP_K,
        keyword_top_k=KEYWORD_TOP_K,
        hist_bins=HISTOGRAM_BINS,
        linear_hist_y=LINEAR_HISTOGRAM_Y,
        write_citation_graph=WRITE_CITATION_GRAPH,
        include_network=INCLUDE_NETWORK,
        network_top_n=NETWORK_TOP_N,
        network_label_top_n=NETWORK_LABEL_TOP_N,
        overwrite_costs_template=OVERWRITE_COSTS_TEMPLATE,
    )


def validate_ide_configuration(args: SimpleNamespace) -> None:
    for name in (
        "dpi", "top_k", "keyword_top_k", "hist_bins",
        "network_top_n", "network_label_top_n",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be greater than zero.")


def main() -> int:
    set_reproducible_seed(123)
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    args = build_ide_arguments(project_root)
    mode = RUN_MODE.strip().lower()

    commands = {
        "all": command_all,
        "cases": command_cases,
        "keywords": command_keywords,
        "citations": command_citations,
        "enrich": command_enrich,
        "costs": command_costs,
        "verify": command_verify,
    }

    try:
        validate_ide_configuration(args)
        if mode not in commands:
            raise ValueError(
                "RUN_MODE must be one of: " + ", ".join(sorted(commands))
            )
        return int(commands[mode](args))
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
