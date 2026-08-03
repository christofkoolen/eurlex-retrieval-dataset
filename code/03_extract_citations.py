"""Extract CJEU citations, enrich case JSON, and build citation outputs.

Edit the IDE configuration near the end of the file and run the complete
script. All default paths are relative to the repository root.
"""

import csv
import json
import re
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
)


# ============================================================
# Citation extraction and JSON enrichment
# ============================================================

ECLI_RE = re.compile(r"\b(?:ECLI:)?(EU:(?:C|T|F):\d{4}:\d{1,6})\b", re.IGNORECASE)


@dataclass(frozen=True)
class CitationEdge:
    source: str
    target: str


@dataclass
class EnrichmentResult:
    input_path: Path
    output_path: Path
    citations_found: int


@dataclass
class GraphBuildResult:
    files_scanned: int
    raw_edges_found: int
    unique_edges_kept: int
    edges: List[CitationEdge]


@dataclass
class TopCaseTables:
    top_in: "pd.DataFrame"
    top_out: "pd.DataFrame"


def canonicalize_ecli(value: Any) -> Optional[str]:
    """Return canonical ``ECLI:EU:...`` if a recognizable ECLI is present."""
    if not isinstance(value, str):
        return None
    match = ECLI_RE.search(value)
    return f"ECLI:{match.group(1).upper()}" if match else None


def build_case_id(data: dict) -> Optional[str]:
    """Prefer the document's own ECLI, else fall back to case number."""
    if not isinstance(data, dict):
        return None

    meta = data.get("case_metadata", {})
    if not isinstance(meta, dict):
        return None

    own_ecli = canonicalize_ecli(meta.get("ecli"))
    if own_ecli:
        return own_ecli

    case_number = meta.get("case_number")
    if isinstance(case_number, str) and case_number.strip():
        return case_number.strip()

    return None


def iter_json_files(root: Path) -> Iterator[Path]:
    """Yield JSON files from a single file or recursively from a directory."""
    if root.is_file() and root.suffix.lower() == ".json":
        yield root
    elif root.is_dir():
        yield from root.rglob("*.json")


def load_json(path: Path) -> Optional[dict]:
    """Safely load JSON. Returns None on failure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"[skip] Failed to read {path}: {exc}")
        return None


def _walk_strings(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_strings(item)
    elif isinstance(node, str):
        yield node


def extract_eclis_from_case_segments(
    case_segments: Any,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Return unique ECLIs in encounter order from ``case_segments``."""
    exclude = exclude or set()
    seen: Set[str] = set()
    ordered: List[str] = []

    for text in _walk_strings(case_segments):
        for match in ECLI_RE.finditer(text):
            canonical = f"ECLI:{match.group(1).upper()}"
            if canonical in exclude or canonical in seen:
                continue
            seen.add(canonical)
            ordered.append(canonical)

    return ordered


def extract_eclis_with_locations_from_case_segments(
    case_segments: Any,
    exclude: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, List[Dict[str, Optional[str]]]]]:
    """Extract ECLIs and segment-level locations from ``case_segments``."""
    exclude = exclude or set()
    ordered: List[str] = []
    seen: Set[str] = set()
    locations: Dict[str, List[Dict[str, Optional[str]]]] = {}

    if not isinstance(case_segments, dict):
        return ordered, locations

    for segment_name, segment_value in case_segments.items():
        for text in _walk_strings(segment_value):
            for match in ECLI_RE.finditer(text):
                canonical = f"ECLI:{match.group(1).upper()}"
                if canonical in exclude:
                    continue
                if canonical not in seen:
                    seen.add(canonical)
                    ordered.append(canonical)
                locations.setdefault(canonical, []).append(
                    {"segment": segment_name if isinstance(segment_name, str) else None}
                )

    return ordered, locations


def add_citations_field(data: dict) -> dict:
    """Add or update the top-level ``citations`` field in a case JSON object."""
    if not isinstance(data, dict):
        return data

    meta = data.get("case_metadata", {}) if isinstance(data.get("case_metadata"), dict) else {}
    own_ecli = canonicalize_ecli(meta.get("ecli"))
    exclude: Set[str] = {own_ecli} if own_ecli else set()

    case_segments = data.get("case_segments", {})
    seg_ordered, locations_map = extract_eclis_with_locations_from_case_segments(
        case_segments,
        exclude=exclude,
    )

    all_ordered = list(seg_ordered)
    seen = set(all_ordered)
    for citation in extract_eclis_from_case_segments(case_segments, exclude=exclude):
        if citation not in seen:
            seen.add(citation)
            all_ordered.append(citation)

    existing_citations = data.get("citations", {}) if isinstance(data.get("citations"), dict) else {}
    existing_list = existing_citations.get("list")

    if isinstance(existing_list, list):
        merged: List[str] = []
        merged_seen: Set[str] = set()

        for citation in existing_list:
            canonical = canonicalize_ecli(citation) or citation
            if isinstance(canonical, str) and canonical not in exclude and canonical not in merged_seen:
                merged.append(canonical)
                merged_seen.add(canonical)

        for citation in all_ordered:
            if citation not in merged_seen:
                merged.append(citation)
                merged_seen.add(citation)

        final_list = merged
    else:
        final_list = all_ordered

    detailed_items: List[Dict[str, Any]] = [
        {
            "type": "ECLI",
            "citation": citation,
            "locations": locations_map.get(citation, []),
        }
        for citation in final_list
    ]

    source_id = build_case_id(data)
    edges: List[Dict[str, str]] = []
    if source_id:
        seen_edges: Set[Tuple[str, str]] = set()
        for target in final_list:
            edge_key = (source_id, target)
            if edge_key in seen_edges:
                continue
            edges.append({"source": source_id, "target": target})
            seen_edges.add(edge_key)

    data["citations"] = {
        "list": final_list,
        "detailed": detailed_items,
        "edges": edges,
    }

    data.pop("case_citations", None)
    data.pop("case_citations_detailed", None)
    data.pop("citation_edges", None)
    return data


def process_json_string(text: str) -> str:
    """Enrich a JSON string and return pretty-printed JSON."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object at the top level.")
    enriched = add_citations_field(data)
    return json.dumps(enriched, ensure_ascii=False, indent=2)


def process_json_file(
    path: Path,
    write_back: bool = False,
    out_dir: Optional[Path] = None,
    rel_root: Optional[Path] = None,
) -> EnrichmentResult:
    """Read, enrich, and write a single JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON object expected in {path}")

    enriched = add_citations_field(data)
    citations_found = len(enriched.get("citations", {}).get("list", []))

    if write_back:
        target = path
    elif out_dir is not None:
        rel_path = path.relative_to(rel_root) if rel_root and path.is_relative_to(rel_root) else Path(path.name)
        target = out_dir / rel_path
    else:
        target = path.with_name(f"{path.stem}_updated.json")

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(enriched, handle, ensure_ascii=False, indent=2)

    return EnrichmentResult(
        input_path=path,
        output_path=target,
        citations_found=citations_found,
    )


def enrich_json_collection(
    input_path: Path,
    in_place: bool = False,
    output_dir: Optional[Path] = None,
) -> List[EnrichmentResult]:
    """Enrich all JSON files under a directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    results: List[EnrichmentResult] = []

    if input_path.is_file():
        results.append(
            process_json_file(
                input_path,
                write_back=in_place,
                out_dir=None if in_place else output_dir,
                rel_root=None,
            )
        )
        return results

    out_dir_path: Optional[Path] = None
    if not in_place:
        out_dir_path = output_dir or (input_path.parent / f"{input_path.name}_updated")
        out_dir_path.mkdir(parents=True, exist_ok=True)

    for json_file in iter_json_files(input_path):
        results.append(
            process_json_file(
                json_file,
                write_back=in_place,
                out_dir=out_dir_path,
                rel_root=input_path,
            )
        )

    return results


# ============================================================
# Citation graph TSV generation
# ============================================================


def extract_edges(data: dict) -> List[CitationEdge]:
    """Extract ``CitationEdge`` objects from ``data['citations']['edges']``."""
    out: List[CitationEdge] = []
    if not isinstance(data, dict):
        return out

    citations = data.get("citations")
    if not isinstance(citations, dict):
        return out

    edges = citations.get("edges")
    if not isinstance(edges, list):
        return out

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if isinstance(source, str) and source.strip() and isinstance(target, str) and target.strip():
            out.append(CitationEdge(source.strip(), target.strip()))

    return out


def build_citation_graph(input_path: Path) -> GraphBuildResult:
    """Collect globally deduplicated citation edges from enriched JSON."""
    all_edges: List[CitationEdge] = []
    seen: Set[Tuple[str, str]] = set()
    files_scanned = 0
    raw_edges_found = 0

    for json_file in iter_json_files(input_path):
        files_scanned += 1
        data = load_json(json_file)
        if data is None:
            continue

        edges = extract_edges(data)
        raw_edges_found += len(edges)
        for edge in edges:
            key = (edge.source, edge.target)
            if key in seen:
                continue
            seen.add(key)
            all_edges.append(edge)

    return GraphBuildResult(
        files_scanned=files_scanned,
        raw_edges_found=raw_edges_found,
        unique_edges_kept=len(all_edges),
        edges=all_edges,
    )


def write_citation_graph_tsv(path: Path, edges: Sequence[CitationEdge]) -> Path:
    """Write citation edges to a TSV file with ``source`` and ``target`` columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "target"])
        for edge in edges:
            writer.writerow([edge.source, edge.target])
    return path


# ============================================================
# Graph analytics and visualizations
# ============================================================


def _require_analysis_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import pandas as pd  # type: ignore
        import networkx as nx  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This command requires pandas, networkx, and matplotlib. "
            "Install them with: pip install pandas networkx matplotlib"
        ) from exc
    return pd, nx, plt


DEFAULT_DPI = 300
DEFAULT_TOP_N_MOST_CITED = 75
DEFAULT_HIST_BINS = 50
DEFAULT_LABEL_TOP_N = 20
DEFAULT_LAYOUT_SEED = 42
DEFAULT_SPRING_K = 0.7
DEFAULT_HIST_FIGSIZE = (8.5, 5.5)
DEFAULT_NETWORK_FIGSIZE = (13, 11)


def apply_publication_style(plt: Any) -> None:
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
            "legend.fontsize": 10,
            "font.size": 10,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def load_graph_from_tsv(tsv_path: Path) -> Any:
    """Load a directed NetworkX graph from a citation graph TSV."""
    pd, nx, _ = _require_analysis_dependencies()

    if not tsv_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    required = {"source", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"TSV must contain columns: {required}")

    df = df[["source", "target"]].dropna().copy()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")].drop_duplicates()

    return nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )


def save_figure(fig: Any, out_path: Path, dpi: int = DEFAULT_DPI, save_pdf: bool = True) -> None:
    """Save a matplotlib figure as PNG and optionally PDF."""
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def add_histogram_style(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_indegree_histogram(
    graph: Any,
    out_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    bins: int = DEFAULT_HIST_BINS,
    use_log_y: bool = True,
    figsize: Tuple[float, float] = DEFAULT_HIST_FIGSIZE,
    save_pdf: bool = True,
) -> None:
    _, _, plt = _require_analysis_dependencies()
    apply_publication_style(plt)

    indegrees = [degree for _, degree in graph.in_degree()]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(indegrees, bins=bins, edgecolor="black", linewidth=0.5)
    add_histogram_style(
        ax,
        xlabel="Incoming citations per case",
        ylabel="Number of cases",
        title="Distribution of incoming citations in the citation graph",
    )
    if use_log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, save_pdf=save_pdf)
    plt.close(fig)


def save_outdegree_histogram(
    graph: Any,
    out_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    bins: int = DEFAULT_HIST_BINS,
    use_log_y: bool = True,
    figsize: Tuple[float, float] = DEFAULT_HIST_FIGSIZE,
    save_pdf: bool = True,
) -> None:
    _, _, plt = _require_analysis_dependencies()
    apply_publication_style(plt)

    outdegrees = [degree for _, degree in graph.out_degree()]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(outdegrees, bins=bins, edgecolor="black", linewidth=0.5)
    add_histogram_style(
        ax,
        xlabel="Outgoing citations per case",
        ylabel="Number of cases",
        title="Distribution of outgoing citations in the citation graph",
    )
    if use_log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, save_pdf=save_pdf)
    plt.close(fig)


def build_top_cited_subgraph(graph: Any, top_n: int = DEFAULT_TOP_N_MOST_CITED) -> Any:
    """Keep only the top-N most cited nodes and edges among them."""
    _, nx, _ = _require_analysis_dependencies()

    top_nodes = sorted(graph.in_degree, key=lambda item: item[1], reverse=True)[:top_n]
    top_node_ids = {node for node, _ in top_nodes}
    subgraph = graph.subgraph(top_node_ids).copy()

    isolates = list(nx.isolates(subgraph))
    if isolates:
        subgraph.remove_nodes_from(isolates)

    return subgraph


def truncate_label(text: str, max_len: int = 26) -> str:
    return text if len(text) <= max_len else f"{text[: max_len - 1]}…"


def save_network_subgraph(
    graph: Any,
    out_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    top_n: int = DEFAULT_TOP_N_MOST_CITED,
    label_top_n: int = DEFAULT_LABEL_TOP_N,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
    spring_k: float = DEFAULT_SPRING_K,
    figsize: Tuple[float, float] = DEFAULT_NETWORK_FIGSIZE,
    save_pdf: bool = True,
) -> None:
    _, _, plt = _require_analysis_dependencies()
    import networkx as nx  # type: ignore

    apply_publication_style(plt)

    subgraph = build_top_cited_subgraph(graph, top_n=top_n)
    if subgraph.number_of_nodes() == 0:
        raise ValueError("Filtered subgraph is empty. Increase TOP_N_MOST_CITED.")

    node_sizes = [120 + (graph.in_degree(node) * 18) for node in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(subgraph, seed=layout_seed, k=spring_k)

    nx.draw_networkx_edges(
        subgraph,
        pos,
        ax=ax,
        alpha=0.22,
        arrows=True,
        arrowsize=10,
        width=0.7,
        connectionstyle="arc3,rad=0.04",
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.6,
        edgecolors="black",
    )

    label_candidates = sorted(subgraph.nodes(), key=lambda node: graph.in_degree(node), reverse=True)[:label_top_n]
    labels = {node: truncate_label(node, max_len=24) for node in label_candidates}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=7, font_weight="bold", ax=ax)

    ax.set_title(f"Core citation subgraph of the top {top_n} most cited cases", pad=14)
    ax.text(
        0.01,
        0.01,
        f"Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()} | Node size ∝ in-degree",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "black", "linewidth": 0.6},
    )
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, save_pdf=save_pdf)
    plt.close(fig)


def save_summary_stats(graph: Any, out_path: Path) -> Path:
    """Write graph-level summary statistics to a text file."""
    pd, nx, _ = _require_analysis_dependencies()

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    indegrees = [degree for _, degree in graph.in_degree()]
    outdegrees = [degree for _, degree in graph.out_degree()]
    density = nx.density(graph) if node_count > 1 else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("CJEU citation graph summary\n")
        handle.write("===========================\n\n")
        handle.write(f"Nodes (cases): {node_count}\n")
        handle.write(f"Edges (unique citation relations): {edge_count}\n")
        handle.write(f"Graph density: {density:.8f}\n\n")
        if indegrees:
            handle.write(f"Mean in-degree: {sum(indegrees) / len(indegrees):.4f}\n")
            handle.write(f"Median in-degree: {pd.Series(indegrees).median():.4f}\n")
            handle.write(f"Max in-degree: {max(indegrees)}\n\n")
        if outdegrees:
            handle.write(f"Mean out-degree: {sum(outdegrees) / len(outdegrees):.4f}\n")
            handle.write(f"Median out-degree: {pd.Series(outdegrees).median():.4f}\n")
            handle.write(f"Max out-degree: {max(outdegrees)}\n")

    return out_path


def visualize_citation_graph(
    tsv_path: Path,
    output_dir: Path,
    *,
    dpi: int = DEFAULT_DPI,
    bins: int = DEFAULT_HIST_BINS,
    top_n: int = DEFAULT_TOP_N_MOST_CITED,
    label_top_n: int = DEFAULT_LABEL_TOP_N,
    use_log_y: bool = True,
    save_pdf: bool = True,
) -> Dict[str, Path]:
    """Generate standard graph visualizations and summary text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph_from_tsv(tsv_path)

    indegree_path = output_dir / "incoming_citations_histogram.png"
    outdegree_path = output_dir / "outgoing_citations_histogram.png"
    network_path = output_dir / "citation_network_top_nodes.png"
    summary_path = output_dir / "citation_graph_summary.txt"

    save_indegree_histogram(
        graph,
        indegree_path,
        dpi=dpi,
        bins=bins,
        use_log_y=use_log_y,
        save_pdf=save_pdf,
    )
    save_outdegree_histogram(
        graph,
        outdegree_path,
        dpi=dpi,
        bins=bins,
        use_log_y=use_log_y,
        save_pdf=save_pdf,
    )
    save_network_subgraph(
        graph,
        network_path,
        dpi=dpi,
        top_n=top_n,
        label_top_n=label_top_n,
        save_pdf=save_pdf,
    )
    save_summary_stats(graph, summary_path)

    return {
        "incoming_histogram": indegree_path,
        "outgoing_histogram": outdegree_path,
        "network_subgraph": network_path,
        "summary": summary_path,
    }


# ============================================================
# Top cited / top citing case counts
# ============================================================


def compute_top_cases(tsv_path: Path, top_k: int = 10) -> TopCaseTables:
    """Compute top in-degree and out-degree cases from a citation graph TSV."""
    pd, _, _ = _require_analysis_dependencies()

    if not tsv_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    required = {"source", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"TSV must contain columns: {required}")

    df = df[["source", "target"]].dropna().copy()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")].drop_duplicates()

    in_degree = (
        df["target"].value_counts().rename_axis("case_id").reset_index(name="in_degree")
    )
    out_degree = (
        df["source"].value_counts().rename_axis("case_id").reset_index(name="out_degree")
    )

    return TopCaseTables(
        top_in=in_degree.head(top_k),
        top_out=out_degree.head(top_k),
    )


def save_top_cases(tables: TopCaseTables, out_path: Path) -> Path:
    """Save top cited / top citing tables to a TSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["table", "case_id", "value"])
        for _, row in tables.top_in.iterrows():
            writer.writerow(["top_in_degree", row["case_id"], row["in_degree"]])
        for _, row in tables.top_out.iterrows():
            writer.writerow(["top_out_degree", row["case_id"], row["out_degree"]])
    return out_path


def print_top_cases(tables: TopCaseTables) -> None:
    """Print top cited and top citing tables to stdout."""
    print("\nTop most cited cases (in-degree):")
    print(tables.top_in.to_string(index=False))
    print("\nTop most citing cases (out-degree):")
    print(tables.top_out.to_string(index=False))


# ============================================================
# End-to-end pipeline
# ============================================================


def run_pipeline(
    raw_json_path: Path,
    *,
    enriched_dir: Path,
    graph_tsv: Path,
    viz_dir: Optional[Path] = None,
    top_k: int = 10,
    in_place: bool = False,
    dpi: int = DEFAULT_DPI,
    top_n: int = DEFAULT_TOP_N_MOST_CITED,
) -> Dict[str, Any]:
    """Run enrichment, graph export, counts, and optional visualization."""
    enrichment_results = enrich_json_collection(
        raw_json_path,
        in_place=in_place,
        output_dir=None if in_place else enriched_dir,
    )

    graph_source = raw_json_path if in_place else enriched_dir
    graph_result = build_citation_graph(graph_source)
    write_citation_graph_tsv(graph_tsv, graph_result.edges)

    top_tables = compute_top_cases(graph_tsv, top_k=top_k)

    output: Dict[str, Any] = {
        "enrichment_results": enrichment_results,
        "graph_result": graph_result,
        "graph_tsv": graph_tsv,
        "top_tables": top_tables,
    }

    if viz_dir is not None:
        output["visualizations"] = visualize_citation_graph(
            graph_tsv,
            viz_dir,
            dpi=dpi,
            top_n=top_n,
        )

    return output


# ============================================================
# Command-line interface
# ============================================================




# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
RUN_MODE = "pipeline"  # enrich, graph, visualize, counts, or pipeline

INPUT_JSON = Path("data/intermediate/restored_json")
ENRICHED_JSON_DIR = Path("data/enriched_json")
CITATION_GRAPH_TSV = Path("data/citations/citation_graph.tsv")
VISUALIZATION_DIR = Path("facts_and_figures/citations")
TOP_CASES_TSV = Path("data/citations/top_cases.tsv")

TOP_K = 10
DPI = DEFAULT_DPI
HISTOGRAM_BINS = DEFAULT_HIST_BINS
TOP_N_MOST_CITED = DEFAULT_TOP_N_MOST_CITED
LABEL_TOP_N = DEFAULT_LABEL_TOP_N
USE_LOG_Y = True
SAVE_PDF = True
IN_PLACE = False


def main() -> int:
    set_reproducible_seed(123)
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    input_json = resolve_project_path(project_root, INPUT_JSON)
    enriched_dir = resolve_project_path(project_root, ENRICHED_JSON_DIR)
    graph_tsv = resolve_project_path(project_root, CITATION_GRAPH_TSV)
    viz_dir = resolve_project_path(project_root, VISUALIZATION_DIR)
    top_cases_tsv = resolve_project_path(project_root, TOP_CASES_TSV)

    mode = RUN_MODE.strip().lower()
    try:
        if mode == "enrich":
            results = enrich_json_collection(
                input_json,
                in_place=IN_PLACE,
                output_dir=None if IN_PLACE else enriched_dir,
            )
            print(f"Processed {len(results):,} JSON file(s).")
        elif mode == "graph":
            source = input_json if IN_PLACE else enriched_dir
            result = build_citation_graph(source)
            write_citation_graph_tsv(graph_tsv, result.edges)
            print(f"Wrote {result.unique_edges_kept:,} unique edges to {graph_tsv}")
        elif mode == "visualize":
            outputs = visualize_citation_graph(
                graph_tsv,
                viz_dir,
                dpi=DPI,
                bins=HISTOGRAM_BINS,
                top_n=TOP_N_MOST_CITED,
                label_top_n=LABEL_TOP_N,
                use_log_y=USE_LOG_Y,
                save_pdf=SAVE_PDF,
            )
            for name, path in outputs.items():
                print(f"{name}: {path}")
        elif mode == "counts":
            tables = compute_top_cases(graph_tsv, top_k=TOP_K)
            print_top_cases(tables)
            save_top_cases(tables, top_cases_tsv)
        elif mode == "pipeline":
            results = run_pipeline(
                input_json,
                enriched_dir=enriched_dir,
                graph_tsv=graph_tsv,
                viz_dir=viz_dir,
                top_k=TOP_K,
                in_place=IN_PLACE,
                dpi=DPI,
                top_n=TOP_N_MOST_CITED,
            )
            save_top_cases(results["top_tables"], top_cases_tsv)
            graph_result = results["graph_result"]
            print(f"Enriched files: {len(results['enrichment_results']):,}")
            print(f"Unique citation edges: {graph_result.unique_edges_kept:,}")
        else:
            raise ValueError(
                "RUN_MODE must be enrich, graph, visualize, counts, or pipeline."
            )
        return 0
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    main()
