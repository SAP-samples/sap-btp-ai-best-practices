"""
Shared application context and helpers for the RFQx Streamlit UI.

This module centralizes cached service instances, session-state initialization,
and reusable visualization helpers so individual Streamlit pages can stay lean
while reusing the existing business logic from the legacy single-file app.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

# Use non-interactive backend for headless Streamlit execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from country_risk_manager import CountryRiskManager
from graph_processor import GraphProcessor
from main import SimplifiedRFQComparator
from project_manager import ProjectManager
from ui_components import initialize_session_state

ASSETS_BASE = Path(__file__).parent / "static"

GRAPH_COLORS = [
    "skyblue",
    "lightcoral",
    "lightgreen",
    "plum",
    "orange",
    "cyan",
]


@st.cache_resource(show_spinner=False)
def get_comparator() -> SimplifiedRFQComparator:
    """Return a cached comparator instance shared across pages."""
    return SimplifiedRFQComparator()


@st.cache_resource(show_spinner=False)
def get_project_manager() -> ProjectManager:
    """Return a cached project manager instance."""
    return ProjectManager()


@st.cache_resource(show_spinner=False)
def get_country_risk_manager() -> CountryRiskManager:
    """Return a cached country risk manager instance."""
    return CountryRiskManager()


@st.cache_resource(show_spinner=False)
def get_graph_processor() -> GraphProcessor:
    """Return a cached graph processor bound to the shared LLM client."""
    return GraphProcessor(llm_client=get_comparator().client)


def ensure_session_state() -> None:
    """
    Ensure all required session state keys exist.

    This wraps the existing initialization helper and extends it with keys that
    used to be created in the monolithic Streamlit script.
    """
    initialize_session_state()

    defaults = {
        "current_project": None,
        "project_loaded": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_combined_graph_visualization(
    documents_data: Dict[str, Dict[str, Any]]
) -> Tuple[io.BytesIO, go.Figure]:
    """
    Create a combined static and interactive graph visualization.

    Args:
        documents_data: Mapping of document names to extracted JSON data.

    Returns:
        Tuple containing a WebP image buffer and a Plotly figure for interactive use.
    """
    graph_processor = get_graph_processor()

    graphs = {
        name: graph_processor.create_graph_from_json(data)
        for name, data in documents_data.items()
    }

    combined_graph = nx.DiGraph()
    color_map: Dict[str, str] = {}

    for index, (doc_name, graph) in enumerate(graphs.items()):
        doc_color = GRAPH_COLORS[index % len(GRAPH_COLORS)]

        for node, attrs in graph.nodes(data=True):
            combined_graph.add_node(
                node, **attrs, document=doc_name, color=doc_color
            )
            color_map[node] = doc_color

        for source, target, attrs in graph.edges(data=True):
            combined_graph.add_edge(source, target, **attrs, document=doc_name)

    plt.figure(figsize=(24, 18))
    position = nx.spring_layout(combined_graph, seed=42, k=2.0, iterations=100)

    node_labels = {
        node: combined_graph.nodes[node].get("label", node)
        for node in combined_graph.nodes()
    }

    edge_labels = {
        (source, target): attrs.get("label", "")
        for source, target, attrs in combined_graph.edges(data=True)
    }

    node_colors = [color_map.get(node, "gray") for node in combined_graph.nodes()]

    nx.draw(
        combined_graph,
        position,
        labels=node_labels,
        with_labels=True,
        node_color=node_colors,
        node_size=4000,
        font_size=9,
        edge_color="gray",
        arrowsize=22,
        font_weight="bold",
        linewidths=1.5,
        width=1.5,
    )

    nx.draw_networkx_edge_labels(
        combined_graph, position, edge_labels=edge_labels, font_color="red", font_size=7
    )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GRAPH_COLORS[index % len(GRAPH_COLORS)],
            markersize=12,
            label=doc_name,
        )
        for index, doc_name in enumerate(documents_data.keys())
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=10,
    )

    plt.title(
        "Combined Knowledge Graph - RFQ Documents Comparison",
        fontsize=18,
        fontweight="bold",
    )

    buffer = io.BytesIO()
    plt.savefig(
        buffer,
        format="webp",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.3,
    )
    buffer.seek(0)
    plt.close()

    interactive_fig = graph_processor.create_interactive_graph(graphs)
    return buffer, interactive_fig


def template_static_paths() -> Dict[str, Path]:
    """
    Convenience helper returning local static asset paths.

    Returns:
        Dictionary with keys for the SAP logo and CSS files.
    """
    base = ASSETS_BASE
    return {
        "sap_logo_square": base / "images" / "SAP_logo_square.png",
        "sap_logo": base / "images" / "SAP_logo.svg",
        "css_variables": base / "styles" / "variables.css",
        "css_theme": base / "styles" / "style.css",
    }


def apply_template_theme() -> None:
    """
    Load the SAP template styling assets and display the logo.

    This helper mirrors the behaviour of the FastAPI + Streamlit template so
    individual pages can stay minimal.
    """
    assets = template_static_paths()
    if assets["sap_logo"].exists():
        st.logo(str(assets["sap_logo"]))
    load_css_files(
        [
            assets["css_variables"],
            assets["css_theme"],
        ]
    )


def load_css_files(file_paths: Iterable[Path]) -> None:
    """Load and inject CSS styles from the provided file paths."""
    css_chunks = []
    for path in file_paths:
        try:
            css_chunks.append(path.read_text(encoding="utf-8"))
        except FileNotFoundError:  # pragma: no cover - safety net
            continue
    if css_chunks:
        st.markdown(f"<style>{''.join(css_chunks)}</style>", unsafe_allow_html=True)
