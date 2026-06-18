"""Shared type definitions for the metal-composition workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


class MetalCompositionState(TypedDict, total=False):
    product_code: str
    source_row_id: int
    source_summary: Dict[str, Any]
    source_row: Dict[str, Any]
    composition_mode: str
    document_mode: str
    diagram_payloads: List["DiagramPayload"]
    material_master_composition: Optional[Dict[str, Any]]
    include_token_usage: bool
    token_usage: Dict[str, Any]
    token_usage_recorder: Any
    diagram_output: Dict[str, Any]
    hts_fact_profile: Dict[str, Any]
    hana_tree_search_output: Dict[str, Any]
    hts_resolution_output: Dict[str, Any]
    hts_classification: Dict[str, Any]
    section_232_assessment: Dict[str, Any]
    section_232_reasoner_output: Dict[str, Any]
    final_composition: Dict[str, Any]
    timing: Dict[str, Any]


@dataclass(frozen=True)
class DiagramPayload:
    filename: str
    content_type: str
    data: bytes
    source_filename: Optional[str] = None
    page_number: Optional[int] = None


@dataclass(frozen=True)
class RenderedDiagramPage:
    """Internal rendered page/image used during vision analysis."""

    page_ref: str
    source_document_index: int
    filename: str
    content_type: str
    data: bytes
    sequence_index: int = 0
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    rendered_width: int = 0
    rendered_height: int = 0
    input_payload: Optional[DiagramPayload] = None


@dataclass(frozen=True)
class RenderedDiagramTextPage:
    """Internal extracted-text page used during mixed text/image analysis."""

    page_ref: str
    source_document_index: int
    sequence_index: int = 0
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    text: str = ""
    chunk_index: int = 1
    chunk_count: int = 1
    char_count: int = 0
    input_payload: Optional[DiagramPayload] = None


@dataclass(frozen=True)
class ZoomedDiagramCrop:
    """Internal high-resolution crop rendered from a page/image."""

    crop_ref: str
    page_ref: str
    filename: str
    content_type: str
    data: bytes
    normalized_box: Dict[str, float]
    rendered_width: int = 0
    rendered_height: int = 0
    source_filename: Optional[str] = None
    page_number: Optional[int] = None


@dataclass(frozen=True)
class MixedDiagramBatchEntry:
    """One prompt-ready diagram source entry, either image or extracted text."""

    kind: str
    page_ref: str
    sequence_index: int
    label: str
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    content_type: str = ""
    data: bytes = b""
    text: str = ""
    char_count: int = 0


@dataclass(frozen=True)
class MaterializedDiagramSources:
    """Mixed image/text sources plus routing diagnostics for one item."""

    image_pages: List[RenderedDiagramPage]
    text_pages: List[RenderedDiagramTextPage]
    preprocess_details_list: List[Dict[str, Any]]
    routing_summary: Dict[str, Any]
    page_decisions: List[Dict[str, Any]]
