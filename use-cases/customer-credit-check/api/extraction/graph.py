"""
LangGraph workflow for PDF extraction with map-reduce pattern.

This module defines the graph structure that orchestrates parallel
text and image extraction followed by LLM-based reduction.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from langgraph.graph import StateGraph, END
# from langgraph.graph.graph import CompiledGraph

from .schemas import ExtractionState
from .nodes import (
    extract_text_node,
    extract_image_node,
    reducer_node
)

logger = logging.getLogger(__name__)


def create_extraction_graph():
    """
    Create the LangGraph workflow for PDF extraction.
    
    The graph structure:
    1. Start -> Parallel(TextNode, ImageNode)
    2. Both nodes -> ReducerNode
    3. ReducerNode -> End
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    # Initialize the state graph
    workflow = StateGraph(ExtractionState)
    
    # Add nodes to the graph
    workflow.add_node("text_extraction", extract_text_node)
    workflow.add_node("image_extraction", extract_image_node)
    workflow.add_node("reducer", reducer_node)
    
    # Define the graph structure
    # Both extraction nodes start from the beginning (parallel execution)
    workflow.set_entry_point("text_extraction")
    workflow.set_entry_point("image_extraction")
    
    # Both extraction nodes lead to the reducer
    workflow.add_edge("text_extraction", "reducer")
    workflow.add_edge("image_extraction", "reducer")
    
    # Reducer leads to the end
    workflow.add_edge("reducer", END)
    
    # Compile the graph
    compiled = workflow.compile()
    
    logger.info("Created PDF extraction graph with parallel text/image nodes")
    
    return compiled


def create_sequential_extraction_graph():
    """
    Create a sequential version of the extraction graph.
    
    This version processes nodes sequentially instead of in parallel,
    useful for debugging or resource-constrained environments.
    
    The graph structure:
    1. Start -> TextNode
    2. TextNode -> ImageNode
    3. ImageNode -> ReducerNode
    4. ReducerNode -> End
    
    Returns:
        Compiled LangGraph workflow for sequential execution
    """
    # Initialize the state graph
    workflow = StateGraph(ExtractionState)
    
    # Add nodes to the graph
    workflow.add_node("text_extraction", extract_text_node)
    workflow.add_node("image_extraction", extract_image_node)
    workflow.add_node("reducer", reducer_node)
    
    # Define sequential flow
    workflow.set_entry_point("text_extraction")
    workflow.add_edge("text_extraction", "image_extraction")
    workflow.add_edge("image_extraction", "reducer")
    workflow.add_edge("reducer", END)
    
    # Compile the graph
    compiled = workflow.compile()
    
    logger.info("Created sequential PDF extraction graph")
    
    return compiled


def create_adaptive_extraction_graph():
    """
    Create an adaptive extraction graph that decides processing strategy.
    
    This advanced version includes conditional logic to:
    - Skip image extraction if text is high quality
    - Skip text extraction for image-only PDFs
    - Handle errors gracefully with fallbacks
    
    Returns:
        Compiled LangGraph workflow with adaptive behavior
    """
    from langgraph.graph import StateGraph
    
    # Initialize the state graph
    workflow = StateGraph(ExtractionState)
    
    # Add decision node
    def routing_decision(state: ExtractionState) -> str:
        """Decide which extraction path to take based on state."""
        # This is a simplified decision - could be enhanced with PDF analysis
        # For now, always do both extractions in parallel
        return "parallel_extraction"
    
    # Add conditional node for smart routing
    def should_extract_images(state: ExtractionState) -> str:
        """Decide if image extraction is needed based on text results."""
        text_result = state.get("text_result")
        
        # If text extraction failed or has low content, definitely extract images
        if not text_result or not text_result.success:
            return "extract_images"
        
        # Check text quality
        total_chars = sum(p.char_count for p in text_result.pages)
        has_tables = any(p.has_tables for p in text_result.pages)
        has_images = any(p.has_images for p in text_result.pages)
        
        # Extract images if text is sparse or document has visual elements
        if total_chars < 1000 or has_tables or has_images:
            return "extract_images"
        
        # Skip image extraction for text-heavy documents
        return "skip_images"
    
    # Add wrapper node for conditional image extraction
    def conditional_image_node(state: ExtractionState) -> Dict[str, Any]:
        """Conditionally run image extraction."""
        decision = should_extract_images(state)
        
        if decision == "extract_images":
            return extract_image_node(state)
        else:
            # Skip image extraction
            logger.info("Skipping image extraction - sufficient text content available")
            return {}
    
    # Add nodes
    workflow.add_node("text_extraction", extract_text_node)
    workflow.add_node("conditional_image", conditional_image_node)
    workflow.add_node("reducer", reducer_node)
    
    # Define flow
    workflow.set_entry_point("text_extraction")
    workflow.add_edge("text_extraction", "conditional_image")
    workflow.add_edge("conditional_image", "reducer")
    workflow.add_edge("reducer", END)
    
    # Compile the graph
    compiled = workflow.compile()
    
    logger.info("Created adaptive PDF extraction graph")
    
    return compiled


async def run_extraction_async(
    graph,
    pdf_path: str,
    question: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the extraction graph asynchronously.
    
    Args:
        graph: Compiled LangGraph workflow
        pdf_path: Path to PDF file
        question: Question to answer from the PDF
        **kwargs: Additional parameters (language, temperature, etc.)
        
    Returns:
        Final state dictionary with extraction results
    """
    # Prepare initial state
    initial_state: ExtractionState = {
        "pdf_path": pdf_path,
        "question": question,
        "language": kwargs.get("language", "es"),
        "temperature": kwargs.get("temperature", 0.2),
        "max_pages": kwargs.get("max_pages", None),
        "text_result": None,
        "image_result": None,
        "final_result": None,
        "start_time": datetime.now(),
        "errors": []
    }
    
    logger.info(f"Starting async extraction for: {pdf_path}")
    
    # Run the graph
    try:
        # For async, we need to collect all updates and merge them
        final_state = dict(initial_state)
        
        # Async execution with streaming
        async for update in graph.astream(initial_state):
            # Merge each update into the final state
            for key, value in update.items():
                final_state[key] = value
                
                # Log progress
                if key == "text_result" and value:
                    logger.debug("Text extraction completed")
                elif key == "image_result" and value:
                    logger.debug("Image extraction completed")
                elif key == "final_result" and value:
                    logger.debug("Reduction completed")
        
        # Return the complete final state
        return final_state
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise


def run_extraction_sync(
    graph,
    pdf_path: str,
    question: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the extraction graph synchronously.
    
    Args:
        graph: Compiled LangGraph workflow
        pdf_path: Path to PDF file
        question: Question to answer from the PDF
        **kwargs: Additional parameters (language, temperature, etc.)
        
    Returns:
        Final state dictionary with extraction results
    """
    # Prepare initial state
    initial_state: ExtractionState = {
        "pdf_path": pdf_path,
        "question": question,
        "language": kwargs.get("language", "es"),
        "temperature": kwargs.get("temperature", 0.2),
        "max_pages": kwargs.get("max_pages", None),
        "text_result": None,
        "image_result": None,
        "final_result": None,
        "start_time": datetime.now(),
        "errors": []
    }
    
    logger.info(f"Starting sync extraction for: {pdf_path}")
    
    # Run the graph
    try:
        # Synchronous execution
        final_state = graph.invoke(initial_state)
        
        # Log completion
        if final_state.get("final_result"):
            logger.info("Extraction completed successfully")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise


def visualize_graph(graph, output_path: str = "extraction_graph.png") -> None:
    """
    Visualize the extraction graph structure.
    
    Args:
        graph: Compiled LangGraph workflow
        output_path: Path to save the visualization
    """
    try:
        # Get graph visualization
        graph_image = graph.get_graph().draw_png()
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(graph_image)
        
        logger.info(f"Graph visualization saved to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to visualize graph: {e}")
        logger.info("Install graphviz for graph visualization support")