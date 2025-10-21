"""
Main PDF Extractor interface.

Provides high-level API for extracting information from PDFs using
the LangGraph-based parallel processing pipeline.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

from .schemas import (
    PDFExtractionRequest,
    FinalExtractionResult,
    ExtractorConfig
)
from .graph import (
    create_extraction_graph,
    create_sequential_extraction_graph,
    create_adaptive_extraction_graph,
    run_extraction_sync,
    run_extraction_async
)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    High-level interface for PDF information extraction.
    
    This class provides a simple API for extracting information from PDFs
    using a LangGraph-based pipeline with parallel text and image processing.
    
    Example:
        extractor = PDFExtractor()
        result = extractor.extract(
            pdf_path="document.pdf",
            question="What is the customer name?"
        )
        print(result.answer)
    """
    
    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        graph_type: str = "parallel",
        debug: bool = False
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            config: Configuration for extraction nodes
            graph_type: Type of graph to use ("parallel", "sequential", "adaptive")
            debug: Enable debug logging
        """
        self.config = config or ExtractorConfig()
        self.graph_type = graph_type
        self.debug = debug
        
        # Configure logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize the graph
        self._init_graph()
        
        logger.info(f"Initialized PDFExtractor with {graph_type} graph")
    
    def _init_graph(self) -> None:
        """Initialize the LangGraph workflow based on configuration."""
        if self.graph_type == "parallel":
            self.graph = create_extraction_graph()
        elif self.graph_type == "sequential":
            self.graph = create_sequential_extraction_graph()
        elif self.graph_type == "adaptive":
            self.graph = create_adaptive_extraction_graph()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def extract(
        self,
        pdf_path: Union[str, Path],
        question: str,
        language: str = "es",
        max_pages: Optional[int] = None,
        temperature: float = 0.2
    ) -> FinalExtractionResult:
        """
        Extract information from a PDF by answering a specific question.
        
        Args:
            pdf_path: Path to the PDF file
            question: Question to answer from the PDF
            language: Language for the response (default: Spanish)
            max_pages: Maximum number of pages to process
            temperature: LLM temperature for response generation
            
        Returns:
            FinalExtractionResult with the answer and metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If extraction fails
        """
        # Validate input
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting from {pdf_path.name}: {question}")
        
        # Run extraction
        state = run_extraction_sync(
            graph=self.graph,
            pdf_path=str(pdf_path),
            question=question,
            language=language,
            max_pages=max_pages,
            temperature=temperature
        )
        
        # Extract final result
        final_result = state.get("final_result")
        
        if not final_result:
            errors = state.get("errors", [])
            error_msg = "; ".join(errors) if errors else "Unknown error during extraction"
            raise ValueError(f"Extraction failed: {error_msg}")
        
        if not final_result.success:
            raise ValueError(f"Extraction failed: {final_result.error_message}")
        
        return final_result
    
    async def extract_async(
        self,
        pdf_path: Union[str, Path],
        question: str,
        language: str = "es",
        max_pages: Optional[int] = None,
        temperature: float = 0.2
    ) -> FinalExtractionResult:
        """
        Asynchronously extract information from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            question: Question to answer from the PDF
            language: Language for the response
            max_pages: Maximum number of pages to process
            temperature: LLM temperature for response generation
            
        Returns:
            FinalExtractionResult with the answer and metadata
        """
        # Validate input
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Starting async extraction from {pdf_path.name}")
        
        # Run extraction
        state = await run_extraction_async(
            graph=self.graph,
            pdf_path=str(pdf_path),
            question=question,
            language=language,
            max_pages=max_pages,
            temperature=temperature
        )
        
        # Extract final result - check both locations for async compatibility
        final_result = state.get("final_result")
        
        # In async mode, the result might be nested in the reducer output
        if not final_result and "reducer" in state and isinstance(state["reducer"], dict):
            final_result = state["reducer"].get("final_result")
        
        if not final_result:
            errors = state.get("errors", [])
            error_msg = "; ".join(errors) if errors else "Unknown error during extraction"
            raise ValueError(f"Extraction failed: {error_msg}")
        
        if not final_result.success:
            raise ValueError(f"Extraction failed: {final_result.error_message}")
        
        return final_result
    
    def extract_batch(
        self,
        requests: List[PDFExtractionRequest],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[FinalExtractionResult]:
        """
        Extract information from multiple PDFs in batch.
        
        Args:
            requests: List of extraction requests
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            List of extraction results
        """
        results = []
        total = len(requests)
        
        logger.info(f"Starting batch extraction for {total} PDFs")
        
        for i, request in enumerate(requests, 1):
            try:
                # Extract from current PDF
                result = self.extract(
                    pdf_path=request.pdf_path,
                    question=request.question,
                    language=request.language,
                    max_pages=request.max_pages,
                    temperature=request.temperature
                )
                results.append(result)
                
            except Exception as e:
                # Create failed result
                logger.error(f"Failed to process {request.pdf_path}: {e}")
                results.append(
                    FinalExtractionResult(
                        success=False,
                        question=request.question,
                        answer="",
                        source_pages=[],
                        extraction_metadata={"error": str(e)},
                        processing_time_ms=0,
                        error_message=str(e)
                    )
                )
            
            # Progress callback
            if progress_callback:
                progress_callback(i, total)
        
        logger.info(f"Batch extraction completed: {sum(r.success for r in results)}/{total} successful")
        
        return results
    
    async def extract_batch_async(
        self,
        requests: List[PDFExtractionRequest],
        max_concurrent: int = 3
    ) -> List[FinalExtractionResult]:
        """
        Asynchronously extract information from multiple PDFs.
        
        Args:
            requests: List of extraction requests
            max_concurrent: Maximum number of concurrent extractions
            
        Returns:
            List of extraction results
        """
        logger.info(f"Starting async batch extraction for {len(requests)} PDFs")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(request: PDFExtractionRequest) -> FinalExtractionResult:
            """Extract with concurrency control."""
            async with semaphore:
                try:
                    return await self.extract_async(
                        pdf_path=request.pdf_path,
                        question=request.question,
                        language=request.language,
                        max_pages=request.max_pages,
                        temperature=request.temperature
                    )
                except Exception as e:
                    logger.error(f"Failed to process {request.pdf_path}: {e}")
                    return FinalExtractionResult(
                        success=False,
                        question=request.question,
                        answer="",
                        source_pages=[],
                        extraction_metadata={"error": str(e)},
                        processing_time_ms=0,
                        error_message=str(e)
                    )
        
        # Run all extractions concurrently
        results = await asyncio.gather(
            *[extract_with_semaphore(req) for req in requests]
        )
        
        successful = sum(r.success for r in results)
        logger.info(f"Async batch completed: {successful}/{len(requests)} successful")
        
        return results
    
    def extract_with_templates(
        self,
        pdf_path: Union[str, Path],
        template_name: str,
        language: str = "es"
    ) -> Dict[str, Any]:
        """
        Extract information using predefined question templates.
        
        Args:
            pdf_path: Path to the PDF file
            template_name: Name of the template to use
            language: Language for responses
            
        Returns:
            Dictionary with extracted fields
        """
        # Define common extraction templates
        templates = {
            "customer_info": [
                "¿Cuál es el nombre del cliente?",
                "¿Cuál es el RFC del cliente?",
                "¿Cuál es la dirección del cliente?",
                "¿Cuál es el teléfono del cliente?"
            ],
            "invoice_details": [
                "¿Cuál es el número de factura?",
                "¿Cuál es la fecha de la factura?",
                "¿Cuál es el monto total?",
                "¿Cuáles son los productos o servicios?"
            ],
            "contract_terms": [
                "¿Cuál es la fecha de inicio del contrato?",
                "¿Cuál es la duración del contrato?",
                "¿Cuáles son las condiciones de pago?",
                "¿Cuáles son las penalizaciones?"
            ]
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
        
        questions = templates[template_name]
        results = {}
        
        logger.info(f"Extracting with template '{template_name}' from {Path(pdf_path).name}")
        
        for question in questions:
            try:
                result = self.extract(
                    pdf_path=pdf_path,
                    question=question,
                    language=language
                )
                # Extract field name from question
                field_name = question.replace("¿", "").replace("?", "").strip()
                results[field_name] = result.answer
            except Exception as e:
                logger.warning(f"Failed to extract '{question}': {e}")
                results[question] = None
        
        return results
    
    def validate_extraction(
        self,
        result: FinalExtractionResult,
        validation_rules: Optional[Dict[str, Callable]] = None
    ) -> bool:
        """
        Validate extraction results against custom rules.
        
        Args:
            result: Extraction result to validate
            validation_rules: Dictionary of validation functions
            
        Returns:
            True if all validations pass
        """
        if not result.success:
            return False
        
        # Default validation rules
        if not validation_rules:
            validation_rules = {
                "has_answer": lambda r: len(r.answer) > 0,
                "has_sources": lambda r: len(r.source_pages) > 0
            }
        
        # Apply validation rules
        for rule_name, rule_func in validation_rules.items():
            try:
                if not rule_func(result):
                    logger.warning(f"Validation failed: {rule_name}")
                    return False
            except Exception as e:
                logger.error(f"Validation error in {rule_name}: {e}")
                return False
        
        return True
    
    def get_statistics(self, results: List[FinalExtractionResult]) -> Dict[str, Any]:
        """
        Generate statistics from extraction results.
        
        Args:
            results: List of extraction results
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        
        stats = {
            "total_processed": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "average_processing_time_ms": sum(r.processing_time_ms for r in successful) / len(successful) if successful else 0,
            "total_pages_processed": sum(len(r.source_pages) for r in successful),
            "extraction_methods": {}
        }
        
        # Count extraction methods
        for result in successful:
            method = result.extraction_metadata.get("extraction_method", "unknown")
            stats["extraction_methods"][method] = stats["extraction_methods"].get(method, 0) + 1
        
        return stats