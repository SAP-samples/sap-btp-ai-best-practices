"""
LangGraph PDF Extraction Module

A robust PDF extraction system using LangGraph's map-reduce pattern
with parallel text and image processing for reliable information extraction.

Now includes SimpleImageExtractor for streamlined, direct image-based extraction
without LangGraph overhead.
"""

# Version of the module
__version__ = "1.1.0"

# Import main components when available
__all__ = [
    "PDFExtractor",
    "SimpleImageExtractor",
    "PDFExtractionRequest",
    "FinalExtractionResult",
    "__version__",
]

# These will be imported after implementation
try:
    from .pdf_extractor import PDFExtractor
    from .simple_pdf_extractor import SimpleImageExtractor
    from .schemas import PDFExtractionRequest, FinalExtractionResult
except ImportError:
    # Module components not yet fully implemented
    pass