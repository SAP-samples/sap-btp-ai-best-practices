"""
PDF Extraction Module

A streamlined PDF extraction system using direct image processing
for reliable information extraction with GPT-4 Vision.
"""

# Version of the module
__version__ = "1.1.0"

# Import main components
from .simple_pdf_extractor import SimpleImageExtractor

__all__ = [
    "SimpleImageExtractor",
    "__version__",
]