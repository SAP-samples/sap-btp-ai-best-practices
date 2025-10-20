# Knowledge Graph Creation - Document Extractors

## Overview

The extractor system is the entry point for the Knowledge Graph Creation Pipeline, responsible for extracting and preparing content from various document formats for downstream processing. These extractors implement Phase 1 of the KG Creation Pipeline: **Raw Text Extraction & Pre-processing**.

The system supports multiple document formats through specialized extractors, each optimized for its specific format while maintaining a consistent output interface for seamless integration with the rest of the pipeline.

## Architecture

### Core Extractors

1. **PDF Extractors** - Multiple strategies for PDF content extraction
2. **Excel Extractor** - Handles spreadsheet data with sheet preservation
3. **Markdown Extractor** - Direct processing of markdown files
4. **Image-based Extractor** - Direct image-to-knowledge-graph extraction

### Key Design Principles

- **Consistent Interface**: All extractors return standardized output format
- **Metadata Preservation**: Source tracking through `SourceMetadata` objects
- **Structural Context**: Maintains document structure (pages, sheets, sections)
- **Token Awareness**: Calculates token counts for LLM processing limits
- **Error Handling**: Comprehensive validation and error reporting

## Detailed Extractor Descriptions

### 1. PDF Extractors

The system provides three different PDF extraction strategies:

#### a) Basic PDF Extractor (`pdf_extractor.py`)
- **Purpose**: Standard text extraction with structural preservation
- **Methods**: PyMuPDF or pdfplumber
- **Features**:
  - Page boundary preservation with separators
  - Table detection and formatting (pdfplumber)
  - Text formatting preservation (PyMuPDF)
  - Metadata tracking per page

#### b) LLM-Enhanced PDF Extractor (`pdf_extractor_llm.py`)
- **Purpose**: Image-based extraction with AI transcription for better accuracy
- **Methods**: PDF-to-image conversion + Gemini transcription
- **Features**:
  - Superior handling of complex layouts
  - Accurate table transcription
  - Parallel page processing
  - Retry mechanism with exponential backoff
  - Automatic image optimization

#### c) Image-Direct PDF Extractor (`pdf_image_extractor.py`)
- **Purpose**: Converts PDFs to images for direct KG extraction
- **Methods**: PDF-to-image conversion without text transcription
- **Features**:
  - Bypasses text extraction entirely
  - Optimized for visual document analysis
  - Batch processing support
  - Memory-efficient image handling

### 2. Excel Extractor (`excel_extractor.py`)

- **Purpose**: Extract structured data from Excel files
- **Method**: Uses markitdown library for robust conversion
- **Features**:
  - Sheet boundary preservation
  - Table structure maintenance
  - NaN value cleanup
  - Markdown table formatting
  - Multi-sheet support with metadata

### 3. Markdown Extractor (`markdown_extractor.py`)

- **Purpose**: Process pre-formatted markdown documents
- **Method**: Direct file reading with metadata enhancement
- **Features**:
  - Minimal processing (already in target format)
  - Metadata wrapper for pipeline compatibility
  - Token counting
  - Single-chunk treatment

### 4. Image Knowledge Graph Extractor (`image_kg_extractor.py`)

- **Purpose**: Extract knowledge graphs directly from images
- **Method**: Multimodal LLM processing (Gemini)
- **Features**:
  - Direct image-to-KG extraction
  - TQDCS framework integration
  - Batch image processing
  - Structured output validation
  - Automatic entity deduplication

## When to Use Each Extractor

### PDF Processing Decision Tree

```
PDF Document
    ├─ Simple text-based PDF → Use pdf_extractor.py
    ├─ Complex layout/tables → Use pdf_extractor_llm.py
    └─ Direct KG extraction → Use pdf_image_extractor.py + image_kg_extractor.py
```


## Usage Examples

### 1. Basic PDF Extraction

```python
from resources.kg_creation.extractors.pdf_extractor import extract_pdf_for_kg_pipeline

# Extract PDF content
result = extract_pdf_for_kg_pipeline("documents/supplier_offer.pdf")

# Access extracted data
print(f"Pages: {result['page_count']}")
print(f"Words: {result['word_count']}")
print(f"Tokens: {result['token_count']}")

# Full text with page separators
full_text = result['full_text']

# Per-page access
for page in result['pages']:
    print(f"Page {page['page_number']}: {page['word_count']} words")
```

### 2. LLM-Enhanced PDF Extraction

```python
from resources.kg_creation.extractors.pdf_extractor_llm import extract_pdf_content

# Extract with image-based transcription
result = extract_pdf_content(
    file_path="complex_table_document.pdf",
    method="image-transcription",  # Always uses this method
    include_metadata=True
)

# Check context limits
from resources.kg_creation.extractors.pdf_extractor_llm import check_context_limits
limits = check_context_limits(result, max_tokens=100000)
if not limits['within_limits']:
    print(f"Warning: {limits['recommendations']}")
```

### 3. Excel Extraction

```python
from resources.kg_creation.extractors.excel_extractor import extract_excel_for_kg_pipeline

# Extract Excel content
result = extract_excel_for_kg_pipeline("data/cost_breakdown.xlsx")

# Access sheet data
for sheet in result['pages']:
    print(f"Sheet: {sheet['sheet_name']}")
    print(f"Tables found: {sheet['table_count']}")
    print(f"Content preview: {sheet['text'][:200]}...")
```

### 4. Direct Image-to-KG Extraction

```python
from resources.kg_creation.extractors.pdf_image_extractor import extract_pdf_content_for_nodes
from resources.kg_creation.extractors.image_kg_extractor import process_images_to_kg

# Step 1: Convert PDF to images
image_result = extract_pdf_content_for_nodes(
    file_path="visual_document.pdf",
    dpi=200,
    max_dimension=2048
)

# Step 2: Extract knowledge graph from images
kg = process_images_to_kg(
    image_extraction_result=image_result,
    llm_model="gemini-2.5-pro",
    temperature=0.1,
    parallel_processing=True,
    batch_size=3  # Process 3 pages together for context
)

# Access extracted entities
print(f"Extracted {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
```

### 5. Markdown Extraction

```python
from resources.kg_creation.extractors.markdown_extractor import extract_markdown_for_kg_pipeline

# Extract markdown content
result = extract_markdown_for_kg_pipeline("docs/technical_spec.md")

# Already in markdown format
markdown_content = result['full_text']
```

## Common Interfaces and Output Formats

### Standard Extraction Result

All extractors return a dictionary with this structure:

```python
{
    "filename": str,              # Original filename
    "file_path": str,             # Full path to source file
    "full_text": str,             # Complete extracted text/markdown
    "page_count": int,            # Number of pages/sheets
    "word_count": int,            # Total word count
    "token_count": int,           # Estimated LLM tokens
    "extraction_method": str,     # Method used
    "pages": List[Dict],          # Per-page/sheet data
    "extraction_timestamp": str,  # ISO format timestamp
    "source_metadata": List[SourceMetadata]  # Tracking metadata
}
```

### Per-Page/Sheet Structure

```python
{
    "page_number": int,           # 1-indexed page/sheet number
    "text": str,                  # Extracted content
    "character_count": int,       # Character count
    "word_count": int,            # Word count
    # Optional fields:
    "sheet_name": str,            # For Excel sheets
    "table_count": int,           # Number of tables found
    "dimensions": Dict,           # Page/image dimensions
    "has_text": bool,             # For image-based extraction
    "has_images": bool            # For image-based extraction
}
```

### SourceMetadata Schema

```python
class SourceMetadata(BaseModel):
    filename: str      # Source file name
    chunk_id: str      # Unique identifier (e.g., "page_1", "sheet_Sales")
```

## Performance Considerations

### Memory Usage

1. **PDF Processing**:
   - Basic extraction: Low memory, processes page-by-page
   - LLM extraction: Higher memory due to image conversion
   - Image-direct: Temporary storage for images

2. **Excel Processing**:
   - Entire file loaded into memory
   - Consider splitting very large files

### Processing Speed

1. **Fastest**: Markdown extraction (no conversion needed)
2. **Fast**: Basic PDF extraction (text-only)
3. **Moderate**: Excel extraction (depends on file size)
4. **Slowest**: LLM-enhanced extraction (API calls + image processing)

### Optimization Tips

1. **Parallel Processing**:
   ```python
   # Enable for LLM extraction
   result = extract_pdf_content(file_path, max_workers=10)
   
   # Enable for image KG extraction
   kg = process_images_to_kg(
       image_result, 
       parallel_processing=True,
       max_workers=5
   )
   ```

2. **Token Management**:
   ```python
   # Check before processing
   if result['token_count'] > 100000:
       # Consider chunking or optimization
       optimized = optimize_text_for_context(
           result['full_text'], 
           target_tokens=80000
       )
   ```

3. **Batch Processing**:
   ```python
   # For image-based KG extraction
   kg = process_images_to_kg(
       image_result,
       batch_size=5  # Process 5 pages together
   )
   ```

## Error Handling

All extractors implement consistent error handling:

1. **FileNotFoundError**: File doesn't exist
2. **ValueError**: Invalid file format or processing error
3. **Automatic retries**: LLM-based extractors retry on API failures
4. **Graceful degradation**: Falls back to simpler methods when needed

## Integration with KG Pipeline

The extractors are designed to feed directly into the next phases:

```python
# Phase 1: Extraction
extraction_result = extract_pdf_for_kg_pipeline("document.pdf")

# Phase 2: Chunking (uses extraction result)
chunks = create_chunks_from_extraction(extraction_result)

# Phase 3: Knowledge Graph Creation
kg = extract_knowledge_graph(chunks)
```

## Best Practices

1. **Choose the right extractor** based on document complexity
2. **Monitor token counts** to avoid LLM context limits
3. **Enable parallel processing** for large documents
4. **Preserve metadata** for source tracking
5. **Validate extraction results** before downstream processing
6. **Clean up temporary files** (automatic for most extractors)
7. **Handle errors gracefully** with appropriate fallbacks

## Future Enhancements

- OCR support for scanned documents without text layers
- Language detection and multi-language support
- Streaming extraction for very large files
- Custom extraction patterns for domain-specific documents
- Integration with document management systems