# Context-Aware Chunking Module

## Overview

The context-aware chunking module is a critical component of the Knowledge Graph Creation Pipeline, implementing Phase 2 of the document processing workflow. This module intelligently divides documents into logically coherent chunks based on their structural boundaries, preserving contextual integrity for downstream processing.

The chunking system is designed to handle heterogeneous document types with different structural patterns:
- **PDF documents**: Split by page boundaries
- **Excel documents**: Split by sheet boundaries
- **Unknown formats**: Treated as single chunks

## Key Features

### 1. Automatic Document Type Detection
The module automatically detects document types through:
- File extension analysis (`.pdf`, `.xlsx`, `.xls`, `.xlsm`, `.xlsb`)
- Content pattern recognition using regex patterns
- Fallback to single-chunk processing for unknown formats

### 2. Structure-Preserving Chunking
- **PDF Documents**: Preserves page boundaries using `--- PAGE X ---` separators
- **Excel Documents**: Maintains sheet boundaries using `# Sheet X: SheetName` headers
- **Contextual Integrity**: Ensures that related information stays together within logical boundaries

### 3. Flexible Chunk Merging
Optional capability to merge small chunks for optimal LLM processing:
- Configurable maximum chunk size
- Maintains clear boundaries between merged content
- Preserves metadata traceability

## Document Type Handling

### PDF Document Processing

PDF documents are chunked based on page separators in the extracted text:

```python
# PDF page separator pattern
PDF_PAGE_PATTERN = r'--- PAGE (\d+) ---'
```

Each page becomes an individual chunk with metadata:
- `chunk_id`: `page_X` where X is the page number
- Preserves all content between page separators
- Handles content before the first page marker

### Excel Document Processing

Excel documents are chunked based on sheet headers:

```python
# Excel sheet header pattern
EXCEL_SHEET_PATTERN = r'# Sheet (\d+): ([^\n]+)'
```

Each sheet becomes a chunk with metadata:
- `chunk_id`: `sheet_SheetName` (sanitized sheet name)
- Includes the sheet header in the chunk content
- Preserves all content until the next sheet boundary

## Usage Examples

### Basic Usage

```python
from resources.kg_creation.chunking.context_chunker import ContextAwareChunker

# Initialize the chunker
chunker = ContextAwareChunker()

# Chunk a document (auto-detect type)
extraction_result = {
    'full_text': 'Document content with separators...',
    'filename': 'document.pdf'
}
chunks = chunker.chunk_document(extraction_result)

# Process chunks
for chunk in chunks:
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Text: {chunk.chunk_text[:100]}...")
    print(f"Source: {chunk.metadata.filename}")
```

### With Explicit Source Type

```python
# Explicitly specify source type
chunks = chunker.chunk_document(extraction_result, source_type='pdf')
```

### Using the Convenience Function

```python
from resources.kg_creation.chunking.context_chunker import chunk_extraction_result

# Basic chunking
chunks = chunk_extraction_result(extraction_result)

# With chunk merging
chunks = chunk_extraction_result(
    extraction_result,
    merge_small_chunks=True,
    max_chunk_size=5000  # Characters
)
```

### Merging Small Chunks

```python
# Merge chunks that are too small
original_chunks = chunker.chunk_document(extraction_result)
merged_chunks = chunker.merge_chunks(original_chunks, max_size=10000)

# Merged chunks have special IDs like:
# "document.pdf:merged_page_1_to_page_3"
```

## Configuration Options

### ContextAwareChunker Parameters

The main class requires no initialization parameters but provides these methods:

1. **chunk_document()**
   - `extraction_result`: Dictionary with 'full_text' and 'filename'
   - `source_type`: Optional hint ('pdf', 'excel', or None for auto-detect)

2. **merge_chunks()**
   - `chunks`: List of Chunk objects to merge
   - `max_size`: Maximum character count for merged chunks

### Convenience Function Parameters

The `chunk_extraction_result()` function accepts:
- `extraction_result`: Result from pdf_extractor or excel_extractor
- `source_type`: Optional source type hint
- `merge_small_chunks`: Boolean to enable chunk merging
- `max_chunk_size`: Maximum size for merged chunks (if merging enabled)

## Integration with the Pipeline

The chunking module integrates seamlessly with the Knowledge Graph Creation Pipeline:

### 1. Pipeline Flow
```
Document → Extraction → Chunking → KG Extraction → Unification → Export
```

### 2. Input Format
Expects extraction results from:
- `pdf_extractor.extract_pdf_content()`
- `excel_extractor.extract_excel_content()`
- `markdown_extractor.extract_markdown_content()`

### 3. Output Format
Produces `Chunk` objects with:
- `chunk_id`: Unique identifier (format: "filename:chunk_identifier")
- `chunk_text`: The actual text content
- `metadata`: SourceMetadata object with filename and chunk_id

### 4. Usage in kg_pipeline.py
```python
# Example from the pipeline
extraction_result = extract_pdf_for_kg_pipeline(file_path)
chunks = chunk_extraction_result(extraction_result, merge_small_chunks=False)

# Each chunk is then processed independently
for chunk in chunks:
    kg = extractor.extract_from_chunk(chunk, metadata)
```

## Data Structures

### Chunk Model
```python
class Chunk(BaseModel):
    chunk_id: str  # Unique identifier
    chunk_text: str  # The chunk content
    metadata: SourceMetadata  # Source information
```

### SourceMetadata Model
```python
class SourceMetadata(BaseModel):
    filename: str  # Source file name
    chunk_id: str  # Chunk identifier within the file
```

## Best Practices

1. **Preserve Document Structure**: The chunker maintains natural document boundaries to preserve context

2. **Chunk Size Optimization**: Consider merging very small chunks (< 500 characters) for better LLM processing efficiency

3. **Metadata Tracking**: Always preserve metadata for traceability back to source documents

4. **Error Handling**: The module validates extraction results and provides clear error messages

5. **Logging**: Enable logging to track chunking decisions and performance:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

## Performance Considerations

- **Memory Efficiency**: Processes documents sequentially to avoid loading entire documents into memory
- **Regex Performance**: Uses compiled regex patterns for efficient pattern matching
- **Scalability**: Designed to handle documents with thousands of pages/sheets

## Error Handling

The module includes robust error handling for:
- Invalid extraction results (missing 'full_text')
- Empty documents
- Malformed separators
- Unknown document types (fallback to single chunk)

## Future Enhancements

Potential improvements for future versions:
- Support for additional document types (Word, PowerPoint)
- Semantic chunking based on content analysis
- Configurable chunk overlap for context preservation
- Smart chunking based on token limits for specific LLMs
- Custom separator patterns for specialized document formats