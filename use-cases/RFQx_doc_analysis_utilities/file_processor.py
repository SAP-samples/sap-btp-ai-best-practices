"""
Multi-format file processor for RFQ documents.

This module handles processing of PDF, Excel, and CSV files following 2024 best practices
for LLM integration. It focuses on statistical summaries and structured data representation
rather than raw data dumps to optimize token usage and maintain privacy.
"""

import io
import re
import pandas as pd
import tiktoken
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import fitz  # PyMuPDF
import pdfplumber

# Global tokenizer configuration for GPT-4
TOKENIZER_NAME = "cl100k_base"


class FileProcessor:
    """Handles multi-format file processing for RFQ documents."""
    
    def __init__(self):
        """Initialize the file processor."""
        self.supported_types = {
            'pdf': ['application/pdf'],
            'excel': [
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ],
            'csv': ['text/csv', 'application/csv']
        }
    
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded file and return structured content.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Dictionary with processed content and metadata
        """
        result = {
            'filename': uploaded_file.name,
            'file_type': self._detect_file_type(uploaded_file),
            'size_bytes': uploaded_file.size,
            'content': '',
            'metadata': {},
            'error': None,
            'token_count': 0
        }
        
        try:
            if result['file_type'] == 'pdf':
                result.update(self._process_pdf(uploaded_file))
            elif result['file_type'] == 'excel':
                result.update(self._process_excel(uploaded_file))
            elif result['file_type'] == 'csv':
                result.update(self._process_csv(uploaded_file))
            else:
                result['error'] = f"Unsupported file type: {uploaded_file.type}"
                return result
            
            # Calculate token count
            result['token_count'] = self._estimate_tokens(result['content'])
            
        except Exception as e:
            result['error'] = f"Error processing {uploaded_file.name}: {str(e)}"
        
        return result
    
    def aggregate_provider_documents(self, processed_files: List[Dict[str, Any]], 
                                   provider_name: str) -> Dict[str, Any]:
        """
        Aggregate multiple processed files into a single structured document for a provider.
        
        Args:
            processed_files: List of processed file dictionaries
            provider_name: Name of the provider
            
        Returns:
            Dictionary with aggregated content and metadata
        """
        # Filter out files with errors
        valid_files = [f for f in processed_files if f.get('error') is None]
        error_files = [f for f in processed_files if f.get('error') is not None]
        
        if not valid_files:
            return {
                'provider_name': provider_name,
                'content': '',
                'total_files': len(processed_files),
                'valid_files': 0,
                'error_files': len(error_files),
                'errors': [f['error'] for f in error_files],
                'token_count': 0
            }
        
        # Build structured markdown content
        content_parts = [
            f"# RFQ Documents for Provider: {provider_name}",
            "",
            f"**Document Summary:**",
            f"- Total files processed: {len(valid_files)}",
            f"- File types: {', '.join(set(f['file_type'] for f in valid_files))}",
            f"- Combined estimated tokens: {sum(f.get('token_count', 0) for f in valid_files):,}",
            ""
        ]
        
        # Add each document's content
        for i, file_data in enumerate(valid_files, 1):
            content_parts.extend([
                f"## Document {i}: {file_data['filename']}",
                f"**Type:** {file_data['file_type'].upper()}",
                f"**Size:** {file_data['size_bytes']:,} bytes",
                ""
            ])
            
            # Add metadata if available
            if file_data.get('metadata'):
                content_parts.append("**Metadata:**")
                for key, value in file_data['metadata'].items():
                    content_parts.append(f"- {key}: {value}")
                content_parts.append("")
            
            content_parts.extend([
                "**Content:**",
                file_data['content'],
                "",
                "---",
                ""
            ])
        
        aggregated_content = "\n".join(content_parts)
        
        return {
            'provider_name': provider_name,
            'content': aggregated_content,
            'total_files': len(processed_files),
            'valid_files': len(valid_files),
            'error_files': len(error_files),
            'errors': [f['error'] for f in error_files] if error_files else [],
            'token_count': self._estimate_tokens(aggregated_content),
            'file_details': valid_files
        }
    
    def _detect_file_type(self, uploaded_file) -> str:
        """Detect the type of uploaded file."""
        mime_type = uploaded_file.type
        
        if mime_type in self.supported_types['pdf']:
            return 'pdf'
        elif mime_type in self.supported_types['excel']:
            return 'excel'
        elif mime_type in self.supported_types['csv']:
            return 'csv'
        else:
            return 'unknown'
    
    def _process_pdf(self, uploaded_file) -> Dict[str, Any]:
        """Process a PDF file using existing logic."""
        try:
            # Use pdfplumber for better table extraction
            pdf_bytes = io.BytesIO(uploaded_file.getvalue())
            
            with pdfplumber.open(pdf_bytes) as pdf:
                pages_content = []
                total_tables = 0
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_parts = [f"### Page {page_num}"]
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        page_parts.append(text.strip())
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables, 1):
                        if table:
                            table_md = self._format_table_as_markdown(table, f"Table {table_num}")
                            page_parts.append(table_md)
                            total_tables += 1
                    
                    pages_content.append("\n\n".join(page_parts))
                
                content = "\n\n---\n\n".join(pages_content)
                
                return {
                    'content': content,
                    'metadata': {
                        'pages': len(pdf.pages),
                        'tables_found': total_tables,
                        'extraction_method': 'pdfplumber'
                    }
                }
        
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def _process_excel(self, uploaded_file) -> Dict[str, Any]:
        """Process an Excel file following 2024 best practices."""
        try:
            excel_bytes = io.BytesIO(uploaded_file.getvalue())
            
            # Get all sheet names
            excel_file = pd.ExcelFile(excel_bytes)
            sheet_names = excel_file.sheet_names
            
            content_parts = []
            total_rows = 0
            total_cols = 0
            
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_bytes, sheet_name=sheet_name)
                
                if df.empty:
                    content_parts.append(f"### Sheet: {sheet_name}\n*Empty sheet*")
                    continue
                
                # Generate statistical summary (2024 best practice)
                summary = self._generate_dataframe_summary(df, sheet_name)
                content_parts.append(summary)
                
                total_rows += len(df)
                total_cols = max(total_cols, len(df.columns))
            
            return {
                'content': "\n\n".join(content_parts),
                'metadata': {
                    'sheets': len(sheet_names),
                    'sheet_names': sheet_names,
                    'total_rows': total_rows,
                    'max_columns': total_cols,
                    'processing_method': 'statistical_summary'
                }
            }
        
        except Exception as e:
            raise Exception(f"Excel processing failed: {str(e)}")
    
    def _process_csv(self, uploaded_file) -> Dict[str, Any]:
        """Process a CSV file following 2024 best practices."""
        try:
            csv_bytes = io.BytesIO(uploaded_file.getvalue())
            df = pd.read_csv(csv_bytes)
            
            if df.empty:
                return {
                    'content': "*Empty CSV file*",
                    'metadata': {'rows': 0, 'columns': 0}
                }
            
            # Generate statistical summary
            summary = self._generate_dataframe_summary(df, "CSV Data")
            
            return {
                'content': summary,
                'metadata': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'processing_method': 'statistical_summary'
                }
            }
        
        except Exception as e:
            raise Exception(f"CSV processing failed: {str(e)}")
    
    def _generate_dataframe_summary(self, df: pd.DataFrame, data_name: str) -> str:
        """
        Generate a comprehensive summary of a DataFrame following 2024 best practices.
        This avoids sending full data to LLMs while preserving analytical value.
        """
        summary_parts = [
            f"### {data_name}",
            "",
            "**Data Overview:**",
            f"- Rows: {len(df):,}",
            f"- Columns: {len(df.columns)}",
            f"- Memory usage: ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            ""
        ]
        
        # Column information
        summary_parts.extend([
            "**Columns:**",
            ""
        ])
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            
            summary_parts.append(f"- **{col}** ({dtype}): {non_null:,} non-null, {null_count:,} null")
        
        summary_parts.append("")
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.extend([
                "**Numeric Data Summary:**",
                ""
            ])
            
            desc = df[numeric_cols].describe()
            summary_parts.append(desc.to_markdown())
            summary_parts.append("")
        
        # Sample data (first 5 rows) - privacy-conscious approach
        summary_parts.extend([
            "**Sample Data (First 5 rows):**",
            ""
        ])
        
        sample_df = df.head(5)
        summary_parts.append(sample_df.to_markdown(index=False))
        
        # Value counts for categorical columns (top 3 values only)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary_parts.extend([
                "",
                "**Categorical Data (Top 3 values per column):**",
                ""
            ])
            
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                top_values = df[col].value_counts().head(3)
                summary_parts.append(f"**{col}:**")
                for value, count in top_values.items():
                    summary_parts.append(f"- {value}: {count:,} occurrences")
                summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _format_table_as_markdown(self, table: List[List], title: str = "Table") -> str:
        """Format a table as markdown."""
        if not table or not table[0]:
            return f"**{title}:** *Empty table*"
        
        # Clean the table data
        cleaned_rows = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
            cleaned_rows.append(cleaned_row)
        
        if not cleaned_rows:
            return f"**{title}:** *No data*"
        
        # Create markdown table
        md_parts = [f"**{title}:**", ""]
        
        # Header row
        header = "| " + " | ".join(cleaned_rows[0]) + " |"
        separator = "|" + "|".join([" --- " for _ in cleaned_rows[0]]) + "|"
        
        md_parts.extend([header, separator])
        
        # Data rows (limit to 10 rows to avoid token explosion)
        for row in cleaned_rows[1:11]:  # Skip header, max 10 data rows
            md_row = "| " + " | ".join(row) + " |"
            md_parts.append(md_row)
        
        if len(cleaned_rows) > 11:
            md_parts.append(f"*... and {len(cleaned_rows) - 11} more rows*")
        
        return "\n".join(md_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for given text."""
        try:
            tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
            return len(tokenizer.encode(text))
        except Exception:
            # Fallback estimation: roughly 4 characters per token
            return len(text) // 4


def create_file_processor() -> FileProcessor:
    """Factory function to create a FileProcessor instance."""
    return FileProcessor()