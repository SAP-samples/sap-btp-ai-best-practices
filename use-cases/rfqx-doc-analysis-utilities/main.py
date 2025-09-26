"""
Main interface for the Simplified RAG RFQ comparator.

This module provides the main interface for processing RFQ documents using
direct PDF-to-LLM processing with GPT-4.1's large context window.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from document_processor import DocumentProcessor, extract_pdf_content, check_context_limits
from file_processor import FileProcessor
from llm_client import SimplifiedRAGClient
from rfq_schema import (RFQ_EXTRACTION_SCHEMA, get_extraction_instructions, 
                       create_dynamic_schema, get_dynamic_extraction_instructions)


class SimplifiedRFQComparator:
    """
    Enhanced RFQ comparator supporting multi-provider, multi-format processing.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.client = SimplifiedRAGClient()
        self.extraction_schema = RFQ_EXTRACTION_SCHEMA
        self.document_processor = DocumentProcessor()
        self.file_processor = FileProcessor()
        
        print("Initialized Enhanced SimplifiedRFQComparator")
        print("Supports multi-provider processing with PDF, Excel, and CSV documents")
    
    def process_single_document(self, pdf_path: str, 
                                  extraction_method: str = "pdfplumber") -> Dict[str, Any]:
        """
        Process a single RFQ document and extract structured information.
        
        Args:
            pdf_path: Path to the PDF document
            extraction_method: PDF extraction method ("pymupdf" or "pdfplumber")
            
        Returns:
            Extracted RFQ information
        """
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {Path(pdf_path).name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Extract PDF content
            print("\n1. Extracting PDF content...")
            pdf_content = extract_pdf_content(pdf_path, method=extraction_method)
            
            # Step 2: Check context limits
            print("\n2. Checking context limits...")
            limits_check = check_context_limits(pdf_content)
            
            if limits_check["within_limits"]:
                print(f"Content fits within context limits ({limits_check['utilization_percentage']:.1f}% utilization)")
            else:
                print(f"Content exceeds limits - will be optimized")
                for rec in limits_check["recommendations"]:
                    print(f"   - {rec}")
            
            # Step 3: Extract RFQ information using GPT-4.1
            print("\n3. Extracting RFQ information with GPT-4.1...")
            extracted_data = self.client.extract_rfq_information(
                pdf_content, self.extraction_schema
            )
            
            # Step 4: Validate extraction
            if "extraction_error" in extracted_data:
                print(f"❌ Extraction failed: {extracted_data['extraction_error']}")
                return extracted_data
            
            # Count successful extractions
            total_fields = sum(len(fields) for fields in self.extraction_schema.values())
            found_fields = sum(1 for category in self.extraction_schema.keys() 
                             if category in extracted_data
                             for field in self.extraction_schema[category].keys()
                             if extracted_data.get(category, {}).get(field, "Not Found") != "Not Found")
            
            success_rate = (found_fields / total_fields) * 100
            
            print(f"\nExtraction completed!")
            print(f"   - Found information for {found_fields}/{total_fields} fields ({success_rate:.1f}%)")
            print(f"   - Model: {extracted_data.get('_metadata', {}).get('extraction_model', 'unknown')}")
            print(f"   - Response tokens: {extracted_data.get('_metadata', {}).get('response_tokens', 'unknown')}")
            
            return extracted_data
            
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return {
                "processing_error": str(e),
                "_metadata": {
                    "source_document": Path(pdf_path).name,
                    "error_stage": "document_processing"
                }
            }
    
    def compare_documents(self, pdf_paths: List[str], 
                         output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple RFQ documents using the simplified RAG approach.
        
        Args:
            pdf_paths: List of paths to PDF documents to compare
            output_file: Optional path to save comparison results
            
        Returns:
            Comparison results
        """
        
        if len(pdf_paths) < 2:
            return {"error": "At least 2 documents required for comparison"}
        
        print(f"\n{'='*80}")
        print(f"COMPARING RFQ DOCUMENTS")
        print(f"{'='*80}")
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"Document {i}: {Path(pdf_path).name}")
        print(f"{'='*80}")
        
        try:
            # Process all documents
            processed_docs = []
            for i, pdf_path in enumerate(pdf_paths, 1):
                print(f"\nProcessing Document {i}...")
                doc_data = self.process_single_document(pdf_path)
                
                # Check for processing errors
                if "processing_error" in doc_data or "extraction_error" in doc_data:
                    print(f"❌ Failed to process Document {i}")
                    return {"error": f"Document {i} processing failed", f"doc{i}_error": doc_data}
                
                processed_docs.append(doc_data)
            
            # Generate comparison
            print("\nGenerating comparison...")
            comparison_result = self._generate_detailed_comparison(processed_docs)
            
            # Save results if output file specified
            if output_file:
                self._save_comparison_results(comparison_result, output_file)
                print(f"\nResults saved to: {output_file}")
            
            return comparison_result
            
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            return {"comparison_error": str(e)}
    
    def _generate_detailed_comparison(self, docs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed comparison between multiple extracted datasets."""
        
        num_docs = len(docs_data)
        
        # Create structured comparison
        comparison = {
            "metadata": {
                "comparison_date": datetime.now().isoformat(),
                "number_of_documents": num_docs,
                "extraction_model": docs_data[0].get("_metadata", {}).get("extraction_model", "unknown")
            },
            "field_by_field_comparison": {},
            "summary_statistics": {},
            "key_differences": []
        }
        
        # Add document names to metadata
        for i, doc_data in enumerate(docs_data, 1):
            comparison["metadata"][f"document_{i}"] = doc_data.get("_metadata", {}).get("source_document", f"Document {i}")
        
        # Field-by-field comparison
        total_fields = 0
        doc_completeness = [0] * num_docs  # Track completeness for each document
        
        for category_name, fields in self.extraction_schema.items():
            comparison["field_by_field_comparison"][category_name] = {}
            
            for field_name in fields.keys():
                total_fields += 1
                
                # Get values from all documents
                field_comparison = {}
                doc_values = []
                for i, doc_data in enumerate(docs_data, 1):
                    value = doc_data.get(category_name, {}).get(field_name, "Not Found")
                    field_comparison[f"document_{i}"] = value
                    doc_values.append(value)
                    
                    # Track completeness
                    if value != "Not Found":
                        doc_completeness[i-1] += 1
                
                comparison["field_by_field_comparison"][category_name][field_name] = field_comparison
                
                # Detect differences across all documents
                unique_values = set(doc_values)
                if len(unique_values) > 1 or "Not Found" in unique_values:
                    difference_entry = {
                        "field": f"{category_name}.{field_name}",
                        "difference_type": "variation_across_documents"
                    }
                    for i, value in enumerate(doc_values, 1):
                        difference_entry[f"document_{i}"] = value
                    comparison["key_differences"].append(difference_entry)
        
        # Summary statistics
        stats = {
            "total_fields": total_fields,
            "number_of_documents": num_docs
        }
        
        # Calculate completeness for each document
        for i in range(num_docs):
            doc_name = f"doc{i+1}_completeness_percentage"
            stats[doc_name] = (doc_completeness[i] / total_fields) * 100
        
        # Calculate fields found in all documents
        found_in_all = 0
        for category_name, fields in self.extraction_schema.items():
            for field_name in fields.keys():
                field_data = comparison["field_by_field_comparison"][category_name][field_name]
                if all(field_data[f"document_{i}"] != "Not Found" for i in range(1, num_docs + 1)):
                    found_in_all += 1
        
        stats["found_in_all_documents"] = found_in_all
        stats["not_found_in_any"] = total_fields - max(doc_completeness)
        
        comparison["summary_statistics"] = stats
        
        # Print summary
        print(f"\nCOMPARISON SUMMARY:")
        print(f"   - Total fields compared: {stats['total_fields']}")
        print(f"   - Found in all documents: {stats['found_in_all_documents']}")
        for i in range(num_docs):
            doc_name = comparison["metadata"][f"document_{i+1}"]
            completeness = stats[f"doc{i+1}_completeness_percentage"]
            print(f"   - {doc_name} completeness: {completeness:.1f}%")
        
        return comparison
    
    def _save_comparison_results(self, comparison: Dict[str, Any], output_file: str):
        """Save comparison results to file."""
        
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.json':
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
        else:
            # Save as markdown report
            self._generate_markdown_report(comparison, output_path)
    
    def _generate_markdown_report(self, comparison: Dict[str, Any], output_path: Path):
        """Generate a markdown comparison report for multiple documents."""
        
        metadata = comparison["metadata"]
        stats = comparison["summary_statistics"]
        field_comparison = comparison["field_by_field_comparison"]
        
        num_docs = metadata["number_of_documents"]
        doc_names = [metadata[f"document_{i}"] for i in range(1, num_docs + 1)]
        
        print("\n==== GENERATING MARKDOWN REPORT ====")
        
        # Start building the markdown
        lines = [
            "# RFQ Document Comparison Report",
            "",
            f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"**Documents Compared:**"
        ]
        
        for i, doc_name in enumerate(doc_names, 1):
            lines.append(f"- Document {i}: {doc_name}")
        
        lines.extend(["", "---", ""])
        
        # Add comparison tables for each category
        for category_name, features in field_comparison.items():
            # Create table header
            header_row = "| Feature |"
            separator_row = "|---------|"
            
            for doc_name in doc_names:
                header_row += f" {doc_name} |"
                separator_row += "-" * (len(doc_name) + 2) + "|"
            
            lines.extend([
                f"## {category_name}",
                "",
                header_row,
                separator_row
            ])
            
            # Add table rows
            for feature_name, values in features.items():
                row = f"| {feature_name} |"
                
                for i in range(1, num_docs + 1):
                    value = values.get(f"document_{i}", "Not Found")
                    
                    # Format "Not Found" in italics
                    if value == "Not Found":
                        value = "*Not Found*"
                    
                    # Escape pipe characters in values
                    value = str(value).replace("|", "\\|")
                    row += f" {value} |"
                
                lines.append(row)
            
            lines.extend(["", "---", ""])
        
        # Add summary statistics
        lines.extend([
            "## Summary Statistics",
            "",
            f"- **Total Attributes Compared:** {stats['total_fields']}",
            f"- **Found in All Documents:** {stats['found_in_all_documents']}"
        ])
        
        # Add completeness for each document
        for i in range(num_docs):
            doc_name = doc_names[i]
            completeness = stats[f"doc{i+1}_completeness_percentage"]
            lines.append(f"- **{doc_name} Completeness:** {completeness:.1f}%")
        
        lines.append("")
        
        markdown_report = "\n".join(lines)
        print(f"Generated markdown report with {len(lines)} lines")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
    
    def answer_query(self, pdf_path: str, query: str) -> str:
        """
        Answer a specific query about an RFQ document.
        
        Args:
            pdf_path: Path to the PDF document
            query: Question to answer
            
        Returns:
            Answer to the query
        """
        
        print(f"\nAnswering query about {Path(pdf_path).name}")
        print(f"Query: {query}")
        
        try:
            # Extract PDF content
            pdf_content = extract_pdf_content(pdf_path)
            
            # Get answer from GPT-4.1
            answer = self.client.answer_specific_query(pdf_content, query)
            
            print(f"\nAnswer: {answer}")
            return answer
            
        except Exception as e:
            error_msg = f"Error answering query: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def process_provider_documents(self, uploaded_files: List, provider_name: str,
                                 custom_features: List[str] = None,
                                 enable_dynamic_extraction: bool = False,
                                 filtered_schema: Dict[str, Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process multiple documents for a single provider.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            provider_name: Name of the provider
            custom_features: List of custom features to extract
            enable_dynamic_extraction: Whether to enable dynamic feature extraction
            
        Returns:
            Processing results for the provider
        """
        
        print(f"\n{'='*60}")
        print(f"PROCESSING PROVIDER: {provider_name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Process all uploaded files
            print(f"\n1. Processing {len(uploaded_files)} files...")
            processed_files = []
            
            for uploaded_file in uploaded_files:
                print(f"   Processing: {uploaded_file.name}")
                file_result = self.file_processor.process_uploaded_file(uploaded_file)
                processed_files.append(file_result)
                
                if file_result.get('error'):
                    print(f"   Error: {file_result['error']}")
                else:
                    print(f"   Success: {file_result['token_count']:,} tokens")
            
            # Step 2: Aggregate documents
            print(f"\n2. Aggregating documents for {provider_name}...")
            aggregated_content = self.file_processor.aggregate_provider_documents(
                processed_files, provider_name
            )
            
            if aggregated_content['valid_files'] == 0:
                return {
                    "provider_name": provider_name,
                    "processing_error": "No valid files could be processed",
                    "errors": aggregated_content.get('errors', []),
                    "_metadata": {
                        "total_files": len(uploaded_files),
                        "valid_files": 0
                    }
                }
            
            print(f"   Aggregated {aggregated_content['valid_files']} files")
            print(f"   Total content: {aggregated_content['token_count']:,} tokens")
            
            # Step 3: Use filtered schema or create dynamic schema
            print(f"\n3. Preparing extraction schema...")
            if filtered_schema:
                # Use the filtered schema from the UI
                extraction_schema = filtered_schema.copy()
                print(f"   Using filtered schema with selected attributes")
                
                # Add dynamic extraction if enabled
                if enable_dynamic_extraction:
                    extraction_schema["dynamically_fetched_features"] = {
                        "additional_attributes": "string",
                        "unique_requirements": "string", 
                        "special_conditions": "string",
                        "other_relevant_information": "string"
                    }
                    print(f"   Added dynamic extraction category")
            else:
                # Fallback to creating dynamic schema (for backward compatibility)
                extraction_schema = create_dynamic_schema(
                    custom_features=custom_features,
                    include_dynamic_extraction=enable_dynamic_extraction
                )
                print(f"   Created dynamic schema")
            
            schema_summary = len(extraction_schema)
            total_fields = sum(len(fields) for fields in extraction_schema.values())
            print(f"   Final schema: {schema_summary} categories, {total_fields} fields")
            
            if not extraction_schema:
                return {
                    "provider_name": provider_name,
                    "processing_error": "No attributes selected for extraction",
                    "_metadata": {
                        "total_files": len(uploaded_files),
                        "valid_files": aggregated_content['valid_files'],
                        "aggregated_tokens": aggregated_content['token_count']
                    }
                }
            
            # Step 4: Transform aggregated content to LLM client format
            print(f"\n4. Preparing content for extraction...")
            llm_compatible_content = {
                'filename': f"{provider_name}_aggregated_documents",
                'full_text': aggregated_content['content'],
                'token_count': aggregated_content['token_count'],
                'page_count': aggregated_content.get('valid_files', 1),  # Use file count as page equivalent
                'file_count': aggregated_content['valid_files'],
                'provider_name': provider_name
            }
            
            # Step 5: Extract information using extraction schema
            print(f"5. Extracting RFQ information...")
            extracted_data = self.client.extract_rfq_information(
                llm_compatible_content, extraction_schema
            )
            
            # Step 6: Validate extraction
            if "extraction_error" in extracted_data:
                print(f"❌ Extraction failed: {extracted_data['extraction_error']}")
                return {
                    "provider_name": provider_name,
                    "extraction_error": extracted_data['extraction_error'],
                    "_metadata": {
                        "total_files": len(uploaded_files),
                        "valid_files": aggregated_content['valid_files'],
                        "aggregated_tokens": aggregated_content['token_count']
                    }
                }
            
            # Count successful extractions
            total_fields = sum(len(fields) for fields in extraction_schema.values())
            found_fields = sum(1 for category in extraction_schema.keys() 
                             if category in extracted_data
                             for field in extraction_schema[category].keys()
                             if extracted_data.get(category, {}).get(field, "Not Found") != "Not Found")
            
            success_rate = (found_fields / total_fields) * 100
            
            print(f"\nExtraction completed!")
            print(f"   Found information for {found_fields}/{total_fields} fields ({success_rate:.1f}%)")
            print(f"   🤖 Model: {extracted_data.get('_metadata', {}).get('extraction_model', 'unknown')}")
            
            # Add provider metadata
            extracted_data["_metadata"]["provider_name"] = provider_name
            extracted_data["_metadata"]["total_files"] = len(uploaded_files)
            extracted_data["_metadata"]["valid_files"] = aggregated_content['valid_files']
            extracted_data["_metadata"]["aggregated_tokens"] = aggregated_content['token_count']
            extracted_data["_metadata"]["success_rate"] = success_rate
            extracted_data["_metadata"]["found_fields"] = found_fields
            extracted_data["_metadata"]["total_fields"] = total_fields
            
            # Preserve raw aggregated content for Document Chat
            extracted_data["_raw_content"] = aggregated_content['content']
            
            if aggregated_content.get('errors'):
                extracted_data["_metadata"]["file_errors"] = aggregated_content['errors']
            
            return extracted_data
            
        except Exception as e:
            print(f"❌ Error processing provider {provider_name}: {e}")
            return {
                "provider_name": provider_name,
                "processing_error": str(e),
                "_metadata": {
                    "total_files": len(uploaded_files),
                    "error_stage": "provider_processing"
                }
            }
    
    def compare_providers(self, providers_data: List[Dict[str, Any]], 
                         output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple providers using their extracted data.
        
        Args:
            providers_data: List of provider extraction results
            output_file: Optional path to save comparison results
            
        Returns:
            Comparison results
        """
        
        if len(providers_data) < 2:
            return {"error": "At least 2 providers required for comparison"}
        
        # Filter out providers with errors
        valid_providers = [p for p in providers_data 
                          if not any(key in p for key in ["processing_error", "extraction_error"])]
        
        if len(valid_providers) < 2:
            return {"error": "At least 2 valid providers required for comparison"}
        
        print(f"\n{'='*80}")
        print(f"COMPARING {len(valid_providers)} PROVIDERS")
        print(f"{'='*80}")
        for i, provider_data in enumerate(valid_providers, 1):
            provider_name = provider_data.get("_metadata", {}).get("provider_name", f"Provider {i}")
            print(f"Provider {i}: {provider_name}")
        print(f"{'='*80}")
        
        try:
            # Generate comparison using existing logic but adapted for providers
            print(f"\nGenerating provider comparison...")
            comparison_result = self._generate_provider_comparison(valid_providers)
            
            # Save results if output file specified
            if output_file:
                self._save_comparison_results(comparison_result, output_file)
                print(f"\nResults saved to: {output_file}")
            
            return comparison_result
            
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            return {"comparison_error": str(e)}
    
    def _generate_provider_comparison(self, providers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed comparison between multiple providers."""
        
        num_providers = len(providers_data)
        
        # Determine the schema from the first provider
        first_provider = providers_data[0]
        schema_keys = set()
        for category in first_provider.keys():
            if not category.startswith('_') and isinstance(first_provider[category], dict):
                schema_keys.add(category)
        
        # Create structured comparison
        comparison = {
            "metadata": {
                "comparison_date": datetime.now().isoformat(),
                "number_of_providers": num_providers,
                "number_of_documents": num_providers,  # Alias for backward compatibility
                "comparison_type": "multi_provider",
                "extraction_model": first_provider.get("_metadata", {}).get("extraction_model", "unknown")
            },
            "field_by_field_comparison": {},
            "summary_statistics": {},
            "key_differences": []
        }
        
        # Add provider names to metadata
        for i, provider_data in enumerate(providers_data, 1):
            provider_name = provider_data.get("_metadata", {}).get("provider_name", f"Provider {i}")
            comparison["metadata"][f"provider_{i}"] = provider_name
            comparison["metadata"][f"document_{i}"] = provider_name  # Alias for backward compatibility
        
        # Field-by-field comparison
        total_fields = 0
        provider_completeness = [0] * num_providers
        
        for category_name in schema_keys:
            if category_name not in comparison["field_by_field_comparison"]:
                comparison["field_by_field_comparison"][category_name] = {}
            
            # Get all possible fields from all providers in this category
            all_fields = set()
            for provider_data in providers_data:
                if category_name in provider_data and isinstance(provider_data[category_name], dict):
                    all_fields.update(provider_data[category_name].keys())
            
            for field_name in all_fields:
                total_fields += 1
                
                # Get values from all providers
                field_comparison = {}
                provider_values = []
                for i, provider_data in enumerate(providers_data, 1):
                    value = provider_data.get(category_name, {}).get(field_name, "Not Found")
                    field_comparison[f"provider_{i}"] = value
                    provider_values.append(value)
                    
                    # Track completeness
                    if value != "Not Found":
                        provider_completeness[i-1] += 1
                
                comparison["field_by_field_comparison"][category_name][field_name] = field_comparison
                
                # Detect differences across all providers
                unique_values = set(provider_values)
                if len(unique_values) > 1 or "Not Found" in unique_values:
                    difference_entry = {
                        "field": f"{category_name}.{field_name}",
                        "difference_type": "variation_across_providers"
                    }
                    for i, value in enumerate(provider_values, 1):
                        difference_entry[f"provider_{i}"] = value
                    comparison["key_differences"].append(difference_entry)
        
        # Summary statistics
        stats = {
            "total_fields": total_fields,
            "number_of_providers": num_providers
        }
        
        # Calculate completeness for each provider
        for i in range(num_providers):
            provider_key = f"provider{i+1}_completeness_percentage"
            doc_key = f"doc{i+1}_completeness_percentage"  # Alias for backward compatibility
            completeness_value = (provider_completeness[i] / total_fields) * 100 if total_fields > 0 else 0
            stats[provider_key] = completeness_value
            stats[doc_key] = completeness_value
        
        # Calculate fields found in all providers
        found_in_all = 0
        for category_name in schema_keys:
            if category_name in comparison["field_by_field_comparison"]:
                for field_name in comparison["field_by_field_comparison"][category_name]:
                    field_data = comparison["field_by_field_comparison"][category_name][field_name]
                    if all(field_data[f"provider_{i}"] != "Not Found" for i in range(1, num_providers + 1)):
                        found_in_all += 1
        
        stats["found_in_all_providers"] = found_in_all
        stats["found_in_all_documents"] = found_in_all  # Alias for backward compatibility
        stats["not_found_in_any"] = total_fields - max(provider_completeness) if provider_completeness else total_fields
        
        comparison["summary_statistics"] = stats
        
        # Print summary
        print(f"\nPROVIDER COMPARISON SUMMARY:")
        print(f"   - Total fields compared: {stats['total_fields']}")
        print(f"   - Found in all providers: {stats['found_in_all_providers']}")
        for i in range(num_providers):
            provider_name = comparison["metadata"][f"provider_{i+1}"]
            completeness = stats[f"provider{i+1}_completeness_percentage"]
            print(f"   - {provider_name} completeness: {completeness:.1f}%")
        
        return comparison


def main():
    """Main function for command-line usage."""
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python main.py compare <pdf1> <pdf2> [pdf3] ... -o <output_file>")
        print("  python main.py extract <pdf_file>")
        print("  python main.py query <pdf_file> \"<question>\"")
        return
    
    command = sys.argv[1]
    comparator = SimplifiedRFQComparator()
    
    if command == "compare" and len(sys.argv) >= 4:
        # Parse arguments to handle -o flag
        args = sys.argv[2:]
        pdf_files = []
        output_file = None
        
        # Look for -o flag
        if "-o" in args:
            o_index = args.index("-o")
            if o_index + 1 < len(args):
                output_file = args[o_index + 1]
                pdf_files = args[:o_index]
            else:
                print("Error: -o flag requires an output filename")
                return
        else:
            # No -o flag, treat all as PDF files (backward compatibility)
            pdf_files = args
        
        if len(pdf_files) < 2:
            print("Error: At least 2 PDF files are required for comparison")
            return
        
        result = comparator.compare_documents(pdf_files, output_file)
        
        if not output_file:
            print("\n" + "="*80)
            print("COMPARISON RESULTS")
            print("="*80)
            print(json.dumps(result, indent=2))
    
    elif command == "extract" and len(sys.argv) >= 3:
        pdf_file = sys.argv[2]
        result = comparator.process_single_document(pdf_file)
        
        print("\n" + "="*80)
        print("EXTRACTION RESULTS")
        print("="*80)
        print(json.dumps(result, indent=2))
    
    elif command == "query" and len(sys.argv) >= 4:
        pdf_file = sys.argv[2]
        query = " ".join(sys.argv[3:])  # Join all remaining args as query
        
        answer = comparator.answer_query(pdf_file, query)
        print(f"\nAnswer: {answer}")
        
    else:
        print("Invalid command or insufficient arguments")


if __name__ == "__main__":
    main()