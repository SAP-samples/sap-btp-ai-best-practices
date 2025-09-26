"""
LLM client for direct PDF content processing using GPT-4.1.

This module handles communication with GPT-4.1 for processing complete PDF documents
without chunking or embedding retrieval.
"""

import json
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the correct GenAI Hub interface
from gen_ai_hub.proxy.native.openai import chat
# Import the Country Risk Manager for risk analysis
from country_risk_manager import CountryRiskManager


class SimplifiedRAGClient:
    """
    Client for processing complete documents with GPT-4.1 using large context.
    """
    
    def __init__(self):
        """Initialize the client with GPT-4.1 configuration."""
        # Use GPT-4.1 (not GPT-4o) for maximum context window
        self.model_name = "gpt-4.1"  # Use the same model naming as existing interface
        self.max_tokens = 4096  # Response tokens
        self.temperature = 0.1  # Low temperature for factual extraction
        self.country_risk_manager = CountryRiskManager()
        self.max_retries = 3  # Maximum retry attempts
        
        print(f"Initialized SimplifiedRAGClient with model: {self.model_name}")
    
    def extract_rfq_information(self, pdf_content: Dict[str, Any], 
                               extraction_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract RFQ information from complete PDF content using GPT-4.1.
        
        Args:
            pdf_content: Complete PDF content from pdf_processor
            extraction_schema: Schema defining what information to extract
            
        Returns:
            Extracted information structured according to schema
        """
        
        print(f"Extracting RFQ information from {pdf_content['filename']}...")
        print(f"Document: {pdf_content['token_count']} tokens, {pdf_content['page_count']} pages")
        
        # Prepare the extraction prompt
        prompt = self._build_extraction_prompt(pdf_content, extraction_schema)
        system_prompt = self._get_system_prompt()
        
        try:
            # Use the GenAI Hub interface following the existing pattern
            for attempt in range(self.max_retries):
                try:
                    response = chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    content = response.choices[0].message.content.strip()
                    
                    # Parse JSON response with cleanup
                    try:
                        # Clean up common JSON formatting issues
                        cleaned_content = self._clean_json_response(content)
                        extracted_data = json.loads(cleaned_content)
                        
                        # Add metadata
                        extracted_data["_metadata"] = {
                            "source_document": pdf_content["filename"],
                            "extraction_model": self.model_name,
                            "source_tokens": pdf_content["token_count"],
                            "response_tokens": len(content.split()) * 0.75  # Rough token estimate
                        }
                        
                        return extracted_data
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON response on attempt {attempt + 1}: {e}")
                        if attempt == self.max_retries - 1:
                            # Return raw content if JSON parsing fails on final attempt
                            return {
                                "raw_response": content,
                                "extraction_error": f"JSON parsing failed: {e}",
                                "_metadata": {
                                    "source_document": pdf_content["filename"],
                                    "extraction_model": self.model_name,
                                    "source_tokens": pdf_content["token_count"]
                                }
                            }
                        continue  # Retry
                    
                except Exception as e:
                    print(f"LLM request attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"All {self.max_retries} attempts failed. Last error: {e}")
                    
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                
        except Exception as e:
            print(f"Error during extraction: {e}")
            return {
                "extraction_error": str(e),
                "_metadata": {
                    "source_document": pdf_content["filename"],
                    "extraction_model": self.model_name,
                    "source_tokens": pdf_content["token_count"]
                }
            }
    
    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response content to handle common formatting issues."""
        cleaned_content = content
        
        # Remove markdown code blocks if present
        if "```json" in cleaned_content:
            cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_content:
            # Try to extract JSON from any code block
            parts = cleaned_content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") or part.startswith("["):
                    cleaned_content = part
                    break
        
        return cleaned_content
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for RFQ information extraction."""
        
        return """You are an expert at extracting structured information from RFQ (Request for Quotation) documents.

Your task is to carefully read the complete document and extract specific information according to the provided schema. 

IMPORTANT INSTRUCTIONS:
1. Read the ENTIRE document carefully - information may be scattered across different sections
2. Extract information EXACTLY as it appears in the document
3. If information is not found, use "Not Found" as the value
4. For dates, extract in the format they appear (preserve original format)
5. For monetary amounts, include currency symbols and formatting as shown
6. For contact details, include all provided information (name, email, phone, etc.)
7. For technical specifications, extract the complete requirements
8. Always return valid JSON matching the requested schema

IMPORTANT: Respond with valid JSON only, no additional text.

Your extraction should be comprehensive and accurate, leveraging the full context of the document."""
    
    def _build_extraction_prompt(self, pdf_content: Dict[str, Any], 
                                schema: Dict[str, Any]) -> str:
        """Build the extraction prompt with PDF content and schema."""
        
        # Get optimized text content
        from document_processor import optimize_text_for_context, check_context_limits
        
        full_text = pdf_content["full_text"]
        
        # Check if we need to optimize for context
        limits_check = check_context_limits(pdf_content, max_tokens=120000)  # Conservative limit
        
        if not limits_check["within_limits"]:
            print(f"Optimizing text: {pdf_content['token_count']} tokens exceeds limit")
            full_text = optimize_text_for_context(full_text, target_tokens=100000)
        
        prompt = f"""Please extract RFQ information from the following document according to the provided schema.

DOCUMENT: {pdf_content['filename']}
PAGES: {pdf_content['page_count']}

EXTRACTION SCHEMA:
{json.dumps(schema, indent=2)}

DOCUMENT CONTENT:
{full_text}

Please extract all available information according to the schema and return as valid JSON. If any information is not found in the document, use "Not Found" as the value."""

        return prompt
    
    def compare_rfq_documents(self, doc1_data: Dict[str, Any], 
                             doc2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare extracted information from two RFQ documents.
        
        Args:
            doc1_data: Extracted data from first document
            doc2_data: Extracted data from second document
            
        Returns:
            Comparison results
        """
        
        print("Comparing extracted RFQ data...")
        
        try:
            # Create comparison prompt
            comparison_prompt = f"""Please provide a detailed comparison of these two RFQ documents.

DOCUMENT 1: {doc1_data.get('_metadata', {}).get('source_document', 'Document 1')}
{json.dumps({k: v for k, v in doc1_data.items() if not k.startswith('_')}, indent=2)}

DOCUMENT 2: {doc2_data.get('_metadata', {}).get('source_document', 'Document 2')}
{json.dumps({k: v for k, v in doc2_data.items() if not k.startswith('_')}, indent=2)}

Please provide:
1. Key similarities between the documents
2. Key differences between the documents  
3. Notable missing information in either document
4. Overall assessment of completeness for each document

Return your analysis as structured JSON."""

            system_prompt = "You are an expert at comparing RFQ documents and identifying key differences and similarities. IMPORTANT: Respond with valid JSON only, no additional text."
            
            try:
                response = chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": comparison_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    cleaned_content = self._clean_json_response(content)
                    comparison_result = json.loads(cleaned_content)
                    return comparison_result
                except json.JSONDecodeError:
                    return {"raw_comparison": content}
                    
            except Exception as e:
                raise Exception(f"LLM request failed: {e}")
                
        except Exception as e:
            print(f"Error during comparison: {e}")
            return {"comparison_error": str(e)}
    
    def answer_specific_query_streaming(self, pdf_content: Dict[str, Any], 
                                       query: str):
        """
        Answer a specific query about the PDF content with streaming support.
        
        Args:
            pdf_content: Complete PDF content
            query: Specific question to answer
            
        Yields:
            String chunks of the answer as they arrive
        """
        
        print(f"Answering query (streaming) about {pdf_content['filename']}: {query}")
        
        # Get optimized text content  
        from document_processor import optimize_text_for_context, check_context_limits
        
        full_text = pdf_content["full_text"]
        
        # Check if we need to optimize for context
        limits_check = check_context_limits(pdf_content, max_tokens=800000)
        
        if not limits_check["within_limits"]:
            full_text = optimize_text_for_context(full_text, target_tokens=800000)
        
        # Extract country information for risk context
        country_risk_context = ""
        # Try to find country information in the content
        country_indicators = ['country_of_contracting_authority', 'contracting authority country', 'location']
        found_country = None
        
        for indicator in country_indicators:
            if indicator in full_text.lower():
                # Simple pattern to extract country name after the indicator
                import re
                pattern = rf'{indicator}[:\s]+([A-Za-z\s]+?)(?:\n|,|\.|;|$)'
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    potential_country = match.group(1).strip()
                    # Validate if this looks like a country name
                    if len(potential_country.split()) <= 3 and len(potential_country) > 2:
                        found_country = potential_country
                        break
        
        if found_country:
            risk_context = self.country_risk_manager.get_risk_context_for_llm(found_country)
            if risk_context:
                country_risk_context = f"\n\nCOUNTRY RISK CONTEXT:\n{risk_context}\n"

        query_prompt = f"""Based on the following RFQ document{' and country risk information' if country_risk_context else ''}, please answer this specific question:

QUESTION: {query}

DOCUMENT: {pdf_content['filename']}

DOCUMENT CONTENT:
{full_text}
{country_risk_context}
Please provide a detailed and accurate answer based on the document content{' and country risk data' if country_risk_context else ''}. 

If the information is not available in the document, clearly state that it was not found; if it is possible to calculate it or infer it from the document, calculate it and give the result, stating that this is an stimation and cite the pages and documents used. When relevant to the question, consider country risk factors in your response. If asked to make evaluations or comparisons, do so based following the document content strictly and objectively, with strengths and weakneses. Give realistic and practical answers based on the information provided. Whenever possible, cite the document, page and section where the information was found, so that the user can verify the answer."""

        messages = [
            {"role": "system", "content": "You are an expert at analyzing RFQ documents and answering specific questions accurately."},
            {"role": "user", "content": query_prompt}
        ]

        try:
            # Try streaming first
            try:
                response = chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True  # Enable streaming
                )
                
                # Stream the response chunks
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content
                        elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                            # Fallback for different response format
                            content = chunk.choices[0].message.content
                            if content:
                                yield content
                
                return  # Successful streaming
                
            except Exception as stream_error:
                print(f"Streaming not supported or failed: {stream_error}")
                print("Falling back to regular generation...")
                
                # Fallback to regular generation if streaming fails
                response = chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                    # No stream parameter - regular generation
                )
                
                # Return the complete response as a single chunk
                answer = response.choices[0].message.content
                if answer:
                    yield answer.strip()
                    
        except Exception as e:
            error_msg = f"Error answering query: {e}"
            print(f"❌ {error_msg}")
            yield error_msg

    def answer_specific_query(self, pdf_content: Dict[str, Any], 
                             query: str) -> str:
        """
        Answer a specific query about the PDF content.
        
        Args:
            pdf_content: Complete PDF content
            query: Specific question to answer
            
        Returns:
            Answer to the query
        """
        
        print(f"Answering query about {pdf_content['filename']}: {query}")
        
        # Get optimized text content  
        from document_processor import optimize_text_for_context, check_context_limits
        
        full_text = pdf_content["full_text"]
        
        # Check if we need to optimize for context
        limits_check = check_context_limits(pdf_content, max_tokens=120000)
        
        if not limits_check["within_limits"]:
            full_text = optimize_text_for_context(full_text, target_tokens=100000)
        
        # Extract country information for risk context
        country_risk_context = ""
        # Try to find country information in the content
        country_indicators = ['country_of_contracting_authority', 'contracting authority country', 'location']
        found_country = None
        
        for indicator in country_indicators:
            if indicator in full_text.lower():
                # Simple pattern to extract country name after the indicator
                import re
                pattern = rf'{indicator}[:\s]+([A-Za-z\s]+?)(?:\n|,|\.|;|$)'
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    potential_country = match.group(1).strip()
                    # Validate if this looks like a country name
                    if len(potential_country.split()) <= 3 and len(potential_country) > 2:
                        found_country = potential_country
                        break
        
        if found_country:
            risk_context = self.country_risk_manager.get_risk_context_for_llm(found_country)
            if risk_context:
                country_risk_context = f"\n\nCOUNTRY RISK CONTEXT:\n{risk_context}\n"

        query_prompt = f"""Based on the following RFQ document{' and country risk information' if country_risk_context else ''}, please answer this specific question:

QUESTION: {query}

DOCUMENT: {pdf_content['filename']}

DOCUMENT CONTENT:
{full_text}
{country_risk_context}
Please provide a detailed and accurate answer based on the document content{' and country risk data' if country_risk_context else ''}. If the information is not available in the document, clearly state that it was not found. When relevant to the question, consider country risk factors in your response."""

        try:
            response = chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing RFQ documents and answering specific questions accurately."},
                    {"role": "user", "content": query_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            return f"Error answering query: {e}"
    
    def generate_completion_streaming(self, messages: list, temperature: float = None, max_tokens: int = None):
        """
        Generate a streaming completion for general LLM requests.
        
        Args:
            messages: List of message dictionaries for the chat completion
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Yields:
            String chunks of the response as they arrive
        """
        
        # Use instance defaults if not provided
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Try streaming first
            try:
                response = chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    stream=True  # Enable streaming
                )
                
                # Stream the response chunks
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content
                        elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                            # Fallback for different response format
                            content = chunk.choices[0].message.content
                            if content:
                                yield content
                
                return  # Successful streaming
                
            except Exception as stream_error:
                print(f"Streaming not supported or failed: {stream_error}")
                print("Falling back to regular generation...")
                
                # Fallback to regular generation if streaming fails
                response = chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok
                    # No stream parameter - regular generation
                )
                
                # Return the complete response as a single chunk
                content = response.choices[0].message.content
                if content:
                    yield content.strip()
                    
        except Exception as e:
            error_msg = f"Error generating completion: {e}"
            print(f"❌ {error_msg}")
            yield error_msg