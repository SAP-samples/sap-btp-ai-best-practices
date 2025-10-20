"""
Factory for creating LLM instances using native implementations.

This module provides a factory function to create LLM instances
based on model names, using native API implementations for better performance.
It maintains compatibility with the previous LangChain-based interface.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from .config import MODEL_CONFIGS, DEFAULT_MODEL, PROVIDER_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standard response object that mimics LangChain response structure."""
    content: str


class NativeOpenAIWrapper:
    """Wrapper for native OpenAI API that provides LangChain-compatible interface."""
    
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        from gen_ai_hub.proxy.native.openai import chat
        self.chat = chat
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        
    def invoke(self, messages: Union[List[Any], str]) -> LLMResponse:
        """
        Invoke the model with messages.
        
        Args:
            messages: Either a string prompt or list of message objects
            
        Returns:
            LLMResponse object with content attribute
        """
        # Convert to OpenAI message format
        if isinstance(messages, str):
            # Simple string prompt
            openai_messages = [{"role": "user", "content": messages}]
        else:
            # List of messages (potentially HumanMessage objects)
            openai_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    # Handle HumanMessage or similar objects
                    if isinstance(msg.content, str):
                        # Simple text message
                        openai_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg.content, list):
                        # Multimodal content (text + images)
                        content_parts = []
                        for part in msg.content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    content_parts.append({"type": "text", "text": part["text"]})
                                elif part.get("type") == "image_url":
                                    content_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": part["image_url"]["url"]}
                                    })
                        openai_messages.append({"role": "user", "content": content_parts})
                    else:
                        # Fallback for other content types
                        openai_messages.append({"role": "user", "content": str(msg.content)})
                else:
                    # Plain string in list
                    openai_messages.append({"role": "user", "content": str(msg)})
        
        # Make API call
        try:
            # Get model config
            config = MODEL_CONFIGS.get(self.model_name, {})
            
            # Build kwargs for API call
            api_kwargs = {
                "messages": openai_messages,
                "model": self.model_name,
            }
            
            # Add temperature only if model supports it
            if config.get("supports_temperature", True):
                api_kwargs["temperature"] = self.temperature
            
            # Add max_tokens if specified
            if "max_tokens" in self.kwargs:
                api_kwargs["max_tokens"] = self.kwargs["max_tokens"]
            
            # Add other supported parameters, filtering out provider-specific ones
            supported_params = ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop"]
            for param in supported_params:
                if param in self.kwargs:
                    api_kwargs[param] = self.kwargs[param]
            
            response = self.chat.completions.create(**api_kwargs)
            
            # Extract content from response
            content = response.choices[0].message.content
            return LLMResponse(content=content)
            
        except Exception as e:
            logger.error(f"Error invoking OpenAI model {self.model_name}: {str(e)}")
            raise


class NativeBedrockWrapper:
    """Wrapper for native Bedrock API that provides LangChain-compatible interface."""
    
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        from gen_ai_hub.proxy.native.amazon.clients import Session
        self.session = Session()
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        
    def invoke(self, messages: Union[List[Any], str]) -> LLMResponse:
        """
        Invoke the model with messages.
        
        Args:
            messages: Either a string prompt or list of message objects
            
        Returns:
            LLMResponse object with content attribute
        """
        # Create Bedrock client
        bedrock = self.session.client(model_name=self.model_name)
        
        # Convert to Bedrock message format
        if isinstance(messages, str):
            # Simple string prompt
            bedrock_messages = [{
                "role": "user",
                "content": [{"text": messages}]
            }]
        else:
            # List of messages
            bedrock_messages = []
            for msg in messages:
                content_parts = []
                
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        # Simple text message
                        content_parts.append({"text": msg.content})
                    elif isinstance(msg.content, list):
                        # Multimodal content
                        for part in msg.content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    content_parts.append({"text": part["text"]})
                                elif part.get("type") == "image_url":
                                    # Extract base64 data from data URL
                                    url = part["image_url"]["url"]
                                    if url.startswith("data:image"):
                                        # Extract base64 part
                                        base64_data = url.split(",")[1]
                                        content_parts.append({
                                            "image": {
                                                "format": "png",  # Assume PNG for now
                                                "source": {"bytes": base64_data}
                                            }
                                        })
                    else:
                        content_parts.append({"text": str(msg.content)})
                else:
                    content_parts.append({"text": str(msg)})
                
                if content_parts:
                    bedrock_messages.append({
                        "role": "user",
                        "content": content_parts
                    })
        
        # Make API call
        try:
            # Build inference config
            inference_config = {
                "temperature": self.temperature,
                # "maxTokens": self.kwargs.get("max_tokens", 1000)
            }
            
            response = bedrock.converse(
                messages=bedrock_messages,
                inferenceConfig=inference_config
            )
            
            # Extract content from response
            content = response['output']['message']['content'][0]['text']
            return LLMResponse(content=content)
            
        except Exception as e:
            logger.error(f"Error invoking Bedrock model {self.model_name}: {str(e)}")
            raise


class NativeVertexAIWrapper:
    """Wrapper for native Vertex AI (Gemini) API that provides LangChain-compatible interface."""
    
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel
        self.model = GenerativeModel(model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        
    def invoke(self, messages: Union[List[Any], str]) -> LLMResponse:
        """
        Invoke the model with messages.
        
        Args:
            messages: Either a string prompt or list of message objects
            
        Returns:
            LLMResponse object with content attribute
        """
        # Configure generation parameters
        generation_config = {
            "temperature": self.temperature,
            # "max_output_tokens": self.kwargs.get("max_tokens", 8000),
            "top_p": self.kwargs.get("top_p", 0.95),
            "top_k": self.kwargs.get("top_k", 40)
        }
        
        # Convert to Gemini content format
        if isinstance(messages, str):
            # Simple string prompt
            contents = messages
        else:
            # Handle list of messages
            contents = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        # Simple text message
                        contents.append(msg.content)
                    elif isinstance(msg.content, list):
                        # Multimodal content - build parts list
                        parts = []
                        for part in msg.content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    parts.append({"text": part["text"]})
                                elif part.get("type") == "image_url":
                                    # Extract base64 data from data URL
                                    url = part["image_url"]["url"]
                                    if url.startswith("data:image"):
                                        # Extract MIME type and base64 data
                                        mime_part = url.split(";")[0].replace("data:", "")
                                        base64_data = url.split(",")[1]
                                        parts.append({
                                            "inline_data": {
                                                "mime_type": mime_part,
                                                "data": base64_data
                                            }
                                        })
                        
                        # For Gemini, we need to structure the content properly
                        contents = [{
                            "role": "user",
                            "parts": parts
                        }]
                    else:
                        contents.append(str(msg.content))
                else:
                    contents.append(str(msg))
            
            # If contents is a list of strings, join them
            if isinstance(contents, list) and all(isinstance(c, str) for c in contents):
                contents = "\n".join(contents)
        
        # Make API call
        try:
            response = self.model.generate_content(
                contents=contents,
                generation_config=generation_config
            )
            
            # Extract content from response
            content = response.text
            return LLMResponse(content=content)
            
        except Exception as e:
            logger.error(f"Error invoking Vertex AI model {self.model_name}: {str(e)}")
            raise


def create_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> Any:
    """
    Create an LLM instance based on model name using native implementations.
    
    Args:
        model_name: Name of the model to use. If None, uses DEFAULT_MODEL
                   or LLM_MODEL environment variable.
        temperature: Temperature for generation (0.0 = deterministic)
        **kwargs: Additional model-specific parameters
        
    Returns:
        A native LLM wrapper instance with LangChain-compatible interface
        
    Raises:
        ValueError: If model_name is not supported
        ImportError: If required dependencies are not installed
    """
    # Determine model name
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {', '.join(MODEL_CONFIGS.keys())}"
        )
    
    # Get model configuration
    config = MODEL_CONFIGS[model_name]
    provider = config["provider"]
    
    # Handle model name aliases
    actual_model_name = config.get("actual_model_name", model_name)
    
    # Merge default configuration with provided kwargs
    provider_defaults = PROVIDER_DEFAULTS.get(provider, {})
    all_kwargs = {**provider_defaults, **kwargs}
    
    logger.info(
        f"Creating native LLM instance: model={actual_model_name}, "
        f"provider={provider}, temperature={temperature}"
    )
    
    try:
        if provider == "openai":
            return NativeOpenAIWrapper(
                model_name=actual_model_name,
                temperature=temperature,
                **all_kwargs
            )
        
        elif provider == "bedrock":
            return NativeBedrockWrapper(
                model_name=actual_model_name,
                temperature=temperature,
                **all_kwargs
            )
        
        elif provider == "vertexai":
            return NativeVertexAIWrapper(
                model_name=actual_model_name,
                temperature=temperature,
                **all_kwargs
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    except Exception as e:
        logger.error(f"Failed to create native LLM instance: {str(e)}")
        raise


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model is not supported
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_CONFIGS[model_name]