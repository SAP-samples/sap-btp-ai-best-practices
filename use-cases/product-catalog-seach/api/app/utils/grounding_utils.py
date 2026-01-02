"""
Utility functions for grounding operations with AI Core services.
Contains functions for authentication, data retrieval, and API interactions.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, Tuple
from fastapi import HTTPException
import json

logger = logging.getLogger(__name__)

# AICORE Configuration
AICORE_AUTH_URL = os.getenv("AICORE_AUTH_URL")
AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID")
AICORE_CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET")
AICORE_BASE_URL = os.getenv("AICORE_BASE_URL")
AICORE_RESOURCE_GROUP = "grounding"
ORCHESTRATION_URL = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d5c3bb3db9836057/completion"


# === API Functions (Do Not Change) ===
def get_token():
    """Fetches an OAuth token from AICORE."""
    url = f"{AICORE_AUTH_URL}/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": AICORE_CLIENT_ID,
        "client_secret": AICORE_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    resp = requests.post(url, headers=headers, data=data)
    resp.raise_for_status()  # Raises an exception for HTTP errors
    return resp.json()["access_token"]


def list_pipelines(token):
    """Lists all available document grounding pipelines."""
    url = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/lm/document-grounding/pipelines"  # Specific URL from your script
    headers = {
        "Authorization": f"Bearer {token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_pipeline_details(token, pipeline_id):
    """Fetches detailed information for a specific pipeline."""
    url = f"{AICORE_BASE_URL}/lm/document-grounding/pipelines/{pipeline_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching pipeline {pipeline_id}: {e}")
        return None


def list_documents(token, collection_id):
    """Lists documents within a specific vector collection."""
    url = f"{AICORE_BASE_URL}/lm/document-grounding/vector/collections/{collection_id}/documents"
    headers = {
        "Authorization": f"Bearer {token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def list_collections(token):
    """Lists all available vector collections."""
    url = f"{AICORE_BASE_URL}/lm/document-grounding/vector/collections"
    headers = {
        "Authorization": f"Bearer {token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def call_grounding_completion(
    token: str,
    grounding_request: str,
    collection_id: str = "*",
    custom_prompt: str = None,
    max_chunk_count: int = 50,
) -> dict:
    """Calls the grounding completion service."""
    headers = {
        "Authorization": f"Bearer {token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
        "Content-Type": "application/json",
    }

    if custom_prompt is None:
        custom_prompt = (
            "You are a precise and reliable assistant. Using only the provided context, "
            "generate a concise and accurate summary relevant to the request. "
            "Do not infer or generate information beyond the given context. "
            "If the requested information is not available in the context, clearly state that. "
            "Request: {{ ?groundingRequest }} Context: {{ ?groundingOutput }}"
        )

    payload = {
        "orchestration_config": {
            "module_configurations": {
                "grounding_module_config": {
                    "type": "document_grounding_service",
                    "config": {
                        "filters": [
                            {
                                "id": "filter1",
                                "data_repositories": [
                                    collection_id
                                ],  # Can be "*" or a specific ID
                                "search_config": {"max_chunk_count": max_chunk_count},
                                "data_repository_type": "vector",
                            }
                        ],
                        "input_params": ["groundingRequest"],
                        "output_param": "groundingOutput",
                        "metadata_params": [
                            "title",
                            "source",
                            "webUrlPageNo",
                            "webUrl",
                        ],
                    },
                },
                "llm_module_config": {
                    "model_name": "gpt-4.1",  # As per your script
                    "model_params": {},
                    "model_version": "latest",
                },
                "templating_module_config": {
                    "template": [{"role": "user", "content": custom_prompt}],
                    "defaults": {},
                },
                "filtering_module_config": {  # As per your script
                    "input": {
                        "filters": [
                            {
                                "type": "azure_content_safety",
                                "config": {
                                    "Hate": 2,
                                    "SelfHarm": 2,
                                    "Sexual": 2,
                                    "Violence": 2,
                                },
                            }
                        ]
                    },
                    "output": {
                        "filters": [
                            {
                                "type": "azure_content_safety",
                                "config": {
                                    "Hate": 2,
                                    "SelfHarm": 2,
                                    "Sexual": 2,
                                    "Violence": 2,
                                },
                            }
                        ]
                    },
                },
            }
        },
        "input_params": {"groundingRequest": grounding_request},
    }

    resp = requests.post(ORCHESTRATION_URL, headers=headers, json=payload)
    resp.raise_for_status()
    result_json = resp.json()

    # Extracting results as per your script's latest version
    grounding_result_data = (
        result_json.get("module_results", {})
        .get("grounding", {})
        .get("data", {})
        .get("grounding_result", "")
    )
    llm_response_data = (
        result_json.get("module_results", {})
        .get("llm", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    return {
        "grounding_result": grounding_result_data,
        "llm_response": llm_response_data,
    }


# === Helper Functions for UI ===
def get_pipeline_path(pipeline_resource, token=None):
    """Extracts includePath from pipeline resource."""
    try:
        # First try to get path from the main object (if it's already detailed)
        paths = (
            pipeline_resource.get("configuration", {})
            .get("sharePoint", {})
            .get("site", {})
            .get("includePaths", [])
        )
        if paths:
            return paths[0].lstrip("/")  # Remove leading slash

        # If no path found, make additional request for details
        if token and "id" in pipeline_resource:
            detailed_pipeline = get_pipeline_details(token, pipeline_resource["id"])
            if detailed_pipeline:
                paths = (
                    detailed_pipeline.get("configuration", {})
                    .get("sharePoint", {})
                    .get("site", {})
                    .get("includePaths", [])
                )
                if paths:
                    return paths[0].lstrip("/")  # Remove leading slash

        return "N/A"
    except Exception as e:
        print(f"Error extracting pipeline path: {e}")
        return "N/A"


def get_pipeline_id_from_collection_metadata(collection_resource):
    """Extracts the pipeline ID from collection metadata."""
    for meta in collection_resource.get("metadata", []):
        if meta.get("key") == "pipeline" and meta.get("value"):
            return meta["value"][0]
    return None


def extract_document_details(doc_resource):
    """Extracts title and timestamp from document metadata."""
    title = "N/A"
    timestamp = "N/A"
    for meta in doc_resource.get("metadata", []):
        if meta.get("key") == "title" and meta.get("value"):
            title = meta["value"][0]
        elif meta.get("key") == "timestamp" and meta.get(
            "value"
        ):  # Assuming 'timestamp' is the key for indexing time
            timestamp = meta["value"][0]
    return title, timestamp
