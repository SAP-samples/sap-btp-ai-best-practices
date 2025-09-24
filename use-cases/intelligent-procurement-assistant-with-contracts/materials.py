"""
Materials Search Module
Handles material data loading, vectorization, and intelligent material search
"""

import os
import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any
import json
import time
import random

#############################################################################
# CONFIGURATION
#############################################################################
embedding_model = "text-embedding-ada-002"
chat_model = "gpt-4.1"


#############################################################################
# AI FUNCTIONS (with SAP AI Hub or Mock)
#############################################################################
def get_embedding(text: str, use_sap_ai_hub: bool = False) -> List[float]:
    """Get embeddings - uses SAP AI Hub if available, otherwise mock"""
    if use_sap_ai_hub:
        return get_embedding_real(text)
    else:
        return get_embedding_mock(text)


def ask_llm(prompt: str, use_sap_ai_hub: bool = False) -> str:
    """Ask LLM - uses SAP AI Hub if available, otherwise smart mock"""
    if use_sap_ai_hub:
        return ask_llm_real(prompt)
    else:
        return ask_llm_smart_mock(prompt)


def get_embedding_real(text: str, model: str = embedding_model) -> List[float]:
    """Get embeddings using SAP AI Hub"""
    try:
        # Import here to avoid issues if not available
        from gen_ai_hub.proxy.native.openai import embeddings

        response = embeddings.create(model_name=model, input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"DEBUG: SAP AI Hub embedding error: {e}")
        return get_embedding_mock(text)


def ask_llm_real(prompt: str) -> str:
    """Ask LLM using SAP AI Hub"""
    try:
        # Import here to avoid issues if not available
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

        proxy_client = get_proxy_client("gen-ai-hub")
        llm = ChatOpenAI(proxy_model_name=chat_model, proxy_client=proxy_client)
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        print(f"DEBUG: SAP AI Hub LLM error: {e}")
        return ask_llm_smart_mock(prompt)


def get_embedding_mock(text: str) -> List[float]:
    """Mock embedding function - generates consistent embeddings based on text"""
    random.seed(hash(text) % 2**32)
    return [random.random() for _ in range(1536)]


def ask_llm_smart_mock(prompt: str) -> str:
    """Smart mock LLM that simulates intelligent material search"""
    prompt_lower = prompt.lower()

    # Extract material IDs from the prompt context
    if "materials database:" in prompt_lower:
        try:
            # Parse the materials from the prompt
            materials_section = prompt.split("Available Materials Database:")[1].split(
                "Instructions:"
            )[0]
            materials_data = json.loads(materials_section)

            # Determine search intent
            user_query = ""
            if "user query:" in prompt_lower:
                user_query = prompt.split('User Query: "')[1].split('"')[0].lower()

            # Check for completely unrelated queries that should return empty
            unrelated_terms = [
                "elephant",
                "dinosaur",
                "unicorn",
                "dragon",
                "spaceship",
                "castle",
                "rainbow",
                "magic",
            ]
            if any(term in user_query for term in unrelated_terms):
                return ""  # Return empty to indicate no matches

            # Find relevant materials based on query
            relevant_ids = []

            for material in materials_data:
                material_desc = material.get("description", "").lower()
                material_id = material.get("id", 0)

                # Smart matching logic
                if (
                    "laptop" in user_query
                    or "computer" in user_query
                    or "notebook" in user_query
                ):
                    if any(
                        term in material_desc
                        for term in [
                            "laptop",
                            "notebook",
                            "computer",
                            "thinkpad",
                            "elitebook",
                            "macbook",
                            "xps",
                        ]
                    ):
                        relevant_ids.append(material_id)

                elif "office" in user_query and "supplies" in user_query:
                    if any(
                        term in material_desc
                        for term in [
                            "paper",
                            "scissors",
                            "organizer",
                            "folder",
                            "supplies",
                        ]
                    ):
                        relevant_ids.append(material_id)

                elif "chair" in user_query or "furniture" in user_query:
                    if any(
                        term in material_desc for term in ["chair", "desk", "furniture"]
                    ):
                        relevant_ids.append(material_id)

                elif "printer" in user_query:
                    if any(
                        term in material_desc
                        for term in ["printer", "toner", "ink", "cartridge"]
                    ):
                        relevant_ids.append(material_id)

                elif "monitor" in user_query or "display" in user_query:
                    if any(
                        term in material_desc
                        for term in ["monitor", "display", "screen"]
                    ):
                        relevant_ids.append(material_id)

            # If no specific matches and it's a reasonable business query, do broader matching
            if not relevant_ids and not any(
                term in user_query for term in unrelated_terms
            ):
                for material in materials_data:
                    material_desc = material.get("description", "").lower()
                    material_id = material.get("id", 0)

                    # Check if any word from query is in material description
                    query_words = user_query.split()
                    if any(
                        word in material_desc for word in query_words if len(word) > 3
                    ):
                        relevant_ids.append(material_id)

            # Return top 5 most relevant IDs
            return ",".join(map(str, relevant_ids[:5]))

        except Exception as e:
            # Fallback to simple response
            return "0,1,2"

    return "1,2,3"


#############################################################################
# CORE FUNCTIONS
#############################################################################
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2:
        return 0.0

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


#############################################################################
# CONTRACT DATA PROCESSING
#############################################################################
@st.cache_data
def load_contracts_data() -> pd.DataFrame:
    """Load contracts CSV data"""
    file_paths = ["contracts.csv", "data/contracts.csv"]
    contracts_df = pd.DataFrame()

    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                contracts_df = pd.read_csv(file_path)
                print(f"DEBUG: Loaded {len(contracts_df)} contracts from {file_path}")
                break
            except Exception as e:
                print(f"DEBUG: Error reading {file_path}: {e}")
                continue

    return contracts_df


#############################################################################
# MATERIALS DATA PROCESSING
#############################################################################
@st.cache_data
def load_and_vectorize_materials(use_sap_ai_hub: bool = False) -> pd.DataFrame:
    """Load materials CSV and create embeddings for descriptions"""

    # Try multiple file locations
    file_paths = ["materials.csv", "data/materials.csv"]
    df = pd.DataFrame()

    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"DEBUG: Loaded {len(df)} materials from {file_path}")
                break
            except Exception as e:
                print(f"DEBUG: Error reading {file_path}: {e}")
                continue

    # If no file found, create sample data
    if df.empty:
        print("DEBUG: materials.csv not found, creating sample data...")
        sample_data = {
            "MATERIAL_ID": ["MAT001", "MAT002", "MAT003", "MAT004", "MAT005", "MAT006"],
            "MATERIAL_DESCRIPTION": [
                "ThinkPad Laptop Computer 15-inch",
                "Dell Laptop Workstation High Performance",
                "MacBook Pro Laptop 13-inch",
                "Office Chair Ergonomic Black",
                "Desk Organizer Set Wooden",
                "Printer Paper A4 White 500 sheets",
            ],
            "COMMODITY_CODE": [
                "Computer Hardware",
                "Computer Hardware",
                "Computer Hardware",
                "Office Furniture",
                "Office Supplies",
                "Office Supplies",
            ],
            "PRICE": [1200.00, 1500.00, 1800.00, 350.00, 45.00, 12.00],
            "UNIT_OF_MEASURE": ["EA", "EA", "EA", "EA", "EA", "PK"],
        }
        df = pd.DataFrame(sample_data)

    # Required columns check
    required_columns = ["MATERIAL_DESCRIPTION", "PRICE", "COMMODITY_CODE"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"DEBUG: Missing required columns: {missing_columns}")
        print(f"DEBUG: Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Create embeddings for material descriptions silently
    print("DEBUG: Creating embeddings for materials...")

    embeddings_list = []
    for i, description in enumerate(df["MATERIAL_DESCRIPTION"].fillna("")):
        if description.strip():
            embedding = get_embedding(description, use_sap_ai_hub)
            embeddings_list.append(embedding)
        else:
            embeddings_list.append([])

        if not use_sap_ai_hub:
            time.sleep(0.05)  # Simulate processing time for mock

    df["embedding"] = embeddings_list
    print("DEBUG: Material embeddings created successfully")

    return df


#############################################################################
# SEARCH FUNCTIONS
#############################################################################
def search_materials_with_llm(
    query: str,
    materials_df: pd.DataFrame,
    use_sap_ai_hub: bool = False,
    top_k: int = 10,
) -> pd.DataFrame:
    """Use LLM as the primary search engine to find relevant materials"""
    if materials_df.empty:
        return pd.DataFrame()

    # Create a comprehensive context of all materials for the LLM
    materials_context = []
    for idx, row in materials_df.iterrows():
        materials_context.append(
            {
                "id": idx,
                "description": row["MATERIAL_DESCRIPTION"],
                "commodity": row["COMMODITY_CODE"],
                "price": row["PRICE"],
            }
        )

    # Create a focused prompt for the LLM to act as a search engine
    search_prompt = f"""
You are an intelligent search engine for a procurement system. The user is looking for materials to purchase.

User Query: "{query}"

Available Materials Database:
{json.dumps(materials_context, indent=2)}

Instructions:
1. Analyze the user's query to understand their intent and requirements
2. From the materials database, identify the TOP {top_k} most relevant materials that match the user's needs
3. Consider semantic similarity, category matching, and practical relevance
4. Return ONLY the material IDs (the 'id' field) of the most relevant materials, ranked from most to least relevant
5. Be precise - if the user asks for "laptop", return only laptop/notebook computers, not printers or office supplies
6. Format your response as a simple comma-separated list of IDs, nothing else

Example response format: 1,5,12,8
"""

    try:
        # Get LLM response
        llm_response = ask_llm(search_prompt, use_sap_ai_hub).strip()

        # Parse the LLM response to get material IDs
        try:
            selected_ids = [
                int(id_str.strip())
                for id_str in llm_response.split(",")
                if id_str.strip().isdigit()
            ]
        except:
            print("DEBUG: LLM response format error, using fallback search")
            return search_materials_fallback(query, materials_df, use_sap_ai_hub, top_k)

        # Get the selected materials
        if selected_ids:
            selected_materials = materials_df.loc[selected_ids].copy()

            # Add relevance scores based on LLM ranking (higher rank = higher score)
            relevance_scores = []
            for idx in selected_materials.index:
                try:
                    rank_position = selected_ids.index(idx)
                    # Convert rank to relevance score (1.0 for first, decreasing)
                    relevance_score = 1.0 - (rank_position / len(selected_ids))
                    relevance_scores.append(relevance_score)
                except:
                    relevance_scores.append(0.5)

            selected_materials["relevance_score"] = relevance_scores

            return selected_materials.head(top_k)
        else:
            # No results found - return empty DataFrame
            return pd.DataFrame()

    except Exception as e:
        print(f"DEBUG: LLM search failed: {e}")
        return search_materials_fallback(query, materials_df, use_sap_ai_hub, top_k)


def search_materials_fallback(
    query: str,
    materials_df: pd.DataFrame,
    use_sap_ai_hub: bool = False,
    top_k: int = 10,
) -> pd.DataFrame:
    """Fallback vector search if LLM search fails"""
    query_embedding = get_embedding(query, use_sap_ai_hub)
    if not query_embedding:
        return pd.DataFrame()

    similarities = []
    for embedding in materials_df["embedding"]:
        if embedding:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append(similarity)
        else:
            similarities.append(0.0)

    materials_df_copy = materials_df.copy()
    materials_df_copy["relevance_score"] = similarities

    results = (
        materials_df_copy[materials_df_copy["relevance_score"] > 0.1]
        .sort_values("relevance_score", ascending=False)
        .head(top_k)
    )

    return results


#############################################################################
# MAIN INTERFACE FUNCTIONS
#############################################################################
def get_materials_data(use_sap_ai_hub: bool = False) -> pd.DataFrame:
    """Main function to get materials data with contract information - called from app.py"""
    # Load materials data
    materials_df = load_and_vectorize_materials(use_sap_ai_hub)

    if materials_df.empty:
        return materials_df

    # Load contracts data
    contracts_df = load_contracts_data()

    if not contracts_df.empty:
        # Merge materials with contracts based on COMMODITY_CODE and VENDOR_NAME
        materials_with_contracts = materials_df.merge(
            contracts_df[
                [
                    "CONTRACT_NUMBER",
                    "CONTRACT_VALUE",
                    "COMMODITY_CODE",
                    "VENDOR_NAME",
                    "EXHAUSTION_RATE",
                ]
            ],
            on=["COMMODITY_CODE", "VENDOR_NAME"],
            how="left",
        )

        print(
            f"DEBUG: Merged {len(materials_with_contracts)} materials with contract information"
        )
        return materials_with_contracts
    else:
        # If no contracts data, add empty contract columns
        materials_df["CONTRACT_NUMBER"] = None
        materials_df["CONTRACT_VALUE"] = None
        materials_df["EXHAUSTION_RATE"] = None
        print(
            "DEBUG: No contracts data found, returning materials without contract info"
        )
        return materials_df


def search_materials(
    query: str, materials_df: pd.DataFrame, use_sap_ai_hub: bool = False
) -> pd.DataFrame:
    """Main search function - called from app.py"""
    return search_materials_with_llm(query, materials_df, use_sap_ai_hub)


def get_material_info(materials_df: pd.DataFrame) -> Dict[str, Any]:
    """Get materials database info"""
    if materials_df.empty:
        return {"count": 0, "status": "empty"}

    return {
        "count": len(materials_df),
        "columns": list(materials_df.columns),
        "categories": (
            materials_df["COMMODITY_CODE"].unique().tolist()
            if "COMMODITY_CODE" in materials_df.columns
            else []
        ),
        "price_range": {
            "min": (
                float(materials_df["PRICE"].min())
                if "PRICE" in materials_df.columns
                else 0
            ),
            "max": (
                float(materials_df["PRICE"].max())
                if "PRICE" in materials_df.columns
                else 0
            ),
        },
        "status": "loaded",
    }
