"""
FastAPI router exposing endpoints for matching two CSV datasets by embeddings.

This version stores AI catalog embeddings in SAP HANA native vector store and
retrieves nearest neighbors via SQL (COSINE_SIMILARITY), then optionally
delegates ranking and reasoning to an LLM via SAP Gen AI Hub.
"""

import logging
import json
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.match import MatchRequest, MatchResponse, MatchPerClient
from ..utils.hana_vectors import (
    connect,
    create_temp_table,
    insert_rows,
    search_top_k,
    drop_table,
)


logger = logging.getLogger(__name__)

router = APIRouter()


def _concat_row(row: dict, selected_columns: List[str]) -> str:
    """Create a text string from a row using selected columns.

    This mirrors the UI's concat behavior so backend and frontend stay aligned.
    """

    parts: List[str] = []
    for column in selected_columns:
        value = row.get(column)
        if value is None:
            continue
        parts.append(f"{column}: {value}")
    return " - ".join(parts)


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts via SAP Gen AI Hub proxy.

    Falls back to a trivial random embedding if the service is unavailable, so
    that local development can proceed without credentials.
    """

    try:
        from gen_ai_hub.proxy.native.openai import embeddings  # type: ignore

        resp = embeddings.create(model="text-embedding-3-large", input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return vectors
    except Exception as e:  # pragma: no cover - best-effort local fallback
        logger.warning("Embedding service unavailable, using random embeddings: %s", e)
        rng = np.random.default_rng(42)
        # Use 3072 to match text-embedding-3-large default dimensionality
        return rng.normal(size=(len(texts), 3072)).astype(np.float32)


def _sanitize_texts(texts: List[str]) -> List[str]:
    """Replace empty/None texts with a safe placeholder to avoid API 400s.

    Some providers reject empty strings for embeddings. This ensures every
    entry has at least minimal content so the embedding API payload is valid.
    """

    sanitized: List[str] = []
    for text in texts:
        s = (text or "").strip()
        sanitized.append(s if s else "N/A")
    return sanitized


def _rank_with_llm(batch_items: List[dict], system_prompt: str, max_matches: int) -> List[List[List[str]]]:
    """Call LLM once for a batch to get ranked matches and reasons.

    Returns per-task results as list of [match, reason] pairs.
    """

    try:
        from gen_ai_hub.proxy.native.openai import chat  # type: ignore

        user_prompt = "Process the following matching tasks:\n\n"
        for idx, item in enumerate(batch_items):
            user_prompt += f"TASK {idx+1}:\n"
            user_prompt += f"Client Product:\n{item['client_text']}\n\n"
            user_prompt += "Candidates:\n"
            for c_idx, (candidate, name) in enumerate(zip(item["candidates"], item["candidate_names"])):
                user_prompt += f"Candidate {c_idx+1} - {name}:\n{candidate}\n\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = chat.completions.create(model_name="gpt-4o", messages=messages, temperature=0.0)
        output = response.choices[0].message.content.strip()

        # Parse format with TASK, MATCH N:, REASON N:
        task_results: dict[int, list[list[str]]] = {}
        current_task = None
        current_match_num = None
        current_match = None
        lines = output.split("\n")
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("TASK"):
                try:
                    num = int(line.split()[1].rstrip(":"))
                    current_task = num
                    task_results.setdefault(current_task, [])
                except Exception:
                    current_task = (current_task or 0) + 1
                    task_results.setdefault(current_task, [])
            elif line.upper().startswith("MATCH ") and ":" in line:
                try:
                    current_match_num = int(line.split(":", 1)[0].split()[1])
                except Exception:
                    current_match_num = len(task_results.get(current_task, [])) + 1
                current_match = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASON ") and ":" in line:
                if current_task is None or current_match is None:
                    continue
                reason = line.split(":", 1)[1].strip() or "No reason provided"
                if len(task_results[current_task]) < max_matches:
                    task_results[current_task].append([current_match, reason])
                current_match = None

        results: List[List[List[str]]] = []
        for i in range(1, len(batch_items) + 1):
            results.append(task_results.get(i, [["Error", f"Failed to parse result for task {i}"]]))
        return results
    except Exception as e:  # pragma: no cover - tolerate local dev without LLM
        logger.warning("LLM ranking unavailable, passing through candidates: %s", e)
        passthrough: List[List[List[str]]] = []
        for item in batch_items:
            pairs = [[name, "Nearest neighbor candidate"] for name in item["candidate_names"][:max_matches]]
            passthrough.append(pairs)
        return passthrough


@router.post("/match", response_model=MatchResponse)
def match_datasets(payload: MatchRequest) -> MatchResponse:
    """Match client rows to AI catalog rows and return ranked matches with reasons.

    The endpoint accepts pre-loaded rows and column selections from the UI, so
    the server can operate statelessly without file storage.
    """

    # Build combined text and sanitize to prevent invalid embedding inputs
    ai_texts = [_concat_row(row, payload.selected_ai_columns) for row in payload.ai_rows]
    client_texts = [_concat_row(row, payload.selected_client_columns) for row in payload.client_rows]
    ai_texts = _sanitize_texts(ai_texts)
    client_texts = _sanitize_texts(client_texts)

    # Compute embeddings in a single call so dimensions are guaranteed consistent
    combined_texts = ai_texts + client_texts
    combined_vectors = _embed_texts(combined_texts)
    ai_count = len(ai_texts)
    ai_vectors = combined_vectors[:ai_count]
    client_vectors = combined_vectors[ai_count:]

    # Defensive check in case provider returns inconsistent dims across calls
    if ai_vectors.shape[1] != client_vectors.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Embedding dimension mismatch: AI dim {ai_vectors.shape[1]} vs client dim {client_vectors.shape[1]}"
            ),
        )

    # Store AI vectors in HANA and retrieve nearest neighbors for each client
    k = max(payload.num_matches, 5)  # retrieve at least 5 to give LLM room
    cc = connect()
    table_name = create_temp_table(cc)
    try:
        # Insert AI rows: include candidate display name in metadata for later use
        ai_rows_payload = []
        for idx, text in enumerate(ai_texts):
            if payload.matching_column and payload.matching_column in payload.ai_rows[idx]:
                display_name = str(payload.ai_rows[idx][payload.matching_column])
            else:
                display_name = f"Candidate {idx}"
            metadata = {"name": display_name}
            ai_rows_payload.append((text, metadata, ai_vectors[idx].tolist()))
        insert_rows(cc, table_name, ai_rows_payload)

        # Prepare batches for LLM by querying HANA for each client vector
        batches = []
        for start in range(0, len(client_texts), payload.batch_size):
            end = min(start + payload.batch_size, len(client_texts))
            batch_items = []
            for i in range(start, end):
                results = search_top_k(
                    cc,
                    table_name,
                    client_vectors[i].tolist(),
                    k=k,
                    metric="COSINE_SIMILARITY",
                )
                candidate_texts = [t for (t, _m) in results]
                candidate_names: List[str] = []
                for _t, m in results:
                    if m:
                        try:
                            meta = json.loads(m)
                            candidate_names.append(str(meta.get("name", "Candidate")))
                        except Exception:
                            candidate_names.append("Candidate")
                    else:
                        candidate_names.append("Candidate")

                batch_items.append(
                    {
                        "client_id": i,
                        "client_text": client_texts[i],
                        "candidates": candidate_texts,
                        "candidate_names": candidate_names,
                    }
                )
            batches.append(batch_items)
    finally:
        # Best-effort cleanup
        try:
            drop_table(cc, table_name)
        finally:
            cc.connection.close()

    # Default prompt mirrors the UI template
    default_prompt = (
        "You are an expert classifier of AI products. You will be given multiple client products and for each one, a list of candidate matches.\n\n"
        "For EACH client product, select the BEST matching candidates in RANKED ORDER (from best to worst) and provide a reason for each match.\n\n"
        "Format your response using the following structure for each task:\n\n"
        "TASK 1:\n"
        "MATCH 1: [Best Product Name]\n"
        "REASON 1: [Reason for match]\n"
        "MATCH 2: [Second Best Product Name]\n"
        "REASON 2: [Reason for match]\n"
        "... and so on up to MATCH 10\n\n"
        "And so on for each task. Use exactly this format with the exact keywords 'TASK', 'MATCH 1:', 'REASON 1:', etc."
    )
    system_prompt = payload.batch_system_prompt or default_prompt

    # Collect results
    per_client_results: List[MatchPerClient] = []
    result_columns: List[str] = []
    for i in range(1, payload.num_matches + 1):
        result_columns.extend([f"LLM Match {i}", f"LLM Reasoning {i}"])

    for batch_items in batches:
        if payload.use_llm:
            ranked = _rank_with_llm(batch_items, system_prompt, payload.num_matches)
        else:
            # Passthrough top-K names without reasons if LLM disabled
            ranked = []
            for item in batch_items:
                pairs = [[name, "Nearest neighbor candidate"] for name in item["candidate_names"][: payload.num_matches]]
                ranked.append(pairs)

        for local_idx, pairs in enumerate(ranked):
            global_idx = batch_items[local_idx]["client_id"]
            row_out = {}
            for match_idx in range(payload.num_matches):
                if match_idx < len(pairs) and len(pairs[match_idx]) == 2:
                    row_out[f"LLM Match {match_idx + 1}"] = pairs[match_idx][0]
                    row_out[f"LLM Reasoning {match_idx + 1}"] = pairs[match_idx][1]
                else:
                    row_out[f"LLM Match {match_idx + 1}"] = None
                    row_out[f"LLM Reasoning {match_idx + 1}"] = None
            # Extend list until we can assign by position later
            while len(per_client_results) <= global_idx:
                per_client_results.append(MatchPerClient(results={}))
            per_client_results[global_idx] = MatchPerClient(results=row_out)

    return MatchResponse(
        success=True,
        message="Matching completed",
        model="gpt-4o" if payload.use_llm else None,
        result_columns=result_columns,
        matches=per_client_results,
    )


