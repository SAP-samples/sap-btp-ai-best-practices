import logging
from typing import List
from fastapi import APIRouter, HTTPException

from ..models.grounding import (
    GroundingRequest,
    GroundingResponse,
    PipelinesResponse,
    CollectionsResponse,
    DocumentsResponse,
    MappedPipelineCollection,
    FileInfo,
    Pipeline,
    Collection,
    Document,
)
from ..utils import grounding_utils

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/pipelines", response_model=PipelinesResponse)
async def list_pipelines():
    """Lists all available document grounding pipelines."""
    try:
        token = grounding_utils.get_token()
        data = grounding_utils.list_pipelines(token)
        logger.info(f"Pipelines API response: {data}")

        pipelines = [Pipeline(**resource) for resource in data.get("resources", [])]
        logger.info(f"Found {len(pipelines)} pipelines")
        return PipelinesResponse(resources=pipelines, success=True)

    except HTTPException as e:
        # Re-raise HTTP exceptions from utils
        error_msg = f"HTTP error while fetching pipelines: {str(e.detail)}"
        logger.error(error_msg)
        return PipelinesResponse(resources=[], success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error while fetching pipelines: {str(e)}"
        logger.error(error_msg)
        return PipelinesResponse(resources=[], success=False, error=error_msg)


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """Lists all available vector collections."""
    try:
        token = grounding_utils.get_token()
        data = grounding_utils.list_collections(token)
        logger.info(f"Collections API response: {data}")

        collections = [Collection(**resource) for resource in data.get("resources", [])]
        logger.info(f"Found {len(collections)} collections")
        return CollectionsResponse(resources=collections, success=True)

    except HTTPException as e:
        # Re-raise HTTP exceptions from utils
        error_msg = f"HTTP error while fetching collections: {str(e.detail)}"
        logger.error(error_msg)
        return CollectionsResponse(resources=[], success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error while fetching collections: {str(e)}"
        logger.error(error_msg)
        return CollectionsResponse(resources=[], success=False, error=error_msg)


@router.get("/collections/{collection_id}/documents", response_model=DocumentsResponse)
async def list_documents(collection_id: str):
    """Lists documents within a specific vector collection."""
    try:
        token = grounding_utils.get_token()
        data = grounding_utils.list_documents(token, collection_id)

        documents = [Document(**resource) for resource in data.get("resources", [])]
        return DocumentsResponse(resources=documents, success=True)

    except Exception as e:
        logger.error(f"Failed to list documents for collection {collection_id}: {e}")
        return DocumentsResponse(resources=[], success=False, error=str(e))


@router.get("/mapped-collections", response_model=List[MappedPipelineCollection])
async def get_mapped_collections():
    """Gets mapped pipelines and collections for display."""
    try:
        # Get pipelines and collections
        pipelines_resp = await list_pipelines()
        collections_resp = await list_collections()

        if not pipelines_resp.success or not collections_resp.success:
            raise HTTPException(
                status_code=500, detail="Failed to fetch pipelines or collections"
            )

        # Create pipeline mapping with token for detailed pipeline info
        token = grounding_utils.get_token()
        pipeline_info_map = {
            p.id: {"path": grounding_utils.get_pipeline_path(p.dict(), token), "raw": p}
            for p in pipelines_resp.resources
        }

        mapped_items = []
        for coll in collections_resp.resources:
            pipeline_id_from_meta = (
                grounding_utils.get_pipeline_id_from_collection_metadata(coll.dict())
            )

            pipeline_path_display = "N/A (Direct Collection or Unlinked)"
            if pipeline_id_from_meta and pipeline_id_from_meta in pipeline_info_map:
                pipeline_path_display = pipeline_info_map[pipeline_id_from_meta]["path"]

            mapped_items.append(
                MappedPipelineCollection(
                    pipeline_path=pipeline_path_display,
                    collection_title=coll.title,
                    collection_id=coll.id,
                )
            )

        return mapped_items

    except Exception as e:
        logger.error(f"Failed to get mapped collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_id}/files", response_model=List[FileInfo])
async def get_collection_files(collection_id: str):
    """Gets file information for a specific collection."""
    try:
        documents_resp = await list_documents(collection_id)

        if not documents_resp.success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch documents: {documents_resp.error}",
            )

        files_info = []
        for doc in documents_resp.resources:
            title, timestamp = grounding_utils.extract_document_details(doc.dict())
            files_info.append(
                FileInfo(
                    file_name=title, indexed_timestamp=timestamp, document_id=doc.id
                )
            )

        return files_info

    except Exception as e:
        logger.error(f"Failed to get files for collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completion", response_model=GroundingResponse)
async def grounding_completion(request: GroundingRequest):
    """Calls the grounding completion service."""
    try:
        token = grounding_utils.get_token()

        result = grounding_utils.call_grounding_completion(
            token=token,
            grounding_request=request.grounding_request,
            collection_id=request.collection_id,
            custom_prompt=request.custom_prompt,
            max_chunk_count=request.max_chunk_count,
        )

        return GroundingResponse(
            grounding_result=str(result["grounding_result"]),
            llm_response=result["llm_response"],
            success=True,
        )

    except Exception as e:
        logger.error(f"Failed to execute grounding completion: {e}")
        return GroundingResponse(
            grounding_result="", llm_response="", success=False, error=str(e)
        )
