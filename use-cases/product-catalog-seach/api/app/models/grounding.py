from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class GroundingRequest(BaseModel):
    """Request model for grounding completion endpoint.

    Attributes:
        grounding_request: The user's query/request text
        collection_id: Vector collection ID to search in, or "*" for all collections
        custom_prompt: Optional custom prompt template for the LLM
        max_chunk_count: Maximum number of chunks to retrieve (1-300)
    """

    grounding_request: str
    collection_id: str = "*"
    custom_prompt: Optional[str] = None
    max_chunk_count: int = 50


class GroundingResponse(BaseModel):
    """Response model for grounding completion endpoint.

    Attributes:
        grounding_result: The grounding/context data retrieved
        llm_response: The LLM's response based on the grounding context
        success: Whether the request was successful
        error: Error message if request failed
    """

    grounding_result: str
    llm_response: str
    success: bool
    error: Optional[str] = None


class Pipeline(BaseModel):
    """Model for pipeline resource.

    Attributes:
        id: Pipeline ID
        configuration: Pipeline configuration data
        title: Pipeline title
    """

    id: str
    configuration: Dict[str, Any]
    title: Optional[str] = None


class Collection(BaseModel):
    """Model for vector collection resource.

    Attributes:
        id: Collection ID
        title: Collection title
        metadata: Collection metadata
    """

    id: str
    title: str
    metadata: List[Dict[str, Any]]


class Document(BaseModel):
    """Model for document resource.

    Attributes:
        id: Document ID
        metadata: Document metadata containing title, timestamp, etc.
    """

    id: str
    metadata: List[Dict[str, Any]]


class PipelinesResponse(BaseModel):
    """Response model for list pipelines endpoint.

    Attributes:
        resources: List of pipeline resources
        success: Whether the request was successful
        error: Error message if request failed
    """

    resources: List[Pipeline]
    success: bool
    error: Optional[str] = None


class CollectionsResponse(BaseModel):
    """Response model for list collections endpoint.

    Attributes:
        resources: List of collection resources
        success: Whether the request was successful
        error: Error message if request failed
    """

    resources: List[Collection]
    success: bool
    error: Optional[str] = None


class DocumentsResponse(BaseModel):
    """Response model for list documents endpoint.

    Attributes:
        resources: List of document resources
        success: Whether the request was successful
        error: Error message if request failed
    """

    resources: List[Document]
    success: bool
    error: Optional[str] = None


class MappedPipelineCollection(BaseModel):
    """Model for mapped pipeline and collection data for display.

    Attributes:
        pipeline_path: Path from pipeline configuration
        collection_title: Collection title
        collection_id: Collection ID
    """

    pipeline_path: str
    collection_title: str
    collection_id: str


class FileInfo(BaseModel):
    """Model for file information in a collection.

    Attributes:
        file_name: Document title/file name
        indexed_timestamp: When the document was indexed
        document_id: Document ID
    """

    file_name: str
    indexed_timestamp: str
    document_id: str
