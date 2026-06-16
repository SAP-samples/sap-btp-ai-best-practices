"""
Pydantic models for the chatbot API.

Defines request and response schemas for the forecasting agent chatbot endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request to send a message to the chatbot."""

    message: str = Field(..., description="User message to send to the agent")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity. Creates new session if not provided.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What if brand awareness increases 10% in NYC?",
                    "session_id": "abc123-def456",
                }
            ]
        }
    }


class ToolCall(BaseModel):
    """Information about a tool call made by the agent."""

    name: str = Field(..., description="Name of the tool that was called")
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )


class Attachment(BaseModel):
    """A file attachment in the chat response."""

    filename: str = Field(..., description="Name of the file")
    file_type: str = Field(
        ..., description="Type of file: 'image', 'csv', or 'pdf'"
    )
    url: str = Field(..., description="URL endpoint to fetch the file")


class ChatResponse(BaseModel):
    """Response from the chatbot."""

    session_id: str = Field(..., description="Session ID for conversation continuity")
    message: str = Field(..., description="Agent's response message")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools called by the agent during processing"
    )
    attachments: List[Attachment] = Field(
        default_factory=list,
        description="File attachments (images, CSVs, PDFs) generated during processing",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123-def456",
                    "message": "Based on my analysis, a 10% increase in brand awareness in NYC would result in approximately $2.3M additional sales...",
                    "tool_calls": [
                        {"name": "modify_business_lever", "args": {"feature": "awareness", "modification": "increase by 10%"}},
                        {"name": "run_forecast_model", "args": {}},
                    ],
                    "attachments": [
                        {"filename": "scenario_comparison_1_20251216.png", "file_type": "image", "url": "/api/chatbot/files/scenario_comparison_1_20251216.png"},
                    ],
                }
            ]
        }
    }


class ScenarioSummary(BaseModel):
    """Summary of a scenario in the session."""

    name: str = Field(..., description="Scenario name")
    parent: Optional[str] = Field(None, description="Parent scenario name")
    num_rows: int = Field(0, description="Number of data rows in scenario")
    num_modifications: int = Field(0, description="Number of modifications applied")
    stores: List[int] = Field(default_factory=list, description="Store IDs in scenario")
    horizons: List[int] = Field(default_factory=list, description="Horizon weeks in scenario")


class SessionInfo(BaseModel):
    """Information about a session."""

    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="Session creation timestamp (ISO format)")
    last_activity: str = Field(..., description="Last activity timestamp (ISO format)")
    origin_date: Optional[str] = Field(None, description="Forecast origin date (t0)")
    horizon_weeks: int = Field(13, description="Forecast horizon in weeks")
    channel: str = Field("B&M", description="Active channel (B&M or WEB)")
    active_scenario: str = Field("baseline", description="Name of active scenario")
    num_scenarios: int = Field(0, description="Number of scenarios")
    scenario_names: List[str] = Field(default_factory=list, description="List of scenario names")
    num_predictions: int = Field(0, description="Number of cached predictions")
    predictions_for: List[str] = Field(default_factory=list, description="Scenarios with predictions")
    models_loaded: bool = Field(False, description="Whether ML models are loaded")


class ConversationMessage(BaseModel):
    """A message in the conversation history."""

    role: str = Field(..., description="Message role (human, ai, tool)")
    content: str = Field(..., description="Message content")


class ConversationHistory(BaseModel):
    """Conversation history for a session."""

    session_id: str = Field(..., description="Session identifier")
    messages: List[ConversationMessage] = Field(
        default_factory=list, description="Conversation messages"
    )
    total_messages: int = Field(0, description="Total number of messages in session")


class DeleteResponse(BaseModel):
    """Response for delete operations."""

    status: str = Field(..., description="Operation status")
    session_id: str = Field(..., description="Deleted session ID")


class ResetResponse(BaseModel):
    """Response for reset operations."""

    status: str = Field(..., description="Operation status")
    session_id: str = Field(..., description="Reset session ID")


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ToolCall",
    "Attachment",
    "ScenarioSummary",
    "SessionInfo",
    "ConversationMessage",
    "ConversationHistory",
    "DeleteResponse",
    "ResetResponse",
]
