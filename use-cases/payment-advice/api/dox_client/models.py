"""Product-used field models for SAP Document AI schema configuration."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FieldSetup(BaseModel):
    """
    Configure extraction for a schema field.

    Attributes:
        language: Supported extraction languages; SAP expects an array.
        type: Setup type, either ``auto`` or ``manual``.
        priority: Extraction priority.
        modelArtifactId: Referenced SAP model artifacts.
    """

    language: List[str] = Field(default_factory=list)
    type: Literal["auto", "manual"]
    priority: int = 1
    modelArtifactId: List[str] = Field(default_factory=list)


class FieldDefinition(BaseModel):
    """
    Define one SAP Document AI schema field.

    Attributes:
        name: Unique field name within the schema.
        description: Optional field description.
        label: Optional display label.
        categoryName: Optional SAP schema category.
        setupType: SAP setup type identifier.
        setupTypeVersion: SAP setup type version.
        setup: Extraction language, type, and priority configuration.
        formattingType: SAP field formatting type.
        formatting: Formatting options for the selected type.
        formattingTypeVersion: SAP formatting type version.
        defaultExtractor: Optional predefined extractor configuration.
    """

    name: str
    description: Optional[str] = None
    label: Optional[str] = None
    categoryName: Optional[str] = None
    setupType: str = "static"
    setupTypeVersion: str = "2.0.0"
    setup: FieldSetup = Field(default_factory=lambda: FieldSetup(type="auto", priority=1))
    # SAP can add formatting types such as ``country/region`` without a client release.
    formattingType: str = "string"
    formatting: Dict[str, Any] = Field(default_factory=dict)
    formattingTypeVersion: str = "1.0.0"
    defaultExtractor: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return the full field definition expected by the SAP API."""
        return self.model_dump(exclude_none=False)
