"""Assessment framework models for normalized questionnaire data."""

from pydantic import BaseModel, ConfigDict, Field


class AnswerItem(BaseModel):
    """Normalized answer option belonging to a single assessment question.

    Inputs:
        Field values describing one imported answer option.

    Outputs:
        A validated answer item with level and item index bounds enforced.
        Export-only Ecomarket columns such as selected defaults or optional flags
        are rejected as extra fields.

    Attributes:
        answer_item_id: Stable identifier for this specific answer item.
        question_id: Identifier of the assessment question this item belongs to.
        level: Maturity level represented by the item, constrained to values 1-5.
        item_index: One-based ordering value within the question and level.
        text: Human-readable answer text imported from the framework catalog.
    """

    model_config = ConfigDict(extra="forbid")

    answer_item_id: str
    question_id: str
    level: int = Field(ge=1, le=5)
    item_index: int = Field(ge=1)
    text: str


class AssessmentQuestion(BaseModel):
    """Assessment question with its grouped answer catalog items.

    Inputs:
        Field values describing one framework question and its answer options.

    Outputs:
        A validated assessment question carrying answer items for caller-side
        grouping by level.

    Attributes:
        question_id: Stable identifier for the assessment question.
        dimension: High-level framework dimension that owns the question.
        section: Framework section or subsection containing the question.
        topic_title: Concise question-specific title used by benchmark reports.
        question: Human-readable question text presented to reviewers.
        explanation: Optional guidance that explains how to interpret the question.
        answer_items: Answer options associated with the question, grouped by
            their ``level`` values when consumed by callers.
    """

    question_id: str
    dimension: str
    section: str
    topic_title: str | None = None
    question: str
    explanation: str | None = None
    answer_items: list[AnswerItem] = Field(default_factory=list)


class AssessmentDimension(BaseModel):
    """Summary counters for a single assessment framework dimension.

    Inputs:
        Field values describing question and answer progress for a dimension.

    Outputs:
        A validated dimension summary with ``answered_count`` defaulting to 0.

    Attributes:
        dimension: Name of the framework dimension being summarized.
        display_name: Localized dimension label for display. The ``dimension``
            field remains the stable canonical key used by API routes and jobs.
        question_count: Total number of questions in the dimension.
        answered_count: Number of questions that currently have selected answers.
    """

    dimension: str
    display_name: str
    question_count: int
    answered_count: int = 0


class FrameworkImportSummary(BaseModel):
    """Aggregate counts produced after importing a framework catalog.

    Inputs:
        Count values produced by a framework import operation.

    Outputs:
        A validated summary of imported dimensions, questions, answer items, and
        explanations.

    Attributes:
        dimensions: Number of distinct framework dimensions imported.
        questions: Number of assessment questions imported.
        answer_items: Number of normalized answer items imported.
        explanations: Number of question explanations imported.
    """

    dimensions: int
    questions: int
    answer_items: int
    explanations: int
