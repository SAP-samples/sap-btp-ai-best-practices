"""API response models for data import endpoints."""

from pydantic import BaseModel


class AssessmentImportResponse(BaseModel):
    """Normalized summary returned by assessment framework imports.

    Inputs:
        dimensions: Number of unique framework dimensions in the uploaded seed.
        questions: Number of assessment questions in the uploaded seed.
        answer_items: Number of answer items across all uploaded questions.
        explanations: Number of question explanations in the uploaded seed.
        italian_questions: Optional count of Italian translated questions.
        italian_answer_items: Optional count of Italian translated answer items.
        write_completed: Whether the import was committed to HANA.

    Outputs:
        FrameworkImportResponse: A machine-readable import summary for API callers.
    """

    dimensions: int
    questions: int
    answer_items: int
    explanations: int
    italian_questions: int | None = None
    italian_answer_items: int | None = None
    write_completed: bool


class JouleKnowledgeImportResponse(BaseModel):
    """Normalized summary returned by Joule knowledge imports.

    Inputs:
        glossary_terms: Number of glossary rows in the uploaded seed.
        question_explanations: Number of question explanations in the seed.
        dimensions: Number of dimension explanations in the seed.
        embedding_model: Embedding model used for optional semantic indexing.
        batch_size: Embedding batch size used in the importer.
        write_completed: Whether the import was committed to HANA.

    Outputs:
        JouleKnowledgeImportResponse: A machine-readable import summary for API
        callers.
    """

    glossary_terms: int
    question_explanations: int
    dimensions: int
    embedding_model: str
    batch_size: int
    write_completed: bool
