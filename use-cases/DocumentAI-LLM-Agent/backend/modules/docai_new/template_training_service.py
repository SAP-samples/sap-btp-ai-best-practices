"""
template_training_service.py
-----------------------------
DOC AI NEW — Template Training Service.

Correct SAP DOX API workflow (no /train endpoint exists):
1. Load existing templates
2. Select template
3. Upload PDFs
4. Run Free Prompt extraction
5. Generate annotations
6. Attach documents to template (upload → poll → associate)
7. Configure metadata (POST /templates/{id}/metadata)

NOTE: SAP DOX API does NOT expose POST /templates/{id}/train.
      The workflow ends with metadata configuration.
"""

import logging
from pathlib import Path
from typing import Any

from modules.docai_new.annotation_generation_service import AnnotationGenerationService
from modules.docai_new.free_prompt_extraction_service import FreePromptExtractionService
from modules.docai_new.template_creation_service import TemplateCreationService, TemplateCreationError
from modules.docai_new.template_discovery_service import TemplateDiscoveryService

logger = logging.getLogger(__name__)


class TemplateTrainingService:
    """
    Orchestrates the complete template training workflow.

    Supports training with 1 or N PDFs.
    Builds the training dataset automatically.
    """

    def __init__(self) -> None:
        self._extractor = FreePromptExtractionService()
        self._annotator = AnnotationGenerationService()
        self._creator = TemplateCreationService()
        self._discovery = TemplateDiscoveryService()

    def train_template(
        self,
        template_id: str,
        pdf_paths: list[Path],
        client_id: str = "default",
    ) -> dict[str, Any]:
        """
        Train a template with one or more PDFs.

        For each PDF:
        1. Run Free Prompt extraction
        2. Generate annotations
        3. Attach document to template
        4. Trigger training

        Args:
            template_id: SAP Document AI template ID.
            pdf_paths: List of PDF paths to use for training.
            client_id: SAP Document AI client ID.

        Returns:
            Training result dict with status and details.
        """
        if not pdf_paths:
            raise ValueError("At least one PDF is required for training.")

        logger.info(
            "Starting template training. Template: %s | PDFs: %d",
            template_id,
            len(pdf_paths),
        )

        documents_processed = 0
        fields_annotated = 0
        errors: list[str] = []
        extraction_results: list[dict[str, Any]] = []

        for pdf_path in pdf_paths:
            try:
                logger.info("Processing PDF for training: %s", pdf_path.name)

                # Step 1: Free Prompt extraction
                extraction = self._extractor.extract(pdf_path)
                extraction_results.append({
                    "filename": pdf_path.name,
                    "extraction": extraction,
                })

                # Step 2: Generate annotations with real PDF coordinates
                annotations = self._annotator.generate_annotations(
                    extraction, pdf_path=pdf_path
                )
                fields_annotated += len(annotations)

                # Step 3: Attach document to template
                try:
                    self._creator.add_document_to_template(
                        template_id=template_id,
                        pdf_path=pdf_path,
                        client_id=client_id,
                    )
                    documents_processed += 1
                    logger.info("Document attached to template: %s", pdf_path.name)
                except TemplateCreationError as exc:
                    logger.warning("Could not attach document '%s': %s", pdf_path.name, exc)
                    errors.append(f"{pdf_path.name}: {exc}")

            except Exception as exc:
                logger.error("Error processing '%s': %s", pdf_path.name, exc)
                errors.append(f"{pdf_path.name}: {exc}")

        # Step 4: Configure metadata
        # SAP DOX API does NOT expose POST /templates/{id}/train.
        # The correct final step is POST /templates/{id}/metadata.
        metadata_configured = False
        metadata_fields_count = 0

        if documents_processed > 0:
            try:
                # Use fields from the last extraction as reference
                last_extraction = extraction_results[-1]["extraction"] if extraction_results else {}
                metadata_result = self._creator.configure_metadata(
                    template_id=template_id,
                    extracted_fields=last_extraction if isinstance(last_extraction, dict) else {},
                    client_id=client_id,
                )
                metadata_configured = True
                metadata_fields_count = metadata_result.get("fields_count", 0)
                logger.info(
                    "Metadata configured for template '%s'. Fields: %d",
                    template_id,
                    metadata_fields_count,
                )
            except TemplateCreationError as exc:
                errors.append(f"Metadata configuration failed: {exc}")
                logger.error("Metadata configuration failed: %s", exc)
        else:
            logger.warning("No documents were attached. Metadata configuration skipped.")

        return {
            "template_id": template_id,
            "documents_processed": documents_processed,
            "fields_annotated": fields_annotated,
            "metadata_configured": metadata_configured,
            "metadata_fields_count": metadata_fields_count,
            "training_status": "metadata_configured" if metadata_configured else "skipped",
            "extraction_results": extraction_results,
            "errors": errors,
            "success": documents_processed > 0 and metadata_configured and len(errors) == 0,
        }

    def get_available_templates(self, client_id: str = "default") -> list[dict[str, Any]]:
        """Return all available templates for selection."""
        return self._discovery.list_all_templates(client_id=client_id)