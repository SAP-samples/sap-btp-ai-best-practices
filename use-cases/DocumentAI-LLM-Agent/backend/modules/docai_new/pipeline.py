"""
pipeline.py
-----------
DOC AI NEW — Main Pipeline.

Orchestrates the full DOC AI NEW workflow:
1. Detect PDF type (scanned vs searchable)
2. Extract with Free Prompt (always)
3. Discover template name (customer_name)
4. Look up existing template
5a. Template EXISTS → return extraction + template
5b. Template NOT FOUND → auto-create + auto-train with first PDF

When a new template is created, the first PDF automatically becomes
the first training document:
  Create Template → Upload PDF → Generate Annotations → Train Template

Always uses SAP_invoice_schema (schemaId: cf8cc8a9-1eee-42d9-9a3e-507a61baac23).
Template name = customer_name from LLM.
"""

import logging
from pathlib import Path
from typing import Any

from modules.docai_new.annotation_generation_service import AnnotationGenerationService
from modules.docai_new.free_prompt_extraction_service import FreePromptExtractionService
from modules.docai_new.pdf_detection_service import PdfDetectionService
from modules.docai_new.template_creation_service import (
    TemplateCreationError,
    TemplateCreationService,
)
from modules.docai_new.template_discovery_service import TemplateDiscoveryService, TemplatesNotAvailableError
from modules.docai_new.template_training_service import TemplateTrainingService
from modules.routing.template_processor import TemplateInvoiceProcessor, TemplateProcessingError

logger = logging.getLogger(__name__)


class DocAiNewPipeline:
    """
    DOC AI NEW pipeline.

    Workflow for NEW template:
        1. Detect PDF type
        2. Free Prompt extraction
        3. Discover template name (customer_name)
        4. Look up template in SAP → not found
        5. Auto-create template (schemaId required)
        6. Upload PDF as first training document
        7. Generate & submit annotations
        8. Trigger training
        → Returns: template_created=True, training_started=True

    Workflow for EXISTING template:
        1. Detect PDF type
        2. Free Prompt extraction
        3. Discover template name
        4. Look up template → found
        5. Return existing template + extraction result
        → Returns: template_found=True
    """

    def __init__(self) -> None:
        self._detector = PdfDetectionService()
        self._extractor = FreePromptExtractionService()
        self._discoverer = TemplateDiscoveryService()
        self._creator = TemplateCreationService()
        self._trainer = TemplateTrainingService()
        self._annotator = AnnotationGenerationService()
        self._template_processor = TemplateInvoiceProcessor()

    def process(
        self,
        pdf_path: Path,
        client_id: str = "default",
        auto_create_template: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full DOC AI NEW pipeline for a single PDF.

        Args:
            pdf_path: Path to the PDF file.
            client_id: SAP Document AI client ID.
            auto_create_template: Whether to auto-create + auto-train template.

        Returns:
            Pipeline result dict including training status when applicable.
        """
        logger.info("=" * 60)
        logger.info("DOC AI NEW PIPELINE — Starting: %s", pdf_path.name)
        logger.info("=" * 60)

        result: dict[str, Any] = {
            "filename": pdf_path.name,
            "pdf_type": None,
            "extraction": None,
            "customer_name": None,
            "template_found": False,
            "template_created": False,
            "template_activated": False,
            "template_id": None,
            "template_name": None,
            "sap_result": None,
            "document_id": None,
            "document_associated": False,
            "metadata_configured": False,
            "metadata_fields_count": 0,
            "annotations_count": 0,
            "route": None,
            "errors": [],
        }

        # ── Step 1: Detect PDF type ──────────────────────────────────────
        logger.info("[Step 1] Detecting PDF type for '%s'...", pdf_path.name)
        try:
            pdf_info = self._detector.get_pdf_info(pdf_path)
            result["pdf_type"] = pdf_info.get("pdf_type", "unknown")
        except Exception as exc:
            logger.warning("[Step 1] PDF detection failed: %s. Assuming searchable.", exc)
            result["pdf_type"] = "searchable"
        logger.info("[Step 1] PDF type: %s", result["pdf_type"])

        # ── Step 2: Free Prompt Extraction (always) ──────────────────────
        logger.info("[Step 2] Running Free Prompt Extraction...")
        try:
            extraction = self._extractor.extract(pdf_path)
            result["extraction"] = extraction
        except Exception as exc:
            logger.error("[Step 2] Extraction failed: %s", exc)
            result["errors"].append(f"Extraction failed: {exc}")
            result["route"] = "error"
            return result

        # ── Step 3: Discover template name (customer_name) ───────────────
        logger.info("[Step 3] Discovering template name from extraction...")
        raw_customer_name = self._extractor.extract_customer_name(extraction)
        # Apply SAP hardening: Unicode transliteration, max 80 chars, strip special chars
        customer_name = self._discoverer.normalize_template_name(raw_customer_name)
        result["customer_name"] = customer_name
        logger.info("[Step 3] Customer name (raw): '%s' → (normalized): '%s'", raw_customer_name, customer_name)

        # ── Step 4: Look up existing template ────────────────────────────
        logger.info("[Step 4] Looking up template for '%s'...", customer_name)
        existing_template = None
        try:
            existing_template = self._discoverer.find_template_by_customer(
                customer_name=customer_name,
                client_id=client_id,
            )
        except TemplatesNotAvailableError as exc:
            logger.warning("[Step 4] %s — skipping all template steps.", exc)
            result["route"] = "free_prompt_only"
            result["errors"].append(str(exc))
            return result
        except Exception as exc:
            logger.warning("[Step 4] Template lookup failed: %s", exc)

        if existing_template:
            # ── Template EXISTS: reprocess with SAP DocAI + template ─────
            result["template_found"] = True
            result["template_id"] = existing_template.get("id")
            result["template_name"] = existing_template.get("name")
            result["route"] = "existing_template"
            logger.info(
                "[Step 4] Template found: ID=%s | Name=%s — reprocessing with SAP DocAI",
                result["template_id"],
                existing_template.get("name"),
            )
            try:
                sap_result, _ = self._template_processor.process(
                    pdf_path=pdf_path,
                    template_id=result["template_id"],
                    client_id=client_id,
                )
                result["sap_result"] = sap_result
                logger.info("[Step 4] SAP DocAI + template reprocessing done.")
            except TemplateProcessingError as exc:
                logger.warning("[Step 4] Template reprocessing failed: %s. Using LLM extraction.", exc)
                result["errors"].append(f"Template reprocessing warning: {exc}")
                result["sap_result"] = None
            return result

        # ── Step 5: Template NOT FOUND — auto-create + auto-train ────────
        if not auto_create_template or customer_name == "Unknown_Customer":
            result["route"] = "free_prompt_only"
            logger.info("[Step 5] Skipping template creation (disabled or unknown customer).")
            return result

        logger.info("[Step 5] Template not found. Auto-creating for '%s'...", customer_name)

        # 5a. Final validation before create_template()
        assert customer_name, "customer_name must not be empty before create_template()"
        assert len(customer_name) > 0, "customer_name length must be > 0"
        assert len(customer_name) <= 80, f"customer_name too long ({len(customer_name)} chars)"
        logger.info("SAP SAFE TEMPLATE NAME: '%s'", customer_name)

        # 5b. Create template
        try:
            new_template = self._creator.create_template(
                customer_name=customer_name,
                client_id=client_id,
            )
            result["template_created"] = True
            result["template_id"] = new_template.get("id")
            logger.info("[Step 5] Template created: ID=%s", result["template_id"])
        except TemplateCreationError as exc:
            logger.error("[Step 5] Template creation failed: %s", exc)
            result["errors"].append(f"Template creation failed: {exc}")
            result["route"] = "free_prompt_only"
            return result

        # 5c. Activate template (DRAFT → ACTIVE)
        logger.info("[Step 5c] Activating template '%s'...", result["template_id"])
        try:
            self._creator.activate_template(
                template_id=result["template_id"],
                client_id=client_id,
            )
            result["template_activated"] = True
            logger.info("[Step 5c] Template activated successfully.")
        except Exception as exc:
            logger.warning("[Step 5c] Template activation failed (non-fatal): %s", exc)
            result["errors"].append(f"Template activation warning: {exc}")

        template_id = result["template_id"]

        # Step 6: Generate annotations from extraction using real PDF coordinates
        logger.info("[Step 6] Generating annotations with real PDF coordinates...")
        annotations = self._annotator.generate_annotations(extraction, pdf_path=pdf_path)
        result["annotations_count"] = len(annotations)
        logger.info("[Step 6] Generated %d annotations.", len(annotations))

        # Step 7: Associate document to template
        # Workflow: Upload PDF → get document_id → associate
        logger.info("[Step 7] Associating document to template '%s'...", template_id)
        try:
            attach_result = self._creator.add_document_to_template(
                template_id=template_id,
                pdf_path=pdf_path,
                client_id=client_id,
            )
            result["document_id"] = attach_result.get("document_id")
            result["document_associated"] = True
            logger.info("[Step 7] Document associated. document_id=%s", result["document_id"])
        except Exception as exc:
            logger.error("[Step 7] Document association failed: %s", exc)
            result["errors"].append(f"Document association failed: {exc}")
            result["route"] = "template_created_association_failed"

        # Step 8: Configure metadata
        # POST /templates/{id}/metadata — no training endpoint exists in SAP DOX API
        logger.info("[Step 8] Configuring metadata for template '%s'...", template_id)
        try:
            extraction_fields = extraction if isinstance(extraction, dict) else {}
            metadata_result = self._creator.configure_metadata(
                template_id=template_id,
                extracted_fields=extraction_fields,
                client_id=client_id,
            )
            result["metadata_configured"] = True
            result["metadata_fields_count"] = metadata_result.get("fields_count", 0)
            logger.info("[Step 8] Metadata configured. Fields: %d", result["metadata_fields_count"])
        except Exception as exc:
            logger.error("[Step 8] Metadata configuration failed: %s", exc)
            result["errors"].append(f"Metadata configuration failed: {exc}")

        result["route"] = "template_created_and_configured"

        logger.info("=" * 60)
        logger.info("DOC AI NEW PIPELINE completed. Route: %s", result["route"])
        logger.info("=" * 60)
        return result

    def process_batch(
        self,
        pdf_paths: list[Path],
        client_id: str = "default",
        auto_create_template: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Process multiple PDFs through the DOC AI NEW pipeline.

        Args:
            pdf_paths: List of PDF paths.
            client_id: SAP Document AI client ID.
            auto_create_template: Whether to auto-create + auto-train templates.

        Returns:
            List of pipeline result dicts.
        """
        results = []
        for pdf_path in pdf_paths:
            logger.info("[Batch] Processing: %s", pdf_path.name)
            try:
                result = self.process(
                    pdf_path=pdf_path,
                    client_id=client_id,
                    auto_create_template=auto_create_template,
                )
                results.append(result)
            except Exception as exc:
                logger.error("[Batch] Pipeline failed for '%s': %s", pdf_path.name, exc)
                results.append({
                    "filename": pdf_path.name,
                    "errors": [str(exc)],
                    "route": "error",
                })
        return results