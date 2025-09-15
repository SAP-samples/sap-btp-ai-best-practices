// Import all icons because of the dynamic import of the icons for use cases
import "@ui5/webcomponents-icons/dist/AllIcons.js";
import "@ui5/webcomponents-icons-tnt/dist/AllIcons.js";
import "@ui5/webcomponents-icons-business-suite/dist/AllIcons.js";

// import "@ui5/webcomponents-icons/dist/decrease-line-height.js";
// import "@ui5/webcomponents-icons/dist/lateness.js";
// import "@ui5/webcomponents-icons/dist/connected.js";
// import "@ui5/webcomponents-icons/dist/documents.js";
// import "@ui5/webcomponents-icons/dist/hide.js";
// import "@ui5/webcomponents-icons/dist/filter.js";
// import "@ui5/webcomponents-icons/dist/compare.js";
// import "@ui5/webcomponents-icons/dist/discussion.js";
// import "@ui5/webcomponents-icons/dist/translate.js";
// import "@ui5/webcomponents-icons/dist/theater.js";
// import "@ui5/webcomponents-icons/dist/detail-view.js";
// import "@ui5/webcomponents-icons/dist/doc-attachment.js";
// import "@ui5/webcomponents-icons/dist/ppt-attachment.js";
// import "@ui5/webcomponents-icons/dist/background.js";
// import "@ui5/webcomponents-icons/dist/document-text.js";
// import "@ui5/webcomponents-icons/dist/course-program.js";
// import "@ui5/webcomponents-icons/dist/write-new-document.js";
// import "@ui5/webcomponents-icons/dist/attachment-text-file.js";
// import "@ui5/webcomponents-icons/dist/image-viewer.js";
// import "@ui5/webcomponents-icons/dist/comment.js";
// import "@ui5/webcomponents-icons/dist/customer-view.js";
// import "@ui5/webcomponents-icons/dist/sys-enter.js";
// import "@ui5/webcomponents-icons/dist/overview-chart.js";
// import "@ui5/webcomponents-icons/dist/workflow-tasks.js";
// import "@ui5/webcomponents-icons/dist/quality-issue.js";
// import "@ui5/webcomponents-icons/dist/business-objects-explorer.js";
// import "@ui5/webcomponents-icons/dist/collapse-group.js";
// import "@ui5/webcomponents-icons/dist/provision.js";
// import "@ui5/webcomponents-icons/dist/syntax.js";
// import "@ui5/webcomponents-icons/dist/vds-file.js";
// import "@ui5/webcomponents-icons/dist/paste.js";
// import "@ui5/webcomponents-icons/dist/lab.js";
// import "@ui5/webcomponents-icons/dist/machine.js";
// import "@ui5/webcomponents-icons/dist/add-product.js";
// import "@ui5/webcomponents-icons/dist/pipeline-analysis.js";
// import "@ui5/webcomponents-icons/dist/education.js";
// import "@ui5/webcomponents-icons/dist/puzzle.js";
// import "@ui5/webcomponents-icons/dist/scatter-chart.js";
// import "@ui5/webcomponents-icons/dist/area-chart.js";
// import "@ui5/webcomponents-icons/dist/attachment-video.js";
// import "@ui5/webcomponents-icons/dist/timesheet";

const capabilityIcons: Record<string, string> = {
  // === Functional View ===

  // Information Analysis & Processing
  "/docs/functional-view/information-analysis-and-processing/summarization-of-text": "decrease-line-height",
  "/docs/functional-view/information-analysis-and-processing/translation-of-text": "translate",
  "/docs/functional-view/information-analysis-and-processing/sentiment-analysis-of-text": "theater",
  "/docs/functional-view/information-analysis-and-processing/qa-on-enterprise-knowledge-base": "detail-view",
  "/docs/functional-view/information-analysis-and-processing/information-classification-in-categories": "doc-attachment",
  "/docs/functional-view/information-analysis-and-processing/information-extraction-from-documents": "ppt-attachment",
  "/docs/functional-view/information-analysis-and-processing/image-analysis": "background",

  // Content Creation
  "/docs/functional-view/content-creation/generate-text-generic-knowledge": "document-text",
  "/docs/functional-view/content-creation/generate-text-specific-knowledge": "course-program",
  "/docs/functional-view/content-creation/review-and-refine-text": "write-new-document",
  "/docs/functional-view/content-creation/generate-image-from-text": "image-viewer",
  "/docs/functional-view/content-creation/describe-images": "attachment-text-file",

  // Conversational Interaction
  "/docs/functional-view/conversational-interaction/conversational-interaction-with-applications": "comment",
  "/docs/functional-view/conversational-interaction/conversational-interaction-with-analytics": "customer-view",

  // Decision Support
  "/docs/functional-view/decision-support/recommendations-historical-data": "sys-enter",
  "/docs/functional-view/decision-support/data-clustering": "overview-chart",
  "/docs/functional-view/decision-support/forecasting-historical-data": "workflow-tasks",
  "/docs/functional-view/decision-support/anomaly-detection": "quality-issue",
  "/docs/functional-view/decision-support/scenario-simulations": "business-objects-explorer",
  "/docs/functional-view/decision-support/alternative-ranking": "collapse-group",

  // Coding Assistance
  "/docs/functional-view/coding-assistance/synthetic-data-generation": "provision",
  "/docs/functional-view/coding-assistance/code-generation": "syntax",
  "/docs/functional-view/coding-assistance/code-review": "vds-file",
  "/docs/functional-view/coding-assistance/code-documentation": "paste",
  "/docs/functional-view/coding-assistance/test-generation": "lab",

  // Agentic Workflows
  "/docs/functional-view/agentic-workflows/agentic-workflows": "machine",

  // Add other functional view mappings... Ensure NO trailing slashes in keys

  // === Technical View ===

  // Plain
  "/docs/technical-view/generative-ai/plain": "discussion",
  "/docs/technical-view/generative-ai/plain/access-to-generative-ai-models": "connected",
  "/docs/technical-view/generative-ai/plain/prompt-templating": "documents",
  "/docs/technical-view/generative-ai/plain/data-masking": "hide",
  "/docs/technical-view/generative-ai/plain/content-filtering": "filter",

  // RAG
  "/docs/technical-view/generative-ai/rag": "pipeline-analysis",
  "/docs/technical-view/generative-ai/rag/vector-rag-vs-graph-rag": "compare",
  "/docs/technical-view/generative-ai/rag/vector-overall": "lateness",
  "/docs/technical-view/generative-ai/rag/vector-query": "lateness",
  "/docs/technical-view/generative-ai/rag/vector-rag-embedding": "add-product",
  "/docs/technical-view/generative-ai/rag/vector-rag-query-pipeline": "pipeline-analysis",
  "/docs/technical-view/generative-ai/rag/kg-rag-creation": "education",
  "/docs/technical-view/generative-ai/rag/kg-rag-query-pipeline": "pipeline-analysis",

  // Agentic
  "/docs/technical-view/generative-ai/agentic/single-agent": "lateness",
  "/docs/technical-view/generative-ai/agentic/multi-agent": "lateness",

  // Narrow AI
  "/docs/technical-view/narrow-ai/regression": "scatter-chart",
  "/docs/technical-view/narrow-ai/time-series-forecasting": "area-chart",
  "/docs/technical-view/narrow-ai/classification": "lateness",
  "/docs/technical-view/narrow-ai/recommendation": "lateness",
  "/docs/technical-view/narrow-ai/clustering": "lateness",
  "/docs/technical-view/narrow-ai/anomaly-detection": "quality-issue",
  "/docs/technical-view/narrow-ai/predictive-ai-bdc-btp": "lateness",
  "/docs/technical-view/narrow-ai/deployment-custom-predictive-models": "puzzle",
  "/docs/technical-view/narrow-ai/data-preparation-and-model-evaluation": "timesheet",

  // AI Services
  "/docs/technical-view/ai-services/document-information-extraction": "ppt-attachment",
  "/docs/technical-view/ai-services/sap-document-ai": "attachment-video"
};

/**
 * Gets the UI5 icon name associated with a given capability or technical pattern href.
 * Falls back to 'lateness' if no specific icon is mapped.
 * Handles trailing slashes in the input href automatically and attempts lookup
 * both with and without a trailing slash against the map keys.
 * @param href The href string of the link.
 * @returns The name of the UI5 icon.
 */
export const getIconForCapability = (href: string | undefined): string => {
  // Return default if href is undefined or null
  if (!href) {
    return "lateness";
  }

  // Normalize input href by removing trailing slash if present (and not the root '/')
  const normalizedHref = href.length > 1 && href.endsWith("/") ? href.slice(0, -1) : href;

  // 1. Try lookup with the key exactly as normalized (no trailing slash)
  let icon = capabilityIcons[normalizedHref];

  // 2. If not found, try lookup with a trailing slash added to the normalized key
  if (!icon) {
    icon = capabilityIcons[normalizedHref + "/"];
  }

  // Return the found icon or the default 'lateness' icon
  return icon || "lateness";
};
