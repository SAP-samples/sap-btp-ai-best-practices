import { environment } from "@site/src/config/environment";

export const TRACKING_CONFIG = {
  toolName: "sap-btp-ai-best-practices-website",
  dataCenter: "eu1",
  apiKey: environment.tracking.apiKey,
  consentMessage: "We collect anonymous data to improve your experience on this site. Continue?"
} as const;

export const TRACKING_EVENTS = {
  PAGE_VIEW: {
    HOME: "pv-home",
    GLOSSARY: "pv-glossary",

    // Technical View
    TECHNICAL_VIEW: "pv-technical-view",
    TECHNICAL_VIEW_NARROW_AI: "pv-technical-view-narrow-ai",
    TECHNICAL_VIEW_GENERATIVE_AI: "pv-technical-view-generative-ai",
    TECHNICAL_VIEW_GENERATIVE_AI_PLAIN: "pv-technical-view-generative-ai-plain",
    BP_ACCESS_TO_GENERATIVE_AI_MODELS: "pv-bp-access-to-generative-ai-models",
    BP_PROMPT_TEMPLATING: "pv-bp-prompt-templating",
    BP_DATA_MASKING: "pv-bp-data-masking",
    BP_CONTENT_FILTERING: "pv-bp-content-filtering",
    TECHNICAL_VIEW_GENERATIVE_AI_RAG: "pv-technical-view-generative-ai-rag",
    BP_VECTOR_RAG_VS_GRAPH_RAG: "pv-bp-vector-rag-vs-graph-rag",
    BP_VECTOR_BASED_RAG_EMBEDDING: "pv-bp-vector-based-rag-embedding",
    BP_VECTOR_BASED_RAG_QUERY_PIPELINE: "pv-bp-vector-based-rag-query-pipeline",

    // Functional View
    FUNCTIONAL_VIEW: "pv-functional-view",

    FUNCTIONAL_VIEW_INFORMATION_ANALYSIS_AND_PROCESSING: "pv-functional-view-information-analysis-and-processing",
    AI_CAPABILITY_SUMMARIZATION_OF_TEXT: "pv-ai-capability-summarization-of-text",
    AI_CAPABILITY_TRANSLATION_OF_TEXT: "pv-ai-capability-translation-of-text",
    AI_CAPABILITY_SENTIMENT_ANALYSIS_OF_TEXT: "pv-ai-capability-sentiment-analysis-of-text",
    AI_CAPABILITY_QA_ON_ENTERPRISE_KNOWLEDGE_BASE: "pv-ai-capability-qa-on-enterprise-knowledge-base",
    AI_CAPABILITY_INFORMATION_CLASSIFICATION_IN_CATEGORIES: "pv-ai-capability-information-classification-in-categories",
    AI_CAPABILITY_INFORMATION_EXTRACTION_FROM_DOCUMENTS: "pv-ai-capability-information-extraction-from-documents",
    AI_CAPABILITY_IMAGE_ANALYSIS: "pv-ai-capability-image-analysis",

    FUNCTIONAL_VIEW_CONTENT_CREATION: "pv-functional-view-content-creation",
    AI_CAPABILITY_GENERATE_TEXT_BASED_ON_GENERIC_KNOWLEDGE: "pv-ai-capability-generate-text-based-on-generic-knowledge",
    AI_CAPABILITY_GENERATE_TEXT_BASED_ON_SPECIFIC_KNOWLEDGE: "pv-ai-capability-generate-text-based-on-specific-knowledge",
    AI_CAPABILITY_REVIEW_AND_REFINE_TEXT: "pv-ai-capability-review-and-refine-text",
    AI_CAPABILITY_GENERATE_IMAGE_FROM_TEXT: "pv-ai-capability-generate-image-from-text",
    AI_CAPABILITY_DESCRIBE_IMAGES: "pv-ai-capability-describe-images",

    FUNCTIONAL_VIEW_CONVERSATIONAL_INTERACTION: "pv-functional-view-conversational-interaction",
    AI_CAPABILITY_CONVERSATIONAL_INTERACTION_WITH_APPLICATIONS: "pv-ai-capability-conversational-interaction-with-applications",
    AI_CAPABILITY_CONVERSATIONAL_INTERACTION_WITH_ANALYTICS: "pv-ai-capability-conversational-interaction-with-analytics",

    FUNCTIONAL_VIEW_DECISION_SUPPORT: "pv-functional-view-decision-support",
    AI_CAPABILITY_RECOMMENDATIONS_HISTORICAL_DATA: "pv-ai-capability-recommendations-historical-data",
    AI_CAPABILITY_DATA_CLUSTERING: "pv-ai-capability-data-clustering",
    AI_CAPABILITY_FORECASTING_HISTORICAL_DATA: "pv-ai-capability-forecasting-historical-data",
    AI_CAPABILITY_ANOMALY_DETECTION: "pv-ai-capability-anomaly-detection",
    AI_CAPABILITY_SCENARIO_SIMULATIONS: "pv-ai-capability-scenario-simulations",
    AI_CAPABILITY_ALTERNATIVE_RANKING: "pv-ai-capability-alternative-ranking",

    FUNCTIONAL_VIEW_CODING_ASSISTANCE: "pv-functional-view-coding-assistance",
    AI_CAPABILITY_SYNTHETIC_DATA_GENERATION: "pv-ai-capability-synthetic-data-generation",
    AI_CAPABILITY_CODE_GENERATION: "pv-ai-capability-code-generation",
    AI_CAPABILITY_CODE_REVIEW: "pv-ai-capability-code-review",
    AI_CAPABILITY_CODE_DOCUMENTATION: "pv-ai-capability-code-documentation",
    AI_CAPABILITY_TEST_GENERATION: "pv-ai-capability-test-generation",

    FUNCTIONAL_VIEW_AGENTIC_WORKFLOWS: "pv-functional-view-agentic-workflows",
    AI_CAPABILITY_AGENTIC_WORKFLOWS: "pv-ai-capability-agentic-workflows",

    // Technology Pages
    TECHNOLOGY_SAP_BTP: "pv-technology-sap-btp",
    TECHNOLOGY_SAP_AI_CORE: "pv-technology-sap-ai-core",
    TECHNOLOGY_SAP_HANA_CLOUD: "pv-technology-sap-hana-cloud",
    TECHNOLOGY_SAP_AI_LAUNCHPAD: "pv-technology-sap-ai-launchpad"
  },

  // Button click events
  BUTTON_CLICK: {
    DOWNLOAD_ACCESS_TO_GEN_AI_MODELS: "btn-download-access-to-gen-ai-models",
    DOWNLOAD_PROMPT_TEMPLATING: "btn-download-prompt-templating",
    DOWNLOAD_DATA_MASKING: "btn-download-data-masking",
    DOWNLOAD_CONTENT_FILTERING: "btn-download-content-filtering",
    DOWNLOAD_VECTOR_RAG_EMBEDDING: "btn-download-vector-rag-embedding",
    DOWNLOAD_VECTOR_RAG_QUERY_PIPELINE: "btn-download-vector-rag-query-pipeline",
    DOWNLOAD_KG_RAG_CREATION: "btn-download-kg-rag-creation",
    DOWNLOAD_KG_RAG_QUERY_PIPELINE: "btn-download-kg-rag-query-pipeline",
    DOWNLOAD_DOCUMENT_AI: "btn-download-ai",
    DOWNLOAD_DEPLOYMENT_CUSTOM_PREDICTIVE_MODELS: "btn-download-deployment-custom-predictive-models",
    DOWNLOAD_REGRESSION: "btn-download-regression",
    DOWNLOAD_TIME_SERIES_FORECASTING: "btn-download-time-series-forecasting",
    DOWNLOAD_ANOMALY_DETECTION: "btn-download-anomaly-detection",
    DOWNLOAD_CLASSIFICATION: "btn-download-classification",
    DOWNLOAD_CLUSTERING: "btn-download-clustering"
  }
} as const;
