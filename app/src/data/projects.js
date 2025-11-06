/**
 * Projects data for AI4U website
 * This file contains all project information used across different pages
 */

export const projects = [
  {
    id: "agentic-email-automation",
    title: "Agentic Email Automation",
    description:
      "AI Email Agent that prioritizes and answers emails end-to-end, integrating with SAP S/4HANA, Ariba, and more systems to traverse decision trees and compliance gates.",
    categories: [],
    industries: [],
    icon: "workflow-tasks",
    docsPath: "agentic-email-automation",
    sampleFiles: [],
    appUrl: "https://email-agent-cockpit.cfapps.eu10-004.hana.ondemand.com/email-agent",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/email-agent",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/ai-powered-email-cockpit",
    videoUrl: "",
    onePagerFile: "images/Slide5.jpeg",
    isExternal: true
  },
  {
    id: "ai-powered-email-cockpit",
    title: "AI-Powered Email Cockpit",
    description: "An easy-to-use tool with an email-style interface and built-in chatbot to quickly organize, track, and resolve payment-related requests.",
    categories: [],
    industries: [],
    icon: "email",
    docsPath: "ai-powered-email-cockpit",
    sampleFiles: [],
    appUrl: "https://ai-powered-email-cockpit.cfapps.eu10-004.hana.ondemand.com/ai-powered-email-cockpit",
    completionDate: "2025-08-14",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/ai-powered-email-cockpit",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/ai-powered-email-cockpit/tree/release-ai-in-a-day",
    onePagerFile: "images/Slide1.jpeg",
    isExternal: true
  },
  {
    id: "utilities-tariff-mapping-cockpit",
    title: "Utilities Tariff Mapping Cockpit",
    description: "AI‑assisted mapping of utility tariff PDFs/text to SAP IS‑U configuration with review and compliance checks.",
    categories: [],
    industries: [],
    icon: "energy-saving-lightbulb",
    docsPath: "utilities-tariff-mapping-cockpit",
    sampleFiles: ["data/Residential Service – Rate 10 (Small file).pdf", "data/Riverview Tariff Book 2025 (Large file).pdf"],
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_8efsik7w",
    appUrl: "https://utilities-tariff-mapping-cockpit.cfapps.eu10-004.hana.ondemand.com/validate/step1-upload",
    completionDate: "2025-08-11",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/prompt-templating",
        title: "Prompt Templating"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/utilities-tariff-mapping-cockpit",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/utilities-tariff-mapping-cockpit",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    isExternal: true
  },
  {
    id: "intelligent-procurement-assistant",
    title: "Intelligent Procurement Assistant",
    description: "AI assistant that finds materials, recommends approved vendors, and automatically creates purchase requests from catalog, text, or quotation PDFs.",
    categories: [],
    industries: [],
    icon: "cart",
    docsPath: "intelligent-procurement-assistant",
    onePagerFile: "images/Slide52.jpg",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_dv205qgu",
    sampleFiles: [
      "data/01_Contract_Office_Supplies_Enriched.pdf",
      "data/02_Contract_Cloud_Security_Enriched.pdf",
      "data/03_Contract_Catering_Enriched.pdf",
      "data/04_Contract_Software_Maintenance_Enriched.pdf"
    ],
    appUrl: "https://intelligent-procurement-assistant.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "2025-08-08",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/vector-rag-embedding",
        title: "Vector-based RAG (1/2) Embedding"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/vector-rag-query-pipeline",
        title: "Vector-based RAG (2/2) Query Pipeline"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/intelligent-procurement-assistant",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/intelligent-procurement-assistant",
    internalSourceCodeUrl2: "https://github.tools.sap/sap-btp-ai-services-coe/nielsen-streamlit",
    isExternal: true
  },
  {
    id: "product-catalog-search",
    title: "Product Catalog Search",
    description:
      "This AI-powered application searches PDF product catalogs and automatically analyzes contracts to determine if specific products can be purchased under existing agreements.",
    categories: [],
    industries: [],
    icon: "search",
    docsPath: "product-catalog-search",
    onePagerFile: "",
    sampleFiles: [],
    appUrl: "https://product-catalog-seach-public.cfapps.eu10-004.hana.ondemand.com/grounding-product-catalog",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/product-catalog-seach",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/european-commission-product-catalog-seach",
    isExternal: true
  },
  {
    id: "reconciliation-of-medicare-mandated-drug-pricing",
    title: "Underpayment Reasons Identification and Visualization",
    description:
      "Machine learning–based reconciliation of estimated vs. actual reimbursements under Medicare-mandated drug pricing, predicting clerk actions such as write-off, collections, or discount adjustments.",
    categories: [],
    industries: ["Health Care Equipment, Services & Pharmaceuticals"],
    icon: "stethoscope",
    docsPath: "reconciliation-of-medicare-mandated-drug-pricing",
    sampleFiles: [],
    appUrl: "https://btp-ai-sandbox.launchpad.cfapps.eu10.hana.ondemand.com/25c37970-20e2-414b-9963-956309c1d217.claimsservice.claims-0.0.1/index.html",
    completionDate: "2025-08-07",
    relatedBestPractices: [],
    sourceCodeUrl: "",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/mfp-claim-matching",
    onePagerFile: "images/Slide28.jpg",
    isExternal: true
  },
  {
    id: "sales-order-anomaly-detection",
    title: "Sales Order Anomaly Detection",
    description: "Analyze sales orders and detects anomalies in the data, creates clear explanations on what are the anomalies.",
    categories: [],
    industries: ["Health Care Equipment, Services & Pharmaceuticals"],
    icon: "alert",
    docsPath: "sales-order-anomaly-detection",
    sampleFiles: [],
    appUrl: "https://anomaly-dashboard.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/narrow-ai/anomaly-detection",
        title: "Anomaly Detection"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/anomaly-detection",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Amgen-Anomaly_detection",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_k3irwolb",
    onePagerFile: "images/Slide26.jpg",
    isExternal: true
  },
  {
    id: "ai-assisted-procurement-optimizer",
    title: "Vendor Selection Optimization",
    description:
      "Procurement optimization application to optimize the allocation of materials from different vendors to optimize total cost while keeping demand, with different tariffs simulation capabilities. Start by selecting a Material Description or Material Number from the left toolbar.",
    categories: [],
    industries: [],
    icon: "settings",
    docsPath: "ai-assisted-procurement-optimizer",
    sampleFiles: [],
    appUrl: "https://procurement_assistant.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/vendor-selection-optimization",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/procurement_assistant",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    isExternal: true
  },
  {
    id: "apex-assistant-post-sales-chatbot",
    title: "Apex Assistant: Post Sales Chatbot",
    description:
      "This app helps the customers to know the detail program for the car and schedule an appointment for new services. Customer can ask about the previous car services and information related.",
    categories: [],
    industries: ["Automobiles & Components"],
    icon: "car-rental",
    docsPath: "apex-assistant-post-sales-chatbot",
    sampleFiles: ["data/Support data to use.xlsx"],
    appUrl: "https://postsale-chatbot.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/post-sales-chatbot",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Zapata_Bot",
    onePagerFile: "images/Slide8.jpeg",
    isExternal: true
  },
  {
    id: "sap-rfqx-document-analysis-application-v2",
    title: "Intelligent Negotiation Assistant in Procurement",
    description:
      "An AI-powered solution for comparing RFP/RFQ responses, highlighting key offer details, and providing interactive dashboards with part comparisons, certifications, cost breakdowns, and risk insights—complete with a chatbot to access specific information instantly",
    categories: [],
    industries: [],
    icon: "compare-2",
    docsPath: "sap-rfqx-document-analysis-application-v2",
    sampleFiles: [],
    // appUrl: "https://rfqx_analysis.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/kg-rag-creation",
        title: "Graph-based RAG (1/2) KG Creation"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/kg-rag-query-pipeline",
        title: "Graph-based RAG (2/2) KG Query Pipeline"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/intelligent-negotiation-assistant-procurement",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Traton_RFQx",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    isExternal: true
  },
  {
    id: "sap-rfqx-document-analysis-application",
    title: "SAP RFQx Document Analysis Application",
    description: "Quick analyzer of RFQ documents to extract key relevant data and compare offers side by side, suggest best providers and chat about the documents.",
    categories: [],
    industries: [],
    icon: "compare-2",
    docsPath: "sap-rfqx-document-analysis-application",
    sampleFiles: [],
    appUrl: "https://uc-rfq-pdf_4.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/kg-rag-creation",
        title: "Graph-based RAG (1/2) KG Creation"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/kg-rag-query-pipeline",
        title: "Graph-based RAG (2/2) KG Query Pipeline"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/rfqx-doc-analysis-utilities",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/UC_RFQ",
    isExternal: true
  },
  {
    id: "document-information-extraction",
    title: "AI-Powered PDF Document Information Extraction",
    description: "This system enables structured information extraction from PDF documents using advanced artificial intelligence with vision capabilities.",
    categories: [],
    industries: ["Financial Services"],
    icon: "attachment",
    docsPath: "document-information-extraction",
    sampleFiles: [],
    appUrl: "https://data_extraction.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/ai-pdf-information-extraction",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Sesajal_DataExtraction",
    onePagerFile: "",
    isExternal: true
  },
  {
    id: "ai-assisted-customer-credit-check",
    title: "AI Assisted Customer Credit Check",
    description: "A document processing and credit evaluation engine with integrated report generation.",
    categories: [],
    industries: ["Financial Services"],
    icon: "money-bills",
    docsPath: "ai-assisted-customer-credit-check",
    sampleFiles: ["data/CGV.pdf", "data/Commercial Investigation.pdf", "data/CSF.pdf", "data/KYC.pdf", "data/Legal Investigation.pdf", "data/Vendor Comments.pdf"],
    appUrl: "https://credit_creation.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/customer-credit-check",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Sesajal_DataExtraction",
    onePagerFile: "images/Slide44.jpg",
    isExternal: true
  },
  {
    id: "ai-capability-matcher",
    title: "AI Capability Matcher",
    description: "Match AI capabilities from a dataset with products from a different dataset, to recommend technologies based on description.",
    categories: [],
    industries: [],
    icon: "puzzle",
    docsPath: "ai-capability-matcher",
    sampleFiles: ["data/ai_catalog.csv", "data/client_catalog.csv"],
    appUrl: "https://capability-matcher.cfapps.eu10-004.hana.ondemand.com/Capability_Matcher",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/vector-rag-embedding",
        title: "Vector-based RAG (1/2) Embedding"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/ai-capability-matcher",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/AI-Category-Matcher",
    onePagerFile: "images/Slide6.jpeg",
    isExternal: true
  },
  {
    id: "extraction-of-shipments",
    title: "SAP Supply chain: Extraction of shipments",
    description: "AI-driven automation of goods receipt and invoice processing with NLP-enabled notifications and seamless SAP S/4HANA integration.",
    categories: [],
    industries: [],
    icon: "shipping-status",
    docsPath: "extraction-of-shipments",
    sampleFiles: [
      "data/Factura ARM.PDF",
      "data/Factura CARGONET.pdf",
      "data/Factura DEFIBA.pdf",
      "data/Factura JET.pdf",
      "data/Factura LYS.pdf",
      "data/Factura MAERSK.pdf",
      "data/Listado general ICM.xlsx"
    ],
    appUrl: "https://extraction-of-shipments.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [],
    sourceCodeUrl: "",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/nike-argentina.git",
    onePagerFile: "images/Slide8.jpg",
    isExternal: true
  },
  {
    id: "tax-classification",
    title: "SAP Financial Process: Tax Classification",
    description: "AI-assisted tax validation and anomaly detection across invoices with automated extraction, analysis, and posting in SAP S/4HANA.",
    categories: [],
    industries: ["Utilities & Power"],
    icon: "number-sign",
    docsPath: "tax-classification",
    sampleFiles: [
      "data/bejing_2008.pdf",
      "data/Factura_Caso_1_-_00010-00002041.pdf",
      "data/Factura_Caso_2_-_9996-00001444.pdf",
      "data/Factura_Caso_2_-_NC_9996-00000416.pdf",
      "data/Factura_Caso_3_-_Varias_PO.pdf",
      "data/FC1020-00967563_20250430150724.198_X.pdf",
      "data/sample-invoice-2.pdf",
      "data/sample-invoice-3.pdf"
    ],
    appUrl: "https://pampa-en.cfapps.eu01-canary.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [],
    sourceCodeUrl: "",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/pampa-energia.git",
    onePagerFile: "images/Slide34.jpg",
    isExternal: true
  },
  {
    id: "mail-classification",
    title: "Text/Email Classification",
    description: "Application to classify text into configurable categories.",
    categories: [],
    industries: [],
    icon: "email",
    docsPath: "mail-classification",
    sampleFiles: ["data/sample_mail.txt"],
    appUrl: "https://mail.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/text-classification",
    isExternal: true
  },
  {
    id: "kpi-wizard",
    title: "KPI Wizard",
    description: "Analyze sales data and creates KPIs.",
    categories: [],
    industries: [],
    icon: "bar-chart",
    docsPath: "kpi-wizard",
    sampleFiles: ["data/dummy_sales_data.xlsx"],
    appUrl: "https://kpi-wizard.cfapps.eu01-canary.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/bago-argentina.git",
    isExternal: true
  },
  {
    id: "sales-forecast-and-cash-optimization",
    title: "Sales Forecast and Cash Optimization",
    description: "AI-powered sales performance agent for unified data visibility, decline analysis, and actionable recommendations to protect working capital.",
    categories: [],
    industries: ["Automobiles & Components"],
    icon: "future",
    docsPath: "sales-forecast-and-cash-optimization",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    onePagerFile: "images/Slide16.jpg",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_z630actv",
    isExternal: true
  },
  {
    id: "ai-powered-margin-and-cost-control",
    title: "AI-Powered Margin & Cost Control",
    description: "AI-enabled procurement optimization with anomaly detection, predictive cost forecasting, and conversational insights for margin protection and spend control.",
    categories: [],
    industries: ["Energy Equipment & Services"],
    icon: "money-bills",
    docsPath: "ai-powered-margin-and-cost-control",
    sampleFiles: [],
    appUrl:
      "https://btp-ai-sandbox.launchpad.cfapps.eu10.hana.ondemand.com/0d64c625-9ba5-42ea-97a1-7f28a246c45a.equipmentservice.nsequipments-0.0.1/index.html#/Equipment(30108623)",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_846fmk6w",
    onePagerFile: "images/Slide38.jpg",
    isExternal: true
  },
  {
    id: "documents-validation",
    title: "Documents Validation",
    description: "AI-powered project documentation agent for automated validation of PDFs/Excels, missing data detection, and vendor communication.",
    categories: [],
    industries: ["Utilities & Power"],
    icon: "checklist",
    docsPath: "documents-validation",
    sampleFiles: [
      "data/70123399_INTRUSIVE TEST CARD/70123399_INTRUSIVE TEST CARD.xlsx",
      "data/70123399_INTRUSIVE TEST CARD/70123399_INTRUSIVE TEST RESULTS.xlsx",
      "data/70125111_MAKE READY/70123399_MAKE READY.pdf"
    ],
    appUrl: "https://PGE-v2.cfapps.eu10-004.hana.ondemand.com",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/PG-E",
    videoUrl: "",
    onePagerFile: "images/Slide36.jpg",
    isExternal: true
  },
  {
    id: "b2b-post-sales",
    title: "SAP Sales Support: Customer Support Agent",
    description: "AI-powered B2B support agent for real-time order, shipment, and billing status with natural language summaries and customer self-service.",
    categories: [],
    industries: ["Semiconductor Equipment"],
    icon: "customer",
    docsPath: "b2b-post-sales",
    sampleFiles: [],
    appUrl: "https://b2b-post-sales.cfapps.eu10-004.hana.ondemand.com",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    isExternal: true
  },
  {
    id: "financial-statement-agent",
    title: "SAP Financial Process: Financial Statement Assistant",
    description: "AI-powered B2B support agent for real-time order, shipment, and billing status with natural language summaries and customer self-service.",
    categories: [],
    industries: ["Materials & Chemicals"],
    icon: "accounting-document-verification",
    docsPath: "financial-statement-agent",
    sampleFiles: [],
    appUrl: "https://sk-poc.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/sk-poc.git",
    videoUrl: "",
    onePagerFile: "images/Slide40.jpg",
    isExternal: true
  },
  {
    id: "pm-data-science",
    title: "SAP Plant Maintenance: Data-Science Assistant",
    description: "AI-driven maintenance intelligence agent for anomaly detection, dynamic KPIs, and natural language insights across integrated data sources.",
    categories: [],
    industries: ["Materials & Chemicals"],
    icon: "bar-chart",
    docsPath: "pm-data-science",
    sampleFiles: [],
    appUrl: "https://pm-data-science.cfapps.eu01-canary.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/pm-data-science",
    videoUrl: "",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    isExternal: true
  },
  {
    id: "agentic-cold-chain-logistics",
    title: "Agentic Transportation Lane Risk Analysis",
    description: "AI-driven container optimization using predictive risk metrics and historical shipment data with automated assignment and mitigation strategies.",
    categories: [],
    industries: [],
    icon: "fridge",
    docsPath: "agentic-cold-chain-logistics",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/illy.git",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_0tb3z8d5",
    onePagerFile: "images/Slide4.jpg",
    isExternal: true
  },
  {
    id: "cash-optimization",
    title: "Cash Optimization",
    description: "AI-driven payment advisory agent for real-time cash position assessment and proactive payment service recommendations to optimize cash flow.",
    categories: [],
    industries: ["Financial Services"],
    icon: "money-bills",
    docsPath: "cash-optimization",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/jpmc-saphire-demo-custom-chatbot",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_lxucneju",
    onePagerFile: "images/Slide46.jpg",
    isExternal: true
  },
  {
    id: "ai-agents-for-gr-and-invoice-workflows",
    title: "Touchless Transactions: AI Agents for GR & Invoice Workflows",
    description: "AI-powered goods receipt and invoice processing agent with NLP-driven notifications and seamless SAP S/4HANA automation..",
    categories: [],
    industries: ["Consumer Durables & Apparel"],
    icon: "sales-order",
    docsPath: "ai-agents-for-gr-and-invoice-workflows",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/adidas-V1",
    videoUrl: "",
    onePagerFile: "images/Slide6.jpg",
    isExternal: true
  },
  {
    id: "log-analyser",
    title: "Log Analyser",
    description: "AI-powered log analyser for SAP systems, analyzing logs to identify patterns and anomalies.",
    categories: [],
    industries: [],
    icon: "log",
    docsPath: "log-analyser",
    sampleFiles: [],
    appUrl: "https://btp-ai-sandbox.launchpad.cfapps.eu10.hana.ondemand.com/27c95a73-2650-41aa-9bf2-0c7d61e48d12.logreportservice.sapbtpailog-0.0.1/index.html",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/ai-log-analyzer",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/ai-log-analyzer",
    videoUrl: "",
    onePagerFile: "",
    isExternal: true
  },
  {
    id: "supply-chain-agent",
    title: "Supply Chain Agent",
    description: "AI-powered automation streamlines supply chain reporting by consolidating Excel data, reducing analysis time, and optimizing decision-making.",
    categories: [],
    industries: ["Consumer Durables & Apparel"],
    icon: "chain-link",
    docsPath: "supply-chain-agent",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/Slide10.jpg",
    isExternal: true
  },
  {
    id: "sales-order-automation",
    title: "Sales Order Automation",
    description: "AI-driven order processing automates data extraction from customer POs, reducing manual errors and speeding up sales order creation in S4 HANA.",
    categories: [],
    industries: ["Consumer Durables & Apparel"],
    icon: "process",
    docsPath: "sales-order-automation",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/Slide12.jpg",
    isExternal: true
  },
  {
    id: "dairy-maximizing-yield-minimizing-waste",
    title: "AI-Powered Dashboard for Maximizing Yield, Minimizing Waste",
    description:
      "An AI-powered platform unifies dairy supply chain data to enhance visibility, optimize efficiency, strengthen vendor partnerships, and unlock multimillion-pound savings.",
    categories: [],
    industries: ["Food & Beverage"],
    icon: "nutrition-activity",
    docsPath: "dairy-maximizing-yield-minimizing-waste",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/Slide3.jpeg",
    isExternal: true
  },
  {
    id: "sales-order-extractor",
    title: "Sales Order Extractor",
    description: "Extract sales order data from PDF/Excel documents using AI and automatically create sales orders in SAP HANA.",
    categories: [],
    industries: [],
    icon: "sales-order",
    docsPath: "sales-order-extractor",
    sampleFiles: [],
    appUrl: "https://sales-order-extractor-ui.cfapps.eu10-004.hana.ondemand.com/pdf-extraction",
    completionDate: "",
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/sales-order-extractor",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Organic-Valley_final",
    videoUrl: "",
    onePagerFile: "",
    isExternal: true
  },
  {
    id: "partial-payment-investigation",
    title: "Partial Payment Investigation",
    description:
      "SAP BTP AI accelerates deduction clearing by analyzing partial payment data across sources, reducing lead time and supporting AR clerks in process orchestration.",
    categories: [],
    industries: ["Food & Beverage"],
    icon: "inspect",
    docsPath: "partial-payment-investigation",
    sampleFiles: [],
    appUrl: "https://btp-ai-sandbox.launchpad.cfapps.eu10.hana.ondemand.com/9ea8015b-3ef5-408b-aa19-68700fc94320.deductionagentservice.aritems-0.0.1",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/deduction-agent",
    videoUrl: "",
    onePagerFile: "images/Slide22.jpg",
    isExternal: true
  },
  {
    id: "dynamic-pricing-simulation",
    title: "Dynamic Pricing Simulation",
    description:
      "SAP BTP AI empowers smarter pricing decisions by predicting margin impacts, optimizing contract selling prices, and reducing uncertainty during sudden material cost increases.",
    categories: [],
    industries: ["Food & Beverage"],
    icon: "simulate",
    docsPath: "dynamic-pricing-simulation",
    sampleFiles: [],
    appUrl: "https://fgf_margin_prototype.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/Slide24.jpg",
    isExternal: true
  },

  {
    id: "document-data-extraction-agent",
    title: "Document Data Extraction Agent",
    description: "Upload an invoice document to have its information extracted digitally.",
    categories: [],
    industries: [],
    icon: "attachment",
    docsPath: "document-data-extraction-agent",
    sampleFiles: ["invoices/sample-invoice-1.pdf", "invoices/sample-invoice-2.pdf", "invoices/sample-invoice-3.pdf", "invoices/chinese_blogger.pdf", "invoices/bejing_2008.pdf"],
    appUrl: "https://invoiceextraction.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "2024-07-28",
    sourceCodeUrl: "",
    isExternal: false
  },
  {
    id: "llm-qa-chatbot",
    title: "LLM Q&A Chatbot",
    description: "Upload documents to ground chat bot responses.",
    categories: [],
    industries: [],
    icon: "ai",
    docsPath: "llm-qa-chatbot",
    sampleFiles: ["dummy_sales_data.xlsx"],
    appUrl: "https://ai4ucoe-agent.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "2024-07-28",
    sourceCodeUrl: "",
    isExternal: false
  },
  {
    id: "rfp-evaluation-agent",
    title: "RFP Evaluation Agent",
    description: "Upload multiple PDF RFQs and easily compare results.",
    categories: [],
    industries: [],
    icon: "quality-issue",
    docsPath: "rfp-evaluation-agent",
    sampleFiles: [
      "00_RS PRO Squirrel Cage Motor AC Motor, 0.75kW, IE3, 3 Phase, 4 Pole, 400 V, Foot Mount Mounting A700000006779745.pdf",
      "00_RS PRO Squirrel Cage Motor AC Motor, 3kW, IE3, 3 Phase, 2 Pole, 400 V, Foot Mount Mounting A700000006779801.pdf",
      "01_JSP EVO2 White Safety Helmet , Adjustable, Ventilated A700000009347754.pdf",
      "01_Petzl VERTEX VENT White Safety Helmet with Chin Strap, Adjustable, Ventilated 0900766b816f906d.pdf",
      "02_COVERALL CLEAN 0900766b81525eb9.pdf",
      "02_Ultra Safe Goggles A700000012793452.pdf",
      "20191113 Nielsen ServicesAgreement (PDF) Orlando_2019_Nov_13 ver 2.pdf",
      "20191113 Nielsen ServicesAgreement (PDF) Orlando_2019_Sep_17 ver 1.pdf",
      "Third Amendment to Amended and Restated Employment Agreement among MediaAlpha 1.pdf",
      "Third Amendment to Amended and Restated Employment Agreement among MediaAlpha 2.pdf"
    ],
    appUrl: "https://uc-rfq-pdf_3.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "2024-07-28",
    sourceCodeUrl: "",
    isExternal: false
  },
  {
    id: "multi-document-rag",
    title: "Multi Document RAG - HANA VE & GenAI",
    description: "Upload files which are vectorized for memory and accuracy via a chatbot",
    categories: [],
    industries: [],
    icon: "database",
    docsPath: "multi-document-rag",
    sampleFiles: [],
    appUrl: "https://multi-rag.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "2024-07-28",
    sourceCodeUrl: "",
    isExternal: false
  }
];

/**
 * Helper functions for working with project data
 */

export function getProjectById(id) {
  return projects.find((project) => project.id === id);
}

export function getProjectsByCategory(category) {
  return projects.filter((project) => project.categories.includes(category));
}

export function getProjectsByStatus(status) {
  return projects.filter((project) => project.status === status);
}

export function getCategoryLabel(category) {
  const labels = {
    "ai-ml": "AI/ML",
    automation: "Automation",
    analytics: "Analytics",
    integration: "Integration",
    development: "Development",
    "document-processing": "Document Processing",
    utilities: "Utilities",
    procurement: "Procurement",
    rag: "RAG",
    agent: "Agent"
  };
  return labels[category] || category;
}

export function getStatusLabel(status) {
  const labels = {
    active: "Active",
    "in-development": "In Development",
    completed: "Completed",
    archived: "Archived"
  };
  return labels[status] || status;
}

/**
 * Visibility helpers
 */

export function getPublicProjects() {
  return projects.filter((project) => project.isExternal === true);
}

/**
 * Project data file utilities
 * These functions help access readme files and sample files in public/data/
 */

/**
 * Get the README file URL for a project
 */
export function getProjectReadmeUrl(project) {
  return `/data/${project.docsPath}/README.md`;
}

/**
 * Get the project data directory URL
 */
export function getProjectDataUrl(project) {
  return `/data/${project.docsPath}`;
}

/**
 * Fetch and parse the README content for a project
 */
export async function getProjectReadmeContent(project) {
  try {
    const response = await fetch(getProjectReadmeUrl(project));
    if (!response.ok) {
      throw new Error(`Failed to fetch README: ${response.status}`);
    }

    const content = await response.text();

    // Check if we got HTML instead of markdown (happens when server returns 404 page)
    if (content.includes("<!DOCTYPE html>") || content.includes("<html>")) {
      throw new Error("Received HTML instead of markdown content");
    }

    // Check if content is too short to be a real README
    if (content.trim().length < 10) {
      throw new Error("README content is too short or empty");
    }

    return content;
  } catch (error) {
    console.error(`Error fetching README for ${project.title}:`, error);
    // Return null to indicate failure, let the UI handle the missing content
    return null;
  }
}

/**
 * Get file type based on extension
 */
function getFileType(filename) {
  const extension = filename.toLowerCase().split(".").pop();
  const typeMap = {
    pdf: "PDF Document",
    txt: "Text File",
    xlsx: "Excel Spreadsheet",
    xls: "Excel Spreadsheet",
    csv: "CSV Data",
    md: "Markdown",
    pptx: "PowerPoint Presentation",
    docx: "Word Document",
    json: "JSON Data",
    yaml: "YAML Configuration",
    yml: "YAML Configuration",
    py: "Python Script",
    js: "JavaScript File",
    ts: "TypeScript File",
    xml: "XML Document",
    zip: "Archive File"
  };
  return typeMap[extension] || "File";
}

/**
 * Format completion date for display
 */
export function formatCompletionDate(dateString) {
  if (!dateString) return "Not specified";

  const date = new Date(dateString);
  if (isNaN(date.getTime())) return "Invalid date";

  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric"
  });
}

/**
 * Get projects sorted by completion date (newest first)
 */
export function getProjectsByCompletionDate() {
  return [...projects].sort((a, b) => {
    if (!a.completionDate && !b.completionDate) return 0;
    if (!a.completionDate) return 1;
    if (!b.completionDate) return -1;
    return new Date(b.completionDate) - new Date(a.completionDate);
  });
}

/**
 * Complete project data loader for the project page
 */
export async function getCompleteProjectData(projectId) {
  const project = getProjectById(projectId);
  if (!project) {
    return null;
  }

  const readmeContent = await getProjectReadmeContent(project);

  // Add URLs and types to sample files
  const baseUrl = getProjectDataUrl(project);
  const sampleFilesWithUrls = project.sampleFiles.map((filename) => ({
    name: filename,
    url: `${baseUrl}/${filename}`,
    type: getFileType(filename)
  }));

  return {
    ...project,
    readmeContent,
    sampleFiles: sampleFilesWithUrls,
    dataUrl: baseUrl
  };
}
