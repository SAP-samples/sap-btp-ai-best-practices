/**
 * Projects data for AI4U website
 * This file contains all project information used across different pages
 */

export const projects = [
  {
    id: "agentic-email-automation",
    title: "Agentic Email Automation",
    isAgentic: true,
    description:
      "AI Email Agent that prioritizes and answers emails end-to-end, integrating with SAP S/4HANA, Ariba, and more systems to traverse decision trees and compliance gates.",
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
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_51n3w4w0",
    onePagerFile: "images/Slide5.jpeg",
    isExternal: true
  },
  {
    id: "ai-powered-email-cockpit",
    title: "AI-Powered Email Cockpit",
    description: "An easy-to-use tool with an email-style interface and built-in chatbot to quickly organize, track, and resolve payment-related requests.",
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
    id: "utilities-rate-compare-and-export",
    title: "Intelligent Tariff Extraction & Comparison",
    description: "AI-powered extraction and comparison of utility rate tariffs from PDF documents to identify and export rate changes.",
    icon: "compare-2",
    docsPath: "utilities-rate-compare-and-export",
    sampleFiles: ["data/EVERGREEN STATE 2025-11.pdf", "data/EVERGREEN STATE 2025-12.pdf"],
    videoUrl: "",
    appUrl: "https://utilities-rate-compare-and-export-live.cfapps.eu10-004.hana.ondemand.com/utilities-rate-compare-and-export",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/utilities-rate-compare-and-export",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/duke-energy-rate-tariffs",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    isExternal: true
  },
  {
    id: "image-diagram-to-signavio",
    title: "Image Diagram to Signavio",
    description: "Analyze images of diagrams and transform them into standard BPMN format to be used in Signavio",
    icon: "tnt/bdd-diagram",
    docsPath: "image-diagram-to-signavio",
    sampleFiles: ["data/Process Cab Booking Request PowerPoint.png"],
    appUrl: "https://diagram-to-bpmn.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/diagram-to-bpmn",
    originalSourceCodeUrl: "",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_2v0t5j5p",
    onePagerFile: "images/New one pagers.jpg",
    isExternal: true
  },
  {
    id: "video-incident-and-safety-monitoring",
    title: "Video Incident and Safety Monitoring",
    description:
      "Analyze video to identify personnel on the site, describe ongoing high-risk recovery operations, flag potential safety incidents, and infer basic site conditions such as weather and visibility.",
    icon: "video",
    docsPath: "video-incident-and-safety-monitoring",
    sampleFiles: ["data/railings on the stairs are not installed.mp4", "data/walking along the tube.mp4"],
    appUrl: "https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com/analyze-standalone.html",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/video-incident-and-safety-monitoring",
    isExternal: true,
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_zymttv7o"
  },
  {
    id: "wired-for-intelligence",
    title: "Wired for Intelligence: powering clean orders from messy customer inputs",
    description:
      "Automates the extraction and mapping of sales order data from unstructured customer documents into SAP, reducing manual effort, improving accuracy, and enabling scalable, reliable order processing.",
    icon: "workflow-tasks",
    docsPath: "wired-for-intelligence",
    sampleFiles: [],
    appUrl:
      "https://btp-ai-sandbox.launchpad.cfapps.eu10.hana.ondemand.com/0c1d4d1c-2393-498a-8659-b114030ebe73.purchaseorderextractorservice.purchaseorderextraction-0.0.1/index.html",
    completionDate: "",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/purchase-order-extractor",
    isExternal: true,
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_2ql2791x",
    onePagerFile: "images/One-pager.jpg"
  },
  {
    id: "intelligent-procurement-assistant",
    title: "Intelligent Procurement Assistant",
    description: "AI assistant that finds materials, recommends approved vendors, and automatically creates purchase requests from catalog, text, or quotation PDFs.",
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
    icon: "search",
    docsPath: "product-catalog-search",
    onePagerFile: "images/Ai4U - Flipbook.jpg",
    sampleFiles: [],
    appUrl: "https://product-catalog-seach-public.cfapps.eu10-004.hana.ondemand.com/grounding-product-catalog",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      },
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/rag/document-grounding",
        title: "Document Grounding"
      }
    ],
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_p8pta9ul",
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/product-catalog-seach",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/european-commission-product-catalog-seach",
    isExternal: true
  },
  {
    id: "reconciliation-of-medicare-mandated-drug-pricing",
    title: "Underpayment Reasons Identification and Visualization",
    description:
      "Machine learning–based reconciliation of estimated vs. actual reimbursements under Medicare-mandated drug pricing, predicting clerk actions such as write-off, collections, or discount adjustments.",
    icon: "stethoscope",
    docsPath: "reconciliation-of-medicare-mandated-drug-pricing",
    sampleFiles: [],
    appUrl: "https://btp-ai-sandbox.launchpad.cfapps.eu10.hana.ondemand.com/25c37970-20e2-414b-9963-956309c1d217.claimsservice.claims-0.0.1/index.html",
    completionDate: "2025-08-07",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    sourceCodeUrl: "",
    internalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/mfp-claim-matching",
    onePagerFile: "images/Slide28.jpg",
    isExternal: true
  },
  {
    id: "sales-order-anomaly-detection",
    title: "Sales Order Anomaly Detection",
    description: "Analyze sales orders and detects anomalies in the data, creates clear explanations on what are the anomalies.",
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
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_ndxhkobu",
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
    isAgentic: true,
    isExternal: true
  },
  {
    id: "sap-rfqx-document-analysis-application-v2",
    title: "Intelligent Negotiation Assistant in Procurement",
    description:
      "An AI-powered solution for comparing RFP/RFQ responses, highlighting key offer details, and providing interactive dashboards with part comparisons, certifications, cost breakdowns, and risk insights—complete with a chatbot to access specific information instantly",
    icon: "compare-2",
    docsPath: "sap-rfqx-document-analysis-application-v2",
    sampleFiles: [],
    appUrl: "https://rfqx_analysis.cfapps.eu10-004.hana.ondemand.com/",
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
    onePagerFile: "images/Slide9.jpg",
    isExternal: true
  },
  {
    id: "ai-assisted-customer-credit-check",
    title: "AI Assisted Customer Credit Check",
    description: "A document processing and credit evaluation engine with integrated report generation.",
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
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_w8zgrcz7",
    isExternal: true
  },
  {
    id: "extraction-of-shipments",
    title: "SAP Supply chain: Extraction of shipments",
    description: "AI-driven automation of goods receipt and invoice processing with NLP-enabled notifications and seamless SAP S/4HANA integration.",
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
    icon: "email",
    docsPath: "mail-classification",
    sampleFiles: ["data/sample_mail.txt"],
    appUrl: "https://mail.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/text-classification",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    isExternal: true
  },
  {
    id: "kpi-wizard",
    title: "KPI Wizard",
    description: "Analyze sales data and creates KPIs.",
    icon: "bar-chart",
    docsPath: "kpi-wizard",
    sampleFiles: ["data/dummy_sales_data.xlsx"],
    appUrl: "https://kpi-wizard.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/bago-argentina.git",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    isExternal: true
  },
  {
    id: "sales-forecast-and-cash-optimization",
    title: "Sales Forecast and Cash Optimization",
    description: "AI-powered sales performance agent for unified data visibility, decline analysis, and actionable recommendations to protect working capital.",
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
    icon: "fridge",
    docsPath: "agentic-cold-chain-logistics",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/illy.git",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_0tb3z8d5",
    onePagerFile: "images/Slide4.jpeg",
    isExternal: true
  },
  {
    id: "cash-optimization",
    title: "Cash Optimization",
    description: "AI-driven payment advisory agent for real-time cash position assessment and proactive payment service recommendations to optimize cash flow.",
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
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/touchless-transactions-ai-agent-for-gr-and-invoice-workflows",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/adidas-V1",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_8oewer4k",
    onePagerFile: "images/Slide6.jpg",
    isExternal: true
  },
  {
    id: "log-analyser",
    title: "Log Analyser",
    description: "AI-powered log analyser for SAP systems, analyzing logs to identify patterns and anomalies.",
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
    id: "agentic-chatbot",
    title: "Data Analyst Chatbot Agent with SAC Dashboard integration",
    isAgentic: true,
    description: "An advanced AI Agent, embedded in SAC, that uses the same governed dashboard data and analytics tools to deliver concise, table-formatted insights in real time.",
    icon: "business-suite/answered",
    docsPath: "agentic-chatbot",
    sampleFiles: [],
    appUrl: "https://agentic-chatbot.cfapps.eu10-004.hana.ondemand.com/",
    completionDate: "",
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/agentic-chatbot",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/organic-valley-ai-utilities",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_npo3xufq",
    onePagerFile: "images/New one pagers.jpg",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    isExternal: true
  },
  {
    id: "document-outlier-detection",
    title: "Document Outlier Detection",
    description: "Detect outliers in tabular data documents using AI, and generate a report with the outliers found.",
    icon: "alert",
    docsPath: "document-outlier-detection",
    sampleFiles: ["data/Sample Data for Mis-key scenario.xlsx"],
    appUrl: "https://document-outlier-detection.cfapps.eu10-004.hana.ondemand.com/outlier-detection",
    completionDate: "",
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/document-outlier-detection",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/organic-valley-ai-utilities",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_npo3xufq&kalturaStartTime=133",
    onePagerFile: "images/New one pagers.jpg",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    isExternal: true
  },
  {
    id: "sales-order-extractor",
    title: "Sales Order Document Extraction",
    description: "Extract sales order data from PDF/Excel documents using AI and automatically create sales orders in SAP HANA.",
    icon: "sales-order",
    docsPath: "sales-order-extractor",
    sampleFiles: ["data/PO-50871.pdf", "data/PO-84193.pdf"],
    appUrl: "https://sales-order-extractor-ui.cfapps.eu10-004.hana.ondemand.com/pdf-extraction",
    completionDate: "",
    sourceCodeUrl: "https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/use-cases/sales-order-extractor",
    originalSourceCodeUrl: "https://github.tools.sap/sap-btp-ai-services-coe/Organic-Valley_final",
    videoUrl: "https://sapvideo.cfapps.eu10-004.hana.ondemand.com/?entry_id=1_1uxl6mt2",
    onePagerFile: "images/New one pagers.jpg",
    relatedBestPractices: [
      {
        url: "https://btp-ai-bp.docs.sap/docs/technical-view/generative-ai/plain/access-to-generative-ai-models",
        title: "Access to Generative AI Models"
      }
    ],
    isExternal: true
  },
  {
    id: "partial-payment-investigation",
    title: "Partial Payment Investigation",
    description:
      "SAP BTP AI accelerates deduction clearing by analyzing partial payment data across sources, reducing lead time and supporting AR clerks in process orchestration.",
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
    icon: "simulate",
    docsPath: "dynamic-pricing-simulation",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/Slide24.jpg",
    isExternal: true
  },
  {
    id: "ai-powered-ticket-routing-support",
    title: "AI-powered Ticket Routing Support",
    description: "AI-powered real-time guidance to resolve issues independently and automatically classify their requests without manual input.",
    icon: "business-suite/answered",
    docsPath: "ai-powered-ticket-routing-support",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/New one pagers.jpg",
    isExternal: true
  },
  {
    id: "utilities-high-bill-explainer",
    title: "Utilities High Bill Explainer",
    description:
      "Guide the consumer through easy to understand language on what changes have occurred MoM or YoY to help provide the consumer confidence in the accuracy of their bill.",
    icon: "money-bills",
    docsPath: "utilities-high-bill-explainer",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/New one pagers.jpg",
    isExternal: true
  },
  {
    id: "intelligent-store-selling-forecast",
    title: "Intelligent Store Selling Forecast",
    description: "Unified forecasting model that integrates store traffic, historical sales, and promotional data to deliver accurate, actionable insights at the store level.",
    icon: "bar-chart",
    docsPath: "intelligent-store-selling-forecast",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/New one pagers.jpg",
    isExternal: true
  },
  {
    id: "utilities-best-offer-recommendation-assistant",
    title: "Utilities Best Offer Recommendation Assistant",
    description:
      "A recommendation agent that scans every customer profile and highlights the programs they likely qualify for, then guides CSRs and sales agents with follow up questions to confirm eligibility.",
    icon: "money-bills",
    docsPath: "utilities-best-offer-recommendation-assistant",
    sampleFiles: [],
    appUrl: "",
    completionDate: "",
    sourceCodeUrl: "",
    originalSourceCodeUrl: "",
    videoUrl: "",
    onePagerFile: "images/New one pagers.jpg",
    isExternal: true
  }

  // {
  //   id: "document-data-extraction-agent",
  //   title: "Document Data Extraction Agent",
  //   description: "Upload an invoice document to have its information extracted digitally.",
  //   icon: "attachment",
  //   docsPath: "document-data-extraction-agent",
  //   sampleFiles: ["invoices/sample-invoice-1.pdf", "invoices/sample-invoice-2.pdf", "invoices/sample-invoice-3.pdf", "invoices/chinese_blogger.pdf", "invoices/bejing_2008.pdf"],
  //   appUrl: "https://invoiceextraction.cfapps.eu10-004.hana.ondemand.com/",
  //   completionDate: "2024-07-28",
  //   sourceCodeUrl: "",
  //   isExternal: false
  // },
  // {
  //   id: "llm-qa-chatbot",
  //   title: "LLM Q&A Chatbot",
  //   description: "Upload documents to ground chat bot responses.",
  //   icon: "ai",
  //   docsPath: "llm-qa-chatbot",
  //   sampleFiles: ["dummy_sales_data.xlsx"],
  //   appUrl: "https://ai4ucoe-agent.cfapps.eu10-004.hana.ondemand.com/",
  //   completionDate: "2024-07-28",
  //   sourceCodeUrl: "",
  //   isExternal: false
  // },
  // {
  //   id: "rfp-evaluation-agent",
  //   title: "RFP Evaluation Agent",
  //   description: "Upload multiple PDF RFQs and easily compare results.",
  //   icon: "quality-issue",
  //   docsPath: "rfp-evaluation-agent",
  //   sampleFiles: [
  //     "00_RS PRO Squirrel Cage Motor AC Motor, 0.75kW, IE3, 3 Phase, 4 Pole, 400 V, Foot Mount Mounting A700000006779745.pdf",
  //     "00_RS PRO Squirrel Cage Motor AC Motor, 3kW, IE3, 3 Phase, 2 Pole, 400 V, Foot Mount Mounting A700000006779801.pdf",
  //     "01_JSP EVO2 White Safety Helmet , Adjustable, Ventilated A700000009347754.pdf",
  //     "01_Petzl VERTEX VENT White Safety Helmet with Chin Strap, Adjustable, Ventilated 0900766b816f906d.pdf",
  //     "02_COVERALL CLEAN 0900766b81525eb9.pdf",
  //     "02_Ultra Safe Goggles A700000012793452.pdf",
  //     "20191113 Nielsen ServicesAgreement (PDF) Orlando_2019_Nov_13 ver 2.pdf",
  //     "20191113 Nielsen ServicesAgreement (PDF) Orlando_2019_Sep_17 ver 1.pdf",
  //     "Third Amendment to Amended and Restated Employment Agreement among MediaAlpha 1.pdf",
  //     "Third Amendment to Amended and Restated Employment Agreement among MediaAlpha 2.pdf"
  //   ],
  //   appUrl: "https://uc-rfq-pdf_3.cfapps.eu10-004.hana.ondemand.com/",
  //   completionDate: "2024-07-28",
  //   sourceCodeUrl: "",
  //   isExternal: false
  // },
  // {
  //   id: "multi-document-rag",
  //   title: "Multi Document RAG - HANA VE & GenAI",
  //   description: "Upload files which are vectorized for memory and accuracy via a chatbot",
  //   icon: "database",
  //   docsPath: "multi-document-rag",
  //   sampleFiles: [],
  //   appUrl: "https://multi-rag.cfapps.eu10-004.hana.ondemand.com/",
  //   completionDate: "2024-07-28",
  //   sourceCodeUrl: "",
  //   isExternal: false
  // }
];

/**
 * Helper functions for working with project data
 */

export function getProjectById(id) {
  return projects.find((project) => project.id === id);
}

/**
 * Visibility helpers
 */

export function getPublicProjects() {
  return projects.filter((project) => project.isExternal === true);
}

/**
 * Get the category label from the category ID
 */
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
