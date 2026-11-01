# SAP BTP AI Best Practices

This repository provides a curated collection of best practices for implementing AI solutions on the SAP Business Technology Platform (BTP). It covers **Generative AI**, **Agentic AI**, **Narrow AI** (classical machine learning and tabular foundation models) and **SAP AI services**, with practical examples, code, and guidance for enterprise scenarios.

The accompanying articles are published on the [SAP BTP AI Best Practices](https://btp-ai-bp.docs.sap/) documentation site.

## Structure

- **[generative-ai/](./generative-ai/README.md)**: Generative AI and LLM best practices (e.g., model access, prompt engineering, RAG, content filtering)
- **[agentic-ai/](./agentic-ai/README.md)**: Code-based AI agents built on the generative AI hub (e.g., ReAct agents, LangGraph workflows)
- **[narrow-ai/](./narrow-ai/README.md)**: Predictive/classical machine learning best practices (e.g., regression, anomaly detection) and SAP-RPT for tabular predictions
- **[ai-services/](./ai-services/)**: Best practices for using AI services on SAP BTP (e.g., SAP Document AI, SAP Translation Hub)

Each subdirectory contains:

- Topic-specific documentation and guidance
- Example code in one or more programming languages
- Step-by-step setup and usage instructions

## Catalog

### Generative AI

| Best practice | Languages / frameworks |
|---|---|
| [Access to generative AI models](./generative-ai/access-to-generative-ai-models/README.md) | TypeScript, Python, Java, CAP |
| [Content filtering](./generative-ai/content-filtering/README.md) | TypeScript, Python, Java, CAP |
| [Data masking](./generative-ai/data-masking/README.md) | TypeScript, Python, Java, CAP |
| [Prompt templating](./generative-ai/prompt-templating/README.md) | TypeScript, Python, Java, CAP |
| [Prompt registry](./generative-ai/prompt-registry/README.md) | TypeScript, CAP |
| [Document grounding](./generative-ai/document-grounding/README.md) | Python |
| [Vector RAG – embedding](./generative-ai/vector-rag-embedding/README.md) | TypeScript, Python, Java, CAP |
| [Vector RAG – query](./generative-ai/vector-rag-query/README.md) | TypeScript, Python, Java, CAP |
| [Knowledge Graph RAG](./generative-ai/knowledge-graph/README.md) | Python |

Setup of the SAP AI Core service instance, service key and destination is described in [Prerequisites](./generative-ai/Prerequisites.md).

### Agentic AI

| Best practice | Languages / frameworks |
|---|---|
| [Code-based agents – native ReAct](./agentic-ai/code-based-agents/native-react/README.md) | Python |
| [Code-based agents – LangGraph ReAct](./agentic-ai/code-based-agents/langgraph-react/README.md) | Python (LangGraph, LangChain) |

### Narrow AI

| Best practice | Languages / frameworks |
|---|---|
| [Anomaly detection](./narrow-ai/anomaly-detection/README.md) | Python (hana-ml) |
| [Classification](./narrow-ai/classification/README.md) | Python (hana-ml) |
| [Clustering](./narrow-ai/clustering/README.md) | Python (hana-ml) |
| [Regression](./narrow-ai/regression/README.md) | Python (hana-ml) |
| [Time-series forecasting](./narrow-ai/time-series-forecasting/README.md) | Python (hana-ml) |
| [SAP-RPT (tabular foundation model)](./narrow-ai/RPT-1/README.md) | Python, TypeScript, Java |

### AI Services

| Best practice | Languages / frameworks |
|---|---|
| [SAP Document AI](./ai-services/document-ai/python/README.md) | Python |
| [SAP Translation Hub](./ai-services/translation-hub/README.md) | Python (notebooks) |

## SDKs used in the samples

The generative AI, agentic AI and SAP-RPT samples use the SAP Cloud SDK for AI:

- Python: [`sap-ai-sdk-gen`](https://pypi.org/project/sap-ai-sdk-gen/) – source code at [SAP/ai-sdk-python](https://github.com/SAP/ai-sdk-python)
- JavaScript / TypeScript: `@sap-ai-sdk/*` packages – [SAP/ai-sdk-js](https://github.com/SAP/ai-sdk-js)
- Java: [SAP/ai-sdk-java](https://github.com/SAP/ai-sdk-java)

Release notes for all three SDKs are linked from the [SAP Cloud SDK for AI documentation](https://sap.github.io/ai-sdk/). Check them when you upgrade a sample, because some releases contain breaking changes (for example, the move to LangChain 1.x in the Python SDK 6.x line).
