# SAP BTP AI Best Practices

<!--- Register repository https://api.reuse.software/register, then add REUSE badge:
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/REPO-NAME)](https://api.reuse.software/info/github.com/SAP-samples/REPO-NAME)
-->

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/sap-btp-ai-best-practices)](https://api.reuse.software/info/github.com/SAP-samples/sap-btp-ai-best-practices)

For comprehensive documentation related to this code: [SAP BTP AI Best Practices](https://btp-ai-bp.docs.sap/)

## Description

This repository provides a comprehensive collection of best practices and sample implementations for AI-powered applications on SAP Business Technology Platform (BTP). It serves as a practical guide for developers looking to implement enterprise-grade AI solutions using SAP's AI services and integration with generative AI models.

The repository is organized into four best-practice areas plus a set of end-to-end use cases:

- **Generative AI** – securely access generative AI models through the generative AI hub, content filtering, data masking for PII protection, prompt templating and the prompt registry, document grounding, vector-based retrieval-augmented generation (RAG) with SAP HANA Cloud Vector Engine, and Knowledge Graph RAG
- **Agentic AI** – code-based agents that implement the ReAct (Reasoning and Acting) pattern from scratch and with LangGraph, including multi-agent routing, an enterprise procurement workflow, token caching and checkpointed session state
- **Narrow AI** – classical machine learning with the SAP HANA Predictive Analysis Library (anomaly detection, classification, clustering, regression, time-series forecasting) and SAP-RPT, SAP's relational foundation model for predictions on tabular business data
- **AI Services** – SAP Document AI and SAP Translation Hub

Most best practices include code samples in multiple programming languages and frameworks (TypeScript, Python, Java, CAP) to support various technology stacks and development approaches.

## Requirements

- SAP Business Technology Platform account
- Access to the SAP AI Core service. The generative AI hub capabilities used in these samples (orchestration, foundation models, SAP-RPT) require the `extended` service plan. See [Prerequisites](./best-practices/generative-ai/Prerequisites.md) for setup steps.
- The samples use the SAP Cloud SDK for AI in their respective languages:
  - Python: [`sap-ai-sdk-gen`](https://pypi.org/project/sap-ai-sdk-gen/) ([source](https://github.com/SAP/ai-sdk-python))
  - JavaScript / TypeScript: [`@sap-ai-sdk/*` packages](https://github.com/SAP/ai-sdk-js)
  - Java: [SAP Cloud SDK for AI (Java)](https://github.com/SAP/ai-sdk-java)

## Download and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SAP-samples/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices
   ```

2. Follow the setup instructions in the README of each specific best practice directory.

   For example, to run the TypeScript example for accessing generative AI models:

   ```bash
   cd best-practices/generative-ai/access-to-generative-ai-models/typescript
   npm install
   cp .env.example .env
   # Edit the .env file with your service key
   npm start
   ```

## Repository Structure

```
sap-btp-ai-best-practices/
├── best-practices/               # Best practices implementations
│   ├── generative-ai/            # LLM access, orchestration modules, RAG, prompt management
│   ├── agentic-ai/               # Code-based agents (native ReAct, LangGraph)
│   ├── narrow-ai/                # Classical ML with SAP HANA PAL, and SAP-RPT
│   └── ai-services/              # SAP Document AI, SAP Translation Hub
└── use-cases/                    # End-to-end use case implementations
```

For a detailed overview of available best practices, see the [best practices directory](./best-practices/README.md). For the end-to-end applications, see [use-cases](./use-cases/README.md).

## Known Issues

No known issues.

## How to obtain support

[Create an issue](https://github.com/SAP-samples/sap-btp-ai-best-practices/issues) in this repository if you find a bug or have questions about the content.

For additional support, ask a question in the [SAP Community](https://community.sap.com/).

## Contributing

If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License

Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
