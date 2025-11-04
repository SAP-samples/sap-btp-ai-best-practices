# SAP BTP AI Best Practices

<!--- Register repository https://api.reuse.software/register, then add REUSE badge:
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/REPO-NAME)](https://api.reuse.software/info/github.com/SAP-samples/REPO-NAME)
-->

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/sap-btp-ai-best-practices)](https://api.reuse.software/info/github.com/SAP-samples/sap-btp-ai-best-practices)

For comprehensive documentation related to this code: [SAP BTP AI Best Practices](https://btp-ai-bp.docs.sap/)

## Description

This repository provides a comprehensive collection of best practices and sample implementations for AI-powered applications on SAP Business Technology Platform (BTP). It serves as a practical guide for developers looking to implement enterprise-grade AI solutions using SAP's AI services and integration with generative AI models.

The examples provided demonstrate how to:

- Securely access generative AI models
- Implement content filtering for AI-generated outputs
- Apply data masking for PII protection
- Create effective prompt templates
- Build retrieval-augmented generation (RAG) systems

Each best practice includes code samples in multiple programming languages (TypeScript, Python, Java, CAP) to support various technology stacks and development approaches.

## Requirements

- SAP Business Technology Platform account
- Access to SAP AI Core service

## Download and Installation

1. Clone the repository:

   ```bash
   git clone https://github.tools.sap/btp-ai-best-practices/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices
   ```

2. Follow the setup instructions in the README of each specific best practice directory.

   For example, to run the TypeScript example for accessing generative AI models:

   ```bash
   cd best-practices/access-to-generative-ai-models/typescript
   npm install
   cp .env.example .env
   # Edit the .env file with your service key
   npm start
   ```

## Repository Structure

```
sap-btp-ai-best-practices/
├── best-practices/               # Best practices implementations
└── use-cases/                    # End-to-end use case implementations
```

For a detailed overview of available best practices, see the best practices directory.

## Known Issues

No known issues.

## How to obtain support

[Create an issue](https://github.com/SAP-samples/sap-btp-ai-best-practices/issues) in this repository if you find a bug or have questions about the content.

For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing

If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License

Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.

## Devtoberfest Scavenger Hunt

Congrats! You have found the location of another clue in the [Devtoberfest](https://community.sap.com/t5/devtoberfest-blog-posts/devtoberfest-2025-welcome/ba-p/14182817) Scavenger Hunt. You will find the instructions of how to derive the 2 hexadecimal digits using the instructions in the [Scavenger Hunt Blog Post](https://community.sap.com/t5/devtoberfest-blog-posts/introducing-the-devtoberfest-scavenger-hunt/ba-p/14183972).
