# BTP AI Best Practices – Knowledge Graph RAG

This repository collects best practices and prompt engineering examples for working with **Knowledge Graph–powered Retrieval-Augmented Generation (KG-RAG)** on **SAP BTP**, focusing on use cases involving SAP HANA Cloud and AI Core.

> 📚 Related SAP AI best practice guides:
> - [KG RAG – Overall and KG Creation](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20RAG%20-%20Knowledge%20Graph%20-%20Overall%20and%20KG%20creation.aspx)
> - [KG RAG – Query Pipeline](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20RAG%20-%20Knowledge%20Graph%20-%20Query%20pipeline.aspx)

## Overview

**Knowledge Graph–powered RAG** combines structured data representation with generative AI. This approach ensures:
- Better context grounding
- Semantic consistency
- Explainability of AI responses

This repository focuses on two key areas:
- 🧱 **KG Creation and Grounding** — extracting RDF triplets from domain documents using prompt templates with HANA Cloud and AI CORE
- 🔎 **Query Pipeline** — retrieving grounded facts from a KG and injecting them into prompts for LLMs

## Data Samples

You can use PDF documents as domain data sources for knowledge graph creation. Here are a couple of examples:
> - [EBMUD 2312 - Four 1500-HP Main Air Compressor Motors (PDF)](https://www.ebmud.com/application/files/3416/7771/2399/2312_-_Four_1500-HP_Main_Air_Compressor_Motors_FINAL.pdf)
> - [RS Components A700000006779737 (PDF)](https://docs.rs-online.com/d853/A700000006779737.pdf)


Feel free to use your own PDF documents in the repository to experiment with KG extraction and grounding.

## Language Support

Currently, the examples are implemented in:

- [Python](./python/): Main implementation with LangChain, RDFLib, and prompt templating
  - Includes: 
    - Prompt templates for triplet extraction
    - Examples of selective attribute filtering
    - End-to-end demonstration of RDF graph construction and visualization in Jupyter

> 📂 See `prompts.py` for configurable templates used in Knowledge Graph construction  
> 📓 See `KG-RDF-creation-grounding-visualisation-BP07-BP08-BestPractice.ipynb` for the executable demo

## Target Audience

- SAP developers building intelligent data apps
- Data scientists implementing AI Core pipelines
- Architects designing graph-powered RAG solutions

