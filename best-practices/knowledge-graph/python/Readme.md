# Knowledge Graph RAG – Python Implementation

This folder contains the **Python-based implementation** of best practices for building and querying **Knowledge Graphs for RAG** (Retrieval-Augmented Generation) on SAP BTP.

These examples demonstrate how to extract RDF triplets from raw text using prompt engineering and visualize structured knowledge for grounding LLM responses.

## Project Structure
```
knowledge-graph/
└── python/
├── prompts.py # Prompt templates for KG creation and selective filtering
├── KG-RDF-creation-grounding-visualisation-BP07-BP08-BestPractice.ipynb # End-to-end example
└── README.md # Project documentation (this file)
```


## Included Content

- ✅ **Prompt Engineering Templates**  
  `prompts.py` defines reusable LangChain prompt templates for RDF triplet extraction — you can update it for your domain.

- 📊 **Notebook Demo**  
  `KG-RDF-creation-grounding-visualisation-BP07-BP08-BestPractice.ipynb` demonstrates:
  - Extracting entities and relationships
  - Filtering by allowed attributes
  - Creating RDF graphs from plain text
  - Upload KG to HANA DB
  - Using som attributes for searching in a new document
  - Visualizing the result using Python tooling

## Use Case

This implementation supports the BTP AI Best Practices themes:
- **KG RAG – Overall and KG Creation**
- **KG RAG – Query Pipeline** 

> For full pipeline integration and execution, see the parent folder's `README.md`.

## Audience

This example is built for:
- SAP HANA Cloud KG Engine developers
- AI Core pipeline designers
- Anyone building AI-grounded applications using structured semantic data

---

For questions or contributions, open an issue in the main repository.

