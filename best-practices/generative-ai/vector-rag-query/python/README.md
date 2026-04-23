## Project Structure
This project demonstrates best practices for performing Retrieval Augmented Generation (RAG) using LLMs available on SAP GenAI Hub and SAP HANA Vector Store. It provides examples of different ways to implement RAG using Python.

```
├── python
│   ├── LangChain_RAG_with_History.ipynb
│   ├── Native_RAG.ipynb
│   ├── README.md
│   └── requirements.txt
└── README.md
```

## Prerequisites

Source documents must already be chunked and stored along with their embedding vectors in SAP HANA Cloud. Run the corresponding embedding notebook first:

| Query Notebook | Requires Embedding Notebook | Table |
|---|---|---|
| `Native_RAG.ipynb` | `Native_HANA_VectorStore_Embeddings.ipynb` | `SCIENCE_DATA` |
| `LangChain_RAG_with_History.ipynb` | `LangChain_HANA_VectorStore_Embeddings.ipynb` | `SAP_HELP_PUBLIC` |

See the [embedding project](../vector-rag-embedding/python/) for details.

## Clone the repository

```sh
git clone https://github.com/SAP-samples/sap-btp-ai-best-practices/
cd sap-btp-ai-best-practices/best-practices/generative-ai/vector-rag-query/python
```

## Create a virtual environment

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Install dependencies

```sh
pip install -r requirements.txt
```

## Configure environment variables

* Copy the .env-example file to .env
  ```sh
    cp .env-example .env
  ```
* Populate the .env file with the required values.
* The embedding model defaults to `text-embedding-3-small`. To use a different model available on SAP GenAI Hub, set the `EMBEDDING_MODEL` variable in your `.env` file.

## Run the Jupyter Notebook

```sh
jupyter notebook
```

Open LangChain_RAG_with_History.ipynb or Native_RAG.ipynb notebook in your browser to explore based on your implementation preference.

## Usage Examples

The notebooks demonstrate various methods to build a RAG pipeline with SAP HANA Vector Store.

- LangChain Implementations (LangChain_RAG_with_History.ipynb):
  - Creates a LangChain object for operations on HANA Vector Store using the [`langchain-hana`](https://github.com/SAP/langchain-integration-for-sap-hana-cloud) package
  - Retrieves top semantically matching documents from vector store
  - Optionally processes history messages for follow-up queries
  - Uses OpenAI embedding as well as completion models from GenAI Hub
  - Demonstrates seamless integration between LangChain and SAP HANA Vector Store
- Native Client Integrations (Native_RAG.ipynb):
  - Initializes connection context and cursor for all operations on HANA Vector Store
  - Uses SQL queries to retrieve top semantically matching records based on similarity measures
  - Uses OpenAI embedding as well as completion models from GenAI Hub

Each section in the notebook provides a detailed example of how to set up and perform RAG operations.

## Recommended Method

The recommended method depends on the use case:

- If using native features of SAP HANA with SQL and Connection Context: Use the Native_RAG.ipynb.
- If using LangChain functions or modules: Use the LangChain_RAG_with_History.ipynb.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
