## Project Structure
This project demonstrates best practices for accessing Embeddings models using the SAP GenAI Hub and storing them in SAP HANA Vector Store. It provides examples of different ways to create and store embeddings in the vector store using Python.

```
├── python
│   ├── LangChain_HANA_VectorStore_Embeddings.ipynb
│   ├── Native_HANA_VectorStore_Embeddings.ipynb
│   ├── README.md
│   └── requirements.txt
├── sample_files
│   ├── attention_is_all_you_need.pdf
│   ├── sap-hana-cloud.pdf
│   └── science-data-sample.csv
└── README.md
```

## Notebooks

| Notebook | Approach | Data Source | Vector Store Integration |
|----------|----------|-------------|--------------------------|
| `Native_HANA_VectorStore_Embeddings.ipynb` | Direct Python + SQL | CSV (science data) | Manual SQL with `hdbcli` |
| `LangChain_HANA_VectorStore_Embeddings.ipynb` | LangChain framework | PDFs (SAP docs) | [`langchain-hana`](https://github.com/SAP/langchain-integration-for-sap-hana-cloud) |

After creating embeddings, use the corresponding query notebooks in `vector-rag-query/python/` to build a RAG pipeline on top of the stored vectors.

## Clone the repository
``` sh
git clone https://github.com/SAP-samples/sap-btp-ai-best-practices/
cd sap-btp-ai-best-practices/best-practices/generative-ai/vector-rag-embedding/python
```

## Create a virtual environment
``` sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Install dependencies
``` sh
pip install -r requirements.txt
```

## Configure environment variables
* Copy the .env-example file to .env
  ``` sh
    cp .env-example .env
  ```
* Populate the .env file with the required values.
* The embedding model defaults to `text-embedding-3-small`. To use a different model available on SAP GenAI Hub, set the `EMBEDDING_MODEL` variable in your `.env` file.

## Run the Jupyter Notebook
``` sh
jupyter notebook
```

Open LangChain_HANA_VectorStore_Embeddings.ipynb or Native_HANA_VectorStore_Embeddings.ipynb notebook in your browser to explore based on your implementation preference.

## Usage Examples
The notebooks demonstrate various methods to use Embeddings model and SAP HANA Vector Store.

* Native Client Integrations (Native_HANA_VectorStore_Embeddings.ipynb):
  * Reads text from csv file and does preprocessing for metadata
  * Uses an OpenAI-compatible embedding model from GenAI Hub
  * Uses HANA connection context for database operations
* LangChain Implementations (LangChain_HANA_VectorStore_Embeddings.ipynb):
  * Reads multiple PDF files from a directory and extracts text
  * Uses an OpenAI-compatible embedding model from GenAI Hub
  * Demonstrates seamless integration between LangChain and SAP HANA Vector Store using the [`langchain-hana`](https://github.com/SAP/langchain-integration-for-sap-hana-cloud) package

Each section in the notebook provides a detailed example of how to set up and perform embeddings related operations.

## Recommended Method
The recommended method depends on the use case:

* If using native features of SAP HANA with SQL and Connection Context: Use the Native_HANA_VectorStore_Embeddings.ipynb.
* If using LangChain functions or modules: Use the LangChain_HANA_VectorStore_Embeddings.ipynb.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
