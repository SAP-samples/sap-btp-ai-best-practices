## Project Structure
This project demonstrates best practices for accessing Embeddings models using the SAP GenAI Hub and storing them in SAP HANA Vector Store. It provides examples of different ways to create and store embeddings in the vector store using Python.

```
├── python
│   ├── LangChain_HANA_VectorStore_Embeddings.ipynb
│   ├── Native_HANA_VectorStore_Embeddings.ipynb
│   ├── README.md
│   └── requirements.txt
├── sample_files
│   ├── sap-business-ai.pdf
│   ├── sap-hana-cloud.pdf
│   └── science-data-sample.csv
└── README.md
```

## Clone the repository
``` sh
git clone https://github.com/SAP-samples/sap-btp-ai-best-practices/
cd https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/vector-rag-embedding/python
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

## Run the Jupyter Notebook
``` sh
jupyter notebook
```

Open LangChain_HANA_VectorStore_Embeddings.ipynb or Native_HANA_VectorStore_Embeddings.ipynb notebook in your browser to explore based on your implementation preference.

## Usage Examples
The notebooks demonstrate various methods to use Embeddings model and SAP HANA Vector Store.

* Native Client Integrations (Native_HANA_VectorStore_Embeddings.ipynb):
  * Reads text from csv file and does preprocessing for metadata
  * Uses OpenAI embedding model from GenAI Hub
  * Uses HANA connection context for database operations
* LangChain Implementations (LangChain_HANA_VectorStore_Embeddings.ipynb):
  * Reads multiple PDF files a directory and extracts texts from those.
  * Uses OpenAI embedding model from GenAI Hub
  * Demonstrates seemless integration between LangChain and SAP HANA Vector Store for all the operations

Each section in the notebook provides a detailed example of how to set up and perform embeddings related operations.

## Recommended Method
The recommended method depends on the use case:

* If using native features of SAP HANA with SQL and Connection Context: Use the Native_HANA_VectorStore_Embeddings.ipynb.
* If using LangChain functions or modules: Use the LangChain_HANA_VectorStore_Embeddings.ipynb.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.