
## Project Structure
This project demonstrates best practices for using Document Grounding service. It provides examples of different ways to connect various data respositories using differet methods.

```
├── python
│   ├── Document_Grounding_S3_API.ipynb
│   ├── Document_Grounding_S3_SDK.ipynb
│   ├── Document_Grounding_Vector_API.ipynb
│   └── README.md
├── sample_files
│   ├── AI Best Practices.pdf
│   ├── deep seek technical paper.pdf
│   ├── Document AI.pdf
│   ├── multimodal llm paper.pdf
│   ├── NeurIPS 2025 CNN Paper.pdf
│   └── Paper ConTextTab.pdf
├── README.md
└── requirements.txt
```

The Document Grounding service provides out-of-the-box support for different document repositories. The notebooks demonstrate various methods for using the service with corresponding source data repositories. 
* AWS S3
* SAP WorkZone
* SAP Document Management Service
* SharePoint
* SFTP file server

## Pre-requisites
Create a generic secrets key collection on AI Core using the secrets of your source document repository.

## Clone the repository
``` sh
git clone https://github.com/SAP-samples/sap-btp-ai-best-practices/
cd sap-btp-ai-best-practices/best-practices/generative-ai/document-grounding/python
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

Open the notebook in your browser to explore the steps based on your source repository and preferred implementation approach.

## Usage Examples
* Connect to S3
    * Document_Grounding_S3_SDK.ipynb: SAP Cloud AI SDK to implement grounding service
    * Document_Grounding_S3_API.ipynb: Data Management APIs to implement grounding service
* Connect to SharePoint
* Connect to SAP WorkZone
* Connect to SAP Document Management System
* Connect to the SFTP file server
* Direct Vector collection
    * Document_Grounding_Vector_API.ipynb: Create chunks and metadata and push them to the grounding service using the Vector API


## Sample data sources:
* [NeurIPS receptive field paper](https://proceedings.neurips.cc/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf)
* [ConTextTab paper](https://arxiv.org/html/2506.10707v1)
* [Deep Seek paper](https://arxiv.org/pdf/2412.19437)
* [Multi Modal LLMs paper](https://arxiv.org/pdf/2508.21801)

