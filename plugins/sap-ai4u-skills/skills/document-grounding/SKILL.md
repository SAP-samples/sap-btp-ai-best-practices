---
name: document-grounding
description: Implement document-grounded retrieval and question answering with SAP Document Grounding Service through Gen AI Hub. Use when tasks involve connecting repositories (S3, SharePoint, WorkZone, Document Management, SFTP), creating grounding pipelines/collections, retrieving chunks, and answering with grounded context.
---

# Document Grounding with SAP Gen AI Hub

Use this skill to create scripts that retrieve grounded context from enterprise documents.

## Set Required Environment Variables

```bash
AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_BASE_URL=""
AICORE_RESOURCE_GROUP=""
```

Set API endpoint explicitly in code:

```python
AI_API_URL = ""
```

## Get OAuth Token

```python
import os
import requests

client_id = os.getenv("AICORE_CLIENT_ID")
client_secret = os.getenv("AICORE_CLIENT_SECRET")
auth_url = os.getenv("AICORE_AUTH_URL")

response = requests.post(
    auth_url,
    data={"grant_type": "client_credentials"},
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    auth=(client_id, client_secret),
)
access_token = response.json().get("access_token")
```

## Pattern A: S3 Pipeline + Retrieval API

Create pipeline:

```python
url = f"{AI_API_URL}/v2/lm/document-grounding/pipelines"
payload = {
    "type": "S3",
    "configuration": {
        "destination": "your-generic-secret-key-name",
        "s3": {"includePaths": ["/new_papers/"]},
    },
}
headers = {
    "Authorization": f"Bearer {access_token}",
    "AI-Resource-Group": "default",
    "Content-Type": "application/json",
}
print(requests.post(url, headers=headers, json=payload).text)
```

Search grounded chunks:

```python
search_url = f"{AI_API_URL}/v2/lm/document-grounding/retrieval/search"
payload = {
    "query": "What is efficient receptive field?",
    "filters": [{
        "id": "vector",
        "searchConfiguration": {"maxChunkCount": 2},
        "dataRepositories": ["<repository_id>"],
        "dataRepositoryType": "vector",
    }],
}
response = requests.post(search_url, headers=headers, json=payload).json()
```

## Pattern B: SDK Orchestration Grounding

```python
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.document_grounding import (
    GroundingModule, GroundingType, DataRepositoryType,
    GroundingFilterSearch, DocumentGrounding, DocumentGroundingFilter,
)
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService

prompt = Template(messages=[
    SystemMessage("Use the given Context for the answer."),
    UserMessage("Context: {{ ?grounding_response }}\nQuestion: {{ ?query }}"),
])

filters = [DocumentGroundingFilter(
    id="vector",
    data_repositories=["<repository_id>"],
    search_config=GroundingFilterSearch(max_chunk_count=3),
    data_repository_type=DataRepositoryType.VECTOR.value,
)]

config = OrchestrationConfig(
    template=prompt,
    llm=LLM(name="gpt-4o", parameters={"temperature": 0.0}),
    grounding=GroundingModule(
        type=GroundingType.DOCUMENT_GROUNDING_SERVICE.value,
        config=DocumentGrounding(input_params=["query"], output_param="grounding_response", filters=filters),
    ),
)

service = OrchestrationService(api_url="<your_orchestration_deployment_url>")
result = service.run(config=config, template_values=[TemplateValue("query", "what is effective receptive field?")])
print(result.orchestration_result.choices[0].message.content)
```

## Pattern C: Vector Collection API

Use this when ingesting pre-chunked documents directly:

- Create collection: `POST /v2/lm/document-grounding/vector/collections`.
- Upload docs JSONL: `POST /vector/collections/{collection_id}/documents`.
- Run retrieval search with `dataRepositories: [collection_id]`.

## Apply Best Practices

- Keep repository and collection IDs externalized, not hardcoded.
- Limit `maxChunkCount` (start with `2` to `5`) for concise prompts.
- Send stable metadata keys (for example `id`, `url`, chunk `index`) for traceability.
- Preserve a strict instruction: if answer is unknown from context, return unknown.
- Separate retrieval step and generation step for easier debugging.

## Validate with Expected Outputs

Healthy run indicators:

- `"Access token obtained successfully."`
- Pipeline creation returns HTTP `201` and a `pipelineId`.
- Retrieval search returns HTTP `200` and chunk text in response payload.
- Final model answer discusses grounded topic (for sample: receptive field).

## Related Skills

- `vector-rag-embedding` / `vector-rag-query` — use instead when managing your own HANA vector table rather than the managed grounding service.
- `access-to-generative-ai-models` — orchestration service call patterns.
- `sap-btp-ai` — routing and shared environment conventions.
