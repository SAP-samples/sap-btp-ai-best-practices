## API Reference

This FastAPI service exposes endpoints for health checks, chat with grounded context, knowledge graph (KG) access, and analysis workflows (cost, risk, parts, comparison).

### Base URL

- Local: `http://localhost:8000`
- All documented paths below are relative to the base URL.

### Authentication

- Most endpoints require an API key via the `X-API-Key` header.
- Configure the server with environment variable `API_KEY`.

Example header:

```bash
export API_KEY="your-secret-key"
```

Include in requests:

```bash
-H "X-API-Key: $API_KEY"
```

Note: `GET /api/health` and the `/api/grounding/*` endpoints are unauthenticated by default.

---

### Health

- **GET** `/api/health`
  - Returns service health info.

Example:

```bash
curl -s http://localhost:8000/api/health
```

---

### Example Router

- **GET** `/api/example/service1/`
  - Simple test route (requires API key).

Example:

```bash
curl -s \
  -H "X-API-Key: $API_KEY" \
  http://localhost:8000/api/example/service1/
```

---

### Chat

- **POST** `/api/v1/chat/ask`
  - Asks a question about one or two suppliers, using provided KG JSON or packaged supplier IDs.
  - Request body:
    - `question` (string) — required
    - `supplier1` and `supplier2` (object): `{ id?, name?, kg_json? }`
    - `model` (string, optional)
  - Response body: `{ answer_markdown: str, sources: [{...}] }`

Example:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "Which supplier has lower overall cost for component systems?",
    "supplier1": {"id": "supplier1"},
    "supplier2": {"id": "supplier2"}
  }' \
  http://localhost:8000/api/v1/chat/ask
```

---

### Knowledge Graph (KG)

- **GET** `/api/v1/kg/static/list`
  - Lists packaged supplier KGs with metadata (requires API key).
  - Source of truth: files under `template/api/app/data/kg/suppliers`. You can override the directory with `SUPPLIER_KG_DIR`.
  - Response shape:
    - `{ "suppliers": [{ "id": string, "name": string, "filename": string }] }`

  Example:
  ```bash
  curl -s -H "X-API-Key: $API_KEY" \
    http://localhost:8000/api/v1/kg/static/list | jq
  ```
  Example response:
  ```json
  {
    "suppliers": [
      { "id": "supplier1", "name": "SupplierA", "filename": "supplier1.json" },
      { "id": "supplier2", "name": "SupplierB", "filename": "supplier2.json" }
    ]
  }
  ```

- **GET** `/api/v1/kg/static/get/{supplier_id}`
  - Returns KG JSON for a supplier (requires API key). `supplier_id` examples: `supplier1`, `supplier2`.
  - Reads `{filename}` from the list endpoint or `{supplier_id}.json` fallback.

  Example:
  ```bash
  curl -s -H "X-API-Key: $API_KEY" \
    http://localhost:8000/api/v1/kg/static/get/supplier1 | jq '.nodes | length, .relationships | length'
  ```

- **POST** `/api/v1/kg/create`
  - Creates per-file knowledge graphs (image-mode) for two suppliers from uploaded PDFs, saves all outputs under `app/data/kg/suppliers/<slug-timestamp>/`, and returns identifiers/paths to use for unification.
  - Authentication: requires `X-API-Key`.
  - Content-Type: `multipart/form-data`
  - Form fields:
    - `supplier1_name` (string, required)
    - `supplier2_name` (string, required)
    - `supplier1_files` (file[], optional): One or more PDFs; repeat the field for multiple files
    - `supplier2_files` (file[], optional): One or more PDFs; repeat the field for multiple files
  - Response ( abridged ):
    - `{ supplier1: { id, name, root_dir, pdfs_dir, kgs_dir, pdfs: [..], kg_outputs: [{ json, graphml }] }, supplier2: { ... } }`

  Example:
  ```bash
  curl -X POST 'http://localhost:8000/api/v1/kg/create' \
    -H 'X-API-Key: $API_KEY' \
    -F 'supplier1_name=SupplierA' \
    -F 'supplier2_name=SupplierB' \
    -F 'supplier1_files=@"/absolute/path/supplier 1/file1.pdf"' \
    -F 'supplier1_files=@"/absolute/path/supplier 1/file2.pdf"' \
    -F 'supplier2_files=@"/absolute/path/supplier 2/fileA.pdf"'
  ```

  Sample response (truncated):
  ```json
  {
    "supplier1": {
      "id": "supplier_a-20250101_120000",
      "root_dir": "template/api/app/data/kg/suppliers/supplier_a-20250101_120000",
      "pdfs_dir": ".../pdfs",
      "kgs_dir": ".../kgs",
      "pdfs": [".../pdfs/file1.pdf", ".../pdfs/file2.pdf"],
      "kg_outputs": [
        { "json": ".../kgs/file1_kg.json", "graphml": ".../kgs/file1_kg.graphml.gz" },
        { "json": ".../kgs/file2_kg.json", "graphml": ".../kgs/file2_kg.graphml.gz" }
      ]
    },
    "supplier2": { "id": "SupplierB-20250101_120000", "...": "..." }
  }
  ```

- **POST** `/api/v1/kg/unify`
  - Unifies all per-file KGs for each of the two suppliers identified by the IDs returned from `/api/v1/kg/create`. Writes timestamped unified files under each supplier's `kgs` directory and returns their paths.
  - Authentication: requires `X-API-Key`.
  - Content-Type: `application/json`
  - Body:
    - `{ "supplier1_id": string, "supplier2_id": string }`
  - Response ( abridged ):
    - `{ supplier1: { supplier_id, unified_json, unified_graphml, statistics }, supplier2: { ... } }`

  Example:
  ```bash
  curl -X POST 'http://localhost:8000/api/v1/kg/unify' \
    -H 'X-API-Key: $API_KEY' \
    -H 'Content-Type: application/json' \
    -d '{
      "supplier1_id": "supplier_a-20250101_120000",
      "supplier2_id": "SupplierB-20250101_120000"
    }'
  ```

Examples:

```bash
curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/kg/static/list

curl -s -H "X-API-Key: $API_KEY" \
  http://localhost:8000/api/v1/kg/static/get/supplier1
```

Notes:
- Use `@"/absolute/path with spaces/file.pdf"` in curl for file uploads with spaces.
- PDFs and outputs are saved under `template/api/app/data/kg/suppliers/<supplier-id>/`.
- Long-running extractions may take several minutes depending on document count/complexity.

---

### Analysis Suite

These endpoints compute and/or reuse cached analyses. Provide either `supplier_id` or `kg_path`. Optional `model` can select an LLM profile.

- **POST** `/api/v1/analyze/cost`
  - Body: `{ supplier_id?: str, kg_path?: str, model?: str }`
  - Computes cost-related metrics for a supplier from a KG JSON file.
  - Provide either `supplier_id` (packaged/static KG id) or a direct `kg_path`.
  - Caching: results written to `app/data/cache/analyses/cost_<supplier>_<mtime>.json`.

  Example (by supplier id):
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier_id": "supplier1"}' \
    http://localhost:8000/api/v1/analyze/cost | jq
  ```
  Example (by explicit KG path):
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"kg_path": "template/api/app/data/kg/suppliers/supplier1.json"}' \
    http://localhost:8000/api/v1/analyze/cost | jq
  ```

- **POST** `/api/v1/analyze/risk`
  - Body: `{ supplier_id?: str, kg_path?: str, model?: str }`
  - Computes risk-related metrics for a supplier from a KG JSON file.
  - Caching similar to cost.

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier_id": "supplier2"}' \
    http://localhost:8000/api/v1/analyze/risk | jq
  ```

- **POST** `/api/v1/analyze/parts`
  - Body: `{ supplier_id?: str, kg_path?: str, model?: str, core_part_categories?: [str] }`
  - Breaks down parts and categories; cache varies by categories (hash of list).
  - Set `core_part_categories` like `["ELECTRONICS", "MECHANICAL"]`.

  Example with categories:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier_id": "supplier1", "core_part_categories": ["ELECTRONICS", "MECHANICAL"]}' \
    http://localhost:8000/api/v1/analyze/parts | jq
  ```

- **POST** `/api/v1/analyze/homepage`
  - Body: `{ supplier_id?: str, kg_path?: str, model?: str }`
  - Generates high-level homepage-style summary.

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier_id": "supplier1"}' \
    http://localhost:8000/api/v1/analyze/homepage | jq
  ```

- **POST** `/api/v1/analyze/tqdcs`
  - Body: `{ supplier_id?: str, kg_path?: str, model?: str, prior_analyses?: object, weights?: { [k: string]: number } }`
  - Computes detailed analysis using optional weights and prior analyses.
  - If you’ve already called cost/parts/risk, pass them via `prior_analyses` to avoid recomputation.
  - `weights` can adjust T/Q/D/C/S contributions (e.g., `{ "T": 1.0, "C": 1.5 }`).

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier_id": "supplier1", "weights": {"C": 1.5, "D": 1.2}}' \
    http://localhost:8000/api/v1/analyze/tqdcs | jq
  ```

- **POST** `/api/v1/analyze/compare`
  - Body: `{ supplier1_name: str, supplier2_name: str, supplier1_analyses?: object, supplier2_analyses?: object, tqdcs_weights?: object, generate_metrics?: bool, generate_strengths_weaknesses?: bool, generate_recommendation_and_split?: bool, model?: str }`
  - Compares two suppliers; accepts precomputed analyses to avoid recomputation.
  - If you pass `supplier1_analyses` and `supplier2_analyses`, they should mirror outputs from `supplier_full`.

  Minimal example using names only:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier1_name": "SupplierA", "supplier2_name": "SupplierB"}' \
    http://localhost:8000/api/v1/analyze/compare | jq
  ```

- **POST** `/api/v1/analyze/supplier_full`
  - Body: `{ supplier_id?: str, kg_path?: str, model?: str, core_part_categories?: [str], force_refresh?: bool }`
  - Runs cost, risk, parts, homepage, then TQDCS for one supplier and returns a consolidated object.
  - Uses cache when available unless `force_refresh` is true.

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier_id": "supplier2", "core_part_categories": ["ELECTRONICS"]}' \
    http://localhost:8000/api/v1/analyze/supplier_full | jq
  ```

- **POST** `/api/v1/analyze/complete`
  - Body: `{ supplier1_id: str, supplier2_id: str, model?: str, comparator_model?: str, core_part_categories?: [str], tqdcs_weights?: object, force_refresh?: bool }`
  - Full run for two suppliers plus comparison; executes each supplier in parallel and then compares.

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier1_id": "supplier1", "supplier2_id": "supplier2"}' \
    http://localhost:8000/api/v1/analyze/complete | jq
  ```

- **POST** `/api/v1/analyze/ensure`
  - Body: `{ supplier1_id: str, supplier2_id: str, model?: str, comparator_model?: str, core_part_categories?: [str], tqdcs_weights?: object, force_refresh?: bool, generate_metrics?: bool, generate_strengths_weaknesses?: bool, generate_recommendation_and_split?: bool }`
  - Ensures all required analyses exist (reuses cache when possible) and returns comparison. Useful for idempotent pipelines.

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier1_id": "supplier1", "supplier2_id": "supplier2", "generate_recommendation_and_split": true}' \
    http://localhost:8000/api/v1/analyze/ensure | jq
  ```

- **POST** `/api/v1/analyze/cache_status`
  - Body: `{ supplier1_id: str, supplier2_id: str, core_part_categories?: [str] }`
  - Reports what is already cached for each supplier and if a comparison exists.
  - Helpful to decide which analyses to recompute.

  Example:
  ```bash
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"supplier1_id": "supplier1", "supplier2_id": "supplier2"}' \
    http://localhost:8000/api/v1/analyze/cache_status | jq
  ```

Examples:

```bash
# Cost
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"supplier_id": "supplier1"}' \
  http://localhost:8000/api/v1/analyze/cost

# Parts with categories
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"supplier_id": "supplier2", "core_part_categories": ["ELECTRONICS", "MECHANICAL"]}' \
  http://localhost:8000/api/v1/analyze/parts

# Ensure + compare
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"supplier1_id": "supplier1", "supplier2_id": "supplier2"}' \
  http://localhost:8000/api/v1/analyze/ensure
```

---

### Grounding

These endpoints interact with a document-grounding service and are unauthenticated by default.

- **GET** `/api/grounding/pipelines`
  - Lists available grounding pipelines.

- **GET** `/api/grounding/collections`
  - Lists available vector collections.

- **GET** `/api/grounding/collections/{collection_id}/documents`
  - Lists documents within a collection.

- **GET** `/api/grounding/mapped-collections`
  - Maps pipelines to collections with resolved pipeline paths.

- **GET** `/api/grounding/collections/{collection_id}/files`
  - Returns file info (title, timestamp, document id) for a collection.

- **POST** `/api/grounding/completion`
  - Body: `{ grounding_request: str, collection_id: "*"|string, custom_prompt?: str, max_chunk_count?: number }`
  - Executes a grounding completion and returns context and LLM answer.

Examples:

```bash
curl -s http://localhost:8000/api/grounding/pipelines

curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "grounding_request": "Summarize component system docs",
    "collection_id": "*",
    "max_chunk_count": 25
  }' \
  http://localhost:8000/api/grounding/completion
```

---

### Notes

- Default supplier IDs: `supplier1`, `supplier2` (see `app/data/kg/suppliers`).
- Some analysis endpoints cache results under `app/data/cache/analyses` to speed up repeat requests.

