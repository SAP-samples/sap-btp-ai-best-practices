# Pharma Procurement Sales Order Agent Local Runbook

## Backend

Use port `8056` for this prototype so it can run next to other local POCs.

```bash
cd api
uvicorn app.main:app --host 0.0.0.0 --port 8056 --reload
```

Direct backend endpoint:

```bash
curl -X POST http://localhost:8056/api/pharma-order/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <local-api-key>" \
  -d '{"question":"What is the price for Northstar for Glycemor 10 mg?","include_trace":true}'
```

Joule-compatible endpoint:

```bash
curl "http://localhost:8056/api/joule/pharma-order?question=What%20is%20the%20price%20for%20Northstar%20for%20Glycemor%2010%20mg%3F"
```

## UI

Use port `5178` for the UI training environment.

```bash
cd ui
npm run dev -- --host 0.0.0.0 --port 5178
```

## Backend tests

The Pharma Procurement Sales Order Agent tests are located in `tests/unit` and avoid live LLM calls by testing the data/tool layer directly or patching the agent call.

```bash
python -m unittest tests/unit/test_pharma_order_data.py tests/unit/test_pharma_order_tools.py tests/unit/test_pharma_order_router.py
```
