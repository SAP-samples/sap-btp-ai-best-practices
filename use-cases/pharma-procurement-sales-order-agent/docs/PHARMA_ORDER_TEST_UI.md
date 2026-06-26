# Pharma Procurement Sales Order Agent Test UI

This UI is a local and deployed test surface for the Pharma Procurement Sales Order Agent backend. It is not the target productive UX.

The target experience remains Joule / Pro-Code calling the FastAPI backend. The test UI is used to validate:

- Capability discovery through `/api/pharma-order/capabilities`.
- LangGraph orchestration through `/api/pharma-order/ask`.
- Joule streaming integration through `/api/joule/pharma-order/stream`.
- MCP-compatible tool execution.
- Readable trace summaries instead of raw SAP-like JSON payloads.
- Complex multi-step chains where the agent resolves names and identifiers before calling business tools.
- Preview-only handling for action-style requests where the prototype must not update SAP.

## Latest backend test result

Last checked: 2026-06-19.

Command:

```powershell
python -m unittest tests.unit.test_pharma_order_data tests.unit.test_pharma_order_tools tests.unit.test_pharma_order_router tests.unit.test_pharma_order_capabilities tests.unit.test_pharma_order_complex_scenarios
```

Result:

```text
Ran 10 tests
OK
```

## Capability-driven UI

The UI does not hardcode example buttons anymore. It loads capability cards from the backend. Each card includes supported scenarios, expected tools, source structures, and example questions.

## Trace policy

The UI should show compact evidence:

- Tool name.
- Source files or source structures.
- Matched records count.
- Assumptions.
- Short evidence preview.

Raw JSON can remain available in backend logs or debug responses, but it should not be the main user-facing trace.

## Recommended smoke test questions

Use the questions below in Joule first, then compare with the test UI when behavior looks suspicious.

### Pricing and product identification

Expected behavior: the agent should resolve customer, product, NDC/material identifiers, then return synthetic SAP-like pricing evidence.

```text
What is the price for Northstar for Glycemor 10 mg?
```

```text
For Northstar, identify Glycemor 10 mg by NDC 90000-0100-30 and confirm the current net price.
```

```text
Which SAP material number corresponds to NDC 90000-0100-30?
```

### Availability and shipment readiness

Expected behavior: the agent should use inventory and batch/availability evidence and answer whether shipment is feasible this week.

```text
Is Glycemor 10 mg available for shipment this week?
```

```text
For Northstar, identify Glycemor 10 mg by NDC 90000-0100-30, confirm the price, and tell me whether it can ship this week.
```

### Sales order status and fulfillment

Expected behavior: the agent should return order status, block status, shipment/invoice state, and available ETA/tracking evidence.

```text
What is the status of sales order 50214568?
```

```text
Is sales order 50214568 blocked, shipped, invoiced, and what is the ETA or tracking information?
```

### Document and invoice metadata

Expected behavior: the agent can find invoice metadata and document availability, but the prototype should not generate or return actual SAP documents.

```text
Can I get the invoice PDF for order 50214568?
```

### Multi-tool order explanation

Expected behavior: the agent should combine order lookup, item/product context, shipment risk, and invoice metadata. This scenario previously exposed recursion risk and should be retested after graph tuning changes.

```text
For the latest Northstar Glycemor order, explain status, shipment risk, and invoice PDF availability.
```

### Compliance and recent order lookup

Expected behavior: the agent should validate DEA/compliance details and retrieve recent relevant orders. This case is useful for detecting duplicate Joule invocations.

```text
Is DEA number BC1234567 valid for Northstar, and what are the latest relevant orders?
```

Known issue: Joule may call the backend more than once for this exact multi-intent query. The backend processes each individual request normally, but the duplicate visible answer is tracked in `PHARMA_ORDER_BACKLOG.md` as a Joule/DAS invocation issue.

### Preview-only action requests

Expected behavior: the request should route to the backend, be rewritten into a preview-only analysis if needed, and clearly state that no SAP update was performed.

```text
Release delivery blocks Z12 for all open ZOR orders for Northstar.
```

```text
Which Northstar orders have Z12 delivery blocks, and what would be required to release them?
```

```text
Preview the release of Z12 delivery blocks for open Northstar orders. Do not update SAP.
```

```text
Can you add or replace the freight forwarder partner function for sales order 50214568?
```

```text
Can you create a sales order from this PDF or email request?
```

## Backend log checks

When testing through Joule, watch backend logs with:

```powershell
cf logs pharma-order-agent-api
```

Useful signals:

- `Joule service representative stream start` means Joule reached the FastAPI streaming route.
- `Pharma Procurement Sales Order Agent query rewrite` means an action-style request was normalized to preview-only analysis.
- `event_type":"llm_usage` means the metering payload was emitted.
- Repeated `POST /api/joule/pharma-order/stream` entries with the same correlation id indicate duplicate invocation from the Joule/DAS layer, not a backend loop.
## Duplicate invocation regression checks

After deploying the idempotency guard, repeat the known duplicate-prone Joule prompts:

```text
Can I get the invoice PDF for order 50214568?
```

```text
For the latest Northstar Glycemor order, explain status, shipment risk, and invoice PDF availability.
```

```text
Release delivery blocks Z12 for all open ZOR orders for Northstar.
```

Expected behavior: Joule should render one visible Pharma Procurement Sales Order Agent answer. If Joule/DAS still repeats the HTTP call, the second backend request should be logged as `duplicate_suppressed` and should not call the LLM or MCP tools again.