# Pharma Procurement Sales Order Agent Implementation Map

This document describes how the Pharma Procurement Sales Order Agent prototype is wired and where the main agent, tool, data, API, UI, and Joule integration modules are implemented.

## 1. Runtime Flow

```text
Joule or test UI
  -> FastAPI router
  -> Pharma Procurement Sales Order Agent LangGraph/LangChain agent
  -> tool-calling model
  -> MCP-compatible tool layer
  -> synthetic SAP-like JSON datasets
  -> formatted service representative answer
```

The current prototype is read-only by design. Transactional scenarios are answered as previews unless a productive SAP integration, authorization model, and customer approval are added later.

## 2. Entry Points

| Entry point | Module | Purpose |
|---|---|---|
| Test UI API | `api/app/routers/pharma_order.py` | Main test endpoint for `/api/pharma-order/ask`, health, and capabilities. |
| Joule REST adapter | `api/app/routers/joule.py` | Joule-facing endpoints under `/api/joule/pharma-order` and `/api/joule/pharma-order/stream`. |
| FastAPI app registration | `api/app/main.py` | Registers Pharma Procurement Sales Order Agent and Joule routers. |
| UI page | `ui/src/routes/pharma-order` | Test UI for direct backend validation outside Joule. |

## 3. Agent Layer

| Module | Responsibility |
|---|---|
| `api/app/agents/pharma_order/graph.py` | Runs the Pharma Procurement Sales Order Agent agent, configures model/tool execution, invokes MCP-compatible tools, and returns answer/trace metadata. |
| `api/app/agents/pharma_order/prompts.py` | Stores prompt variants and pharmaceutical order support behavior instructions. |
| `api/app/mcp/pharma_order_server.py` | Exposes Pharma Procurement Sales Order Agent tools through an MCP-compatible server surface. |

The agent is expected to resolve names and identifiers first, then call the smallest required set of tools, consolidate evidence, and produce a short service-representative-friendly answer.

## 4. Tool Layer

| Tool | Module | Purpose |
|---|---|---|
| `get_pricing_for_customer_material` | `api/app/tools/pharma_order/sap_mock_tools.py` | Simulates customer/material pricing using sales order simulation-like data. |
| `get_material_availability` | `api/app/tools/pharma_order/sap_mock_tools.py` | Checks ATP-like stock, batch, expiry, and pharma handling context. |
| `get_order_status` | `api/app/tools/pharma_order/sap_mock_tools.py` | Reads sales order header, item, partner, schedule, pricing, text, and logistics status context. |
| `lookup_customer_by_dea` | `api/app/tools/pharma_order/sap_mock_tools.py` | Resolves customer identity and compliance context from DEA/GTS-like data. |
| `lookup_customer_recent_orders` | `api/app/tools/pharma_order/sap_mock_tools.py` | Finds recent synthetic sales orders for a customer. |
| `lookup_batch_expiry` | `api/app/tools/pharma_order/sap_mock_tools.py` | Looks up batch, expiry, recall, quarantine, and serialization data. |
| `lookup_material_by_ndc` | `api/app/tools/pharma_order/sap_mock_tools.py` | Resolves NDC/product names to SAP material context. |
| `check_duplicate_po` | `api/app/tools/pharma_order/sap_mock_tools.py` | Checks synthetic order history for possible duplicate customer POs. |
| `list_blocked_orders` | `api/app/tools/pharma_order/sap_mock_tools.py` | Lists blocked orders and block reasons from synthetic sales order data. |
| `set_or_clear_order_block` | `api/app/tools/pharma_order/sap_mock_tools.py` | Preview-only block/unblock action. No SAP update is performed. |
| `get_invoice_pdf` | `api/app/tools/pharma_order/sap_mock_tools.py` | Returns invoice PDF metadata only, not binary PDF content. |

## 5. Data Layer

| Module / folder | Purpose |
|---|---|
| `api/app/tools/pharma_order/data_store.py` | Generic JSON-backed search and record extraction layer. |
| `api/app/data/API_SALES_ORDER_SIMULATION_SRV__pricing_simulations.json` | Synthetic pricing simulation records and pricing elements. |
| `api/app/data/API_SALES_ORDER_SRV__sales_orders_header_item_partner_status.json` | Synthetic sales order header, item, partner, pricing, schedule, text, and logistics data. |
| `api/app/data/API_MATERIAL_STOCK_SRV__material_stock_availability.json` | Synthetic ATP-like stock and pharma handling data. |
| `api/app/data/API_BATCH_SRV__batch_expiry_lot_status.json` | Synthetic batch/lot expiry and serialization data. |
| `api/app/data/API_BILLING_DOCUMENT_SRV__invoice_pdf_metadata.json` | Synthetic invoice PDF metadata. |
| `api/app/data/ZSD_EXTERNAL_INFO__customers_dea_gts_lookup.json` | Synthetic customer, DEA, GTS, compliance, and commercial context. |
| `api/app/data/ZAPI_DEL_LIST_PRICE_V4__materials_ndc_catalog.json` | Synthetic NDC/catalog/material mapping. |

## 6. Joule Integration

| File | Purpose |
|---|---|
| `template_joule.da.sapdas.yaml` | Digital assistant template wrapper. |
| `joule/capability.sapdas.yaml` | Capability metadata and system alias mapping to the BTP destination. |
| `joule/scenarios/sc_pharma_order.yaml` | Scenario description that routes pharmaceutical order support questions to the Pharma Procurement Sales Order Agent function. |
| `joule/functions/fn_pharma_order.yaml` | Uses `streamed-message` to call `/api/joule/pharma-order/stream` through the `TEMPLATE_API` system alias. |

Required BTP destination:

```text
Name: pharma-order-agent-api
Type: HTTP
URL: https://pharma-order-agent-api.cfapps.eu10-005.hana.ondemand.com
Proxy Type: Internet
Authentication: NoAuthentication
```

## 7. Read-Only vs Transactional Behavior

Read-only scenarios are suitable for the current prototype: pricing, availability, order status, customer/DEA lookup, material/NDC lookup, batch lookup, blocked-order listing, and invoice metadata lookup.

Transactional scenarios must remain preview-only in this prototype: create order, update partner function, change quantities, reject order lines, release delivery blocks, expedite orders, send emails, and generate actual PDF output.

## 8. Recommended Demo Path

1. Start with pricing: `What is the price for Northstar for Glycemor 10 mg?`
2. Show identifier resolution: `For Northstar, identify Glycemor 10 mg by NDC 90000-0100-30 and confirm the current net price.`
3. Show availability: `Is Glycemor 10 mg available for shipment this week?`
4. Show multi-tool orchestration: `For Northstar, identify Glycemor 10 mg by NDC 90000-0100-30, confirm the price, and tell me whether it can ship this week.`
5. Show order/invoice chain: `For the latest Northstar Glycemor order, explain status, shipment risk, and invoice PDF availability.`
