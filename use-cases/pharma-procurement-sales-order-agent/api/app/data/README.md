# Pharma Procurement Sales Order Agent Synthetic Data Fixtures

These files provide SAP-like synthetic data for the Pharma Procurement Sales Order Agent technical prototype.
They are intentionally customer-neutral and do not contain real customer data.

## Files

| File | Purpose |
|---|---|
| `ZSD_EXTERNAL_INFO__customers_dea_gts_lookup.json` | Customer master-like data, DEA/GTS lookup data, and partner roles. |
| `ZAPI_DEL_LIST_PRICE_V4__materials_ndc_catalog.json` | Pharmaceutical material catalog data, NDC aliases, storage, serialization, and compliance attributes. |
| `API_SALES_ORDER_SIMULATION_SRV__pricing_simulations.json` | Mock output for `API_SALES_ORDER_SIMULATION_SRV`. |
| `API_MATERIAL_STOCK_SRV__material_stock_availability.json` | Mock output for `API_MATERIAL_STOCK_SRV`. |
| `API_BATCH_SRV__batch_expiry_lot_status.json` | Mock output for `API_BATCH_SRV`. |
| `API_SALES_ORDER_SRV__sales_orders_header_item_partner_status.json` | Mock nested shape for `API_SALES_ORDER_SRV` entities such as `A_SalesOrder`, `A_SalesOrderItem`, partners, pricing elements, schedule lines, texts, and billing plan. |
| `API_BILLING_DOCUMENT_SRV__invoice_pdf_metadata.json` | Mock output for `API_BILLING_DOCUMENT_SRV`. |
| `PHARMA_ORDER_SCENARIOS__sample_questions_tool_mapping.json` | Representative questions mapped to intended tools and fixture records. |

## Design Notes

- The data is built around the first anchor question: "What is the price for Northstar for Glycemor 10 mg?"
- `Glycemor 10 mg` is treated as a demo alias for material `GLYCE10MG`.
- Pharmaceutical-specific fields include NDC, GTIN, batch/lot control, serialization, DEA relevance, storage conditions, expiry, temperature range, recall status, and cold-chain flag.
- The files are mocks. Real implementation should replace them with SAP OData responses or adapters once metadata and access are available.
