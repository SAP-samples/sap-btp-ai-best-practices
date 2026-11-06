# Anonymized Application Test Questions

Use these questions in the manual email form after seeding HANA from `data/anonymized`.

```bash
cd api
../.venv/bin/python scripts/setup_hana_document_request_data.py
```

## Sales Orders

| Scenario | Sales order | Country | Status | Available documents |
| --- | --- | --- | --- | --- |
| Scenario 1 and 2 | `6090004185` | `DE` | Picked, packed, and PGI completed | Billing document, delivery note, packing list |
| Scenario 3 | `6090005272` | `PL` | Picking partly available, PGI not completed | No shipment documents should be attached |

## Completed Order: `6090004185`

Known values:

- Outbound delivery: `90000785`
- Invoice: `9087400156`
- Packing list: `PackingList_ODO0090000785_HU00987500364128609231.pdf`
- Delivery note: `DeliveryNote90000785.pdf`
- Billing document: `Billing Document - 9087400156.pdf`
- Line `0010`: product `A100247-DE`, batch `B72814`, quantity `1.00`, packed quantity `1`
- Line `0020`: product `A100248-DE`, batch `B72815`, quantity `1.00`, packed quantity `1`

Questions to ask:

1. "What is the status of sales order 6090004185?"
2. "Please send all shipment documents for order 6090004185."
3. "Can you send me the invoice for order 6090004185?"
4. "Can you send the delivery note and packing list for outbound delivery 90000785?"
5. "What items and quantities are in sales order 6090004185?"
6. "What is the status of line 20 of order 6090004185?"
7. "For order 6090004185, what is the GI status and shipper box for product A100248-DE?"
8. "Please send the packing list for line item 20 of order 6090004185."
9. "What is the batch and expiry date for product A100247-DE in order 6090004185?"

Expected behavior:

- The app should find HANA order context for `6090004185`.
- Status questions should report completed picking, packing, and PGI.
- Broad document requests should attach available billing, delivery note, and packing list files.
- Line-specific document requests for line `20` should prefer the packing list.

## Pending Order: `6090005272`

Known values:

- Country: `PL`
- Outbound deliveries: `90001542` and `90001591`
- Line `0030`: product `P510258`, batch `L674161`, quantity `1`, picked but PGI not completed
- The order has 41 status lines and no item-detail rows in the anonymized seed.

Questions to ask:

1. "What is the status of sales order 6090005272?"
2. "Is there a back order on sales order 6090005272?"
3. "Can you share shipment documents for order 6090005272 if they are available?"
4. "What is the GI status for product P510258 in order 6090005272?"
5. "What line contains product P510258 in sales order 6090005272?"
6. "What is the status of line 30 of order 6090005272?"
7. "Can you send the delivery note and packing list for order 6090005272?"

Expected behavior:

- The app should find HANA status context for `6090005272`.
- The app should explain that PGI is not completed.
- The app should not attach shipment documents for this order.
- Product `P510258` should match line `0030` and batch `L674161`.

## Negative Checks

Use these to confirm the app does not rely on old non-anonymized IDs:

1. "What is the status of sales order 5030000237?"
2. "Please send shipment documents for order 5030002384."
3. "Can you find outbound delivery 50000214?"

Expected behavior:

- These old source IDs should not match the anonymized HANA seed data.
- The app should either ask for review or report that no matching HANA context/documents were found.
