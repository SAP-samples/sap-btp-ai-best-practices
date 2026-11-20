# Anonymized sample data

Shareable sample data for demos, tests and new team members. Everything in this folder is
fictitious: company, person and brand names, addresses, phone and tax numbers, customer,
material, product and order numbers are replaced by consistent fakes. Prices, quantities,
pack sizes, dates, cities and countries are kept, so the orders still validate against the
master data.

```
orders/                 12 sample purchase orders (PDF), logos removed
  6k <customer> <no>.pdf          German body-shop orders (DE10, EUR)
  ... Promotion Order (...).pdf   Vietnamese promotion order (VN10, VND)
  PO ... / Purchase_Order ...     Singapore yacht orders (SG10, SGD), one with a 15 % discount
seed/
  master_data.json      mock ECC master data (customers, partners, contacts, materials,
                        customer material codes, customer prices, unit aliases)
  expected_lines.json   acceptance list: every order line with the material it must resolve to
  de_order_lines.json   lines of the German orders (printed code, description, price)
```

The seed and the PDFs were anonymized in one run with one mapping, so a customer name, a
customer number, a printed product code or an e-mail domain has the same fake value in the
PDF and in the master data.

## How to use it

Run from the `api/` folder with the project virtual environment and `api/.env` filled in.

1. Load the anonymized master data into HANA. This **replaces** the content of the mock ECC
   tables (`*_POAGENT`) of the HANA user in `api/.env`:

   ```bash
   ../.venv/bin/python -m app.cli.seed_master_data --file ../data/seed/master_data.json --dry-run
   ../.venv/bin/python -m app.cli.seed_master_data --file ../data/seed/master_data.json
   ```

2. Check that the material search finds the expected material for every line (no LLM):

   ```bash
   ../.venv/bin/python -m app.cli.check_material_matching --file ../data/seed/expected_lines.json
   ```

   Expected result: `139/139` among the candidates, `137/139` ranked first (same as the
   original data).

3. Process the sample orders end to end with the agent:

   ```bash
   ../.venv/bin/python -m app.cli.run_samples ../data/orders --concurrency 4
   ```

   Or start the API and UI (see the main `README.md`) and drop PDFs from `orders/` into the
   **Upload** dialog.

## Notes

* `expected_lines.json` also has 4 lines of an e-mail order (`.eml`) that is not part of this
  folder; the matching check uses only the line data, so these lines still count.
* The fake brand and product names are invented. Any similarity with a real brand is not
  intended.
