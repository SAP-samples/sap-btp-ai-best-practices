# Anonymization for Publication

## What it is

A configuration-driven tool that turns this private project into a publishable one. It rewrites
two surfaces with **one shared set of rules**:

1. **Data**: writes `data/`, a mirror of the source files the application and the
   development-fixture tests use (Excel workbooks and a reduced ECN JSON export).
2. **Repository**: rewrites git-tracked code, tests and docs (`api/`, `ui/`, `docs/`, `tests/`,
   `scripts/`, `joule/`, root README/GLOSSARY/manifest files) and renames files whose names
   contain customer terms.

Both surfaces share the same rules, and part numbers use the same keyed permutation. A part
number in a test assertion, in the BOM workbook and in the ECN JSON therefore becomes the same
fake value, and the tests still pass against the anonymized data.

## What it is for

Publishing the use case (for example to a public samples repository) without customer names,
plants, brands, people, supplier names, model ranges or recognisable part numbers. No customer
term may be used as an identifier (file, module, column, variable or table name) in the result.

## Inputs and outputs

| Input | Location | Published? |
|---|---|---|
| Private configuration: terms, code patterns, secret key | `data/anonymization/terms.yaml` | No (git-ignored) |
| Source data | `data/` (except `data/`) | No (git-ignored) |

| Output | Location | Published? |
|---|---|---|
| Anonymized dataset | `data/` | Yes |
| Rewritten repository files and renames | working tree (commit them) | Yes |
| Generated fakes (re-identification key) | `data/anonymization/mapping.json` | **Never** |
| Harvest cache, review file, leak report | `data/anonymization/*.json`, `*.md` | **Never** |

`mapping.json`, `harvest.json`, `repository_review.md` and `leak_report.md` contain original
values. They stay in the git-ignored `data/anonymization/` folder. Keep that folder private: it
re-identifies the published data.

## How it works

Text passes, applied in this order to every published string (cells, sheet titles, JSON keys
and values, file names, lines of code):

1. E-mail addresses on private domains become `first.last@example.com` or `userNNN@example.com`.
2. Person names in every variant: `Last, First`, `First Last`, upper case, `F.LAST`,
   `first.last`, and the bare surname. A bare surname is not replaced when it is a common
   dictionary word. People are harvested from the **full** JSON export, including
   `DisplayName` fields. Decorations such as `DELETED-`, `(External)` and `Admin_02` are
   normalized away.
3. Supplier names become `SUPPLIER NN <legal form>`.
4. Literal terms from the config: company, plants, brands, plant pipe standard, PLM host,
   local checkout path.
5. Model-family codes (letter + 3 digits before a series letter, e.g. `A123V`) become `Q9NN`,
   including the dashed spelling `A1-23`.
6. Short codes as whole tokens (brand suffixes, plant codes).
7. Digit rules: OEM-style part numbers get a new prefix and a keyed permutation of their
   digits. The revision character is kept, so `X1234560` and `X123456A` stay related.
8. Plain numeric material numbers (harvested from material columns) keep their first digit
   and permute the other digits. That avoids a leading zero, which Excel number cells would
   drop.

The digit permutation (`scripts/anonymization/digits.py`) is a keyed Feistel network over
base-10 strings. It is deterministic for a key, bijective per length, and not reversible
without the key.

### Choices that keep the application working

- Replacement codes keep the **sort order** of the originals where tests or the UI sort by
  them. The plant codes and the rule brand suffixes are chosen this way.
- Upper-case names become identifier-safe (`PLANT_A`), because some appear as constants in tests.
- Plant names that appear as unquoted values in rule expressions become `Plant_A` / `Plant_B`,
  so every rule still parses.
- Workbook structure is unchanged (sheets, headers, row and column counts). Quantities, dates,
  ECN numbers and PLM object IDs are kept.
- The JSON export is reduced to the ECNs the demo and the tests need (11 of 557). One
  integration test was updated to the subset count.
- Derived helper columns that glue several original codes into one string (the BOM
  workbook's `Concat`, `Concat 2`, `Concat NA`) are emptied through `datasets.clear_columns`.
  The text rules match codes on word boundaries, so codes embedded inside such strings would
  otherwise survive. The header cell stays; the application does not read these columns.

### PDF drawing fixtures

The drawing fixtures (`api/app/ecn/fixtures/drawings/QKX*.pdf`) originally carry the title
block, the logo (vector art) and designer names. These are covered by hand with white boxes in
a PDF editor. A white box only hides content on screen: the words stay in the text layer and
the logo paths stay in the drawing. `scripts/anonymization/redact_hidden_pdf_text.py` removes
them for real:

1. Every large white-filled shape painted over earlier drawing is treated as a white-out. All
   paths and characters inside it (plus a 2-point margin) are removed with a PyMuPDF redaction.
2. Every remaining word whose box renders pure white is removed from the text layer.
3. The XMP packet and the info dictionary are dropped (CAD exports keep title, number and
   owner there).
4. The saved file must render pixel-identical to the input and contain no hidden word;
   otherwise nothing is written.

```bash
python3 scripts/anonymization/redact_hidden_pdf_text.py api/app/ecn/fixtures/drawings/*.pdf --dry-run
```

```bash
python3 scripts/anonymization/redact_hidden_pdf_text.py api/app/ecn/fixtures/drawings/*.pdf
```

Check the result with `pdftotext <file> -` and the `scan` command below.

### Not handled automatically
- Images (PNG screenshots) are not rewritten. Generated screenshots in `ui/test-output/` are
  git-ignored.
- Git history still contains the original names. Publish a copy of the files, not this
  repository with its history.

## Related files

| File | Role |
|---|---|
| `scripts/anonymize_for_publication.py` | CLI (`data`, `repo`, `scan`) |
| `scripts/anonymization/config.py` | Loads and validates the private YAML |
| `scripts/anonymization/terms.example.yaml` | Documented example configuration |
| `scripts/anonymization/replacer.py` | Ordered text engine shared by all surfaces |
| `scripts/anonymization/harvest.py` | Collects people, suppliers, numbers, model families |
| `scripts/anonymization/digits.py` | Keyed format-preserving digit permutation |
| `scripts/anonymization/mapping.py` | Persistent generated fakes (`mapping.json`) |
| `scripts/anonymization/datasets.py` | Workbook, JSON and PDF writers |
| `scripts/anonymization/repository.py` | Repository rewrite and `git mv` renames |
| `scripts/anonymization/leaks.py` | Leak scan over files, workbooks, JSON and PDFs |
| `scripts/anonymization/redact_hidden_pdf_text.py` | Removes text and drawing hidden under white-out boxes in PDFs |

Key snippet (all passes run through one call per string):

```python
store = MappingStore.load(config.work_dir / "mapping.json")
anonymizer = TextAnonymizer(config, store, harvest_result)
anonymizer.apply("Replace XYZ1234567 at plant P1")
```

## How to run and test

Requirements: `pip install openpyxl pyyaml rich pymupdf`.

```bash
python scripts/anonymize_for_publication.py data
```

```bash
python scripts/anonymize_for_publication.py repo --dry-run
```

```bash
python scripts/anonymize_for_publication.py repo
```

```bash
python scripts/anonymize_for_publication.py scan
```

`repo --dry-run` writes `data/anonymization/repository_review.md` with every rename and changed
line. `scan` exits with status 1 when any configured term, harvested person, supplier, model
family or material number survives. Findings are written to `data/anonymization/leak_report.md`.

Verify the result:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit
```

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/integration/test_real_bom_changes_fixture.py tests/integration/test_development_fixtures.py
```

```bash
cd ui && npx vitest run
```

Re-running is deterministic: `mapping.json` and the secret key are reused. Delete `mapping.json`
only to regenerate every fake from scratch. The repository rewrite must then also run again from
the original files.
