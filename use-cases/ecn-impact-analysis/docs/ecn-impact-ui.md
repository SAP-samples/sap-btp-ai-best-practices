# ECN Impact user interface

## Purpose

The browser experience turns deterministic ECN extraction and optional AI
generation into a business-readable inbox and detail view. It separates source
readiness from AI availability: a provider outage can block narrative generation
without hiding material, BOM, configuration, or data-quality evidence.

The product routes are `/ecns`, `/settings`, and four ECN views: `/ecns/:ecn_row_id` (Overview), `/impact`, `/attention`, and `/references` beneath that ECN route. `/` and
`/home` redirect to `/ecns`. The only decision control is the Engineering review
(Accept ECN / Reject, see [engineering-review.md](engineering-review.md)); no
override or downstream write-back controls are exposed, and Accept writes nothing to SAP.

## Inputs and outputs

The ECN inbox accepts one JSON file. Reusable workbooks are published in Settings
and pinned by the API when the ECN upload is committed. The list shows business identifier, a two-line accessible description, upload date, material/warning
counts, readable analysis state, overall impact, and one View action. Four KPI
cards summarize total, complete, in-progress, and needs-attention counts. All
four are keyboard-accessible filter buttons with an active selection highlight:
**Total ECNs** clears the analysis filter, **Summaries available** includes
`COMPLETED` and `COMPLETED_WITH_GAPS`, **In progress** includes `QUEUED` and
`RUNNING`, and **Generation needs attention** filters `FAILED`. The list API
and filtered bulk actions accept `SUMMARIES_AVAILABLE` and `IN_PROGRESS` as
filter groups; individual row states remain unchanged.

At 600 px and below, Search, Analysis, Sort, and Go remain immediately
available. ECN number, Plant, Warnings, Order, and Rows are grouped in a named
**Filters** disclosure with an active secondary-filter count and Clear filters
control; defaults and empty values are not counted. Desktop layouts keep every
filter directly visible. Each KPI action preserves the other applied filters
and sort, updates the dropdown and URL, resets the page to 1, and clears row
selections. Counts remain global; list pagination and filtered bulk actions use
the same server-side group membership.

Route stylesheets remain loaded during same-tab navigation. Shared surface,
table, helper-text and pagination rules are scoped to `.ecn-page`, `.ecn-detail`
or `.settings-page` with `:where()` to preserve selector specificity. This keeps
the inbox's fixed column widths, cell spacing and status wrapping stable after
opening an ECN or Settings, regardless of stylesheet load order.

Descriptions use a fixed-width table column and a two-line visual clamp on desktop. The full text remains available through the element's accessible label, focus target, and native title, as well as on the detail page. Finite analysis, review, and impact states use UI5 semantic tags in both the table and mobile cards; internal enum values are not shown as customer-facing labels. Reset, filter, generation, loading, and list-error feedback uses UI5 message strips while retaining explicit `status` or `alert` live-region behavior. At 600 px and below, the desktop table is replaced by accessible ECN result cards that keep the ECN, description, analysis/review state, impact, warnings, plant state, selection, and View action visibly associated without document-level horizontal overflow.

## Business-first detail

The ECN landing view is a decision brief, not a source-data dump. It contains three visual areas only: **Change**, **Impact at a glance**, and **Decision**. Together they answer the change-integration-board questions without requiring the reviewer to reconcile multiple summaries.

- **Change** combines the ECN number, stored one-sentence AI change brief, business reason, owner, and earliest known cutover. Existing analyses use their AI "What changes" line until regenerated. Raw source text never appears in the headline.
- **Impact at a glance** contains at most three named rows: affected plants and assembly/part with any decision-relevant EBOM/MBOM gap; the incoming/outgoing phase transition and date; and the make-or-buy/cost, lead-time, sourcing, or manufacturing factor.
- **Decision** shows one approve, hold, or investigate outcome and at most one direct, owned action that prevents it. Readiness counts, severity chips, generic confirmation tasks, and missing-field inventories do not appear here.
- Parts, detailed BOMs, procurement detail, source documents, audit records, original ECN text, and full AI narrative are drill-down evidence. Descriptions are the primary part labels; an identifier remains the fallback and opens the source evidence.
- The floating footer is the only visible Accept/Send back control. The review tab keeps the acceptance status, plan, and history without repeating actions.

The side navigation retains ECNs and Settings, with one ECN Overview item immediately after ECNs while an ECN is open (`syncEcnNavigation` in `ui/src/modules/navigation.js`). The duplicate Impact analysis, What needs attention and Reference documents sidebar shortcuts are removed; their routes and underlying detail content remain available. Changing ECN updates the Overview link, and leaving removes the ECN-specific item. The main-page Overview, Plants, BOM Change Evidence and Audit Log navigation is unchanged. Switching between the retained ECN route views keeps the assessment loaded and polling shared: Overview selects the decision brief, Impact focuses Impact at a glance, Attention focuses the Decision outcome, and References opens the audit tab.

Detailed tooling inventory, machine utilization, work-instruction steps, and predicted changeover minutes are intentionally not displayed in the decision workflow. The supplied source data does not establish those facts. Manufacturing is represented only as an implementation-timing signal until evidence-led production data is available.

Prototype cost, sourcing, and manufacturing timing enrichments remain demonstration-only inputs. They are used only where they directly support the decision story and are not presented as source evidence.

The snapshot supplies complete part and reference lists, including BOM-only parts absent from the ECN export. Older responses fall back to the existing paginated business-evidence API. A stored summary is labeled **Regenerate summary**; opening any page never starts a model call.

Chat is mounted in the app shell on all pages. It is enabled while an ECN is open and disabled elsewhere with a clear prompt to open an ECN. Conversation history stays in memory per ECN and clears on reload. Wide layouts use a side panel; narrow layouts use the named Shell Bar action to open a modal assistant without a floating launcher.

Polling, detail reads, evidence pages, source-detail reads, retries, source
publication, provider checks, and reset each have isolated abort lanes. One
refresh cannot cancel an unrelated mutation.

### Dynamic Page scrolling boundary

ECN detail keeps the approved shared scrolling page: route views target the
existing evidence sections in place, navigation buttons update the visible
reading position, and the scroll spy maintains `aria-current` on the active
section. The pinned-title offset and DynamicPage scroller currently require two
internal selectors from installed `@ui5/webcomponents-fiori` 2.26.0. They are
contained exclusively in `ui/src/pages/ecn-detail/dynamic-page-scroll-adapter.js`;
page code neither queries nor retains an internal node. This is documented
technical debt because no verified public DynamicPage API preserves the required
pinned-title precision. If the internal template is unavailable or changes, the
adapter performs a null-safe native section `scrollIntoView` fallback and leaves
the current navigation state intact rather than failing the detail page.

## Settings source management

Settings contains five maintained source cards: EBOM/MBOM, conditional
configuration rules, material master, purchase info records, and stock. Cards
show every active type/scope publication with filename, scope, row and warning
counts, upload time, and shortened hash. A valid replacement activates
atomically.

Make/Buy status is a direct material-master attribute (ProcType column) and is
not a separate ingestion source. The retired historical Make/Buy decisions and
Make/Buy business-rules card types are no longer accepted or displayed.

The provider card displays cached OAuth and deployment availability for the
configured text-generation model (default `gpt-5.4`). Analysis retries a
timed-out text-generation
request once on `gpt-5.4`. **Check now** is the
only action that contacts AI Core. Credentials, endpoints, request/response
bodies, prompts, and tenant identifiers are never displayed.

Settings keeps these tasks on the existing `/settings` route in three visible
areas: **Provider readiness**, **Reference datasets**, and progressively
disclosed **Advanced administration**. A source replacement clearly progresses
through choosing the supported file, describing its model-range/data-group/
environment scope, then validating and atomically activating it. Validation
failure leaves the current active publication unchanged; successful activation
applies only to future ECN imports in that exact scope, while existing ECNs
retain their pinned source versions. Publication cards show safe row/warning,
scope, date, active-source, and secondary traceability metadata only.

Reset remains fail-closed and requires the exact phrase `RESET HISTORY`. Any
queued, provider-blocked, or running job keeps reset disabled. Reset preserves
Settings-managed reference datasets.

## Related files

- `ui/src/pages/ecns/`: JSON intake, KPIs, filters, bounded table, and pagination.
- `ui/src/pages/ecn-detail/`: four views, shared summary/polling, complete snapshot reference lists, and
  optional technical provenance.
- `ui/src/pages/ecn-detail/dynamic-page-scroll-adapter.js`: contained internal-DOM scrolling, pinned-title offset, and scroll-spy boundary for the installed DynamicPage.
- `ui/src/pages/settings/`: safe provider diagnostics, immutable source cards,
  status legend, and guarded reset.
- `ui/src/components/analysis-sections.js`: readable AI narrative renderer.
- `ui/src/services/ecn-api.js`: exact JSON upload, evidence, reference, provider,
  progress, retry, and reset HTTP contracts.
- `api/app/ecn/api_models.py`: metadata-only detail and list overall-impact DTOs.
- `api/app/persistence/codec.py`: persisted analysis severity projection.

## Test commands

```bash
npm --prefix ui test -- tests/dynamic-page-scroll-adapter.test.js tests/ecn-detail-page.test.js
npm --prefix ui test
npm --prefix ui run build
PYTHONPATH=api .venv/bin/python -m pytest -q \
  tests/unit/test_ecn_repository.py \
  tests/integration/test_ecn_workflow.py
```

Browser acceptance should inspect `/ecns`, `/settings`, and one real detail at a
desktop width and a narrow width. Confirm the root does not overflow, descriptions
stay at two lines, evidence is not fetched before expansion, and raw technical
data does not appear before explicit source inspection.

The original ECN description and full AI assessment are collapsed in History &
Audit. The initial Overview emphasizes only the plain-language decision story.
