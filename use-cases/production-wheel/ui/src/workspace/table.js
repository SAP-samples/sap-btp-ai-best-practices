import { request } from "../services/api.js";
import { el, options, fieldLabel } from "./dom.js";
import { fieldDescription, fieldValue, viewLabel, viewDescription } from "./vocabulary.js";
import { renderGlossary } from "./glossary.js";

/** A server-side, searchable, sortable, paginated evidence table; no local data storage. */
export class QueryTable {
  /** Create controls in target; context supplies current dataset/run/point identifiers. */
  constructor(target, { context, views, fail, onSelect }) {
    this.context = context;
    this.fail = fail;
    this.onSelect = onSelect;
    this.offset = 0;
    this.limit = 25;
    this.sort = [];
    this.sequence = 0;
    this.total = 0;
    target.innerHTML = `<div class="table-toolbar"><label>View <select data-view></select></label><label>Search column <select data-field><option value="">Choose a column</option></select></label><label>Contains <input data-search placeholder="Search evidence"></label><ui5-button data-go>Search</ui5-button></div><p data-view-help class="muted"></p><details data-glossary><summary>Column glossary</summary><div data-definitions></div></details><div class="table-scroll" tabindex="0"><table><thead></thead><tbody></tbody></table></div><div class="pagination"><ui5-button data-prev>Previous</ui5-button><span data-count role="status"></span><ui5-button data-next>Next</ui5-button></div>`;
    this.target = target;
    this.view = target.querySelector("[data-view]");
    options(
      this.view,
      views.map((view) => [view, viewLabel(view)]),
    );
    this.field = target.querySelector("[data-field]");
    this.search = target.querySelector("[data-search]");
    this.view.addEventListener("change", () => {
      this.offset = 0;
      this.sort = [];
      this.search.value = "";
      this.field.value = "";
      this.refresh().catch(fail);
    });
    target.querySelector("[data-go]").addEventListener("click", () => {
      this.offset = 0;
      this.refresh().catch(fail);
    });
    this.search.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        this.offset = 0;
        this.refresh().catch(fail);
      }
    });
    target.querySelector("[data-prev]").addEventListener("click", () => {
      this.offset = Math.max(0, this.offset - this.limit);
      this.refresh().catch(fail);
    });
    target.querySelector("[data-next]").addEventListener("click", () => {
      if (this.offset + this.limit < this.total) {
        this.offset += this.limit;
        this.refresh().catch(fail);
      }
    });
  }

  /** Remove evidence and invalidate in-flight responses when selection changes. */
  clear() {
    this.target.querySelector("[data-view-help]").textContent = "";
    this.target.querySelector("[data-definitions]").replaceChildren();
    ++this.sequence;
    this.offset = 0;
    this.total = 0;
    this.target.querySelector("thead").replaceChildren();
    this.target.querySelector("tbody").replaceChildren();
    this.target.querySelector("[data-count]").textContent =
      "Select a dataset or frontier point.";
  }

  /** Query current view and discard stale responses after a selection change. */
  async refresh(reset = false) {
    if (reset) this.offset = 0;
    const sequence = ++this.sequence;
    const scope = this.context(this.view.value);
    if (!scope?.dataset_id && !scope?.run_id) {
      this.target.querySelector("tbody").replaceChildren();
      this.target.querySelector("[data-count]").textContent =
        "Select a dataset or run.";
      return;
    }
    this.target.querySelector("[data-count]").textContent = "Loading evidence…";
    const result = await request("/api/query", "POST", {
      view: this.view.value,
      ...scope,
      fields: [],
      filters:
        this.search.value && this.field.value
          ? [
              {
                field: this.field.value,
                op: "contains",
                value: this.search.value,
              },
            ]
          : [],
      group_by: [],
      metrics: [],
      sort: this.sort,
      offset: this.offset,
      limit: this.limit,
    });
    if (sequence !== this.sequence) return;
    const rows = result.rows || [];
    const columns = (
      result.columns?.length ? result.columns : Object.keys(rows[0] || {})
    ).map((column) =>
      typeof column === "string" ? column : column.name || column.field,
    );
    options(
      this.field,
      [["", "Choose a column"], ...columns.map((column) => [column, fieldLabel(column)])],
      this.field.value,
    );
    this.target.querySelector("[data-view-help]").textContent = viewDescription(this.view.value);
    renderGlossary(this.target.querySelector("[data-definitions]"), columns);
    const header = el("tr");
    for (const column of columns) {
      const cell = el("th");
      cell.title = [fieldDescription(column), result.units?.[column] ? `Unit: ${result.units[column]}` : ""].filter(Boolean).join(" ");
      const button = el(
        "button",
        `${fieldLabel(column)}${this.sort[0]?.field === column ? (this.sort[0].direction === "asc" ? " ↑" : " ↓") : ""}`,
        "sort-button",
      );
      button.addEventListener("click", () => {
        this.sort = [
          {
            field: column,
            direction:
              this.sort[0]?.field === column &&
              this.sort[0]?.direction === "asc"
                ? "desc"
                : "asc",
          },
        ];
        this.refresh().catch(this.fail);
      });
      cell.append(button);
      header.append(cell);
    }
    this.target.querySelector("thead").replaceChildren(header);
    this.target.querySelector("tbody").replaceChildren(
      ...rows.map((row) => {
        const tr = el("tr");
        for (const column of columns) tr.append(el("td", fieldValue(column, row[column])));
        if (this.onSelect) {
          tr.tabIndex = 0;
          tr.addEventListener("click", () => this.onSelect(row));
          tr.addEventListener("keydown", (event) => {
            if (event.key === "Enter") this.onSelect(row);
          });
        }
        return tr;
      }),
    );
    this.total = Number(result.total || 0);
    this.target.querySelector("[data-count]").textContent =
      `${this.total ? this.offset + 1 : 0}–${this.offset + rows.length} of ${this.total} records${Object.values(result.units || {}).some(Boolean) ? " · Hover column headers for units" : ""}`;
    this.target.querySelector("[data-prev]").disabled = this.offset === 0;
    this.target.querySelector("[data-next]").disabled =
      this.offset + this.limit >= this.total;
  }
}
