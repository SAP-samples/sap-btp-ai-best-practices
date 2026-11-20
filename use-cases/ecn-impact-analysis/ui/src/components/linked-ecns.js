import { element } from "./workspace-dom.js";

/** Render verified outgoing references as navigation actions, or disabled buttons. */
export function renderLinkedEcns(field, references = [], listQuery = "") {
  const value = field.querySelector("dd");
  value.className = "linked-ecn-actions";
  value.replaceChildren();
  const items = references.length ? references : [{ ecn_number: "No linked ECN", ecn_row_id: null }];
  for (const { ecn_number: number, ecn_row_id: rowId } of items) {
    const action = element(rowId ? "a" : "button", number, "linked-ecn-action");
    if (rowId) {
      action.href = `/ecns/${encodeURIComponent(rowId)}${listQuery ? `?${listQuery}` : ""}`;
      action.setAttribute("aria-label", `Open linked ECN ${number}`);
      action.title = `Open ${number}`;
      action.dataset.linkedEcn = number;
    } else {
      action.type = "button";
      action.disabled = true;
      action.title = references.length
        ? `${number} is referenced in the description but is not available.`
        : "No other ECN is referenced in the source description.";
      action.setAttribute("aria-label", references.length ? `${number} unavailable` : "No linked ECN");
    }
    value.append(action);
  }
}
