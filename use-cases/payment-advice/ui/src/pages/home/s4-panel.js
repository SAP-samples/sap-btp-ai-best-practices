/**
 * "S/4 posting" tab of the advice review pane.
 *
 * Step 1 "Check against S/4" (read-only): the server looks every invoice reference up in S/4,
 * derives company code / customer / currency, validates and stores a payment advice preview.
 * Step 2 "Post to S/4" (final, reviewed advices only): the server re-checks against live S/4
 * data and creates the S/4 Payment Advice; the advice then becomes immutable.
 */
import { node, value, keyValueList, button } from "./dom.js";

const kindLabels = {matched: "S/4 item", deduction: "Deduction", unresolved: "Not resolved"};
const statusLabels = {open: "Open", cleared: "Already cleared", not_found: "Not in S/4", ambiguous: "Ambiguous"};

/** One-line state of the S/4 step, shown above the actions. */
function summary(advice, s4) {
  if (s4?.posted) return ["Posted to S/4 as payment advice " + s4.posted.key.PaymentAdvice + ".", "s4-ok"];
  if (!s4) return ["Not checked against S/4 yet.", ""];
  if (s4.post_error) return ["S/4 rejected the last post: " + s4.post_error, "s4-blocking"];
  const blocking = s4.issues.filter(i => i.severity === "blocking").length;
  if (blocking) return [blocking + " blocking issue(s): resolve them, then check again.", "s4-blocking"];
  if (advice.status !== "reviewed") return ["Ready for S/4. Mark the advice reviewed to enable posting.", "s4-ok"];
  return ["Ready to post. Posting re-checks against live S/4 data first.", "s4-ok"];
}

/** Table of advice lines with their S/4 match. */
function linesTable(lines) {
  const scroll = node("div", "", "inbox-table-scroll"); scroll.tabIndex = 0;
  scroll.setAttribute("aria-label", "S/4 match per advice line, horizontally scrollable");
  const table = node("table", "", "s4-lines"), head = node("tr"), body = node("tbody");
  ["#", "Reference", "Kind", "S/4 status", "S/4 document", "Account (rules)", "S/4 customer", "S/4 amount", "Advice net", "Reason"].forEach(h => head.append(node("th", h)));
  for (const line of lines) {
    const item = line.s4_item || {};
    const tr = node("tr", "", "s4-" + line.kind);
    // A deduction never has an S/4 item before clearing creates its residual, so "Not in S/4" would read as an error.
    const status = line.kind === "deduction" ? "New deduction" : statusLabels[line.status] || line.status;
    [line.index + 1, line.reference, kindLabels[line.kind] || line.kind, status,
     item.accounting_document ? item.accounting_document + "/" + item.fiscal_year + " (" + line.matched_by + ")" : "",
     line.account || "", item.customer || "", item.amount ? item.amount + " " + item.currency : "", line.net, line.reason_code || ""]
      .forEach(cell => tr.append(node("td", value(cell))));
    body.append(tr);
  }
  const thead = node("thead"); thead.append(head); table.append(thead, body); scroll.append(table);
  return scroll;
}

/** Collapsible JSON block (payload preview, S/4 call trace, read-back). */
function disclosure(title, data) {
  const details = node("details"), pre = node("pre", JSON.stringify(data, null, 2));
  details.append(node("summary", title), pre);
  return details;
}

/**
 * Render the S/4 tab.
 * @param {HTMLElement} content Tab panel to fill.
 * @param {object} advice Current advice record (with optional `s4` state).
 * @param {{post: (path: string, body: object) => Promise<void>, onError: (e: Error) => void}} actions
 *   `post` sends a revision-checked advice mutation and refreshes the view.
 */
export function renderS4Panel(content, advice, {post, onError}) {
  const s4 = advice.s4, [text, cls] = summary(advice, s4);
  const state = node("p", text, "s4-summary " + cls); state.setAttribute("role", "status");
  content.append(state);
  const busy = ["queued", "processing", "failed"].includes(advice.status) || !advice.result;
  const bar = node("div", "", "advice-actions");
  if (!s4?.posted) {
    const check = button("Check against S/4", () => post("/s4/check", {revision: advice.revision}), onError);
    check.disabled = busy;
    const derived = s4?.derived || {};
    const send = button("Post to S/4", async () => {
      const ok = window.confirm("Create the payment advice in S/4 for company code " + derived.company_code +
        ", customer " + derived.customer + ", amount " + s4.payload?.PaidAmountInPaytCurrency + " " + derived.currency + "?");
      if (ok) await post("/s4/post", {revision: advice.revision});
    }, onError, "primary");
    send.disabled = busy || advice.status !== "reviewed" || !s4?.ready;
    bar.append(check, send);
  }
  content.append(bar);
  if (!s4) return;
  const derived = s4.derived || {};
  content.append(keyValueList({"company code": derived.company_code, "company code (rules)": derived.rule_company_code || "none",
    "advice account": derived.customer, "payer (rules)": derived.payer_account || "none",
    "other customers": (derived.customers || []).slice(1).join(", ") || "none", currency: derived.currency,
    "checked at": s4.checked_at, connection: s4.mode}));
  if (s4.issues?.length) {
    const list = node("ul", "", "s4-issues");
    s4.issues.forEach(issue => list.append(node("li", issue.message, "s4-" + issue.severity)));
    content.append(list);
  }
  content.append(linesTable(s4.lines || []));
  if (s4.posted) content.append(disclosure("Stored in S/4 (read-back)", s4.posted.read_back));
  if (s4.payload) content.append(disclosure("Payment advice payload", s4.payload));
  content.append(disclosure("S/4 calls made by the check", s4.trace || []));
}
