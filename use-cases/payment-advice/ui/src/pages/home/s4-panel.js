/**
 * "S/4 posting" tab of the advice review pane.
 *
 * Step 1 "Check against S/4" (read-only): the server looks every invoice reference up in S/4,
 * derives company code / customer / currency, validates and stores a payment advice preview.
 * Step 2 "Post to S/4" (final, reviewed advices only): the server re-checks against live S/4
 * data and creates the S/4 Payment Advice; the advice then becomes immutable.
 */
import { node, value, keyValueList, button } from "./dom.js";

const kindLabels = {matched: "S/4 item", deduction: "Deduction", unresolved: "Not resolved", residual: "Residual"};
const statusLabels = {open: "Open", cleared: "Already cleared", reversed: "Reversed in S/4", not_found: "Not in S/4",
  ambiguous: "Ambiguous", residual: "Posted as open item"};
/** Blocking issues the reviewer can resolve by posting the paid line as a residual open item. */
const residualIssues = new Set(["invoice_reversed", "invoice_not_found"]);

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

/**
 * Post a reversed / not-found invoice line as a residual open item, or undo that choice. Shown next to
 * the message it resolves; saved as a normal correction (audited, undoable), then the advice is checked again.
 */
function residualAction(line, advice, {post, onError}) {
  const residual = line.kind === "residual";
  return button(residual ? "Undo residual" : "Post line " + (line.index + 1) + " as residual", async () => {
    if (!residual && !window.confirm("Post line " + (line.index + 1) + " (" + line.reference + ") to S/4 as an open item " +
      "on the customer instead of applying it to an invoice? Set its reason code in the advice table, then check again.")) return;
    await post("/edit", {revision: advice.revision, target: line.row_id, field: "s4_treatment",
      value: residual ? null : "residual", reason: residual ? "Undo residual posting" : "Post as residual in S/4"});
  }, onError, "inline-action");
}

/**
 * Issue messages, with the residual action inside the message it resolves, and one note per line the
 * reviewer already posts as residual (with Undo). No actions once the advice is posted.
 */
function issuesList(s4, advice, actions) {
  const list = node("ul", "", "s4-issues"), editable = advice.status !== "posted";
  const lines = new Map((s4.lines || []).map(line => [line.index, line]));
  for (const issue of s4.issues || []) {
    const item = node("li", issue.message, "s4-" + issue.severity), line = lines.get(issue.line);
    if (editable && line?.row_id && residualIssues.has(issue.code)) item.append(residualAction(line, advice, actions));
    list.append(item);
  }
  for (const line of s4.lines || []) {
    if (line.kind !== "residual") continue;
    const item = node("li", "Line " + (line.index + 1) + " (" + line.reference + ") is posted as an open item on the customer.", "s4-info");
    if (editable && line.row_id) item.append(residualAction(line, advice, actions));
    list.append(item);
  }
  return list;
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

/** "/Date(1758153600000)/" (OData V2) -> "2025-09-18"; other values unchanged. */
function odataDate(text) {
  const ms = /\/Date\((-?\d+)\)\//.exec(text || "")?.[1];
  return ms ? new Date(Number(ms)).toISOString().slice(0, 10) : text || "";
}

/** "PT13H32M35S" (OData V2 time) -> "13:32:35". */
function odataTime(text) {
  const [, h = 0, m = 0, sec = 0] = /PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/.exec(text || "") || [];
  return text ? [h, m, sec].map(part => String(part).padStart(2, "0")).join(":") : "";
}

/**
 * Result card after posting: what S/4 created, read back from S/4 itself, and whether it holds exactly
 * what was sent (`posted.verification`, computed by the server at post time).
 */
function postedCard(posted) {
  const card = node("section", "", "s4-result"), stored = posted.read_back || {};
  card.setAttribute("aria-label", "Payment advice created in S/4");
  card.append(node("h4", "Payment advice " + posted.key.PaymentAdvice + " created in S/4"));
  if (stored.read_back_error) {
    card.append(node("p", "Created, but reading it back from S/4 failed: " + stored.read_back_error, "s4-blocking"));
    return card;
  }
  card.append(keyValueList({"company code": stored.CompanyCode, "customer account": stored.PaymentAdviceAccount,
    amount: stored.PaidAmountInPaytCurrency + " " + stored.PaymentCurrency, "payment date": odataDate(stored.PaymentDate),
    "payment reference": stored.PaymentAdviceHeaderText, "created in S/4": odataDate(stored.CreationDate) + " " +
    odataTime(stored.CreationTime) + " by " + stored.CreatedByUser, "posted from the app": posted.posted_at}));
  const check = posted.verification;
  if (!check) card.append(node("p", "No comparison with the sent payload was recorded for this post.", ""));
  else if (!check.differences.length)
    card.append(node("p", "Verified: S/4 holds " + check.items_stored + " of " + check.items_sent +
      " items, and every value matches what was sent.", "s4-ok"));
  else {
    card.append(node("p", "S/4 differs from what was sent:", "s4-blocking"));
    const list = node("ul", "", "s4-issues");
    check.differences.forEach(text => list.append(node("li", text, "s4-blocking")));
    card.append(list);
  }
  card.append(storedItemsTable(stored));
  return card;
}

/** Items as S/4 stored them (from the read-back), one row per payment advice item. */
function storedItemsTable(stored) {
  const scroll = node("div", "", "inbox-table-scroll"); scroll.tabIndex = 0;
  scroll.setAttribute("aria-label", "Payment advice items stored in S/4, horizontally scrollable");
  const table = node("table", "", "s4-lines"), head = node("tr"), body = node("tbody");
  ["Item", "Reference", "Text", "Invoice document", "Customer", "Gross", "Discount", "Net", "Reason"].forEach(h => head.append(node("th", h)));
  for (const item of stored.items || []) {
    const tr = node("tr");
    [item.PaymentAdviceItem, item.DocumentReferenceID || item.AssignmentReference, item.DocumentItemText,
     item.AccountingDocument ? item.AccountingDocument + "/" + item.FiscalYear : "",
     item.PaymentAdviceAltvAccount || stored.PaymentAdviceAccount, item.GrossAmountInPaymentCurrency,
     item.CashDiscountAmountInPaytCrcy, item.NetPaymentAmountInPaytCurrency, item.PaymentDifferenceReason]
      .forEach(cell => tr.append(node("td", value(cell))));
    body.append(tr);
  }
  const thead = node("thead"); thead.append(head); table.append(thead, body); scroll.append(table);
  return scroll;
}

/** Collapsible JSON block (payload preview, S/4 call trace). */
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
  if (s4.posted) content.append(postedCard(s4.posted));
  const derived = s4.derived || {};
  content.append(keyValueList({"company code": derived.company_code, "company code (rules)": derived.rule_company_code || "none",
    "advice account": derived.customer, "payer (rules)": derived.payer_account || "none",
    "other customers": (derived.customers || []).slice(1).join(", ") || "none", currency: derived.currency,
    "checked at": s4.checked_at, connection: s4.mode}));
  const list = issuesList(s4, advice, {post, onError});
  if (list.children.length) content.append(list);
  content.append(linesTable(s4.lines || []));
  if (s4.payload) content.append(disclosure("Payment advice payload", s4.payload));
  content.append(disclosure("S/4 calls made by the check", s4.trace || []));
}
