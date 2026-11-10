/** Email inbox. Chat state lives only in this module, never browser storage. */
import { request, upload, API_BASE_URL, API_KEY } from "../../services/api.js";
import { renderSafeMarkdown } from "../../services/safe-markdown.js";
import { node, value, keyValueList, button } from "./dom.js";
import { renderS4Panel } from "./s4-panel.js";
import { renderAdviceTable } from "./advice-table.js";
const conversations = new Map();
const drafts = new Map();
const labels = {processing:"Processing",needs_review:"Needs review",ready:"Ready",reviewed:"Reviewed",posted:"Posted to S/4",failed:"Failed"};
const base = "/api/email-ingestion";
const adviceBase = "/api/payment-advice/advices/";

/** Initialize one inbox view; navigation disposes polling and attachment object URLs. */
export default function initHomePage() {
  const root = document.getElementById("inbox");
  if (!root) return;
  const get = id => root.querySelector("#" + id);
  const error = e => { get("inbox-error").textContent = e.message || String(e); };
  let customers = [], filter = "", offset = 0, selectedEmail = null, selectedAdvice = null, activeTab = "Original email";
  let details = null, refreshing = null, refreshQueued = false, detailRequest = 0, timer = null, disposed = false, blobUrls = [];
  let selectedAttachment = null;
  const chatBusy = new Set();
  /** Release preview URLs before replacing panels or leaving this page. */
  function release() { blobUrls.forEach(url => URL.revokeObjectURL(url)); blobUrls = []; }
  /** Ignore obsolete detail responses, including A-to-B-to-A selection races. */
  async function loadDetails() {
    const id = selectedEmail, generation = ++detailRequest;
    if (!id) return;
    const response = await request(base + "/inbox/" + id);
    if (!disposed && selectedEmail === id && generation === detailRequest) details = response;
  }
  /** Resolve a registered customer label without guessing an unknown identity. */
  function customerName(key) { return customers.find(c => c.client_key === key)?.display_name || key || "Customer unconfirmed"; }
  /** Create a registered-customer selector with an optional automatic choice. */
  function customerSelect(selected, automatic = false) {
    const select = node("select");
    if (automatic) { const opt = node("option", "Automatic matching"); opt.value = ""; select.append(opt); }
    else if (!selected) { const opt = node("option", "Customer unconfirmed"); opt.value = ""; select.append(opt); }
    customers.forEach(c => { const opt = node("option", c.display_name + (c.is_critical ? " · Priority" : "")); opt.value = c.client_key; select.append(opt); });
    select.value = selected || "";
    return select;
  }
  /** Refresh persisted data after a mutation without dropping browser-only chat. */
  async function updateAdvice(path, body) {
    const id = selectedAdvice;
    await request(adviceBase + id + path, "POST", body);
    await refresh(true);
  }
  /** Confirm and remove the selected email plus all of its persisted workspace children. */
  async function deleteEntry() {
    if (!window.confirm("Delete this email and all attachments, payment advices, corrections, and processing history? This cannot be undone.")) return;
    const emailId = selectedEmail, revision = details.email.revision, adviceIds = details.advices.map(advice => advice.id);
    await request(base + "/inbox/" + emailId, "DELETE", {revision});
    adviceIds.forEach(id => { conversations.delete(id); drafts.delete(id); });
    if (selectedEmail === emailId) {
      selectedEmail = null; selectedAdvice = null; selectedAttachment = null; details = null; ++detailRequest;
    }
    await refresh(true);
  }
  /** Render the current advice table and right-hand source/chat tabs. */
  async function workspace(host) {
    release();
    const advice = details.advices.find(a => a.id === selectedAdvice) || details.advices[0];
    if (!advice) { host.append(node("p", "No processable advice. See email warnings: " + value(details.email.warnings))); return; }
    selectedAdvice = advice.id;
    const wrap = node("div", "", "advice-workspace"), left = node("section", "", "advice-panel"), right = node("section", "", "advice-panel");
    wrap.append(left, right); host.append(wrap);
    const pickerLabel = node("label", "Payment advice ");
    const picker = node("select"); picker.setAttribute("aria-label", "Selected payment advice");
    details.advices.forEach(a => { const option = node("option", a.filename + " · " + (labels[a.status] || a.status)); option.value = a.id; picker.append(option); });
    picker.value = advice.id;
    picker.onchange = () => { selectedAdvice = picker.value; selectedAttachment = null; renderRows().catch(error); };
    pickerLabel.append(picker); left.append(pickerLabel);
    const customerLabel = node("label", "Customer ");
    const select = customerSelect(advice.client_key);
    select.setAttribute("aria-label", "Advice customer");
    customerLabel.append(select, button("Apply customer", () => updateAdvice("/customer", {revision:advice.revision, client_key:select.value}), error));
    left.append(customerLabel, node("span", labels[advice.status] || advice.status, "status-pill status-" + advice.status));
    const actions = node("div", "", "advice-actions");
    const pending = ["queued","processing"].includes(advice.status), posted = advice.status === "posted";
    select.disabled = pending || posted;
    customerLabel.children[1].disabled = pending || posted;
    const displayedResult = advice.result || advice.original_extraction;
    if (advice.result && !pending && !posted && advice.status !== "failed") {
      actions.append(button(advice.status === "reviewed" ? "Reopen" : "Mark reviewed",
        () => updateAdvice("/review/" + (advice.status === "reviewed" ? "reopen" : "reviewed"), {revision:advice.revision}), error));
    }
    if (!pending) {
      if (!posted) actions.append(button("Retry", () => updateAdvice("/retry", {revision:advice.revision}), error));
      if (!details.advices.some(item => ["queued","processing"].includes(item.status))) {
        const remove = button("Delete", deleteEntry, error, "danger");
        remove.title = "Delete this complete inbox entry";
        remove.setAttribute("aria-label", "Delete this email and all related payment advice data");
        actions.append(remove);
      }
    }
    if (advice.corrections?.length && !pending && !posted) actions.append(button("Undo last edit", () => updateAdvice("/undo", {revision:advice.revision}), error));
    left.append(actions);
    if (advice.error) left.append(node("p", advice.error));
    if (pending) {
      const stage = advice.status === "queued" ? "Waiting for a processing slot" :
        advice.stage === "interpreting" ? "Interpreting deductions" : "Extracting with Document AI";
      const progress = node("p", stage + ". Other emails can be submitted and processed in parallel. This view refreshes automatically.");
      progress.setAttribute("role", "status"); left.append(progress);
      if (advice.result) left.append(node("p", "Previous saved result shown until reprocessing finishes; analyst corrections are retained."));
    }
    if (!advice.result && advice.original_extraction) left.append(node("p", "Document AI extraction is available below. Deduction interpretation is not complete; these values are read-only."));
    const header = node("dl", "", "advice-header");
    Object.entries(displayedResult?.header || {}).forEach(([key,v]) => {
      const pair = node("div"), detail = node("dd");
      if (key === "interpretation" && v && typeof v === "object" && !Array.isArray(v)) detail.append(keyValueList(v));
      else detail.textContent = value(v);
      pair.append(node("dt",key.replaceAll("_"," ")),detail); header.append(pair);
    });
    left.append(header);
    const verification = advice.verification;
    if (verification) left.append(node("p", [...verification.issues,...verification.warnings].join(" · ")));
    renderAdviceTable(left, actions, advice, displayedResult, {
      editable: Boolean(advice.result) && !pending && !posted && advice.status !== "failed",
      post: (path, body) => request(adviceBase + advice.id + path, "POST", body),
      rerender: () => refresh(true), onError: error});
    const tabs = node("div", "", "advice-tabs"); tabs.setAttribute("role","tablist");
    const content = node("div"); content.setAttribute("role","tabpanel");
    for (const tab of ["Original email","Attachments","Advice chat","S/4 posting"]) {
      const b = button(tab, async () => { activeTab = tab; await renderRows(); },error);
      b.setAttribute("role","tab"); b.setAttribute("aria-selected",String(tab === activeTab)); tabs.append(b);
    }
    right.append(tabs,content);
    if (activeTab === "Original email") {
      content.append(node("h2", details.email.subject), node("p", "From: " + details.email.sender), node("p",details.email.received_at), node("pre",details.email.body || "(No text body)"));
      if (details.email.warnings?.length) content.append(node("pre",value(details.email.warnings)));
    } else if (activeTab === "Attachments") {
      const attachment = details.attachments.find(a => a.id === (selectedAttachment || advice.attachment_id));
      if (!attachment) return;
      const attachmentPicker = node("select");
      details.attachments.forEach(a => { const option = node("option",a.filename); option.value = a.id; attachmentPicker.append(option); });
      attachmentPicker.value = attachment.id; attachmentPicker.setAttribute("aria-label","Original attachment");
      attachmentPicker.onchange = () => {
        selectedAttachment = attachmentPicker.value;
        selectedAdvice = details.advices.find(a => a.attachment_id === selectedAttachment)?.id || selectedAdvice;
        renderRows().catch(error);
      };
      content.append(attachmentPicker);
      /** Fetch authenticated original bytes, never expose API credentials in URLs. */
      async function original() {
        const response = await fetch(API_BASE_URL + base + "/attachments/" + attachment.id, {headers:{"X-API-Key":API_KEY}});
        if (!response.ok) throw new Error("Attachment download failed");
        return response.blob();
      }
      content.append(button("Download original", async () => {
        const url = URL.createObjectURL(await original()); blobUrls.push(url);
        const a = node("a"); a.href = url; a.download = attachment.filename; a.click();
      },error));
      if (attachment.mime_type === "application/pdf" || attachment.mime_type.startsWith("image/")) {
        const blob = await original();
        const isPdf = attachment.mime_type === "application/pdf";
        // ponytail: forced PDF type replaces iframe sandbox; sandboxed frames block the browser PDF viewer
        const url = URL.createObjectURL(isPdf ? new Blob([blob], {type: "application/pdf"}) : blob);
        if (!content.isConnected) { URL.revokeObjectURL(url); return; }
        blobUrls.push(url);
        const preview = node(isPdf ? "iframe" : "img");
        preview.src = url;
        if (isPdf) preview.title = attachment.filename;
        else preview.alt = attachment.filename;
        content.append(preview);
      } else {
        const preview = await request(base + "/attachments/" + attachment.id + "?preview=true");
        content.append(node("pre",preview.text || "Preview unavailable. Download the original."));
      }
    } else if (activeTab === "S/4 posting") {
      renderS4Panel(content, advice, {post: (path, body) => updateAdvice(path, body), onError: error});
    } else {
      renderChat(content, advice);
    }
  }
  /** Append one chat message, rendering Markdown only for trusted assistant output. */
  function appendAdviceMessage(log, message) {
    const element = node("div", "", "advice-chat-message " + message.role);
    if (message.role === "assistant") renderSafeMarkdown(element, message.content);
    else element.textContent = message.content;
    log.append(element);
    return element;
  }
  /** Render per-advice ephemeral history and explicit rule-proposal confirmation cards. */
  function renderChat(content, advice) {
    content.className = "advice-chat";
    content.append(node("p","Context: " + advice.filename + ". Corrections persist; chat resets on reload."));
    const messages = conversations.get(advice.id) || []; conversations.set(advice.id,messages);
    const log = node("div", "", "advice-chat-log"); log.setAttribute("aria-live","polite");
    messages.forEach(message => appendAdviceMessage(log, message));
    const input = node("textarea"); input.rows = 3; input.maxLength = 10000; input.setAttribute("aria-label","Ask about or correct this payment advice");
    input.value = drafts.get(advice.id) || "";
    input.oninput = () => drafts.set(advice.id,input.value);
    input.placeholder = "8700838008CR should have code 321 because…";
    /** Send one turn bound to the revision displayed beside this composer. */
    async function send() {
      const message = input.value.trim(); if (!message || chatBusy.has(advice.id)) return;
      const history = messages.slice(-20); messages.push({role:"user",content:message}); input.value = ""; drafts.delete(advice.id);
      appendAdviceMessage(log,{role:"user",content:message}); chatBusy.add(advice.id);
      try {
        const reply = await request(adviceBase + advice.id + "/chat","POST",{revision:advice.revision,message,history});
        messages.push({role:"assistant",content:reply.reply});
        await refresh(true);
      } catch (e) { messages.push({role:"assistant",content:e.message}); await refresh(true); throw e; }
      finally { chatBusy.delete(advice.id); if (selectedAdvice === advice.id) await renderRows(); }
    }
    const submit = button(chatBusy.has(advice.id) ? "Working…" : "Send",send,error,"primary");
    submit.disabled = chatBusy.has(advice.id) || ["queued","processing"].includes(advice.status);
    content.append(log,input,submit);
    request(adviceBase + advice.id + "/proposals").then(items => {
      if (!content.isConnected) return;
      items.filter(p => p.status === "pending").forEach(p => {
        const card = node("details"); card.append(node("summary","Proposed customer rule change"),node("p",p.scope),node("p",p.reason),node("pre",p.playbook_text));
        card.append(button("Confirm this rule proposal",async () => {
          await request(adviceBase + advice.id + "/proposals/" + p.id + "/confirm","POST");
          await refresh(true);
        },error)); content.append(card);
      });
    }).catch(error);
  }
  let page = [];
  /** Render email rows, expanding the selected email inline with contained panels. */
  async function renderRows() {
    if (disposed) return;
    release();
    const rows = get("inbox-rows"); rows.replaceChildren();
    if (!page.length) { const tr = node("tr"), td = node("td","No emails yet. Add a manual email or fetch Gmail to begin."); td.colSpan = 7; tr.append(td); rows.append(tr); }
    for (const email of page) {
      const tr = node("tr");
      const customer = node("td");
      const toggle = button(customerName(email.client_key),async () => {
        selectedEmail = selectedEmail === email.id ? null : email.id; selectedAdvice = null; selectedAttachment = null;
        details = null; ++detailRequest;
        if (selectedEmail) await loadDetails();
        await renderRows();
      },error,"email-link");
      toggle.setAttribute("aria-expanded",String(selectedEmail === email.id));
      customer.append(toggle,node("small",email.sender,"email-sender")); tr.append(customer);
      [email.subject,(email.payment_references || []).join(", "),new Date(email.received_at).toLocaleString(),email.attachment_count,email.source].forEach(v => tr.append(node("td",value(v))));
      const status = node("td"); status.append(node("span",labels[email.status] || email.status,"status-pill status-" + email.status)); tr.append(status); rows.append(tr);
      if (selectedEmail === email.id && details?.email.id === email.id) {
        const expanded = node("tr","","expanded-email"), cell = node("td"); cell.colSpan = 7; expanded.append(cell); rows.append(expanded);
        // Build all rows synchronously; an older preview request must never resume
        // appending stale rows after a newer selection has replaced the table.
        workspace(cell).catch(error);
      }
    }
  }
  /** Poll persisted statuses without replacing an active chat composer on every tick. */
  function refresh(force = false) {
    if (disposed) return Promise.resolve();
    if (refreshing) { refreshQueued ||= force; return refreshing; }
    refreshing = (async () => {
    try {
      const requestedOffset = offset, requestedFilter = filter;
      const [response, connection] = await Promise.all([request(base + "/inbox?offset=" + requestedOffset + (requestedFilter ? "&status=" + requestedFilter : "")),request(base + "/status")]);
      if (!root.isConnected) return;
      if (requestedOffset !== offset || requestedFilter !== filter) { refreshQueued = true; return; }
      const rowsChanged = JSON.stringify(page) !== JSON.stringify(response.emails);
      page = response.emails;
      get("inbox-connection").textContent = connection.message + (connection.mailbox ? " · " + connection.mailbox : "") +
        " · Last successful fetch: " + (connection.last_successful_fetch ? new Date(connection.last_successful_fetch).toLocaleString() : "Never") +
        (connection.last_fetch ? " · Fetch " + connection.last_fetch.status + (connection.last_fetch.error ? ": " + connection.last_fetch.error : "") : "");
      get("gmail-fetch").disabled = !connection.configured || ["queued","processing"].includes(connection.last_fetch?.status);
      const counts = response.counts, stats = get("inbox-stats"); stats.replaceChildren();
      for (const [key,label] of [["","Total emails"],...Object.entries(labels)]) {
        const b = button(label,async () => { filter = key; offset = 0; await refresh(true); },error);
        b.append(node("strong",key ? counts[key] || 0 : Object.values(counts).reduce((a,b) => a+b,0)));
        b.setAttribute("aria-pressed",String(filter === key)); stats.append(b);
      }
      get("inbox-count").textContent = page.length + " emails shown";
      get("inbox-previous").disabled = offset === 0; get("inbox-next").disabled = page.length < 50;
      const pending = details?.advices.some(a => ["queued","processing"].includes(a.status));
      const detailChanged = selectedEmail && page.find(e => e.id === selectedEmail)?.revision !== details?.email.revision;
      if (selectedEmail && (force || pending || detailChanged)) await loadDetails();
      if (force || pending || rowsChanged || detailChanged) await renderRows();
    } catch (e) { error(e); }
    finally {
      refreshing = null;
      if (refreshQueued) { refreshQueued = false; await refresh(true); }
    }
    })();
    return refreshing;
  }
  get("manual-open").onclick = () => get("manual-dialog").showModal();
  get("manual-cancel").onclick = () => get("manual-dialog").close();
  get("manual-form").onsubmit = async event => {
    event.preventDefault();
    const submit = event.submitter; submit.disabled = true; get("manual-error").textContent = "";
    try {
      const form = new FormData(event.target);
      if (!form.getAll("files").some(f => f.size)) form.delete("files");
      const result = await upload(base + "/manual",form);
      get("manual-dialog").close(); event.target.reset(); selectedEmail = result.email_id; selectedAdvice = null; selectedAttachment = null; details = null; ++detailRequest; offset = 0; filter = "";
      await refresh(true);
    } catch (e) { get("manual-error").textContent = e.message; } finally { submit.disabled = false; }
  };
  get("gmail-fetch").onclick = async () => { try { get("gmail-fetch").disabled = true; await request(base + "/fetch","POST"); await refresh(true); } catch(e) {error(e);get("gmail-fetch").disabled=false;} };
  get("inbox-previous").onclick = () => { offset = Math.max(0,offset-50); refresh(true); };
  get("inbox-next").onclick = () => { offset += 50; refresh(true); };
  /** Dispose only once navigation actually removes this instance. */
  function onPageChange() {
    if (!root.isConnected) { disposed = true; clearInterval(timer); release(); window.removeEventListener("pageChanged",onPageChange); }
  }
  window.addEventListener("pageChanged",onPageChange);
  request("/api/payment-advice/customers").then(response => {
    customers = Array.isArray(response) ? response : response.customers || [];
    get("manual-customer").replaceWith(Object.assign(customerSelect("",true),{id:"manual-customer",name:"customer"}));
    return refresh(true);
  }).catch(error);
  refresh(true);
  timer = setInterval(() => refresh(),4000);
}
