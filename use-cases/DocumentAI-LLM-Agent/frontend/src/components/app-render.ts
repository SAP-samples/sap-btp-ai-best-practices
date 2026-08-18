import type { AppState, MethodScore } from "../api/types.js";
import { scoreClass, escapeHtml } from "../utils/formatters.js";

export function invoiceResultsHtml(state: AppState): string {
  const r = state.invoiceResult;
  if (!r) return "";
  const ext = (r.result?.extraction ?? r.result?.document ?? {}) as { headerFields?: Array<{name?:string;value?:unknown;rawValue?:string;confidence?:number}>; lineItems?: Array<Record<string,unknown>> };
  const hf = ext.headerFields ?? [];
  const li = ext.lineItems ?? [];
  const rows = hf.map((f) => {
    const pct = f.confidence!=null?Math.round(f.confidence*100):null;
    const cls = f.confidence!=null?(f.confidence>=0.8?"high":f.confidence>=0.5?"medium":"low"):"low";
    return `<tr>
      <td><span class="field-name">${escapeHtml(f.name??"—")}</span></td>
      <td><span class="field-value">${escapeHtml(String(f.value??f.rawValue??"—"))}</span></td>
      <td>${pct!=null?`<div class="confidence-bar"><div class="confidence-track"><div class="confidence-fill ${cls}" style="width:${pct}%"></div></div><span class="confidence-text">${pct}%</span></div>`:"—"}</td>
    </tr>`;
  }).join("");
  const liRows = li.map((item,i) => `<tr><td>${i+1}</td><td>${escapeHtml(String(item["description"]??"—"))}</td><td>${escapeHtml(String(item["quantity"]??"—"))}</td><td>${escapeHtml(String(item["unitPrice"]??"—"))}</td><td>${escapeHtml(String(item["netAmount"]??"—"))}</td></tr>`).join("");
  return `<div class="fade-in" style="margin-top:1rem;">
    <ui5-card>
      <ui5-card-header slot="header" title-text="SAP Document AI Extraction" subtitle-text="Invoice processing result"></ui5-card-header>
      <div style="padding:1rem;">
        <div class="method-header"><span class="method-title">🔵 SAP Document AI</span><div class="method-stats"><span class="method-stat">📊 ${hf.length} fields</span><span class="method-stat">📦 ${li.length} line items</span></div></div>
        ${hf.length>0?`<div style="overflow-x:auto;"><table class="field-table"><thead><tr><th>Field</th><th>Value</th><th>Confidence</th></tr></thead><tbody>${rows}</tbody></table></div>`:`<div style="padding:2rem;text-align:center;color:var(--sapContent_LabelColor,#6a6d70);">No fields extracted</div>`}
        ${li.length>0?`<div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">📦 Line Items</div><div style="overflow-x:auto;"><table class="line-items-table"><thead><tr><th>#</th><th>Description</th><th>Qty</th><th>Unit Price</th><th>Net Amount</th></tr></thead><tbody>${liRows}</tbody></table></div></div>`:""}
      </div>
    </ui5-card>
  </div>`;
}

export function schemasHtml(state: AppState): string {
  const r = state.schemasResult;
  if (!r) return "";
  const items = r.schemas ?? r.value ?? (Array.isArray(r) ? r as unknown[] : []);
  return `<div class="fade-in" style="margin-top:1rem;">
    <ui5-card>
      <ui5-card-header slot="header" title-text="SAP Document AI Schemas" subtitle-text="${items.length} schema(s) available"></ui5-card-header>
      <div style="padding:1rem;">
        ${items.length>0?`<div style="overflow-x:auto;"><table class="field-table"><thead><tr><th>ID</th><th>Name</th><th>Document Type</th><th>Status</th></tr></thead><tbody>
          ${items.map((item) => {
            const s = item as Record<string,unknown>;
            return `<tr><td><span class="field-name">${escapeHtml(String(s["id"]??"—"))}</span></td><td>${escapeHtml(String(s["name"]??"—"))}</td><td>${escapeHtml(String(s["documentType"]??"—"))}</td><td><span class="status-badge ${s["state"]==="ACTIVE"?"success":"info"}">${escapeHtml(String(s["state"]??s["status"]??"—"))}</span></td></tr>`;
          }).join("")}
        </tbody></table></div>`:`<div style="padding:2rem;text-align:center;color:var(--sapContent_LabelColor,#6a6d70);">No schemas found</div>`}
        <div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">📄 Raw Response</div><div class="json-viewer">${escapeHtml(JSON.stringify(r,null,2))}</div></div>
      </div>
    </ui5-card>
  </div>`;
}

export function templatesHtml(state: AppState): string {
  const r = state.templatesResult;
  if (!r) return "";
  const items = r.results ?? r.templates ?? r.value ?? (Array.isArray(r) ? r as unknown[] : []);
  return `<div class="fade-in" style="margin-top:1rem;">
    <ui5-card>
      <ui5-card-header slot="header" title-text="SAP Document AI Templates" subtitle-text="${items.length} template(s) available"></ui5-card-header>
      <div style="padding:1rem;">
        ${items.length>0?`<div style="overflow-x:auto;"><table class="field-table"><thead><tr><th>ID</th><th>Name</th><th>Document Type</th><th>Status</th></tr></thead><tbody>
          ${items.map((item) => {
            const t = item as Record<string,unknown>;
            return `<tr><td><span class="field-name">${escapeHtml(String(t["id"]??"—"))}</span></td><td>${escapeHtml(String(t["name"]??"—"))}</td><td>${escapeHtml(String(t["documentType"]??"—"))}</td><td><span class="status-badge ${t["state"]==="ACTIVE"?"success":"info"}">${escapeHtml(String(t["state"]??t["status"]??"—"))}</span></td></tr>`;
          }).join("")}
        </tbody></table></div>`:`<div style="padding:2rem;text-align:center;color:var(--sapContent_LabelColor,#6a6d70);">No templates found</div>`}
        <div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">📄 Raw Response</div><div class="json-viewer">${escapeHtml(JSON.stringify(r,null,2))}</div></div>
      </div>
    </ui5-card>
  </div>`;
}

export function evalResultsCard(state: AppState): string {
  const ev = state.evaluationResult;
  if (!ev) return "";
  const scores = ev.scores ?? {};
  const llmEval = ev.llm_evaluation ?? {};
  const methods = [
    { key:"sap", label:"SAP Document AI", emoji:"🔵" },
    { key:"llm_prompting", label:"LLM Technique 1", emoji:"🟣" },
    { key:"llm_structured", label:"LLM Technique 2", emoji:"🟢" },
  ];
  const scoreCards = methods.map((m) => {
    const sc = (scores as Record<string,MethodScore>)[m.key] ?? {};
    const raw = sc.overall_score ?? sc.field_coverage ?? 0;
    const pct = Math.round(raw <= 1 ? raw*100 : raw);
    const cls = scoreClass(pct);
    return `<div class="kpi-card ${cls}">
      <div style="font-size:1.25rem;margin-bottom:0.25rem;">${m.emoji}</div>
      <div class="kpi-value">${pct}%</div>
      <div class="kpi-label">${m.label}</div>
      <div style="font-size:0.75rem;color:var(--sapContent_LabelColor,#6a6d70);margin-top:0.25rem;">${sc.fields_found??0} fields found</div>
    </div>`;
  }).join("");
  const metricRows = methods.map((m) => {
    const sc = (scores as Record<string,MethodScore>)[m.key] ?? {};
    const confPct = sc.confidence_avg!=null?Math.round(sc.confidence_avg*100)+"%":"—";
    const compPct = sc.completeness!=null?Math.round(sc.completeness*100)+"%":"—";
    return `<tr>
      <td>${m.emoji} ${m.label}</td>
      <td>${sc.fields_found??0}</td>
      <td>${sc.missing_fields??0}</td>
      <td>${confPct}</td>
      <td>${compPct}</td>
    </tr>`;
  }).join("");
  const recs = llmEval.recommendations ?? [];
  const bestMethod = String(llmEval.best_method ?? "");
  const assessment = String(llmEval.overall_assessment ?? "");
  const summary = String(ev.summary ?? "");
  return `<div class="eval-section fade-in">
    <ui5-card>
      <ui5-card-header slot="header" title-text="Extraction Quality Evaluation" subtitle-text="AI-powered quality assessment of all extraction methods"></ui5-card-header>
      <div style="padding:1rem;">
        ${bestMethod?`<div style="margin-bottom:1rem;padding:0.75rem 1rem;background:var(--sapSuccessBackground,#f1fdf6);border:1px solid var(--sapSuccessBorderColor,#107e3e);border-radius:0.375rem;"><span style="font-weight:700;color:var(--sapSuccessColor,#107e3e);">🏆 Best Method: ${escapeHtml(bestMethod)}</span></div>`:""}
        <div class="kpi-grid">${scoreCards}</div>
        <div style="margin-top:1.5rem;">
          <div class="section-title" style="margin-bottom:0.75rem;">📊 Detailed Metrics</div>
          <div style="overflow-x:auto;"><table class="comparison-table">
            <thead><tr><th>Method</th><th>Fields Found</th><th>Missing</th><th>Confidence Avg</th><th>Completeness</th></tr></thead>
            <tbody>${metricRows}</tbody>
          </table></div>
        </div>
        ${assessment?`<div style="margin-top:1.5rem;"><div class="section-title" style="margin-bottom:0.75rem;">🤖 AI Assessment</div><div class="summary-text">${escapeHtml(assessment)}</div></div>`:""}
        ${recs.length>0?`<div style="margin-top:1.5rem;"><div class="section-title" style="margin-bottom:0.75rem;">💡 Recommendations</div><ul style="padding-left:1.5rem;display:flex;flex-direction:column;gap:0.5rem;">${recs.map((r)=>`<li style="font-size:0.875rem;color:var(--sapTextColor,#32363a);">${escapeHtml(String(r))}</li>`).join("")}</ul></div>`:""}
        ${summary?`<div style="margin-top:1.5rem;"><div class="section-title" style="margin-bottom:0.75rem;">📋 Summary Report</div><div class="summary-text">${escapeHtml(summary)}</div></div>`:""}
      </div>
    </ui5-card>
  </div>`;
}