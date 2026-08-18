import type { AppState, ProcessingStep, GenAIPipelineResponse, HeaderField, LineItem, LlmExtractionResult, ComparisonResult, PostInvoiceRequest, PostInvoiceResponse, PostPOInvoiceRequest, PostPOInvoiceResponse, PurchaseOrderResult, ChatMessage, SOLineItem, ExtractedPurchaseOrder, SOValidationResult, CreateSORequest, CreateSOResponse, ExtractedPaymentAdvice, PostPaymentAdviceRequest, PostPaymentAdviceResponse } from "../api/types.js";
import { api } from "../api/client.js";
import { confidenceClass, escapeHtml, truncate } from "../utils/formatters.js";
import { invoiceResultsHtml, schemasHtml, templatesHtml, evalResultsCard } from "./app-render.js";
import { renderDocAiNewApp } from "./docai-new-app.js";
import { renderTrainTemplateApp } from "./train-template-app.js";

const INVOICE_FIELDS = ["documentNumber","documentDate","grossAmount","netAmount","taxAmount","taxRate","currencyCode","senderName","receiverName","purchaseOrderNumber","deliveryDate","senderAddress","receiverAddress","senderBankAccount","taxId","receiverContact","senderCity","senderStreet","senderPostalCode","senderCountry","receiverCity","receiverStreet","receiverPostalCode","receiverCountry"];

// Top-level view: "legacy" = existing DOC AI, "docai-new" = DOC AI NEW, "train" = Train Template
type TopView = "legacy" | "docai-new" | "train";

export class App {
  private root: HTMLElement;
  private state: AppState;
  private topView: TopView = "legacy";

  constructor(root: HTMLElement) {
    this.root = root;
    this.state = {
      scenario: "genai",
      selectedFile: null,
      selectedFiles: [],
      processingStatus: "idle",
      processingSteps: this.buildSteps(),
      pipelineResult: null,
      invoiceResult: null,
      schemasResult: null,
      templatesResult: null,
      evaluationResult: null,
      fiPostResult: null,
      soResult: null,
      paResult: null,
      errorMessage: null,
      apiHealthy: false,
      chatMessages: [],
      streamingText: "",
    };
  }

  async init(): Promise<void> {
    this.render();
    try { await api.health(); this.state.apiHealthy = true; } catch { this.state.apiHealthy = false; }
    this.updateHealthIndicator();
  }

  private buildSteps(): ProcessingStep[] {
    return [
      { id:"sap", label:"SAP Document AI Extraction", description:"Submitting document and polling for results…", status:"pending" },
      { id:"llm1", label:"LLM Technique 1 — Free Prompting", description:"Multimodal extraction with free prompting…", status:"pending" },
      { id:"llm2", label:"LLM Technique 2 — Structured JSON", description:"Multimodal extraction with structured JSON schema…", status:"pending" },
      { id:"compare", label:"Comparison & Analysis", description:"Comparing results from all three methods…", status:"pending" },
    ];
  }

  private buildTemplateSteps(templateName: string): ProcessingStep[] {
    return [
      { id:"sap", label:"Extracting Data", description:`Processing with template "${templateName}"…`, status:"pending" },
      { id:"done", label:"Done", description:"Extraction complete.", status:"pending" },
    ];
  }

  private buildSOSteps(): ProcessingStep[] {
    return [
      { id:"extract", label:"Extract Customer PO", description:"Submitting document to SAP DocAI for extraction…", status:"pending" },
      { id:"validate", label:"Validate Against S4", description:"Resolving customer BP and validating materials in S4…", status:"pending" },
    ];
  }

  private setStep(id: string, status: ProcessingStep["status"]): void {
    const s = this.state.processingSteps.find((x) => x.id === id);
    if (s) s.status = status;
    const c = document.getElementById("steps-container");
    if (c) c.innerHTML = this.state.processingSteps.map((x,i) => this.stepHtml(x,i)).join("");
  }

  private updateHealthIndicator(): void {
    const el = document.getElementById("health-item");
    if (el) { el.setAttribute("icon", this.state.apiHealthy?"connected":"disconnected"); el.setAttribute("text", this.state.apiHealthy?"API Online":"API Offline"); }
  }

  private statusText(): string {
    const m: Record<string,string> = { idle:"Ready to execute", uploading:"Uploading file…", processing:"Pipeline running…", completed:"Completed successfully", error:"Error occurred" };
    return m[this.state.processingStatus] ?? "";
  }

  render(): void {
    if (this.topView === "docai-new") {
      this.root.innerHTML = this.buildShellOnly("DOC AI NEW");
      this.bindShellEvents();
      const content = this.root.querySelector<HTMLDivElement>("#main-content")!;
      renderDocAiNewApp(content);
      return;
    }
    if (this.topView === "train") {
      this.root.innerHTML = this.buildShellOnly("Train Template");
      this.bindShellEvents();
      const content = this.root.querySelector<HTMLDivElement>("#main-content")!;
      renderTrainTemplateApp(content);
      return;
    }
    this.root.innerHTML = this.buildHtml();
    this.bindEvents();
    // Scroll assistant messages to bottom
    const scroll = this.root.querySelector(".assistant-scroll");
    if (scroll) scroll.scrollTop = scroll.scrollHeight;
  }

  private buildShellOnly(activeLabel: string): string {
    return (
      `<div style="display:flex;flex-direction:column;height:100%;overflow:hidden;">` +
      `<ui5-shellbar id="shellbar" primary-title="AI4U - Document AI Agent" secondary-title="${activeLabel}" logo-area-interactive style="flex-shrink:0;">` +
      `<img slot="logo" src="/logo.png" alt="SAP" style="height:2rem;cursor:pointer;" />` +
      `<ui5-shellbar-item id="nav-back" icon="nav-back" text="Volver" slot="items"></ui5-shellbar-item>` +
      `<ui5-shellbar-item id="nav-legacy" icon="home" text="DOC AI" slot="items"></ui5-shellbar-item>` +
      `<ui5-shellbar-item id="nav-docai-new" icon="add-document" text="DOC AI NEW" slot="items"></ui5-shellbar-item>` +
      `<ui5-shellbar-item id="nav-train" icon="learning-assistant" text="Train Template" slot="items"></ui5-shellbar-item>` +
      `</ui5-shellbar>` +
      `<div class="fiori-content" id="main-content" style="overflow-y:auto;"></div>` +
      `</div>`
    );
  }

  private bindShellEvents(): void {
    const goHome = () => { this.topView = "legacy"; this.render(); };
    this.root.querySelector("#nav-back")?.addEventListener("click", goHome);
    this.root.querySelector("#nav-legacy")?.addEventListener("click", goHome);
    this.root.querySelector("#nav-docai-new")?.addEventListener("click", () => { this.topView = "docai-new"; this.render(); });
    this.root.querySelector("#nav-train")?.addEventListener("click", () => { this.topView = "train"; this.render(); });
    this.root.querySelector("#shellbar")?.addEventListener("logo-click", goHome);
    this.root.querySelector("#shellbar")?.addEventListener("logoClick", goHome);
  }

  private stepHtml(step: ProcessingStep, index: number): string {
    const cls = step.status==="pending"?"":step.status;
    const icon = step.status==="completed"?"✓":step.status==="error"?"✗":step.status==="active"?"⟳":String(index+1);
    return `<div class="step-item ${cls}" id="step-${step.id}"><div class="step-number">${icon}</div><div class="step-content"><div class="step-title">${step.label}</div><div class="step-subtitle">${step.description}</div></div>${step.status==="active"?`<ui5-busy-indicator active size="Small"></ui5-busy-indicator>`:""}</div>`;
  }

  // ─── LEFT PANEL ─────────────────────────────────────────────────────────────

  private buildUploadPanel(): string {
    const isProc = this.state.processingStatus === "processing";
    const isDone = this.state.processingStatus === "completed";
    const needsFile = this.state.scenario === "genai" || this.state.scenario === "sales-order";
    const scenarios = [
      { id:"genai", label:"GenAI Invoice Pipeline", desc:"SAP DocAI + LLM Technique 1 + LLM Technique 2 + Comparison" },
      { id:"evaluation", label:"Evaluate Extraction Quality", desc:"Run quality evaluation on last pipeline results" },
      { id:"doc-ai-new", label:"DOC AI NEW", desc:"Automatic template discovery, creation and training pipeline" },
      { id:"train-template", label:"Train Template", desc:"Train an existing SAP Document AI template with PDFs" },
      { id:"sales-order", label:"Sales Order Process", desc:"Extract customer PO via SAP DocAI → validate against S4 → create Sales Order" },
    ];
    const scenarioItems = scenarios.map((s) =>
      `<label class="scenario-item ${this.state.scenario===s.id?"selected":""}">` +
      `<input type="radio" name="scenario" value="${s.id}" ${this.state.scenario===s.id?"checked":""} ${isProc?"disabled":""} />` +
      `<div><div class="scenario-label">${s.label}</div><div class="scenario-desc">${s.desc}</div></div>` +
      `</label>`
    ).join("");

    const fileSection = needsFile
      ? `<div class="upload-zone ${this.state.selectedFile?"has-file":""}" id="upload-zone">` +
        `<input type="file" id="file-input" accept=".pdf,.jpg,.jpeg,.png,.tif,.tiff" ${isProc?"disabled":""} />` +
        `<div class="upload-icon">📄</div>` +
        `<div class="upload-title">${this.state.selectedFile?"File Selected":"Drop file here or click to browse"}</div>` +
        `<div class="upload-subtitle">PDF, JPG, PNG, TIFF</div>` +
        `${this.state.selectedFile?`<div class="upload-filename">✓ ${escapeHtml(this.state.selectedFile.name)}</div>`:""}` +
        `</div>`
      : "";

    const errorStrip = this.state.errorMessage
      ? `<ui5-message-strip design="Negative" style="margin-top:0.75rem;display:block;">${escapeHtml(this.state.errorMessage)}</ui5-message-strip>`
      : "";

    const stepsSection = isProc
      ? `<div class="steps-container" id="steps-container">${this.state.processingSteps.map((s,i) => this.stepHtml(s,i)).join("")}</div>`
      : `<div id="steps-container" class="hidden"></div>`;

    const busyIndicator = isProc
      ? `<div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;"><ui5-busy-indicator active size="Small"></ui5-busy-indicator><span style="font-size:0.8125rem;color:var(--sapInformationColor,#0070f2);">Pipeline running…</span></div>`
      : "";

    const resetBtn = isDone
      ? `<ui5-button id="reset-btn" design="Transparent" icon="refresh">New Execution</ui5-button>`
      : "";

    return `
      <aside class="workspace-panel upload-panel" aria-labelledby="upload-title">
        <header class="panel-heading">
          <div>
            <div class="panel-eyebrow">1 · Upload</div>
            <h1 id="upload-title">Document &amp; Pipeline</h1>
          </div>
        </header>
        <div class="panel-body upload-body">
          <div class="scenario-group">${scenarioItems}</div>
          ${fileSection}
          ${stepsSection}
          ${busyIndicator}
          ${errorStrip}
        </div>
        <footer class="panel-footer">
          <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <ui5-button id="execute-btn" design="Emphasized" icon="play" ${isProc?"disabled":""} style="flex:1;min-width:120px;">${isProc?"Processing…":"Execute Pipeline"}</ui5-button>
            ${resetBtn}
          </div>
          <div style="font-size:0.75rem;color:var(--sapContent_LabelColor,#6a6d70);margin-top:0.375rem;">${this.statusText()}</div>
        </footer>
      </aside>`;
  }

  // ─── CENTER PANEL ────────────────────────────────────────────────────────────

  private buildReviewPanel(): string {
    const isProc = this.state.processingStatus === "processing";
    const isDone = this.state.processingStatus === "completed";
    const isError = this.state.processingStatus === "error";

    let reviewContent: string;
    if (isProc) {
      reviewContent = `<div class="workspace-empty" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;color:var(--sapContent_LabelColor,#6a6d70);">
        <ui5-busy-indicator active size="Large"></ui5-busy-indicator>
        <p style="font-size:0.9375rem;">Pipeline running, please wait…</p>
      </div>`;
    } else if (isError) {
      reviewContent = `<div class="workspace-empty" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;">
        <div style="font-size:3rem;">✗</div>
        <p style="font-size:1rem;font-weight:600;color:var(--sapErrorColor,#bb0000);">Pipeline Error</p>
        <p style="font-size:0.875rem;color:var(--sapContent_LabelColor,#6a6d70);">${escapeHtml(this.state.errorMessage ?? "An unexpected error occurred.")}</p>
      </div>`;
    } else if (isDone && this.state.pipelineResult) {
      const r = this.state.pipelineResult;
      if (r.document_type === "purchase_order") {
        const po = r.extracted_po as ExtractedPurchaseOrder | null;
        const validation = r.so_validation as SOValidationResult | null;
        reviewContent = po
          ? this.soResultHtml(po, validation)
          : `<div class="workspace-empty" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;color:var(--sapContent_LabelColor,#6a6d70);"><div style="font-size:3rem;opacity:0.3;">📦</div><p>Customer PO detected — extraction in progress.</p></div>`;
      } else if (r.document_type === "payment_advice") {
        const pa = r.extracted_pa as ExtractedPaymentAdvice | null;
        reviewContent = pa
          ? this.paResultHtml(pa)
          : `<div class="workspace-empty" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;color:var(--sapContent_LabelColor,#6a6d70);"><div style="font-size:3rem;opacity:0.3;">💳</div><p>Payment Advice detected — extraction in progress.</p></div>`;
      } else {
        reviewContent = this.pipelineResultsHtml(r);
      }
    } else if (isDone && this.state.invoiceResult) {
      reviewContent = invoiceResultsHtml(this.state);
    } else if (isDone && this.state.schemasResult) {
      reviewContent = schemasHtml(this.state);
    } else if (isDone && this.state.templatesResult) {
      reviewContent = templatesHtml(this.state);
    } else if (isDone && this.state.evaluationResult) {
      reviewContent = evalResultsCard(this.state);
    } else if (isDone && this.state.scenario === "sales-order") {
      const po = (this.state as unknown as Record<string, unknown>)["soExtracted"] as ExtractedPurchaseOrder | null;
      const validation = (this.state as unknown as Record<string, unknown>)["soValidation"] as SOValidationResult | null;
      reviewContent = po ? this.soResultHtml(po, validation) : `<div class="workspace-empty" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;color:var(--sapContent_LabelColor,#6a6d70);"><div style="font-size:3rem;opacity:0.3;">📄</div><p>No extraction result.</p></div>`;
    } else {
      reviewContent = `<div class="workspace-empty" style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;color:var(--sapContent_LabelColor,#6a6d70);">
        <div style="font-size:3rem;opacity:0.3;">📄</div>
        <p style="font-size:1rem;font-weight:600;">No Results Yet</p>
        <p style="font-size:0.875rem;">Select a pipeline, upload a document and click Execute.</p>
      </div>`;
    }

    // Footer: eval + FI post buttons (only shown when pipeline result exists)
    let footerContent = "";
    if (isDone && this.state.pipelineResult) {
      const r = this.state.pipelineResult;

      if (r.document_type === "purchase_order") {
        // PO footer — Create Sales Order
        const validation = r.so_validation as SOValidationResult | null;
        const soResultLocal = this.state.soResult;
        if (soResultLocal) {
          const resultHtml = soResultLocal.success
            ? `<span style="font-size:0.8125rem;color:var(--sapSuccessColor,#107e3e);">✓ Sales Order: ${escapeHtml(soResultLocal.sales_order)} — ${soResultLocal.items_created} item(s)</span>`
            : `<span style="font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);">Error: ${escapeHtml(soResultLocal.error)}</span>`;
          footerContent = `<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">${resultHtml}</div>`;
        } else {
          footerContent = `
            <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
              <ui5-button id="btn-create-so" design="Emphasized" icon="add-document">POST S4 — Create Sales Order</ui5-button>
              <span id="so-create-status" style="font-size:0.8125rem;color:var(--sapNeutralColor,#6a6d70);">${validation?.ready_to_create ? "Ready to create" : validation ? "Review issues before posting" : ""}</span>
            </div>`;
        }
      } else if (r.document_type === "payment_advice") {
        // Payment Advice footer
        const paResultLocal = this.state.paResult;
        if (paResultLocal) {
          const resultHtml = paResultLocal.success
            ? `<span style="font-size:0.8125rem;color:var(--sapSuccessColor,#107e3e);">✓ Payment Advice: ${escapeHtml(paResultLocal.payment_advice)} — ${escapeHtml(paResultLocal.payer_name_matched)}</span>`
            : `<span style="font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);">Error: ${escapeHtml(paResultLocal.error)}</span>`;
          footerContent = `<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">${resultHtml}</div>`;
        } else {
          footerContent = `
            <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
              <ui5-button id="btn-post-fi" design="Emphasized" icon="post">POST S4 — Payment Advice</ui5-button>
              <span id="fi-post-status" style="font-size:0.8125rem;color:var(--sapNeutralColor,#6a6d70);"></span>
            </div>`;
        }
      } else {
        // Invoice footer — POST S4 (FI / MM)
        const fiResultHtml = this.state.fiPostResult
          ? (this.state.fiPostResult.success
              ? `<span style="font-size:0.8125rem;color:var(--sapSuccessColor,#107e3e);">FI Doc: ${escapeHtml(this.state.fiPostResult.fi_document)}</span>`
              : `<span style="font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);">Error: ${escapeHtml(this.state.fiPostResult.error)}</span>`)
          : `<span id="fi-post-status" style="font-size:0.8125rem;color:var(--sapNeutralColor,#6a6d70);"></span>`;
        footerContent = `
          <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <ui5-button id="btn-post-fi" design="Emphasized" icon="post">POST S4</ui5-button>
            ${fiResultHtml}
          </div>`;
      }
    } else if (isDone && this.state.scenario === "sales-order") {
      const validation = (this.state as unknown as Record<string, unknown>)["soValidation"] as SOValidationResult | null;
      const soResultLocal = this.state.soResult;
      if (soResultLocal) {
        const resultHtml = soResultLocal.success
          ? `<span style="font-size:0.8125rem;color:var(--sapSuccessColor,#107e3e);">Sales Order: ${escapeHtml(soResultLocal.sales_order)}</span>`
          : `<span style="font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);">Error: ${escapeHtml(soResultLocal.error)}</span>`;
        footerContent = `<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">${resultHtml}</div>`;
      } else if (validation?.ready_to_create) {
        footerContent = `
          <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
            <ui5-button id="btn-create-so" design="Emphasized" icon="add-document">Create Sales Order</ui5-button>
            <span id="so-create-status" style="font-size:0.8125rem;color:var(--sapNeutralColor,#6a6d70);"></span>
          </div>`;
      }
    } else if (isDone && !this.state.evaluationResult && this.state.pipelineResult?.route !== "template") {
      footerContent = `<ui5-button id="evaluate-btn" design="Default" icon="ai">Evaluate Quality</ui5-button>`;
    }

    return `
      <section class="workspace-panel review-panel" aria-labelledby="review-title">
        <header class="panel-heading review-heading">
          <div>
            <div class="panel-eyebrow">2 · Results</div>
            <h1 id="review-title">Extraction Review</h1>
          </div>
        </header>
        <div id="review-content" class="panel-body review-content">
          ${reviewContent}
        </div>
        ${footerContent ? `<footer class="panel-footer review-actions">${footerContent}</footer>` : ""}
      </section>`;
  }

  // ─── RIGHT PANEL (ASSISTANT) ─────────────────────────────────────────────────

  private buildAssistantPanel(): string {
    const messagesHtml = this.buildMessagesHtml();
    return `
      <aside class="workspace-panel assistant-panel" aria-labelledby="assistant-title">
        <header class="panel-heading assistant-heading">
          <div style="display:flex;align-items:center;gap:0.75rem;">
            <div>
              <div class="panel-eyebrow">3 · Assistant</div>
              <h1 id="assistant-title">DocAI Assistant</h1>
            </div>
          </div>
          <div class="online-indicator ${this.state.apiHealthy?"online":""}">
            <span></span>${this.state.apiHealthy?"Online":"Offline"}
          </div>
        </header>
        <div class="assistant-scroll">
          <div id="assistant-messages" class="assistant-messages">
            ${messagesHtml}
          </div>
        </div>
        <footer class="chat-composer">
          <label class="sr-only" for="chat-input">Message</label>
          <textarea id="chat-input" rows="2" placeholder="Ask about the extraction results…"></textarea>
          <ui5-button id="chat-send" design="Emphasized" icon="paper-plane" accessible-name="Send"></ui5-button>
        </footer>
      </aside>`;
  }

  private buildMessagesHtml(): string {
    const msgs = this.state.chatMessages;
    if (!msgs.length && !this.state.streamingText) {
      return `<div class="assistant-empty">
        <div style="font-size:2.5rem;opacity:0.4;margin-bottom:0.75rem;">🤖</div>
        <strong>DocAI Assistant</strong>
        <span>Ask about extraction results, schemas, or FI posting.</span>
      </div>`;
    }
    const html = msgs.map((msg) => {
      const isUser = msg.role === "user";
      return `<article class="chat-message ${isUser?"user":""}">
        <span class="chat-author">${isUser?"You":"Assistant"}</span>
        <div>${escapeHtml(msg.content)}</div>
      </article>`;
    });
    if (this.state.streamingText) {
      html.push(`<article class="chat-message streaming">
        <span class="chat-author">Assistant</span>
        <div>${escapeHtml(this.state.streamingText)}</div>
      </article>`);
    }
    return html.join("");
  }

  // ─── MAIN BUILD ─────────────────────────────────────────────────────────────

  private buildHtml(): string {
    const parts: string[] = [];
    parts.push(`<div style="display:flex;flex-direction:column;height:100%;overflow:hidden;">`);
    parts.push(
      `<ui5-shellbar id="shellbar" primary-title="AI4U - Document AI Agent" style="flex-shrink:0;">` +
      `<img slot="logo" src="/logo.png" alt="SAP" style="height:2rem;" />` +
      `<ui5-shellbar-item id="nav-legacy" icon="home" text="DOC AI" slot="items"></ui5-shellbar-item>` +
      `<ui5-shellbar-item id="nav-docai-new" icon="add-document" text="DOC AI NEW" slot="items"></ui5-shellbar-item>` +
      `<ui5-shellbar-item id="nav-train" icon="learning-assistant" text="Train Template" slot="items"></ui5-shellbar-item>` +
      `<ui5-shellbar-item id="health-item" icon="${this.state.apiHealthy?"connected":"disconnected"}" text="${this.state.apiHealthy?"API Online":"API Offline"}" slot="items"></ui5-shellbar-item>` +
      `</ui5-shellbar>`
    );
    parts.push(`<div class="workspace-container">`);
    parts.push(`<div class="workspace-grid">`);
    parts.push(this.buildUploadPanel());
    parts.push(this.buildReviewPanel());
    parts.push(this.buildAssistantPanel());
    parts.push(`</div>`);
    parts.push(`</div>`);
    parts.push(`</div>`);
    parts.push(`<div id="toast-container" style="position:fixed;bottom:2rem;right:2rem;z-index:9999;display:flex;flex-direction:column;gap:0.5rem;pointer-events:none;"></div>`);
    return parts.join("");
  }

  private pipelineResultsHtml(r: GenAIPipelineResponse): string {
    if (r.route === "template") {
      const tName = r.routing_decision?.template_name ?? "Template";
      const sapData = r.template_result ?? r.sap_result;
      const evalHtml = this.state.evaluationResult
        ? evalResultsCard(this.state)
        : `<div class="eval-section"><div class="eval-cta"><div class="eval-cta-title">Evaluate Extraction Quality</div><div class="eval-cta-desc">Run an AI-powered quality evaluation.</div><ui5-button id="evaluate-btn" design="Emphasized" icon="ai" style="min-width:200px;">Evaluate Result?</ui5-button></div></div>`;
      return `<div class="fade-in" style="padding:1rem;">
        <div class="results-tabs" id="results-tabs">
          <button class="results-tab active" data-tab="sap">SAP Document AI</button>
        </div>
        <div id="tab-sap" class="tab-panel">${this.sapCard(sapData, tName)}</div>
        ${evalHtml}
      </div>`;
    }
    const evalHtml = this.state.evaluationResult
      ? evalResultsCard(this.state)
      : `<div class="eval-section"><div class="eval-cta"><div class="eval-cta-title">Evaluate Extraction Quality</div><div class="eval-cta-desc">Run an AI-powered quality evaluation comparing all three extraction methods.</div><ui5-button id="evaluate-btn" design="Emphasized" icon="ai" style="min-width:200px;">Evaluate Result?</ui5-button></div></div>`;
    return `<div class="fade-in" style="padding:1rem;">
      <div class="results-tabs" id="results-tabs">
        <button class="results-tab active" data-tab="sap">SAP Document AI</button>
        <button class="results-tab" data-tab="llm1">LLM Technique 1</button>
        <button class="results-tab" data-tab="llm2">LLM Technique 2</button>
        <button class="results-tab" data-tab="comparison">Comparison</button>
        <button class="results-tab" data-tab="summary">Summary</button>
      </div>
      <div id="tab-sap" class="tab-panel">${this.sapCard(r.sap_result)}</div>
      <div id="tab-llm1" class="tab-panel hidden">${this.llmCard(r.llm_prompting,"LLM Technique 1 — Free Prompting","")}</div>
      <div id="tab-llm2" class="tab-panel hidden">${this.llmCard(r.llm_structured,"LLM Technique 2 — Structured JSON","")}</div>
      <div id="tab-comparison" class="tab-panel hidden">${this.comparisonCard(r.comparison)}</div>
      <div id="tab-summary" class="tab-panel hidden">${this.summaryCard(r)}</div>
      ${evalHtml}
    </div>`;
  }

  private sapCard(sapResult: GenAIPipelineResponse["sap_result"], templateName?: string): string {
    const ext = (sapResult?.extraction ?? sapResult?.document ?? {}) as { headerFields?: HeaderField[]; lineItems?: unknown[] };
    const hf: HeaderField[] = ext.headerFields ?? [];
    const rawLi: unknown[] = ext.lineItems ?? [];

    const li: LineItem[] = rawLi.map((row) => {
      if (Array.isArray(row)) {
        const item: LineItem = {};
        for (const field of row as HeaderField[]) {
          if (field.name) item[field.name as keyof LineItem] = field.value ?? field.rawValue ?? null;
        }
        return item;
      }
      return row as LineItem;
    });

    const subtitle = templateName
      ? `Template: ${escapeHtml(templateName)}`
      : "Structured extraction from SAP Document AI service";

    const rows = hf.map((f) => { const conf=f.confidence??null; const pct=conf!=null?Math.round(conf*100):null; const cls=confidenceClass(conf); return `<tr><td><span class="field-name">${escapeHtml(f.name??"—")}</span></td><td><span class="field-value" title="${escapeHtml(String(f.value??f.rawValue??"—"))}">${escapeHtml(truncate(String(f.value??f.rawValue??"—")))}</span></td><td>${pct!=null?`<div class="confidence-bar"><div class="confidence-track"><div class="confidence-fill ${cls}" style="width:${pct}%"></div></div><span class="confidence-text">${pct}%</span></div>`:"—"}</td></tr>`; }).join("");
    const liRows = li.map((item,i) => `<tr><td>${i+1}</td><td>${escapeHtml(String(item.description??"—"))}</td><td>${escapeHtml(String(item.quantity??"—"))}</td><td>${escapeHtml(String(item.unitPrice??"—"))}</td><td>${escapeHtml(String(item.netAmount??"—"))}</td></tr>`).join("");
    return `<ui5-card><ui5-card-header slot="header" title-text="SAP Document AI Extraction" subtitle-text="${subtitle}"></ui5-card-header><div style="padding:1rem;"><div class="method-header"><span class="method-title">SAP Document AI</span><div class="method-stats"><span class="method-stat">${hf.length} fields</span><span class="method-stat">${li.length} line items</span></div></div>${hf.length>0?`<div style="overflow-x:auto;"><table class="field-table"><thead><tr><th>Field</th><th>Value</th><th>Confidence</th></tr></thead><tbody>${rows}</tbody></table></div>`:`<div style="padding:2rem;text-align:center;color:var(--sapContent_LabelColor,#6a6d70);">No header fields extracted</div>`}${li.length>0?`<div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">Line Items (${li.length})</div><div style="overflow-x:auto;"><table class="line-items-table"><thead><tr><th>#</th><th>Description</th><th>Qty</th><th>Unit Price</th><th>Net Amount</th></tr></thead><tbody>${liRows}</tbody></table></div></div>`:""}</div></ui5-card>`;
  }

  private llmCard(llm: LlmExtractionResult, title: string, _emoji: string): string {
    const conf = llm?.fieldConfidence ?? {};
    const li: LineItem[] = llm?.lineItems ?? [];
    const fields = INVOICE_FIELDS.filter((f) => llm?.[f]!=null);
    const rows = fields.map((f) => { const val=llm[f]; const c=(conf as Record<string,number>)[f]??null; const pct=c!=null?Math.round(c*100):null; const cls=confidenceClass(c); return `<tr><td><span class="field-name">${escapeHtml(f)}</span></td><td><span class="field-value" title="${escapeHtml(String(val))}">${escapeHtml(truncate(String(val)))}</span></td><td>${pct!=null?`<div class="confidence-bar"><div class="confidence-track"><div class="confidence-fill ${cls}" style="width:${pct}%"></div></div><span class="confidence-text">${pct}%</span></div>`:"—"}</td></tr>`; }).join("");
    const liRows = li.map((item,i) => `<tr><td>${i+1}</td><td>${escapeHtml(String(item.description??"—"))}</td><td>${escapeHtml(String(item.quantity??"—"))}</td><td>${escapeHtml(String(item.unitPrice??"—"))}</td><td>${escapeHtml(String(item.netAmount??"—"))}</td></tr>`).join("");
    return `<ui5-card><ui5-card-header slot="header" title-text="${escapeHtml(title)}" subtitle-text="GenAI multimodal extraction result"></ui5-card-header><div style="padding:1rem;"><div class="method-header"><span class="method-title">${escapeHtml(title)}</span><div class="method-stats"><span class="method-stat">${fields.length} fields</span><span class="method-stat">${li.length} line items</span></div></div>${fields.length>0?`<div style="overflow-x:auto;"><table class="field-table"><thead><tr><th>Field</th><th>Value</th><th>Confidence</th></tr></thead><tbody>${rows}</tbody></table></div>`:`<div style="padding:2rem;text-align:center;color:var(--sapContent_LabelColor,#6a6d70);">No fields extracted</div>`}${li.length>0?`<div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">Line Items (${li.length})</div><div style="overflow-x:auto;"><table class="line-items-table"><thead><tr><th>#</th><th>Description</th><th>Qty</th><th>Unit Price</th><th>Net Amount</th></tr></thead><tbody>${liRows}</tbody></table></div></div>`:""}</div></ui5-card>`;
  }

  private comparisonCard(cmp: ComparisonResult): string {
    const s = cmp?.summary;
    if (!s) return `<ui5-card><div style="padding:2rem;text-align:center;">No comparison data available</div></ui5-card>`;
    const conflicts = cmp.conflicts ?? [];
    const onlySap = cmp.only_in_sap ?? [];
    const onlyLlm = cmp.only_in_llm ?? [];
    const conflictRows = conflicts.map((c) => `<tr><td><span class="field-name">${escapeHtml(c.field)}</span></td><td class="mismatch">${escapeHtml(truncate(String(c.sap??"—")))}</td><td class="mismatch">${escapeHtml(truncate(String(c.llm_prompting??"—")))}</td><td class="mismatch">${escapeHtml(truncate(String(c.llm_structured??"—")))}</td></tr>`).join("");
    const pb = (label: string, val: number) => { const pct=Math.round(val*100); const cls=pct>=80?"success":pct>=50?"":"warning"; return `<div class="progress-container"><div class="progress-label"><span>${escapeHtml(label)}</span><span>${pct}%</span></div><div class="progress-track"><div class="progress-fill ${cls}" style="width:${pct}%"></div></div></div>`; };
    return `<ui5-card><ui5-card-header slot="header" title-text="Extraction Comparison" subtitle-text="Side-by-side analysis of all three methods"></ui5-card-header><div style="padding:1rem;"><div class="kpi-grid"><div class="kpi-card success"><div class="kpi-value">${s.agreements??0}</div><div class="kpi-label">Agreements</div></div><div class="kpi-card error"><div class="kpi-value">${s.conflicts??0}</div><div class="kpi-label">Conflicts</div></div><div class="kpi-card"><div class="kpi-value">${s.sap_fields_found??0}</div><div class="kpi-label">SAP Fields</div></div><div class="kpi-card"><div class="kpi-value">${s.llm_prompting_fields_found??0}</div><div class="kpi-label">LLM1 Fields</div></div><div class="kpi-card"><div class="kpi-value">${s.llm_structured_fields_found??0}</div><div class="kpi-label">LLM2 Fields</div></div><div class="kpi-card"><div class="kpi-value">${s.total_unique_fields??0}</div><div class="kpi-label">Total Unique</div></div></div>${conflicts.length>0?`<div style="margin-top:1.5rem;"><div class="section-title" style="margin-bottom:0.75rem;">Conflicts (${conflicts.length})</div><div style="overflow-x:auto;"><table class="comparison-table"><thead><tr><th>Field</th><th>SAP DocAI</th><th>LLM Technique 1</th><th>LLM Technique 2</th></tr></thead><tbody>${conflictRows}</tbody></table></div></div>`:""}${onlySap.length>0?`<div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">Only in SAP DocAI</div><div style="display:flex;flex-wrap:wrap;gap:0.5rem;">${onlySap.map((f)=>`<span class="status-badge info">${escapeHtml(f)}</span>`).join("")}</div></div>`:""}${onlyLlm.length>0?`<div style="margin-top:1rem;"><div class="section-title" style="margin-bottom:0.5rem;">Only in LLM Methods</div><div style="display:flex;flex-wrap:wrap;gap:0.5rem;">${onlyLlm.map((f)=>`<span class="status-badge success">${escapeHtml(f)}</span>`).join("")}</div></div>`:""}<div style="margin-top:1.5rem;"><div class="section-title" style="margin-bottom:0.75rem;">Confidence Averages</div>${pb("SAP Document AI",s.sap_confidence_avg??0)}${pb("LLM Technique 1",s.llm_prompting_confidence_avg??0)}${pb("LLM Technique 2",s.llm_structured_confidence_avg??0)}</div></div></ui5-card>`;
  }

  private summaryCard(r: GenAIPipelineResponse): string {
    const s = r.comparison?.summary;
    const fiResultHtml = this.state.fiPostResult
      ? (this.state.fiPostResult.success
          ? `<div style="margin-top:1rem;padding:0.75rem 1rem;background:var(--sapSuccessBackground,#f1fdf6);border:1px solid var(--sapSuccessBorderColor,#107e3e);border-radius:0.375rem;"><span style="font-weight:700;color:var(--sapSuccessColor,#107e3e);">FI Document Posted: ${escapeHtml(this.state.fiPostResult.fi_document)}</span><span style="margin-left:1rem;font-size:0.8125rem;color:var(--sapContent_LabelColor,#6a6d70);">Company Code: ${escapeHtml(this.state.fiPostResult.company_code)} | Fiscal Year: ${escapeHtml(this.state.fiPostResult.fiscal_year)} | Supplier: ${escapeHtml(this.state.fiPostResult.supplier_name_matched)}</span></div>`
          : `<div style="margin-top:1rem;padding:0.75rem 1rem;background:var(--sapErrorBackground,#fff1f1);border:1px solid var(--sapErrorBorderColor,#bb0000);border-radius:0.375rem;"><span style="font-weight:700;color:var(--sapErrorColor,#bb0000);">FI Posting Failed: ${escapeHtml(this.state.fiPostResult.error)}</span></div>`)
      : "";
    return `<ui5-card><ui5-card-header slot="header" title-text="Pipeline Summary" subtitle-text="Executive overview of extraction results"></ui5-card-header><div style="padding:1rem;"><div class="kpi-grid"><div class="kpi-card"><div class="kpi-value">${s?.sap_fields_found??0}</div><div class="kpi-label">SAP Fields</div></div><div class="kpi-card"><div class="kpi-value">${s?.llm_prompting_fields_found??0}</div><div class="kpi-label">LLM1 Fields</div></div><div class="kpi-card"><div class="kpi-value">${s?.llm_structured_fields_found??0}</div><div class="kpi-label">LLM2 Fields</div></div><div class="kpi-card success"><div class="kpi-value">${s?.agreements??0}</div><div class="kpi-label">Agreements</div></div><div class="kpi-card error"><div class="kpi-value">${s?.conflicts??0}</div><div class="kpi-label">Conflicts</div></div></div>${fiResultHtml}<div style="margin-top:1.5rem;"><div class="section-title" style="margin-bottom:0.75rem;">Raw Pipeline Output</div><div class="json-viewer">${escapeHtml(JSON.stringify({job_id:r.job_id,output_dir:r.output_dir,comparison_summary:s},null,2))}</div></div></div></ui5-card>`;
  }

  private soResultHtml(po: ExtractedPurchaseOrder, validation: SOValidationResult | null): string {
    // ── PO header card ───────────────────────────────────────────────────────
    const headerRows = [
      ["Customer", po.customer_name ?? "—"],
      ["PO Number", po.purchase_order_number ?? "—"],
      ["Order Date", po.order_date ?? "—"],
      ["Requested Delivery", po.requested_delivery_date ?? "—"],
      ["Currency", po.currency ?? "—"],
      ["Total Amount", po.total_amount != null ? String(po.total_amount) : "—"],
    ].map(([label, val]) =>
      `<tr><td><span class="field-name">${escapeHtml(label)}</span></td><td><span class="field-value">${escapeHtml(val)}</span></td></tr>`
    ).join("");

    const lineItemRows = (po.line_items ?? []).map((item: SOLineItem, i: number) =>
      `<tr>
        <td>${i + 1}</td>
        <td>${escapeHtml(item.material_code ?? "—")}</td>
        <td>${escapeHtml(item.sap_material ?? "—")}</td>
        <td>${escapeHtml(item.description ?? "—")}</td>
        <td>${escapeHtml(String(item.quantity ?? "—"))}</td>
        <td>${escapeHtml(item.uom ?? "—")}</td>
        <td>${escapeHtml(item.unit_price != null ? String(item.unit_price) : "—")}</td>
      </tr>`
    ).join("");

    const poCard = `
      <ui5-card>
        <ui5-card-header slot="header" title-text="Extracted Customer PO" subtitle-text="SAP DocAI extraction result"></ui5-card-header>
        <div style="padding:1rem;">
          <div style="overflow-x:auto;">
            <table class="field-table">
              <thead><tr><th>Field</th><th>Value</th></tr></thead>
              <tbody>${headerRows}</tbody>
            </table>
          </div>
          ${po.special_instructions ? `<div style="margin-top:0.75rem;font-size:0.8125rem;color:var(--sapContent_LabelColor,#6a6d70);"><strong>Special Instructions:</strong> ${escapeHtml(po.special_instructions)}</div>` : ""}
          ${lineItemRows ? `
            <div style="margin-top:1rem;">
              <div class="section-title" style="margin-bottom:0.5rem;">Line Items (${(po.line_items ?? []).length})</div>
              <div style="overflow-x:auto;">
                <table class="line-items-table">
                  <thead><tr><th>#</th><th>Material (doc)</th><th>SAP Material</th><th>Description</th><th>Qty</th><th>UoM</th><th>Unit Price</th></tr></thead>
                  <tbody>${lineItemRows}</tbody>
                </table>
              </div>
            </div>` : ""}
        </div>
      </ui5-card>`;

    // ── Validation card ──────────────────────────────────────────────────────
    if (!validation) return `<div class="fade-in" style="padding:1rem;">${poCard}</div>`;

    const customerBadge = validation.customer_resolved
      ? `<span class="status-badge success">Resolved</span>`
      : `<span class="status-badge error">Not Resolved</span>`;

    const readinessBadge = validation.ready_to_create
      ? `<span class="status-badge success">Ready to Create</span>`
      : `<span class="status-badge error">Not Ready</span>`;

    const itemRows = (validation.items_validation ?? []).map((item) =>
      `<tr>
        <td>${escapeHtml(item.material_code_extracted || "—")}</td>
        <td>${escapeHtml(item.sap_material || "—")}</td>
        <td>${escapeHtml(item.description || "—")}</td>
        <td>${item.matched ? `<span class="status-badge success">Matched</span>` : `<span class="status-badge error">Not Matched</span>`}</td>
      </tr>`
    ).join("");

    const issuesList = validation.issues?.length
      ? `<div style="margin-top:0.75rem;">
          <div class="section-title" style="margin-bottom:0.4rem;color:var(--sapErrorColor,#bb0000);">Issues</div>
          <ul style="margin:0;padding-left:1.25rem;font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);">
            ${validation.issues.map((iss) => `<li>${escapeHtml(iss)}</li>`).join("")}
          </ul>
        </div>`
      : "";

    const validationCard = `
      <ui5-card style="margin-top:1rem;">
        <ui5-card-header slot="header" title-text="S4 Validation" subtitle-text="Customer and material validation against S4HANA"></ui5-card-header>
        <div style="padding:1rem;">
          <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
            <div class="kpi-card ${validation.customer_resolved ? "success" : "error"}">
              <div class="kpi-label">Customer BP</div>
              <div class="kpi-value" style="font-size:1rem;">${escapeHtml(validation.customer_bp || "—")}</div>
              <div style="margin-top:0.25rem;">${customerBadge}</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-label">Customer Name</div>
              <div class="kpi-value" style="font-size:0.9rem;">${escapeHtml(validation.customer_name_matched || "—")}</div>
            </div>
            <div class="kpi-card ${validation.ready_to_create ? "success" : "error"}">
              <div class="kpi-label">Readiness</div>
              <div style="margin-top:0.5rem;">${readinessBadge}</div>
            </div>
          </div>
          ${itemRows ? `
            <div style="margin-top:1rem;">
              <div class="section-title" style="margin-bottom:0.5rem;">Item Validation</div>
              <div style="overflow-x:auto;">
                <table class="field-table">
                  <thead><tr><th>Extracted Material</th><th>SAP Material</th><th>Description</th><th>Status</th></tr></thead>
                  <tbody>${itemRows}</tbody>
                </table>
              </div>
            </div>` : ""}
          ${issuesList}
        </div>
      </ui5-card>`;

    return `<div class="fade-in" style="padding:1rem;">${poCard}${validationCard}</div>`;
  }

  private bindEvents(): void {
    // Shell navigation
    this.root.querySelector("#nav-legacy")?.addEventListener("click", () => { this.topView = "legacy"; this.render(); });
    this.root.querySelector("#nav-docai-new")?.addEventListener("click", () => { this.topView = "docai-new"; this.render(); });
    this.root.querySelector("#nav-train")?.addEventListener("click", () => { this.topView = "train"; this.render(); });

    // Scenario radio buttons
    document.querySelectorAll('input[name="scenario"]').forEach((el) => {
      el.addEventListener("change", (e) => {
        const val = (e.target as HTMLInputElement).value;
        this.state.scenario = val as AppState["scenario"];
        this.state.errorMessage = null;
        this.render();
      });
    });

    // File input
    const fileInput = document.getElementById("file-input") as HTMLInputElement | null;
    if (fileInput) {
      fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0] ?? null;
        if (file) { this.state.selectedFile = file; this.render(); }
      });
    }

    // Drag & drop
    const uploadZone = document.getElementById("upload-zone");
    if (uploadZone) {
      uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
      uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
      uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        const file = (e as DragEvent).dataTransfer?.files?.[0] ?? null;
        if (file) { this.state.selectedFile = file; this.render(); }
      });
    }

    // Execute button
    const execBtn = document.getElementById("execute-btn");
    if (execBtn) execBtn.addEventListener("click", () => this.execute());

    // Reset button
    const resetBtn = document.getElementById("reset-btn");
    if (resetBtn) resetBtn.addEventListener("click", () => this.reset());

    // Evaluate button
    const evalBtn = document.getElementById("evaluate-btn");
    if (evalBtn) evalBtn.addEventListener("click", () => this.runEvaluation());

    // POST in FI button (event delegation)
    this.root.addEventListener("click", (e) => {
      const target = e.target as HTMLElement;
      if (target.id === "btn-post-fi" || target.closest("#btn-post-fi")) {
        void this.postInFI();
      }
      if (target.id === "btn-create-so" || target.closest("#btn-create-so")) {
        void this.createSalesOrder();
      }
    });

    // Results tabs
    document.querySelectorAll(".results-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const tabId = (tab as HTMLElement).dataset["tab"];
        if (!tabId) return;
        document.querySelectorAll(".results-tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
        tab.classList.add("active");
        document.getElementById(`tab-${tabId}`)?.classList.remove("hidden");
      });
    });

    // Chat: send on button click
    const chatSend = document.getElementById("chat-send");
    if (chatSend) chatSend.addEventListener("click", () => void this.sendChat());

    // Chat: send on Enter (not Shift+Enter)
    const chatInput = document.getElementById("chat-input") as HTMLTextAreaElement | null;
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          void this.sendChat();
        }
      });
    }
  }

  private reset(): void {
    this.state.processingStatus = "idle";
    this.state.processingSteps = this.buildSteps();
    this.state.pipelineResult = null;
    this.state.invoiceResult = null;
    this.state.schemasResult = null;
    this.state.templatesResult = null;
    this.state.evaluationResult = null;
    this.state.fiPostResult = null;
    this.state.soResult = null;
    this.state.paResult = null;
    (this.state as unknown as Record<string, unknown>)["soExtracted"] = null;
    (this.state as unknown as Record<string, unknown>)["soValidation"] = null;
    this.state.errorMessage = null;
    this.state.selectedFile = null;
    this.state.chatMessages = [];
    this.state.streamingText = "";
    this.render();
  }

  private async execute(): Promise<void> {
    this.state.errorMessage = null;
    const scenario = this.state.scenario;

    if ((scenario === "genai" || scenario === "sales-order") && !this.state.selectedFile) {
      this.state.errorMessage = "Please select a document file before executing.";
      this.render();
      return;
    }

    this.state.processingStatus = "processing";
    this.state.processingSteps = this.buildSteps();
    this.render();

    try {
      if (scenario === "evaluation") {
        this.state.evaluationResult = await api.runEvaluation();
        this.state.processingStatus = "completed";
        this.render();
        this.toast("Evaluation completed", "success");
      } else if (scenario === "genai") {
        const file = this.state.selectedFile!;
        this.setStep("sap", "active");
        this.state.pipelineResult = await api.runGenAIPipeline(file);

        const isTemplateRoute = this.state.pipelineResult?.route === "template";
        if (isTemplateRoute) {
          const tName = this.state.pipelineResult?.routing_decision?.template_name ?? "Template";
          this.state.processingSteps = this.buildTemplateSteps(tName);
          this.setStep("sap", "completed");
          this.setStep("done", "completed");
        } else {
          this.setStep("sap", "completed");
          this.setStep("llm1", "completed");
          this.setStep("llm2", "completed");
          this.setStep("compare", "completed");
        }
        this.state.processingStatus = "completed";
        this.render();
        this.toast("Pipeline completed successfully", "success");
      } else if (scenario === "doc-ai-new") {
        this.state.processingStatus = "idle";
        this.topView = "docai-new";
        this.render();
      } else if (scenario === "train-template") {
        this.state.processingStatus = "idle";
        this.topView = "train";
        this.render();
      } else if (scenario === "sales-order") {
        const file = this.state.selectedFile!;
        this.state.processingSteps = this.buildSOSteps();
        this.render();

        // Step 1: Extract
        this.setStep("extract", "active");
        const po = await api.extractSalesOrder(file);
        this.setStep("extract", "completed");

        // Step 2: Validate
        this.setStep("validate", "active");
        const validation = await api.validateSalesOrder(po);
        this.setStep("validate", "completed");

        // Store results on state using index signature
        (this.state as unknown as Record<string, unknown>)["soExtracted"] = po;
        (this.state as unknown as Record<string, unknown>)["soValidation"] = validation;
        this.state.processingStatus = "completed";
        this.render();
        this.toast("Sales Order extraction complete", "success");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.state.errorMessage = msg;
      this.state.processingStatus = "error";
      this.state.processingSteps.forEach((s) => { if (s.status === "active") s.status = "error"; });
      this.render();
      this.toast("Pipeline failed: " + msg, "error");
    }
  }

  private async runEvaluation(): Promise<void> {
    const evalBtn = document.getElementById("evaluate-btn");
    if (evalBtn) { evalBtn.setAttribute("disabled", ""); evalBtn.textContent = "Evaluating..."; }
    try {
      this.state.evaluationResult = await api.runEvaluation();
      this.render();
      this.toast("Evaluation completed successfully", "success");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.toast("Evaluation failed: " + msg, "error");
      if (evalBtn) { evalBtn.removeAttribute("disabled"); evalBtn.textContent = "Evaluate Result?"; }
    }
  }

  private async postInFI(): Promise<void> {
    const btn = document.getElementById("btn-post-fi");
    const statusSpan = document.getElementById("fi-post-status");
    if (btn) btn.setAttribute("disabled", "");
    if (statusSpan) statusSpan.textContent = "";

    const r = this.state.pipelineResult;
    if (!r) {
      if (statusSpan) statusSpan.textContent = "No pipeline result available.";
      if (btn) btn.removeAttribute("disabled");
      return;
    }

    // ── Field extraction priority: SAP DocAI → LLM structured → LLM prompting ──
    // SAP DocAI is the primary source. LLM values fill gaps or override conflicts.

    // Helper: get value from SAP DocAI headerFields by field name
    const sapField = (name: string): string => {
      const src = r.template_result ?? r.sap_result;
      const ext = (src?.extraction ?? src?.document ?? {}) as { headerFields?: HeaderField[] };
      return String(ext.headerFields?.find((f) => f.name === name)?.value ?? "");
    };

    // Helper: get SAP field confidence (0–1)
    const sapConf = (name: string): number => {
      const src = r.template_result ?? r.sap_result;
      const ext = (src?.extraction ?? src?.document ?? {}) as { headerFields?: HeaderField[] };
      return Number(ext.headerFields?.find((f) => f.name === name)?.confidence ?? 0);
    };

    // Normalize any date to YYYY-MM-DD
    const normalizeDate = (raw: unknown): string => {
      if (!raw) return "";
      const s = String(raw).trim();
      if (!s) return "";
      const odataMatch = s.match(/\/Date\((-?\d+)/);
      if (odataMatch) return new Date(parseInt(odataMatch[1], 10)).toISOString().slice(0, 10);
      if (s.length >= 10 && s[4] === "-") return s.slice(0, 10);
      return s;
    };

    // Best value: prefer SAP if confidence >= 0.7, else prefer LLM, else SAP as fallback
    const best = (fieldName: string, llmValue: unknown): string => {
      const sapVal  = sapField(fieldName);
      const sapC    = sapConf(fieldName);
      const llmVal  = String(llmValue ?? "").trim();
      if (sapVal && sapC >= 0.7) return sapVal;       // SAP high confidence → use SAP
      if (llmVal) return llmVal;                      // LLM available → use LLM
      return sapVal;                                   // fallback to SAP even if low conf
    };

    const llmS = r.llm_structured as LlmExtractionResult;
    const llmP = r.llm_prompting as LlmExtractionResult;

    const supplier_name  = best("senderName",      llmS?.senderName  ?? llmP?.senderName  ?? r.routing_decision?.supplier_name ?? "");
    const invoice_number = best("documentNumber",  llmS?.documentNumber ?? llmP?.documentNumber ?? "");
    const invoice_date   = normalizeDate(best("documentDate", llmS?.documentDate ?? llmP?.documentDate ?? ""));
    const gross_raw      = best("grossAmount",     llmS?.grossAmount ?? llmP?.grossAmount ?? "0");
    const total_amount   = parseFloat(String(gross_raw ?? "0").replace(/[^0-9.]/g, "")) || 0;
    const rawCurrency    = best("currencyCode",    llmS?.currencyCode ?? llmP?.currencyCode ?? "");
    const currency       = rawCurrency.trim() || "EUR";
    const po_number      = r.po_number ?? null;

    if (btn) btn.removeAttribute("disabled");

    // ── Transparent routing based on detected document type ──────────────────
    // "purchase_order" → create Sales Order in SD
    // "invoice" with PO → post via MIRO (MM)
    // "invoice" no PO  → post to FI (GL) or search PO manually

    if (r.document_type === "purchase_order") {
      await this.createSalesOrder();
      return;
    }

    if (r.document_type === "payment_advice") {
      await this._postPaymentAdvice();
      return;
    }

    if (po_number) {
      await this._postPOInvoice({ supplier_name, invoice_number, invoice_date, total_amount, currency, po_number });
    } else {
      await this._openNoPOModal({ supplier_name, invoice_number, invoice_date, total_amount, currency });
    }
  }

  private async createSalesOrder(): Promise<void> {
    // Read extracted PO from pipeline result (transparent routing path)
    // or from standalone SO extraction (direct SO pipeline path)
    const r = this.state.pipelineResult;
    const po: ExtractedPurchaseOrder | null =
      (r?.extracted_po as ExtractedPurchaseOrder | null) ??
      ((this.state as unknown as Record<string, unknown>)["soExtracted"] as ExtractedPurchaseOrder | null);
    const validation: SOValidationResult | null =
      (r?.so_validation as SOValidationResult | null) ??
      ((this.state as unknown as Record<string, unknown>)["soValidation"] as SOValidationResult | null);

    if (!po || !validation) {
      this._showErrorModal("Sales Order", "No extraction result available. Run the pipeline first.");
      return;
    }

    // Show confirmation popup BEFORE posting to S4
    await this._showSOConfirmModal(po, validation);
  }

  private async _showSOConfirmModal(po: ExtractedPurchaseOrder, validation: SOValidationResult): Promise<void> {
    const existing = document.getElementById("so-confirm-backdrop");
    if (existing) existing.remove();

    const itemRows = (po.line_items ?? []).map((item: SOLineItem, i: number) => {
      const iv = validation.items_validation?.[i];
      const sapMat = iv?.sap_material || item.sap_material || "—";
      const matched = iv?.matched ?? false;
      const badge = matched
        ? `<span style="color:var(--sapSuccessColor,#107e3e);font-weight:700;font-size:0.72rem;">MATCHED</span>`
        : `<span style="color:var(--sapWarningColor,#e9730c);font-weight:700;font-size:0.72rem;">REVIEW</span>`;
      return `<tr>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;">${escapeHtml(item.material_code || "—")}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;font-weight:600;">${escapeHtml(sapMat)}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;">${escapeHtml(item.description || "—")}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;text-align:center;">${escapeHtml(String(item.quantity))}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;text-align:center;">${badge}</td>
      </tr>`;
    }).join("");

    const backdrop = document.createElement("div");
    backdrop.id = "so-confirm-backdrop";
    backdrop.style.cssText = "position:fixed;inset:0;z-index:2000;background:rgba(29,45,62,0.48);display:flex;align-items:center;justify-content:center;";
    backdrop.innerHTML = `
      <div style="width:min(52rem,calc(100vw-2rem));max-height:calc(100dvh-3rem);display:flex;flex-direction:column;background:#fff;border-radius:0.5rem;box-shadow:0 1rem 3rem rgba(29,45,62,0.28);overflow:hidden;">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:0.75rem 1rem;background:#f7f9fa;border-bottom:1px solid #d5dadd;">
          <div>
            <div style="font-size:0.625rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#5b738b;">Confirm before posting to S4</div>
            <h2 style="margin:0;font-size:0.95rem;font-weight:700;color:#183a5a;">Create Sales Order — Data Validation</h2>
          </div>
          <ui5-button id="so-confirm-close" design="Transparent" icon="decline" accessible-name="Close"></ui5-button>
        </div>
        <div style="flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:0.85rem;">

          <!-- SoldTo / ShipTo -->
          <div style="border:1px solid #d5dadd;border-radius:0.35rem;overflow:hidden;">
            <div style="background:#eef5fa;padding:0.45rem 0.75rem;font-size:0.72rem;font-weight:700;color:#354a5f;text-transform:uppercase;letter-spacing:0.04em;">
              Customer (SoldTo / ShipTo)
            </div>
            <div style="padding:0.65rem 0.75rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.8125rem;">
              <div>
                <div style="font-size:0.7rem;color:#5b738b;font-weight:700;margin-bottom:0.15rem;">SoldTo Party</div>
                <div style="font-weight:700;color:#183a5a;">${escapeHtml(validation.customer_bp || "—")}</div>
                <div style="color:#354a5f;">${escapeHtml(validation.customer_name_matched || po.customer_name || "—")}</div>
              </div>
              <div>
                <div style="font-size:0.7rem;color:#5b738b;font-weight:700;margin-bottom:0.15rem;">ShipTo Party</div>
                <div style="font-weight:700;color:#183a5a;">${escapeHtml(validation.customer_bp || "—")}</div>
                <div style="color:#354a5f;font-size:0.75rem;">(defaults to SoldTo — can be changed in S4)</div>
              </div>
            </div>
          </div>

          <!-- PO header -->
          <div style="border:1px solid #d5dadd;border-radius:0.35rem;overflow:hidden;">
            <div style="background:#eef5fa;padding:0.45rem 0.75rem;font-size:0.72rem;font-weight:700;color:#354a5f;text-transform:uppercase;letter-spacing:0.04em;">
              Order Header
            </div>
            <div style="padding:0.65rem 0.75rem;display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;font-size:0.8125rem;">
              ${[["PO Number", po.purchase_order_number],["Order Date", po.order_date],["Delivery Date", po.requested_delivery_date],
                 ["Currency", po.currency],["Total Amount", po.total_amount != null ? String(po.total_amount) : "—"],
                 ["Special Instr.", po.special_instructions || "—"]]
                .map(([l,v])=>`<div><div style="font-size:0.7rem;color:#5b738b;font-weight:700;margin-bottom:0.1rem;">${escapeHtml(String(l))}</div><div style="font-weight:600;color:#183a5a;">${escapeHtml(String(v||"—"))}</div></div>`).join("")}
            </div>
          </div>

          <!-- Materials -->
          <div style="border:1px solid #d5dadd;border-radius:0.35rem;overflow:hidden;">
            <div style="background:#eef5fa;padding:0.45rem 0.75rem;font-size:0.72rem;font-weight:700;color:#354a5f;text-transform:uppercase;letter-spacing:0.04em;">
              Material Codes (${(po.line_items ?? []).length} items)
            </div>
            <div style="overflow-x:auto;">
              <table style="width:100%;border-collapse:collapse;font-size:0.8125rem;">
                <thead>
                  <tr style="background:#f0f5fa;">
                    <th style="padding:0.4rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;">Extracted Material</th>
                    <th style="padding:0.4rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;">SAP Material Code</th>
                    <th style="padding:0.4rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;">Description</th>
                    <th style="padding:0.4rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:center;font-size:0.72rem;font-weight:700;color:#354a5f;">Qty</th>
                    <th style="padding:0.4rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:center;font-size:0.72rem;font-weight:700;color:#354a5f;">Match</th>
                  </tr>
                </thead>
                <tbody>${itemRows}</tbody>
              </table>
            </div>
          </div>

          ${!validation.ready_to_create && validation.issues?.length ? `
          <div style="background:#fff1f1;border:1px solid var(--sapErrorBorderColor,#bb0000);border-radius:0.35rem;padding:0.65rem 0.75rem;">
            <div style="font-weight:700;font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);margin-bottom:0.35rem;">Validation Issues</div>
            <ul style="margin:0;padding-left:1.25rem;font-size:0.8125rem;color:var(--sapErrorColor,#bb0000);">
              ${validation.issues.map(i=>`<li>${escapeHtml(i)}</li>`).join("")}
            </ul>
          </div>` : ""}
        </div>
        <div style="display:flex;justify-content:flex-end;align-items:center;gap:0.5rem;padding:0.65rem 1rem;border-top:1px solid #d5dadd;background:#fafafa;">
          <span id="so-confirm-status" style="font-size:0.8125rem;color:var(--sapNeutralColor,#6a6d70);flex:1;"></span>
          <ui5-button id="so-confirm-cancel" design="Transparent">Cancel</ui5-button>
          <ui5-button id="so-confirm-post" design="Emphasized" icon="add">Confirm &amp; Create Sales Order</ui5-button>
        </div>
      </div>`;

    document.body.appendChild(backdrop);

    const close = () => backdrop.remove();
    const statusEl = document.getElementById("so-confirm-status")!;

    document.getElementById("so-confirm-close")!.addEventListener("click", close);
    document.getElementById("so-confirm-cancel")!.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });

    document.getElementById("so-confirm-post")!.addEventListener("click", async () => {
      const postBtn = document.getElementById("so-confirm-post") as HTMLElement;
      postBtn.setAttribute("disabled", "");
      statusEl.textContent = "Creating Sales Order in S4…";

      const payload: CreateSORequest = {
        customer_bp:           validation.customer_bp,
        purchase_order_number: po.purchase_order_number,
        currency:              po.currency || "EUR",
        items: (po.line_items ?? []).map((item: SOLineItem, i: number) => ({
          material_code: (validation.items_validation?.[i]?.sap_material || item.sap_material || item.material_code),
          sap_material:  validation.items_validation?.[i]?.sap_material || item.sap_material || "",
          description:   item.description || "",
          quantity:      item.quantity,
          uom:           item.uom || "EA",
          unit_price:    item.unit_price,
          currency:      item.currency ?? po.currency,
        })),
        special_instructions: po.special_instructions,
      };

      try {
        const result: CreateSOResponse = await api.createSalesOrder(payload);
        close();
        this.state.soResult = result;
        if (result.success) {
          this.toast(`Sales Order ${result.sales_order} created — ${result.items_created} item(s)`, "success");
        } else {
          this._showErrorModal("Sales Order Creation Failed", result.error || result.message);
        }
        this.render();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        postBtn.removeAttribute("disabled");
        statusEl.style.color = "var(--sapErrorColor,#bb0000)";
        statusEl.textContent = "Failed — see details";
        this._showErrorModal("Sales Order Creation Failed", msg);
      }
    });
  }

  private async _postPaymentAdvice(): Promise<void> {
    const r = this.state.pipelineResult;
    const pa = r?.extracted_pa as ExtractedPaymentAdvice | null;
    if (!pa) {
      this._showErrorModal("Payment Advice", "No extraction result available.");
      return;
    }

    const btn = document.getElementById("btn-post-fi");
    const statusSpan = document.getElementById("fi-post-status");
    if (btn) btn.setAttribute("disabled", "");
    if (statusSpan) { statusSpan.style.color = ""; statusSpan.textContent = "Posting Payment Advice…"; }

    const payload: PostPaymentAdviceRequest = {
      payer_name:          pa.payer_name || "",
      payer_bp:            pa.payer_bp   || "",
      payment_date:        pa.payment_date || new Date().toISOString().slice(0, 10),
      total_amount:        pa.total_amount || 0,
      currency:            pa.currency   || "EUR",
      bank_reference:      pa.bank_reference || "",
      payment_advice_note: pa.payment_advice_note || "Payment Advice from Document AI",
      line_items:          pa.line_items || [],
    };

    try {
      const result = await api.postPaymentAdvice(payload);
      this.state.paResult = result;
      if (result.success) {
        if (statusSpan) { statusSpan.style.color = "var(--sapSuccessColor,#107e3e)"; statusSpan.textContent = `PA: ${result.payment_advice}`; }
        this.toast(`Payment Advice ${result.payment_advice} posted — ${result.payer_name_matched}`, "success");
      } else {
        this._showErrorModal("Payment Advice Posting Failed", result.error);
        if (statusSpan) { statusSpan.style.color = "var(--sapErrorColor,#bb0000)"; statusSpan.textContent = "Posting failed — see details"; }
      }
      this.render();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this._showErrorModal("Payment Advice Posting Failed", msg);
      if (statusSpan) { statusSpan.style.color = "var(--sapErrorColor,#bb0000)"; statusSpan.textContent = "Posting failed — see details"; }
      if (btn) btn.removeAttribute("disabled");
    }
  }

  private paResultHtml(pa: ExtractedPaymentAdvice): string {
    const fmt = (n: number | null | undefined, curr?: string) =>
      n != null && n > 0 ? `${curr || ""} ${n.toLocaleString("de-DE", {minimumFractionDigits: 2})}`.trim() : "—";

    const headerRows = [
      ["Payer / Zahler",    pa.payer_name     || "—"],
      ["Payment Date",      pa.payment_date   || "—"],
      ["Currency",          pa.currency       || "—"],
      ["Total Amount",      fmt(pa.total_amount, pa.currency)],
      ["Bank Reference",    pa.bank_reference || "—"],
      ["Our Reference",     pa.our_reference  || "—"],
      ["Note",              pa.payment_advice_note || "—"],
    ].map(([label, val]) =>
      `<tr><td style="width:40%;"><span class="field-name">${escapeHtml(String(label))}</span></td><td><span class="field-value">${escapeHtml(String(val))}</span></td></tr>`
    ).join("");

    const lineItemRows = (pa.line_items || []).map((item, i) => {
      const gross = item.gross_amount || item.net_payment_amount || 0;
      const net   = item.net_payment_amount || item.gross_amount || 0;
      const disc  = item.discount_amount || 0;
      return `<tr>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;">${i + 1}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;font-weight:600;">${escapeHtml(item.invoice_number || "—")}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;">${escapeHtml(item.invoice_date || "—")}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;text-align:right;">${fmt(gross)}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;text-align:right;">${disc > 0 ? fmt(disc) : "—"}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;text-align:right;font-weight:700;color:var(--sapSuccessColor,#107e3e);">${fmt(net)}</td>
        <td style="padding:0.4rem 0.6rem;border-bottom:1px solid #edf0f2;">${escapeHtml(item.currency || pa.currency || "—")}</td>
      </tr>`;
    }).join("");

    return `<div class="fade-in" style="padding:1rem;display:flex;flex-direction:column;gap:1rem;">
      <ui5-card>
        <ui5-card-header slot="header" title-text="Payment Advice" subtitle-text="SAP DocAI extraction — SAP_paymentAdvice_schema"></ui5-card-header>
        <div style="padding:1rem;">
          <div style="overflow-x:auto;">
            <table class="field-table" style="width:100%">
              <thead><tr><th>Field</th><th>Value</th></tr></thead>
              <tbody>${headerRows}</tbody>
            </table>
          </div>
          ${lineItemRows ? `
          <div style="margin-top:1rem;">
            <div class="section-title" style="margin-bottom:0.5rem;">Line Items (${(pa.line_items || []).length})</div>
            <div style="overflow-x:auto;border:1px solid #d5dadd;border-radius:0.35rem;">
              <table style="width:100%;border-collapse:collapse;font-size:0.8125rem;">
                <thead>
                  <tr style="background:#f0f5fa;">
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;width:8%">#</th>
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;">Invoice No.</th>
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;">Date</th>
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:right;font-size:0.72rem;font-weight:700;color:#354a5f;">Gross Amount</th>
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:right;font-size:0.72rem;font-weight:700;color:#354a5f;">Discount</th>
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:right;font-size:0.72rem;font-weight:700;color:#354a5f;">Net Payment</th>
                    <th style="padding:0.45rem 0.6rem;border-bottom:2px solid #d5dadd;text-align:left;font-size:0.72rem;font-weight:700;color:#354a5f;">Currency</th>
                  </tr>
                </thead>
                <tbody>${lineItemRows}</tbody>
              </table>
            </div>
          </div>` : ""}
        </div>
      </ui5-card>
    </div>`;
  }

  private async _postPOInvoice(fields: {
    supplier_name: string; invoice_number: string; invoice_date: string;
    total_amount: number; currency: string; po_number: string; po_item?: string;
  }): Promise<void> {
    const btn = document.getElementById("btn-post-fi");
    const statusSpan = document.getElementById("fi-post-status");
    if (btn) btn.setAttribute("disabled", "");
    if (statusSpan) { statusSpan.style.color = ""; statusSpan.textContent = "Posting via PO (MIRO)…"; }

    const payload: PostPOInvoiceRequest = {
      supplier_name:       fields.supplier_name,
      invoice_number:      fields.invoice_number,
      invoice_date:        fields.invoice_date,
      total_amount:        fields.total_amount,
      currency:            fields.currency,
      purchase_order:      fields.po_number,
      purchase_order_item: fields.po_item ?? "00010",
    };

    try {
      const result = await api.postPOInvoice(payload);
      this.state.fiPostResult = result as PostInvoiceResponse;
      if (result.success) {
        if (statusSpan) { statusSpan.style.color = "var(--sapSuccessColor,#107e3e)"; statusSpan.textContent = `PO Invoice posted — FI Doc: ${result.fi_document}`; }
        this.toast(`PO invoice posted — FI Doc: ${result.fi_document} (PO: ${result.purchase_order})`, "success");
      } else {
        if (statusSpan) { statusSpan.style.color = "var(--sapErrorColor,#bb0000)"; statusSpan.textContent = "Posting failed — see details"; }
        this._showErrorModal("PO Invoice Posting Failed", result.error);
      }
      this.render();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (statusSpan) { statusSpan.style.color = "var(--sapErrorColor,#bb0000)"; statusSpan.textContent = "Posting failed — see details"; }
      this._showErrorModal("PO Invoice Posting Failed", msg);
      if (btn) btn.removeAttribute("disabled");
    }
  }

  // ─── NO-PO MODAL ────────────────────────────────────────────────────────────

  private _buildModalHtml(fields: {
    supplier_name: string; invoice_number: string;
    total_amount: number; currency: string;
    defaultGLAccount?: string;
  }): string {
    const glDefault = escapeHtml(fields.defaultGLAccount || "");
    return `
<div id="po-modal-backdrop" class="po-modal-backdrop open" role="dialog" aria-modal="true" aria-labelledby="po-modal-title">
  <div class="po-modal">
    <div class="po-modal-header">
      <h2 id="po-modal-title">No PO detected — verify before posting</h2>
      <ui5-button id="po-modal-close" design="Transparent" icon="decline" accessible-name="Close"></ui5-button>
    </div>
    <div class="po-modal-body">
      <p style="font-size:0.875rem;color:#5b738b;margin:0;">
        No Purchase Order number was found in the extracted invoice
        (<strong>${escapeHtml(fields.invoice_number || "—")}</strong>,
        <strong>${fields.currency} ${fields.total_amount.toLocaleString()}</strong>,
        supplier: <strong>${escapeHtml(fields.supplier_name || "—")}</strong>).
      </p>
      <p style="font-size:0.875rem;color:#5b738b;margin:0;">
        You can search for an open PO by vendor and select it, or post directly to FI via GL account.
      </p>

      <!-- GL Account selection -->
      <div style="display:flex;align-items:flex-end;gap:0.75rem;padding:0.65rem 0;border-top:1px solid #edf0f2;border-bottom:1px solid #edf0f2;margin:0.5rem 0;">
        <div style="flex:0 0 auto;">
          <label for="gl-account-input" style="display:block;font-size:0.72rem;font-weight:700;color:#354a5f;margin-bottom:0.25rem;">GL Account</label>
          <input id="gl-account-input" type="text"
            placeholder="e.g. 11001000"
            value="${glDefault}"
            style="width:11rem;padding:0.4rem 0.55rem;border:1px solid #c0c8d0;border-radius:0.3rem;font-size:0.8125rem;color:#183a5a;font-family:inherit;" />
        </div>
        <div style="font-size:0.75rem;color:#5b738b;padding-bottom:0.45rem;">
          Account used when posting directly to FI (no PO). Leave blank to use the server default.
        </div>
      </div>

      <!-- Vendor / BP search -->
      <div class="po-search-row">
        <div>
          <label for="po-vendor-input">Vendor name or BP number</label>
          <input id="po-vendor-input" type="text"
            placeholder="e.g. Inlandslieferant DE 3 or 10300003"
            value="${escapeHtml(fields.supplier_name || "")}" />
        </div>
        <div style="padding-bottom:0.05rem">
          <label style="visibility:hidden">Search</label>
          <ui5-button id="po-search-btn" design="Default" icon="search">Search POs</ui5-button>
        </div>
        <div style="padding-bottom:0.05rem">
          <label style="visibility:hidden">Clear</label>
          <ui5-button id="po-clear-btn" design="Transparent">Clear</ui5-button>
        </div>
      </div>

      <div id="po-status" class="po-status-msg"></div>

      <!-- PO results table (initially hidden) -->
      <div id="po-table-section" style="display:none;">
        <div class="po-table-wrap" style="max-height:16rem;">
          <table class="po-table" id="po-results-table">
            <thead>
              <tr>
                <th>Select</th>
                <th>PO Number</th>
                <th>Date</th>
                <th>Currency</th>
                <th>Supplier Name</th>
                <th>Company</th>
                <th>PO Item</th>
              </tr>
            </thead>
            <tbody id="po-results-body"></tbody>
          </table>
        </div>
        <p style="font-size:0.75rem;color:#5b738b;margin:0.35rem 0 0;">
          Click a row to select the PO. Adjust the item number if needed.
        </p>
      </div>

      <!-- Selected PO summary -->
      <div id="po-selected-summary" style="display:none;background:#eaf4fb;border:1px solid #0a6ed1;border-radius:0.35rem;padding:0.55rem 0.75rem;font-size:0.8125rem;color:#183a5a;">
        Selected: <strong id="po-selected-label"></strong>
      </div>
    </div>

    <div class="po-modal-footer">
      <ui5-button id="po-post-gl-btn" design="Default">Post to FI (GL account)</ui5-button>
      <ui5-button id="po-post-po-btn" design="Emphasized" disabled>Post via PO (MIRO)</ui5-button>
      <ui5-button id="po-cancel-btn" design="Transparent">Find PO manually later</ui5-button>
    </div>
  </div>
</div>`;
  }

  private async _openNoPOModal(fields: {
    supplier_name: string; invoice_number: string; invoice_date: string;
    total_amount: number; currency: string;
  }): Promise<void> {
    // Fetch default GL account from backend before building the modal
    const fiConfig = await api.getFIConfig();

    // Inject modal into body (outside #app to avoid re-render wipe)
    const existing = document.getElementById("po-modal-backdrop");
    if (existing) existing.remove();

    const wrapper = document.createElement("div");
    wrapper.innerHTML = this._buildModalHtml({ ...fields, defaultGLAccount: fiConfig.gl_account });
    document.body.appendChild(wrapper.firstElementChild!);

    const backdrop  = document.getElementById("po-modal-backdrop")!;
    const statusEl  = document.getElementById("po-status")!;
    const tableSection = document.getElementById("po-table-section")!;
    const tbody     = document.getElementById("po-results-body")!;
    const selectedSummary = document.getElementById("po-selected-summary")!;
    const selectedLabel   = document.getElementById("po-selected-label")!;
    const postPoBtn = document.getElementById("po-post-po-btn") as HTMLButtonElement;
    const vendorInput = document.getElementById("po-vendor-input") as HTMLInputElement;
    const glAccountInput = document.getElementById("gl-account-input") as HTMLInputElement;

    let selectedPO: PurchaseOrderResult | null = null;
    let selectedItem = "00010";

    const close = () => backdrop.remove();

    const setStatus = (msg: string, tone: ""|"error"|"success" = "") => {
      statusEl.textContent = msg;
      statusEl.className = `po-status-msg${tone ? " " + tone : ""}`;
    };

    const renderRows = (pos: PurchaseOrderResult[]) => {
      if (!pos.length) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#5b738b;padding:1rem;">No Purchase Orders found for this vendor.</td></tr>`;
        tableSection.style.display = "";
        return;
      }
      tbody.innerHTML = pos.map((po) => {
        const badge = po.status === "02" || po.status === "open"
          ? `<span class="po-badge open">Open</span>`
          : `<span class="po-badge other">${escapeHtml(po.status)}</span>`;
        return `<tr data-po="${escapeHtml(po.purchase_order)}" data-supplier="${escapeHtml(po.supplier)}" data-name="${escapeHtml(po.supplier_name)}" data-currency="${escapeHtml(po.currency)}">
          <td style="text-align:center"><input type="radio" name="po-select" value="${escapeHtml(po.purchase_order)}" /></td>
          <td><strong>${escapeHtml(po.purchase_order)}</strong> ${badge}</td>
          <td>${escapeHtml(po.document_date ?? "—")}</td>
          <td>${escapeHtml(po.currency)}</td>
          <td>${escapeHtml(po.supplier_name || po.supplier)}</td>
          <td>${escapeHtml(po.company_code)}</td>
          <td><input class="po-item-input" type="text" value="00010" title="PO Item" data-po="${escapeHtml(po.purchase_order)}" /></td>
        </tr>`;
      }).join("");
      tableSection.style.display = "";
    };

    // Row click → select
    tbody.addEventListener("click", (e) => {
      const row = (e.target as HTMLElement).closest("tr") as HTMLTableRowElement | null;
      if (!row || !row.dataset["po"]) return;
      // deselect all
      tbody.querySelectorAll("tr").forEach((r) => r.classList.remove("selected"));
      row.classList.add("selected");
      row.querySelector<HTMLInputElement>("input[type=radio]")!.checked = true;

      const poNum   = row.dataset["po"] ?? "";
      const poName  = row.dataset["name"] ?? row.dataset["supplier"] ?? "";
      const poCurr  = row.dataset["currency"] ?? "";
      const itemInput = row.querySelector<HTMLInputElement>(".po-item-input");
      selectedItem = itemInput?.value.trim() || "00010";

      // Build a minimal PurchaseOrderResult for selectedPO
      selectedPO = {
        purchase_order:          poNum,
        supplier:                row.dataset["supplier"] ?? "",
        supplier_name:           poName,
        company_code:            "",
        purchasing_organization: "",
        purchasing_group:        "",
        document_date:           null,
        currency:                poCurr,
        status:                  "",
      };

      selectedLabel.textContent = `PO ${poNum}  |  ${poName}  |  ${poCurr}`;
      selectedSummary.style.display = "";
      postPoBtn.removeAttribute("disabled");
    });

    // PO item input change
    tbody.addEventListener("change", (e) => {
      const input = e.target as HTMLInputElement;
      if (input.classList.contains("po-item-input") && input.dataset["po"] === selectedPO?.purchase_order) {
        selectedItem = input.value.trim() || "00010";
        if (selectedLabel.textContent) {
          selectedLabel.textContent = selectedLabel.textContent.replace(/Item: \S+/, `Item: ${selectedItem}`);
        }
      }
    });

    // Search
    const doSearch = async () => {
      const q = vendorInput.value.trim();
      if (!q) { setStatus("Enter a vendor name or BP number to search.", "error"); return; }
      setStatus("Searching…");
      tableSection.style.display = "none";
      selectedSummary.style.display = "none";
      selectedPO = null;
      postPoBtn.setAttribute("disabled", "");

      try {
        // If q looks like a number, use it directly as BP; otherwise search by name first
        let bp = q;
        if (!/^\d+$/.test(q)) {
          setStatus("Resolving vendor name to BP…");
          const bpResult = await api.searchCustomers(q, 1);
          if (!bpResult.results.length) {
            setStatus(`No Business Partner found for "${q}". Try using the BP number directly.`, "error");
            return;
          }
          bp = bpResult.results[0].business_partner;
          setStatus(`BP resolved: ${bp} — ${bpResult.results[0].customer_name}. Fetching POs…`);
        } else {
          setStatus(`Fetching POs for vendor ${bp}…`);
        }

        const result = await api.searchPurchaseOrders(bp, 30);
        if (!result.purchase_orders.length) {
          setStatus(`No open Purchase Orders found for vendor ${bp}.`, "error");
          tableSection.style.display = "none";
          return;
        }
        setStatus(`Found ${result.count} PO(s) for vendor ${bp}.`, "success");
        renderRows(result.purchase_orders);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setStatus(`Search failed: ${msg}`, "error");
      }
    };

    document.getElementById("po-search-btn")!.addEventListener("click", () => void doSearch());
    document.getElementById("po-clear-btn")!.addEventListener("click", () => {
      vendorInput.value = "";
      tableSection.style.display = "none";
      selectedSummary.style.display = "none";
      selectedPO = null;
      postPoBtn.setAttribute("disabled", "");
      setStatus("");
    });
    vendorInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); void doSearch(); }
    });

    // GL post button
    document.getElementById("po-post-gl-btn")!.addEventListener("click", async () => {
      const glAccount = glAccountInput?.value.trim() || "";
      close();
      const btn = document.getElementById("btn-post-fi");
      const statusSpan = document.getElementById("fi-post-status");
      if (btn) btn.setAttribute("disabled", "");
      if (statusSpan) { statusSpan.style.color = ""; statusSpan.textContent = "Posting to FI (GL)…"; }
      const payload: PostInvoiceRequest = {
        supplier_name:  fields.supplier_name,
        invoice_number: fields.invoice_number,
        invoice_date:   fields.invoice_date,
        total_amount:   fields.total_amount,
        currency:       fields.currency,
        ...(glAccount ? { gl_account: glAccount } : {}),
      };
      try {
        const result = await api.postSupplierInvoice(payload);
        this.state.fiPostResult = result;
        if (result.success) {
          if (statusSpan) { statusSpan.style.color = "var(--sapSuccessColor,#107e3e)"; statusSpan.textContent = `FI Doc: ${result.fi_document}`; }
          this.toast(`FI document ${result.fi_document} posted successfully`, "success");
        } else {
          if (statusSpan) { statusSpan.style.color = "var(--sapErrorColor,#bb0000)"; statusSpan.textContent = "Posting failed — see details"; }
          this._showErrorModal("FI Invoice Posting Failed", result.error);
        }
        this.render();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        if (statusSpan) { statusSpan.style.color = "var(--sapErrorColor,#bb0000)"; statusSpan.textContent = "Posting failed — see details"; }
        this._showErrorModal("FI Invoice Posting Failed", msg);
        if (btn) btn.removeAttribute("disabled");
      }
    });

    // PO post button
    document.getElementById("po-post-po-btn")!.addEventListener("click", async () => {
      if (!selectedPO) return;
      close();
      await this._postPOInvoice({
        supplier_name:  fields.supplier_name,
        invoice_number: fields.invoice_number,
        invoice_date:   fields.invoice_date,
        total_amount:   fields.total_amount,
        currency:       fields.currency,
        po_number:      selectedPO.purchase_order,
        po_item:        selectedItem,
      });
    });

    // Close handlers
    document.getElementById("po-cancel-btn")!.addEventListener("click", () => {
      close();
      const statusSpan = document.getElementById("fi-post-status");
      if (statusSpan) { statusSpan.style.color = "var(--sapNeutralColor,#6a6d70)"; statusSpan.textContent = "No PO selected. Click POST S4 to try again."; }
      const btn = document.getElementById("btn-post-fi");
      if (btn) btn.removeAttribute("disabled");
    });
    document.getElementById("po-modal-close")!.addEventListener("click", () => {
      close();
      const btn = document.getElementById("btn-post-fi");
      if (btn) btn.removeAttribute("disabled");
    });
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) {
        close();
        const btn = document.getElementById("btn-post-fi");
        if (btn) btn.removeAttribute("disabled");
      }
    });
  }

  private buildChatContext(): Record<string, unknown> {
    const r = this.state.pipelineResult;
    if (!r) return {};
    const llmS = r.llm_structured as LlmExtractionResult;
    const llmP = r.llm_prompting as LlmExtractionResult;
    return {
      route: r.route,
      supplier_name: llmS?.senderName ?? llmP?.senderName ?? r.routing_decision?.supplier_name ?? null,
      invoice_number: llmS?.documentNumber ?? llmP?.documentNumber ?? null,
      total_amount: llmS?.grossAmount ?? llmP?.grossAmount ?? null,
      currency: llmS?.currencyCode ?? llmP?.currencyCode ?? null,
      template_name: r.routing_decision?.template_name ?? null,
    };
  }

  private async sendChat(): Promise<void> {
    const chatInput = document.getElementById("chat-input") as HTMLTextAreaElement | null;
    if (!chatInput) return;
    const message = chatInput.value.trim();
    if (!message) return;

    chatInput.value = "";

    // Add user message
    this.state.chatMessages.push({ role: "user", content: message });
    this.state.streamingText = "";
    this.updateAssistantMessages();

    // Scroll to bottom
    const scroll = this.root.querySelector(".assistant-scroll");
    if (scroll) scroll.scrollTop = scroll.scrollHeight;

    try {
      const history: ChatMessage[] = this.state.chatMessages.slice(0, -1);
      const response = await api.streamChat({
        message,
        history,
        context: this.buildChatContext(),
      });

      if (!response.body) {
        this.state.chatMessages.push({ role: "assistant", content: "No response from assistant." });
        this.updateAssistantMessages();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const event = JSON.parse(trimmed) as { type: string; content?: string; error?: string };
            if (event.type === "delta" && event.content) {
              this.state.streamingText += event.content;
              this.updateAssistantMessages();
              const scrollEl = this.root.querySelector(".assistant-scroll");
              if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
            } else if (event.type === "done") {
              const finalText = this.state.streamingText || (event.content ?? "");
              this.state.chatMessages.push({ role: "assistant", content: finalText });
              this.state.streamingText = "";
              this.updateAssistantMessages();
              const scrollEl = this.root.querySelector(".assistant-scroll");
              if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
            } else if (event.type === "error") {
              this.state.chatMessages.push({ role: "assistant", content: `Error: ${event.error ?? "Unknown error"}` });
              this.state.streamingText = "";
              this.updateAssistantMessages();
            }
          } catch {
            // Non-JSON line — skip
          }
        }
      }

      // If streaming ended without a "done" event, finalize what we have
      if (this.state.streamingText) {
        this.state.chatMessages.push({ role: "assistant", content: this.state.streamingText });
        this.state.streamingText = "";
        this.updateAssistantMessages();
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.state.chatMessages.push({ role: "assistant", content: `Error: ${msg}` });
      this.state.streamingText = "";
      this.updateAssistantMessages();
    }
  }

  /** Update only the assistant messages area without full re-render */
  private updateAssistantMessages(): void {
    const container = document.getElementById("assistant-messages");
    if (container) container.innerHTML = this.buildMessagesHtml();
  }

  private toast(message: string, type: "success" | "error" | "info" = "info"): void {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const colors: Record<string, string> = {
      success: "var(--sapSuccessColor,#107e3e)",
      error: "var(--sapErrorColor,#bb0000)",
      info: "var(--sapInformationColor,#0070f2)",
    };
    const icons: Record<string, string> = { success: "OK", error: "X", info: "i" };
    const toast = document.createElement("div");
    toast.style.cssText = "background:var(--sapBaseColor,#fff);border:1px solid " + colors[type] + ";border-left:4px solid " + colors[type] + ";border-radius:0.375rem;padding:0.75rem 1rem;font-size:0.875rem;color:var(--sapTextColor,#32363a);box-shadow:0 4px 12px rgba(0,0,0,0.15);display:flex;align-items:center;gap:0.5rem;min-width:280px;max-width:400px;pointer-events:auto;";
    toast.innerHTML = "<span style=\"color:" + colors[type] + ";font-weight:700;\">[" + icons[type] + "]</span><span>" + message + "</span>";
    container.appendChild(toast);
    setTimeout(function() { toast.style.opacity = "0"; toast.style.transition = "opacity 0.3s"; setTimeout(function() { toast.remove(); }, 300); }, 4000);
  }

  private _showErrorModal(title: string, message: string): void {
    const existing = document.getElementById("fi-error-modal-backdrop");
    if (existing) existing.remove();

    // Extract the clean human-readable part — strip raw JSON details after "Details:"
    const cleanMessage = message
      .replace(/\s*Details:\s*\{.*$/s, "")
      .replace(/^Error:\s*/i, "")
      .trim();

    const backdrop = document.createElement("div");
    backdrop.id = "fi-error-modal-backdrop";
    backdrop.style.cssText = "position:fixed;inset:0;z-index:2000;background:rgba(29,45,62,0.48);display:flex;align-items:center;justify-content:center;";
    backdrop.innerHTML = `
      <div style="width:min(36rem,calc(100vw - 2rem));background:#fff;border-radius:0.5rem;box-shadow:0 1rem 3rem rgba(29,45,62,0.28);overflow:hidden;">
        <div style="display:flex;align-items:center;gap:0.75rem;padding:0.75rem 1rem;background:#fff1f1;border-bottom:1px solid var(--sapErrorBorderColor,#bb0000);">
          <span style="font-size:1.25rem;color:var(--sapErrorColor,#bb0000);">✕</span>
          <span style="font-weight:700;font-size:0.9375rem;color:var(--sapErrorColor,#bb0000);">${escapeHtml(title)}</span>
        </div>
        <div style="padding:1.25rem 1rem;">
          <p style="margin:0;font-size:0.875rem;color:var(--sapTextColor,#32363a);line-height:1.5;white-space:pre-wrap;">${escapeHtml(cleanMessage)}</p>
        </div>
        <div style="display:flex;justify-content:flex-end;padding:0.65rem 1rem;border-top:1px solid #d5dadd;background:#fafafa;">
          <ui5-button id="fi-error-close-btn" design="Emphasized">Close</ui5-button>
        </div>
      </div>`;

    document.body.appendChild(backdrop);

    const close = () => backdrop.remove();
    document.getElementById("fi-error-close-btn")!.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
  }
}
