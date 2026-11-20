import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { snapshotWorkspace } from "../src/components/snapshot-workspace.js";
import { ecnChat } from "../src/components/ecn-chat.js";
import { mountShellChat, resetShellChat, setChatContext } from "../src/modules/shell-chat.js";
import detailHtml from "../src/pages/ecn-detail/ecn-detail.html?raw";
import * as api from "../src/services/workspace-api.js";

vi.mock("../src/services/workspace-api.js", () => ({ getSnapshot:vi.fn(), listChanges:vi.fn(), listObjects:vi.fn(), getObject:vi.fn(), listReviews:vi.fn(),saveReview:vi.fn(),getAcceptancePlan:vi.fn(),versions:vi.fn(),reassess:vi.fn(),chat:vi.fn(),globalChat:vi.fn() }));
const flush = () => new Promise(resolve => setTimeout(resolve, 0));
const id="11111111-1111-4111-8111-111111111111";
const snapshot={assessment_id:id,evidence_hash:"a".repeat(64),version:"ecn-snapshot-v1",counts:{technical_objects:1,supporting_objects:0,bom_change_objects:1,located_objects:1,unlocated_objects:0},questions:[],limitations:[]};

/** Render the detail template with a shell chat, as the application does. */
function mountPage() {
  document.body.innerHTML = `<div id="app"><button id="assistant-shell-action" type="button" hidden>Open ECN chat</button><main class="content">${detailHtml}</main></div>`;
  mountShellChat(document.querySelector("#app"));
  return document.querySelector(".ecn-detail");
}

beforeEach(() => {
  vi.clearAllMocks(); resetShellChat(); document.body.replaceChildren();
  vi.stubGlobal("localStorage", {setItem:vi.fn(),getItem:vi.fn()});
  vi.stubGlobal("sessionStorage", {setItem:vi.fn(),getItem:vi.fn()});
  HTMLDialogElement.prototype.showModal=function(){this.open=true;};
  HTMLDialogElement.prototype.close=function(){this.open=false;};
  api.getSnapshot.mockResolvedValue(snapshot); api.versions.mockResolvedValue([]); api.listReviews.mockResolvedValue([]);
  api.getAcceptancePlan.mockResolvedValue({blocked:false});
});
afterEach(() => { vi.unstubAllGlobals(); resetShellChat(); });

it("binds the shell chat to the ECN and keeps the draft when collapsed and reopened", async () => {
 const root=mountPage();const controller=new AbortController();
 await snapshotWorkspace(root,id,controller.signal).ready;
 const panel=document.querySelector("#ecn-assistant-panel");
 const shellAction=document.querySelector("#assistant-shell-action");
 const chatIcon=panel.querySelector(".assistant-title ui5-icon");
 expect(chatIcon?.getAttribute("name")).toBe("discussion-2");
 expect(chatIcon?.getAttribute("aria-hidden")).toBe("true");
 expect(panel.hidden).toBe(true);
 shellAction.click();
 expect(panel.hidden).toBe(false);
 expect(panel.querySelector("textarea").disabled).toBe(false);
 panel.querySelector("textarea").value="Which BOMs use this part?";
 panel.querySelector(".assistant-close").click();
 expect(panel.hidden).toBe(true);
 expect(document.querySelector(".assistant-launcher")).toBeNull();
 expect(shellAction.hidden).toBe(false);
 expect(shellAction.accessibilityAttributes.expanded).toBe("false");
 shellAction.click();
 expect(panel.hidden).toBe(false);
 expect(panel.querySelector("textarea").value).toBe("Which BOMs use this part?");
 document.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));
 expect(panel.hidden).toBe(true);
 controller.abort();
});

it("offers the global chat outside an ECN and returns to it after leaving one", async () => {
 const root=mountPage();
 const panel=document.querySelector("#ecn-assistant-panel");
 expect(panel.hidden).toBe(true);
 expect(panel.textContent).toContain("Ask about any stored ECN");
 expect(panel.querySelector("textarea").disabled).toBe(false);
 const controller=new AbortController();
 await snapshotWorkspace(root,id,controller.signal).ready;
 expect(panel.textContent).not.toContain("Ask about any stored ECN");
 controller.abort();
 expect(panel.textContent).toContain("Ask about any stored ECN");
 expect(panel.querySelector("textarea").disabled).toBe(false);
});

it("sends global questions to /api/chat and links the ECNs the answer read", async () => {
 mountPage();
 api.globalChat.mockResolvedValue({answer:"ECN1 has high impact.",consulted_objects:[],consulted_ecns:[{ecn_number:"ECN1",assessment_id:"row-1"}]});
 const panel=document.querySelector("#ecn-assistant-panel");
 panel.querySelector("textarea").value="Which ECNs have high impact?";
 panel.querySelector(".chat-composer").requestSubmit();await flush();
 expect(api.globalChat).toHaveBeenCalledWith({message:"Which ECNs have high impact?",history:[]},expect.any(AbortSignal));
 expect(api.chat).not.toHaveBeenCalled();
 const link=[...panel.querySelectorAll(".chat-sources a")].find(a=>a.textContent==="ECN1");
 expect(link.getAttribute("href")).toBe("/ecns/row-1");
});

it("uses a named shell action with nonmodal focus in a narrow viewport", async () => {
 const mediaQuery = vi.fn((query) => ({ matches: query === "(max-width: 600px)" }));
 vi.stubGlobal("matchMedia", mediaQuery);
 const root=mountPage();const controller=new AbortController();
 await snapshotWorkspace(root,id,controller.signal).ready;
 const panel=document.querySelector("#ecn-assistant-panel");
 const shellAction=document.querySelector("#assistant-shell-action");
 expect(panel.hidden).toBe(true);
 expect(document.querySelector(".assistant-launcher")).toBeNull();
 expect(shellAction.hidden).toBe(false);
 shellAction.click();
 expect(panel.hidden).toBe(false);
 expect(panel.getAttribute("role")).toBe("complementary");
 expect(panel.hasAttribute("aria-modal")).toBe(false);
 expect(document.activeElement).toBe(panel.querySelector("textarea"));
 document.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));
 expect(panel.hidden).toBe(true);
 expect(document.activeElement).toBe(shellAction);
 controller.abort();
});

it("focuses the close control when chat is unavailable", () => {
 const mediaQuery = vi.fn((query) => ({ matches: query === "(max-width: 600px)" }));
 vi.stubGlobal("matchMedia", mediaQuery);
 mountPage();
 setChatContext({ id, signal: new AbortController().signal, inspect: vi.fn(), graphReady: false });
 const panel=document.querySelector("#ecn-assistant-panel");
 const shellAction=document.querySelector("#assistant-shell-action");
 shellAction.click();
 expect(panel.querySelector("textarea").disabled).toBe(true);
 expect(document.activeElement).toBe(panel.querySelector(".assistant-close"));
});

it("does not trap keyboard Tab focus inside the floating panel", async () => {
 const root=mountPage();const controller=new AbortController();
 await snapshotWorkspace(root,id,controller.signal).ready;
 const panel=document.querySelector("#ecn-assistant-panel");
 document.querySelector("#assistant-shell-action").click();
 panel.querySelector(".assistant-close").focus();
 const tab=new KeyboardEvent("keydown",{key:"Tab",bubbles:true,cancelable:true});
 document.dispatchEvent(tab);
 expect(tab.defaultPrevented).toBe(false);
 expect(document.querySelector(".assistant-backdrop")).toBeNull();
 controller.abort();
});

it("starts with the chat collapsed in a narrow viewport", async () => {
 const mediaQuery = vi.fn((query) => ({ matches: query === "(max-width: 600px)" }));
 vi.stubGlobal("matchMedia", mediaQuery);
 const root=mountPage();const controller=new AbortController();
 await snapshotWorkspace(root,id,controller.signal).ready;
 expect(document.querySelector("#ecn-assistant-panel").hidden).toBe(true);
 expect(document.querySelector(".assistant-launcher")).toBeNull();
 expect(document.querySelector("#assistant-shell-action").hidden).toBe(false);
 controller.abort();
});

it("records a send-back as a safe human-decision Audit Log event", async () => {
   const root=mountPage();const controller=new AbortController();
   const workspace=snapshotWorkspace(root,id,controller.signal); await workspace.ready;
   const reviewApi=await workspace.reviewApi;
   reviewApi.setReviewerName("Engineer");
   api.saveReview.mockResolvedValue([{reviewer:"Engineer",outcome:"CHANGES_REQUESTED",comments:'<img src=x onerror="alert(1)">',created_at:"2026-09-16",snapshot_version:snapshot.version,evidence_hash:snapshot.evidence_hash}]);
   await reviewApi.reject('<img src=x onerror="alert(1)">');
   expect(api.saveReview.mock.lastCall[1]).toMatchObject({outcome:"CHANGES_REQUESTED",evidence_hash:snapshot.evidence_hash,snapshot_version:snapshot.version});
   const audit=root.querySelector("#audit-log");
   expect(audit.querySelector("img")).toBeNull();
   expect(audit.textContent).toContain("Human decision");
   expect(audit.textContent).toContain("Engineer");
   expect(audit.textContent).toContain("Rejected");
   expect(audit.textContent).toContain('<img src=x onerror="alert(1)">');
   controller.abort();
});

it("accepts through the Audit Log review API and retains earlier decisions", async () => {
  const root=mountPage();const controller=new AbortController();
  const oldReview={reviewer:"Old",outcome:"CORRECTION_RECORDED",comments:"legacy",created_at:"2026-09-01"};
  api.listReviews.mockResolvedValue([oldReview]);
   vi.stubGlobal("confirm",vi.fn(()=>true));
   const workspace=snapshotWorkspace(root,id,controller.signal); await workspace.ready;
   const audit=root.querySelector("#audit-log");
   await vi.waitFor(() => expect(audit.textContent).toContain("Correction recorded"));
   const reviewApi=await workspace.reviewApi;
   expect(reviewApi.accept()).toBe(false);
   expect(api.saveReview).not.toHaveBeenCalled();
   reviewApi.setReviewerName("Engineer");
   api.saveReview.mockResolvedValue([oldReview,{reviewer:"Engineer",outcome:"ACCEPTED",comments:"",created_at:"2026-09-29"}]);
   await reviewApi.accept();
   expect(confirm).toHaveBeenCalledOnce();
   expect(api.saveReview.mock.lastCall[1]).toMatchObject({outcome:"ACCEPTED",comments:""});
   expect(audit.textContent).toContain("Accepted");
   expect(audit.textContent).toContain("Correction recorded");
   controller.abort();
});

it("keeps chat in memory per ECN and sends only that ECN's conversation", async () => {
 const first=document.createElement("div");document.body.append(first);const controller=new AbortController();
 api.chat.mockResolvedValue({answer:"Observed <script>alert(1)</script>",consulted_objects:[]});
 ecnChat(first,"chat-a",controller.signal,vi.fn());first.querySelector("textarea").value="Question A";
 first.querySelector(".chat-composer").requestSubmit();await flush();
 const second=document.createElement("div");document.body.append(second);ecnChat(second,"chat-b",controller.signal,vi.fn());
 expect(second.textContent).not.toContain("Question A");second.querySelector("textarea").value="Question B";
 second.querySelector(".chat-composer").requestSubmit();await flush();
 expect(api.chat.mock.lastCall[1]).toMatchObject({assessment_id:"chat-b",history:[]});
 expect(document.querySelector("script")).toBeNull();
 const reopened=document.createElement("div");ecnChat(reopened,"chat-a",controller.signal,vi.fn());
 expect(reopened.textContent).toContain("Question A");
 expect(localStorage.setItem).not.toHaveBeenCalled();expect(sessionStorage.setItem).not.toHaveBeenCalled();controller.abort();
});

it("keeps the conversation of an ECN when the shell chat is rebound to it", async () => {
 mountPage();
 api.chat.mockResolvedValue({answer:"Answer one",consulted_objects:[]});
 const first=new AbortController();
 setChatContext({id:"chat-rebind",signal:first.signal,inspect:vi.fn(),graphReady:true});
 const panel=document.querySelector("#ecn-assistant-panel");
 panel.querySelector("textarea").value="First question";
 panel.querySelector(".chat-composer").requestSubmit();await flush();
 first.abort();
 expect(panel.textContent).not.toContain("First question");
 setChatContext({id:"chat-rebind",signal:new AbortController().signal,inspect:vi.fn(),graphReady:true});
 expect(panel.textContent).toContain("First question");
});

it("keeps technical tool traces out of the simplified conversation", async () => {
 const host=document.createElement("div");document.body.append(host);const controller=new AbortController();
 api.chat.mockResolvedValue({answer:"Creator found.",consulted_objects:[],tool_trace:[{sequence:1,tool:"list_ecn_fields",arguments:{query:"created",page:1},duration_ms:12,outcome:"success",result:{total_matches:1,page:1,has_next:false}}]});
 ecnChat(host,"chat-trace",controller.signal,vi.fn());host.querySelector("textarea").value="Who created it?";
 host.querySelector(".chat-composer").requestSubmit();await flush();
 expect(host.querySelector(".chat-assistant").textContent).toContain("Creator found.");
 expect(host.querySelector(".chat-tool-trace")).toBeNull();
 expect(host.textContent).not.toContain("list_ecn_fields");
 controller.abort();
});

it("renders source import, analysis, assessment version, and review Audit Log events", async () => {
 const root=mountPage();const controller=new AbortController();
 api.getSnapshot.mockResolvedValue({...snapshot,assessment_date:"2026-09-23"});
 api.versions.mockResolvedValue([
   {ecn_row_id:id,created_at:"2026-09-23T10:00:00Z"},
   {ecn_row_id:"earlier-assessment",created_at:"2026-09-01T10:00:00Z"},
 ]);
 api.listReviews.mockResolvedValue([{reviewer:"Ana",outcome:"CHANGES_REQUESTED",comments:"Confirm PL10 date",created_at:"2026-09-30"}]);
 const auditDetail={
   uploaded_at:"2026-09-14T08:00:00Z",
   ecn_artifact:{file_name:"ecn.json"},
   analysis:{created_at:"2026-09-14T08:01:00Z",completeness:"COMPLETED_WITH_GAPS"},
 };
 await snapshotWorkspace(root,id,controller.signal,{auditDetail}).ready;
 const audit=root.querySelector("#audit-log");
 await vi.waitFor(() => expect(audit.textContent).toContain("Confirm PL10 date"));
 expect(audit.textContent).toContain("Source import");
 expect(audit.textContent).toContain("ECN source imported");
 expect(audit.textContent).toContain("ecn.json");
 expect(audit.textContent).toContain("Generated analysis");
 expect(audit.textContent).toContain("Assessment status: Completed With Gaps");
 expect(audit.textContent).toContain("Assessment version created");
 expect(audit.textContent).toContain("Assessed as of 2026-09-23");
 expect(audit.textContent).toContain("Earlier assessment version");
 expect(audit.textContent).toContain("Human decision");
 expect(audit.textContent).toContain("Ana");
 expect(audit.textContent).toContain("Rejected");
 controller.abort();
});

it("reloads the Audit Log when the assistant saves a confirmed review", async () => {
 const root=mountPage();const controller=new AbortController();
 const workspace=snapshotWorkspace(root,id,controller.signal);await workspace.ready;await flush();
 expect(api.listReviews).toHaveBeenCalledOnce();
 api.listReviews.mockResolvedValue([{reviewer:"Ana",outcome:"CHANGES_REQUESTED",comments:"Confirm PL10 date",created_at:"2026-09-30"}]);
 api.chat.mockResolvedValue({answer:"Saved.",consulted_objects:[],review_saved:true});
 const chat=document.createElement("div");document.body.append(chat);ecnChat(chat,id,controller.signal,vi.fn());
  chat.querySelector("textarea").value="yes";
  chat.querySelector(".chat-composer").requestSubmit();await flush();await flush();
  expect(api.listReviews).toHaveBeenCalledTimes(2);
  expect(root.querySelector("#audit-log").textContent).toContain("Confirm PL10 date");
  controller.abort();
});
