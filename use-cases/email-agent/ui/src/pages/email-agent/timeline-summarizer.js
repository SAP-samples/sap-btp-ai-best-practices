// Deterministic helpers for summarizing timeline steps (Assistant tool calls and Tool results)

export function systemFromToolName(name) {
  if (!name) return null;
  if (name.startsWith("s4_")) return "S/4";
  if (name.startsWith("ariba_")) return "Ariba";
  if (name.startsWith("relish_")) return "Relish";
  if (name.startsWith("vendor_list_")) return "Vendor List";
  if (name === "email_template_get") return "Email Templates";
  if (name === "write_todos" || name === "read_todos") return "TODOs";
  if (name === "calculator_tool") return "Calculator";
  return null;
}

export function extractEntitiesFromArgs(args) {
  const invoices = new Set();
  const pos = new Set();
  const suppliers = new Set();
  const domains = new Set();
  const addValue = (k, v) => {
    if (v == null) return;
    const val = String(v);
    if (k.includes("invoice")) invoices.add(val);
    if (k === "q") {
      const nums = val.match(/\b\d{5,}\b/g);
      if (nums) nums.forEach((n) => invoices.add(n));
    }
    if (k.includes("po") || /(^|_)po(_|$)/.test(k)) pos.add(val);
    if (k === "domain") domains.add(val.toLowerCase());
    if (k.includes("supplier") || k.includes("vendor")) suppliers.add(val);
  };
  try {
    Object.entries(args || {}).forEach(([k, v]) => {
      if (Array.isArray(v)) v.forEach((x) => addValue(k, x));
      else addValue(k, v);
    });
  } catch (_) {
    // ignore parsing errors; fallback to empty sets
  }
  return { invoices: Array.from(invoices), pos: Array.from(pos), suppliers: Array.from(suppliers), domains: Array.from(domains) };
}

export function summarizeToolCalls(calls) {
  if (!Array.isArray(calls) || calls.length === 0) return "";
  const systemsSet = new Set();
  const invoices = new Set();
  const pos = new Set();
  const suppliers = new Set();
  const domains = new Set();

  calls.forEach((c) => {
    const sys = systemFromToolName(c?.name);
    if (sys) systemsSet.add(sys);
    const e = extractEntitiesFromArgs(c?.args || {});
    e.invoices.forEach((x) => invoices.add(x));
    e.pos.forEach((x) => pos.add(x));
    e.suppliers.forEach((x) => suppliers.add(x));
    e.domains.forEach((x) => domains.add(x));
  });

  const systems = Array.from(systemsSet);
  if (systems.length === 0) return "";

  const listLimit = (arr, n) => {
    const a = Array.from(arr);
    return a.length > n ? `${a.slice(0, n).join(", ")}…` : a.join(", ");
  };

  const parts = [];
  if (invoices.size) parts.push(`invoices ${listLimit(invoices, 5)}`);
  if (pos.size) parts.push(`POs ${listLimit(pos, 5)}`);
  if (!parts.length && domains.size) parts.push(`domain ${listLimit(domains, 1)}`);
  if (!parts.length && suppliers.size) parts.push(`supplier ${listLimit(suppliers, 2)}`);
  if (!parts.length) {
    // Generic fallback if we know the systems but not entities
    if (systems.length === 1 && systems[0] === "Vendor List") parts.push("supplier enablement");
    else parts.push("relevant records");
  }

  const sysText = systems.join(", ").replace(/, ([^,]*)$/, " and $1");
  const what = parts.join(" and ");
  return `Agent is checking ${sysText} for ${what}`;
}

export function tryParseJson(s) {
  if (typeof s !== "string") return null;
  try {
    return JSON.parse(s);
  } catch (_) {
    return null;
  }
}

export function tryExtractJsonFromText(text) {
  if (!text) return null;
  try {
    const fencedJson = text.match(/```json\s*([\s\S]*?)```/i);
    if (fencedJson) {
      const candidate = fencedJson[1].trim();
      const parsed = JSON.parse(candidate);
      return parsed;
    }
  } catch (_) {}
  try {
    const fenced = text.match(/```\s*([\s\S]*?)```/);
    if (fenced) {
      const candidate = fenced[1].trim();
      const parsed = JSON.parse(candidate);
      return parsed;
    }
  } catch (_) {}
  try {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start !== -1 && end !== -1 && end > start) {
      const candidate = text.slice(start, end + 1);
      const parsed = JSON.parse(candidate);
      return parsed;
    }
  } catch (_) {}
  return null;
}

export function countRecordsInPayload(payload) {
  if (payload == null) return null;
  try {
    if (Array.isArray(payload)) return payload.length;
    if (typeof payload === "object") {
      const candidates = [payload.results, payload.items, payload.data, payload.records, payload.rows].filter((x) => Array.isArray(x));
      if (candidates.length > 0) return candidates[0].length;
      return Object.keys(payload).length || 1;
    }
    return null;
  } catch (_) {
    return null;
  }
}

export function summarizeToolResult(toolCall, toolMessage) {
  const sys = systemFromToolName(toolCall?.name) || "Tool";
  const entities = extractEntitiesFromArgs(toolCall?.args || {});
  const parts = [];
  if (entities.invoices.length) parts.push(`invoices ${entities.invoices.slice(0, 5).join(", ")}${entities.invoices.length > 5 ? "…" : ""}`);
  if (entities.pos.length && parts.length === 0) parts.push(`POs ${entities.pos.slice(0, 5).join(", ")}${entities.pos.length > 5 ? "…" : ""}`);
  if (entities.domains.length && parts.length === 0) parts.push(`domain ${entities.domains[0]}`);
  if (entities.suppliers.length && parts.length === 0) parts.push(`supplier ${entities.suppliers.slice(0, 2).join(", ")}${entities.suppliers.length > 2 ? "…" : ""}`);

  let resultNote = "";
  let contentText = "";
  const content = toolMessage?.content;
  if (typeof content === "string") contentText = content;
  else if (Array.isArray(content)) {
    for (const part of content) {
      if (part?.type === "text" && typeof part.text === "string") {
        contentText += (contentText ? "\n" : "") + part.text;
      }
    }
  } else if (content != null) {
    try {
      contentText = JSON.stringify(content);
    } catch (_) {}
  }

  const parsed = tryParseJson(contentText);
  const count = countRecordsInPayload(parsed);
  if (typeof count === "number") {
    resultNote = count === 0 ? "no records" : `${count} record${count === 1 ? "" : "s"}`;
  } else if (contentText) {
    const t = contentText.trim();
    if (/not\s+found/i.test(t)) resultNote = "no records";
    else resultNote = "result";
  } else {
    resultNote = "result";
  }

  const what = parts.length ? ` for ${parts.join(" and ")}` : "";
  return `Result from ${sys}: ${resultNote}${what}`;
}

// UI helpers for timeline rendering
export function iconForType(type) {
  switch (type) {
    case "SystemMessage":
      return "settings";
    case "AIMessage":
      return "ai";
    case "HumanMessage":
      return "customer";
    case "ToolMessage":
    case "ToolMessageChunk":
      return "wrench";
    default:
      return "information";
  }
}

export function titleForType(type) {
  if (type === "AIMessage") return "Assistant";
  if (type === "HumanMessage") return "Human";
  if (type?.startsWith?.("Tool")) return "Tool";
  if (type === "SystemMessage") return "System";
  return type || "Message";
}

export function stateForType(type) {
  if (type === "AIMessage") return "Information";
  if (type?.startsWith?.("Tool")) return "Critical";
  if (type === "SystemMessage") return "";
  if (type === "HumanMessage") return "Positive";
  return "Information";
}

export function toDisplay(content, toolCalls) {
  let text = "";
  const calls = [];
  if (typeof content === "string") {
    text = content;
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (part?.type === "text" && typeof part.text === "string") {
        text += (text ? "\n" : "") + part.text;
      } else if (part?.type === "tool_use") {
        calls.push({ name: part.name, args: part.input, id: part.id });
      }
    }
  } else if (content != null) {
    try {
      text = JSON.stringify(content, null, 2);
    } catch (_) {
      text = String(content);
    }
  }
  if (Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      calls.push({ name: tc?.name, args: tc?.args, id: tc?.id });
    }
  }
  return { text, calls };
}

// Subtitles for non-tool steps
export function humanSubtitleFromText(text) {
  let subtitle = "Provided email bundle";
  const metaMatch = (text || "").match(/\[METADATA\]\s*([\s\S]*?)\s*\[\/METADATA\]/i);
  if (metaMatch) {
    const md = tryParseJson(metaMatch[1].trim());
    if (md && typeof md === "object") {
      const subj = md.subject || md.Subject;
      const from = md.from || md.From;
      const att = md.attachments || md.Attachments;
      const count = (att && (att.count || att.Count)) || 0;
      if (subj && from) subtitle = `Email from ${from} — “${subj}”${count ? ` (${count} attachment${count === 1 ? "" : "s"})` : ""}`;
      else if (subj) subtitle = `Selected email: “${subj}”${count ? ` (${count} attachment${count === 1 ? "" : "s"})` : ""}`;
    }
  }
  return subtitle;
}

export function assistantSubtitleFromText(text) {
  const json = tryExtractJsonFromText(text || "");
  if (json && typeof json === "object") {
    const decision = json.decision || json.Decision;
    const subject = json.replySubject || json.ReplySubject;
    const moveToFolder = json.moveToFolder || json.MoveToFolder;
    const needs = json.needsHumanReview;
    const yesNo = (v) => (v === true ? "Yes" : v === false ? "No" : "");
    const pieces = [];
    if (decision) pieces.push(`Decision: ${decision}`);
    if (typeof needs === "boolean") pieces.push(`Needs review: ${yesNo(needs)}`);
    if (subject) pieces.push(`Subject: “${subject}”`);
    if (moveToFolder) pieces.push(`Folder: ${moveToFolder}`);
    if (pieces.length > 0) return pieces.join(" — ");
    return "Assistant produced final JSON result";
  }
  return "Assistant response";
}
