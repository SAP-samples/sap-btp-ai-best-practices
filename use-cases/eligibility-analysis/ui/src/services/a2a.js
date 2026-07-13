import { request } from "./api.js";

function makeId(prefix) {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function extractTextFromTask(task) {
  const messageParts = task?.status?.message?.parts;
  if (Array.isArray(messageParts)) {
    const text = messageParts
      .map((part) => (part && typeof part.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("\n");
    if (text) return text;
  }

  const artifactParts = task?.artifacts?.[0]?.parts;
  if (Array.isArray(artifactParts)) {
    const text = artifactParts
      .map((part) => (part && typeof part.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("\n");
    if (text) return text;
  }

  return "";
}

export async function sendA2AUserMessage(text, { contextId, includeToolCalls = false } = {}) {
  const requestId = makeId("rpc");
  const messageId = makeId("msg");

  const params = {
    message: {
      role: "user",
      parts: [{ kind: "text", text }],
      messageId
    }
  };

  if (contextId) {
    params.message.contextId = contextId;
  }

  if (includeToolCalls) {
    params.metadata = { includeToolCalls: true };
  }

  const payload = {
    jsonrpc: "2.0",
    id: requestId,
    method: "message/send",
    params
  };

  const data = await request("/api/a2a", "POST", payload);

  if (data?.error) {
    const error = new Error(data.error.message || "A2A request failed");
    error.code = data.error.code;
    error.data = data.error.data;
    throw error;
  }

  const task = data?.result;
  const answer = extractTextFromTask(task);

  return {
    text: answer,
    task,
    contextId: task?.contextId || contextId
  };
}
