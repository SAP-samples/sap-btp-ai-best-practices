import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({ gfm: true, breaks: false });

const INTERNAL_EVIDENCE_ID = "E-[0-9a-f]{16}";
const INTERNAL_EVIDENCE_LIST = new RegExp(
  `\\[\\s*${INTERNAL_EVIDENCE_ID}(?:\\s*,\\s*${INTERNAL_EVIDENCE_ID})*\\s*\\]`,
  "gi",
);
const INTERNAL_EVIDENCE_TOKEN = new RegExp(`\\b${INTERNAL_EVIDENCE_ID}\\b`, "gi");

/** Remove audit-only evidence handles from business-facing prose. */
export function stripInternalEvidenceIds(value) {
  return String(value || "")
    .replace(INTERNAL_EVIDENCE_LIST, "")
    .replace(INTERNAL_EVIDENCE_TOKEN, "supporting evidence")
    .replace(/[ \t]+([.,;:!?])/g, "$1")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

/** Parse GFM, remove executable markup, and return detached safe DOM. */
export function renderMarkdown(markdown) {
  const template = document.createElement("template");
  template.innerHTML = DOMPurify.sanitize(marked.parse(stripInternalEvidenceIds(markdown)), {
    FORBID_TAGS: ["img", "script", "iframe", "object", "embed", "form"],
    FORBID_ATTR: ["style", "srcset"],
  });

  for (const anchor of template.content.querySelectorAll("a")) {
    const rawHref = anchor.getAttribute("href") || "";
    let parsed;
    try {
      parsed = new URL(rawHref, window.location.href);
    } catch {
      parsed = null;
    }
    if (!parsed || !["http:", "https:"].includes(parsed.protocol)) {
      anchor.removeAttribute("href");
      anchor.removeAttribute("target");
      anchor.removeAttribute("rel");
    } else if (parsed.origin !== window.location.origin) {
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
    }
  }
  return template.content.cloneNode(true);
}
