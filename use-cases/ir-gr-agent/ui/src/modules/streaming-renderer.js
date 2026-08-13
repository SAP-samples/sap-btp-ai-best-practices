import { marked } from "marked";
import DOMPurify from "dompurify";

/** Single authoritative place for markdown → safe HTML. */
export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text));
}

/**
 * Buffers streaming text tokens and renders markdown once on finish(),
 * avoiding O(n²) re-parses during streaming. All agent output goes
 * through DOMPurify before touching the DOM.
 *
 * Usage:
 *   const r = createStreamingRenderer(bubbleElement);
 *   r.append(chunk);   // call per token — no parse, no layout
 *   r.finish();        // parse + sanitize + set innerHTML once
 */
export function createStreamingRenderer(container) {
  let buf = "";

  return {
    append(chunk) {
      buf += chunk;
      // Show raw text while streaming — no markdown parse, no layout thrash
      container.textContent = buf;
    },
    finish() {
      container.innerHTML = renderMarkdown(buf);
    },
    get text() {
      return buf;
    },
  };
}
