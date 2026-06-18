export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * Renders inline markdown emphasis after escaping unsafe HTML characters.
 *
 * @param {string} value - Inline markdown text to render.
 * @returns {string} HTML-safe inline content with strong and emphasis tags.
 */
function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

/**
 * Converts trusted application markdown text into sanitized HTML for UI display.
 *
 * @param {string | null | undefined} value - Markdown text from the application state or API response.
 * @returns {string} Sanitized HTML containing paragraphs, headings, lists, rules, and inline emphasis.
 */
export function renderMarkdownHtml(value) {
  const markdown = String(value ?? "").replace(/\r\n?/g, "\n").trim();
  if (!markdown) return "";

  const lines = markdown.split("\n");
  const blocks = [];
  const paragraphLines = [];

  /**
   * Flushes buffered paragraph text into the rendered block list.
   *
   * @returns {void}
   */
  function flushParagraph() {
    if (!paragraphLines.length) return;
    blocks.push(`<p>${paragraphLines.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraphLines.length = 0;
  }

  for (let index = 0; index < lines.length; index += 1) {
    const trimmedLine = lines[index].trim();

    if (!trimmedLine) {
      flushParagraph();
      continue;
    }

    if (/^-{3,}$/.test(trimmedLine)) {
      flushParagraph();
      blocks.push("<hr>");
      continue;
    }

    const headingMatch = trimmedLine.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      const level = headingMatch[1].length;
      blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    if (/^[-*]\s+/.test(trimmedLine)) {
      const items = [];
      flushParagraph();

      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }

      index -= 1;
      blocks.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    paragraphLines.push(trimmedLine);
  }

  flushParagraph();
  return blocks.join("");
}
