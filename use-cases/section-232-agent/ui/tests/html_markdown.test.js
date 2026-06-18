import test from "node:test";
import assert from "node:assert/strict";

import { renderMarkdownHtml } from "../src/utils/html.js";

test("renderMarkdownHtml renders candidate reasoning markdown as sanitized HTML", () => {
  const markdown = [
    "### **8414.90.9180**",
    "",
    "Confidence: 58.00%",
    "- Air or vacuum pumps",
    "- Parts",
    "",
    "Potential **alternative** if impeller is for air/gas usage.",
    "",
    "---",
    "",
    "<script>alert('xss')</script>"
  ].join("\n");

  assert.equal(
    renderMarkdownHtml(markdown),
    [
      "<h3><strong>8414.90.9180</strong></h3>",
      "<p>Confidence: 58.00%</p>",
      "<ul><li>Air or vacuum pumps</li><li>Parts</li></ul>",
      "<p>Potential <strong>alternative</strong> if impeller is for air/gas usage.</p>",
      "<hr>",
      "<p>&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;</p>"
    ].join("")
  );
});
