import { expect, it } from "vitest";
import { renderMarkdown, stripInternalEvidenceIds } from "../src/modules/markdown.js";

it("keeps GFM tables and safe links while removing executable content", () => {
  const host = document.createElement("div");
  host.append(renderMarkdown(`
| Part | Impact |
|---|---|
| 100 | **Changed** |

[local](/ecns/row-1) [external](https://example.test/x)
![tracker](https://example.test/pixel.png)
<script>window.pwned = true</script>
<a href="javascript:window.pwned=true" onclick="window.pwned=true">bad</a>
`));
  expect(host.querySelector("table")).not.toBeNull();
  expect(host.querySelector("img,script,[onclick]")).toBeNull();
  expect(host.querySelector('a[href^="javascript:"]')).toBeNull();
  expect(host.querySelector('a[href="/ecns/row-1"]').target).toBe("");
  const external = host.querySelector('a[href="https://example.test/x"]');
  expect(external.target).toBe("_blank");
  expect(external.rel).toContain("noopener");
  expect(external.rel).toContain("noreferrer");
});

it("removes audit-only evidence handles from business prose", () => {
  const text = stripInternalEvidenceIds(
    "Supported conclusion. [E-0123456789abcdef, E-fedcba9876543210] "
      + "See E-1111111111111111 for context.",
  );
  expect(text).toBe("Supported conclusion. See supporting evidence for context.");
  const host = document.createElement("div");
  host.append(renderMarkdown("Finding [E-0123456789abcdef]"));
  expect(host.textContent.trim()).toBe("Finding");
});
