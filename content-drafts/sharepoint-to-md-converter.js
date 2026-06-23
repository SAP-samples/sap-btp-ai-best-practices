/**
 * SharePoint Page → Markdown Converter
 *
 * Usage:
 *   1. Open the SharePoint page in Chrome/Edge
 *   2. Open DevTools console (F12 → Console tab)
 *   3. Paste this entire script and press Enter
 *   4. Run: await copySharePointPageAsMarkdown()
 *   5. The markdown is copied to your clipboard (or a dialog appears)
 *   6. Paste into a new .md file in this content-drafts/ folder
 *
 * Options (pass as object):
 *   await copySharePointPageAsMarkdown({ includeImages: false })
 *   await copySharePointPageAsMarkdown({ includeFrontMatter: true })
 *   await copySharePointPageAsMarkdown({ rootSelector: '#my-section' })
 */

(() => {
  const DEFAULT_OPTIONS = {
    rootSelector: null,
    includeImages: true,
    includeLinks: true,
    includePageTitle: true,
    includeSourceUrl: true,
    includeFrontMatter: false,
    removeSharePointChrome: true,
    preferSelection: false,

    skipBlobImages: true,
    skipDataImages: true,
    unexportableImageHandling: "placeholder", // "placeholder" | "skip"
    imagePlaceholderText: "Image not exported from SharePoint",

    debug: false
  };

  const BLOCK_TAGS = new Set([
    "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "CANVAS", "DD", "DETAILS",
    "DIV", "DL", "DT", "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM",
    "H1", "H2", "H3", "H4", "H5", "H6", "HEADER", "HR", "LI", "MAIN", "NAV",
    "OL", "P", "PRE", "SECTION", "SUMMARY", "TABLE", "TBODY", "TD", "TFOOT",
    "TH", "THEAD", "TR", "UL"
  ]);

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "META", "LINK", "SVG"
  ]);

  const SP_CHROME_SELECTORS = [
    "#SuiteNavWrapper",
    "#spSiteHeader",
    "#spLeftNav",
    "#spCommandBar",
    "#CommentsWrapper",

    "[data-automation-id='SiteHeader']",
    "[data-automation-id='pageCommandBar']",
    "[data-automation-id='commandBar']",
    "[data-automation-id='CommentsWrapper']",
    "[data-sp-feature-tag='Comments']",

    "header[role='banner']",
    "footer",

    ".ms-CommandBar",
    ".od-SuiteNav",
    ".sp-appBar",
    ".spLeftNav",
    ".spCommandBar",
    ".ms-Dialog",
    ".ms-Panel",
    ".ms-Callout",
    ".ms-ContextualMenu",
    ".ms-Layer",
    ".ms-TooltipHost"
  ];

  function absoluteUrl(url) {
    if (!url) return "";
    try {
      return new URL(url, window.location.href).href;
    } catch {
      return url;
    }
  }

  function normalizeSpaces(text) {
    return String(text || "")
      .replace(/ /g, " ")
      .replace(/[ \t\f\v]+/g, " ")
      .replace(/\n[ \t]+/g, "\n")
      .replace(/[ \t]+\n/g, "\n");
  }

  function cleanText(text) {
    return normalizeSpaces(text)
      .replace(/\s+/g, " ")
      .trim();
  }

  function escapeMarkdownText(text) {
    return String(text || "")
      .replace(/\\/g, "\\\\")
      .replace(/`/g, "\\`")
      .replace(/\[/g, "\\[")
      .replace(/\]/g, "\\]");
  }

  function escapeInline(text) {
    return escapeMarkdownText(cleanText(text));
  }

  function escapeInlinePreserveSpaces(text) {
    return escapeMarkdownText(normalizeSpaces(text));
  }

  function escapeTableCell(text) {
    return cleanText(text)
      .replace(/\|/g, "\\|")
      .replace(/\n+/g, "<br>");
  }

  function codeFence(text, language = "") {
    const raw = String(text || "").replace(/\n+$/g, "");
    const longestTicks = Math.max(
      2,
      ...(raw.match(/`+/g) || []).map(x => x.length)
    );

    const fence = "`".repeat(Math.max(3, longestTicks + 1));
    return `\n\n${fence}${language}\n${raw}\n${fence}\n\n`;
  }

  function isProbablyHidden(el) {
    if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;

    if (
      el.hidden ||
      el.getAttribute("aria-hidden") === "true" ||
      el.getAttribute("data-is-focusable") === "false"
    ) {
      return true;
    }

    const style = (el.getAttribute("style") || "").toLowerCase();

    return (
      style.includes("display: none") ||
      style.includes("visibility: hidden") ||
      style.includes("opacity: 0")
    );
  }

  function removeNoise(root, options) {
    const skipSelector = Array.from(SKIP_TAGS).join(",");

    if (skipSelector) {
      root.querySelectorAll(skipSelector).forEach(n => n.remove());
    }

    root.querySelectorAll("[hidden], [aria-hidden='true']").forEach(n => n.remove());

    root.querySelectorAll("*").forEach(el => {
      const style = (el.getAttribute("style") || "").toLowerCase();

      if (
        style.includes("display: none") ||
        style.includes("visibility: hidden")
      ) {
        el.remove();
      }
    });

    if (options.removeSharePointChrome) {
      SP_CHROME_SELECTORS.forEach(selector => {
        root.querySelectorAll(selector).forEach(n => n.remove());
      });
    }

    root.querySelectorAll("button, [role='button']").forEach(button => {
      const text = cleanText(button.innerText || button.textContent);
      const aria = cleanText(button.getAttribute("aria-label"));
      const title = cleanText(button.getAttribute("title"));
      const combined = [text, aria, title].join(" ");

      const likelyContentButton =
        text.length > 30 ||
        button.closest("details") ||
        button.closest("[data-automation-id='Canvas']");

      const likelyChromeButton =
        /edit|share|follow|copy link|like|comment|analytics|details|settings|new|upload|sync|automate|integrate|export|open in|expand|collapse|more/i
          .test(combined);

      if (likelyChromeButton && !likelyContentButton) {
        button.remove();
      }
    });

    return root;
  }

  function getPageTitle() {
    const candidates = [
      "[data-automation-id='TitleTextId']",
      "[data-automation-id='pageHeader'] h1",
      "h1",
      "title"
    ];

    for (const selector of candidates) {
      const el = document.querySelector(selector);
      const text = el ? cleanText(el.innerText || el.textContent) : "";

      if (text) return text;
    }

    return cleanText(document.title || "")
      .replace(/\s*-\s*SharePoint\s*$/i, "");
  }

  function hasBlockChild(el) {
    if (!el || !el.childNodes) return false;

    return Array.from(el.childNodes).some(child => {
      return child.nodeType === Node.ELEMENT_NODE && BLOCK_TAGS.has(child.tagName);
    });
  }

  function repairInlineSpacing(text) {
    let s = normalizeSpaces(text);

    s = s.replace(/`([^`\n]*)`/g, (_, code) => {
      return "`" + code.trim() + "`";
    });

    s = s.replace(/\*\*([^*\n]+?)\*\*/g, (_, bold) => {
      return "**" + bold.trim() + "**";
    });

    s = s.replace(/([^\s\[(])(\*\*[^*\n]+?\*\*)/g, "$1 $2");
    s = s.replace(/(\*\*[^*\n]+?\*\*)([A-Za-z0-9(])/g, "$1 $2");

    s = s.replace(/([^\s\[(])(`[^`\n]+?`)/g, "$1 $2");
    s = s.replace(/(`[^`\n]+?`)([A-Za-z0-9(])/g, "$1 $2");

    s = s.replace(/(\]\([^)]+\))(\()/g, "$1 $2");
    s = s.replace(/(\]\([^)]+\))([A-Za-z0-9(])/g, "$1 $2");

    s = s.replace(/\s+([.,;:!?])/g, "$1");
    s = s.replace(/\(\s+/g, "(");
    s = s.replace(/\s+\)/g, ")");
    s = s.replace(/[ \t]{2,}/g, " ");

    return s.trim();
  }

  function inlineChildren(el, options) {
    return repairInlineSpacing(
      Array.from(el.childNodes)
        .map(node => renderInline(node, options))
        .join("")
    );
  }

  function getImageSource(el) {
    const candidates = [
      el.currentSrc,
      el.getAttribute("src"),
      el.getAttribute("data-src"),
      el.getAttribute("data-originalsrc"),
      el.getAttribute("data-sp-originalimgsrc"),
      el.getAttribute("data-fullres-src"),
      el.getAttribute("data-imageurl")
    ].filter(Boolean);

    return candidates[0] || "";
  }

  function renderUnexportableImage(el, options, reason) {
    if (options.unexportableImageHandling === "skip") return "";

    const alt = cleanText(
      el.getAttribute("alt") ||
      el.getAttribute("aria-label") ||
      el.getAttribute("title") ||
      "image"
    );

    const label = alt && alt !== "image"
      ? `${options.imagePlaceholderText}: ${alt}`
      : options.imagePlaceholderText;

    return `_[${escapeInline(label)} (${reason})]_`;
  }

  function cleanCodeLine(text) {
    return String(text || "")
      .replace(/ /g, " ")
      .replace(/​/g, "")
      .replace(/\r/g, "")
      .replace(/[ \t]+$/g, "");
  }

  function getPreviousTechHeading(container) {
    if (!container) return "";

    const root =
      container.closest?.("[data-automation-id='Canvas'], article, main, [role='main']") ||
      container.ownerDocument?.body ||
      document.body;

    const headings = Array.from(root.querySelectorAll("h1,h2,h3,h4,h5,h6"));
    let lastTechHeading = "";

    for (const heading of headings) {
      if (heading === container || heading.contains(container)) break;

      const position = heading.compareDocumentPosition(container);
      const isBefore = Boolean(position & Node.DOCUMENT_POSITION_FOLLOWING);

      if (!isBefore) continue;

      const text = cleanText(heading.innerText || heading.textContent).toLowerCase();

      if (
        text === "python" ||
        text === "java" ||
        text.includes("javascript") ||
        text.includes("typescript")
      ) {
        lastTechHeading = text;
      }
    }

    return lastTechHeading;
  }

  function inferCodeLanguage(code, container) {
    const cls = [
      String(container?.className || ""),
      String(container?.parentElement?.className || ""),
      String(container?.closest?.("[class]")?.className || "")
    ].join(" ").toLowerCase();

    const c = String(code || "");
    const previousTechHeading = getPreviousTechHeading(container);

    if (cls.includes("xml") || /^\s*</.test(c)) return "xml";
    if (cls.includes("json") || /^\s*[\[{]/.test(c)) return "json";
    if (previousTechHeading === "python") return "python";
    if (previousTechHeading === "java") return "java";
    if (previousTechHeading.includes("typescript")) return "ts";
    if (previousTechHeading.includes("javascript")) return "js";

    if (cls.includes("typescript") || /\b(import|export|interface|type)\s+\w+/m.test(c)) return "ts";
    if (cls.includes("python") || /\b(def|import|from)\s+\w+/m.test(c)) return "python";
    if (cls.includes("java") || /\b(public|private|protected|class|interface)\s+\w+/m.test(c)) return "java";
    if (cls.includes("javascript") || /\b(const|let|var|function)\s+\w+/m.test(c)) return "js";
    if (cls.includes("bash") || cls.includes("shell") || /^\s*(npm|yarn|pnpm|cd|curl|export|echo|mkdir)\s+/m.test(c)) return "bash";
    if (/^\s*<dependency[\s>]/m.test(c)) return "xml";
    if (/^\s*@(Before|After|Test|Override)\b/m.test(c)) return "java";
    if (/^\s*SELECT\s+/im.test(c)) return "sql";

    return "";
  }

  function makePreFromCode(code, language = "") {
    const pre = document.createElement("pre");
    const codeEl = document.createElement("code");

    if (language) {
      codeEl.className = `language-${language}`;
    }

    codeEl.textContent = String(code || "").replace(/\n+$/g, "");
    pre.appendChild(codeEl);
    pre.setAttribute("data-md-extracted-code", "true");

    return pre;
  }

  function extractCodeMirrorCode(cm) {
    if (!cm) return "";

    if (
      cm.CodeMirror &&
      typeof cm.CodeMirror.getValue === "function"
    ) {
      return cm.CodeMirror.getValue()
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n");
    }

    const textarea = cm.querySelector("textarea");
    if (textarea && textarea.value && textarea.value.trim()) {
      return textarea.value
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n");
    }

    const lines = Array.from(
      cm.querySelectorAll(".CodeMirror-code pre.CodeMirror-line, pre.CodeMirror-line")
    )
      .filter(line => !line.closest(".CodeMirror-measure"))
      .map(line => cleanCodeLine(line.textContent));

    return lines.join("\n").replace(/\n+$/g, "");
  }

  function extractCodeMirror6Code(cm) {
    if (!cm) return "";

    const lines = Array.from(cm.querySelectorAll(".cm-content .cm-line, .cm-line"))
      .map(line => cleanCodeLine(line.textContent));

    return lines.join("\n").replace(/\n+$/g, "");
  }

  function extractMonacoCode(editor) {
    if (!editor) return "";

    const textarea = editor.querySelector("textarea");
    if (textarea && textarea.value && textarea.value.trim()) {
      return textarea.value
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n");
    }

    const lines = Array.from(editor.querySelectorAll(".view-lines .view-line"))
      .map(line => cleanCodeLine(line.textContent));

    return lines.join("\n").replace(/\n+$/g, "");
  }

  function extractGenericCodeBlockCode(el) {
    if (!el) return "";

    const lines = Array.from(el.querySelectorAll("[class*='line'], [data-line]"))
      .map(line => cleanCodeLine(line.textContent))
      .filter(Boolean);

    if (lines.length >= 2) {
      return lines.join("\n").replace(/\n+$/g, "");
    }

    return cleanCodeLine(el.textContent || "");
  }

  function convertRichCodeEditors(clonedRoot, originalRoot) {
    const clonedCodeMirrors = Array.from(clonedRoot.querySelectorAll(".CodeMirror"));
    const originalCodeMirrors = originalRoot
      ? Array.from(originalRoot.querySelectorAll(".CodeMirror"))
      : [];

    clonedCodeMirrors.forEach((cm, index) => {
      const originalCm = originalCodeMirrors[index];

      const code =
        extractCodeMirrorCode(originalCm) ||
        extractCodeMirrorCode(cm);

      if (!code.trim()) return;

      const language = inferCodeLanguage(code, originalCm || cm);
      const pre = makePreFromCode(code, language);
      const wrapper = cm.closest(".react-codemirror2") || cm;

      wrapper.replaceWith(pre);
    });

    const clonedCodeMirror6Editors = Array.from(clonedRoot.querySelectorAll(".cm-editor"));
    const originalCodeMirror6Editors = originalRoot
      ? Array.from(originalRoot.querySelectorAll(".cm-editor"))
      : [];

    clonedCodeMirror6Editors.forEach((cm, index) => {
      const originalCm = originalCodeMirror6Editors[index];

      const code =
        extractCodeMirror6Code(originalCm) ||
        extractCodeMirror6Code(cm);

      if (!code.trim()) return;

      const language = inferCodeLanguage(code, originalCm || cm);
      const pre = makePreFromCode(code, language);

      cm.replaceWith(pre);
    });

    const clonedMonacoEditors = Array.from(clonedRoot.querySelectorAll(".monaco-editor"));
    const originalMonacoEditors = originalRoot
      ? Array.from(originalRoot.querySelectorAll(".monaco-editor"))
      : [];

    clonedMonacoEditors.forEach((editor, index) => {
      const originalEditor = originalMonacoEditors[index];

      const code =
        extractMonacoCode(originalEditor) ||
        extractMonacoCode(editor);

      if (!code.trim()) return;

      const language = inferCodeLanguage(code, originalEditor || editor);
      const pre = makePreFromCode(code, language);

      const wrapper =
        editor.closest(".monaco-editor-container") ||
        editor.closest("[class*='monaco']") ||
        editor;

      wrapper.replaceWith(pre);
    });

    const genericSelectors = [
      "[class*='syntaxHighlighter']",
      "[class*='codeBlock']",
      "[class*='CodeBlock']",
      "[class*='hljs']",
      "[class*='prism']"
    ];

    genericSelectors.forEach(selector => {
      clonedRoot.querySelectorAll(selector).forEach(el => {
        if (el.closest("pre")) return;
        if (el.querySelector(".CodeMirror, .cm-editor, .monaco-editor")) return;

        const code = extractGenericCodeBlockCode(el);

        if (!code.trim()) return;
        if (code.length < 20 && !code.includes("\n")) return;

        const language = inferCodeLanguage(code, el);
        const pre = makePreFromCode(code, language);

        el.replaceWith(pre);
      });
    });
  }

  function renderInline(node, options) {
    if (!node) return "";

    if (node.nodeType === Node.TEXT_NODE) {
      return escapeInlinePreserveSpaces(node.nodeValue);
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return "";

    const el = node;

    if (SKIP_TAGS.has(el.tagName) || isProbablyHidden(el)) return "";

    const tag = el.tagName.toLowerCase();

    if (tag === "br") return "\n";

    if (tag === "img") {
      if (!options.includeImages) return "";

      const src = getImageSource(el);

      if (!src) {
        return renderUnexportableImage(el, options, "missing source");
      }

      if (options.skipBlobImages && src.startsWith("blob:")) {
        return renderUnexportableImage(el, options, "blob URL");
      }

      if (options.skipDataImages && src.startsWith("data:")) {
        return renderUnexportableImage(el, options, "embedded data URL");
      }

      const alt = escapeInline(
        el.getAttribute("alt") ||
        el.getAttribute("aria-label") ||
        el.getAttribute("title") ||
        "image"
      );

      return `![${alt}](${absoluteUrl(src)})`;
    }

    if (tag === "a") {
      const text = inlineChildren(el, options);
      const href = el.getAttribute("href");

      if (
        !href ||
        href.startsWith("#") ||
        href.toLowerCase().startsWith("javascript:")
      ) {
        return text;
      }

      const url = absoluteUrl(href);

      if (!options.includeLinks) return text || url;
      if (!text || text === url) return url;

      return `[${text}](${url})`;
    }

    if (tag === "strong" || tag === "b") {
      const text = inlineChildren(el, options).trim();
      return text ? `**${text}**` : "";
    }

    if (tag === "em" || tag === "i") {
      const text = inlineChildren(el, options).trim();
      return text ? `*${text}*` : "";
    }

    if (tag === "s" || tag === "strike" || tag === "del") {
      const text = inlineChildren(el, options).trim();
      return text ? `~~${text}~~` : "";
    }

    if (tag === "code" && el.closest("pre") == null) {
      const raw = cleanText(el.textContent || "").trim();
      if (!raw) return "";

      if (raw.includes("`")) {
        return "`` " + raw.replace(/`/g, "\\`").trim() + " ``";
      }

      return "`" + raw + "`";
    }

    if (tag === "sup") {
      const text = inlineChildren(el, options).trim();
      return text ? `^${text}^` : "";
    }

    if (tag === "sub") {
      const text = inlineChildren(el, options).trim();
      return text ? `~${text}~` : "";
    }

    if (tag === "time") {
      return escapeInlinePreserveSpaces(
        el.getAttribute("datetime") || el.textContent || ""
      );
    }

    return Array.from(el.childNodes)
      .map(child => renderInline(child, options))
      .join("");
  }

  function renderBlockChildren(el, options, ctx = {}) {
    return Array.from(el.childNodes)
      .map(node => renderBlock(node, options, ctx))
      .join("");
  }

  function renderListItem(li, options, ctx) {
    const clone = li.cloneNode(true);

    clone.querySelectorAll(":scope > ul, :scope > ol").forEach(n => n.remove());

    let main = renderBlockChildren(clone, options, ctx)
      .trim()
      .replace(/\n{3,}/g, "\n\n");

    if (!main) {
      main = inlineChildren(clone, options);
    }

    main = main
      .replace(/^\s*:\s+/gm, ": ")
      .replace(/\n:\s+/g, ": ");

    const termMatch = main.match(/^([^\n:]{1,140})\n:\s*(.+)$/s);

    if (termMatch) {
      main = `**${termMatch[1].trim()}:** ${termMatch[2].trim()}`;
    }

    return main;
  }

  function renderList(el, options, ctx = {}) {
    const ordered = el.tagName.toLowerCase() === "ol";
    const start = Number.parseInt(el.getAttribute("start") || "1", 10);
    const indentLevel = ctx.indent || 0;
    const indent = "  ".repeat(indentLevel);

    const items = Array.from(el.children)
      .filter(child => child.tagName && child.tagName.toLowerCase() === "li");

    let out = "\n";

    items.forEach((li, index) => {
      const marker = ordered ? `${start + index}. ` : "- ";
      const main = renderListItem(li, options, ctx);

      const lines = main
        .split("\n")
        .map(x => x.trim())
        .filter(Boolean);

      if (!lines.length) {
        out += `${indent}${marker}\n`;
      } else {
        out += `${indent}${marker}${lines[0]}\n`;

        lines.slice(1).forEach(line => {
          out += `${indent}${" ".repeat(marker.length)}${line}\n`;
        });
      }

      Array.from(li.children)
        .filter(child => ["ul", "ol"].includes(child.tagName.toLowerCase()))
        .forEach(nested => {
          out += renderList(nested, options, {
            ...ctx,
            indent: indentLevel + 1
          });
        });
    });

    return out + "\n";
  }

  function renderTable(el, options) {
    const rows = Array.from(el.querySelectorAll("tr"))
      .map(tr =>
        Array.from(tr.children)
          .filter(cell => ["td", "th"].includes(cell.tagName.toLowerCase()))
          .map(cell => {
            const text =
              renderBlockChildren(cell, options).trim() ||
              inlineChildren(cell, options);

            return escapeTableCell(text);
          })
      )
      .filter(row => row.length > 0);

    if (!rows.length) return "";

    const maxCols = Math.max(...rows.map(row => row.length));

    const normalizedRows = rows.map(row => {
      const copy = [...row];
      while (copy.length < maxCols) copy.push("");
      return copy;
    });

    const firstRowHasHeaders =
      Array.from(el.querySelectorAll("tr:first-child th")).length > 0;

    const header = firstRowHasHeaders
      ? normalizedRows[0]
      : Array.from({ length: maxCols }, (_, i) => `Column ${i + 1}`);

    const body = firstRowHasHeaders
      ? normalizedRows.slice(1)
      : normalizedRows;

    const separator = header.map(() => "---");

    const lines = [
      `| ${header.join(" | ")} |`,
      `| ${separator.join(" | ")} |`,
      ...body.map(row => `| ${row.join(" | ")} |`)
    ];

    return `\n\n${lines.join("\n")}\n\n`;
  }

  function renderMedia(el) {
    const tag = el.tagName.toLowerCase();

    if (tag === "iframe") {
      const src = absoluteUrl(el.getAttribute("src") || "");
      const title = cleanText(el.getAttribute("title") || "Embedded content");

      return src ? `\n\n[${escapeInline(title)}](${src})\n\n` : "";
    }

    if (["video", "audio", "source"].includes(tag)) {
      const src = absoluteUrl(el.getAttribute("src") || "");
      return src ? `\n\n${src}\n\n` : "";
    }

    return "";
  }

  function renderDetails(el, options, ctx) {
    const summaryEl = el.querySelector(":scope > summary");
    const summary = cleanText(
      summaryEl?.innerText ||
      summaryEl?.textContent ||
      "Details"
    );

    const clone = el.cloneNode(true);
    clone.querySelectorAll(":scope > summary").forEach(n => n.remove());

    const body = renderBlockChildren(clone, options, ctx).trim();

    return `\n\n### ${escapeInline(summary)}\n\n${body}\n\n`;
  }

  function renderBlock(node, options, ctx = {}) {
    if (!node) return "";

    if (node.nodeType === Node.TEXT_NODE) {
      const text = cleanText(node.nodeValue);
      return text ? `${escapeMarkdownText(text)} ` : "";
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return "";

    const el = node;

    if (SKIP_TAGS.has(el.tagName) || isProbablyHidden(el)) return "";

    const tag = el.tagName.toLowerCase();

    if (/^h[1-6]$/.test(tag)) {
      const level = Number(tag[1]);
      const text = inlineChildren(el, options);

      return text ? `\n\n${"#".repeat(level)} ${text}\n\n` : "";
    }

    if (tag === "p") {
      const text = inlineChildren(el, options);
      return text ? `\n\n${text}\n\n` : "";
    }

    if (tag === "br") return "\n";

    if (tag === "hr") return "\n\n---\n\n";

    if (tag === "pre") {
      const code = el.textContent || "";
      const codeEl = el.querySelector("code");

      const langClass = codeEl
        ? Array.from(codeEl.classList).find(c => /^language-/.test(c))
        : null;

      const language = langClass
        ? langClass.replace(/^language-/, "")
        : inferCodeLanguage(code, el);

      return codeFence(code, language);
    }

    if (tag === "blockquote") {
      const inner = renderBlockChildren(el, options, ctx)
        .trim()
        .split("\n")
        .map(line => line.trim() ? `> ${line}` : ">")
        .join("\n");

      return inner ? `\n\n${inner}\n\n` : "";
    }

    if (tag === "ul" || tag === "ol") {
      return renderList(el, options, ctx);
    }

    if (tag === "table") {
      return renderTable(el, options);
    }

    if (tag === "img") {
      const image = renderInline(el, options);
      return image ? `\n\n${image}\n\n` : "";
    }

    if (tag === "figure") {
      const body = renderBlockChildren(el, options, ctx).trim();
      const caption = cleanText(el.querySelector("figcaption")?.innerText || "");

      if (caption && !body.includes(caption)) {
        return `\n\n${body}\n\n_${escapeInline(caption)}_\n\n`;
      }

      return body ? `\n\n${body}\n\n` : "";
    }

    if (tag === "details") {
      return renderDetails(el, options, ctx);
    }

    if (["iframe", "video", "audio", "source"].includes(tag)) {
      return renderMedia(el);
    }

    if (tag === "dl") {
      const parts = [];
      let currentTerm = "";

      Array.from(el.children).forEach(child => {
        const childTag = child.tagName.toLowerCase();

        if (childTag === "dt") {
          currentTerm = inlineChildren(child, options);
        } else if (childTag === "dd") {
          const desc = inlineChildren(child, options);

          if (currentTerm || desc) {
            parts.push(`- **${currentTerm}:** ${desc}`);
          }

          currentTerm = "";
        }
      });

      return parts.length ? `\n\n${parts.join("\n")}\n\n` : "";
    }

    if (tag === "li") {
      return inlineChildren(el, options);
    }

    if (hasBlockChild(el)) {
      return renderBlockChildren(el, options, ctx);
    }

    const inline = inlineChildren(el, options);

    return inline ? `\n\n${inline}\n\n` : "";
  }

  function removeDuplicateTitleHeadings(md, title) {
    const cleanTitle = cleanText(title).toLowerCase();

    if (!cleanTitle) return md;

    return String(md || "")
      .split("\n")
      .filter(line => {
        const m = line.match(/^#\s+(.+)\s*$/);
        if (!m) return true;

        return cleanText(m[1]).toLowerCase() !== cleanTitle;
      })
      .join("\n");
  }

  function protectFencedCodeBlocks(markdown, transformOutsideCode) {
    const blocks = [];
    const tokenPrefix = "§§MD_CODE_BLOCK_";

    let protectedMd = String(markdown || "").replace(
      /(^|\n)(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n\2(?=\n|$)/g,
      match => {
        const token = `${tokenPrefix}${blocks.length}§§`;
        blocks.push(match);
        return `\n${token}\n`;
      }
    );

    protectedMd = transformOutsideCode(protectedMd);

    blocks.forEach((block, index) => {
      protectedMd = protectedMd.replace(`${tokenPrefix}${index}§§`, block.trim());
    });

    return protectedMd;
  }

  function fixBrokenBoldOutsideCode(md) {
    let out = md;

    out = out.replace(/\*\s+\*\s+([^*\n]+?)\s+\*\s+\*/g, "**$1**");
    out = out.replace(/\*\s+\*\s+([^*\n]+?)\s+\*\*/g, "**$1**");
    out = out.replace(/\*\*([^*\n]+?)\s+\*\s+\*/g, "**$1**");

    out = out.replace(/\*\*([^*\n]+?)\*\*/g, (_, text) => {
      return `**${text.trim()}**`;
    });

    return out;
  }

  function fixInlineCodeOutsideCode(md) {
    return md.replace(/`([^`\n]*)`/g, (_, code) => {
      return "`" + code.trim() + "`";
    });
  }

  function fixStepLinksOutsideCode(md) {
    let out = md;

    out = out.replace(
      /^\s*\*\*(\d+)\*\*\s*\[([^\]]+)\]\(([^)]+)\)\s*$/gm,
      "$1. [$2]($3)"
    );

    out = out.replace(
      /^\s*\*\s+\*\s+(\d+)\*\*?\s*\[([^\]]+)\]\(([^)]+)\)\s*$/gm,
      "$1. [$2]($3)"
    );

    out = out.replace(
      /(\]\([^\n)]+\))(?=\d+\.\s+\[)/g,
      "$1\n"
    );

    return out;
  }

  function fixListDefinitionOutsideCode(md) {
    let out = md;

    out = out.replace(
      /^- ([^\n:][^\n]{1,120})\n:\s+/gm,
      "- **$1:** "
    );

    out = out.replace(
      /^- ([A-Z][^:\n]{2,100}):\s+/gm,
      "- **$1:** "
    );

    return out;
  }

  function fixParagraphLineBreaksOutsideCode(md) {
    return md
      .replace(/\bProvide\n(data_schema|task_type|index_column)\n/g, "Provide `$1` ")
      .replace(/\n(data_schema|task_type|index_column|classification|regression|string|numeric|date)\n/g, " `$1` ")
      .replace(/\n(sap-rpt-1-small|sap-rpt-1-large)\n/g, " `$1` ")
      .replace(/\n(SAP Cloud SDK for AI Python|SAP AI SDK for JavaScript\/TypeScript|SAP AI SDK for Java)\n/g, " $1 ");
  }

  function fixPunctuationOutsideCode(md) {
    return md
      .replace(/(`[^`\n]+`)\s+([.,;:!?])/g, "$1$2")
      .replace(/\*\*([^*\n]+)\*\*\s+([.,;:!?])/g, "**$1**$2")
      .replace(/\s+([.,;:!?])/g, "$1")
      .replace(/\b(string|numeric|date|classification|regression)\s*,/g, "$1,")
      .replace(/\b(default)\s*\./g, "$1.");
  }

  function fixHeadingImageSpacingOutsideCode(md) {
    return md.replace(/^(#{1,6} [^\n#]+)(!\[)/gm, "$1\n\n$2");
  }

  function postProcessMarkdown(md, options, title) {
    let out = String(md || "").replace(/\r\n/g, "\n");

    if (options.includePageTitle && title) {
      out = removeDuplicateTitleHeadings(out, title);
      out = `# ${escapeInline(title)}\n\n${out}`;
    }

    out = protectFencedCodeBlocks(out, outside => {
      return outside
        .replace(/\r\n/g, "\n")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n[ \t]+/g, "\n");
    });

    out = protectFencedCodeBlocks(out, outside => {
      outside = fixBrokenBoldOutsideCode(outside);
      outside = fixInlineCodeOutsideCode(outside);
      outside = fixStepLinksOutsideCode(outside);
      outside = fixListDefinitionOutsideCode(outside);
      outside = fixParagraphLineBreaksOutsideCode(outside);
      outside = fixPunctuationOutsideCode(outside);
      outside = fixHeadingImageSpacingOutsideCode(outside);

      outside = outside
        .replace(/\n{4,}/g, "\n\n\n")
        .replace(/([^\n])\n(Source: https?:\/\/)/g, "$1\n\n$2")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

      return outside;
    });

    return out.trim() + "\n";
  }

  function getSelectedHtmlAsRoot() {
    const selection = window.getSelection();

    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
      return null;
    }

    const wrapper = document.createElement("div");

    for (let i = 0; i < selection.rangeCount; i++) {
      wrapper.appendChild(selection.getRangeAt(i).cloneContents());
    }

    return wrapper;
  }

  function scoreRoot(el) {
    if (!el) return 0;

    const text = cleanText(el.innerText || el.textContent || "");
    const headings = el.querySelectorAll("h1,h2,h3,h4,h5,h6").length;
    const paragraphs = el.querySelectorAll("p,li,table,img,pre,.CodeMirror,.cm-editor,.monaco-editor").length;
    const chromePenalty = el.querySelectorAll("nav,header,footer,button,[role='navigation']").length;

    return text.length + headings * 200 + paragraphs * 80 - chromePenalty * 50;
  }

  function findBestRoot(options) {
    if (options.rootSelector) {
      const selected = document.querySelector(options.rootSelector);

      if (selected) return selected;

      console.warn(`No element found for rootSelector: ${options.rootSelector}`);
    }

    if (options.preferSelection) {
      const selectedRoot = getSelectedHtmlAsRoot();

      if (selectedRoot) return selectedRoot;
    }

    const selectors = [
      "[data-automation-id='Canvas']",
      "[data-automation-id='pageContent']",
      "#spPageCanvasContent",
      ".CanvasComponent",
      "article",
      "main[role='main']",
      "[role='main']",
      "main"
    ];

    const candidates = selectors
      .flatMap(selector => Array.from(document.querySelectorAll(selector)))
      .filter(Boolean);

    if (!candidates.length) return document.body;

    return candidates
      .map(el => ({ el, score: scoreRoot(el) }))
      .sort((a, b) => b.score - a.score)[0].el;
  }

  function showManualCopyDialog(text) {
    const existing = document.getElementById("sp-md-copy-dialog");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.id = "sp-md-copy-dialog";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "2147483647";
    overlay.style.background = "rgba(0,0,0,0.55)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.padding = "24px";

    const panel = document.createElement("div");
    panel.style.background = "white";
    panel.style.color = "black";
    panel.style.width = "min(1000px, 95vw)";
    panel.style.height = "min(720px, 90vh)";
    panel.style.borderRadius = "10px";
    panel.style.boxShadow = "0 20px 60px rgba(0,0,0,0.35)";
    panel.style.display = "flex";
    panel.style.flexDirection = "column";
    panel.style.overflow = "hidden";
    panel.style.fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";

    const header = document.createElement("div");
    header.style.padding = "14px 16px";
    header.style.borderBottom = "1px solid #ddd";
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.innerHTML = `
      <strong>Markdown ready</strong>
      <span style="font-size: 13px; color: #555;">Press Cmd+C / Ctrl+C to copy</span>
    `;

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.flex = "1";
    textarea.style.width = "100%";
    textarea.style.border = "0";
    textarea.style.padding = "16px";
    textarea.style.resize = "none";
    textarea.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
    textarea.style.fontSize = "13px";
    textarea.style.lineHeight = "1.45";
    textarea.style.outline = "none";

    const footer = document.createElement("div");
    footer.style.padding = "12px 16px";
    footer.style.borderTop = "1px solid #ddd";
    footer.style.display = "flex";
    footer.style.gap = "8px";
    footer.style.justifyContent = "flex-end";

    const selectButton = document.createElement("button");
    selectButton.textContent = "Select all";
    selectButton.style.padding = "8px 12px";
    selectButton.onclick = () => {
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);
    };

    const copyButton = document.createElement("button");
    copyButton.textContent = "Try copy again";
    copyButton.style.padding = "8px 12px";
    copyButton.onclick = async () => {
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);

      let ok = false;

      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(textarea.value);
          ok = true;
        }
      } catch {}

      if (!ok) {
        try {
          ok = document.execCommand("copy");
        } catch {}
      }

      copyButton.textContent = ok ? "Copied" : "Still blocked";
      setTimeout(() => {
        copyButton.textContent = "Try copy again";
      }, 1600);
    };

    const downloadButton = document.createElement("button");
    downloadButton.textContent = "Download .md";
    downloadButton.style.padding = "8px 12px";
    downloadButton.onclick = () => {
      const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "sharepoint-page.md";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    };

    const closeButton = document.createElement("button");
    closeButton.textContent = "Close";
    closeButton.style.padding = "8px 12px";
    closeButton.onclick = () => overlay.remove();

    footer.appendChild(selectButton);
    footer.appendChild(copyButton);
    footer.appendChild(downloadButton);
    footer.appendChild(closeButton);

    panel.appendChild(header);
    panel.appendChild(textarea);
    panel.appendChild(footer);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    setTimeout(() => {
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);
    }, 50);
  }

  async function copyToClipboard(text) {
    const value = String(text || "");

    try {
      if (typeof copy === "function") {
        copy(value);
        return true;
      }
    } catch {}

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
        return true;
      }
    } catch {}

    try {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "0";
      textarea.style.top = "0";
      textarea.style.width = "1px";
      textarea.style.height = "1px";
      textarea.style.opacity = "0";
      textarea.style.zIndex = "2147483647";

      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);

      const ok = document.execCommand("copy");
      document.body.removeChild(textarea);

      if (ok) return true;
    } catch {}

    showManualCopyDialog(value);
    return false;
  }

  window.copySharePointPageAsMarkdown = async function copySharePointPageAsMarkdown(userOptions = {}) {
    const options = { ...DEFAULT_OPTIONS, ...userOptions };

    const sourceRoot = findBestRoot(options);
    const clonedRoot = sourceRoot.cloneNode(true);

    convertRichCodeEditors(clonedRoot, sourceRoot);
    removeNoise(clonedRoot, options);

    const title = getPageTitle();

    let md = renderBlockChildren(clonedRoot, options);

    if (options.includeSourceUrl) {
      md += `\n\n---\n\nSource: ${window.location.href}\n`;
    }

    md = postProcessMarkdown(md, options, title);

    if (options.includeFrontMatter) {
      const frontMatter = [
        "---",
        `title: "${String(title || document.title || "").replace(/"/g, '\\"')}"`,
        `source: "${window.location.href}"`,
        `captured: "${new Date().toISOString()}"`,
        "---",
        ""
      ].join("\n");

      md = frontMatter + md;
    }

    window.__lastSharePointMarkdown = md;

    const copied = await copyToClipboard(md).catch(() => false);

    if (copied) {
      console.info("Markdown copied to clipboard.");
    } else {
      console.warn("Automatic clipboard copy failed. Use the copy dialog, download the .md file, or read window.__lastSharePointMarkdown.");
    }

    console.info(
      `Markdown generated: ${md.length.toLocaleString()} characters, ${md.split("\n").length.toLocaleString()} lines. Stored in window.__lastSharePointMarkdown.`
    );

    if (options.debug) {
      console.info("Root used:", sourceRoot);
      console.info("Options:", options);
    }

    // Intentionally no return, to avoid dumping the full Markdown in DevTools.
  };

  console.info("Ready. Run: await copySharePointPageAsMarkdown()");
})();
