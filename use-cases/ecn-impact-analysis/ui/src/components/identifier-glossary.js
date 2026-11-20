// Make the dense SAP identifiers inside AI-written prose understandable in place.
//
// The impact-factor assessments are LLM prose that names raw part numbers,
// assembly numbers and model ranges (e.g. "916100410830", "G916100410040",
// "Q900V"). buildIdentifierIndex(snapshot) collects every identifier the
// snapshot already knows about, each with a human label or description.
// linkifyIdentifiers(container, index, inspect) walks the rendered text and
// turns each recognised identifier into either a clickable part link (opens the
// material dialog on click, description on hover) or a tooltip-annotated span
// (assembly / model range). Only identifiers present in the index are touched,
// so dates like "2022-05-30" and quantities like "quantity 1" are never matched.

/** Collect {identifier → {kind, description/label, clickable}} from one snapshot. */
export function buildIdentifierIndex(snapshot) {
  const index = new Map();
  if (!snapshot) return index;

  // A part always wins over an annotation and keeps the first description found.
  const addPart = (number, description) => {
    if (!number) return;
    const key = String(number);
    const prev = index.get(key);
    if (prev?.kind === "part") { if (!prev.description && description) prev.description = description; return; }
    index.set(key, { kind: "part", description: description || "", clickable: true });
  };
  const annotate = (value, kind) => {
    if (!value) return;
    const key = String(value);
    if (index.has(key)) return; // never downgrade a part to an annotation
    index.set(key, { kind, clickable: false });
  };

  const bom = snapshot.bom_changes || {};
  for (const part of snapshot.part_descriptions || []) addPart(part.number, part.description);
  for (const part of snapshot.parts || []) addPart(part.number, part.description);
  for (const material of bom.materials || []) addPart(material.number, material.description);
  for (const line of bom.lines || []) {
    for (const side of [line.incoming, line.outgoing]) if (side) addPart(side.number, side.description);
    annotate(line.assembly, "assembly");
    annotate(line.model_range, "model");
  }
  for (const plant of snapshot.plant_impact?.plants || []) {
    for (const material of plant.materials || []) addPart(material.number, material.description);
  }
  for (const program of snapshot.programs || []) {
    annotate(program.model_range, "model");
    for (const level of ["ebom", "mbom"]) {
      for (const row of program[level] || []) {
        addPart(row.number, row.description);
        if (row.parent) annotate(row.parent.split("_")[0], "assembly");
      }
    }
  }
  return index;
}

/** Replace only known part identifiers with their supplied descriptions. */
export function describePartIdentifiers(text, index) {
  if (!text || !index?.size) return text;
  const parts = [...index.entries()]
    .filter(([, entry]) => entry.kind === "part" && entry.description)
    .sort(([left], [right]) => right.length - left.length)
    .map(([identifier]) => escapeRegExp(identifier));
  if (!parts.length) return text;
  const pattern = new RegExp(`(?<![A-Za-z0-9])(${parts.join("|")})(?![A-Za-z0-9])`, "g");
  return String(text).replace(pattern, (identifier) => index.get(identifier).description);
}

/** Human hover text for one identifier entry. */
function tooltipFor(entry) {
  if (entry.kind === "part") return entry.description ? `${entry.description} (part — click for details)` : "Part — click for details";
  if (entry.kind === "assembly") return "Assembly (parent BOM item)";
  if (entry.kind === "model") return "Model range (vehicle program)";
  return "";
}

/** A clickable part button or an annotated span for one matched identifier. */
function makeRef(id, entry, inspect) {
  const el = document.createElement(entry.clickable ? "button" : "span");
  el.className = `id-ref id-ref--${entry.kind}`;
  el.textContent = id;
  el.title = tooltipFor(entry);
  if (entry.clickable) {
    el.type = "button";
    el.setAttribute("aria-label", `${id}: ${entry.description || "part"} — open details`);
    el.addEventListener("click", () => void inspect?.(id));
  }
  return el;
}

/** Escape a string for safe inclusion in a RegExp alternation. */
const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Replace every recognised identifier in one text node with a ref element. */
function replaceInTextNode(textNode, pattern, index, inspect) {
  const text = textNode.nodeValue;
  pattern.lastIndex = 0;
  let match, last = 0, frag = null;
  while ((match = pattern.exec(text))) {
    const entry = index.get(match[1]);
    if (!entry) continue;
    frag = frag || document.createDocumentFragment();
    if (match.index > last) frag.append(text.slice(last, match.index));
    frag.append(makeRef(match[1], entry, inspect));
    last = match.index + match[1].length;
  }
  if (!frag) return;
  if (last < text.length) frag.append(text.slice(last));
  textNode.replaceWith(frag);
}

/** Turn recognised identifiers inside `container` into clickable/annotated refs. */
export function linkifyIdentifiers(container, index, inspect) {
  if (!container || !index?.size) return;
  // Longest identifiers first so a prefix never shadows a longer match.
  const keys = [...index.keys()].sort((a, b) => b.length - a.length).map(escapeRegExp);
  const pattern = new RegExp(`(?<![A-Za-z0-9])(${keys.join("|")})(?![A-Za-z0-9])`, "g");
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      if (!node.nodeValue?.trim()) return NodeFilter.FILTER_REJECT;
      // Leave text that is already inside a link, button or earlier ref alone.
      if (node.parentElement?.closest("a, button, code, pre, kbd, samp, math, .id-ref, .plant-name-ref")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const targets = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) targets.push(node);
  for (const node of targets) replaceInTextNode(node, pattern, index, inspect);
}
