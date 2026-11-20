// Presentation only: names come from the current response, never a customer map.
const text = (value) => typeof value === "string" ? value.trim() : "";

/** Build a code → name index; conflicting supplied names leave the code unresolved. */
export function buildPlantNameIndex(plants = []) {
  const names = new Map();
  const conflicts = new Set();
  for (const plant of plants || []) {
    const code = text(plant?.plant_code), name = text(plant?.plant_name);
    if (!code || !name || code === name || conflicts.has(code)) continue;
    if (names.has(code) && names.get(code) !== name) {
      names.delete(code);
      conflicts.add(code);
    } else names.set(code, name);
  }
  return names;
}

/** Description first, with a code retained for traceability in explicit plant fields. */
export function plantLabel(code, names, fallback = "Plant not supplied") {
  const key = text(code);
  const name = names?.get(key);
  return name ? `${name} (${key})` : key || fallback;
}

/** The list contract supplies `CODE Name` labels, not structured pairs. */
export function suppliedPlantLabel(label) {
  const value = text(label);
  const match = /^([A-Z0-9]+)\s+(.+)$/.exec(value);
  return match ? `${match[2]} (${match[1]})` : value;
}

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Replace whole-token codes in rendered prose only; never change source or attributes. */
export function describeRenderedPlants(container, names, protectedTexts = []) {
  if (!container || !names?.size) return;
  const keys = [...names.keys()].sort((a, b) => b.length - a.length).map(escapeRegExp);
  // Underscores/hyphens protect technical identifiers; URI/path and math text stay literal.
  const pattern = new RegExp(`(?<![\\p{L}\\p{N}_/\\-])(${keys.join("|")})(?![\\p{L}\\p{N}_/\\-])`, "gu");
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue?.trim() || node.parentElement?.closest("a, button, code, pre, kbd, samp, math, .id-ref, .plant-name-ref")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const targets = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) targets.push(node);
  for (const node of targets) {
    const value = node.nodeValue;
    const protectedRanges = [...value.matchAll(/(?:[a-z][a-z0-9+.-]*:\/\/|www\.|mailto:|urn:|tel:)\S+|\$[^$]+\$|[\p{L}\p{N}_.]+(?:\s*[=+*^<>]\s*[\p{L}\p{N}_.]+)+/giu)]
      .map(match => [match.index, match.index + match[0].length]);
    for (const literal of protectedTexts.filter(Boolean)) {
      let start = value.indexOf(literal);
      while (start !== -1) {
        protectedRanges.push([start, start + literal.length]);
        start = value.indexOf(literal, start + literal.length);
      }
    }
    pattern.lastIndex = 0;
    let match, last = 0, fragment = null;
    while ((match = pattern.exec(value))) {
      if (protectedRanges.some(([start, end]) => match.index >= start && match.index < end)) continue;
      fragment ||= document.createDocumentFragment();
      fragment.append(value.slice(last, match.index));
      const label = document.createElement("span");
      label.className = "plant-name-ref";
      label.textContent = names.get(match[1]);
      label.title = `Plant ${match[1]}`;
      fragment.append(label);
      last = match.index + match[1].length;
    }
    if (fragment) {
      fragment.append(value.slice(last));
      node.replaceWith(fragment);
    }
  }
}
