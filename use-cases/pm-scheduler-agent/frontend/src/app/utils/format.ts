/**
 * Format an ISO date/datetime string for display in tables.
 *
 * The backend returns SAP dates as full ISO timestamps (e.g.
 * "2026-08-25T00:00:00"). For the grids we only care about the calendar day,
 * so strip the time portion. Empty/null values render as an em dash.
 */
export function formatDisoDate(value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return '—';
  }
  const s = String(value);
  const tIndex = s.indexOf('T');
  return tIndex > 0 ? s.slice(0, tIndex) : s;
}

/**
 * Convert a subset of Markdown (headings, bold, bullets, paragraphs) to HTML.
 * Handles the patterns produced by the AI explanation endpoint.
 */
export function markdownToHtml(text: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const inlineFormat = (s: string) =>
    escape(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>');

  const lines = text.split('\n');
  const out: string[] = [];
  let inList = false;

  for (const raw of lines) {
    const line = raw.trimEnd();

    // Headings
    const h3 = line.match(/^###\s+(.*)/);
    const h2 = line.match(/^##\s+(.*)/);
    const h1 = line.match(/^#\s+(.*)/);
    if (h1 || h2 || h3) {
      if (inList) { out.push('</ul>'); inList = false; }
      const content = inlineFormat((h1 ?? h2 ?? h3)![1]);
      out.push(`<p style="font-weight:700;font-size:15px;margin:1.1em 0 0.3em;color:#0a6ed1;">${content}</p>`);
      continue;
    }

    // Bullet items
    const bullet = line.match(/^[-*]\s+(.*)/);
    if (bullet) {
      if (!inList) { out.push('<ul style="margin:0.3em 0 0.3em 1.2em;padding:0;">'); inList = true; }
      out.push(`<li style="margin:0.25em 0;line-height:1.55;">${inlineFormat(bullet[1])}</li>`);
      continue;
    }

    // Blank line — close list, add spacing
    if (line === '') {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<div style="height:0.5em;"></div>');
      continue;
    }

    // Normal paragraph line
    if (inList) { out.push('</ul>'); inList = false; }
    out.push(`<p style="margin:0.2em 0;line-height:1.65;">${inlineFormat(line)}</p>`);
  }

  if (inList) out.push('</ul>');
  return out.join('\n');
}
