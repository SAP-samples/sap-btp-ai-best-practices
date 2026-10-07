/** Safe, report-friendly Markdown formatting for model-authored chat messages. */
import {marked} from 'marked';

/**
 * Escape text before inserting it into generated HTML or an HTML attribute.
 * @param {unknown} value Untrusted text to escape.
 * @returns {string} HTML-safe text.
 */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'",'&#39;');
}

/**
 * Accept only absolute HTTP(S) destinations for links authored by the model.
 * @param {unknown} value Candidate link destination.
 * @returns {string|null} Normalized safe URL, or null when the link is blocked.
 */
function safeHttpUrl(value) {
  const href=String(value ?? '').trim();
  if(!/^https?:\/\//i.test(href))return null;
  try {
    const parsed=new URL(href);
    return ['http:','https:'].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
}

const renderer=new marked.Renderer();
renderer.html=html=>escapeHtml(html);
renderer.image=(_href,_title,text)=>escapeHtml(text);
renderer.link=(href,title,text)=>{
  const safe=safeHttpUrl(href);
  if(!safe)return text;
  const titleAttribute=title ? ` title="${escapeHtml(title)}"` : '';
  return `<a href="${escapeHtml(safe)}" target="_blank" rel="noopener noreferrer"${titleAttribute}>${text}</a>`;
};

/**
 * Convert assistant Markdown into allowlisted HTML with raw HTML and images disabled.
 * @param {unknown} markdown Model-authored Markdown text.
 * @returns {string} Safe HTML for an assistant message body.
 */
export function formatAssistantMarkdown(markdown) {
  return marked.parse(String(markdown ?? ''),{renderer,gfm:true,breaks:true});
}

/**
 * Choose HTML only for successful assistant responses; all other messages stay literal.
 * @param {unknown} content Message content.
 * @param {string} role Message author role.
 * @param {{isError?: boolean}} options Rendering flags.
 * @returns {{mode: 'html'|'text', value: string}} Safe rendering instruction.
 */
export function messageContent(content,role,{isError=false}={}) {
  const value=String(content ?? '');
  return role==='assistant'&&!isError ? {mode:'html',value:formatAssistantMarkdown(value)} : {mode:'text',value};
}
