/**
 * Type declarations for `npm run typecheck` only; Vite never bundles this file.
 *
 * The UI5 Web Components the workspace creates with el("ui5-button") or reads
 * from index.html are registered here, so the checker knows their tag names
 * and the few properties the workspace sets on them.
 */

/** `<ui5-button>` properties used by the workspace. */
interface UI5ButtonElement extends HTMLElement {
  disabled: boolean;
  design: string;
  icon: string;
}

interface HTMLElementTagNameMap {
  "ui5-button": UI5ButtonElement;
}
