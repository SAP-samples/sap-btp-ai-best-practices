/** Local fictional relationship preview. Imported only by the DEV opt-in branch. No IO. */
export function mountLinkedEcnDemo(root, signal) {
  const doc = root.ownerDocument;
  const make = (tag, text) => {
    const node = doc.createElement(tag);
    if (text) node.textContent = text;
    return node;
  };
  const dialog = make("dialog");
  dialog.id = "linked-ecn-demo-dialog";
  dialog.className = "ecn-local-dialog";
  dialog.setAttribute("aria-labelledby", "linked-ecn-demo-title");
  dialog.setAttribute("aria-describedby", "linked-ecn-demo-notice");
  const title = make("h2", "ECN-DEMO-001 — Fictional related ECN");
  title.id = "linked-ecn-demo-title";
  const notice = make("p", "Simulated demo relationship. No source relationship is supplied for the current ECN. This fictional record is local only, not a server ECN or an engineering decision.");
  notice.id = "linked-ecn-demo-notice";
  const description = make("p", "Illustrative follow-up: revise the protective sleeve on a demonstration hydraulic line to improve clearance during assembly. Reviewers would compare the sleeve revision and installation sequence before coordinating the two changes.");
  const closeButton = make("button", "Close and return to current ECN");
  closeButton.type = "button";
  closeButton.autofocus = true;
  const actions = make("div");
  actions.className = "dialog-actions";
  actions.append(closeButton);
  dialog.append(title, notice, description, actions);
  root.append(dialog);
  let disposed = false;
  let returnFocus = false;
  const restoreFocus = () => {
    if (!disposed && returnFocus) root.querySelector("[data-linked-ecn-demo]")?.focus();
    returnFocus = false;
  };
  const close = () => {
    if (typeof dialog.close === "function") dialog.close(); else dialog.open = false;
    restoreFocus();
  };
  const open = (event) => {
    const link = event.target.closest?.("[data-linked-ecn-demo]");
    if (!link || !root.contains(link) || disposed || signal.aborted) return;
    event.preventDefault();
    if (dialog.open) return;
    returnFocus = true;
    if (typeof dialog.showModal === "function") dialog.showModal(); else dialog.open = true;
    closeButton.focus();
  };
  const cancel = (event) => {
    event.preventDefault();
    close();
  };
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    root.removeEventListener("click", open);
    closeButton.removeEventListener("click", close);
    dialog.removeEventListener("cancel", cancel);
    dialog.removeEventListener("close", restoreFocus);
    signal.removeEventListener("abort", dispose);
    close();
    dialog.remove();
  };
  root.addEventListener("click", open);
  closeButton.addEventListener("click", close);
  dialog.addEventListener("cancel", cancel);
  dialog.addEventListener("close", restoreFocus);
  signal.addEventListener("abort", dispose, { once: true });
  if (signal.aborted) dispose();
  return {
    render(field) {
      if (disposed) return;
      const focused = doc.activeElement?.matches?.("[data-linked-ecn-demo]");
      const link = make("a", "ECN-DEMO-001");
      link.href = "#linked-ecn-demo-dialog";
      link.dataset.linkedEcnDemo = "";
      link.setAttribute("aria-haspopup", "dialog");
      link.setAttribute("aria-controls", dialog.id);
      const label = make("span", "Simulated demo relationship");
      label.className = "helper-text";
      label.id = "linked-ecn-demo-label";
      link.setAttribute("aria-describedby", label.id);
      field.querySelector("dd").replaceChildren(link, doc.createTextNode(" — "), label);
      if (focused) queueMicrotask(() => {
        if (!disposed && link.isConnected && !dialog.open) link.focus();
      });
    },
    dispose,
  };
}
