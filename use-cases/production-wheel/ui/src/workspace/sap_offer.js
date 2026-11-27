/**
 * After-run notice: offer SAP settings generation when a run has just completed.
 *
 * Nothing is selected or generated on its own: the notice names the scenario
 * (stored selection, else Balanced), lists what would be blocked and why, and
 * the user decides with one click ("Select B and generate"), asks the
 * Assistant, or dismisses it. See offerModel in sap_readiness.js for the text.
 */
import { el } from "./dom.js";
import { offerModel } from "./sap_readiness.js";

/**
 * Mount the notice controller on an empty, hidden container.
 * @param {HTMLElement} box Container (starts hidden).
 * @param {object} app Workspace app (askAssistant, fail).
 * @param {{select: (pointIndex: number) => Promise<void>, generate: () => Promise<object>}} actions
 *   `select` stores the decision; `generate` runs SAP settings generation for the stored selection.
 * @returns {{show: (decisions: object, readiness: object, summary: object|null) => boolean, hide: () => void}}
 */
export function mountSapOffer(box, app, actions) {
  /** Hide and empty the notice. */
  function hide() {
    box.hidden = true;
    box.replaceChildren();
  }

  /** Button helper with a click handler that reports errors and re-enables on failure. */
  function button(label, design, onClick) {
    const control = document.createElement("ui5-button");
    control.textContent = label;
    if (design) control.setAttribute("design", design);
    control.addEventListener("click", async () => {
      control.disabled = true;
      try {
        await onClick();
      } catch (error) {
        control.disabled = false;
        app.fail(error);
      }
    });
    return control;
  }

  /**
   * Render the offer, or hide it when there is nothing to offer.
   * @returns {boolean} Whether the notice is shown.
   */
  function show(decisions, readiness, summary) {
    hide();
    const offer = offerModel(decisions, readiness, summary);
    if (!offer) return false;
    box.append(el("p", offer.text, "sap-offer-title"));
    if (offer.gaps.length) {
      const list = el("ul");
      for (const gap of offer.gaps) {
        const hint = gap.tab === "planning_calendar" && offer.suggestion
          ? ` Codes found in an earlier wheel ("${offer.suggestion.dataset_name}"): ${offer.suggestion.values.calendar_odd_profile} / ${offer.suggestion.values.calendar_even_profile}; save them in Settings or ask the Assistant.`
          : gap.fix ? ` ${gap.fix}` : "";
        list.append(el("li", `${gap.label}: ${gap.status}. ${gap.text}${hint}`));
      }
      box.append(el("p", "Generating now is possible; these tabs would be reported as blocked or partial:", "muted"), list);
    }
    const controls = el("div", null, "actions");
    controls.append(
      button(offer.primaryLabel, "Emphasized", async () => {
        if (offer.needsSelection) await actions.select(offer.pointIndex);
        await actions.generate();
        hide();
      }),
      button("Ask the Assistant", null, async () => app.askAssistant(offer.question)),
      button("Dismiss", "Transparent", async () => hide()),
    );
    box.append(controls);
    box.hidden = false;
    return true;
  }

  return { show, hide };
}
