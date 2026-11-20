/**
 * Settings "Plant Code Mapping" subsection.
 *
 * Conditional rules name brands (GEN_E_BRAND = E_BRAND_JB), not plants. This
 * editor maps each brand suffix (JB) to the SAP plant that builds it (PL10
 * Plant A) and chooses which rule lifecycle states may decide a VC group's
 * plants. Values are saved in HANA through PUT /api/plant-settings and frozen
 * into every ECN imported afterwards.
 */

const FIELDS = Object.freeze([
  ["suffix", "Brand suffix", "JB"],
  ["plant_code", "Plant code", "PL10"],
  ["plant_name", "Plant name", "Plant A"],
]);

/** Build one editable mapping row with its own remove control.
 * @param {{suffix?: string, plant_code?: string, plant_name?: string}} value - Saved row, or empty for a new one.
 * @returns {HTMLTableRowElement} Row whose inputs are named after the API fields.
 */
function mappingRow(value = {}) {
  const row = document.createElement("tr");
  for (const [name, label, placeholder] of FIELDS) {
    const cell = document.createElement("td");
    const input = document.createElement("input");
    input.name = name;
    input.value = value[name] || "";
    input.placeholder = placeholder;
    input.setAttribute("aria-label", label);
    cell.append(input);
    row.append(cell);
  }
  const cell = document.createElement("td");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "plant-mapping-remove";
  remove.textContent = "×";
  remove.setAttribute("aria-label", `Remove mapping ${value.suffix || "row"}`);
  remove.addEventListener("click", () => row.remove());
  cell.append(remove);
  row.append(cell);
  return row;
}

/** Build one rule-state checkbox; checked means rules in this state decide plants.
 * @param {string} state - Rule lifecycle state (Etat) as exported by the rules workbook.
 * @param {boolean} used - Whether the state is currently not excluded.
 * @returns {HTMLLabelElement} Labelled checkbox carrying the state as its value.
 */
function stateCheckbox(state, used) {
  const label = document.createElement("label");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.name = "rule_state";
  box.value = state;
  box.checked = used;
  label.append(box, ` ${state}`);
  return label;
}

/** Render saved settings into the table and the rule-state checkboxes. */
function render(root, settings) {
  root.querySelector("#plant-mapping-rows").replaceChildren(...settings.mappings.map((mapping) => mappingRow(mapping)));
  const excluded = new Set(settings.excluded_rule_states);
  const states = root.querySelector("#rule-states-options");
  states.replaceChildren(...settings.available_rule_states.map((state) => stateCheckbox(state, !excluded.has(state))));
  if (!settings.available_rule_states.length) states.textContent = "Publish a conditional-rules workbook to choose rule states.";
}

/** Read the form into the PUT body; fully blank rows are ignored. */
function collect(root) {
  const mappings = [...root.querySelectorAll("#plant-mapping-rows tr")]
    .map((row) => Object.fromEntries(FIELDS.map(([name]) => [name, row.querySelector(`[name="${name}"]`).value.trim()])))
    .filter((mapping) => mapping.suffix || mapping.plant_code || mapping.plant_name);
  const excluded = [...root.querySelectorAll('[name="rule_state"]')].filter((box) => !box.checked).map((box) => box.value);
  return { mappings, excluded_rule_states: excluded };
}

/** Load the Plant Code Mapping and wire its add, remove and save controls.
 * @param {HTMLElement} root - Settings page container holding the #plant-mapping section.
 * @param {{api: {getPlantSettings: Function, savePlantSettings: Function}, signal?: AbortSignal}} options
 * @returns {Promise<void>} Resolves once the saved values are shown.
 */
export async function initPlantMapping(root, { api, signal }) {
  const status = root.querySelector("#plant-mapping-status");
  root.querySelector("#add-plant-mapping").addEventListener("click", () => {
    root.querySelector("#plant-mapping-rows").append(mappingRow());
  });
  root.querySelector("#plant-mapping-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = event.currentTarget.querySelector('[type="submit"]');
    submit.disabled = true;
    status.textContent = "Saving…";
    try {
      render(root, await api.savePlantSettings(collect(root), { signal }));
      status.textContent = "Plant Code Mapping saved. It applies to ECNs imported from now on.";
    } catch (error) {
      if (error.name !== "AbortError") status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });
  render(root, await api.getPlantSettings({ signal }));
}
