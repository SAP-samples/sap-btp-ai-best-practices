/* Scenario Maker UI5 Components */
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Slider.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/MultiComboBox.js";
import "@ui5/webcomponents/dist/MultiComboBoxItem.js";
import "@ui5/webcomponents/dist/DatePicker.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Toast.js";
import "@ui5/webcomponents/dist/Bar.js";

/* Icons */
import "@ui5/webcomponents-icons/dist/reset.js";
import "@ui5/webcomponents-icons/dist/save.js";
import "@ui5/webcomponents-icons/dist/simulate.js";
import "@ui5/webcomponents-icons/dist/syntax.js";
import "@ui5/webcomponents-icons/dist/download.js";
import "@ui5/webcomponents-icons/dist/upload.js";
import "@ui5/webcomponents-icons/dist/copy.js";
import "@ui5/webcomponents-icons/dist/calendar.js";

import { request } from "../../services/api.js";
import { buildScenarioAnalysisPrompt } from "../../services/scenarioPromptBuilder.js";

// Lever Configuration
const LEVER_CONFIG = {
  financing: {
    label: "Financing",
    containerId: "financing-levers",
    levers: [
      {
        key: "pct_primary_financing_roll_mean_4",
        displayName: "Primary Financing",
        min: 0,
        max: 100,
        step: 1,
        default: 35,
        unit: "%",
        bmOnly: false
      },
      {
        key: "pct_secondary_financing_roll_mean_4",
        displayName: "Secondary Financing",
        min: 0,
        max: 100,
        step: 1,
        default: 20,
        unit: "%",
        bmOnly: false
      },
      {
        key: "pct_tertiary_financing_roll_mean_4",
        displayName: "Tertiary Financing",
        min: 0,
        max: 100,
        step: 1,
        default: 5,
        unit: "%",
        bmOnly: false
      }
    ]
  },
  staffing: {
    label: "Staffing",
    containerId: "staffing-levers",
    bmOnly: true,
    levers: [
      {
        key: "staffing_unique_associates_roll_mean_4",
        displayName: "Unique Associates",
        min: 0,
        max: 50,
        step: 1,
        default: 12,
        unit: "count",
        bmOnly: true
      },
      {
        key: "staffing_hours_roll_mean_4",
        displayName: "Staffing Hours",
        min: 0,
        max: 2000,
        step: 50,
        default: 400,
        unit: "hours",
        bmOnly: true
      }
    ]
  },
  productMix: {
    label: "Product Mix",
    containerId: "product-mix-levers",
    levers: [
      {
        key: "pct_omni_channel_roll_mean_4",
        displayName: "Omni-Channel",
        min: 0,
        max: 100,
        step: 1,
        default: 40,
        unit: "%",
        bmOnly: false
      },
      {
        key: "pct_value_product_roll_mean_4",
        displayName: "Value Product",
        min: 0,
        max: 100,
        step: 1,
        default: 30,
        unit: "%",
        bmOnly: false
      },
      {
        key: "pct_premium_product_roll_mean_4",
        displayName: "Premium Product",
        min: 0,
        max: 100,
        step: 1,
        default: 25,
        unit: "%",
        bmOnly: false
      },
      {
        key: "pct_white_glove_roll_mean_4",
        displayName: "White Glove",
        min: 0,
        max: 100,
        step: 1,
        default: 15,
        unit: "%",
        bmOnly: false
      }
    ]
  },
  marketSignals: {
    label: "Market Signals",
    containerId: "market-signals-levers",
    levers: [
      {
        key: "brand_awareness_dma_roll_mean_4",
        displayName: "Brand Awareness",
        min: 0,
        max: 100,
        step: 1,
        default: 50,
        unit: "%",
        bmOnly: false
      },
      {
        key: "brand_consideration_dma_roll_mean_4",
        displayName: "Brand Consideration",
        min: 0,
        max: 100,
        step: 1,
        default: 40,
        unit: "%",
        bmOnly: false
      }
    ]
  }
};

// State object
const state = {
  scenarioName: "New Scenario",
  channel: "B&M",
  timeWindow: {
    originDate: null,
    horizonStart: 1,
    horizonEnd: 13
  },
  scope: {
    profitCenters: [],
    dmas: [],
    allProfitCenters: false,
    allDmas: false
  },
  levers: {},
  availableStores: [],
  availableDmas: [],
  validationErrors: []
};

// Initialize lever state from config
function initializeLeverState() {
  Object.values(LEVER_CONFIG).forEach(category => {
    category.levers.forEach(lever => {
      state.levers[lever.key] = {
        enabled: false,
        currentValue: lever.default,
        targetValue: lever.default,
        operation: "set",
        config: lever,
        useWeeklyValues: false,
        weeklyValues: {}
      };
    });
  });
}

// Create lever row HTML
function createLeverRow(lever) {
  return `
    <div class="lever-row" data-lever-key="${lever.key}">
      <ui5-checkbox
        id="lever-enabled-${lever.key}"
        class="lever-checkbox">
      </ui5-checkbox>

      <ui5-label class="lever-label">${lever.displayName}</ui5-label>

      <ui5-input
        id="lever-current-${lever.key}"
        type="Number"
        class="lever-input current-value"
        placeholder="Current"
        readonly
        value="${lever.default}">
      </ui5-input>

      <ui5-slider
        id="lever-slider-${lever.key}"
        class="lever-slider"
        min="${lever.min}"
        max="${lever.max}"
        step="${lever.step}"
        value="${lever.default}"
        show-tickmarks
        label-interval="0">
      </ui5-slider>

      <ui5-input
        id="lever-target-${lever.key}"
        type="Number"
        class="lever-input target-value"
        value="${lever.default}"
        placeholder="Target">
      </ui5-input>

      <ui5-text class="lever-unit">${lever.unit}</ui5-text>

      <ui5-select id="lever-operation-${lever.key}" class="lever-operation">
        <ui5-option value="set" selected>Set</ui5-option>
        <ui5-option value="increase">+%</ui5-option>
        <ui5-option value="decrease">-%</ui5-option>
      </ui5-select>

      <ui5-button
        id="lever-timeline-${lever.key}"
        class="lever-timeline-btn"
        icon="calendar"
        design="Transparent"
        tooltip="Configure weekly values"
        disabled>
      </ui5-button>
    </div>
  `;
}

// Render all lever rows
function renderLevers() {
  Object.entries(LEVER_CONFIG).forEach(([categoryKey, category]) => {
    const container = document.getElementById(category.containerId);
    if (!container) return;

    container.innerHTML = category.levers
      .map(lever => createLeverRow(lever))
      .join("");
  });

  updateLeversVisibility();
}

// Setup click handlers for timeline buttons - REMOVED (Replaced by event delegation)
// function setupTimelineButtonHandlers() { ... }

// Update visibility of channel-specific levers
function updateLeversVisibility() {
  const staffingPanel = document.getElementById("staffing-panel");
  if (staffingPanel) {
    staffingPanel.style.display = state.channel === "WEB" ? "none" : "block";
  }

  // Disable B&M-only levers when WEB is selected
  Object.values(LEVER_CONFIG).forEach(category => {
    category.levers.forEach(lever => {
      if (lever.bmOnly) {
        const row = document.querySelector(`[data-lever-key="${lever.key}"]`);
        if (row) {
          if (state.channel === "WEB") {
            row.classList.add("disabled");
            state.levers[lever.key].enabled = false;
            const checkbox = document.getElementById(`lever-enabled-${lever.key}`);
            if (checkbox) checkbox.checked = false;
          } else {
            row.classList.remove("disabled");
          }
        }
      }
    });
  });
}

// Handle channel change
function handleChannelChange(e) {
  const selectedItem = e.target.querySelector("[pressed]");
  if (selectedItem) {
    state.channel = selectedItem.textContent.trim();
    updateLeversVisibility();
  }
}

// Handle slider change
function handleSliderChange(leverKey, value) {
  const numValue = parseFloat(value);
  state.levers[leverKey].targetValue = numValue;

  const input = document.getElementById(`lever-target-${leverKey}`);
  if (input && input.value !== String(numValue)) {
    input.value = numValue;
  }
}

// Handle target value input change
function handleTargetValueChange(leverKey, value) {
  const lever = state.levers[leverKey];
  const config = lever.config;

  let numValue = parseFloat(value) || 0;
  numValue = Math.max(config.min, Math.min(config.max, numValue));

  lever.targetValue = numValue;

  const slider = document.getElementById(`lever-slider-${leverKey}`);
  if (slider && slider.value !== numValue) {
    slider.value = numValue;
  }
}

// Handle lever toggle
function handleLeverToggle(leverKey, enabled) {
  state.levers[leverKey].enabled = enabled;

  // Enable/disable the timeline button
  const timelineBtn = document.getElementById(`lever-timeline-${leverKey}`);
  if (timelineBtn) {
    timelineBtn.disabled = !enabled;
  }

  // Reset to single value mode when disabled
  if (!enabled) {
    state.levers[leverKey].useWeeklyValues = false;
    state.levers[leverKey].weeklyValues = {};
    updateLeverRowVisualState(leverKey);
  }
}

// Handle operation change
function handleOperationChange(leverKey, operation) {
  state.levers[leverKey].operation = operation;
}

// Update horizon summary display
function updateHorizonSummary() {
  const summary = document.getElementById("horizon-summary");
  if (summary) {
    summary.textContent = `Week ${state.timeWindow.horizonStart} to Week ${state.timeWindow.horizonEnd}`;
  }
}

// Handle horizon change - adjust weekly values when horizon changes
function handleHorizonChange() {
  const start = state.timeWindow.horizonStart;
  const end = state.timeWindow.horizonEnd;

  Object.values(state.levers).forEach(lever => {
    if (lever.useWeeklyValues) {
      const newWeeklyValues = {};
      for (let w = start; w <= end; w++) {
        // Keep existing value if available, otherwise use target value
        newWeeklyValues[w] = lever.weeklyValues[w] !== undefined
          ? lever.weeklyValues[w]
          : lever.targetValue;
      }
      lever.weeklyValues = newWeeklyValues;
    }
  });
}

// Format date to YYYY-MM-DD
function formatDate(date) {
  if (!date) return null;
  const d = new Date(date);
  return d.toISOString().split("T")[0];
}

// Build output JSON
function buildScenarioJSON() {
  const modifications = [];

  Object.entries(state.levers).forEach(([key, lever]) => {
    if (lever.enabled) {
      const modification = {
        feature: key,
        operation: lever.operation,
        is_percentage: lever.operation !== "set"
      };

      // Include either single value or weekly values
      if (lever.useWeeklyValues) {
        modification.weekly_values = { ...lever.weeklyValues };
      } else {
        modification.value = lever.targetValue;
      }

      modifications.push(modification);
    }
  });

  return {
    scenario_name: state.scenarioName,
    time_window: {
      origin_date: formatDate(state.timeWindow.originDate),
      horizon_start: state.timeWindow.horizonStart,
      horizon_end: state.timeWindow.horizonEnd
    },
    scope: {
      profit_centers: state.scope.profitCenters,
      dmas: state.scope.dmas
    },
    modifications: modifications,
    channel: state.channel
  };
}

// Validate scenario
function validateScenario() {
  const errors = [];

  // Rule 1: At least one profit center or DMA
  if (state.scope.profitCenters.length === 0 && state.scope.dmas.length === 0) {
    errors.push("At least one profit center or DMA must be selected.");
  }

  // Rule 2: Horizon end >= horizon start
  if (state.timeWindow.horizonEnd < state.timeWindow.horizonStart) {
    errors.push("Horizon end week must be greater than or equal to start week.");
  }

  // Rule 3: Horizon values within 1-52
  if (state.timeWindow.horizonStart < 1 || state.timeWindow.horizonStart > 52) {
    errors.push("Horizon start must be between 1 and 52.");
  }
  if (state.timeWindow.horizonEnd < 1 || state.timeWindow.horizonEnd > 52) {
    errors.push("Horizon end must be between 1 and 52.");
  }

  // Rule 4: Values within min/max bounds
  Object.entries(state.levers).forEach(([key, lever]) => {
    if (lever.enabled) {
      const config = lever.config;
      if (lever.useWeeklyValues) {
        // Validate each weekly value
        Object.entries(lever.weeklyValues).forEach(([week, value]) => {
          if (value < config.min || value > config.max) {
            errors.push(`${config.displayName} value for Week ${week} must be between ${config.min} and ${config.max}.`);
          }
        });
      } else {
        if (lever.targetValue < config.min || lever.targetValue > config.max) {
          errors.push(`${config.displayName} value must be between ${config.min} and ${config.max}.`);
        }
      }
    }
  });

  // Rule 5: At least one lever modification enabled (warning)
  const hasModifications = Object.values(state.levers).some(l => l.enabled);
  if (!hasModifications) {
    errors.push("Warning: No lever modifications enabled. The scenario will use baseline values.");
  }

  // Rule 6: Origin date required
  if (!state.timeWindow.originDate) {
    errors.push("Origin date is required.");
  }

  state.validationErrors = errors;
  displayValidationErrors();

  return errors.filter(e => !e.startsWith("Warning:")).length === 0;
}

// Display validation errors
function displayValidationErrors() {
  const container = document.getElementById("validation-messages");
  if (!container) return;

  if (state.validationErrors.length === 0) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }

  container.classList.remove("hidden");
  container.innerHTML = state.validationErrors
    .map(error => {
      const isWarning = error.startsWith("Warning:");
      return `<ui5-message-strip design="${isWarning ? "Warning" : "Negative"}" hide-close-button>${error}</ui5-message-strip>`;
    })
    .join("");
}

// Show toast notification
function showToast(message) {
  const toast = document.getElementById("notification-toast");
  if (toast) {
    toast.textContent = message;
    toast.open = true;
  }
}

// Apply global operation mode to all levers
function applyGlobalMode() {
  const globalSelect = document.getElementById("global-operation");
  if (!globalSelect) return;

  const operation = globalSelect.value || "set";

  Object.keys(state.levers).forEach(leverKey => {
    state.levers[leverKey].operation = operation;

    // Update the UI select element
    const operationSelect = document.getElementById(`lever-operation-${leverKey}`);
    if (operationSelect) {
      operationSelect.value = operation;
    }
  });

  showToast(`Applied "${operation}" operation to all levers.`);
}

// Show JSON preview dialog
function showJsonPreview() {
  const json = buildScenarioJSON();
  const preview = document.getElementById("json-preview-content");
  const dialog = document.getElementById("json-preview-dialog");

  if (preview && dialog) {
    preview.textContent = JSON.stringify(json, null, 2);
    dialog.open = true;
  }
}

// Close JSON preview dialog
function closeJsonPreview() {
  const dialog = document.getElementById("json-preview-dialog");
  if (dialog) dialog.open = false;
}

// Weekly values dialog state
let currentDialogLeverKey = null;
let dialogWeeklyValues = {};

// Open weekly values dialog
function openWeeklyValuesDialog(leverKey) {
  currentDialogLeverKey = leverKey;
  const leverState = state.levers[leverKey];
  const config = leverState.config;

  // Set dialog header
  const dialog = document.getElementById("weekly-values-dialog");
  dialog.headerText = `Configure ${config.displayName} by Week`;

  // Set baseline value
  document.getElementById("dialog-baseline-value").textContent =
    `${leverState.currentValue} ${config.unit}`;

  // Set single value checkbox and input
  const singleValueCheckbox = document.getElementById("use-single-value");
  const singleValueInput = document.getElementById("single-value-input");
  singleValueCheckbox.checked = !leverState.useWeeklyValues;
  singleValueInput.value = String(leverState.targetValue);
  singleValueInput.setAttribute("min", config.min);
  singleValueInput.setAttribute("max", config.max);
  singleValueInput.setAttribute("step", config.step);
  document.getElementById("single-value-unit").textContent = config.unit;

  // Populate weekly values
  dialogWeeklyValues = { ...leverState.weeklyValues };
  if (Object.keys(dialogWeeklyValues).length === 0) {
    // Initialize with target value for all weeks
    for (let w = state.timeWindow.horizonStart; w <= state.timeWindow.horizonEnd; w++) {
      dialogWeeklyValues[w] = leverState.targetValue;
    }
  }

  renderWeeklyInputs(config);
  updateWeeklyInputsVisibility(!singleValueCheckbox.checked);

  dialog.open = true;
}

// Render weekly inputs in the dialog
function renderWeeklyInputs(config) {
  const container = document.getElementById("weekly-values-container");
  container.innerHTML = "";

  for (let week = state.timeWindow.horizonStart; week <= state.timeWindow.horizonEnd; week++) {
    const row = document.createElement("div");
    row.className = "week-input-row";

    const label = document.createElement("ui5-label");
    label.textContent = `Week ${week}`;

    const input = document.createElement("ui5-input");
    input.type = "Number";
    input.value = String(dialogWeeklyValues[week] !== undefined ? dialogWeeklyValues[week] : config.default);
    input.dataset.week = week;
    input.setAttribute("min", config.min);
    input.setAttribute("max", config.max);
    input.setAttribute("step", config.step);

    // Update dialogWeeklyValues when input changes
    input.addEventListener("change", (e) => {
      dialogWeeklyValues[e.target.dataset.week] = parseFloat(e.target.value);
    });

    const unit = document.createElement("ui5-label");
    unit.className = "unit-label";
    unit.textContent = config.unit;

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(unit);
    container.appendChild(row);
  }
}

// Show/hide weekly inputs container
function updateWeeklyInputsVisibility(show) {
  document.getElementById("weekly-values-container").style.display = show ? "block" : "none";
  document.querySelector(".weekly-dialog-content .quick-actions").style.display = show ? "flex" : "none";
}

// Apply weekly values from dialog to state
function applyWeeklyValues() {
  const leverState = state.levers[currentDialogLeverKey];
  const singleValueCheckbox = document.getElementById("use-single-value");

  if (singleValueCheckbox.checked) {
    // Single value mode
    leverState.useWeeklyValues = false;
    leverState.targetValue = parseFloat(document.getElementById("single-value-input").value);
    leverState.weeklyValues = {};
  } else {
    // Weekly values mode
    leverState.useWeeklyValues = true;
    leverState.weeklyValues = {};

    const inputs = document.querySelectorAll("#weekly-values-container ui5-input");
    inputs.forEach(input => {
      leverState.weeklyValues[input.dataset.week] = parseFloat(input.value);
    });
  }

  updateLeverRowVisualState(currentDialogLeverKey);
  document.getElementById("weekly-values-dialog").open = false;
  showToast("Weekly values applied.");
}

// Update lever row visual state based on weekly mode
function updateLeverRowVisualState(leverKey) {
  const leverState = state.levers[leverKey];
  const slider = document.getElementById(`lever-slider-${leverKey}`);
  const targetInput = document.getElementById(`lever-target-${leverKey}`);
  const timelineBtn = document.getElementById(`lever-timeline-${leverKey}`);

  if (leverState.useWeeklyValues) {
    // Disable slider/input when using weekly values
    if (slider) slider.disabled = true;
    if (targetInput) targetInput.disabled = true;
    if (timelineBtn) timelineBtn.design = "Attention";
  } else {
    if (slider) slider.disabled = !leverState.enabled;
    if (targetInput) targetInput.disabled = !leverState.enabled;
    if (timelineBtn) timelineBtn.design = "Transparent";
  }
}

// Close weekly values dialog
function closeWeeklyValuesDialog() {
  document.getElementById("weekly-values-dialog").open = false;
}

// Fill all weeks with a value
function fillAllWeeks() {
  const value = prompt("Enter value for all weeks:");
  if (value !== null && !isNaN(parseFloat(value))) {
    const inputs = document.querySelectorAll("#weekly-values-container ui5-input");
    inputs.forEach(input => {
      input.value = value;
      dialogWeeklyValues[input.dataset.week] = parseFloat(value);
    });
  }
}

// Copy Week 1 value to all other weeks
function copyWeek1ToAll() {
  const inputs = document.querySelectorAll("#weekly-values-container ui5-input");
  if (inputs.length > 0) {
    const week1Value = inputs[0].value;
    inputs.forEach(input => {
      input.value = week1Value;
      dialogWeeklyValues[input.dataset.week] = parseFloat(week1Value);
    });
  }
}

// Reset weekly values to baseline
function resetWeeklyValues() {
  const leverState = state.levers[currentDialogLeverKey];
  const inputs = document.querySelectorAll("#weekly-values-container ui5-input");
  inputs.forEach(input => {
    input.value = leverState.currentValue;
    dialogWeeklyValues[input.dataset.week] = leverState.currentValue;
  });
}

// Copy JSON to clipboard
async function copyJsonToClipboard() {
  const json = buildScenarioJSON();
  try {
    await navigator.clipboard.writeText(JSON.stringify(json, null, 2));
    showToast("JSON copied to clipboard!");
  } catch (err) {
    showToast("Failed to copy to clipboard.");
  }
}

// Reset to defaults
function resetToDefaults() {
  Object.values(LEVER_CONFIG).forEach(category => {
    category.levers.forEach(lever => {
      state.levers[lever.key] = {
        enabled: false,
        currentValue: lever.default,
        targetValue: lever.default,
        operation: "set",
        config: lever,
        useWeeklyValues: false,
        weeklyValues: {}
      };

      // Update UI
      const checkbox = document.getElementById(`lever-enabled-${lever.key}`);
      const slider = document.getElementById(`lever-slider-${lever.key}`);
      const target = document.getElementById(`lever-target-${lever.key}`);
      const operation = document.getElementById(`lever-operation-${lever.key}`);
      const timelineBtn = document.getElementById(`lever-timeline-${lever.key}`);

      if (checkbox) checkbox.checked = false;
      if (slider) {
        slider.value = lever.default;
        slider.disabled = false;
      }
      if (target) {
        target.value = lever.default;
        target.disabled = false;
      }
      if (operation) operation.value = "set";
      if (timelineBtn) {
        timelineBtn.disabled = true;
        timelineBtn.design = "Transparent";
      }
    });
  });

  showToast("All levers reset to default values.");
}

// Format date for filename (YYYY-MM-DD_HH-mm-ss)
function formatDateForFilename(date) {
  const d = new Date(date);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const hours = String(d.getHours()).padStart(2, "0");
  const minutes = String(d.getMinutes()).padStart(2, "0");
  const seconds = String(d.getSeconds()).padStart(2, "0");
  return `${year}-${month}-${day}_${hours}-${minutes}-${seconds}`;
}

// Download scenario as JSON file
function downloadScenarioJSON() {
  const scenario = buildScenarioJSON();
  const jsonString = JSON.stringify(scenario, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const scenarioName = (scenario.scenario_name || "scenario").replace(/[^a-z0-9]/gi, "_");
  const filename = `${scenarioName}_${formatDateForFilename(new Date())}.json`;

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast(`Scenario "${scenario.scenario_name}" downloaded.`);
}

// Trigger file input for loading scenario
function loadScenarioFromFile() {
  const fileInput = document.getElementById("scenario-file-input");
  if (fileInput) {
    fileInput.value = "";
    fileInput.click();
  }
}

// Validate uploaded scenario JSON
function validateUploadedScenario(scenario) {
  const errors = [];

  if (!scenario.scenario_name) {
    errors.push("Missing required field: scenario_name");
  }

  if (!scenario.time_window) {
    errors.push("Missing required field: time_window");
  }

  if (!scenario.scope) {
    errors.push("Missing required field: scope");
  }

  if (!scenario.channel) {
    errors.push("Missing required field: channel");
  } else if (!["B&M", "WEB"].includes(scenario.channel)) {
    errors.push("Invalid channel: must be 'B&M' or 'WEB'");
  }

  if (scenario.scope && scenario.scope.profit_centers) {
    const availableStoreIds = state.availableStores.map(s => s.id);
    const invalidStores = scenario.scope.profit_centers.filter(
      id => !availableStoreIds.includes(id)
    );
    if (invalidStores.length > 0) {
      errors.push(`Invalid store IDs: ${invalidStores.join(", ")}`);
    }
  }

  if (scenario.scope && scenario.scope.dmas) {
    const availableDmaNames = state.availableDmas.map(d => d.dma || d);
    const invalidDmas = scenario.scope.dmas.filter(
      dma => !availableDmaNames.includes(dma)
    );
    if (invalidDmas.length > 0) {
      errors.push(`Invalid DMAs: ${invalidDmas.join(", ")}`);
    }
  }

  return { valid: errors.length === 0, errors };
}

// Populate form from uploaded scenario
function populateFormFromScenario(scenario) {
  // Set scenario name
  state.scenarioName = scenario.scenario_name;
  const nameInput = document.getElementById("scenario-name-input");
  if (nameInput) nameInput.value = scenario.scenario_name;

  // Set channel
  state.channel = scenario.channel;
  const channelSelector = document.getElementById("channel-selector");
  if (channelSelector) {
    const items = channelSelector.querySelectorAll("ui5-segmented-button-item");
    items.forEach(item => {
      item.pressed = item.textContent.trim() === scenario.channel;
    });
  }
  updateLeversVisibility();

  // Set time window
  if (scenario.time_window) {
    state.timeWindow.originDate = scenario.time_window.origin_date;
    state.timeWindow.horizonStart = scenario.time_window.horizon_start || 1;
    state.timeWindow.horizonEnd = scenario.time_window.horizon_end || 13;

    const originDatePicker = document.getElementById("origin-date");
    if (originDatePicker) originDatePicker.value = scenario.time_window.origin_date || "";

    const horizonStartInput = document.getElementById("horizon-start");
    if (horizonStartInput) horizonStartInput.value = state.timeWindow.horizonStart;

    const horizonEndInput = document.getElementById("horizon-end");
    if (horizonEndInput) horizonEndInput.value = state.timeWindow.horizonEnd;

    updateHorizonSummary();
  }

  // Clear and set profit centers (stores)
  state.scope.profitCenters = scenario.scope.profit_centers || [];
  const storesContainer = document.getElementById("profit-centers-list");
  if (storesContainer) {
    storesContainer.querySelectorAll("ui5-checkbox").forEach(cb => {
      const item = cb.closest(".checkbox-list-item");
      const storeId = parseInt(item.dataset.id);
      cb.checked = state.scope.profitCenters.includes(storeId);
    });
  }
  updateSelectedCount("profit-centers");

  // Clear and set DMAs
  state.scope.dmas = scenario.scope.dmas || [];
  const dmasContainer = document.getElementById("dmas-list");
  if (dmasContainer) {
    dmasContainer.querySelectorAll("ui5-checkbox").forEach(cb => {
      const item = cb.closest(".checkbox-list-item");
      const dmaName = item.dataset.dma;
      cb.checked = state.scope.dmas.includes(dmaName);
    });
  }
  updateSelectedCount("dmas");

  // Reset all levers to defaults first
  Object.values(LEVER_CONFIG).forEach(category => {
    category.levers.forEach(lever => {
      state.levers[lever.key] = {
        enabled: false,
        currentValue: lever.default,
        targetValue: lever.default,
        operation: "set",
        config: lever,
        useWeeklyValues: false,
        weeklyValues: {}
      };

      const checkbox = document.getElementById(`lever-enabled-${lever.key}`);
      const slider = document.getElementById(`lever-slider-${lever.key}`);
      const target = document.getElementById(`lever-target-${lever.key}`);
      const operation = document.getElementById(`lever-operation-${lever.key}`);
      const timelineBtn = document.getElementById(`lever-timeline-${lever.key}`);

      if (checkbox) checkbox.checked = false;
      if (slider) {
        slider.value = lever.default;
        slider.disabled = false;
      }
      if (target) {
        target.value = lever.default;
        target.disabled = false;
      }
      if (operation) operation.value = "set";
      if (timelineBtn) {
        timelineBtn.disabled = true;
        timelineBtn.design = "Transparent";
      }
    });
  });

  // Apply modifications from scenario
  if (scenario.modifications && Array.isArray(scenario.modifications)) {
    scenario.modifications.forEach(mod => {
      const leverKey = mod.feature;
      if (!state.levers[leverKey]) return;

      const leverState = state.levers[leverKey];
      leverState.enabled = true;
      leverState.operation = mod.operation || "set";

      if (mod.weekly_values && Object.keys(mod.weekly_values).length > 0) {
        leverState.useWeeklyValues = true;
        leverState.weeklyValues = { ...mod.weekly_values };
      } else {
        leverState.useWeeklyValues = false;
        leverState.targetValue = mod.value !== undefined ? mod.value : leverState.config.default;
      }

      // Update UI elements
      const checkbox = document.getElementById(`lever-enabled-${leverKey}`);
      const slider = document.getElementById(`lever-slider-${leverKey}`);
      const target = document.getElementById(`lever-target-${leverKey}`);
      const operation = document.getElementById(`lever-operation-${leverKey}`);
      const timelineBtn = document.getElementById(`lever-timeline-${leverKey}`);

      if (checkbox) checkbox.checked = true;
      if (timelineBtn) timelineBtn.disabled = false;
      if (operation) operation.value = leverState.operation;

      if (leverState.useWeeklyValues) {
        if (slider) slider.disabled = true;
        if (target) target.disabled = true;
        if (timelineBtn) timelineBtn.design = "Attention";
      } else {
        if (slider) slider.value = leverState.targetValue;
        if (target) target.value = leverState.targetValue;
      }
    });
  }

  showToast(`Scenario "${scenario.scenario_name}" loaded successfully.`);
}

// Handle file upload
function handleScenarioFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = function(e) {
    try {
      const scenario = JSON.parse(e.target.result);

      // Validate the scenario
      const validation = validateUploadedScenario(scenario);
      if (!validation.valid) {
        showToast(`Error: ${validation.errors.join("; ")}`);
        return;
      }

      // Populate the form
      populateFormFromScenario(scenario);

    } catch (error) {
      console.error("Error parsing scenario file:", error);
      showToast("Error: Invalid JSON file format.");
    }
  };

  reader.onerror = function() {
    showToast("Error reading file.");
  };

  reader.readAsText(file);
}

// Submit scenario - redirects to chatbot for AI analysis
function submitScenario() {
  if (!validateScenario()) {
    showToast("Please fix validation errors before submitting.");
    return;
  }

  const scenario = buildScenarioJSON();

  // Convert scenario to analysis prompt for the AI agent
  const analysisPrompt = buildScenarioAnalysisPrompt(scenario);

  // Store in sessionStorage for chatbot to pick up
  sessionStorage.setItem("pendingScenarioAnalysis", JSON.stringify({
    scenario: scenario,
    prompt: analysisPrompt,
    timestamp: Date.now()
  }));

  showToast("Redirecting to AI Assistant for analysis...");

  // Navigate to chatbot page
  window.pageRouter.navigate("/chatbot");
}

// Populate stores checkbox list
async function loadStores() {
  try {
    const response = await request("/api/stores");
    if (response && response.stores) {
      state.availableStores = response.stores;
      const container = document.getElementById("profit-centers-list");
      if (container) {
        // Build checkbox list items
        container.innerHTML = response.stores.map(store =>
          `<div class="checkbox-list-item" data-id="${store.id}" data-searchtext="store ${store.id} ${(store.dma || '').toLowerCase()}">
            <ui5-checkbox text="Store ${store.id} - ${store.dma || 'Unknown'}"></ui5-checkbox>
          </div>`
        ).join('');

        // Add event listeners to each checkbox
        container.querySelectorAll('ui5-checkbox').forEach(cb => {
          cb.addEventListener('change', handleStoreCheckboxChange);
        });

        updateSelectedCount('profit-centers');
      }
    }
  } catch (error) {
    console.error("Error loading stores:", error);
  }
}

// Populate DMAs checkbox list
async function loadDmas() {
  try {
    const response = await request("/api/dma");
    if (response && response.dmas) {
      state.availableDmas = response.dmas;
      const container = document.getElementById("dmas-list");
      if (container) {
        // Build checkbox list items
        container.innerHTML = response.dmas.map(dma => {
          const dmaName = dma.dma || dma;
          return `<div class="checkbox-list-item" data-dma="${dmaName}" data-searchtext="${dmaName.toLowerCase()}">
            <ui5-checkbox text="${dmaName}"></ui5-checkbox>
          </div>`;
        }).join('');

        // Add event listeners to each checkbox
        container.querySelectorAll('ui5-checkbox').forEach(cb => {
          cb.addEventListener('change', handleDmaCheckboxChange);
        });

        updateSelectedCount('dmas');
      }
    }
  } catch (error) {
    console.error("Error loading DMAs:", error);
  }
}

// Handle store checkbox change
function handleStoreCheckboxChange(e) {
  const item = e.target.closest('.checkbox-list-item');
  const id = parseInt(item.dataset.id);

  if (e.target.checked) {
    if (!state.scope.profitCenters.includes(id)) {
      state.scope.profitCenters.push(id);
    }
  } else {
    state.scope.profitCenters = state.scope.profitCenters.filter(x => x !== id);
  }

  updateSelectedCount('profit-centers');
}

// Handle DMA checkbox change
function handleDmaCheckboxChange(e) {
  const item = e.target.closest('.checkbox-list-item');
  const dmaName = item.dataset.dma;

  if (e.target.checked) {
    if (!state.scope.dmas.includes(dmaName)) {
      state.scope.dmas.push(dmaName);
    }
  } else {
    state.scope.dmas = state.scope.dmas.filter(x => x !== dmaName);
  }

  updateSelectedCount('dmas');
}

// Update selected count display
function updateSelectedCount(type) {
  if (type === 'profit-centers') {
    const countEl = document.getElementById('profit-centers-selected');
    if (countEl) {
      const count = state.scope.profitCenters.length;
      countEl.textContent = `${count} store${count !== 1 ? 's' : ''} selected`;
    }
  } else if (type === 'dmas') {
    const countEl = document.getElementById('dmas-selected');
    if (countEl) {
      const count = state.scope.dmas.length;
      countEl.textContent = `${count} DMA${count !== 1 ? 's' : ''} selected`;
    }
  }
}

// Handle select all stores
function handleSelectAllStores(e) {
  const checked = e.target.checked;
  state.scope.allProfitCenters = checked;

  // Update state
  if (checked) {
    state.scope.profitCenters = state.availableStores.map(s => s.id);
  } else {
    state.scope.profitCenters = [];
  }

  // Update UI checkboxes
  const container = document.getElementById("profit-centers-list");
  if (container) {
    container.querySelectorAll('ui5-checkbox').forEach(cb => {
      cb.checked = checked;
    });
  }

  updateSelectedCount('profit-centers');
}

// Handle select all DMAs
function handleSelectAllDmas(e) {
  const checked = e.target.checked;
  state.scope.allDmas = checked;

  // Update state
  if (checked) {
    state.scope.dmas = state.availableDmas.map(d => d.dma || d);
  } else {
    state.scope.dmas = [];
  }

  // Update UI checkboxes
  const container = document.getElementById("dmas-list");
  if (container) {
    container.querySelectorAll('ui5-checkbox').forEach(cb => {
      cb.checked = checked;
    });
  }

  updateSelectedCount('dmas');
}

// Setup search filtering for checkbox lists
function setupSearchFiltering() {
  // Store search
  const storeSearch = document.getElementById('profit-centers-search');
  if (storeSearch) {
    storeSearch.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const items = document.querySelectorAll('#profit-centers-list .checkbox-list-item');
      items.forEach(item => {
        const text = item.dataset.searchtext || '';
        item.classList.toggle('hidden', query && !text.includes(query));
      });
    });
  }

  // DMA search
  const dmaSearch = document.getElementById('dmas-search');
  if (dmaSearch) {
    dmaSearch.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const items = document.querySelectorAll('#dmas-list .checkbox-list-item');
      items.forEach(item => {
        const text = item.dataset.searchtext || '';
        item.classList.toggle('hidden', query && !text.includes(query));
      });
    });
  }
}

// Setup event handlers
function setupEventHandlers() {
  // Channel selector
  const channelSelector = document.getElementById("channel-selector");
  if (channelSelector) {
    channelSelector.addEventListener("ui5-selection-change", handleChannelChange);
  }

  // Scenario name
  const scenarioNameInput = document.getElementById("scenario-name-input");
  if (scenarioNameInput) {
    scenarioNameInput.addEventListener("change", (e) => {
      state.scenarioName = e.target.value;
    });
  }

  // Time window controls
  const originDate = document.getElementById("origin-date");
  if (originDate) {
    originDate.addEventListener("change", (e) => {
      state.timeWindow.originDate = e.target.value;
    });
  }

  const horizonStart = document.getElementById("horizon-start");
  if (horizonStart) {
    horizonStart.addEventListener("change", (e) => {
      state.timeWindow.horizonStart = parseInt(e.target.value) || 1;
      updateHorizonSummary();
      handleHorizonChange();
    });
  }

  const horizonEnd = document.getElementById("horizon-end");
  if (horizonEnd) {
    horizonEnd.addEventListener("change", (e) => {
      state.timeWindow.horizonEnd = parseInt(e.target.value) || 13;
      updateHorizonSummary();
      handleHorizonChange();
    });
  }

  // Select All checkboxes
  const selectAllStores = document.getElementById("select-all-stores");
  if (selectAllStores) {
    selectAllStores.addEventListener("change", handleSelectAllStores);
  }

  const selectAllDmas = document.getElementById("select-all-dmas");
  if (selectAllDmas) {
    selectAllDmas.addEventListener("change", handleSelectAllDmas);
  }

  // Apply global mode button
  const applyGlobalModeBtn = document.getElementById("apply-global-mode");
  if (applyGlobalModeBtn) {
    applyGlobalModeBtn.addEventListener("click", applyGlobalMode);
  }

  // Action buttons
  const resetBtn = document.getElementById("reset-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", resetToDefaults);
  }

  const loadScenarioBtn = document.getElementById("load-scenario-btn");
  if (loadScenarioBtn) {
    loadScenarioBtn.addEventListener("click", loadScenarioFromFile);
  }

  const scenarioFileInput = document.getElementById("scenario-file-input");
  if (scenarioFileInput) {
    scenarioFileInput.addEventListener("change", handleScenarioFileUpload);
  }

  const previewJsonBtn = document.getElementById("preview-json-btn");
  if (previewJsonBtn) {
    previewJsonBtn.addEventListener("click", showJsonPreview);
  }

  const saveScenarioBtn = document.getElementById("save-scenario-btn");
  if (saveScenarioBtn) {
    saveScenarioBtn.addEventListener("click", downloadScenarioJSON);
  }

  const submitScenarioBtn = document.getElementById("submit-scenario-btn");
  if (submitScenarioBtn) {
    submitScenarioBtn.addEventListener("click", submitScenario);
  }

  // Dialog buttons
  const copyJsonBtn = document.getElementById("copy-json-btn");
  if (copyJsonBtn) {
    copyJsonBtn.addEventListener("click", copyJsonToClipboard);
  }

  const closePreviewBtn = document.getElementById("close-preview-btn");
  if (closePreviewBtn) {
    closePreviewBtn.addEventListener("click", closeJsonPreview);
  }

  // Weekly values dialog buttons
  const useSingleValueCheckbox = document.getElementById("use-single-value");
  if (useSingleValueCheckbox) {
    useSingleValueCheckbox.addEventListener("change", (e) => {
      updateWeeklyInputsVisibility(!e.target.checked);
    });
  }

  const dialogCancelBtn = document.getElementById("dialog-cancel-btn");
  if (dialogCancelBtn) {
    dialogCancelBtn.addEventListener("click", closeWeeklyValuesDialog);
  }

  const dialogApplyBtn = document.getElementById("dialog-apply-btn");
  if (dialogApplyBtn) {
    dialogApplyBtn.addEventListener("click", applyWeeklyValues);
  }

  const fillAllBtn = document.getElementById("fill-all-btn");
  if (fillAllBtn) {
    fillAllBtn.addEventListener("click", fillAllWeeks);
  }

  const copyWeek1Btn = document.getElementById("copy-week1-btn");
  if (copyWeek1Btn) {
    copyWeek1Btn.addEventListener("click", copyWeek1ToAll);
  }

  const resetWeeklyBtn = document.getElementById("reset-weekly-btn");
  if (resetWeeklyBtn) {
    resetWeeklyBtn.addEventListener("click", resetWeeklyValues);
  }

  // Lever event delegation
  setupLeverEventDelegation();
}

// Setup lever event delegation
function setupLeverEventDelegation() {
  // Use specific container IDs for better reliability than .lever-panel
  const containerIds = Object.values(LEVER_CONFIG).map(c => c.containerId);
  
  containerIds.forEach(id => {
    const container = document.getElementById(id);
    if (!container) return;

    // Handle change events (checkboxes, selects)
    container.addEventListener("change", (e) => {
      const leverRow = e.target.closest(".lever-row");
      if (!leverRow) return;

      const leverKey = leverRow.dataset.leverKey;

      if (e.target.classList.contains("lever-checkbox")) {
        handleLeverToggle(leverKey, e.target.checked);
      } else if (e.target.classList.contains("lever-slider")) {
        handleSliderChange(leverKey, e.target.value);
      } else if (e.target.classList.contains("target-value")) {
        handleTargetValueChange(leverKey, e.target.value);
      } else if (e.target.classList.contains("lever-operation")) {
        handleOperationChange(leverKey, e.target.value);
      }
    });

    // Also handle input events for real-time slider sync
    container.addEventListener("input", (e) => {
      const leverRow = e.target.closest(".lever-row");
      if (!leverRow) return;

      const leverKey = leverRow.dataset.leverKey;

      if (e.target.classList.contains("lever-slider")) {
        handleSliderChange(leverKey, e.target.value);
      }
    });

    // Handle click events (delegation) for timeline buttons
    container.addEventListener("click", (e) => {
      // Look for the timeline button
      // Note: e.target might be the icon inside the button, so we look up the tree
      const timelineBtn = e.target.closest(".lever-timeline-btn");
      
      if (timelineBtn) {
        if (timelineBtn.disabled) return; // Ignore disabled buttons
        
        const leverRow = timelineBtn.closest(".lever-row");
        if (leverRow) {
          const leverKey = leverRow.dataset.leverKey;
          if (leverKey && state.levers[leverKey]) {
            openWeeklyValuesDialog(leverKey);
          }
        }
      }
    });
  });
}

// Main initialization
export default async function initScenarioMakerPage() {
  // Initialize lever state
  initializeLeverState();

  // Render lever rows
  renderLevers();

  // Load data from API
  try {
    await Promise.all([loadStores(), loadDmas()]);
  } catch (error) {
    console.error("Error loading initial data:", error);
  }

  // Setup event handlers and search filtering
  setupEventHandlers();
  setupSearchFiltering();

  // Initialize horizon summary
  updateHorizonSummary();
}
