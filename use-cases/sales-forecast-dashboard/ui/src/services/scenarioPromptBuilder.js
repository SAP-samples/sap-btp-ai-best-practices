/**
 * Scenario Prompt Builder
 *
 * Converts scenario JSON from Scenario Maker into a natural language prompt
 * for the AI agent to process using its standard analysis workflow.
 */

// Feature display names and units (mirrors LEVER_CONFIG in scenario-maker.js)
const FEATURE_INFO = {
  "pct_primary_financing_roll_mean_4": { displayName: "Primary Financing", unit: "%" },
  "pct_secondary_financing_roll_mean_4": { displayName: "Secondary Financing", unit: "%" },
  "pct_tertiary_financing_roll_mean_4": { displayName: "Tertiary Financing", unit: "%" },
  "staffing_unique_associates_roll_mean_4": { displayName: "Unique Associates", unit: "count" },
  "staffing_hours_roll_mean_4": { displayName: "Staffing Hours", unit: "hours" },
  "pct_omni_channel_roll_mean_4": { displayName: "Omni-Channel", unit: "%" },
  "pct_value_product_roll_mean_4": { displayName: "Value Product", unit: "%" },
  "pct_premium_product_roll_mean_4": { displayName: "Premium Product", unit: "%" },
  "pct_white_glove_roll_mean_4": { displayName: "White Glove", unit: "%" },
  "brand_awareness_dma_roll_mean_4": { displayName: "Brand Awareness", unit: "%" },
  "brand_consideration_dma_roll_mean_4": { displayName: "Brand Consideration", unit: "%" }
};

const OPERATION_LABELS = {
  "set": "set to",
  "increase": "increase by",
  "decrease": "decrease by"
};

/**
 * Format the scope (stores/DMAs) as a human-readable string
 */
function formatScope(scenario) {
  const parts = [];

  if (scenario.scope.profit_centers && scenario.scope.profit_centers.length > 0) {
    const count = scenario.scope.profit_centers.length;
    if (count <= 5) {
      parts.push(`Stores: ${scenario.scope.profit_centers.join(", ")}`);
    } else {
      parts.push(`${count} stores`);
    }
  }

  if (scenario.scope.dmas && scenario.scope.dmas.length > 0) {
    const count = scenario.scope.dmas.length;
    if (count <= 3) {
      parts.push(`DMAs: ${scenario.scope.dmas.join(", ")}`);
    } else {
      parts.push(`${count} DMAs`);
    }
  }

  return parts.length > 0 ? parts.join(" and ") : "All stores";
}

/**
 * Format a single modification as a human-readable string
 */
function formatModification(mod) {
  const info = FEATURE_INFO[mod.feature] || { displayName: mod.feature, unit: "" };
  const displayName = info.displayName;
  const opLabel = OPERATION_LABELS[mod.operation] || mod.operation;
  const unit = mod.is_percentage ? "%" : (info.unit !== "%" ? ` ${info.unit}` : "");

  if (mod.weekly_values && Object.keys(mod.weekly_values).length > 0) {
    // Has weekly values - format each week
    const weeks = Object.entries(mod.weekly_values)
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

    const weeklyParts = weeks.map(([week, value]) =>
      `Week ${week}: ${opLabel} ${value}${unit}`
    );

    return `- **${displayName}**:\n  ${weeklyParts.join("\n  ")}`;
  } else {
    // Single value
    return `- **${displayName}**: ${opLabel} ${mod.value}${unit}`;
  }
}

/**
 * Build scope parameters for the agent prompt
 */
function buildScopeParams(scenario) {
  const params = [];

  if (scenario.scope.profit_centers && scenario.scope.profit_centers.length > 0) {
    params.push(`store_ids: [${scenario.scope.profit_centers.join(", ")}]`);
  }

  if (scenario.scope.dmas && scenario.scope.dmas.length > 0) {
    const dmaList = scenario.scope.dmas.map(d => `"${d}"`).join(", ");
    params.push(`dmas: [${dmaList}]`);
  }

  return params;
}

/**
 * Convert scenario JSON to a natural language prompt for the AI agent
 *
 * @param {Object} scenario - The scenario JSON from buildScenarioJSON()
 * @returns {string} - Natural language prompt for the agent
 */
export function buildScenarioAnalysisPrompt(scenario) {
  const scope = formatScope(scenario);
  const scopeParams = buildScopeParams(scenario);

  const modificationsText = scenario.modifications.length > 0
    ? scenario.modifications.map(formatModification).join("\n")
    : "No modifications (baseline analysis only)";

  // Build the prompt
  const prompt = `Analyze this what-if scenario and provide full predictions with SHAP explanations:

**SCENARIO CONFIGURATION**
- Name: "${scenario.scenario_name}"
- Channel: ${scenario.channel}
- Origin Date: ${scenario.time_window.origin_date}
- Forecast Horizon: Week ${scenario.time_window.horizon_start} to Week ${scenario.time_window.horizon_end}
- Scope: ${scope}

**LEVER MODIFICATIONS**
${modificationsText}

Please perform the following analysis:

1. **Initialize** the forecast simulation with:
   - origin_date: "${scenario.time_window.origin_date}"
   - horizon_weeks: ${scenario.time_window.horizon_end}
   - channel: "${scenario.channel}"
   ${scopeParams.length > 0 ? `- ${scopeParams.join("\n   - ")}` : ""}

2. **Create a scenario** named "${scenario.scenario_name}" forked from baseline

3. **Apply all modifications** listed above to the scenario

4. **Run the forecast model** for both baseline and "${scenario.scenario_name}"

5. **Compare scenarios** showing:
   - Total dollar impact on predicted sales
   - Percentage change from baseline
   - Weekly breakdown of differences

6. **Explain the forecast change** using SHAP attribution to identify the key drivers

7. **Generate comparison charts** visualizing the forecast differences

8. **Provide a narrative summary** with:
   - Key findings from the analysis
   - Top drivers of the sales change
   - Recommendations for optimization`;

  return prompt;
}

/**
 * Get a brief summary of the scenario for UI display
 *
 * @param {Object} scenario - The scenario JSON
 * @returns {Object} - Summary object with key properties
 */
export function getScenarioSummary(scenario) {
  return {
    name: scenario.scenario_name,
    channel: scenario.channel,
    originDate: scenario.time_window.origin_date,
    horizon: `Week ${scenario.time_window.horizon_start}-${scenario.time_window.horizon_end}`,
    modificationCount: scenario.modifications.length,
    storeCount: scenario.scope.profit_centers?.length || 0,
    dmaCount: scenario.scope.dmas?.length || 0,
    scope: formatScope(scenario)
  };
}
