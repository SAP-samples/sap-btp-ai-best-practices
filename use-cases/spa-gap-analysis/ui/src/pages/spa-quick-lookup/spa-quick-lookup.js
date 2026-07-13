/* SPA Quick Lookup page - UI5 components */
/* VERSION: 2026-04-20-v2 - Checkbox fix with server reload */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Tag.js";  // Badge replacement - use Tag instead
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/CheckBox.js";

import { request } from "../../services/api.js";

// Version check - MUST be after imports in ES modules
console.log('🔄 SPA Quick Lookup JS loaded - VERSION: 2026-04-20-v2');

// LocalStorage keys
const STORAGE_KEYS = {
  LAST_CUSTOMER_ID: 'spa_quick_lookup_last_customer_id',
  LAST_RESULTS: 'spa_quick_lookup_last_results'
};

// Utility: Format currency values
function formatCurrency(value) {
  if (value === null || value === undefined || isNaN(value)) {
    return '$0.00';
  }
  return '$' + value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

function formatDateValue(value) {
  if (!value) {
    return 'Open';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit'
  });
}

function formatAssignmentScope(scope) {
  const labels = {
    sold_to: 'Direct Sold-To',
    direct: 'Direct Sold-To',
    sales_office: 'Sales Office',
    price_list: 'Price List',
    price_group: 'Price Group',
    sold_to_plant: 'Sold-To + Plant',
    plant: 'Plant',
    payer: 'Payer',
    legacy: 'Legacy'
  };
  return labels[scope] || scope || 'Unknown scope';
}

function formatPricingSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return 'No priced materials';
  }
  return sources.join(', ');
}

function formatBoolChip(value, trueText, falseText) {
  if (value === true) {
    return trueText;
  }
  if (value === false) {
    return falseText;
  }
  return '';
}

// Save state to sessionStorage
function saveState(customerId, results) {
  try {
    sessionStorage.setItem(STORAGE_KEYS.LAST_CUSTOMER_ID, customerId);
    sessionStorage.setItem(STORAGE_KEYS.LAST_RESULTS, JSON.stringify(results));
  } catch (error) {
    console.error('Failed to save Quick Lookup state:', error);
  }
}

// Load state from sessionStorage
function loadState() {
  try {
    const savedCustomerId = sessionStorage.getItem(STORAGE_KEYS.LAST_CUSTOMER_ID);
    const savedResults = sessionStorage.getItem(STORAGE_KEYS.LAST_RESULTS);

    if (savedCustomerId && savedResults) {
      return {
        customerId: savedCustomerId,
        results: JSON.parse(savedResults)
      };
    }
  } catch (error) {
    console.error('Failed to load Quick Lookup state:', error);
  }
  return null;
}

// API functions
async function quickLookup(customerId, excludeUnknown = false) {
  return await request(
    "/api/spa/quick-lookup",
    "POST",
    {
      customer_id: customerId,
      top_n_similar: 50,
      min_similar_count: 2,
      include_rfm: true,
      include_price_group: true,
      exclude_unknown: excludeUnknown
    }
  );
}

async function getSpaDetails(spaId) {
  return await request(`/api/spa/${spaId}`, "GET");
}

async function getPotentialBreakdown(customerId) {
  return await request(`/api/spa/customer/${customerId}/potential-breakdown`, "GET");
}

async function getSpaMaterials(spaId, customerId) {
  return await request(`/api/spa/spa/${spaId}/materials?customer_id=${customerId}`, "GET");
}

// RFM Segment explanations
function getRFMExplanation(segment) {
  const explanations = {
    'Champions': 'Best customers - Recent buyers, frequent orders, high spending',
    'Loyal': 'Consistent customers - Regular orders, good spending',
    'Potential Loyalist': 'Promising customers - Recent buyers with potential',
    'At Risk': 'Declining customers - Haven\'t ordered recently, need attention',
    'Need Attention': 'Below average activity - Require engagement',
    'About to Sleep': 'Low recent activity - Risk of churn',
    'Lost': 'Inactive customers - No recent orders',
    'N/A': 'No transaction history available'
  };
  return explanations[segment] || 'Customer activity segment based on Recency, Frequency, Monetary analysis';
}

export default function initSpaQuickLookupPage() {
  console.log("SPA Quick Lookup page initialized");
  console.log('[Quick Lookup] Init function started');

  const customerIdInput = document.getElementById("customer-id-input");
  const analyzeButton = document.getElementById("analyze-button");
  const hasNamesCheckbox = document.getElementById("filter-has-names");

  console.log('[Quick Lookup] Elements found:', {
    customerIdInput: !!customerIdInput,
    analyzeButton: !!analyzeButton,
    hasNamesCheckbox: !!hasNamesCheckbox
  });
  const loadingIndicator = document.getElementById("loading-indicator");
  const resultsContainer = document.getElementById("results-container");
  const errorPanel = document.getElementById("error-panel");
  const errorMessage = document.getElementById("error-message");

  // Store last results for re-filtering
  let lastResults = null;

  // Load saved state from sessionStorage
  const savedState = loadState();
  if (savedState) {
    customerIdInput.value = savedState.customerId;
    lastResults = savedState.results;
    displayResults(savedState.results);
  }

  // Check if navigated from Agent Chat or Summary View with customer ID
  function checkNavigationCustomerId() {
    console.log('[Quick Lookup] checkNavigationCustomerId called');
    const navigateCustomerId = sessionStorage.getItem('navigate_to_customer_id');
    console.log('[Quick Lookup] sessionStorage value:', navigateCustomerId);

    if (navigateCustomerId) {
      console.log('[Quick Lookup] Found customer ID in sessionStorage:', navigateCustomerId);
      sessionStorage.removeItem('navigate_to_customer_id'); // Clear after reading
      console.log('[Quick Lookup] Removed from sessionStorage');

      customerIdInput.value = navigateCustomerId;
      console.log('[Quick Lookup] Set input value to:', customerIdInput.value);

      // Trigger analysis automatically
      setTimeout(() => {
        console.log('[Quick Lookup] Attempting to click analyze button. Button disabled?', analyzeButton?.disabled);
        if (analyzeButton && !analyzeButton.disabled) {
          console.log('[Quick Lookup] Clicking analyze button now');
          analyzeButton.click();
        } else {
          console.log('[Quick Lookup] Cannot click - button not available or disabled');
        }
      }, 300);
    } else {
      console.log('[Quick Lookup] No customer ID found in sessionStorage');
    }
  }

  // Check on initial load
  console.log('[Quick Lookup] Initial check on page load');
  checkNavigationCustomerId();

  // Poll for navigation requests every 500ms (only when page is visible and not analyzing)
  console.log('[Quick Lookup] Setting up polling interval');
  const navigationCheckInterval = setInterval(() => {
    // Only check if page is visible and not currently analyzing
    if (!document.hidden && analyzeButton && !analyzeButton.disabled) {
      console.log('[Quick Lookup] Polling check triggered');
      checkNavigationCustomerId();
    }
  }, 500);

  // Cleanup interval when page unloads (optional, but good practice)
  window.addEventListener('beforeunload', () => {
    clearInterval(navigationCheckInterval);
  });

  // Checkbox filter handler - re-filter similar customers
  if (hasNamesCheckbox) {
    hasNamesCheckbox.addEventListener("change", () => {
      console.log('[DEBUG] Checkbox changed, checked:', hasNamesCheckbox.checked);
      console.log('[DEBUG] Customer ID:', customerIdInput.value.trim());
      // Re-run analysis with new filter setting
      if (customerIdInput.value.trim()) {
        console.log('[DEBUG] Re-running analysis with new filter...');
        analyzeButton.click();
      } else {
        console.log('[DEBUG] No customer ID, skipping re-analysis');
      }
    });
  }

  // Handle analyze button click
  if (analyzeButton && customerIdInput) {
    analyzeButton.addEventListener("click", async () => {
      const customerId = customerIdInput.value.trim();

      if (!customerId) {
        showError("Please enter a customer ID");
        return;
      }

      // Get checkbox value
      const excludeUnknown = hasNamesCheckbox ? hasNamesCheckbox.checked : false;

      // Show loading
      loadingIndicator.style.display = "block";
      resultsContainer.style.display = "none";
      errorPanel.style.display = "none";
      analyzeButton.disabled = true;

      try {
        const results = await quickLookup(customerId, excludeUnknown);
        lastResults = results; // Store for re-filtering
        saveState(customerId, results); // Save results to sessionStorage
        displayResults(results);
      } catch (error) {
        showError(error.message || "Failed to analyze customer");
      } finally {
        loadingIndicator.style.display = "none";
        analyzeButton.disabled = false;
      }
    });

    // Allow Enter key to trigger analysis
    customerIdInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        analyzeButton.click();
      }
    });
  }

  function displayResults(data) {
    // Display customer profile
    displayCustomerProfile(data.target_customer);

    // Display savings analysis
    displaySavingsAnalysis(data.target_customer);

    // Initialize AI insights button for this customer
    if (window.initializeInsightsButton) {
      window.initializeInsightsButton(data.target_customer.customer_id);
    }

    // Display dynamic example calculation
    displayExampleCalculation(data.target_customer);

    // Display missing SPAs
    displayMissingSPAs(data.missing_spas);

    // Display similar customers
    displaySimilarCustomers(data.similar_customers);

    // NEW: Fetch and display material summary (pass total_cogs for coverage calculation)
    fetchMaterialSummary(data.target_customer.customer_id)
      .then(materialData => displayMaterialSummary(materialData, data.target_customer.total_cogs))
      .catch(error => {
        console.error('Failed to load material summary:', error);
        // Hide card on error
        const card = document.getElementById('material-summary-card');
        if (card) card.style.display = 'none';
      });

    // Show results
    resultsContainer.style.display = "block";
  }

  function displayCustomerProfile(customer) {
    const profileContainer = document.getElementById("customer-profile");
    const currentSpaDetails = Array.isArray(customer.current_spa_details)
      ? [...customer.current_spa_details].sort((a, b) => {
          const coveredDiff = (Number(b.covered_cogs) || 0) - (Number(a.covered_cogs) || 0);
          if (coveredDiff !== 0) {
            return coveredDiff;
          }

          const savingsDiff = (Number(b.current_savings) || 0) - (Number(a.current_savings) || 0);
          if (savingsDiff !== 0) {
            return savingsDiff;
          }

          return String(a.sales_deal || '').localeCompare(String(b.sales_deal || ''));
        })
      : [];
    const spaCountUnique = customer.current_spa_count_unique ?? customer.spa_count ?? currentSpaDetails.length;
    const spaRowCount = customer.current_spa_row_count ?? spaCountUnique;
    const snapshotDate = customer.snapshot_date ? formatDateValue(customer.snapshot_date) : null;
    const countSemantics = spaRowCount > spaCountUnique
      ? `${spaCountUnique} unique active agreements across ${spaRowCount} active A701 rows`
      : `${spaCountUnique} unique active agreements`;
    const spaListMarkup = currentSpaDetails.length > 0
      ? `
          <div style="overflow-x: auto; margin-top: 0.75rem;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem;">
              <thead>
                <tr style="background: #f8fafc; border-bottom: 1px solid #dbe3ea;">
                  <th style="padding: 0.55rem; text-align: left;">SPA</th>
                  <th style="padding: 0.55rem; text-align: left;">Valid From</th>
                  <th style="padding: 0.55rem; text-align: left;">Valid To</th>
                  <th style="padding: 0.55rem; text-align: right;">Materials</th>
                  <th style="padding: 0.55rem; text-align: right;">Covered COGS</th>
                  <th style="padding: 0.55rem; text-align: right;">Current Savings</th>
                </tr>
              </thead>
              <tbody>
                ${currentSpaDetails.map(spa => {
                  const description = spa.agreement_description || spa.description_of_agreement || spa.external_description || '';
                  const metadata = [
                    spa.grouping ? `Grouping ${spa.grouping}` : null,
                    formatAssignmentScope(spa.assignment_scope),
                    spa.agreement_type,
                    formatPricingSources(spa.pricing_sources)
                  ].filter(Boolean).join(' | ');
                  return `
                  <tr style="border-bottom: 1px solid #eef2f6;">
                    <td style="padding: 0.55rem;">
                      <div>
                        <span class="clickable-spa-id" data-spa-id="${spa.sales_deal}" style="cursor: pointer; color: #0070f2; text-decoration: underline; font-family: monospace; font-weight: 600;">${spa.sales_deal}</span>
                      </div>
                      ${description ? `<div style="font-size: 0.78rem; color: #374151; margin-top: 0.15rem; max-width: 32rem;">${description}</div>` : ''}
                      <div style="font-size: 0.76rem; color: #777; margin-top: 0.15rem;">${metadata || '—'}</div>
                      ${spa.is_supplyforce ? `<div style="font-size: 0.74rem; color: #8a5a00; margin-top: 0.15rem;">Supplyforce: attached/current only, excluded from opportunity recommendations</div>` : ''}
                      <div style="display: none;">
                      <div style="font-size: 0.76rem; color: #777; margin-top: 0.15rem;">
                        ${[spa.grouping, spa.assignment_scope, spa.agreement_type].filter(Boolean).join(' • ') || '—'}
                      </div>
                      </div>
                    </td>
                    <td style="padding: 0.55rem;">${formatDateValue(spa.valid_from)}</td>
                    <td style="padding: 0.55rem;">${formatDateValue(spa.valid_to)}</td>
                    <td style="padding: 0.55rem; text-align: right;">${spa.covered_materials ?? 0}</td>
                    <td style="padding: 0.55rem; text-align: right;">${formatCurrency(spa.covered_cogs || 0)}</td>
                    <td style="padding: 0.55rem; text-align: right; color: #1f7a3d; font-weight: 600;">${formatCurrency(spa.current_savings || 0)}</td>
                  </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        `
      : (customer.current_spas && customer.current_spas.length > 0
        ? `<div style="font-size: 0.85em; color: #666; margin-top: 0.5rem;">
            ${customer.current_spas.map(spa =>
              `<span class="clickable-spa-id" data-spa-id="${spa}" style="cursor: pointer; color: #0070f2; text-decoration: underline; margin-right: 8px;">${spa}</span>`
            ).join('')}
           </div>`
        : '<div style="font-size: 0.85em; color: #666; margin-top: 0.5rem;">No active SPAs found.</div>');

    // Build location string only if data exists
    const city = customer.city && customer.city !== 'N/A' ? customer.city : '';
    const state = customer.state && customer.state !== 'N/A' ? customer.state : '';
    const location = [city, state].filter(Boolean).join(', ') || '—';

    profileContainer.innerHTML = `
      <div class="customer-info-grid">
        <div class="info-item">
          <div class="info-label">Customer ID</div>
          <div class="info-value">${customer.customer_id}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Name</div>
          <div class="info-value">${customer.customer_name || "Unknown"}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Sales Office</div>
          <div class="info-value">${customer.sales_office || "—"}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Location</div>
          <div class="info-value">${location}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Customer Type</div>
          <div class="info-value">${customer.pl_type || "—"}</div>
        </div>
        <div class="info-item">
          <div class="info-label">RFM Segment</div>
          <div class="info-value">
            ${customer.rfm_segment || "—"}
            ${customer.rfm_segment
              ? `<div style="font-size: 0.85em; color: #666; margin-top: 4px; font-style: italic;">${getRFMExplanation(customer.rfm_segment)}</div>`
              : ''}
          </div>
        </div>
        <div class="info-item">
          <div class="info-label">Total COGS</div>
          <div class="info-value">$${(customer.total_cogs || 0).toLocaleString()}</div>
        </div>
      </div>
      <div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
        <div style="font-size: 0.9rem; font-weight: 600; color: #374151;">Current SPAs</div>
        <div style="font-size: 0.88rem; color: #111827; margin-top: 0.35rem;">${spaCountUnique}</div>
        <div style="font-size: 0.8rem; color: #666; margin-top: 0.25rem; line-height: 1.4;">
          ${countSemantics}
        </div>
        ${snapshotDate
          ? `<div style="font-size: 0.75rem; color: #888; margin-top: 0.2rem;">Active as of ${snapshotDate}</div>`
          : ''}
        ${spaListMarkup}
      </div>
    `;

    // Add click handlers to SPA IDs
    const clickableSpaIds = profileContainer.querySelectorAll('.clickable-spa-id');
    clickableSpaIds.forEach(element => {
      element.addEventListener('click', async (e) => {
        const spaId = e.target.getAttribute('data-spa-id');
        await showSpaDetailsModal(spaId);
      });
    });
  }

  function displaySavingsAnalysis(customer) {
    const savingsContainer = document.getElementById("savings-analysis-content");
    const savingsCard = document.getElementById("savings-analysis-card");

    if (!customer.savings || !customer.savings.total_savings) {
      // No savings data available
      savingsCard.style.display = "none";
      return;
    }

    savingsCard.style.display = "block";

    const savings = customer.savings;
    const totalCogs = customer.total_cogs || 0;
    const pricingSources = savings.current_pricing_sources || {};
    const pricingSourceNote = savings.pricing_source_note || '';
    const pricingSourceText = Object.keys(pricingSources).length
      ? Object.entries(pricingSources).map(([source, count]) => `${source}: ${count}`).join(', ')
      : '';
    const materialCountText = Number.isFinite(savings.material_count) ? savings.material_count : '—';
    const totalMaterialCountText = Number.isFinite(savings.total_material_count) ? savings.total_material_count : '—';

    // Calculate coverage metrics
    const coveragePercent = totalCogs > 0 ? (savings.total_spa_cost / totalCogs) * 100 : 0;
    const notCoveredPercent = 100 - coveragePercent;
    const notCoveredAmount = totalCogs - savings.total_spa_cost;

    // Calculate percentages
    const savingsPercentOfTotal = totalCogs > 0
      ? (savings.total_savings / totalCogs * 100).toFixed(2)
      : 0;

    // Check if savings are negative
    const isNegative = savings.total_savings < 0;
    const warningBanner = isNegative ? `
      <div style="background: #ffe6e6; border-left: 4px solid #d32f2f; padding: 1rem; margin-bottom: 1.5rem; border-radius: 4px;">
        <div style="display: flex; align-items: start; gap: 0.75rem;">
          <span style="font-size: 1.5em; color: #d32f2f;">⚠️</span>
          <div>
            <strong style="color: #d32f2f; font-size: 1.1em;">Current SPAs are NOT Profitable</strong>
            <p style="margin: 0.5rem 0 0 0; line-height: 1.6; color: #666;">
              This customer's SPA pricing ($${Math.abs(savings.total_spa_cost).toLocaleString(undefined, {minimumFractionDigits: 2})})
              is <strong>HIGHER</strong> than standard base cost pricing ($${Math.abs(savings.total_base_cost).toLocaleString(undefined, {minimumFractionDigits: 2})}).
              The customer is <strong>losing $${Math.abs(savings.total_savings).toLocaleString(undefined, {minimumFractionDigits: 2})}</strong> by using these SPAs.
            </p>
            <p style="margin: 0.5rem 0 0 0; line-height: 1.6; color: #666;">
              <strong>Action:</strong> Review and consider removing these unprofitable SPA agreements.
              The customer would save money by purchasing at standard base cost pricing instead.
            </p>
          </div>
        </div>
      </div>
    ` : '';

    savingsContainer.innerHTML = `
      <div class="savings-analysis">
        ${warningBanner}

        <div class="savings-intro">
          <p style="font-size: 1.1em; margin-bottom: 1rem; color: ${isNegative ? '#d32f2f' : '#0070f2'};">
            <strong>📊 Current ${isNegative ? 'Loss' : 'Savings'} Analysis</strong>
          </p>
          <p style="line-height: 1.6;">
            This customer currently has <strong>${customer.spa_count} SPA(s)</strong>.
            ${isNegative ?
              `Analysis shows these SPAs are <strong style="color: #d32f2f;">costing MORE</strong> than standard pricing for <strong>${materialCountText}</strong> covered materials.` :
              `This current-state view shows <strong>${materialCountText}</strong> purchased materials currently receiving assigned SPA pricing out of <strong>${totalMaterialCountText}</strong> total materials in the Q4 snapshot.`
            }
          </p>
          ${pricingSourceNote ? `
            <p style="line-height: 1.5; font-size: 0.88rem; color: #5f6b7a; margin-top: 0.5rem;">
              <strong>Pricing source:</strong> ${pricingSourceText ? `${pricingSourceText}. ` : ''}${pricingSourceNote}
            </p>
          ` : ''}
        </div>

        <div class="savings-metrics" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
          <div class="metric-box" style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #0070f2;">
            <div style="font-size: 0.85em; color: #666; text-transform: uppercase; margin-bottom: 0.5rem;">Total Spending (COGS)</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #0070f2;">$${totalCogs.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
            <div style="font-size: 0.8em; color: #999; margin-top: 0.25rem;">Cost of Goods Sold</div>
          </div>

          <div class="metric-box" style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
            <div style="font-size: 0.85em; color: #666; text-transform: uppercase; margin-bottom: 0.5rem;">Current SPA Spending</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #f59e0b;">$${savings.total_spa_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
            <div style="font-size: 0.8em; color: #999; margin-top: 0.25rem;">Already using SPA pricing</div>
          </div>

          <div class="metric-box" style="background: ${isNegative ? '#ffe6e6' : '#d1f4e0'}; padding: 1rem; border-radius: 8px; border-left: 4px solid ${isNegative ? '#d32f2f' : '#28a745'};">
            <div style="font-size: 0.85em; color: #666; text-transform: uppercase; margin-bottom: 0.5rem;">Current ${isNegative ? 'Loss' : 'Savings'}</div>
            <div style="font-size: 1.5em; font-weight: bold; color: ${isNegative ? '#d32f2f' : '#28a745'};">${isNegative ? '-' : ''}$${Math.abs(savings.total_savings).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
            <div style="font-size: 0.8em; color: #999; margin-top: 0.25rem;">${Math.abs(savingsPercentOfTotal)}% of total COGS ${isNegative ? '(LOSS)' : ''}</div>
          </div>

          <div class="metric-box" style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #6c757d;">
            <div style="font-size: 0.85em; color: #666; text-transform: uppercase; margin-bottom: 0.5rem;">Materials Covered by SPA</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #6c757d;">${materialCountText}</div>
            <div style="font-size: 0.8em; color: #999; margin-top: 0.25rem;">of ${totalMaterialCountText} purchased materials in Q4</div>
          </div>
        </div>

        <!-- SPA Coverage Section -->
        <div class="coverage-section" style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem; border: 2px solid #e9ecef;">
          <h4 style="margin: 0 0 1rem 0; color: #495057;">📊 SPA Coverage Analysis</h4>

          <!-- Coverage Bar -->
          <div style="margin-bottom: 1rem;">
            <div style="display: flex; height: 40px; border-radius: 6px; overflow: hidden; border: 1px solid #dee2e6;">
              <div style="width: ${coveragePercent}%; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9em; transition: width 0.3s ease;">
                ${coveragePercent > 15 ? `${coveragePercent.toFixed(1)}% Covered` : ''}
              </div>
              <div style="width: ${notCoveredPercent}%; background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9em; transition: width 0.3s ease;">
                ${notCoveredPercent > 15 ? `${notCoveredPercent.toFixed(1)}% Opportunity` : ''}
              </div>
            </div>
          </div>

          <!-- Coverage Details -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div style="background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #28a745;">
              <div style="font-size: 0.85em; color: #666; margin-bottom: 0.25rem;">✅ Covered by SPA</div>
              <div style="font-size: 1.3em; font-weight: bold; color: #28a745;">$${savings.total_spa_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
              <div style="font-size: 0.8em; color: #999; margin-top: 0.25rem;">${coveragePercent.toFixed(1)}% of total COGS</div>
            </div>
            <div style="background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #ff9800;">
              <div style="font-size: 0.85em; color: #666; margin-bottom: 0.25rem;">⚠️ Not Covered</div>
              <div style="font-size: 1.3em; font-weight: bold; color: #ff9800;">$${notCoveredAmount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
              <div style="font-size: 0.8em; color: #999; margin-top: 0.25rem;">${notCoveredPercent.toFixed(1)}% - Expansion opportunity</div>
            </div>
          </div>

          <div style="padding: 1rem; background: white; border-radius: 6px; border: 1px solid #e9ecef;">
            <p style="margin: 0; font-size: 0.9em; color: #666; line-height: 1.6;">
              <strong>What this means:</strong> ${coveragePercent.toFixed(0)}% of this customer's spending is already optimized with SPA pricing.
              ${notCoveredPercent > 20 ? `The remaining ${notCoveredPercent.toFixed(0)}% represents significant expansion potential - these materials could benefit from SPA agreements.` :
                'Most spending is already covered by SPAs.'}
            </p>
          </div>
        </div>

        <div class="savings-explanation" style="background: ${isNegative ? '#fff9f9' : '#f0f7ff'}; padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem;">
          <h4 style="margin: 0 0 1rem 0; color: ${isNegative ? '#d32f2f' : '#0070f2'};">💡 How ${isNegative ? 'Loss' : 'Savings'} ${isNegative ? 'is' : 'are'} Calculated</h4>
          <div style="line-height: 1.8; color: #333;">
            <p style="margin-bottom: 1rem;">
              <strong>1. Total COGS (Q4 Last 12 Months):</strong> This customer spent <strong>$${totalCogs.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong> total.
            </p>
            <p style="margin-bottom: 1rem;">
              <strong>2. Materials with SPA Pricing (${materialCountText} materials):</strong>
            </p>
            <div style="margin-left: 2rem; margin-bottom: 1rem;">
              <p style="margin-bottom: 0.5rem;">• <strong>Without SPA</strong> (baseline cost): $${savings.total_base_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
              <p style="margin-bottom: 0.5rem;">• <strong>With SPA</strong> (actual paid): $${savings.total_spa_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
              <p style="margin-bottom: 0.5rem; color: ${isNegative ? '#d32f2f' : '#28a745'}; font-weight: 600;">• <strong>${isNegative ? 'Loss' : 'Savings'}</strong>: ${isNegative ? '-' : ''}$${Math.abs(savings.total_savings).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} (${Math.abs(savings.savings_percent).toFixed(1)}%${savings.savings_percent > 60 ? '*' : ''} ${isNegative ? 'increase' : 'discount'})</p>
              ${savings.savings_percent > 60 ? `<p style="margin-top: 0.5rem; font-size: 0.85em; color: #666; font-style: italic;">*Very high savings rate. Conservative estimate (40%) used in opportunity calculations.</p>` : ''}
            </div>
            <p style="margin-bottom: 1rem;">
              <strong>3. Breakdown by Coverage:</strong>
            </p>
            <div style="margin-left: 2rem; margin-bottom: 1rem; background: #f8f9fa; padding: 1rem; border-radius: 4px;">
              <p style="margin-bottom: 0.5rem;">📊 <strong>Total COGS:</strong> $${totalCogs.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
              <p style="margin-bottom: 0.5rem; padding-left: 1.5rem;">├─ <strong>Covered by SPA:</strong> $${savings.total_spa_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} (${coveragePercent.toFixed(1)}%)</p>
              <p style="margin-bottom: 0.5rem; padding-left: 1.5rem;">└─ <strong>Not Covered:</strong> $${notCoveredAmount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} (${notCoveredPercent.toFixed(1)}%)</p>
              <p style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #dee2e6; font-weight: 600; color: ${isNegative ? '#d32f2f' : '#28a745'};">
                💰 <strong>Actual ${isNegative ? 'Loss' : 'Savings'} on Covered Materials:</strong> ${isNegative ? '-' : ''}$${Math.abs(savings.total_savings).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
              </p>
              <p style="margin-top: 0.5rem; font-size: 0.9em; color: #666;">
                ${isNegative ?
                  `⚠️ These ${materialCountText} covered materials cost ${Math.abs(savings.savings_percent).toFixed(1)}% MORE with current SPAs.` :
                  `✅ These ${materialCountText} covered materials receive ${savings.savings_percent.toFixed(1)}%${savings.savings_percent > 60 ? '*' : ''} discount through SPAs.`
                }
              </p>
            </div>
          </div>
        </div>

        <div class="savings-note" style="margin-top: 1rem; padding: 1rem; background: ${isNegative ? '#fff9f9' : '#fff'}; border-left: 3px solid ${isNegative ? '#d32f2f' : '#17a2b8'}; color: #666; font-size: 0.9em;">
          <strong>${isNegative ? '⚠️ Warning' : 'ℹ️ Note'}:</strong> ${isNegative ?
            'These SPAs are more expensive than standard base cost pricing. Consider removing these agreements to reduce costs.' :
            'This analysis compares baseline pricing to SPA pricing for materials that have SPA agreements available. The "Missing SPAs" section below shows additional SPAs that similar customers have, which could unlock further savings opportunities.'
          }
        </div>

        <!-- AI-Powered Strategic Insights -->
        <div id="llm-insights-container" style="margin-top: 1.5rem; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white; font-size: 1.1em;">
              🤖 AI-Powered Strategic Analysis
            </h4>
            <div style="font-size: 0.75em; color: rgba(255,255,255,0.8); background: rgba(255,255,255,0.2); padding: 0.25rem 0.75rem; border-radius: 12px;">
              GPT-4.1 through SAP BTP AI Core
            </div>
          </div>

          <div id="llm-insights-content" style="background: white; padding: 1.5rem; border-radius: 6px; min-height: 100px;">
            <div style="text-align: center; color: #999;">
              <div style="font-size: 1.5em; margin-bottom: 0.5rem;">⏳</div>
              <div>Loading strategic insights...</div>
            </div>
          </div>
        </div>

        <!-- Detailed Calculation Breakdown -->
        <details style="margin-top: 1.5rem;">
          <summary style="cursor: pointer; padding: 1rem; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; font-weight: 600; color: #495057;">
            📋 Show Detailed Calculation Breakdown (Formulas, Components, Assumptions)
          </summary>
          <div style="margin-top: 0.5rem; padding: 1.5rem; background: #fafbfc; border: 1px solid #e1e4e8; border-radius: 6px;">

            <!-- Formula Section -->
            <div style="margin-bottom: 1.5rem;">
              <div style="font-size: 0.95em; font-weight: 600; color: #586069; margin-bottom: 0.75rem;">📐 Core Formulas:</div>
              <div style="background: white; padding: 1rem; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 0.85em; line-height: 1.8;">
                <div style="margin-bottom: 0.5rem;"><strong>Coverage %</strong> = (COGS Covered / Total COGS) × 100</div>
                <div style="margin-bottom: 0.5rem; padding-left: 2rem; color: #666;">= ($${savings.total_spa_cost.toLocaleString()} / $${totalCogs.toLocaleString()}) × 100</div>
                <div style="margin-bottom: 0.5rem; padding-left: 2rem; color: #28a745; font-weight: 600;">= ${coveragePercent.toFixed(1)}%</div>
                <div style="margin-top: 1rem; margin-bottom: 0.5rem;"><strong>Savings Amount</strong> = Base Cost - SPA Cost</div>
                <div style="margin-bottom: 0.5rem; padding-left: 2rem; color: #666;">= $${savings.total_base_cost.toLocaleString()} - $${savings.total_spa_cost.toLocaleString()}</div>
                <div style="margin-bottom: 0.5rem; padding-left: 2rem; color: #28a745; font-weight: 600;">= $${savings.total_savings.toLocaleString()}</div>
                <div style="margin-top: 1rem; margin-bottom: 0.5rem;"><strong>Savings %</strong> = (Savings Amount / Base Cost) × 100</div>
                <div style="margin-bottom: 0.5rem; padding-left: 2rem; color: #666;">= ($${savings.total_savings.toLocaleString()} / $${savings.total_base_cost.toLocaleString()}) × 100</div>
                <div style="margin-bottom: 0.5rem; padding-left: 2rem; color: #28a745; font-weight: 600;">= ${savings.savings_percent.toFixed(1)}%${savings.savings_percent > 60 ? '*' : ''}</div>
              </div>
            </div>

            <!-- Component Definitions -->
            <div style="margin-bottom: 1.5rem;">
              <div style="font-size: 0.95em; font-weight: 600; color: #586069; margin-bottom: 0.75rem;">📊 Component Definitions:</div>
              <div style="background: white; padding: 1rem; border-radius: 6px; font-size: 0.9em; line-height: 1.8;">
                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #e1e4e8;">
                  <div style="font-weight: 600; color: #0366d6; margin-bottom: 0.25rem;">Total COGS (Cost of Goods Sold)</div>
                  <div style="color: #586069; padding-left: 1rem;">• Amount: <strong>$${totalCogs.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong></div>
                  <div style="color: #586069; padding-left: 1rem;">• Definition: Total amount spent on all materials in Q4 (last 12 months)</div>
                  <div style="color: #586069; padding-left: 1rem;">• Source: Actual transaction data from quarterly_sales_raw.parquet</div>
                  <div style="color: #586069; padding-left: 1rem;">• Coverage: 100% of spending (all materials)</div>
                </div>

                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #e1e4e8;">
                  <div style="font-weight: 600; color: #28a745; margin-bottom: 0.25rem;">COGS Covered by SPA</div>
                  <div style="color: #586069; padding-left: 1rem;">• Amount: <strong>$${savings.total_spa_cost.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong> (${coveragePercent.toFixed(1)}%)</div>
                  <div style="color: #586069; padding-left: 1rem;">• Definition: Actual amount paid for materials with SPA pricing</div>
                  <div style="color: #586069; padding-left: 1rem;">• Materials: ${materialCountText} purchased materials are currently covered by assigned SPAs</div>
                  <div style="color: #586069; padding-left: 1rem;">• Interpretation: ${coveragePercent > 80 ? 'Excellent' : coveragePercent > 60 ? 'Good' : coveragePercent > 40 ? 'Moderate' : 'Low'} SPA adoption</div>
                </div>

                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #e1e4e8;">
                  <div style="font-weight: 600; color: #ff9800; margin-bottom: 0.25rem;">COGS Not Covered</div>
                  <div style="color: #586069; padding-left: 1rem;">• Amount: <strong>$${notCoveredAmount.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong> (${notCoveredPercent.toFixed(1)}%)</div>
                  <div style="color: #586069; padding-left: 1rem;">• Definition: Spending on materials without SPA pricing</div>
                  <div style="color: #586069; padding-left: 1rem;">• Calculation: Total COGS - COGS Covered = $${totalCogs.toLocaleString()} - $${savings.total_spa_cost.toLocaleString()}</div>
                  <div style="color: #586069; padding-left: 1rem;">• Opportunity: ${notCoveredPercent > 30 ? 'High' : notCoveredPercent > 15 ? 'Medium' : 'Low'} potential for SPA expansion</div>
                </div>

                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #e1e4e8;">
                  <div style="font-weight: 600; color: #6f42c1; margin-bottom: 0.25rem;">Base Cost (Baseline Pricing)</div>
                  <div style="color: #586069; padding-left: 1rem;">• Amount: <strong>$${savings.total_base_cost.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong></div>
                  <div style="color: #586069; padding-left: 1rem;">• Definition: What these ${materialCountText} covered materials would cost WITHOUT SPA agreements</div>
                  <div style="color: #586069; padding-left: 1rem;">• Source: Standard pricing from BASE_COST_Z001 table</div>
                  <div style="color: #586069; padding-left: 1rem; ${savings.total_base_cost > totalCogs * 1.5 ? 'color: #d32f2f; font-weight: 600;' : ''}">• Note: ${savings.total_base_cost > totalCogs * 1.5 ? '⚠️ Base cost significantly exceeds total COGS - may include pricing from different periods' : 'Calculated from standard rate tables'}</div>
                </div>

                <div>
                  <div style="font-weight: 600; color: #28a745; margin-bottom: 0.25rem;">Actual Savings</div>
                  <div style="color: #586069; padding-left: 1rem;">• Amount: <strong>$${savings.total_savings.toLocaleString(undefined, {minimumFractionDigits: 2})}</strong></div>
                  <div style="color: #586069; padding-left: 1rem;">• Discount Rate: <strong>${savings.savings_percent.toFixed(1)}%${savings.savings_percent > 60 ? '*' : ''}</strong></div>
                  <div style="color: #586069; padding-left: 1rem;">• Definition: Difference between baseline pricing and actual SPA pricing</div>
                  <div style="color: #586069; padding-left: 1rem;">• Calculation: Base Cost - SPA Cost = $${savings.total_base_cost.toLocaleString()} - $${savings.total_spa_cost.toLocaleString()}</div>
                </div>
              </div>
            </div>

            <!-- Assumptions & Limitations -->
            <div>
              <div style="font-size: 0.95em; font-weight: 600; color: #586069; margin-bottom: 0.75rem;">⚠️ Assumptions & Limitations:</div>
              <div style="background: #fff3cd; padding: 1rem; border-radius: 6px; border-left: 4px solid #ffc107;">
                <ul style="margin: 0; padding-left: 1.5rem; line-height: 1.8; font-size: 0.9em;">
                  ${savings.savings_percent > 60 ? `
                    <li style="margin-bottom: 0.5rem;"><strong>* High Savings Rate (${savings.savings_percent.toFixed(1)}%):</strong> Discount rate exceeds 60%. For conservative projections, we use a normalized rate of 40% in opportunity calculations. Actual savings shown are real.</li>
                  ` : ''}
                  <li style="margin-bottom: 0.5rem;"><strong>Coverage represents actual adoption:</strong> ${coveragePercent.toFixed(0)}% of spending uses SPA pricing. This is the key metric for SPA program maturity.</li>
                  <li style="margin-bottom: 0.5rem;"><strong>Base Cost reference only:</strong> Base cost provides baseline for comparison but may include pricing from different time periods. Trust COGS for actual spending analysis.</li>
                  <li style="margin-bottom: 0.5rem;"><strong>Savings calculated on covered materials:</strong> The ${savings.savings_percent.toFixed(0)}% savings rate applies only to the ${coveragePercent.toFixed(0)}% of spending that uses SPAs, not total COGS.</li>
                  <li><strong>Potential expansion opportunity:</strong> The ${notCoveredPercent.toFixed(0)}% uncovered spending ($${notCoveredAmount.toLocaleString()}) represents materials that could potentially benefit from SPA agreements.</li>
                </ul>
              </div>
            </div>
          </div>
        </details>

        <div class="savings-abbreviations" style="margin-top: 1rem; padding: 1rem; background: #fafafa; border-radius: 6px; font-size: 0.85em; color: #666;">
          <strong>Abbreviations:</strong>
          <ul style="margin: 0.5rem 0 0 1.5rem; line-height: 1.8;">
            <li><strong>COGS</strong> = Cost of Goods Sold (total amount spent on materials)</li>
            <li><strong>SPA</strong> = Special Price Agreement (negotiated pricing contract with suppliers)</li>
            <li><strong>Q4 12M</strong> = Fourth quarter, 12-month rolling period (last 12 months of data)</li>
            <li><strong>Coverage %</strong> = Percentage of spending that uses SPA pricing</li>
          </ul>
        </div>

        <div style="margin-top: 1.5rem; text-align: center;">
          <ui5-button id="show-material-breakdown-btn" design="Emphasized">
            Show Material Details
          </ui5-button>
        </div>
      </div>
    `;

    // Add event listener for material breakdown button
    setTimeout(() => {
      const breakdownBtn = document.getElementById('show-material-breakdown-btn');
      console.log('[DEBUG] Material breakdown button found:', !!breakdownBtn);
      if (breakdownBtn) {
        console.log('[DEBUG] Adding click listener to material breakdown button');
        breakdownBtn.addEventListener('click', () => {
          console.log('[DEBUG] Material breakdown button clicked, customer:', customer.customer_id);
          showMaterialBreakdownModal(customer.customer_id);
        });
      } else {
        console.error('[ERROR] Material breakdown button NOT found in DOM');
      }
    }, 100);
  }

  /**
   * Display dynamic Example Calculation based on current customer data
   */
  function displayExampleCalculation(customer) {
    const container = document.getElementById('example-calculation-container');
    const pctExplanation = document.getElementById('savings-pct-explanation');

    if (!customer.savings || !customer.savings.total_savings) {
      // No savings data - hide example
      if (container) container.style.display = 'none';
      return;
    }

    const savings = customer.savings;
    const totalCogs = customer.total_cogs || 0;

    // Calculate metrics
    const discountRateOnSPA = savings.total_base_cost > 0
      ? (savings.total_savings / savings.total_base_cost * 100).toFixed(1)
      : 0;

    const savingsPercentOfCOGS = totalCogs > 0
      ? (savings.total_savings / totalCogs * 100).toFixed(1)
      : 0;

    const totalMaterialsWithoutSPA = 0; // We don't have this data directly
    const cogsWithoutSPA = totalCogs - savings.total_base_cost;

    if (container) {
      container.innerHTML = `
        <h5 style="margin: 1rem 0 0.5rem 0; font-size: 0.95em;">
          Example Calculation (Customer ${customer.customer_id}):
        </h5>
        <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 4px; font-family: monospace; font-size: 0.9em; margin: 0.5rem 0; line-height: 1.8;">
          Total COGS: $${totalCogs.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}<br>
          ├─ Materials WITH active SPAs: ${savings.material_count} materials<br>
          │  ├─ Base Cost (standard pricing): $${savings.total_base_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}<br>
          │  ├─ SPA Cost (discounted pricing): $${savings.total_spa_cost.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}<br>
          │  └─ <strong>Current Savings: $${savings.total_savings.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong><br>
          │     ├─ Discount rate on SPA materials: ${discountRateOnSPA}%<br>
          │     └─ Savings as % of total COGS: <strong>${savingsPercentOfCOGS}%</strong> ← shown in Summary View<br>
          └─ Materials WITHOUT SPAs: ($${cogsWithoutSPA.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})})<br>
             └─ Potential opportunity (not included in savings)
        </div>
      `;
      container.style.display = 'block';
    }

    // Update percentage explanation
    if (pctExplanation) {
      pctExplanation.innerHTML = `
        • <strong>${discountRateOnSPA}%</strong> = Average discount on SPA materials ($${(savings.total_savings / 1000).toFixed(0)}K / $${(savings.total_base_cost / 1000).toFixed(0)}K base cost)<br>
        • <strong>${savingsPercentOfCOGS}%</strong> = Savings as % of total spending ($${(savings.total_savings / 1000).toFixed(0)}K / $${(totalCogs / 1000).toFixed(0)}K total COGS) ← <u>shown in tables</u>
      `;
    }
  }

  function displayMissingSPAs(missingSPAs) {
    const listContainer = document.getElementById("missing-spas-list");
    const badge = document.getElementById("missing-spa-count");

    // Update badge
    badge.textContent = missingSPAs.length;

    if (missingSPAs.length === 0) {
      listContainer.innerHTML = `
        <ui5-text>
          <b>No missing SPAs found!</b> This customer has all recommended SPAs.
        </ui5-text>
      `;
      return;
    }

    const eligibilityStyle = (status) => {
      const styles = {
        addable_candidate: 'background: #e5f5ec; color: #107e3e; border: 1px solid #9dd7b5;',
        review_required: 'background: #fff4ce; color: #8a5a00; border: 1px solid #f0c36a;',
        reference_only: 'background: #e8f0fe; color: #2454a6; border: 1px solid #b7caf7;',
        out_of_area: 'background: #fde7e9; color: #a10000; border: 1px solid #f4b6bd;',
        unknown_eligibility: 'background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db;'
      };
      return styles[status] || styles.unknown_eligibility;
    };

    const formatList = (items) => {
      if (!Array.isArray(items) || items.length === 0) {
        return 'N/A';
      }
      return items.slice(0, 4).join(', ') + (items.length > 4 ? ` +${items.length - 4}` : '');
    };

    // Display each missing SPA
    listContainer.innerHTML = missingSPAs.map((spa, index) => `
      <div class="spa-item" id="spa-item-${index}">
        <div class="spa-item-header">
          <div class="spa-deal-id">${spa.sales_deal}</div>
          <div class="confidence-badge confidence-${spa.confidence_level.toLowerCase()}">
            ${getConfidenceVisual(spa.confidence_score)} ${spa.confidence_level} (${spa.confidence_score.toFixed(0)})
          </div>
        </div>
        <div class="spa-description">${spa.description || "No description available"}</div>
        ${spa.external_description && spa.external_description !== spa.description ? `
          <div style="font-size: 0.78rem; color: #5f6b7a; margin-top: 0.15rem;">
            External description: ${spa.external_description}
          </div>
        ` : ''}
        <div style="margin: 0.35rem 0 0.25rem 0; display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;">
          <span style="font-size: 0.78rem; font-weight: 700; padding: 0.18rem 0.45rem; border-radius: 999px; ${eligibilityStyle(spa.eligibility_status)}">
            ${spa.eligibility_label || 'Eligibility unknown'}
          </span>
          ${spa.customer_count !== null && spa.customer_count !== undefined ? `<span style="font-size: 0.78rem; color: #5f6b7a;">Customer count: ${spa.customer_count}</span>` : ''}
          ${spa.opportunity_scope ? `<span style="font-size: 0.78rem; color: #5f6b7a;">Scope: ${spa.opportunity_scope.replaceAll('_', ' ')}</span>` : ''}
          ${spa.a700_vendor_scope === true ? `<span style="font-size: 0.78rem; color: #107e3e;">A700 vendor scope</span>` : ''}
          ${spa.is_supplyforce ? `<span style="font-size: 0.78rem; color: #a10000;">Supplyforce excluded</span>` : ''}
          ${spa.geo_relevance ? `<span style="font-size: 0.78rem; color: #5f6b7a;">Geo: ${spa.geo_relevance.replaceAll('_', ' ')}</span>` : ''}
          ${spa.customer_area ? `<span style="font-size: 0.78rem; color: #5f6b7a;">Customer area: ${spa.customer_area}</span>` : ''}
          ${spa.candidate_areas?.length ? `<span style="font-size: 0.78rem; color: #5f6b7a;">SPA area: ${formatList(spa.candidate_areas)}</span>` : ''}
          ${spa.candidate_sales_offices?.length ? `<span style="font-size: 0.78rem; color: #5f6b7a;">SO: ${formatList(spa.candidate_sales_offices)}</span>` : ''}
          ${spa.candidate_plants?.length ? `<span style="font-size: 0.78rem; color: #5f6b7a;">Plant: ${formatList(spa.candidate_plants)}</span>` : ''}
          ${spa.valid_from || spa.valid_to ? `<span style="font-size: 0.78rem; color: #5f6b7a;">Valid: ${formatDateValue(spa.valid_from)} to ${formatDateValue(spa.valid_to)}</span>` : ''}
        </div>
        ${spa.eligibility_reason ? `
          <div style="font-size: 0.8rem; color: #5f6b7a; margin-bottom: 0.45rem;">
            ${spa.eligibility_reason}
          </div>
        ` : ''}
        <div class="spa-stats">
          <span>${
            spa.count_in_similar > 0 || spa.percentage_in_similar > 0
              ? `📊 <b>${spa.count_in_similar}/${spa.percentage_in_similar}%</b> of similar customers have this`
              : `📊 <b>Pricing opportunity model</b> recommendation without direct similar-customer support`
          }</span>
          <span>🏢 Vendor: ${spa.vendor || "N/A"}</span>
          <span>🏷️ Type: ${getSPATypeLabel(spa.grouping)}</span>
        </div>

        <!-- Why Recommended Section -->
        <div class="why-recommended-section">
          <button class="why-recommended-toggle" onclick="toggleWhyRecommended('${spa.sales_deal}', '${index}')">
            ▶ Why Recommended?
          </button>
          <div class="why-recommended-content" id="why-content-${index}" style="display: none;">
            <div class="loading-placeholder">Loading details...</div>
          </div>
        </div>
      </div>
    `).join("");
  }

  async function toggleWhyRecommended(spaId, index) {
    const contentDiv = document.getElementById(`why-content-${index}`);
    const button = document.querySelector(`#spa-item-${index} .why-recommended-toggle`);

    if (contentDiv.style.display === 'none') {
      // Expand - load data
      contentDiv.style.display = 'block';
      button.textContent = '▼ Why Recommended?';

      // Fetch breakdown data
      try {
        const customerId = document.getElementById("customer-id-input").value;
        const breakdown = await request(`/api/spa/${spaId}/recommendation-breakdown?customer_id=${customerId}`, 'GET');

        // Display breakdown
        contentDiv.innerHTML = `
          <div class="recommendation-factors">
            ${breakdown.potential_value > 0 ? `
              <div class="factor-item" style="background: #d1f4e0; padding: 0.75rem; border-radius: 4px; margin-bottom: 0.75rem;">
                <strong style="color: #28a745;">💰 Potential Savings: $${breakdown.potential_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong>
                <ul style="margin: 0.5rem 0 0 1.5rem; color: #666; font-size: 0.9em;">
                  <li>→ Based on ${breakdown.materials_count_for_customer} materials you purchase</li>
                  <li>→ Covering $${breakdown.cogs_covered.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} in COGS</li>
                </ul>
              </div>
            ` : ''}

            <div class="factor-item">
              <strong>✓ ${breakdown.similarity_factors.customers_with_spa} of ${breakdown.similarity_factors.total_similar_customers} similar customers have this</strong>
              <ul style="margin-left: 1.5rem; color: #666; font-size: 0.9em;">
                <li>→ Same Sales Office (${breakdown.customer_sales_office}): ${breakdown.similarity_factors.same_sales_office} customers</li>
                <li>→ Same Customer Type (${breakdown.customer_pl_type}): ${breakdown.similarity_factors.same_customer_type} customers</li>
              </ul>
            </div>

            <div class="factor-item">
              <strong>✓ ${breakdown.spa_type}</strong>
              <ul style="margin-left: 1.5rem; color: #666; font-size: 0.9em;">
                <li>→ ${breakdown.spa_type === 'Blanket SPA' ? 'Easy to add, typically no approval needed' : 'Customer-specific pricing agreement'}</li>
              </ul>
            </div>

            ${breakdown.material_coverage.total_materials_in_spa > 0 ? `
              <div class="factor-item material-coverage-demo">
                <strong>⚠️ Material Coverage: Demo Data</strong>
                <ul style="margin-left: 1.5rem; color: #666; font-size: 0.9em;">
                  <li>→ This SPA covers ${breakdown.material_coverage.total_materials_in_spa} materials (from supplier catalog)</li>
                  <li>→ Customer-specific overlap calculation pending</li>
                  <li>→ <button class="link-button" onclick="showMaterialsModal('${spaId}')">View Sample Materials</button></li>
                </ul>
                <div style="font-size: 0.85em; color: #999; margin-top: 0.5rem;">
                  Note: Actual coverage requires material mapping table (A703 SKU → Transaction Material Number)
                </div>
              </div>
            ` : ''}
          </div>
        `;
      } catch (error) {
        contentDiv.innerHTML = `<div style="color: #bb0000;">Error loading details: ${error.message}</div>`;
      }
    } else {
      // Collapse
      contentDiv.style.display = 'none';
      button.textContent = '▶ Why Recommended?';
    }
  }

  async function showMaterialsModal(spaId) {
    try {
      const response = await request(`/api/spa/${spaId}/materials?limit=20`, 'GET');

      const modal = document.createElement('dialog');
      modal.id = 'materials-modal';
      modal.innerHTML = `
        <div class="modal-header">
          <h3>SPA ${spaId} - Covered Materials (Sample)</h3>
          <button class="close-button" onclick="this.closest('dialog').close()">✕</button>
        </div>
        <div class="modal-body">
          <p style="color: #f0ab00; font-size: 0.9em; padding: 0.5rem; background: #fff9e6; border-radius: 0.25rem;">
            ⚠️ ${response.disclaimer}
          </p>

          <table class="materials-table">
            <thead>
              <tr>
                <th>Material ID</th>
                <th>Description</th>
                <th>Unit Price</th>
                <th>UoM</th>
              </tr>
            </thead>
            <tbody>
              ${response.materials.map(m => `
                <tr>
                  <td>${m.material_id}</td>
                  <td>${m.description}</td>
                  <td>${m.unit_price ? '$' + m.unit_price.toFixed(2) : 'N/A'}</td>
                  <td>${m.uom || 'N/A'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>

          <p style="font-size: 0.85em; color: #666; margin-top: 1rem;">
            Showing ${response.showing_count} of ${response.total_count} materials in this SPA
          </p>
        </div>
        <div class="modal-footer">
          <button onclick="this.closest('dialog').close()">Close</button>
        </div>
      `;

      document.body.appendChild(modal);
      modal.showModal();

      // Remove modal when closed
      modal.addEventListener('close', () => {
        modal.remove();
      });

    } catch (error) {
      alert(`Error loading materials: ${error.message}`);
    }
  }

  function getConfidenceVisual(score) {
    // Convert 0-100 score to 0-5 filled circles
    const filled = Math.ceil(score / 20);
    const empty = 5 - filled;
    return '●'.repeat(filled) + '○'.repeat(empty);
  }

  function getSPATypeLabel(grouping) {
    if (!grouping) return "N/A";

    // Extract simplified type from grouping
    // "B - SPA/Rebate: Multi-customer Blanket" → "Blanket SPA"
    // "D - SPA/Rebate: Customer Specific" → "Customer Specific"
    if (grouping.includes("Blanket")) {
      return "Blanket SPA";
    } else if (grouping.includes("Customer Specific")) {
      return "Customer Specific";
    } else {
      return grouping; // Fallback to full grouping text
    }
  }

  function displaySimilarCustomers(similarCustomers) {
    const listContainer = document.getElementById("similar-customers-list");
    const hasNamesCheckbox = document.getElementById("filter-has-names");

    // Filter customers by checkbox if checked
    let filteredCustomers = similarCustomers;
    if (hasNamesCheckbox && hasNamesCheckbox.checked) {
      filteredCustomers = similarCustomers.filter(c => c.customer_name && c.customer_name !== "Unknown");
    }

    if (filteredCustomers.length === 0) {
      const message = hasNamesCheckbox && hasNamesCheckbox.checked
        ? `<ui5-text>No similar customers with names found. Uncheck the filter to see all similar customers.</ui5-text>`
        : `<ui5-text>No similar customers found</ui5-text>`;
      listContainer.innerHTML = message;
      return;
    }

    listContainer.innerHTML = filteredCustomers.map(customer => {
      // Build customer details only if data exists
      const detailsParts = [];

      // Location - only show if city or state exists
      const city = customer.city && customer.city !== 'N/A' ? customer.city : '';
      const state = customer.state && customer.state !== 'N/A' ? customer.state : '';
      if (city || state) {
        const location = [city, state].filter(Boolean).join(', ');
        detailsParts.push(`<span>📍 ${location}</span>`);
      }

      // Sales office - only show if exists
      if (customer.sales_office && customer.sales_office !== 'N/A') {
        detailsParts.push(`<span>🏢 ${customer.sales_office}</span>`);
      }

      // RFM segment - only show if exists
      if (customer.rfm_segment && customer.rfm_segment !== 'N/A') {
        detailsParts.push(`<span title="${getRFMExplanation(customer.rfm_segment)}">🎯 ${customer.rfm_segment}</span>`);
      }

      const detailsHtml = detailsParts.length > 0
        ? `<div class="customer-details">${detailsParts.join('\n          ')}</div>`
        : '';

      return `
      <div class="similar-customer-item">
        <div class="similar-customer-header">
          <div>
            <div class="customer-name">
              <span class="clickable-customer-id" data-customer-id="${customer.customer_id}" style="cursor: pointer; color: #0070f2; text-decoration: underline;">${customer.customer_id}</span>
              - ${customer.customer_name || "Unknown"}
            </div>
          </div>
          <div class="similarity-score">${customer.similarity_score.toFixed(1)}</div>
        </div>
        ${detailsHtml}
      </div>
    `;
    }).join("");

    // Add click handlers to customer IDs
    const clickableIds = listContainer.querySelectorAll('.clickable-customer-id');
    clickableIds.forEach(element => {
      element.addEventListener('click', (e) => {
        const customerId = e.target.getAttribute('data-customer-id');
        loadCustomer(customerId);
      });
    });
  }

  function loadCustomer(customerId) {
    // Update input field
    if (customerIdInput) {
      customerIdInput.value = customerId;
      // Trigger analysis
      analyzeButton.click();
      // Scroll to top
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  async function showSpaDetailsModal(spaId) {
    try {
      // Get SPA details from API
      const details = await getSpaDetails(spaId);

      // Get material list for this SPA (filtered to current customer)
      let materialsHtml = '';
      try {
        const currentCustomerId = customerIdInput.value;
        if (currentCustomerId) {
          const materialsData = await getSpaMaterials(spaId, currentCustomerId);

          if (materialsData.materials && materialsData.materials.length > 0) {
            materialsHtml = `
              <div style="margin-top: 1.5rem;">
                <strong>Materials Covered (Customer ${currentCustomerId}):</strong>
                <div style="max-height: 300px; overflow-y: auto; margin-top: 0.5rem;">
                  <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
                    <thead style="position: sticky; top: 0; background: #f5f5f5;">
                      <tr style="border-bottom: 2px solid #ddd;">
                        <th style="padding: 0.5rem; text-align: left;">Material</th>
                        <th style="padding: 0.5rem; text-align: right;">Base Cost</th>
                        <th style="padding: 0.5rem; text-align: right;">SPA Price</th>
                        <th style="padding: 0.5rem; text-align: right;">Savings %</th>
                        ${materialsData.materials[0].customer_cogs ? '<th style="padding: 0.5rem; text-align: right;">Your COGS</th>' : ''}
                        ${materialsData.materials[0].potential_savings ? '<th style="padding: 0.5rem; text-align: right;">Potential</th>' : ''}
                      </tr>
                    </thead>
                    <tbody>
                      ${materialsData.materials.slice(0, 20).map(mat => `
                        <tr style="border-bottom: 1px solid #eee;">
                          <td style="padding: 0.5rem; font-family: monospace;">${mat.material}</td>
                          <td style="padding: 0.5rem; text-align: right;">${formatCurrency(mat.base_cost)}</td>
                          <td style="padding: 0.5rem; text-align: right;">${formatCurrency(mat.spa_price)}</td>
                          <td style="padding: 0.5rem; text-align: right; color: #2e7d32; font-weight: bold;">${mat.savings_percent.toFixed(1)}%</td>
                          ${mat.customer_cogs ? `<td style="padding: 0.5rem; text-align: right;">${formatCurrency(mat.customer_cogs)}</td>` : ''}
                          ${mat.potential_savings ? `<td style="padding: 0.5rem; text-align: right; font-weight: bold;">${formatCurrency(mat.potential_savings)}</td>` : ''}
                        </tr>
                      `).join('')}
                    </tbody>
                  </table>
                  ${materialsData.materials.length > 20 ? `<div style="padding: 0.5rem; text-align: center; color: #666; font-size: 0.875rem;">Showing top 20 of ${materialsData.total_materials} materials</div>` : ''}
                </div>
              </div>
            `;
          }
        }
      } catch (err) {
        console.warn('Could not load materials for SPA:', err);
      }

      // Format dates
      const validFrom = details.valid_from ? new Date(details.valid_from).toLocaleDateString() : 'N/A';
      const validTo = details.valid_to ? new Date(details.valid_to).toLocaleDateString() : 'N/A';

      // Create or get modal dialog (native HTML dialog)
      let dialog = document.getElementById('spa-details-dialog');
      if (!dialog) {
        dialog = document.createElement('dialog');
        dialog.id = 'spa-details-dialog';
        dialog.style.cssText = 'padding: 2rem; border: 1px solid #ccc; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); min-width: 500px; max-width: 90%;';
        document.body.appendChild(dialog);
      }

      // Update content
      dialog.innerHTML = `
        <h3 style="margin-top: 0; color: #0070f2;">SPA Details</h3>
        <div style="margin-bottom: 1rem;">
          <strong>SPA ID:</strong> ${details.spa_id}
        </div>
        ${details.description ? `
        <div style="margin-bottom: 1rem;">
          <strong>Description:</strong> ${details.description}
        </div>
        ` : ''}
        ${details.vendor ? `
        <div style="margin-bottom: 1rem;">
          <strong>Vendor:</strong> ${details.vendor}
        </div>
        ` : ''}
        ${details.grouping ? `
        <div style="margin-bottom: 1rem;">
          <strong>Type:</strong> ${details.grouping}
        </div>
        ` : ''}
        <div style="margin-bottom: 1rem;">
          <strong>Valid From:</strong> ${validFrom}
        </div>
        <div style="margin-bottom: 1rem;">
          <strong>Valid To:</strong> ${validTo}
        </div>
        <div style="margin-bottom: 1rem;">
          <strong>Customer Count:</strong> ${details.customer_count}
        </div>
        <div style="margin-bottom: 1.5rem;">
          <strong>Customers (first 10):</strong>
          <div style="margin-top: 0.5rem; font-size: 0.9em; color: #666;">
            ${details.customers.join(', ')}
          </div>
        </div>
        ${materialsHtml}
        <button id="close-spa-dialog" style="margin-top: 1rem; padding: 0.5rem 2rem; background: #0070f2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem;">Close</button>
      `;

      // Add close handler
      const closeBtn = dialog.querySelector('#close-spa-dialog');
      closeBtn.addEventListener('click', () => dialog.close());

      // Close on backdrop click
      dialog.addEventListener('click', (e) => {
        if (e.target === dialog) {
          dialog.close();
        }
      });

      // Show dialog
      dialog.showModal();

    } catch (error) {
      console.error('Failed to load SPA details:', error);
      showError(`Failed to load SPA details: ${error.message}`);
    }
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorPanel.style.display = "block";
    errorPanel.collapsed = false;
    resultsContainer.style.display = "none";
  }

  // ============================================================
  // MATERIAL HIERARCHY FUNCTIONS
  // ============================================================

  async function fetchMaterialSummary(customerId) {
    return await request(
      `/api/spa/customer/${customerId}/material-summary`,
      "POST",
      { level: 1 }
    );
  }

  async function fetchMaterialDrilldown(customerId, level, parentCategory) {
    return await request(
      `/api/spa/customer/${customerId}/material-drilldown`,
      "POST",
      { level, parent_category: parentCategory }
    );
  }

  function displayMaterialSummary(data, totalCogs) {
    // Calculate coverage percentage
    const coveragePercent = totalCogs > 0 ? (data.total_cogs / totalCogs * 100) : 0;

    // Update summary stats
    document.getElementById('material-total-cogs').textContent =
      '$' + (data.total_cogs / 1000).toFixed(1) + 'K';

    // Update coverage explanation with percentage
    const coveragePercentText = coveragePercent.toFixed(1) + '%';
    document.getElementById('material-coverage-percent').textContent =
      `${coveragePercentText} of $${(totalCogs / 1000).toFixed(1)}K total COGS`;

    // Update coverage explanation
    const explanationText = coveragePercent < 50
      ? `This hierarchy-mapped subset represents ${coveragePercentText} of customer total COGS. The remaining ${(100 - coveragePercent).toFixed(1)}% is outside this partial hierarchy view. This card is directional and is not directly comparable to the canonical Current SPA Coverage above.`
      : `This hierarchy-mapped subset represents ${coveragePercentText} of customer total COGS. This card is directional and is not directly comparable to the canonical Current SPA Coverage above.`;
    document.getElementById('material-coverage-explanation').textContent = explanationText;

    document.getElementById('material-spa-coverage').textContent =
      data.overall_spa_coverage_percentage.toFixed(0) + '%';
    document.getElementById('material-spa-coverage').title =
      'Percentage of hierarchy-mapped spend whose materials appear in legacy A703 material pricing. This is not the canonical assigned-SPA coverage metric.';
    document.getElementById('material-transaction-count').textContent =
      data.total_transactions.toLocaleString();
    document.getElementById('material-category-count').textContent =
      data.categories.length;

    // Render category tree
    const treeContainer = document.getElementById('material-hierarchy-tree');
    treeContainer.innerHTML = data.categories.map((category, idx) =>
      renderCategoryNode(category, idx)
    ).join('');

    // Show card
    document.getElementById('material-summary-card').style.display = 'block';
  }

  function renderCategoryNode(category, index) {
    const coverageBarWidth = category.spa_coverage_percentage;
    const coverageColor = getCoverageColor(category.spa_coverage_percentage);

    return `
      <div class="material-category-item" id="category-${index}">
        <div class="category-header">
          ${category.has_children
            ? `<span class="category-toggle" onclick="toggleCategoryDrilldown('${category.category_code}', ${index})">▶</span>`
            : '<span class="category-no-toggle">•</span>'}
          <strong>${category.category_name}</strong>
          <span class="category-cogs">$${(category.total_cogs / 1000).toFixed(1)}K (${category.percentage_of_total.toFixed(1)}%)</span>
        </div>

        <div class="category-details">
          <span>📦 ${category.unique_materials} materials</span>
          <span>📊 ${category.transaction_count} transactions</span>
        </div>

        <!-- Coverage bar -->
        <div class="coverage-bar-container">
          <div class="coverage-bar" style="width: ${coverageBarWidth}%; background-color: ${coverageColor}"></div>
          <span class="coverage-label">${category.spa_coverage_percentage.toFixed(0)}% A703 Match Coverage</span>
        </div>

        ${category.spas_covering.length > 0
          ? `<div class="spas-covering">Matched by A703 agreements: ${category.spas_covering.join(', ')}</div>`
          : '<div class="spas-covering">No A703 match coverage</div>'}

        <!-- Drill-down container (hidden until expanded) -->
        <div class="category-children" id="category-children-${index}" style="display: none; padding-left: 1.5rem;">
          <div class="loading-placeholder">Loading...</div>
        </div>
      </div>
    `;
  }

  async function toggleCategoryDrilldown(categoryCode, index) {
    const childrenDiv = document.getElementById(`category-children-${index}`);
    const toggle = document.querySelector(`#category-${index} .category-toggle`);

    if (childrenDiv.style.display === 'none') {
      // Expand - fetch children
      toggle.textContent = '▼';
      childrenDiv.style.display = 'block';

      const customerId = customerIdInput.value;
      const nextLevel = categoryCode.length === 2 ? 2 : categoryCode.length === 4 ? 3 : 4;

      try {
        const children = await fetchMaterialDrilldown(customerId, nextLevel, categoryCode);
        childrenDiv.innerHTML = children.categories.map((child, idx) =>
          renderCategoryNode(child, `${index}-${idx}`)
        ).join('');
      } catch (error) {
        childrenDiv.innerHTML = '<div style="color: red;">Error loading details</div>';
      }
    } else {
      // Collapse
      toggle.textContent = '▶';
      childrenDiv.style.display = 'none';
    }
  }

  function getCoverageColor(percentage) {
    if (percentage >= 80) return '#27AE60';  // Green
    if (percentage >= 50) return '#F39C12';  // Yellow/Orange
    return '#E74C3C';  // Red
  }

  // Expose functions to global scope for onclick handlers
  window.toggleWhyRecommended = toggleWhyRecommended;
  window.showMaterialsModal = showMaterialsModal;
  window.showSpaDetailsModal = showSpaDetailsModal;  // Fixed: was showSPADetails
  window.toggleCategoryDrilldown = toggleCategoryDrilldown;  // NEW

  // Expose function to manually trigger navigation check (for cross-page navigation)
  window.triggerQuickLookupCheck = () => {
    console.log('[Quick Lookup] Manual trigger from another page');
    checkNavigationCustomerId();
  };
}

/**
 * Show Material Breakdown Modal with detailed potential analysis
 */
async function showMaterialBreakdownModal(customerId) {
  console.log('[DEBUG] showMaterialBreakdownModal called with customer:', customerId);
  try {
    console.log('[DEBUG] Fetching breakdown data...');
    const breakdown = await getPotentialBreakdown(customerId);
    console.log('[DEBUG] Breakdown data received:', breakdown);

    const dialog = document.createElement('ui5-dialog');
    dialog.setAttribute('header-text', `Material-Level Breakdown - ${breakdown.customer_name}`);
    dialog.style.width = '90%';
    dialog.style.maxWidth = '1200px';

    const cov = breakdown.coverage;
    const totalPotentialFormatted = formatCurrency(breakdown.total_potential);
    const potentialPercent = breakdown.potential_percent.toFixed(2);
    const fullPotentialFormatted = formatCurrency(breakdown.full_potential || breakdown.total_potential);
    const estimateScope = breakdown.estimate_scope || {};
    const topEstimateMaterials = breakdown.top_estimate_materials || breakdown.top_materials || [];
    const topCategories = breakdown.top_categories || [];

    dialog.innerHTML = `
      <div style="padding: 1rem;">
        <!-- Summary Stats -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
          <div style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">
            <div style="font-size: 0.875rem; color: #666;">Total COGS</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${formatCurrency(breakdown.total_cogs)}</div>
          </div>
          <div style="background: #e8f5e9; padding: 1rem; border-radius: 4px;">
            <div style="font-size: 0.875rem; color: #666;">Estimated Potential*</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #2e7d32;">${totalPotentialFormatted}*</div>
            <div style="font-size: 0.875rem; color: #666;">${potentialPercent}% of COGS</div>
          </div>
          <div style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">
            <div style="font-size: 0.875rem; color: #666;">Missing SPAs</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${breakdown.missing_spas_count}</div>
          </div>
          <div style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">
            <div style="font-size: 0.875rem; color: #666;">Materials Coverage</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${cov.materials_with_spa_pricing_pct.toFixed(1)}%</div>
            <div style="font-size: 0.875rem; color: #666;">${cov.materials_with_spa_pricing} / ${cov.total_materials}</div>
          </div>
        </div>

        <div style="margin-bottom: 2rem; padding: 1rem; background: #fff8e1; border-left: 4px solid #f59e0b; border-radius: 4px; font-size: 0.9rem;">
          <strong>* Rough estimate:</strong> ${estimateScope.note || 'Based on top 10 opportunity materials by spend.'}
          <div style="margin-top: 0.5rem; color: #666;">
            Scope: ${estimateScope.materials_considered || 0} materials,
            ${formatCurrency(estimateScope.cogs_considered || 0)} spend basis,
            full bundle potential across recommended SPAs ${fullPotentialFormatted}.
          </div>
          <div style="margin-top: 0.5rem; color: #666;">
            Pricing source: A703 rows are treated as exact/netted/rebated cost. A704 multiplier-based opportunity is Phase 2 and is not included in exact POC monetary opportunity.
          </div>
        </div>

        <!-- Coverage Details -->
        <ui5-title level="H5" style="margin-bottom: 0.5rem;">Coverage Statistics</ui5-title>
        <div style="background: #fafafa; padding: 1rem; border-radius: 4px; margin-bottom: 2rem;">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.875rem;">
            <div>Materials in Missing SPAs:</div>
            <div style="text-align: right; font-weight: bold;">${cov.materials_in_missing_spas} materials</div>

            <div>COGS with SPA Pricing:</div>
            <div style="text-align: right; font-weight: bold;">${formatCurrency(cov.cogs_with_spa_pricing)} (${cov.cogs_with_spa_pricing_pct.toFixed(1)}%)</div>

            <div>COGS in Missing SPAs:</div>
            <div style="text-align: right; font-weight: bold;">${formatCurrency(cov.cogs_in_missing_spas)} (${cov.cogs_in_missing_spas_pct.toFixed(1)}%)</div>
          </div>
        </div>

        <!-- Top Materials -->
        <ui5-title level="H5" style="margin-bottom: 0.5rem;">Top 10 Opportunity Materials by Spend*</ui5-title>
        <div style="overflow-x: auto; margin-bottom: 2rem;">
          <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
            <thead>
              <tr style="background: #f5f5f5; border-bottom: 2px solid #ddd;">
                <th style="padding: 0.5rem; text-align: left;">#</th>
                <th style="padding: 0.5rem; text-align: left;">Material</th>
                <th style="padding: 0.5rem; text-align: left;">Description / Group</th>
                <th style="padding: 0.5rem; text-align: right;">Customer COGS</th>
                <th style="padding: 0.5rem; text-align: right;">Current Price</th>
                <th style="padding: 0.5rem; text-align: right;">SPA Price</th>
                <th style="padding: 0.5rem; text-align: left;">Pricing Source</th>
                <th style="padding: 0.5rem; text-align: left;">Best SPA</th>
                <th style="padding: 0.5rem; text-align: right;">Savings %</th>
                <th style="padding: 0.5rem; text-align: right;">Potential Savings</th>
              </tr>
            </thead>
            <tbody>
              ${topEstimateMaterials.slice(0, 10).map((mat, idx) => {
                // Format material name/description - prefer category_name > description > group
                let materialName = mat.category_name || mat.material_description || mat.material_group || 'N/A';
                if (materialName.length > 50) {
                  materialName = materialName.substring(0, 47) + '...';
                }

                return `
                <tr style="border-bottom: 1px solid #eee;">
                  <td style="padding: 0.5rem;">${idx + 1}</td>
                  <td style="padding: 0.5rem; font-family: monospace; color: #666;">${mat.material}</td>
                  <td style="padding: 0.5rem; font-size: 0.85em;" title="${mat.material_description || mat.material_group || ''}">${materialName}</td>
                  <td style="padding: 0.5rem; text-align: right;">${formatCurrency(mat.cogs_12m)}</td>
                  <td style="padding: 0.5rem; text-align: right; color: #666;">${formatCurrency(mat.base_cost || 0)}</td>
                  <td style="padding: 0.5rem; text-align: right; color: #2e7d32; font-weight: 600;">${formatCurrency(mat.spa_price || 0)}</td>
                  <td style="padding: 0.5rem; font-size: 0.8rem;" title="${mat.rebated_cost_status || ''}">${mat.pricing_source || 'N/A'}</td>
                  <td style="padding: 0.5rem; font-family: monospace;">${mat.sales_deal}</td>
                  <td style="padding: 0.5rem; text-align: right; color: #2e7d32; font-weight: bold;">${mat.savings_percent.toFixed(1)}%</td>
                  <td style="padding: 0.5rem; text-align: right; font-weight: bold;">${formatCurrency(mat.potential_savings)}*</td>
                </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>

        <ui5-title level="H5" style="margin-bottom: 0.5rem;">Top Categories in Estimate*</ui5-title>
        <div style="overflow-x: auto; margin-bottom: 2rem;">
          <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
            <thead>
              <tr style="background: #f5f5f5; border-bottom: 2px solid #ddd;">
                <th style="padding: 0.5rem; text-align: left;">#</th>
                <th style="padding: 0.5rem; text-align: left;">Category</th>
                <th style="padding: 0.5rem; text-align: right;">Materials</th>
                <th style="padding: 0.5rem; text-align: right;">Customer COGS</th>
                <th style="padding: 0.5rem; text-align: right;">Estimated Potential</th>
              </tr>
            </thead>
            <tbody>
              ${topCategories.slice(0, 10).map((cat, idx) => `
                <tr style="border-bottom: 1px solid #eee;">
                  <td style="padding: 0.5rem;">${idx + 1}</td>
                  <td style="padding: 0.5rem;">${cat.category_name || cat.material_group || 'N/A'}</td>
                  <td style="padding: 0.5rem; text-align: right;">${cat.materials_count}</td>
                  <td style="padding: 0.5rem; text-align: right;">${formatCurrency(cat.cogs_12m)}</td>
                  <td style="padding: 0.5rem; text-align: right; font-weight: bold;">${formatCurrency(cat.potential_savings)}*</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <!-- SPA Breakdown -->
        <ui5-title level="H5" style="margin-bottom: 0.5rem;">Potential by Missing SPA</ui5-title>
        <div style="overflow-x: auto;">
          <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
            <thead>
              <tr style="background: #f5f5f5; border-bottom: 2px solid #ddd;">
                <th style="padding: 0.5rem; text-align: left;">#</th>
                <th style="padding: 0.5rem; text-align: left;">SPA ID</th>
                <th style="padding: 0.5rem; text-align: right;">Materials</th>
                <th style="padding: 0.5rem; text-align: right;">COGS Covered</th>
                <th style="padding: 0.5rem; text-align: right;">Avg Savings %</th>
                <th style="padding: 0.5rem; text-align: right;">Potential Savings</th>
              </tr>
            </thead>
            <tbody>
              ${breakdown.spa_breakdown.slice(0, 10).map((spa, idx) => `
                <tr style="border-bottom: 1px solid #eee;">
                  <td style="padding: 0.5rem;">${idx + 1}</td>
                  <td style="padding: 0.5rem; font-family: monospace;">${spa.spa_id}</td>
                  <td style="padding: 0.5rem; text-align: right;">${spa.materials_count}</td>
                  <td style="padding: 0.5rem; text-align: right;">${formatCurrency(spa.cogs_covered)}</td>
                  <td style="padding: 0.5rem; text-align: right; color: #2e7d32;">${spa.avg_savings_pct.toFixed(1)}%</td>
                  <td style="padding: 0.5rem; text-align: right; font-weight: bold;">${formatCurrency(spa.potential_savings)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <div style="margin-top: 1.5rem; padding: 1rem; background: #e3f2fd; border-radius: 4px; font-size: 0.875rem;">
          <strong>Note:</strong> Values marked with * are rough estimates for UI guidance.
          They are based on top spend opportunity materials, prioritize uncovered materials, and may be capped at 50% of customer COGS.
          The non-starred bundle list below sums to the full bundle potential of ${fullPotentialFormatted} across the recommended SPAs.
        </div>
      </div>

      <div slot="footer" style="display: flex; justify-content: flex-end; padding: 0.5rem;">
        <ui5-button id="close-breakdown-dialog" design="Emphasized">Close</ui5-button>
      </div>
    `;

    document.body.appendChild(dialog);

    // Wait for custom element to be defined and show dialog
    setTimeout(() => {
      try {
        console.log('[DEBUG] Opening dialog, element:', dialog);
        console.log('[DEBUG] Dialog type:', dialog.constructor.name);
        console.log('[DEBUG] Dialog.show available:', typeof dialog.show);

        // Try UI5 Dialog API first
        if (typeof dialog.show === 'function') {
          dialog.show();
        } else if (typeof dialog.open !== 'undefined') {
          // Try setting open property
          dialog.open = true;
        } else {
          console.error('[ERROR] No method to open dialog found');
          alert('Cannot open dialog - UI5 Dialog API not available');
        }
      } catch (err) {
        console.error('[ERROR] Failed to open dialog:', err);
        alert(`Failed to open dialog: ${err.message}`);
      }
    }, 50);

    // Close button handler
    setTimeout(() => {
      const closeBtn = dialog.querySelector('#close-breakdown-dialog');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          console.log('[DEBUG] Close button clicked');
          // Try close() or open=false
          if (typeof dialog.close === 'function') {
            dialog.close();
          } else {
            dialog.open = false;
          }
          setTimeout(() => dialog.remove(), 300);
        });
      }
    }, 100);

  } catch (error) {
    console.error('[ERROR] Error showing material breakdown:', error);
    console.error('[ERROR] Error details:', error.message, error.stack);
    alert(`Failed to load material breakdown: ${error.message || 'Unknown error'}`);
  }
}


// ============================================================================
// AI INSIGHTS - LLM-powered strategic analysis
// ============================================================================

let currentCustomerIdForInsights = null;
let insightsCache = {}; // Cache insights per customer

async function loadAIInsights(customerId) {
  console.log(`Loading AI insights for customer ${customerId}`);

  const contentDiv = document.getElementById('llm-insights-content');

  if (!contentDiv) {
    console.error('LLM insights content div not found');
    return;
  }

  // Check cache first
  if (insightsCache[customerId]) {
    console.log('Using cached insights');
    displayInsights(insightsCache[customerId]);
    return;
  }

  // Show loading state
  contentDiv.innerHTML = `
    <div style="text-align: center; padding: 2rem;">
      <div style="font-size: 2em; margin-bottom: 1rem; animation: spin 2s linear infinite;">🤖</div>
      <div style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">Analyzing customer data with GPT-4.1...</div>
      <div style="color: #999; font-size: 0.9em;">through SAP BTP AI Core</div>
    </div>
    <style>
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    </style>
  `;

  try {
    const response = await request(
      `/api/spa/customer-insights?customer_id=${customerId}`,
      'POST',
      {}
    );

    console.log('AI Insights response:', response);

    // Cache the result
    insightsCache[customerId] = response;

    // Display insights
    displayInsights(response);

  } catch (error) {
    console.error('Error loading AI insights:', error);
    contentDiv.innerHTML = `
      <div style="text-align: center; padding: 2rem;">
        <div style="font-size: 2em; margin-bottom: 1rem; color: #d32f2f;">⚠️</div>
        <div style="color: #d32f2f; font-weight: 600; margin-bottom: 0.5rem;">Failed to generate insights</div>
        <div style="color: #666; font-size: 0.9em;">${error.message || 'Unknown error'}</div>
      </div>
    `;
  }
}

function displayInsights(response) {
  const contentDiv = document.getElementById('llm-insights-content');

  if (!response || !response.insight) {
    contentDiv.innerHTML = '<div style="text-align: center; color: #999;">No insights available</div>';
    return;
  }

  // Parse markdown to HTML (simple conversion)
  let insightHtml = response.insight
    .replace(/## (.+)/g, '<h4 style="color: #667eea; margin: 1.5rem 0 0.75rem 0; font-size: 1.1em;">$1</h4>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p style="margin: 0.75rem 0; line-height: 1.6; color: #333;">')
    .replace(/^- (.+)/gm, '<li style="margin: 0.5rem 0;">$1</li>')
    .replace(/(<li[^>]*>.*<\/li>)/s, '<ul style="margin: 0.5rem 0 1rem 1.5rem; line-height: 1.8;">$1</ul>');

  // Wrap in paragraph if not already
  if (!insightHtml.startsWith('<h4') && !insightHtml.startsWith('<p')) {
    insightHtml = `<p style="margin: 0.75rem 0; line-height: 1.6; color: #333;">${insightHtml}</p>`;
  }

  // Confidence badge
  const confidenceBadge = response.confidence === 'high'
    ? '<span style="background: #d1f4e0; color: #28a745; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85em; font-weight: 600;">High Confidence</span>'
    : response.confidence === 'medium'
    ? '<span style="background: #fff3cd; color: #f59e0b; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Medium Confidence</span>'
    : '<span style="background: #ffe6e6; color: #d32f2f; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85em; font-weight: 600;">Low Confidence</span>';

  contentDiv.innerHTML = `
    <div style="margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
      <div style="font-weight: 600; color: #667eea;">Strategic Analysis</div>
      ${confidenceBadge}
    </div>
    <div style="color: #333; line-height: 1.8;">
      ${insightHtml}
    </div>
    ${response.data_summary ? `
      <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e9ecef; font-size: 0.85em; color: #999;">
        <strong>Analysis based on:</strong>
        $${response.data_summary.total_cogs?.toLocaleString() || 0} COGS,
        ${response.data_summary.coverage_percent?.toFixed(1) || 0}% SPA coverage,
        ${response.data_summary.top_categories || 0} top categories
      </div>
    ` : ''}
  `;
}

// Initialize insights button handler when savings analysis is displayed
function initializeInsightsAutoLoad(customerId) {
  currentCustomerIdForInsights = customerId;

  // Automatically load insights when customer changes
  console.log(`Auto-loading insights for customer ${customerId}`);
  loadAIInsights(customerId);
}

// Export for global access
window.initializeInsightsButton = initializeInsightsAutoLoad;
