/* SPA Summary View page - UI5 components */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Popover.js";
import "@ui5/webcomponents/dist/CheckBox.js";

import { request } from "../../services/api.js";
import { pageRouter } from "../../modules/router.js";

// State
let currentData = null;
let currentFilters = {
  rfm_segment: null,
  niels_id: null,
  customer_group: null,
  sales_office: null,
  min_cogs: null,
  has_names: true,  // Default: only show customers with names
  search_customer_id: null  // NEW: Customer ID search
};
let currentSort = {
  sort_by: "total_savings",
  sort_order: "desc"
};

// API function
async function getCustomerSummary(filters, sort_by, sort_order, limit = 100) {
  // Convert has_names filter to exclude_unknown for new API
  const exclude_unknown = filters.has_names === true;

  // Remove has_names from filters (it's a separate parameter now)
  const apiFilters = { ...filters };
  delete apiFilters.has_names;

  return await request(
    "/api/customer/summary",  // NEW endpoint with core_market, customer_group
    "POST",
    {
      filters: apiFilters,
      sort_by: sort_by,
      sort_order: sort_order,
      limit: limit,
      exclude_unknown: exclude_unknown  // NEW parameter
    }
  );
}

export default function initSpaSummaryViewPage() {
  console.log("SPA Summary View page initialized");

  // Get UI elements
  const loadButton = document.getElementById("load-summary-button");
  const refreshButton = document.getElementById("refresh-summary-button");
  const exportButton = document.getElementById("export-summary-button");
  const loadingIndicator = document.getElementById("summary-loading");
  const resultsContainer = document.getElementById("summary-results");
  const errorPanel = document.getElementById("summary-error");
  const errorMessage = document.getElementById("summary-error-message");

  // Filters
  const rfmFilter = document.getElementById("filter-rfm-segment");
  const nielsFilter = document.getElementById("filter-niels-id");
  const custGroupInput = document.getElementById("filter-customer-group");
  const salesOfficeInput = document.getElementById("filter-sales-office");
  const minCogsInput = document.getElementById("filter-min-cogs");
  const hasNamesCheckbox = document.getElementById("filter-has-names");
  const searchCustomerIdInput = document.getElementById("search-customer-id");  // NEW

  // Sort controls
  const sortBySelect = document.getElementById("sort-by-select");
  const sortOrderButton = document.getElementById("sort-order-button");

  // Auto-load on page init
  setTimeout(() => loadButton.click(), 500);

  // Load summary button
  if (loadButton) {
    loadButton.addEventListener("click", async () => {
      await loadSummary();
    });
  }

  // Refresh button (clears filters)
  if (refreshButton) {
    refreshButton.addEventListener("click", () => {
      // Reset filters
      if (rfmFilter) rfmFilter.value = "";
      if (nielsFilter) nielsFilter.value = "";
      if (custGroupInput) custGroupInput.value = "";
      if (salesOfficeInput) salesOfficeInput.value = "";
      if (minCogsInput) minCogsInput.value = "";
      if (hasNamesCheckbox) hasNamesCheckbox.checked = true;  // Default: show only with names
      if (searchCustomerIdInput) searchCustomerIdInput.value = "";  // NEW

      currentFilters = {
        rfm_segment: null,
        niels_id: null,
        customer_group: null,
        sales_office: null,
        min_cogs: null,
        has_names: true,  // Reset to default
        search_customer_id: null  // NEW
      };

      loadButton.click();
    });
  }

  // Export button
  if (exportButton) {
    exportButton.addEventListener("click", () => {
      exportToCSV();
    });
  }

  // Filter change handlers
  if (rfmFilter) {
    rfmFilter.addEventListener("change", (e) => {
      currentFilters.rfm_segment = e.target.value || null;
    });
  }

  if (nielsFilter) {
    nielsFilter.addEventListener("change", (e) => {
      currentFilters.niels_id = e.target.value || null;
    });
  }

  if (custGroupInput) {
    custGroupInput.addEventListener("input", (e) => {
      currentFilters.customer_group = e.target.value || null;
    });
  }

  if (salesOfficeInput) {
    salesOfficeInput.addEventListener("input", (e) => {
      currentFilters.sales_office = e.target.value || null;
    });
  }

  if (minCogsInput) {
    minCogsInput.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      currentFilters.min_cogs = isNaN(val) ? null : val;
    });
  }

  if (hasNamesCheckbox) {
    hasNamesCheckbox.addEventListener("change", (e) => {
      currentFilters.has_names = e.target.checked;
    });
  }

  // NEW: Search Customer ID handler
  if (searchCustomerIdInput) {
    searchCustomerIdInput.addEventListener("input", (e) => {
      currentFilters.search_customer_id = e.target.value.trim() || null;
    });
  }

  // Sort controls
  if (sortBySelect) {
    sortBySelect.addEventListener("change", (e) => {
      currentSort.sort_by = e.target.value;
      if (currentData) {
        displayResults(currentData); // Re-render with new sort
      }
    });
  }

  if (sortOrderButton) {
    sortOrderButton.addEventListener("click", () => {
      currentSort.sort_order = currentSort.sort_order === "desc" ? "asc" : "desc";
      sortOrderButton.textContent = currentSort.sort_order === "desc" ? "↓ Desc" : "↑ Asc";
      loadButton.click(); // Reload with new order
    });
  }

  // Clickable table headers for sorting
  document.addEventListener("click", (e) => {
    const th = e.target.closest("th[data-sort]");
    if (th) {
      const sortField = th.getAttribute("data-sort");

      // Toggle sort order if clicking same field
      if (currentSort.sort_by === sortField) {
        currentSort.sort_order = currentSort.sort_order === "desc" ? "asc" : "desc";
      } else {
        currentSort.sort_by = sortField;
        currentSort.sort_order = "desc";
      }

      // Update UI
      sortBySelect.value = sortField;
      sortOrderButton.textContent = currentSort.sort_order === "desc" ? "↓ Desc" : "↑ Asc";

      // Update sort indicators
      document.querySelectorAll(".sort-indicator").forEach(ind => ind.textContent = "");
      th.querySelector(".sort-indicator").textContent = currentSort.sort_order === "desc" ? " ↓" : " ↑";

      // Reload
      loadButton.click();
    }
  });

  async function loadSummary() {
    loadingIndicator.style.display = "block";
    resultsContainer.style.display = "none";
    errorPanel.style.display = "none";
    loadButton.disabled = true;

    try {
      const data = await getCustomerSummary(
        currentFilters,
        currentSort.sort_by,
        currentSort.sort_order,
        500 // Limit to 500 customers
      );

      currentData = data;
      displayResults(data);
    } catch (error) {
      showError(error.message || "Failed to load customer summary");
    } finally {
      loadingIndicator.style.display = "none";
      loadButton.disabled = false;
    }
  }

  function displayResults(data) {
    // DEBUG: Log received data
    console.log('[Summary View] Received data:', data);
    console.log('[Summary View] First customer:', data.customers[0]);
      console.log('[Summary View] Customer fields:', Object.keys(data.customers[0]));
      console.log('[Summary View] total_savings:', data.customers[0].total_savings);
      console.log('[Summary View] savings_percent:', data.customers[0].savings_percent);

    // Update KPIs with NEW data structure
    document.getElementById("kpi-total-customers").textContent = data.total_customers.toLocaleString();
    document.getElementById("kpi-filtered-customers").textContent = data.filtered_customers.toLocaleString();

    // NEW: Show savings instead of missing SPAs
    const totalSavings = data.summary_stats.total_savings || 0;
    document.getElementById("kpi-total-missing-spas").textContent =
      "$" + (totalSavings / 1000000).toFixed(1) + "M";

    const championsCount = data.summary_stats.champions_count || 0;
    document.getElementById("kpi-high-confidence").textContent =
      championsCount.toLocaleString();

    const totalCogs = data.summary_stats.total_cogs || 0;
    document.getElementById("kpi-potential-value").textContent =
      "$" + (totalCogs / 1000000).toFixed(0) + "M";

    // Build table
    const tableBody = document.getElementById("summary-table-body");
    tableBody.innerHTML = "";

    // NEW: Apply client-side Customer ID filter
    let filteredCustomers = data.customers;
    const searchId = currentFilters.search_customer_id;

    if (searchId) {
      filteredCustomers = data.customers.filter(c =>
        c.customer_id.toString().includes(searchId)
      );

      // Update filtered customers count
      document.getElementById("kpi-filtered-customers").textContent = filteredCustomers.length.toLocaleString();

      if (filteredCustomers.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="10" style="padding: 2rem; text-align: center; color: #666;">No customers found matching ID: ' + searchId + '</td></tr>';
        resultsContainer.style.display = "block";
        return;
      }
    }

    filteredCustomers.forEach((customer) => {
      const row = document.createElement("tr");
      row.style.borderBottom = "1px solid #e0e0e0";

      // Customer ID (clickable)
      const idCell = document.createElement("td");
      idCell.style.padding = "0.75rem";
      idCell.style.border = "1px solid #d9d9d9";
      const idLink = document.createElement("a");
      idLink.href = "#";
      idLink.textContent = customer.customer_id;
      idLink.style.color = "#0854a0";
      idLink.style.fontWeight = "bold";
      idLink.style.textDecoration = "none";
      idLink.style.cursor = "pointer";
      idLink.addEventListener("mouseenter", () => {
        idLink.style.textDecoration = "underline";
      });
      idLink.addEventListener("mouseleave", () => {
        idLink.style.textDecoration = "none";
      });
      idLink.addEventListener("click", (e) => {
        e.preventDefault();
        console.log('[Summary View] Customer ID link clicked:', customer.customer_id);
        navigateToQuickLookup(customer.customer_id);
      });
      idCell.appendChild(idLink);
      row.appendChild(idCell);

      // Customer Name
      const nameCell = document.createElement("td");
      nameCell.style.padding = "0.75rem";
      nameCell.style.border = "1px solid #d9d9d9";
      nameCell.textContent = customer.customer_name || "Unknown";
      nameCell.style.fontSize = "0.9em";
      nameCell.style.maxWidth = "200px";
      nameCell.style.overflow = "hidden";
      nameCell.style.textOverflow = "ellipsis";
      nameCell.style.whiteSpace = "nowrap";
      nameCell.title = customer.customer_name || "Unknown"; // Tooltip on hover
      row.appendChild(nameCell);

      // Core Market
      const marketCell = document.createElement("td");
      marketCell.style.padding = "0.75rem";
      marketCell.style.border = "1px solid #d9d9d9";
      marketCell.textContent = customer.core_market || "Unknown";
      marketCell.style.fontSize = "0.9em";
      row.appendChild(marketCell);

      // Customer Group
      const custGroupCell = document.createElement("td");
      custGroupCell.style.padding = "0.75rem";
      custGroupCell.style.border = "1px solid #d9d9d9";
      custGroupCell.textContent = customer.customer_group || "-";
      custGroupCell.style.fontWeight = "500";
      row.appendChild(custGroupCell);

      // RFM Segment
      const rfmCell = document.createElement("td");
      rfmCell.style.padding = "0.75rem";
      rfmCell.style.border = "1px solid #d9d9d9";
      const rfmTag = document.createElement("ui5-tag");
      rfmTag.textContent = customer.rfm_segment || "N/A";
      rfmTag.setAttribute("color-scheme", getRFMColor(customer.rfm_segment));
      rfmCell.appendChild(rfmTag);
      row.appendChild(rfmCell);

      // Total COGS
      const cogsCell = document.createElement("td");
      cogsCell.style.padding = "0.75rem";
      cogsCell.style.border = "1px solid #d9d9d9";
      cogsCell.textContent = "$" + (customer.total_cogs / 1000).toFixed(1) + "K";
      cogsCell.style.textAlign = "right";
      row.appendChild(cogsCell);

      // Total Savings
      const savingsCell = document.createElement("td");
      savingsCell.style.padding = "0.75rem";
      savingsCell.style.border = "1px solid #d9d9d9";
      savingsCell.textContent = "$" + (customer.total_savings / 1000).toFixed(1) + "K";
      savingsCell.style.textAlign = "right";
      savingsCell.style.fontWeight = customer.total_savings > 50000 ? "bold" : "normal";
      savingsCell.style.color = customer.total_savings > 50000 ? "#107e3e" : "inherit";
      row.appendChild(savingsCell);

      // Savings % (calculate as percentage of total COGS)
      const savingsPctCell = document.createElement("td");
      savingsPctCell.style.padding = "0.75rem";
      savingsPctCell.style.border = "1px solid #d9d9d9";
      savingsPctCell.style.textAlign = "right";
      const pctTag = document.createElement("ui5-tag");
      const savingsPercent = customer.total_cogs > 0
        ? (customer.total_savings / customer.total_cogs * 100)
        : 0;
      pctTag.textContent = savingsPercent.toFixed(0) + "%";
      pctTag.setAttribute("color-scheme", getSavingsColor(savingsPercent));
      savingsPctCell.appendChild(pctTag);
      row.appendChild(savingsPctCell);

      // Coverage % (how much COGS is covered by SPA)
      const coverageCell = document.createElement("td");
      coverageCell.style.padding = "0.75rem";
      coverageCell.style.border = "1px solid #d9d9d9";
      coverageCell.style.textAlign = "right";
      const coverageTag = document.createElement("ui5-tag");
      const coveragePercent = customer.coverage_percent || 0;
      coverageTag.textContent = coveragePercent.toFixed(0) + "%";
      coverageTag.setAttribute("color-scheme", getCoverageColor(coveragePercent));
      coverageCell.appendChild(coverageTag);
      row.appendChild(coverageCell);

      // Potential Savings
      const potentialCell = document.createElement("td");
      potentialCell.style.padding = "0.75rem";
      potentialCell.style.border = "1px solid #d9d9d9";
      potentialCell.style.textAlign = "right";
      const potentialValue = customer.potential_value_estimate || 0;
      const potentialDisplay = "$" + (potentialValue / 1000).toFixed(1) + "K*";
      potentialCell.title = customer.potential_estimate_note || "Rough estimate based on top 10 opportunity materials by spend.";

      if (potentialValue === 0) {
        potentialCell.textContent = "—";
        potentialCell.style.color = "#9ca3af";
      } else {
        potentialCell.textContent = potentialDisplay;

        if (potentialValue > 100000) {
          potentialCell.style.fontWeight = "bold";
          potentialCell.style.color = "#10b981";
        } else if (potentialValue > 50000) {
          potentialCell.style.color = "#f59e0b";
        }
      }
      row.appendChild(potentialCell);

      // PL Type
      const plTypeCell = document.createElement("td");
      plTypeCell.style.padding = "0.75rem";
      plTypeCell.style.border = "1px solid #d9d9d9";
      plTypeCell.style.textAlign = "center";
      plTypeCell.style.fontSize = "0.85em";
      plTypeCell.textContent = customer.pl_type || "-";
      row.appendChild(plTypeCell);

      tableBody.appendChild(row);
    });

    resultsContainer.style.display = "block";
  }

  function getRFMColor(segment) {
    const colors = {
      "Champions": "8",
      "Loyal": "7",
      "Potential Loyalist": "2",
      "Promising": "2",
      "At Risk": "1",
      "Need Attention": "3",
      "N/A": "6"
    };
    return colors[segment] || "6";
  }

  function getConfidenceColor(score) {
    if (score >= 80) return "8"; // Green
    if (score >= 60) return "2"; // Yellow
    return "1"; // Red
  }

  function getSavingsColor(percent) {
    if (percent >= 70) return "8"; // Green - excellent savings
    if (percent >= 50) return "7"; // Blue - good savings
    if (percent >= 30) return "2"; // Yellow - moderate savings
    return "6"; // Grey - low savings
  }

  function getCoverageColor(percent) {
    if (percent >= 80) return "8"; // Green - excellent coverage
    if (percent >= 60) return "7"; // Blue - good coverage
    if (percent >= 40) return "2"; // Yellow - moderate coverage
    if (percent >= 20) return "3"; // Orange - low coverage
    return "1"; // Red - very low coverage
  }

  function navigateToQuickLookup(customerId) {
    console.log('[Summary View] navigateToQuickLookup called with:', customerId);

    // Store customer ID in sessionStorage
    sessionStorage.setItem('navigate_to_customer_id', customerId);
    console.log('[Summary View] sessionStorage set:', sessionStorage.getItem('navigate_to_customer_id'));

    // Always navigate to Quick Lookup - router will handle if already there
    console.log('[Summary View] Calling pageRouter.navigate(/spa-quick-lookup)');
    pageRouter.navigate('/spa-quick-lookup');
  }

  function exportToCSV() {
    if (!currentData || !currentData.customers) {
      alert("No data to export");
      return;
    }

    // CSV headers
    const headers = [
      "Customer ID",
      "Customer Name",
      "Sales Office",
      "RFM Segment",
      "PL Type",
      "Total COGS",
      "Current Savings",
      "Coverage %",
      "Estimated Potential*",
      "Full Bundle Potential",
      "Missing SPAs Count"
    ];

    // CSV rows
    const rows = currentData.customers.map(c => [
      c.customer_id,
      c.customer_name,
      c.sales_office,
      c.rfm_segment,
      c.pl_type,
      c.total_cogs,
      c.total_savings,
      c.coverage_percent,
      c.potential_value_estimate,
      c.potential_value_full_bundle || 0,
      c.missing_spas_count
    ]);

    // Build CSV string
    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(","))
    ].join("\n");

    // Download
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `customer_summary_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorPanel.style.display = "block";
    resultsContainer.style.display = "none";
  }
}
