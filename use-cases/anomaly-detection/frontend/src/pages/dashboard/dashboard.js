import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents-icons/dist/detail-view.js";
import Chart from 'chart.js/auto';
import { marked } from "marked";

import { DashboardAPI, OrdersAPI, AnomalyAPI } from "../../services/api.js";
import { pageRouter } from "../../modules/router.js";

/**
 * Format feature value - converts boolean features (starting with "Is ") to True/False
 */
function formatFeatureValue(feature, value) {
    if (feature.startsWith("Is ")) {
        const numValue = parseFloat(value);
        if (numValue === 1.0 || numValue === 1) return "True";
        if (numValue === 0.0 || numValue === 0) return "False";
    }
    return value;
}

/**
 * Convert anomaly score to human-readable probability label
 */
function getAnomalyProbabilityLabel(score) {
    if (score >= 0.05) return "Normal";
    if (score > -0.05) return "Low Probability Anomaly";
    if (score >= -0.1) return "Medium Probability Anomaly";
    return "High Probability Anomaly";
}

// Application state for the dashboard page
let state = {
    selectedDate: null,
    selectedYear: null,
    selectedMonth: null,
    calendarData: {},           // Full calendar data for heatmap
    availableYears: [],
    availableMonths: [],
    chart: null,
    currentOrders: [],          // Store current orders for client-side sorting
    sortColumn: 'col-score',
    sortDirection: 'Ascending', // 'Ascending' or 'Descending'
    selectedDoc: null,
    selectedItem: null
};

// Month names for display
const MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
];

/**
 * Populate year dropdown with available years
 */
function populateYearDropdown(years) {
    const yearSelect = document.getElementById("year-select");
    yearSelect.innerHTML = "";

    years.forEach(year => {
        const option = document.createElement("ui5-option");
        option.value = year.toString();
        option.textContent = year.toString();
        yearSelect.appendChild(option);
    });
}

/**
 * Populate month dropdown with month names
 */
function populateMonthDropdown(months) {
    const monthSelect = document.getElementById("month-select");
    monthSelect.innerHTML = "";

    months.forEach(month => {
        const option = document.createElement("ui5-option");
        option.value = month.toString();
        option.textContent = MONTH_NAMES[month - 1];
        monthSelect.appendChild(option);
    });
}

/**
 * Set dropdown selected value programmatically
 */
function setDropdownValue(selectId, value) {
    const select = document.getElementById(selectId);
    if (!select) return;

    // UI5 Select uses selectedIndex or we need to iterate options
    const options = select.querySelectorAll("ui5-option");
    options.forEach((opt, index) => {
        if (opt.value === value.toString()) {
            opt.selected = true;
        }
    });
}

/**
 * Get months available for a specific year from calendar data
 */
function getMonthsForYear(year) {
    const months = new Set();
    Object.keys(state.calendarData).forEach(dateStr => {
        const [y, m] = dateStr.split("-").map(Number);
        if (y === year) {
            months.add(m);
        }
    });
    return Array.from(months).sort((a, b) => a - b);
}

/**
 * Get color class based on anomaly rate
 */
function getHeatmapColorClass(anomalyRate, hasData) {
    if (!hasData) return "heatmap-no-data";
    if (anomalyRate === 0) return "heatmap-green";
    if (anomalyRate <= 0.1) return "heatmap-light-red";
    if (anomalyRate <= 0.2) return "heatmap-medium-red";
    return "heatmap-dark-red";
}

/**
 * Get weeks array for a given month (array of 7-day arrays)
 */
function getMonthWeeks(year, month) {
    const weeks = [];
    const firstDay = new Date(year, month - 1, 1);
    const lastDay = new Date(year, month, 0);

    // Start from Sunday of the week containing the first day
    const startDate = new Date(firstDay);
    startDate.setDate(startDate.getDate() - startDate.getDay());

    let currentDate = new Date(startDate);

    while (currentDate <= lastDay || currentDate.getDay() !== 0) {
        const week = [];
        for (let i = 0; i < 7; i++) {
            week.push(new Date(currentDate));
            currentDate.setDate(currentDate.getDate() + 1);
        }
        weeks.push(week);

        // Stop if we've passed the last day and completed the week
        if (currentDate > lastDay && currentDate.getDay() === 0) {
            break;
        }
    }

    return weeks;
}

/**
 * Format date as YYYY-MM-DD string
 */
function formatDateKey(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * Get array of days that have data in the specified month
 */
function getDaysWithDataInMonth(year, month) {
    const days = [];
    Object.keys(state.calendarData).forEach(dateStr => {
        const [y, m, d] = dateStr.split("-").map(Number);
        if (y === year && m === month) {
            days.push(d);
        }
    });
    return days.sort((a, b) => a - b);
}

/**
 * Render the heatmap calendar for the selected year and month
 */
function renderHeatmapCalendar() {
    const container = document.getElementById("heatmap-calendar");
    const headerEl = document.getElementById("calendar-header");

    if (!state.selectedYear || !state.selectedMonth) {
        container.innerHTML = '<p style="text-align: center; color: #666;">Select a year and month to view the calendar.</p>';
        return;
    }

    // Update card header with month/year
    headerEl.titleText = `Daily Activity - ${MONTH_NAMES[state.selectedMonth - 1]} ${state.selectedYear}`;

    // Get weeks for the month
    const weeks = getMonthWeeks(state.selectedYear, state.selectedMonth);

    // Build table HTML
    let tableHTML = '<table>';

    // Header row with day names
    tableHTML += '<thead><tr>';
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    dayNames.forEach(day => {
        tableHTML += `<th>${day}</th>`;
    });
    tableHTML += '</tr></thead>';

    // Body rows with day cells
    tableHTML += '<tbody>';
    weeks.forEach(week => {
        tableHTML += '<tr>';
        week.forEach(date => {
            const isCurrentMonth = date.getMonth() === state.selectedMonth - 1;

            if (!isCurrentMonth) {
                // Empty cell for days outside current month
                tableHTML += '<td class="empty-cell"></td>';
            } else {
                // Format date key to match API format
                const dateKey = formatDateKey(date);
                const dayData = state.calendarData[dateKey];

                const hasData = dayData && dayData.total_orders > 0;
                const anomalyRate = hasData ? (dayData.anomaly_count / dayData.total_orders) : 0;
                const colorClass = getHeatmapColorClass(anomalyRate, hasData);

                // Check if this is the selected day
                const isSelected = state.selectedDate &&
                    state.selectedDate.getDate() === date.getDate() &&
                    state.selectedDate.getMonth() === date.getMonth() &&
                    state.selectedDate.getFullYear() === date.getFullYear();

                const selectedClass = isSelected ? ' selected-day' : '';

                // Build cell content
                let cellContent = `<span class="day-number">${date.getDate()}</span>`;
                if (hasData) {
                    cellContent += `<span class="day-stats">${dayData.anomaly_count} (${(anomalyRate * 100).toFixed(0)}%)</span>`;
                }

                tableHTML += `<td class="${colorClass}${selectedClass}" data-date="${dateKey}">${cellContent}</td>`;
            }
        });
        tableHTML += '</tr>';
    });
    tableHTML += '</tbody></table>';

    container.innerHTML = tableHTML;

    // Add click handlers to day cells
    const dayCells = container.querySelectorAll('td[data-date]');
    dayCells.forEach(cell => {
        cell.addEventListener('click', () => {
            const dateKey = cell.getAttribute('data-date');
            const [year, month, day] = dateKey.split("-").map(Number);
            const date = new Date(year, month - 1, day);
            selectDay(date);
        });
    });
}

/**
 * Select a specific day and load its data
 */
function selectDay(date) {
    state.selectedDate = date;

    // Update day input to show selected day
    const dayInput = document.getElementById("day-input");
    if (dayInput) {
        dayInput.value = date.getDate();
    }

    // Re-render calendar to show selection
    renderHeatmapCalendar();

    // Load daily data
    loadDailyData(date);
}

/**
 * Select the most recent day with data in the current month
 */
function selectMostRecentDay() {
    const days = getDaysWithDataInMonth(state.selectedYear, state.selectedMonth);
    if (days.length > 0) {
        const lastDay = days[days.length - 1];
        const date = new Date(state.selectedYear, state.selectedMonth - 1, lastDay);
        selectDay(date);
    } else {
        clearDailyView();
    }
}

/**
 * Initialize the dashboard page
 * Sets up event handlers and loads initial data
 */
export default function initDashboard() {
    setupControls();
    setupPanelControls();
    loadInitialData();
}

/**
 * Setup event handlers for dashboard controls (filters, table, sorting)
 */
function setupControls() {
    const yearSelect = document.getElementById("year-select");
    const monthSelect = document.getElementById("month-select");
    const dayInput = document.getElementById("day-input");
    const applyDayBtn = document.getElementById("apply-day-btn");
    const ordersTable = document.getElementById("orders-table");

    // Year dropdown change handler
    yearSelect.addEventListener("change", (e) => {
        const selectedYear = parseInt(e.target.selectedOption.value);
        state.selectedYear = selectedYear;

        // Update months dropdown for new year
        const monthsInYear = getMonthsForYear(selectedYear);
        populateMonthDropdown(monthsInYear);

        // Select the last available month in the year
        if (monthsInYear.length > 0) {
            state.selectedMonth = monthsInYear[monthsInYear.length - 1];
            setDropdownValue("month-select", state.selectedMonth);
        }

        // Re-render calendar and select most recent day
        renderHeatmapCalendar();
        selectMostRecentDay();
    });

    // Month dropdown change handler
    monthSelect.addEventListener("change", (e) => {
        const selectedMonth = parseInt(e.target.selectedOption.value);
        state.selectedMonth = selectedMonth;

        // Re-render calendar and select most recent day
        renderHeatmapCalendar();
        selectMostRecentDay();
    });

    // Day input + Apply button handler
    applyDayBtn.addEventListener("click", () => {
        const dayValue = parseInt(dayInput.value);
        if (dayValue && dayValue >= 1 && dayValue <= 31) {
            const targetDate = new Date(state.selectedYear, state.selectedMonth - 1, dayValue);
            // Verify this is a valid date in the month
            if (targetDate.getMonth() === state.selectedMonth - 1) {
                selectDay(targetDate);
            }
        }
    });

    // Allow Enter key in day input
    dayInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            applyDayBtn.click();
        }
    });

    // Handle row selection in the orders table
    ordersTable.addEventListener("selection-change", (e) => {
        const selectedRow = e.detail.selectedRows[0];
        if (selectedRow) {
            const docNum = selectedRow.getAttribute("data-doc");
            const docItem = selectedRow.getAttribute("data-item");
            openOrderDetails(docNum, docItem);
        }
    });

    // Fallback: also react to row-click so clicking anywhere on the row works
    ordersTable.addEventListener("row-click", (e) => {
        const row = e.detail && e.detail.row;
        if (!row) return;

        const docNum = row.getAttribute("data-doc");
        const docItem = row.getAttribute("data-item");

        if (docNum && docItem) {
            openOrderDetails(docNum, docItem);
        }
    });

    // Sorting handlers for sortable columns
    const sortableCols = ['col-doc', 'col-item', 'col-qty', 'col-price', 'col-score'];
    sortableCols.forEach(colId => {
        const col = document.getElementById(colId);
        if (col) {
            col.addEventListener("click", () => handleSort(colId));
        }
    });
}

/**
 * Setup event handlers for the order details panel (AI buttons)
 */
function setupPanelControls() {
    const explainBtn = document.getElementById("explain-btn");
    const classifyBtn = document.getElementById("classify-btn");

    if (explainBtn) {
        explainBtn.addEventListener("click", generateExplanation);
    }
    if (classifyBtn) {
        classifyBtn.addEventListener("click", runClassification);
    }
}

/**
 * Handle column sorting when user clicks a sortable column header
 */
function handleSort(colId) {
    // Toggle direction if clicking same column, else reset to Ascending
    if (state.sortColumn === colId) {
        state.sortDirection = state.sortDirection === 'Ascending' ? 'Descending' : 'Ascending';
    } else {
        state.sortColumn = colId;
        state.sortDirection = 'Ascending';
    }
    
    sortAndRenderOrders();
}

/**
 * Sort current orders and re-render the table
 */
function sortAndRenderOrders() {
    const colMap = {
        'col-doc': 'Sales Document Number',
        'col-item': 'Sales Document Item',
        'col-qty': 'Sales Order item qty',
        'col-price': 'Unit Price',
        'col-score': 'anomaly_score'
    };

    const key = colMap[state.sortColumn];
    if (!key) return;

    state.currentOrders.sort((a, b) => {
        let valA = a[key];
        let valB = b[key];

        // Numeric sort for numbers
        if (key === 'Sales Order item qty' || key === 'Unit Price' || key === 'anomaly_score' || key === 'Sales Document Item') {
            valA = parseFloat(valA);
            valB = parseFloat(valB);
        } else {
            // String sort
            valA = String(valA).toLowerCase();
            valB = String(valB).toLowerCase();
        }

        if (valA < valB) return state.sortDirection === 'Ascending' ? -1 : 1;
        if (valA > valB) return state.sortDirection === 'Ascending' ? 1 : -1;
        return 0;
    });

    renderOrdersTable(state.currentOrders);
}

/**
 * Load initial dashboard data (calendar summary and populate filters)
 */
async function loadInitialData() {
    const container = document.getElementById("heatmap-calendar");

    try {
        // Show loading state
        container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">Loading calendar data...</p>';

        const data = await DashboardAPI.getSummary(); // No params = load all

        // Store calendar data in state
        state.calendarData = data.calendar_data || {};

        // Use years from API response if available, otherwise extract from calendar data
        if (data.years && data.years.length > 0) {
            state.availableYears = data.years;
        } else {
            // Extract available years from the calendar data
            const yearsSet = new Set();
            Object.keys(state.calendarData).forEach(dateStr => {
                const [year] = dateStr.split("-").map(Number);
                yearsSet.add(year);
            });
            state.availableYears = Array.from(yearsSet).sort((a, b) => b - a); // Descending
        }

        // Populate year dropdown
        populateYearDropdown(state.availableYears);

        // If we have data, set initial selections
        if (state.availableYears.length > 0) {
            // Select most recent year
            state.selectedYear = state.availableYears[0];
            setDropdownValue("year-select", state.selectedYear);

            // Get months for selected year and populate dropdown
            const monthsInYear = getMonthsForYear(state.selectedYear);
            populateMonthDropdown(monthsInYear);

            // Select most recent month in that year
            if (monthsInYear.length > 0) {
                state.selectedMonth = monthsInYear[monthsInYear.length - 1];
                setDropdownValue("month-select", state.selectedMonth);
            }

            // Render heatmap calendar
            renderHeatmapCalendar();

            // Select most recent day with data
            selectMostRecentDay();
        } else {
            container.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">No data available. Please ensure the backend is running and data is loaded.</p>';
            clearDailyView();
        }
    } catch (error) {
        console.error("Failed to load summary data", error);
        container.innerHTML = `<p style="text-align: center; color: #d04343; padding: 2rem;">Failed to load data. Please check if the backend server is running at http://localhost:8000</p>`;
        clearDailyView();
    }
}

/**
 * Load data for a specific day (orders and metrics)
 */
async function loadDailyData(date) {
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();

    // Reset the order details panel when changing days
    resetOrderDetailsPanel();

    try {
        const data = await DashboardAPI.getDailyDetails(year, month, day);
        state.currentOrders = data.orders; // Store for client-side sorting
        
        updateMetrics(data.metrics);
        
        // Initial sort
        sortAndRenderOrders();
        
        updateCharts(data.metrics);
    } catch (error) {
        console.error("Failed to load daily details", error);
        clearDailyView();
    }
}

/**
 * Update the KPI metric cards with daily data
 */
function updateMetrics(metrics) {
    document.getElementById("kpi-total-orders").textContent = metrics.total_orders;
    document.getElementById("kpi-anomalies").textContent = metrics.anomaly_count;
    document.getElementById("kpi-rate").textContent = `${(metrics.anomaly_rate * 100).toFixed(1)}%`;
}

/**
 * Render the orders table with the given orders data
 */
function renderOrdersTable(orders) {
    const table = document.getElementById("orders-table");
    // Clear existing rows (keep header)
    const rows = table.querySelectorAll("ui5-table-row");
    rows.forEach(r => r.remove());

    orders.forEach(order => {
        const row = document.createElement("ui5-table-row");
        // Mark row as interactive so clicking anywhere on the row fires the table's row-click event
        row.setAttribute("interactive", "");
        row.setAttribute("data-doc", order["Sales Document Number"]);
        row.setAttribute("data-item", order["Sales Document Item"]);

        const cells = [
            order["Sales Document Number"],
            order["Sales Document Item"],
            order["Material Description"] ? order["Material Description"].substring(0, 20) + "..." : "",
            order["Sales Order item qty"],
            `$${order["Unit Price"] ? order["Unit Price"].toFixed(2) : "0.00"}`,
            order["anomaly_score"] ? order["anomaly_score"].toFixed(3) : "0.000",
            order["predicted_anomaly"] === 1 ? "Anomaly" : "Normal"
        ];

        cells.forEach((text, index) => {
            const cell = document.createElement("ui5-table-cell");
            if (index === 6) {
                // Status column: use a colored tag
                const tag = document.createElement("ui5-tag");
                tag.design = text === "Anomaly" ? "Negative" : "Positive";
                tag.textContent = text;
                cell.appendChild(tag);
            } else {
                cell.textContent = text;
            }
            row.appendChild(cell);
        });

        table.appendChild(row);
    });
}

/**
 * Update the anomaly distribution chart
 */
function updateCharts(metrics) {
    const ctx = document.getElementById('anomaly-chart');
    if (!ctx) return;

    if (state.chart) {
        state.chart.destroy();
    }

    state.chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Anomaly'],
            datasets: [{
                data: [metrics.total_orders - metrics.anomaly_count, metrics.anomaly_count],
                backgroundColor: ['#e5e5e5', '#d04343']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

/**
 * Clear the daily view when no data is available
 */
function clearDailyView() {
    updateMetrics({
        total_orders: "-",
        anomaly_count: "-",
        anomaly_rate: 0,
        total_value: 0
    });
    state.currentOrders = [];
    renderOrdersTable([]);
    resetOrderDetailsPanel();
}

/**
 * Reset the order details panel to show the placeholder
 */
function resetOrderDetailsPanel() {
    state.selectedDoc = null;
    state.selectedItem = null;
    
    const placeholder = document.getElementById("order-placeholder");
    const content = document.getElementById("order-details-content");
    
    if (placeholder) placeholder.style.display = "flex";
    if (content) content.style.display = "none";
}

/**
 * Open/show order details in the side panel (replaces the old dialog approach)
 */
async function openOrderDetails(docNum, docItem) {
    state.selectedDoc = docNum;
    state.selectedItem = docItem;
    
    const placeholder = document.getElementById("order-placeholder");
    const content = document.getElementById("order-details-content");
    
    // Hide placeholder, show content area
    if (placeholder) placeholder.style.display = "none";
    if (content) content.style.display = "flex";

    try {
        const data = await OrdersAPI.getOrder(docNum, docItem);
        renderOrderDetails(data);
    } catch (error) {
        console.error("Failed to load order details", error);
        // Show error in the panel
        const banner = document.getElementById("status-banner");
        if (banner) {
            banner.textContent = "Error loading order details";
            banner.className = "status-banner status-anomaly";
        }
    }
}

/**
 * Render order details in the side panel
 */
function renderOrderDetails(data) {
    const order = data.order;

    // Status Banner
    const banner = document.getElementById("status-banner");
    const isAnomaly = order.predicted_anomaly === 1;
    banner.textContent = isAnomaly ? "ANOMALY DETECTED" : "NORMAL ORDER";
    banner.className = `status-banner ${isAnomaly ? 'status-anomaly' : 'status-normal'}`;

    // Order Details
    document.getElementById("val-sold-to").textContent = order["Sold To number"];
    document.getElementById("val-ship-to").textContent = order["Ship-To Party"];
    document.getElementById("val-material-number").textContent = order["Material Number"];
    document.getElementById("val-material-description").textContent = order["Material Description"];
    document.getElementById("val-qty").textContent = order["Sales Order item qty"];
    document.getElementById("val-value").textContent = `$${order["Order item value"].toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("val-score").textContent = getAnomalyProbabilityLabel(order["anomaly_score"]);

    // SHAP Table
    const table = document.getElementById("shap-table");
    const rows = table.querySelectorAll("ui5-table-row");
    rows.forEach(r => r.remove());

    if (data.shap_explanation && data.shap_explanation.length > 0) {
        data.shap_explanation.forEach(row => {
            const tr = document.createElement("ui5-table-row");
            const formattedValue = formatFeatureValue(row.Feature, row.Value);
            [row.Feature, formattedValue, row.Contribution].forEach(text => {
                const cell = document.createElement("ui5-table-cell");
                cell.textContent = text;
                tr.appendChild(cell);
            });
            table.appendChild(tr);
        });
    } else {
        const tr = document.createElement("ui5-table-row");
        const cell = document.createElement("ui5-table-cell");
        cell.textContent = "No significant contributors found or SHAP not available.";
        tr.appendChild(cell);
        table.appendChild(tr);
    }

    // Reset AI result areas for the new order
    document.getElementById("ai-result").style.display = "none";
    document.getElementById("binary-result").style.display = "none";
    document.getElementById("ai-busy").style.display = "none";
}

/**
 * Generate AI explanation for the selected order
 */
async function generateExplanation() {
    if (!state.selectedDoc) return;

    const busy = document.getElementById("ai-busy");
    const resultBox = document.getElementById("ai-result");
    const btn = document.getElementById("explain-btn");

    busy.style.display = "block";
    btn.disabled = true;
    resultBox.style.display = "none";

    try {
        const data = await OrdersAPI.generateExplanation(state.selectedDoc, state.selectedItem);
        resultBox.innerHTML = marked.parse(data.explanation);
        resultBox.style.display = "block";
    } catch (error) {
        resultBox.textContent = "Error generating explanation.";
        resultBox.style.display = "block";
    } finally {
        busy.style.display = "none";
        btn.disabled = false;
    }
}

/**
 * Run binary classification for the selected order
 */
async function runClassification() {
    if (!state.selectedDoc) return;

    const busy = document.getElementById("classify-busy");
    const resultBox = document.getElementById("binary-result");
    const btn = document.getElementById("classify-btn");

    busy.style.display = "block";
    btn.disabled = true;
    resultBox.style.display = "none";

    try {
        const data = await AnomalyAPI.explainBinary(state.selectedDoc, state.selectedItem);
        resultBox.textContent = data.classification;
        resultBox.style.display = "block";
    } catch (error) {
        resultBox.textContent = "Error running classification.";
        resultBox.style.display = "block";
    } finally {
        busy.style.display = "none";
        btn.disabled = false;
    }
}
