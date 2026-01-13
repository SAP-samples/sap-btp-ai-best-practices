/* Dashboard UI5 Components */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/List.js";
import "@ui5/webcomponents/dist/ListItemStandard.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";

/* Icons */
import "@ui5/webcomponents-icons/dist/map.js";
import "@ui5/webcomponents-icons/dist/decline.js";
import "@ui5/webcomponents-icons/dist/navigation-right-arrow.js";

import { request } from "../../services/api.js";

// ============================================================================
// Global State
// ============================================================================

const state = {
  map: null,
  dmaData: [],
  storesData: [],
  dmaMarkers: {},
  storeMarkers: {},
  dmaLayer: null,
  storeLayer: null,
  selectedDma: null,
  selectedStore: null,
  selectedChannel: "All",  // "All", "B&M", or "WEB"
  currentZoom: 5,
  currentTimeseries: null,
  selectedTimePointIndex: null
};

// Constants
const ZOOM_THRESHOLD = 7;
const US_CENTER = [39.8283, -98.5795];
const INITIAL_ZOOM = 5;

// ============================================================================
// API Functions
// ============================================================================

async function fetchDMAs() {
  const response = await request("/api/dma");
  return response.dmas;
}

async function fetchStores() {
  const response = await request("/api/stores");
  return response.stores;
}

async function fetchStoreTimeseries(storeId, channel = "B&M") {
  return await request(`/api/timeseries/store/${storeId}?channel=${encodeURIComponent(channel)}`);
}

async function fetchDMATimeseries(dmaName) {
  const safeName = encodeURIComponent(dmaName);
  return await request(`/api/timeseries/dma/${safeName}`);
}

// ============================================================================
// Initialization
// ============================================================================

export default async function initDashboardPage() {
  console.log("Dashboard page initialized");

  try {
    // Load data from API
    [state.dmaData, state.storesData] = await Promise.all([
      fetchDMAs(),
      fetchStores()
    ]);

    // Render summary panel with overview data
    renderSummaryPanel();

    // Initialize map
    initMap();

    // Setup event handlers
    setupEventHandlers();

  } catch (error) {
    console.error("Error initializing dashboard:", error);
  }
}

function setupEventHandlers() {
  // Clear selection button
  const clearBtn = document.getElementById("clear-selection-btn");
  if (clearBtn) {
    clearBtn.addEventListener("click", clearSelection);
  }

  // Popup close button
  const popupCloseBtn = document.getElementById("popup-close-btn");
  if (popupCloseBtn) {
    popupCloseBtn.addEventListener("click", closeStorePopup);
  }

  // AI Agent button
  const aiAgentBtn = document.getElementById("ai-agent-btn");
  if (aiAgentBtn) {
    aiAgentBtn.addEventListener("click", handleAIAgent);
  }

  // Channel filter toggle
  const channelFilter = document.getElementById("channel-filter");
  if (channelFilter) {
    channelFilter.addEventListener("selection-change", handleChannelChange);
  }

  // Store list selection
  const storeList = document.getElementById("dma-store-list");
  if (storeList) {
    storeList.addEventListener("item-click", (e) => {
      const storeId = parseInt(e.detail.item.getAttribute("data-store-id"));
      const store = state.storesData.find(s => s.id === storeId);
      if (store) {
        selectStore(store);
      }
    });
  }

  // Window resize for charts
  window.addEventListener("resize", resizeCharts);
}

// ============================================================================
// Channel Toggle Handler
// ============================================================================

function handleChannelChange(event) {
  // Get the selected item text
  const selectedItem = event.detail.selectedItems[0];
  if (!selectedItem) return;

  const newChannel = selectedItem.textContent.trim();
  if (state.selectedChannel === newChannel) return;

  state.selectedChannel = newChannel;
  console.log(`Channel changed to: ${state.selectedChannel}`);

  // Re-render all components with new channel data
  refreshDataForChannel();
}

function refreshDataForChannel() {
  // Re-render summary panel
  renderSummaryPanel();

  // Re-create markers with new AUV values
  state.dmaLayer.clearLayers();
  state.storeLayer.clearLayers();
  createDmaMarkers();
  createStoreMarkers();

  // Re-render visible markers based on zoom
  if (state.currentZoom >= ZOOM_THRESHOLD) {
    showStoreMarkers();
  } else {
    showDmaMarkers();
  }

  // Re-populate DMA store table if a DMA is selected
  if (state.selectedDma) {
    populateDmaStoreTable(state.selectedDma.dma);
  }

  // Re-load timeseries for selected store with new channel
  if (state.selectedStore) {
    loadStoreTimeSeries(state.selectedStore.id);
    // Re-show popup with new channel data
    const markerData = state.storeMarkers[state.selectedStore.id];
    if (markerData) {
      showStorePopup(state.selectedStore, markerData.marker);
    }
  }
}

/**
 * Get the display AUV for a store based on the selected channel.
 * @param {Object} store - Store object with channel-specific AUV fields
 * @returns {number} AUV value for the selected channel
 */
function getDisplayAUV(store) {
  switch (state.selectedChannel) {
    case "All":
      return (store.bm_auv_p50 || 0) + (store.web_auv_p50 || 0);
    case "B&M":
      return store.bm_auv_p50 || 0;
    case "WEB":
      return store.web_auv_p50 || 0;
    default:
      return store.auv_p50 || 0;
  }
}

/**
 * Get the display YoY change for a store based on the selected channel.
 * @param {Object} store - Store object with channel-specific YoY fields
 * @returns {number|null} YoY change for the selected channel
 */
function getDisplayYoY(store) {
  switch (state.selectedChannel) {
    case "All":
      // For "All", use B&M YoY as the primary indicator (stores are omnichannel)
      return store.bm_yoy_auv_change;
    case "B&M":
      return store.bm_yoy_auv_change;
    case "WEB":
      return store.web_yoy_auv_change;
    default:
      return store.yoy_auv_change;
  }
}

/**
 * Get the API channel parameter for the selected channel.
 * @returns {string} API channel parameter value
 */
function getApiChannel() {
  switch (state.selectedChannel) {
    case "All":
    case "B&M":
      return "B&M";  // Default to B&M for "All" and "B&M"
    case "WEB":
      return "WEB";
    default:
      return "B&M";
  }
}

// ============================================================================
// Summary Panel
// ============================================================================

function renderSummaryPanel() {
  // Compute overall summary from stores data using channel-specific AUV
  const storesWithYoy = state.storesData.filter(s => {
    const yoy = getDisplayYoY(s);
    const auv = getDisplayAUV(s);
    return yoy !== null && yoy !== undefined && auv > 0;
  });

  // Overall average AUV (average per store) for selected channel
  const totalAUV = state.storesData.reduce((sum, s) => sum + getDisplayAUV(s), 0);
  const avgAUV = state.storesData.length > 0 ? totalAUV / state.storesData.length : 0;

  // Overall weighted YoY change (weighted by AUV) for selected channel
  let overallYoyPct = null;
  if (storesWithYoy.length > 0) {
    const totalWeightedYoy = storesWithYoy.reduce(
      (sum, s) => sum + (getDisplayYoY(s) * getDisplayAUV(s)), 0
    );
    const totalYoyAUV = storesWithYoy.reduce((sum, s) => sum + getDisplayAUV(s), 0);
    overallYoyPct = totalYoyAUV > 0 ? totalWeightedYoy / totalYoyAUV : null;
  }

  // Update headline text
  const summaryText = document.getElementById("overall-summary-text");
  if (summaryText) {
    const direction = overallYoyPct >= 0 ? "an increase" : "a decrease";
    const yoyFormatted = overallYoyPct !== null
      ? `${direction} of ${Math.abs(overallYoyPct).toFixed(1)}%`
      : "no year-over-year data available";

    const channelLabel = state.selectedChannel === "All" ? "" : ` (${state.selectedChannel})`;
    summaryText.textContent = `Average AUV${channelLabel} for comp stores is ${formatCurrency(avgAUV)}, ${yoyFormatted} compared to prior year.`;
  }

  // Populate TOP 10 DMAs table
  const table = document.getElementById("top-dma-table");
  if (table) {
    // Compute DMA aggregates from stores data using channel-specific AUV
    const dmaAggregates = {};
    state.storesData.forEach(store => {
      const dmaName = store.dma;
      const storeAUV = getDisplayAUV(store);
      const storeYoY = getDisplayYoY(store);

      if (!dmaAggregates[dmaName]) {
        dmaAggregates[dmaName] = {
          dma: dmaName,
          totalAUV: 0,
          storeCount: 0,
          totalWeightedYoy: 0,
          totalYoyAUV: 0
        };
      }
      dmaAggregates[dmaName].totalAUV += storeAUV;
      dmaAggregates[dmaName].storeCount += 1;

      // Track weighted YoY for computing DMA YoY percentage
      if (storeYoY !== null && storeYoY !== undefined && storeAUV > 0) {
        dmaAggregates[dmaName].totalWeightedYoy += storeYoY * storeAUV;
        dmaAggregates[dmaName].totalYoyAUV += storeAUV;
      }
    });

    // Convert to array and compute averages
    const dmaList = Object.values(dmaAggregates).map(dma => ({
      dma: dma.dma,
      avgAUV: dma.storeCount > 0 ? dma.totalAUV / dma.storeCount : 0,
      totalAUV: dma.totalAUV,
      storeCount: dma.storeCount,
      yoyPct: dma.totalYoyAUV > 0 ? dma.totalWeightedYoy / dma.totalYoyAUV : null
    }));

    // Sort by total AUV descending (sum of AUV for all stores in DMA)
    const sortedDmas = dmaList
      .sort((a, b) => b.totalAUV - a.totalAUV)
      .slice(0, 10);

    // Clear existing rows
    table.querySelectorAll("ui5-table-row").forEach(row => row.remove());

    // Add rows for top 10 DMAs
    sortedDmas.forEach(dma => {
      const row = document.createElement("ui5-table-row");

      // DMA Name cell - clickable to navigate to DMA on map
      const nameCell = document.createElement("ui5-table-cell");
      nameCell.textContent = dma.dma;
      nameCell.classList.add("dma-link");
      nameCell.addEventListener("click", () => {
        // Find the DMA in state.dmaData to get full DMA object with lat/lng
        const dmaData = state.dmaData.find(d => d.dma === dma.dma);
        if (dmaData) {
          selectDma(dmaData);
        }
      });
      row.appendChild(nameCell);

      // Avg AUV cell (average AUV per store in DMA)
      const avgCell = document.createElement("ui5-table-cell");
      avgCell.textContent = formatCurrency(dma.avgAUV);
      row.appendChild(avgCell);

      // YoY Change % cell
      const deltaCell = document.createElement("ui5-table-cell");
      if (dma.yoyPct !== null) {
        const sign = dma.yoyPct >= 0 ? "+" : "";
        deltaCell.textContent = `${sign}${dma.yoyPct.toFixed(1)}%`;
        deltaCell.className = dma.yoyPct < -5 ? "yoy-decrease" :
                              dma.yoyPct > 5 ? "yoy-increase" : "yoy-stable";
      } else {
        deltaCell.textContent = "N/A";
      }
      row.appendChild(deltaCell);

      // Total AUV cell (sum of AUV for all stores in DMA - justifies ranking)
      const totalCell = document.createElement("ui5-table-cell");
      totalCell.textContent = formatCurrency(dma.totalAUV);
      row.appendChild(totalCell);

      table.appendChild(row);
    });
  }
}

// ============================================================================
// Map Initialization
// ============================================================================

function initMap() {
  // Create map centered on US
  state.map = L.map("map").setView(US_CENTER, INITIAL_ZOOM);

  // Add OpenStreetMap tile layer
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 18
  }).addTo(state.map);

  // Create layer groups for DMAs and stores
  state.dmaLayer = L.layerGroup().addTo(state.map);
  state.storeLayer = L.layerGroup().addTo(state.map);

  // Create markers
  createDmaMarkers();
  createStoreMarkers();

  // Set up event handlers
  state.map.on("zoomend", handleZoomChange);
  state.map.on("moveend", handleMoveEnd);

  // Reposition popup when map moves
  state.map.on("move", function() {
    const popup = document.getElementById("store-popup");
    if (!popup.classList.contains("hidden") && state.selectedStore) {
      const markerData = state.storeMarkers[state.selectedStore.id];
      if (markerData) {
        const point = state.map.latLngToContainerPoint(markerData.marker.getLatLng());
        popup.style.left = (point.x + 20) + "px";
        popup.style.top = (point.y - 50) + "px";
      }
    }
  });

  // Initial render
  handleZoomChange();
}

// ============================================================================
// Marker Creation
// ============================================================================

function createDmaMarkers() {
  // Clear existing markers
  state.dmaMarkers = {};

  // Get channel-specific AUV field based on selected channel
  const getAUV = (dma) => {
    switch (state.selectedChannel) {
      case "All":
        return dma.total_auv_p50 || 0;
      case "B&M":
        return dma.bm_total_auv_p50 || 0;
      case "WEB":
        return dma.web_total_auv_p50 || 0;
      default:
        return dma.total_auv_p50 || 0;
    }
  };

  // Get channel-specific YoY status
  const getYoYStatus = (dma) => {
    switch (state.selectedChannel) {
      case "All":
        return dma.yoy_status;
      case "B&M":
        return dma.bm_yoy_status;
      case "WEB":
        return dma.web_yoy_status;
      default:
        return dma.yoy_status;
    }
  };

  // Use total AUV for marker sizing
  const auvValues = state.dmaData
    .filter(d => getAUV(d) > 0)
    .map(d => getAUV(d));

  const minAUV = auvValues.length > 0 ? Math.min(...auvValues) : 0;
  const maxAUV = auvValues.length > 0 ? Math.max(...auvValues) : 1;

  state.dmaData.forEach(dma => {
    if (!dma.lat || !dma.lng || isNaN(dma.lat) || isNaN(dma.lng)) return;

    const dmaAUV = getAUV(dma);
    let size = 35;
    if (dmaAUV > 0 && maxAUV > minAUV) {
      const normalized = (dmaAUV - minAUV) / (maxAUV - minAUV);
      size = 30 + normalized * 30;
    }

    // Traffic light class based on channel-specific YoY status
    const yoyStatus = getYoYStatus(dma);
    let markerClass = "dma-marker";
    if (yoyStatus === "decrease") {
      markerClass += " dma-yoy-decrease";
    } else if (yoyStatus === "stable") {
      markerClass += " dma-yoy-stable";
    } else if (yoyStatus === "increase") {
      markerClass += " dma-yoy-increase";
    }

    const icon = L.divIcon({
      className: markerClass,
      html: `<div style="width:${size}px;height:${size}px;line-height:${size}px;font-size:${Math.max(10, size / 3)}px">${dma.store_count}</div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2]
    });

    const marker = L.marker([dma.lat, dma.lng], { icon })
      .on("click", () => selectDma(dma));

    state.dmaMarkers[dma.dma] = {
      marker: marker,
      data: dma,
      size: size,
      markerClass: markerClass
    };
  });
}

function createStoreMarkers() {
  // Clear existing markers
  state.storeMarkers = {};

  // Use channel-specific AUV for marker sizing
  const auvValues = state.storesData
    .filter(s => getDisplayAUV(s) > 0)
    .map(s => getDisplayAUV(s));

  const minAUV = auvValues.length > 0 ? Math.min(...auvValues) : 0;
  const maxAUV = auvValues.length > 0 ? Math.max(...auvValues) : 1;

  state.storesData.forEach(store => {
    if (!store.lat || !store.lng || isNaN(store.lat) || isNaN(store.lng)) return;

    const storeAUV = getDisplayAUV(store);
    const storeYoY = getDisplayYoY(store);

    let size = 12;
    if (storeAUV > 0 && maxAUV > minAUV) {
      const normalized = (storeAUV - minAUV) / (maxAUV - minAUV);
      size = 8 + normalized * 12;
    }

    let markerClass = "store-marker";
    // Traffic light color based on channel-specific YoY AUV change
    if (storeYoY !== null && storeYoY !== undefined) {
      if (storeYoY < -5) {
        markerClass += " yoy-decrease";
      } else if (storeYoY > 5) {
        markerClass += " yoy-increase";
      } else {
        markerClass += " yoy-stable";
      }
    } else {
      markerClass += " yoy-unknown";
    }

    const icon = L.divIcon({
      className: markerClass,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2]
    });

    const marker = L.marker([store.lat, store.lng], { icon })
      .on("click", () => selectStore(store));

    state.storeMarkers[store.id] = {
      marker: marker,
      data: store,
      size: size,
      markerClass: markerClass
    };
  });
}

// ============================================================================
// Zoom and Move Handlers
// ============================================================================

function handleZoomChange() {
  const zoom = state.map.getZoom();
  state.currentZoom = zoom;

  if (zoom >= ZOOM_THRESHOLD) {
    showStoreMarkers();
  } else {
    showDmaMarkers();
  }
}

function handleMoveEnd() {
  if (state.currentZoom >= ZOOM_THRESHOLD) {
    updateVisibleStores();
  }
}

function showDmaMarkers() {
  state.storeLayer.clearLayers();

  Object.values(state.dmaMarkers).forEach(({ marker }) => {
    if (!state.dmaLayer.hasLayer(marker)) {
      state.dmaLayer.addLayer(marker);
    }
  });
}

function showStoreMarkers() {
  state.dmaLayer.clearLayers();

  if (state.selectedDma && state.dmaMarkers[state.selectedDma.dma]) {
    state.dmaLayer.addLayer(state.dmaMarkers[state.selectedDma.dma].marker);
  }

  updateVisibleStores();
}

function updateVisibleStores() {
  const bounds = state.map.getBounds();

  Object.values(state.storeMarkers).forEach(({ marker, data }) => {
    const inBounds = bounds.contains([data.lat, data.lng]);

    if (inBounds) {
      if (!state.storeLayer.hasLayer(marker)) {
        state.storeLayer.addLayer(marker);
      }
    } else {
      state.storeLayer.removeLayer(marker);
    }
  });
}

// ============================================================================
// Selection Handlers
// ============================================================================

function selectDma(dma) {
  state.selectedDma = dma;
  state.selectedStore = null;

  updateMarkerSelection();

  document.getElementById("welcome-card").classList.add("hidden");
  document.getElementById("dma-card").classList.remove("hidden");
  closeStorePopup();

  // Update DMA header
  const dmaHeader = document.getElementById("dma-header");
  if (dmaHeader) {
    dmaHeader.titleText = dma.dma;
  }

  // Update store count text
  const storeCountText = document.getElementById("dma-store-count-text");
  if (storeCountText) {
    storeCountText.textContent = `${dma.store_count} store${dma.store_count !== 1 ? 's' : ''}`;
  }

  populateDmaStoreTable(dma.dma);
  zoomToDma(dma.dma);
  loadDmaTimeSeries(dma.dma);
}

function selectStore(store) {
  state.selectedStore = store;

  const dma = state.dmaData.find(d => d.dma === store.dma);
  if (dma) {
    state.selectedDma = dma;
  }

  updateMarkerSelection();

  document.getElementById("welcome-card").classList.add("hidden");

  if (state.selectedDma) {
    document.getElementById("dma-card").classList.remove("hidden");

    const dmaHeader = document.getElementById("dma-header");
    if (dmaHeader) {
      dmaHeader.titleText = state.selectedDma.dma;
    }

    // Update store count text
    const storeCountText = document.getElementById("dma-store-count-text");
    if (storeCountText) {
      storeCountText.textContent = `${state.selectedDma.store_count} store${state.selectedDma.store_count !== 1 ? 's' : ''}`;
    }

    populateDmaStoreTable(state.selectedDma.dma);
  }

  const markerData = state.storeMarkers[store.id];
  if (markerData) {
    showStorePopup(store, markerData.marker);
  }

  loadStoreTimeSeries(store.id);

  if (state.currentZoom < ZOOM_THRESHOLD) {
    state.map.setView([store.lat, store.lng], ZOOM_THRESHOLD + 1);
  } else {
    state.map.panTo([store.lat, store.lng]);
  }
}

function clearSelection() {
  state.selectedDma = null;
  state.selectedStore = null;

  updateMarkerSelection();

  document.getElementById("welcome-card").classList.remove("hidden");
  document.getElementById("dma-card").classList.add("hidden");
  closeStorePopup();

  clearCharts();

  state.map.setView(US_CENTER, INITIAL_ZOOM);
}

// ============================================================================
// Store Popup
// ============================================================================

function showStorePopup(store, marker) {
  const popup = document.getElementById("store-popup");

  // Get channel-specific values
  const storeAUV = getDisplayAUV(store);
  const storeYoY = getDisplayYoY(store);

  document.getElementById("popup-store-name").textContent = store.name || `Store #${store.id}`;
  document.getElementById("popup-location").textContent = `${store.city || ""}, ${store.state || ""}`;
  document.getElementById("popup-store-id").textContent = store.id;
  document.getElementById("popup-store-dma").textContent = store.dma;
  document.getElementById("popup-store-channel").textContent = state.selectedChannel;  // Show selected channel view
  document.getElementById("popup-store-outlet").textContent = store.is_outlet ? "Yes" : "No";
  document.getElementById("popup-design-sf").textContent =
    store.store_design_sf ? formatNumber(store.store_design_sf) + " SF" : "N/A";
  document.getElementById("popup-merch-sf").textContent =
    store.merchandising_sf ? formatNumber(store.merchandising_sf) + " SF" : "N/A";
  document.getElementById("popup-auv").textContent = formatCurrency(storeAUV);

  // Display channel-specific YoY AUV change
  const yoyEl = document.getElementById("popup-yoy-change");
  if (yoyEl) {
    if (storeYoY !== null && storeYoY !== undefined) {
      const sign = storeYoY >= 0 ? "+" : "";
      yoyEl.textContent = `${sign}${storeYoY.toFixed(1)}%`;
      yoyEl.className = storeYoY < -5 ? "yoy-decrease" :
                        storeYoY > 5 ? "yoy-increase" : "yoy-stable";
    } else {
      yoyEl.textContent = "N/A";
      yoyEl.className = "";
    }
  }

  const markerLatLng = marker.getLatLng();
  const point = state.map.latLngToContainerPoint(markerLatLng);

  popup.style.left = (point.x + 20) + "px";
  popup.style.top = (point.y - 50) + "px";

  popup.classList.remove("hidden");

  const mapContainer = document.getElementById("map");
  adjustPopupPosition(popup, mapContainer);
}

function closeStorePopup() {
  document.getElementById("store-popup").classList.add("hidden");
}

function adjustPopupPosition(popup, container) {
  const popupRect = popup.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();

  if (popupRect.right > containerRect.right - 10) {
    popup.style.left = (parseInt(popup.style.left) - popupRect.width - 40) + "px";
  }

  if (popupRect.bottom > containerRect.bottom - 10) {
    popup.style.top = (containerRect.height - popupRect.height - 20) + "px";
  }

  if (popupRect.top < containerRect.top + 10) {
    popup.style.top = "20px";
  }
}

function handleAIAgent() {
  // Store the selected store ID for the chatbot page
  if (state.selectedStore) {
    window.pendingChatbotQuery = {
      storeId: state.selectedStore.id
    };
  }
  // Navigate to chatbot page
  window.pageRouter.navigate("/chatbot");
}

// ============================================================================
// Marker Selection Styling
// ============================================================================

function updateMarkerSelection() {
  Object.entries(state.dmaMarkers).forEach(([dmaName, { marker, data, size, markerClass }]) => {
    const isSelected = state.selectedDma && state.selectedDma.dma === dmaName;
    const className = isSelected ? markerClass + " selected" : markerClass;

    const icon = L.divIcon({
      className: className,
      html: `<div style="width:${size}px;height:${size}px;line-height:${size}px;font-size:${Math.max(10, size / 3)}px">${data.store_count}</div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2]
    });

    marker.setIcon(icon);
  });

  Object.entries(state.storeMarkers).forEach(([storeId, { marker, data, size, markerClass }]) => {
    const isSelected = state.selectedStore && state.selectedStore.id === parseInt(storeId);
    const className = isSelected ? markerClass + " selected" : markerClass;

    const icon = L.divIcon({
      className: className,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2]
    });

    marker.setIcon(icon);
  });
}

function populateDmaStoreTable(dmaName) {
  const table = document.getElementById("dma-store-table");
  if (!table) return;

  // Clear existing rows
  table.querySelectorAll("ui5-table-row").forEach(row => row.remove());

  // Get and sort stores by channel-specific AUV descending
  const dmaStores = state.storesData.filter(s => s.dma === dmaName);
  dmaStores.sort((a, b) => getDisplayAUV(b) - getDisplayAUV(a));

  dmaStores.forEach(store => {
    const storeAUV = getDisplayAUV(store);
    const storeYoY = getDisplayYoY(store);

    const row = document.createElement("ui5-table-row");

    // Store name cell - clickable
    const nameCell = document.createElement("ui5-table-cell");
    nameCell.textContent = store.name || `#${store.id} - ${store.city}`;
    nameCell.classList.add("store-link");
    nameCell.addEventListener("click", () => {
      selectStore(store);
    });
    row.appendChild(nameCell);

    // AUV cell (channel-specific)
    const auvCell = document.createElement("ui5-table-cell");
    auvCell.textContent = formatCurrency(storeAUV);
    row.appendChild(auvCell);

    // YoY AUV Change cell (channel-specific)
    const yoyCell = document.createElement("ui5-table-cell");
    if (storeYoY !== null && storeYoY !== undefined) {
      const sign = storeYoY >= 0 ? "+" : "";
      yoyCell.textContent = `${sign}${storeYoY.toFixed(1)}%`;
      yoyCell.className = storeYoY < -5 ? "yoy-decrease" :
                          storeYoY > 5 ? "yoy-increase" : "yoy-stable";
    } else {
      yoyCell.textContent = "N/A";
    }
    row.appendChild(yoyCell);

    table.appendChild(row);
  });
}

function zoomToDma(dmaName) {
  const dmaStores = state.storesData.filter(s => s.dma === dmaName);

  if (dmaStores.length === 0) return;

  if (dmaStores.length === 1) {
    state.map.setView([dmaStores[0].lat, dmaStores[0].lng], ZOOM_THRESHOLD + 1);
  } else {
    const bounds = L.latLngBounds(dmaStores.map(s => [s.lat, s.lng]));
    state.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 12 });
  }
}

// ============================================================================
// Time Series and Charts
// ============================================================================

async function loadStoreTimeSeries(storeId) {
  try {
    const channel = getApiChannel();
    const data = await fetchStoreTimeseries(storeId, channel);

    data.timeseries.sort((a, b) => new Date(a.date) - new Date(b.date));

    state.currentTimeseries = data.timeseries;
    state.selectedTimePointIndex = data.timeseries.length - 1;

    renderTimeSeriesChart(data.timeseries);

    const selectedPoint = data.timeseries[state.selectedTimePointIndex];

    // Get YoY percentage from pre-computed value or calculate it
    const yoyPct = selectedPoint.yoy_change_pct ||
      (selectedPoint.baseline_sales_p50 && selectedPoint.baseline_sales_p50 > 0
        ? ((selectedPoint.pred_sales_p50 - selectedPoint.baseline_sales_p50) / selectedPoint.baseline_sales_p50 * 100)
        : null);

    renderExplanation(selectedPoint.explanation, selectedPoint.date, yoyPct, selectedPoint.fiscal_week);
  } catch (error) {
    console.error("Error loading time series:", error);
    state.currentTimeseries = null;
    state.selectedTimePointIndex = null;
    clearCharts();
  }
}

async function loadDmaTimeSeries(dmaName) {
  try {
    const data = await fetchDMATimeseries(dmaName);

    data.timeseries.sort((a, b) => new Date(a.date) - new Date(b.date));

    renderDmaTimeSeriesChart(data.timeseries);
    showDmaExplanationPlaceholder();
  } catch (error) {
    console.error("Error loading DMA time series:", error);
    clearCharts();
  }
}

function renderTimeSeriesChart(timeseries) {
  if (!timeseries || timeseries.length === 0) {
    clearTimeSeriesChart();
    return;
  }

  const dates = timeseries.map(d => d.date);
  const p50_2025 = timeseries.map(d => d.pred_sales_p50);
  const p90_2025 = timeseries.map(d => d.pred_sales_p90);
  const p50_2024 = timeseries.map(d => d.baseline_sales_p50);

  // Pre-calculate deltas for tooltip
  const deltas = timeseries.map(d => {
    if (d.baseline_sales_p50 && d.baseline_sales_p50 > 0) {
      const dollarDelta = d.pred_sales_p50 - d.baseline_sales_p50;
      const pctDelta = (dollarDelta / d.baseline_sales_p50) * 100;
      return { dollar: dollarDelta, pct: pctDelta, hasBaseline: true };
    }
    return { dollar: null, pct: null, hasBaseline: false };
  });

  const markerSizes = dates.map((_, i) =>
    i === state.selectedTimePointIndex ? 12 : 6
  );
  const markerColors = dates.map((_, i) =>
    i === state.selectedTimePointIndex ? "#e74c3c" : "#3498db"
  );

  // 2025 confidence band
  const confidenceBand = {
    x: [...dates, ...dates.slice().reverse()],
    y: [...p90_2025, ...p50_2025.slice().reverse()],
    fill: "toself",
    fillcolor: "rgba(52, 152, 219, 0.2)",
    line: { color: "transparent" },
    name: "2025 p50-p90 Range",
    hoverinfo: "skip",
    showlegend: true
  };

  // 2024 baseline line (dashed)
  const line2024 = {
    x: dates,
    y: p50_2024,
    mode: "lines",
    name: "2024 Baseline",
    line: { color: "#95a5a6", width: 2, dash: "dot" },
    hovertemplate: "2024: %{y:$,.0f}<extra></extra>",
    connectgaps: false
  };

  // 2025 forecast line
  const line2025 = {
    x: dates,
    y: p50_2025,
    mode: "lines+markers",
    name: "2025 Forecast",
    line: { color: "#3498db", width: 2 },
    marker: {
      size: markerSizes,
      color: markerColors,
      line: { color: "white", width: 1 }
    },
    hovertemplate: "2025: %{y:$,.0f}<extra></extra>"
  };

  // Delta trace (for tooltip - invisible line)
  const deltaHoverText = deltas.map(d => {
    if (!d.hasBaseline) return "";
    const sign = d.dollar >= 0 ? "+" : "";
    return `Delta: ${sign}$${Math.abs(d.dollar).toLocaleString("en-US", {maximumFractionDigits: 0})} (${sign}${d.pct.toFixed(1)}%)`;
  });

  const deltaTrace = {
    x: dates,
    y: p50_2025,
    mode: "lines",
    line: { color: "transparent", width: 0 },
    showlegend: false,
    hovertemplate: deltaHoverText.map(t => t ? `<b>${t}</b><extra></extra>` : "<extra></extra>"),
    name: "Delta"
  };

  const container = document.getElementById("timeseries-chart");
  const layout = {
    autosize: true,
    height: container.clientHeight || 230,
    margin: { t: 10, r: 15, b: 50, l: 60 },
    xaxis: {
      title: { text: "Week (click to see explanation)", standoff: 5, font: { size: 10 } },
      tickformat: "%b %d",
      tickangle: -45,
      tickfont: { size: 9 },
      gridcolor: "#f0f0f0"
    },
    yaxis: {
      title: { text: "Sales ($)", standoff: 5, font: { size: 10 } },
      tickformat: "$,.0f",
      tickfont: { size: 9 },
      gridcolor: "#f0f0f0"
    },
    legend: {
      orientation: "h",
      y: -0.25,
      x: 0.5,
      xanchor: "center",
      font: { size: 9 }
    },
    hovermode: "x unified",
    plot_bgcolor: "white",
    paper_bgcolor: "transparent"
  };

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot("timeseries-chart", [confidenceBand, line2024, line2025, deltaTrace], layout, config)
    .then(() => {
      resizeCharts();

      const chartEl = document.getElementById("timeseries-chart");
      chartEl.on("plotly_click", function(data) {
        if (data.points && data.points.length > 0) {
          // Find click on the 2025 forecast line (curveNumber === 2)
          const point = data.points.find(p => p.curveNumber === 2);
          if (point) {
            selectTimePoint(point.pointIndex);
          }
        }
      });
    });
}

function renderDmaTimeSeriesChart(timeseries) {
  if (!timeseries || timeseries.length === 0) {
    clearTimeSeriesChart();
    return;
  }

  const dates = timeseries.map(d => d.date);
  const p50 = timeseries.map(d => d.pred_sales_p50);
  const p90 = timeseries.map(d => d.pred_sales_p90);

  const confidenceBand = {
    x: [...dates, ...dates.slice().reverse()],
    y: [...p90, ...p50.slice().reverse()],
    fill: "toself",
    fillcolor: "rgba(52, 152, 219, 0.2)",
    line: { color: "transparent" },
    name: "p50-p90 Range",
    hoverinfo: "skip",
    showlegend: true
  };

  const p50Line = {
    x: dates,
    y: p50,
    mode: "lines+markers",
    name: "Aggregated Sales (p50)",
    line: { color: "#3498db", width: 2 },
    marker: { size: 6, color: "#3498db" },
    hovertemplate: "%{x}<br>p50: %{y:$,.0f}<extra></extra>"
  };

  const p90Line = {
    x: dates,
    y: p90,
    mode: "lines",
    name: "p90 Upper Bound",
    line: { color: "#2980b9", width: 1, dash: "dash" },
    hovertemplate: "%{x}<br>p90: %{y:$,.0f}<extra></extra>"
  };

  const container = document.getElementById("timeseries-chart");
  const layout = {
    autosize: true,
    height: container.clientHeight || 230,
    margin: { t: 10, r: 15, b: 50, l: 65 },
    xaxis: {
      title: { text: "Week", standoff: 5, font: { size: 10 } },
      tickformat: "%b %d",
      tickangle: -45,
      tickfont: { size: 9 },
      gridcolor: "#f0f0f0"
    },
    yaxis: {
      title: { text: "Aggregated Sales ($)", standoff: 5, font: { size: 10 } },
      tickformat: "$,.0f",
      tickfont: { size: 9 },
      gridcolor: "#f0f0f0"
    },
    legend: {
      orientation: "h",
      y: -0.25,
      x: 0.5,
      xanchor: "center",
      font: { size: 9 }
    },
    hovermode: "closest",
    plot_bgcolor: "white",
    paper_bgcolor: "transparent"
  };

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.newPlot("timeseries-chart", [confidenceBand, p50Line, p90Line], layout, config)
    .then(() => resizeCharts());
}

function showDmaExplanationPlaceholder() {
  const header = document.getElementById("explanation-header");
  const content = document.getElementById("explanation-content");

  if (header) {
    header.titleText = "Sales Change Explanation";
    header.subtitleText = "Select a store to see details";
  }
  if (content) {
    content.innerHTML = '<p class="explanation-placeholder">Explanations are store-specific. Select a store to view sales drivers.</p>';
  }
}

function selectTimePoint(index) {
  if (!state.currentTimeseries || index < 0 || index >= state.currentTimeseries.length) {
    return;
  }

  state.selectedTimePointIndex = index;
  const selectedPoint = state.currentTimeseries[index];

  renderTimeSeriesChart(state.currentTimeseries);

  // Get YoY percentage from pre-computed value or calculate it
  const yoyPct = selectedPoint.yoy_change_pct ||
    (selectedPoint.baseline_sales_p50 && selectedPoint.baseline_sales_p50 > 0
      ? ((selectedPoint.pred_sales_p50 - selectedPoint.baseline_sales_p50) / selectedPoint.baseline_sales_p50 * 100)
      : null);

  renderExplanation(selectedPoint.explanation, selectedPoint.date, yoyPct, selectedPoint.fiscal_week);
}

function renderExplanation(explanation, selectedDate, yoyChangePct, fiscalWeek) {
  const header = document.getElementById("explanation-header");
  const content = document.getElementById("explanation-content");

  if (!explanation) {
    if (header) {
      header.titleText = "Sales Change Explanation";
      header.subtitleText = "Click a week to see details";
    }
    if (content) {
      content.innerHTML = '<p class="explanation-placeholder">Select a point on the chart to see why sales changed.</p>';
    }
    return;
  }

  const formattedDate = new Date(selectedDate).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric"
  });

  // Format title with fiscal week if available
  const titleText = fiscalWeek
    ? `Fiscal Week ${fiscalWeek}, ${formattedDate}`
    : `Week of ${formattedDate}`;

  // Handle new stores without baseline
  if (explanation.no_baseline) {
    if (header) {
      header.titleText = titleText;
      header.subtitleText = "New store - no YoY comparison";
    }

    if (content) {
      content.innerHTML = `
        <p class="explanation-summary">${escapeHtml(explanation.summary)}</p>
        <p class="explanation-disclaimer">
          This store opened recently. Year-over-year comparisons will be
          available once 52 weeks of historical data exist.
        </p>
      `;
    }
    return;
  }

  // Update header
  if (header) {
    header.titleText = titleText;
    const yoyText = yoyChangePct !== null
      ? `${yoyChangePct >= 0 ? "+" : ""}${yoyChangePct.toFixed(1)}% vs last year`
      : "No baseline data";
    header.subtitleText = yoyText;
  }

  // Build explanation HTML
  let html = `<p class="explanation-summary">${escapeHtml(explanation.summary)}</p>`;

  if (explanation.drivers && explanation.drivers.length > 0) {
    html += `<div class="explanation-section">`;
    html += `<div class="explanation-section-title">Factors reducing sales:</div>`;
    for (const driver of explanation.drivers) {
      const text = typeof driver === "string" ? driver : driver.description;
      html += `<div class="explanation-driver">- ${escapeHtml(text)}</div>`;
    }
    html += `</div>`;
  }

  if (explanation.offsets && explanation.offsets.length > 0) {
    html += `<div class="explanation-section">`;
    html += `<div class="explanation-section-title">Factors adding to sales:</div>`;
    for (const offset of explanation.offsets) {
      const text = typeof offset === "string" ? offset : offset.description;
      html += `<div class="explanation-offset">+ ${escapeHtml(text)}</div>`;
    }
    html += `</div>`;
  }

  if (explanation.other_factors_impact !== null && explanation.other_factors_impact !== undefined) {
    const impact = explanation.other_factors_impact;
    const absImpact = Math.abs(impact);
    const formattedImpact = absImpact.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

    html += `<div class="explanation-section">`;
    html += `<div class="explanation-section-title">Other factors:</div>`;
    if (impact < 0) {
      html += `<div class="explanation-driver">- Market conditions, seasonality, and other model factors: ~${formattedImpact}</div>`;
    } else {
      html += `<div class="explanation-offset">+ Market conditions, seasonality, and other model factors: ~${formattedImpact}</div>`;
    }
    html += `</div>`;
  }

  if (explanation.is_estimated) {
    html += `<p class="explanation-disclaimer">
      These are estimated impacts based on model analysis.
      Actual effects may vary due to factors not captured in the model.
    </p>`;
  }

  if (content) {
    content.innerHTML = html;
  }
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function clearCharts() {
  clearTimeSeriesChart();
  clearExplanation();
}

function resizeCharts() {
  // Only resize the timeseries chart (explanation panel is not a Plotly chart)
  const el = document.getElementById("timeseries-chart");
  if (el && el.data && el.data.length) {
    Plotly.Plots.resize(el);
  }
}

function clearTimeSeriesChart() {
  Plotly.purge("timeseries-chart");
  document.getElementById("timeseries-chart").innerHTML =
    '<ui5-text class="chart-placeholder">Select a store to view predictions</ui5-text>';
}

function clearExplanation() {
  const header = document.getElementById("explanation-header");
  const content = document.getElementById("explanation-content");

  if (header) {
    header.titleText = "Sales Change Explanation";
    header.subtitleText = "Click a week to see details";
  }
  if (content) {
    content.innerHTML = '<p class="explanation-placeholder">Select a store to view sales drivers.</p>';
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatCurrency(value) {
  if (value === null || value === undefined || isNaN(value)) return "N/A";

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(value);
}

function formatNumber(value) {
  if (value === null || value === undefined || isNaN(value)) return "N/A";

  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function formatFeatureName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, l => l.toUpperCase())
    .substring(0, 25);
}
