// Base URL for API - uses environment variable in production, localhost in development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

/**
 * Generic request helper
 * @param {string} endpoint - API endpoint (e.g., "/dashboard")
 * @param {object} options - Fetch options (method, headers, body)
 * @returns {Promise<any>} - JSON response
 */
export async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const defaultHeaders = {
    "Content-Type": "application/json",
  };

  const config = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, config);
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("API Request Failed:", error);
    throw error;
  }
}

// Dashboard API
export const DashboardAPI = {
  getSummary: (year = null, month = null) => {
    let query = "";
    if (year) query += `year=${year}&`;
    if (month) query += `month=${month}`;
    return request(`/dashboard/summary?${query}`);
  },
  getDailyDetails: (year, month, day) => request(`/dashboard/daily?year=${year}&month=${month}&day=${day}`),
};

// Orders API
export const OrdersAPI = {
  getOrder: (docNumber, docItem) => request(`/orders/${docNumber}/${docItem}`),
  generateExplanation: (docNumber, docItem) => 
    request(`/orders/${docNumber}/${docItem}/explain`, { method: "POST" }),
  getRandomAnomalous: () => request(`/orders/random-anomalous`),
};

// Anomaly API
export const AnomalyAPI = {
  explainBinary: (docNumber, docItem) => request(`/anomaly/explain-binary?doc_number=${docNumber}&doc_item=${docItem}`, { method: "POST" }),
};

// Fine Tuning API
export const FineTuningAPI = {
    upload: (formData) => {
        return fetch(`${API_BASE_URL}/fine-tuning/upload`, {
            method: 'POST',
            body: formData
        }).then(res => res.json());
    },
    train: (params) => request(`/fine-tuning/train`, { 
        method: "POST", 
        body: JSON.stringify(params) 
    }),
    getStatistics: () => request(`/fine-tuning/statistics`),
    getFeatures: () => request(`/fine-tuning/features`),
};
