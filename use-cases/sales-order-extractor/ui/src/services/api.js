const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_API_KEY || "your-super-secret-api-key";

// General Request function to the API
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...headers
    },
    body: body ? JSON.stringify(body) : null
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

// File upload request function (for multipart/form-data)
export async function uploadRequest(endpoint, formData, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      ...headers
      // Note: Don't set Content-Type for FormData, let browser set it with boundary
    },
    body: formData
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

// API Service object with all endpoints
export const apiService = {
  // Health check
  async checkHealth() {
    return request("/api/health");
  },

  // PDF extraction endpoints
  async uploadPDF(formData) {
    const endpoint = `/api/pdf/upload`;
    return uploadRequest(endpoint, formData);
  },

  // Excel/CSV extraction endpoints
  async uploadExcel(formData) {
    const endpoint = `/api/excel/upload`;
    return uploadRequest(endpoint, formData);
  },

  async extractFromBase64(fileContent, filename, extractionModel = "anthropic", temperature = 0.1, maxTokens = 2000) {
    return request("/api/pdf/extract", "POST", {
      file_content: fileContent,
      filename,
      extraction_model: extractionModel,
      temperature,
      max_tokens: maxTokens
    });
  }
};
