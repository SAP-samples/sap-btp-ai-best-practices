/* Home page specific UI5 components */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Panel.js";

import { request } from "../../services/api.js";

// Home-specific API function
async function checkHealth() {
  return await request("/api/health");
}

export default function initHomePage() {
  console.log("Home page initialized");

  // Initialize health check functionality
  const healthCheckButton = document.getElementById("health-check-button");
  const healthResponsePanel = document.getElementById("health-response-panel");
  const healthResponseContainer = document.getElementById("health-response-container");

  if (healthCheckButton && healthResponseContainer) {
    healthCheckButton.addEventListener("click", async () => {
      healthCheckButton.loading = true;
      healthResponseContainer.textContent = "Loading...";

      if (healthResponsePanel) {
        healthResponsePanel.collapsed = false;
      }

      try {
        const healthData = await checkHealth();
        healthResponseContainer.textContent = JSON.stringify(healthData, null, 2);
      } catch (error) {
        healthResponseContainer.textContent = `Error: ${error.message}`;
      } finally {
        healthCheckButton.loading = false;
      }
    });
  }
}
