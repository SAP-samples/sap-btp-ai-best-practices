/* Home page specific UI5 components */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Label.js";

import { request } from "../../services/api.js";

// Home-specific API function
async function checkHealth() {
  return await request("/api/health");
}

export default function initHomePage() {
  console.log("Home page initialized");

  // Helper to handle textarea enter key
  const handleTextAreaEnter = (textarea, button) => {
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        button.click();
      }
    });
  };

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

  // Initialize LLM Demo functionality
  const llmInput = document.getElementById("llm-input");
  const llmSubmitButton = document.getElementById("llm-submit-button");
  const llmResponsePanel = document.getElementById("llm-response-panel");
  const llmResponseContainer = document.getElementById("llm-response-container");

  if (llmSubmitButton && llmInput && llmResponseContainer) {
    handleTextAreaEnter(llmInput, llmSubmitButton);

    llmSubmitButton.addEventListener("click", async () => {
      const message = llmInput.value;
      if (!message.trim()) return;

      llmSubmitButton.loading = true;
      llmResponseContainer.textContent = "Thinking...";
      
      if (llmResponsePanel) {
        llmResponsePanel.collapsed = false;
      }

      try {
        const response = await request("/api/llm-demo/chat", "POST", { message });
        if (response.success) {
             llmResponseContainer.textContent = response.text;
        } else {
             llmResponseContainer.textContent = `Error: ${response.error || "Unknown error"}`;
        }
      } catch (error) {
        llmResponseContainer.textContent = `Error: ${error.message}`;
      } finally {
        llmSubmitButton.loading = false;
      }
    });
  }

  // Initialize Agent Demo functionality
  const agentInput = document.getElementById("agent-input");
  const agentSubmitButton = document.getElementById("agent-submit-button");
  const agentResponsePanel = document.getElementById("agent-response-panel");
  const agentResponseContainer = document.getElementById("agent-response-container");

  if (agentSubmitButton && agentInput && agentResponseContainer) {
    handleTextAreaEnter(agentInput, agentSubmitButton);

    agentSubmitButton.addEventListener("click", async () => {
      const message = agentInput.value;
      if (!message.trim()) return;

      agentSubmitButton.loading = true;
      agentResponseContainer.textContent = "Thinking...";
      
      if (agentResponsePanel) {
        agentResponsePanel.collapsed = false;
      }

      try {
        const response = await request("/api/agent-demo/chat", "POST", { message });
        if (response.success) {
             agentResponseContainer.textContent = response.text;
        } else {
             agentResponseContainer.textContent = `Error: ${response.error || "Unknown error"}`;
        }
      } catch (error) {
        agentResponseContainer.textContent = `Error: ${error.message}`;
      } finally {
        agentSubmitButton.loading = false;
      }
    });
  }

  // Initialize HANA Demo functionality
  const hanaTestButton = document.getElementById("hana-test-button");
  const hanaResponsePanel = document.getElementById("hana-response-panel");
  const hanaResponseContainer = document.getElementById("hana-response-container");

  if (hanaTestButton && hanaResponseContainer) {
    hanaTestButton.addEventListener("click", async () => {
      hanaTestButton.loading = true;
      hanaResponseContainer.textContent = "Connecting to HANA...";
      
      if (hanaResponsePanel) {
        hanaResponsePanel.collapsed = false;
      }

      try {
        const response = await request("/api/hana-demo/test", "POST");
        hanaResponseContainer.textContent = response.message;
        if (!response.success) {
            hanaResponseContainer.style.color = "var(--sapNegativeColor)";
        } else {
             hanaResponseContainer.style.color = "var(--sapPositiveColor)";
        }
      } catch (error) {
        hanaResponseContainer.textContent = `Error: ${error.message}`;
        hanaResponseContainer.style.color = "var(--sapNegativeColor)";
      } finally {
        hanaTestButton.loading = false;
      }
    });
  }
}
