/* Chat-specific UI5 components */
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Panel.js";

import { request } from "../../services/api.js";

// Chat-specific API function
async function postChatMessage(model, message) {
  return await request(`/api/chat/${model}`, "POST", { message });
}

export default function initChatPage() {
  console.log("Chat page initialized");

  // Initialize chat functionality
  const sendButton = document.getElementById("send-button");

  if (sendButton) {
    sendButton.addEventListener("click", async () => {
      const modelSelect = document.getElementById("model-select");
      const promptInput = document.getElementById("prompt-input");
      const responsePanel = document.getElementById("response-panel");
      const responseContainer = document.getElementById("response-container");

      const selectedModel = modelSelect.value;
      const prompt = promptInput.value;

      if (!prompt) {
        alert("Please enter a prompt.");
        return;
      }

      responsePanel.collapsed = false;
      responseContainer.innerHTML = "<i>Loading...</i>";
      sendButton.loading = true;

      try {
        const data = await postChatMessage(selectedModel, prompt);
        responseContainer.textContent = data.text;
      } catch (error) {
        console.error("Error:", error);
        responseContainer.textContent = `An error occurred: ${error.message}`;
      } finally {
        sendButton.loading = false;
      }
    });
  }
}
