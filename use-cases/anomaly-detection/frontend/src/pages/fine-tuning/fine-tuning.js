import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Slider.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/StepInput.js";
import "@ui5/webcomponents/dist/Checkbox.js";

import { FineTuningAPI } from "../../services/api.js";

export default function initFineTuning() {
    const fileUploader = document.getElementById("file-uploader");
    const slider = document.getElementById("contamination-slider");
    const sliderValue = document.getElementById("contamination-value");
    const autoContamination = document.getElementById("auto-contamination");
    const trainBtn = document.getElementById("train-btn");
    
    // Initial Load
    loadStatistics();
    loadFeatures();

    // Slider interaction
    slider.addEventListener("input", (e) => {
        sliderValue.textContent = e.target.value;
    });

    // Auto Contamination Toggle
    autoContamination.addEventListener("change", (e) => {
        if (e.target.checked) {
            slider.disabled = true;
            sliderValue.textContent = "Auto";
            sliderValue.style.color = "var(--sapContent_DisabledTextColor)";
        } else {
            slider.disabled = false;
            sliderValue.textContent = slider.value;
            sliderValue.style.color = "inherit";
        }
    });

    // File Upload interaction
    fileUploader.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (file) {
            await handleFileUpload(file);
        }
    });

    // Train Button
    trainBtn.addEventListener("click", startTraining);
    
    document.getElementById("dialog-close-btn").addEventListener("click", () => {
        document.getElementById("training-dialog").close();
    });
}

async function loadStatistics() {
    try {
        const stats = await FineTuningAPI.getStatistics();
        
        document.getElementById("stat-total-records").textContent = stats.total_records.toLocaleString();
        
        const anomalyRate = stats.anomaly_rate !== null 
            ? (stats.anomaly_rate * 100).toFixed(2) + "%" 
            : "N/A";
        document.getElementById("stat-anomaly-rate").textContent = anomalyRate;
        
        document.getElementById("stat-unique-customers").textContent = stats.unique_customers.toLocaleString();
        document.getElementById("stat-unique-materials").textContent = stats.unique_materials.toLocaleString();
        
    } catch (error) {
        console.error("Failed to load statistics:", error);
    }
}

async function loadFeatures() {
    const container = document.getElementById("features-container");
    container.innerHTML = '<ui5-busy-indicator active size="Medium"></ui5-busy-indicator>';

    try {
        const categories = await FineTuningAPI.getFeatures();
        container.innerHTML = ""; // Clear loader
        
        let totalFeatures = 0;
        let selectedCount = 0;

        for (const [category, features] of Object.entries(categories)) {
            const categoryDiv = document.createElement("div");
            categoryDiv.className = "feature-group";
            
            const title = document.createElement("div");
            title.className = "feature-group-title";
            title.textContent = category;
            categoryDiv.appendChild(title);

            features.forEach(feature => {
                totalFeatures++;
                // Default all to selected
                selectedCount++;

                const itemDiv = document.createElement("div");
                itemDiv.className = "feature-item";

                const checkbox = document.createElement("ui5-checkbox");
                checkbox.text = feature.name;
                checkbox.checked = true;
                checkbox.setAttribute("data-feature-id", feature.id);
                checkbox.title = feature.description; // Tooltip

                // Update count on change
                checkbox.addEventListener("change", (e) => {
                    selectedCount += e.target.checked ? 1 : -1;
                    updateFeatureCount(selectedCount, totalFeatures);
                });

                itemDiv.appendChild(checkbox);
                categoryDiv.appendChild(itemDiv);
            });

            container.appendChild(categoryDiv);
        }

        updateFeatureCount(selectedCount, totalFeatures);

    } catch (error) {
        console.error("Failed to load features:", error);
        container.innerHTML = '<ui5-message-strip design="Negative">Failed to load features.</ui5-message-strip>';
    }
}

function updateFeatureCount(selected, total) {
    document.getElementById("selected-features-count").textContent = `Selected Features: ${selected} / ${total}`;
}

async function handleFileUpload(file) {
    const msgStrip = document.getElementById("upload-msg");
    const statusDiv = document.getElementById("file-status");
    
    statusDiv.style.display = "block";
    msgStrip.design = "Information";
    msgStrip.textContent = "Uploading...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await FineTuningAPI.upload(formData);
        msgStrip.design = "Positive";
        msgStrip.textContent = `File uploaded successfully: ${response.filename}`;
    } catch (error) {
        msgStrip.design = "Negative";
        msgStrip.textContent = "Upload failed. Please try again.";
    }
}

async function startTraining() {
    const dialog = document.getElementById("training-dialog");
    const busy = document.getElementById("train-busy");
    const statusText = document.getElementById("train-status-text");
    const closeBtn = document.getElementById("dialog-close-btn");
    
    const contamination = document.getElementById("contamination-slider").value;
    const nEstimators = document.getElementById("n-estimators").value;
    const maxSamples = document.getElementById("max-samples").selectedOption.value;
    const autoContamination = document.getElementById("auto-contamination").checked;
    const customerStratified = document.getElementById("enable-stratification").checked;

    // Collect selected features
    const selectedFeatures = [];
    document.querySelectorAll("#features-container ui5-checkbox").forEach(cb => {
        if (cb.checked) {
            selectedFeatures.push(cb.getAttribute("data-feature-id"));
        }
    });

    dialog.show();
    busy.active = true;
    closeBtn.style.display = "none";
    statusText.textContent = "Training in progress...";

    try {
        const response = await FineTuningAPI.train({
            contamination: autoContamination ? "auto" : parseFloat(contamination),
            n_estimators: parseInt(nEstimators),
            max_samples: maxSamples,
            customer_stratified: customerStratified,
            features: selectedFeatures
        });
        
        busy.active = false;
        statusText.textContent = "Training Completed Successfully!";
        closeBtn.style.display = "block";
    } catch (error) {
        busy.active = false;
        statusText.textContent = "Training Failed.";
        closeBtn.style.display = "block";
    }
}
