sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel"
], function (Controller, MessageToast, MessageBox, JSONModel) {
    "use strict";

    return Controller.extend("videoincidentmonitor.controller.AnalyzeMedia", {

        onInit: function () {
            // Initialize view model
            const oViewModel = new JSONModel({
                selectedFile: null,
                mediaType: "video",
                fileName: "",
                fileSize: 0,
                analyzing: false
            });
            this.getView().setModel(oViewModel, "view");

            // Set default instruction based on media type
            this._setDefaultInstruction("video");
        },

        onNavBack: function () {
            window.location.href = "/test/flpSandbox.html";
        },

        onMediaTypeChange: function (oEvent) {
            const sMediaType = oEvent.getParameter("item").getKey();
            this.getView().getModel("view").setProperty("/mediaType", sMediaType);
            this._setDefaultInstruction(sMediaType);

            // Update file uploader file types
            const oFileUploader = this.byId("fileUploader");
            if (sMediaType === "video") {
                oFileUploader.setFileType(["mp4", "avi", "mov", "webm"]);
            } else {
                oFileUploader.setFileType(["wav", "mp3", "ogg", "flac"]);
            }
        },

        onFileChange: function (oEvent) {
            const oFileUploader = oEvent.getSource();
            const oFile = oEvent.getParameter("files")[0];

            if (!oFile) {
                return;
            }

            const oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/selectedFile", oFile);
            oViewModel.setProperty("/fileName", oFile.name);
            oViewModel.setProperty("/fileSize", (oFile.size / 1024 / 1024).toFixed(2));

            // Update file info text
            this.byId("fileInfo").setText(`Selected: ${oFile.name} (${(oFile.size / 1024 / 1024).toFixed(2)} MB)`);

            // Show preview
            this._showMediaPreview(oFile);

            // Enable analyze button
            this.byId("analyzeButton").setEnabled(true);
        },

        onTemperatureChange: function (oEvent) {
            const fValue = oEvent.getParameter("value");
            this.byId("temperatureValue").setText(fValue.toFixed(1));
        },

        onAnalyze: async function () {
            const oViewModel = this.getView().getModel("view");
            const oFile = oViewModel.getProperty("/selectedFile");

            if (!oFile) {
                MessageBox.warning("Please select a media file first.");
                return;
            }

            // Get parameters
            const sInstruction = this.byId("instructionText").getValue();
            const fTemperature = parseFloat(this.byId("temperatureSlider").getValue());
            const iMaxTokens = parseInt(this.byId("maxTokensInput").getValue());

            if (!sInstruction) {
                MessageBox.warning("Please provide an instruction for analysis.");
                return;
            }

            // Show results panel and progress
            this.byId("resultsPanel").setVisible(true);
            this.byId("analysisProgress").setVisible(true);
            this.byId("resultsBox").setVisible(false);
            this.byId("analysisStatus").setText("Processing...").setState("Information");
            this.byId("analyzeButton").setEnabled(false);

            try {
                // Prepare form data
                const formData = new FormData();
                formData.append("file", oFile);
                formData.append("instruction", sInstruction);
                formData.append("temperature", fTemperature);
                formData.append("maxTokens", iMaxTokens);
                formData.append("autoAnalyze", "true");

                // Upload and analyze
                const response = await fetch("/odata/v4/VideoIncidentService/MediaAnalysis", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();

                // Hide progress, show results
                this.byId("analysisProgress").setVisible(false);
                this.byId("resultsBox").setVisible(true);

                // Update status
                if (result.status === "completed") {
                    this.byId("analysisStatus").setText("Completed").setState("Success");
                } else if (result.status === "failed") {
                    this.byId("analysisStatus").setText("Failed").setState("Error");
                } else {
                    this.byId("analysisStatus").setText(result.status).setState("Warning");
                }

                // Display results
                this.byId("analysisResultText").setText(result.analysisResult || "No result available");

                // Incident detection
                if (result.incidentDetected) {
                    this.byId("incidentStatus").setText("Yes").setState("Error");
                    this.byId("severityBox").setVisible(true);

                    // Set severity
                    const severityMap = {
                        "critical": "Error",
                        "high": "Warning",
                        "medium": "Warning",
                        "low": "Success"
                    };
                    const severityState = severityMap[result.severity] || "None";
                    this.byId("severityStatus").setText(result.severity ? result.severity.toUpperCase() : "Unknown").setState(severityState);
                } else {
                    this.byId("incidentStatus").setText("No").setState("Success");
                    this.byId("severityBox").setVisible(false);
                }

                // Metrics
                this.byId("promptTokens").setText(result.promptTokens || "-");
                this.byId("completionTokens").setText(result.completionTokens || "-");
                this.byId("totalTokens").setText(result.totalTokens || "-");
                this.byId("processingTime").setText(result.processingTime ? `${result.processingTime}s` : "-");

                // Store result ID for download
                this._currentResultId = result.ID;

                MessageToast.show("Analysis completed successfully!");

            } catch (error) {
                console.error("Analysis error:", error);
                this.byId("analysisProgress").setVisible(false);
                this.byId("analysisStatus").setText("Error").setState("Error");
                MessageBox.error(`Analysis failed: ${error.message}`);
            } finally {
                this.byId("analyzeButton").setEnabled(true);
            }
        },

        onDownloadReport: function () {
            const sResult = this.byId("analysisResultText").getText();
            const sFileName = this.getView().getModel("view").getProperty("/fileName");
            const sReportName = `${sFileName}_analysis.txt`;

            // Create blob and download
            const blob = new Blob([sResult], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = sReportName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            MessageToast.show("Report downloaded");
        },

        onViewAll: function () {
            window.location.href = "/test/flpSandbox.html#MediaAnalysisList-display";
        },

        // Helper methods

        _setDefaultInstruction: function (sMediaType) {
            const aDefaultInstructions = {
                "video": "Analyze this video for safety violations. Are there any incidents, missing safety equipment, or hazardous conditions?",
                "audio": "Transcribe this audio file. Write the text verbatim and identify the language."
            };

            this.byId("instructionText").setValue(aDefaultInstructions[sMediaType]);
        },

        _showMediaPreview: function (oFile) {
            const sMediaType = this.getView().getModel("view").getProperty("/mediaType");
            const sObjectURL = URL.createObjectURL(oFile);

            let sHTML;
            if (sMediaType === "video") {
                sHTML = `<video controls style="max-width: 100%; max-height: 400px; border-radius: 4px;">
                            <source src="${sObjectURL}" type="${oFile.type}">
                            Your browser does not support the video tag.
                         </video>`;
            } else {
                sHTML = `<audio controls style="width: 100%;">
                            <source src="${sObjectURL}" type="${oFile.type}">
                            Your browser does not support the audio tag.
                         </audio>`;
            }

            this.byId("mediaPreview").setContent(sHTML);
            this.byId("mediaPreviewBox").setVisible(true);
        }

    });
});
