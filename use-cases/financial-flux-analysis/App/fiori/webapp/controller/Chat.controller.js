sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/CustomListItem",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/Text",
    "sap/m/FormattedText",
    "sap/ui/core/Icon",
    "financial/flux/analysis/services/apiService"
], function (
    Controller,
    JSONModel,
    CustomListItem,
    VBox,
    HBox,
    Text,
    FormattedText,
    Icon,
    ApiService
) {
    "use strict";

    return Controller.extend("financial.flux.analysis.controller.Chat", {

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------

        onInit: function () {
            this._conversationId = null;
            this._messages       = [];
            this._attachKeyHandler();
        },

        // ------------------------------------------------------------------
        // Keyboard: Enter to send, Shift+Enter for newline
        // ------------------------------------------------------------------

        _attachKeyHandler: function () {
            var that = this;
            this.getView().addEventDelegate({
                onAfterRendering: function () {
                    var oTextArea = that.byId("idChatInput");
                    if (!oTextArea) { return; }
                    var oDom = oTextArea.getDomRef
                        ? (oTextArea.getDomRef("inner") || oTextArea.getDomRef())
                        : null;
                    if (!oDom) { return; }
                    oDom.addEventListener("keydown", function (oEvent) {
                        if (oEvent.key === "Enter" && !oEvent.shiftKey) {
                            oEvent.preventDefault();
                            var oBtn = that.byId("idChatSendBtn");
                            if (oBtn && oBtn.getEnabled()) {
                                that.onSendMessage();
                            }
                        }
                    });
                }
            });
        },

        // ------------------------------------------------------------------
        // Input live change
        // ------------------------------------------------------------------

        onInputLiveChange: function (oEvent) {
            var sVal     = oEvent.getSource().getValue() || "";
            var bHasText = sVal.trim().length > 0;
            this.byId("idChatSendBtn").setEnabled(bHasText);
            var oCount = this.byId("idChatCharCount");
            if (oCount) {
                oCount.setText(sVal.length > 0 ? sVal.length + " / 1000" : "");
            }
        },

        // ------------------------------------------------------------------
        // Send message
        // ------------------------------------------------------------------

        onSendMessage: function () {
            var oInput   = this.byId("idChatInput");
            var sMessage = (oInput.getValue() || "").trim();
            if (!sMessage) { return; }

            oInput.setValue("");
            this.byId("idChatSendBtn").setEnabled(false);
            var oCount = this.byId("idChatCharCount");
            if (oCount) { oCount.setText(""); }

            var oEmptyState = this.byId("idChatEmptyState");
            if (oEmptyState) { oEmptyState.setVisible(false); }

            this._appendMessage("user", sMessage);
            this._setBusy(true);
            this._hideError();

            var oContext = this._buildContext();
            var that     = this;

            ApiService.sendChatMessage({
                message:         sMessage,
                conversation_id: that._conversationId,
                context:         oContext
            }).then(function (oData) {
                that._conversationId = oData.conversation_id;
                that._appendMessage("assistant", oData.response || "");
                that._setBusy(false);
            }).catch(function (oErr) {
                that._setBusy(false);
                var sErrMsg = (oErr && oErr.message)
                    ? oErr.message
                    : "An error occurred while contacting the AI assistant.";
                that._showError(sErrMsg);
                that._appendMessage("assistant", "\u26a0\ufe0f " + sErrMsg, true);
            });
        },

        // ------------------------------------------------------------------
        // Suggestion chips
        // ------------------------------------------------------------------

        onSuggestionPress: function (oEvent) {
            var sText  = oEvent.getSource().getText();
            var oInput = this.byId("idChatInput");
            oInput.setValue(sText);
            this.byId("idChatSendBtn").setEnabled(true);
            var oCount = this.byId("idChatCharCount");
            if (oCount) { oCount.setText(sText.length + " / 1000"); }
            this.onSendMessage();
        },

        // ------------------------------------------------------------------
        // Clear conversation
        // ------------------------------------------------------------------

        onClearConversation: function () {
            this._conversationId = null;
            this._messages       = [];

            var oList = this.byId("idChatMessageList");
            if (oList) { oList.removeAllItems(); }

            var oEmptyState = this.byId("idChatEmptyState");
            if (oEmptyState) { oEmptyState.setVisible(true); }

            this._hideError();
            this._setBusy(false);

            var oInput = this.byId("idChatInput");
            if (oInput) { oInput.setValue(""); }
            this.byId("idChatSendBtn").setEnabled(false);
            var oCount = this.byId("idChatCharCount");
            if (oCount) { oCount.setText(""); }
        },

        // ------------------------------------------------------------------
        // Error strip close
        // ------------------------------------------------------------------

        onErrorStripClose: function () {
            this._hideError();
        },

        // ------------------------------------------------------------------
        // Private: append message bubble
        // ------------------------------------------------------------------

        _appendMessage: function (sRole, sText, bError) {
            var oList   = this.byId("idChatMessageList");
            var bIsUser = (sRole === "user");
            var sHtml   = this._markdownToHtml(sText);

            var sTime = new Date().toLocaleTimeString([], {
                hour:   "2-digit",
                minute: "2-digit"
            });

            // Bubble container
            var oBubble = new VBox({
                class: bIsUser
                    ? "fluxChatBubbleUser"
                    : (bError ? "fluxChatBubbleError" : "fluxChatBubbleAssistant")
            });

            // Bubble header: role + timestamp
            var oHeader = new HBox({
                alignItems:     "Center",
                justifyContent: "SpaceBetween",
                class:          "fluxChatBubbleHeader"
            });

            if (!bIsUser) {
                oHeader.addItem(new Icon({
                    src:   "sap-icon://ai",
                    class: "fluxChatBubbleRoleIcon sapUiTinyMarginEnd"
                }));
            }

            oHeader.addItem(new Text({
                text:  bIsUser ? "You" : "AI Assistant",
                class: "fluxChatBubbleLabel"
            }));

            oHeader.addItem(new Text({
                text:  sTime,
                class: "fluxChatTimestamp"
            }));

            oBubble.addItem(oHeader);

            // Message content
            oBubble.addItem(new FormattedText({
                htmlText: sHtml,
                class:    "fluxChatBubbleText"
            }));

            // Alignment wrapper
            var oWrapper = new HBox({
                justifyContent: bIsUser ? "End" : "Start",
                class:          "fluxChatMessageRow"
            });
            oWrapper.addItem(oBubble);

            oList.addItem(new CustomListItem({ content: [oWrapper] }));

            this._scrollToBottom();
            this._messages.push({ role: sRole, text: sText });
        },

        // ------------------------------------------------------------------
        // Private: markdown to safe HTML
        // ------------------------------------------------------------------

        _markdownToHtml: function (sText) {
            if (!sText) { return ""; }

            var s = sText
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");

            // Bold: **text**
            s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

            // Bullet list lines starting with "- " or "* "
            var aLines  = s.split("\n");
            var inList  = false;
            var aResult = [];

            for (var i = 0; i < aLines.length; i++) {
                var sLine    = aLines[i];
                var bBullet  = /^[-*]\s+/.test(sLine);

                if (bBullet) {
                    if (!inList) { aResult.push("<ul>"); inList = true; }
                    aResult.push("<li>" + sLine.replace(/^[-*]\s+/, "") + "</li>");
                } else {
                    if (inList) { aResult.push("</ul>"); inList = false; }
                    if (sLine.trim() === "") {
                        aResult.push("<br/>");
                    } else {
                        aResult.push(sLine + "<br/>");
                    }
                }
            }
            if (inList) { aResult.push("</ul>"); }

            return aResult.join("");
        },

        // ------------------------------------------------------------------
        // Private: scroll to bottom
        // ------------------------------------------------------------------

        _scrollToBottom: function () {
            var oContainer = this.byId("idChatMessagesContainer");
            if (!oContainer) { return; }
            setTimeout(function () {
                var oDomRef = oContainer.getDomRef();
                if (oDomRef) { oDomRef.scrollTop = oDomRef.scrollHeight; }
            }, 60);
        },

        // ------------------------------------------------------------------
        // Private: busy indicator
        // ------------------------------------------------------------------

        _setBusy: function (bBusy) {
            var oBusyRow = this.byId("idChatBusyRow");
            if (oBusyRow) { oBusyRow.setVisible(bBusy); }
            var oSendBtn = this.byId("idChatSendBtn");
            if (oSendBtn) { oSendBtn.setEnabled(!bBusy); }
            if (bBusy) { this._scrollToBottom(); }
        },

        // ------------------------------------------------------------------
        // Private: error strip
        // ------------------------------------------------------------------

        _showError: function (sMsg) {
            var oStrip = this.byId("idChatErrorStrip");
            if (oStrip) { oStrip.setText(sMsg); oStrip.setVisible(true); }
        },

        _hideError: function () {
            var oStrip = this.byId("idChatErrorStrip");
            if (oStrip) { oStrip.setVisible(false); oStrip.setText(""); }
        },

        // ------------------------------------------------------------------
        // Private: build context from parent app model
        // ------------------------------------------------------------------

        _buildContext: function () {
            try {
                var oComponent = this.getOwnerComponent();
                var oRootView  = oComponent ? oComponent.getRootControl() : null;
                var oAppModel  = oRootView  ? oRootView.getModel("app")   : null;
                if (oAppModel) {
                    return {
                        currentYear:    oAppModel.getProperty("/currentYear")            || null,
                        previousYear:   oAppModel.getProperty("/previousYear")           || null,
                        companyCode:    oAppModel.getProperty("/selectedCompanyCode")    || null,
                        segment:        oAppModel.getProperty("/selectedSegment")        || null,
                        functionalArea: oAppModel.getProperty("/selectedFunctionalArea") || null
                    };
                }
            } catch (e) {
                // Context is optional — silently ignore
            }
            return {};
        }

    });
});