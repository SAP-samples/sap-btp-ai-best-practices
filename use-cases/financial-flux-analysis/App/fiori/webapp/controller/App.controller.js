sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Item",
    "sap/m/BusyDialog"
], function (Controller, JSONModel, MessageToast, MessageBox, CoreItem, BusyDialog) {
    "use strict";

    var BASE_URL = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
        ? "" : "https://flux-backend.cfapps.eu10-005.hana.ondemand.com";

    return Controller.extend("financial.flux.analysis.controller.App", {

        onInit: function () {
            this._initModels();
            // Create the initial-load BusyDialog — same component used by "Explain"
            this._initBusyDialog = new BusyDialog({ title: "Please wait" });
            this._initBusyDialog.setText("Loading financial data\u2026");
            this._initBusyDialog.open();
            this._loadInitData();
        },

        // ------------------------------------------------------------------
        // Model initialization
        // ------------------------------------------------------------------

        _initModels: function () {
            var oAppModel = new JSONModel({
                currentYear:          "",
                previousYear:         "",
                selectedCompanyCode:  "",
                selectedSegment:      "",
                selectedFunctionalArea: "",
                availableYears:       [],
                companyCodes:         [],
                segments:             [],
                functionalAreas:      [],
                busy: false
            });
            this.getView().setModel(oAppModel, "app");
        },

        // ------------------------------------------------------------------
        // Load initial data from /api/init-data
        // ------------------------------------------------------------------

        _loadInitData: function () {
            var that = this;
            var oModel = this.getView().getModel("app");

            fetch(BASE_URL + "/api/init-data")
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.status !== "success") {
                        // Close busy indicator on error too
                        if (that._initBusyDialog) { that._initBusyDialog.close(); }
                        MessageBox.error("Failed to load initial data: " + (data.message || "Unknown error"));
                        return;
                    }

                    var aYears = (data.years || []).map(function (y) {
                        return { key: String(y), text: String(y) };
                    });
                    var aCodes = (data.companyCodes || []).map(function (c) {
                        return { key: String(c), text: String(c) };
                    });
                    var aSegments = (data.segments || []).map(function (s) {
                        return { key: String(s), text: String(s) };
                    });
                    var aFuncAreas = (data.functionalAreas || []).map(function (f) {
                        return { key: String(f), text: String(f) };
                    });

                    oModel.setProperty("/availableYears",   aYears);
                    oModel.setProperty("/companyCodes",     aCodes);
                    oModel.setProperty("/segments",         aSegments);
                    oModel.setProperty("/functionalAreas",  aFuncAreas);

                    // Auto-select: currentYear = most recent, previousYear = second most recent
                    var sortedYears = aYears.map(function (y) { return parseInt(y.key, 10); })
                        .sort(function (a, b) { return b - a; });

                    var iCurrentYear  = sortedYears.length > 0 ? sortedYears[0] : new Date().getFullYear();
                    var iPreviousYear = sortedYears.length > 1 ? sortedYears[1] : iCurrentYear - 1;

                    oModel.setProperty("/currentYear",  String(iCurrentYear));
                    oModel.setProperty("/previousYear", String(iPreviousYear));

                    // Populate year selects
                    that._populateYearSelects(aYears, String(iCurrentYear), String(iPreviousYear));

                    // Populate company code select
                    that._populateCompanySelect(aCodes);

                    // Populate segment select
                    that._populateSelect("idSegmentSelect", aSegments, "All Segments");

                    // Populate functional area select
                    that._populateSelect("idFunctionalAreaSelect", aFuncAreas, "All Functional Areas");

                    // Auto-trigger analysis
                    that._triggerAnalysis();
                })
                .catch(function (err) {
                    if (that._initBusyDialog) { that._initBusyDialog.close(); }
                    MessageBox.error("Error loading initial data: " + err.message);
                });
        },

        // ------------------------------------------------------------------
        // Populate year selects
        // ------------------------------------------------------------------

        _populateYearSelects: function (aYears, sCurrentKey, sPreviousKey) {
            var oCurSelect  = this.byId("idCurrentYearSelect");
            var oPrevSelect = this.byId("idPreviousYearSelect");

            if (oCurSelect) {
                oCurSelect.destroyItems();
                aYears.forEach(function (y) {
                    oCurSelect.addItem(new CoreItem({ key: y.key, text: y.text }));
                });
                oCurSelect.setSelectedKey(sCurrentKey);
            }

            if (oPrevSelect) {
                oPrevSelect.destroyItems();
                aYears.forEach(function (y) {
                    oPrevSelect.addItem(new CoreItem({ key: y.key, text: y.text }));
                });
                oPrevSelect.setSelectedKey(sPreviousKey);
            }
        },

        // ------------------------------------------------------------------
        // Populate company code select
        // ------------------------------------------------------------------

        _populateCompanySelect: function (aCodes) {
            var oSelect = this.byId("idCompanyCodeSelect");
            if (!oSelect) return;
            oSelect.destroyItems();
            oSelect.addItem(new CoreItem({ key: "", text: "All Companies" }));
            aCodes.forEach(function (c) {
                oSelect.addItem(new CoreItem({ key: c.key, text: c.text }));
            });
            oSelect.setSelectedKey("");
        },

        /**
         * Generic helper to populate any Select control with an "All" option + items.
         * @param {string} sId       byId control ID
         * @param {Array}  aItems    [{key, text}]
         * @param {string} sAllText  Label for the "All" option
         */
        _populateSelect: function (sId, aItems, sAllText) {
            var oSelect = this.byId(sId);
            if (!oSelect) return;
            oSelect.destroyItems();
            oSelect.addItem(new CoreItem({ key: "", text: sAllText || "All" }));
            aItems.forEach(function (item) {
                oSelect.addItem(new CoreItem({ key: item.key, text: item.text }));
            });
            oSelect.setSelectedKey("");
        },

        // ------------------------------------------------------------------
        // Year / Company / Segment / Functional Area change handlers
        // ------------------------------------------------------------------

        onCurrentYearChange: function (oEvent) {
            var sKey = oEvent.getSource().getSelectedKey();
            this.getView().getModel("app").setProperty("/currentYear", sKey);
            this._triggerAnalysis();
        },

        onPreviousYearChange: function (oEvent) {
            var sKey = oEvent.getSource().getSelectedKey();
            this.getView().getModel("app").setProperty("/previousYear", sKey);
            this._triggerAnalysis();
        },

        onCompanyCodeChange: function (oEvent) {
            var sKey  = oEvent.getSource().getSelectedKey();
            var sText = oEvent.getSource().getSelectedItem()
                ? oEvent.getSource().getSelectedItem().getText()
                : "";
            this.getView().getModel("app").setProperty("/selectedCompanyCode", sKey);
            // Update snapped company label
            var oSnapped = this.byId("idSnappedCompany");
            if (oSnapped) {
                oSnapped.setText(sKey ? "· " + sText : "");
            }
            this._triggerAnalysis();
        },

        onSegmentChange: function (oEvent) {
            var sKey = oEvent.getSource().getSelectedKey();
            this.getView().getModel("app").setProperty("/selectedSegment", sKey);
            this._triggerAnalysis();
        },

        onFunctionalAreaChange: function (oEvent) {
            var sKey = oEvent.getSource().getSelectedKey();
            this.getView().getModel("app").setProperty("/selectedFunctionalArea", sKey);
            this._triggerAnalysis();
        },

        // ------------------------------------------------------------------
        // Refresh Data button
        // ------------------------------------------------------------------

        onRefreshDataPress: function () {
            var that = this;
            MessageToast.show("Refreshing data from SAP HANA...");

            fetch(BASE_URL + "/api/refresh", { method: "POST" })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.status === "success") {
                        MessageToast.show("Data refreshed: " + (data.rows || 0) + " rows loaded.");
                        that._triggerAnalysis();
                    } else {
                        MessageBox.error("Refresh failed: " + (data.message || "Unknown error"));
                    }
                })
                .catch(function (err) {
                    MessageBox.error("Error refreshing data: " + err.message);
                });
        },

        // ------------------------------------------------------------------
        // Tab select
        // ------------------------------------------------------------------

        onIconTabBarMainSelect: function (oEvent) {
            var sKey = oEvent.getParameter("key");
            this.getView().getModel("app").setProperty("/currentView", sKey);
        },

        // ------------------------------------------------------------------
        // Trigger analysis in Variances view
        // ------------------------------------------------------------------

        _triggerAnalysis: function () {
            var oModel       = this.getView().getModel("app");
            var sCurrentYear = oModel.getProperty("/currentYear");
            var sPrevYear    = oModel.getProperty("/previousYear");
            var sCompany     = oModel.getProperty("/selectedCompanyCode");
            var sSegment     = oModel.getProperty("/selectedSegment");
            var sFuncArea    = oModel.getProperty("/selectedFunctionalArea");

            if (!sCurrentYear || !sPrevYear) return;

            var oParams = {
                current_year:     parseInt(sCurrentYear, 10),
                previous_year:    parseInt(sPrevYear, 10),
                company_code:     sCompany  || null,
                segment:          sSegment  || null,
                functional_area:  sFuncArea || null,
                // Pass the initial-load BusyDialog reference (null after first load)
                _initBusyDialog: this._initBusyDialog || null
            };

            // Clear the reference so subsequent triggers don't re-close it
            this._initBusyDialog = null;

            var oVariancesCtrl = this._getVariancesController();
            if (oVariancesCtrl && oVariancesCtrl.loadGroupedData) {
                oVariancesCtrl.loadGroupedData(oParams);
            } else {
                // Retry after short delay — Variances view may still be loading asynchronously
                var that = this;
                setTimeout(function () {
                    var oCtrl = that._getVariancesController();
                    if (oCtrl && oCtrl.loadGroupedData) {
                        oCtrl.loadGroupedData(oParams);
                    }
                }, 600);
            }
        },

        // ------------------------------------------------------------------
        // Helper: get Variances controller
        // ------------------------------------------------------------------

        _getVariancesController: function () {
            var oTabBar = this.byId("idMainIconTabBar");
            if (!oTabBar) return null;
            var aItems = oTabBar.getItems();
            if (!aItems || !aItems.length) return null;
            var oFirstTab = aItems[0];
            var aContent  = oFirstTab.getContent ? oFirstTab.getContent() : [];
            if (!aContent.length) return null;
            var oView = aContent[0];
            return oView && oView.getController ? oView.getController() : null;
        }
    });
});