sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Table",
    "sap/m/Column",
    "sap/m/ColumnListItem",
    "sap/m/Text",
    "sap/m/Label",
    "sap/m/ObjectNumber",
    "sap/m/BusyDialog",
    "sap/ui/core/Fragment"
], function (
    Controller, JSONModel, MessageToast, MessageBox,
    Dialog, Button, MTable, MColumn, ColumnListItem, MText, MLabel, ObjectNumber,
    BusyDialog, Fragment
) {
    var BASE_URL = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
        ? "" : "https://flux-backend.cfapps.eu10-005.hana.ondemand.com";

    "use strict";

    var GROUP_OPTIONS = ["G/L Account", "Profit Center", "Cost Center", "Financial Statement Line Item"];

    // ------------------------------------------------------------------
    // Amount formatter helpers
    // ------------------------------------------------------------------

    function _formatMillions(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return "";
        var abs = Math.abs(n);
        var sign = n < 0 ? "-" : "";
        if (abs >= 1000000) {
            return sign + "$" + (abs / 1000000).toFixed(2) + "M";
        } else if (abs >= 1000) {
            return sign + "$" + (abs / 1000).toFixed(1) + "K";
        } else {
            return sign + "$" + abs.toFixed(2);
        }
    }

    function _formatDelta(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return "";
        var abs = Math.abs(n);
        var sign = n > 0 ? "+" : (n < 0 ? "-" : "");
        if (abs >= 1000000) {
            return sign + "$" + (abs / 1000000).toFixed(2) + "M";
        } else if (abs >= 1000) {
            return sign + "$" + (abs / 1000).toFixed(1) + "K";
        } else {
            return sign + "$" + abs.toFixed(2);
        }
    }

    function _formatPercent(v) {
        if (v === null || v === undefined || v === "") return "N/A";
        var n = Number(v);
        if (isNaN(n)) return "N/A";
        var sign = n > 0 ? "+" : "";
        return sign + n.toFixed(1) + "%";
    }

    /**
     * Accounting-aware CSS class for variance coloring.
     * Uses isFavorable (bool | null) from backend — NOT raw delta sign.
     *
     * isFavorable = true  → GREEN  (favorable P&L impact)
     * isFavorable = false → RED    (unfavorable P&L impact)
     * isFavorable = null  → NEUTRAL (balance sheet / unknown account nature)
     */
    function _favorabilityClass(isFavorable) {
        if (isFavorable === null || isFavorable === undefined) {
            return "fluxNeutralValue fluxNumericCell";
        }
        return isFavorable
            ? "fluxPositiveValue fluxNumericCell"
            : "fluxNegativeValue fluxNumericCell";
    }

    /**
     * Accounting-aware sap.ui.core.ValueState for ObjectNumber.
     * Uses isFavorable (bool | null) from backend — NOT raw delta sign.
     */
    function _favorabilityState(isFavorable) {
        if (isFavorable === null || isFavorable === undefined) return "None";
        return isFavorable ? "Success" : "Error";
    }

    function _processRow(row) {
        var sName = (row.name || "").trim();
        // isFavorable comes from the backend variance_interpreter.
        // null = neutral (balance sheet / unknown), true = favorable, false = unfavorable.
        var isFav = (row.isFavorable !== undefined && row.isFavorable !== null)
            ? row.isFavorable
            : null;
        return {
            key:                      row.key,
            name:                     sName,
            // displayName: show name if available, otherwise fall back to key code
            displayName:              sName || row.key,
            currentAmount:            row.currentAmount,
            previousAmount:           row.previousAmount,
            variance:                 row.variance,
            variancePercent:          row.variancePercent,
            records:                  row.records,
            isFavorable:              isFav,
            accountNature:            row.accountNature   || "UNKNOWN",
            varianceDirection:        row.varianceDirection || "NEUTRAL",
            currentAmountFormatted:   _formatMillions(row.currentAmount),
            previousAmountFormatted:  _formatMillions(row.previousAmount),
            varianceFormatted:        _formatDelta(row.variance),
            variancePercentFormatted: _formatPercent(row.variancePercent),
            // Coloring is accounting-aware: expense decrease = GREEN, revenue increase = GREEN
            varianceClass:            _favorabilityClass(isFav),
            variancePercentClass:     _favorabilityClass(isFav),
            varianceState:            _favorabilityState(isFav)
        };
    }

    // ------------------------------------------------------------------

    return Controller.extend("financial.flux.analysis.controller.Variances", {

        onInit: function () {
            this._initVarianceModel();
            this._drilldownDialog  = null;
            this._aiDialog         = null;
            this._busyDialog       = null;
            this._searchQuery      = "";
        },

        _initVarianceModel: function () {
            var oModel = new JSONModel({
                groupedData:          [],
                filteredData:         [],
                filteredCount:        0,
                totalCount:           0,
                selectedGroupBy:      "G/L Account",
                selectedGroupByName:  "G/L Account Name",
                selectedGroupByIndex: 0,
                loading:              false,
                error:                null,
                // Context stored for drilldown and AI explain
                _current_year:    null,
                _previous_year:   null,
                _company_code:    null,
                _segment:         null,
                _functional_area: null,
                _dataLoaded:      false
            });
            this.getView().setModel(oModel, "variances");
        },

        // ------------------------------------------------------------------
        // Public: called by App.controller.js
        // ------------------------------------------------------------------

        loadGroupedData: function (oParams) {
            var oModel = this.getView().getModel("variances");
            var oP = oParams || {};

            oModel.setProperty("/_current_year",    oP.current_year    || null);
            oModel.setProperty("/_previous_year",   oP.previous_year   || null);
            oModel.setProperty("/_company_code",    oP.company_code    || null);
            oModel.setProperty("/_segment",         oP.segment         || null);
            oModel.setProperty("/_functional_area", oP.functional_area || null);
            oModel.setProperty("/_dataLoaded",      true);

            // Store the initial-load BusyDialog reference (only present on first load)
            this._initBusyDialog = oP._initBusyDialog || null;

            this._searchQuery = "";
            var oSearch = this.byId("idSearchField");
            if (oSearch) oSearch.setValue("");

            this._loadGroupedData(
                oModel.getProperty("/selectedGroupBy"),
                oP.current_year,
                oP.previous_year,
                oP.company_code,
                oP.segment         || null,
                oP.functional_area || null
            );
        },

        // ------------------------------------------------------------------
        // RadioButtonGroup handler
        // ------------------------------------------------------------------

        onRadioButtonGroupGroupBySelect: function (oEvent) {
            var iIndex   = oEvent.getParameter("selectedIndex");
            var sSelected = GROUP_OPTIONS[iIndex] || GROUP_OPTIONS[0];
            var oModel   = this.getView().getModel("variances");

            oModel.setProperty("/selectedGroupBy",      sSelected);
            oModel.setProperty("/selectedGroupByName",  sSelected + " Name");
            oModel.setProperty("/selectedGroupByIndex", iIndex);

            if (!oModel.getProperty("/_dataLoaded")) return;

            this._searchQuery = "";
            var oSearch = this.byId("idSearchField");
            if (oSearch) oSearch.setValue("");

            this._loadGroupedData(
                sSelected,
                oModel.getProperty("/_current_year"),
                oModel.getProperty("/_previous_year"),
                oModel.getProperty("/_company_code"),
                oModel.getProperty("/_segment"),
                oModel.getProperty("/_functional_area")
            );
        },

        // ------------------------------------------------------------------
        // Search handlers
        // ------------------------------------------------------------------

        onSearchLiveChange: function (oEvent) {
            this._searchQuery = (oEvent.getParameter("newValue") || "").toLowerCase().trim();
            this._applyFilter();
        },

        onSearch: function (oEvent) {
            this._searchQuery = (oEvent.getParameter("query") || "").toLowerCase().trim();
            this._applyFilter();
        },

        _applyFilter: function () {
            var oModel    = this.getView().getModel("variances");
            var aAll      = oModel.getProperty("/groupedData") || [];
            var sQuery    = this._searchQuery;

            var aFiltered = sQuery
                ? aAll.filter(function (row) {
                    return (row.key || "").toLowerCase().indexOf(sQuery) !== -1
                        || (row.name || "").toLowerCase().indexOf(sQuery) !== -1;
                })
                : aAll.slice();

            oModel.setProperty("/filteredData",  aFiltered);
            oModel.setProperty("/filteredCount", aFiltered.length);
        },

        // ------------------------------------------------------------------
        // Row press (type="Active") — drilldown
        // ------------------------------------------------------------------

        onRowPress: function (oEvent) {
            var oItem = oEvent.getSource();
            var oCtx  = oItem.getBindingContext("variances");
            if (!oCtx) return;
            this._openDrilldown(oCtx.getProperty("key"));
        },

        // Key link press — drilldown
        onKeyLinkPress: function (oEvent) {
            var oCtx = oEvent.getSource().getBindingContext("variances");
            if (!oCtx) return;
            this._openDrilldown(oCtx.getProperty("key"));
        },

        // ------------------------------------------------------------------
        // Explain button — AI Analysis dialog
        // ------------------------------------------------------------------

        onExplainPress: function (oEvent) {
            // Stop event from bubbling to the row press handler
            if (oEvent.stopPropagation) { oEvent.stopPropagation(); }

            var oCtx = oEvent.getSource().getBindingContext("variances");
            if (!oCtx) return;

            var oRowData     = oCtx.getObject();
            var oVarModel    = this.getView().getModel("variances");
            var sKey         = oRowData.key;
            var sDisplayName = oRowData.displayName || sKey;
            var sGroupBy     = oVarModel.getProperty("/selectedGroupBy");
            var iCurrentYear = oVarModel.getProperty("/_current_year");
            var iPrevYear    = oVarModel.getProperty("/_previous_year");
            var sCompany     = oVarModel.getProperty("/_company_code");
            var sSegment     = oVarModel.getProperty("/_segment");
            var sFuncArea    = oVarModel.getProperty("/_functional_area");

            if (!iCurrentYear || !iPrevYear) {
                MessageBox.warning("Please select fiscal years before running AI analysis.");
                return;
            }

            // Show busy dialog while LLM processes
            this._showBusyDialog("Generating AI analysis\u2026");

            var that = this;

            fetch(BASE_URL + "/api/ai-explain", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    account_number:  sKey,
                    group_by:        sGroupBy,
                    current_year:    iCurrentYear,
                    previous_year:   iPrevYear,
                    company_code:    sCompany  || null,
                    segment:         sSegment  || null,
                    functional_area: sFuncArea || null
                })
            })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                that._closeBusyDialog();

                if (data.status !== "success") {
                    MessageBox.error("AI analysis failed: " + (data.message || "Unknown error"));
                    return;
                }

                // Build aiResult model
                var cur   = data.currentTotal  || 0;
                var prev  = data.previousTotal || 0;
                var delta = data.delta         || 0;
                var pct   = data.deltaPercent  || 0;

                // Map docTypeBreakdown → docTypes with formatted amount
                var aDocTypes = (data.docTypeBreakdown || []).map(function (dt) {
                    return {
                        jeType:          dt.jeType     || "",
                        jeTypeName:      dt.jeTypeName || "",
                        amountFormatted: that._fmtMoney(dt.amount || 0),
                        count:           dt.count || 0
                    };
                });

                // Map supplierBreakdown → suppliers with formatted amount
                var aSuppliers = (data.supplierBreakdown || []).map(function (s) {
                    return {
                        supplier:        s.supplier || "",
                        amountFormatted: that._fmtMoney(s.amount || 0)
                    };
                });

                // Map flashVariance → flashItems with full comparative data.
                // Flash categories are expense-type: delta < 0 = less expense = favorable (GREEN).
                // isFavorable comes from backend (accounting-aware) — not raw delta sign.
                var aFlashItems = (data.flashVariance || []).map(function (item) {
                    var d    = item.delta        || 0;
                    var pct  = item.deltaPercent;
                    var isFav = (item.isFavorable !== undefined && item.isFavorable !== null)
                        ? item.isFavorable
                        : (d < 0 ? true : (d > 0 ? false : null));
                    return {
                        category:       item.category || "",
                        previousFmt:    that._fmtMoney(item.previous || 0),
                        currentFmt:     that._fmtMoney(item.current  || 0),
                        deltaFmt:       that._fmtDelta(d),
                        deltaPctFmt:    (pct !== null && pct !== undefined) ? that._fmtPct(pct) : "N/A",
                        deltaState:     isFav === true ? "Success" : (isFav === false ? "Error" : "None"),
                        impactText:     isFav === true ? "Favorable" : (isFav === false ? "Unfavorable" : "Neutral"),
                        delta:          d,
                        hasNetZero:     item.hasNetZeroActivity || false,
                        netZeroNote:    item.hasNetZeroActivity ? "⚠ Contains net-zero internal reallocations" : ""
                    };
                });

                // Build flat period breakdown list (all categories × all months, sorted by month)
                var aFlashPeriods = [];
                (data.flashVariance || []).forEach(function (item) {
                    (item.periods || []).forEach(function (p) {
                        var d    = p.delta || 0;
                        var pct  = p.deltaPercent;
                        var isFav = (p.isFavorable !== undefined && p.isFavorable !== null)
                            ? p.isFavorable
                            : (d < 0 ? true : (d > 0 ? false : null));
                        aFlashPeriods.push({
                            category:    item.category || "",
                            monthLabel:  p.monthLabel  || "",
                            month:       p.month       || 0,
                            previousFmt: that._fmtMoney(p.previous || 0),
                            currentFmt:  that._fmtMoney(p.current  || 0),
                            deltaFmt:    that._fmtDelta(d),
                            deltaPctFmt: (pct !== null && pct !== undefined) ? that._fmtPct(pct) : "N/A",
                            deltaState:  isFav === true ? "Success" : (isFav === false ? "Error" : "None"),
                            impactText:  isFav === true ? "Favorable" : (isFav === false ? "Unfavorable" : "Neutral")
                        });
                    });
                });
                // Sort period rows by month then category
                aFlashPeriods.sort(function (a, b) {
                    return a.month !== b.month ? a.month - b.month : a.category.localeCompare(b.category);
                });

                var sFlashTitle      = "Flash Expense Comparative Analysis";
                var sFlashPrevLabel  = "FY" + iPrevYear;
                var sFlashCurrLabel  = "FY" + iCurrentYear;

                // Accounting-aware state: use isFavorable from backend (not raw delta sign)
                // isFavorable=true  → GREEN (favorable P&L impact)
                // isFavorable=false → RED   (unfavorable P&L impact)
                // isFavorable=null  → NONE  (balance sheet / unknown)
                var isFav = (data.isFavorable !== undefined && data.isFavorable !== null)
                    ? data.isFavorable : null;
                var sDeltaState = isFav === true ? "Success" : (isFav === false ? "Error" : "None");

                var oAIModel = new JSONModel({
                    accountName:           sDisplayName,
                    accountKey:            sKey,
                    explanation:           data.explanation || "No explanation available.",
                    segment:               data.segment    || "N/A",
                    previousYearFormatted: that._fmtMoney(prev),
                    currentYearFormatted:  that._fmtMoney(cur),
                    deltaFormatted:        that._fmtDelta(delta),
                    deltaPercentFormatted: that._fmtPct(pct),
                    deltaState:            sDeltaState,
                    recordsSummary:        (data.currentRecords || 0) + " (current) / " +
                                           (data.previousRecords || 0) + " (previous)",
                    docTypes:              aDocTypes,
                    suppliers:             aSuppliers,
                    flashVarianceTitle:    sFlashTitle,
                    flashPrevYearLabel:    sFlashPrevLabel,
                    flashCurrYearLabel:    sFlashCurrLabel,
                    flashItems:            aFlashItems,
                    flashPeriods:          aFlashPeriods,
                    lineItems:             []   // loaded on demand via "Load Detail"
                });

                // Store context for line-item loading
                oAIModel.setProperty("/_key",             sKey);
                oAIModel.setProperty("/_groupBy",         sGroupBy);
                oAIModel.setProperty("/_currentYear",     iCurrentYear);
                oAIModel.setProperty("/_previousYear",    iPrevYear);
                oAIModel.setProperty("/_company",         sCompany  || null);
                oAIModel.setProperty("/_segment",         sSegment  || null);
                oAIModel.setProperty("/_functionalArea",  sFuncArea || null);

                that._openAIDialog(oAIModel);
            })
            .catch(function (err) {
                that._closeBusyDialog();
                MessageBox.error("Error calling AI analysis: " + err.message);
            });
        },

        // ------------------------------------------------------------------
        // AI Dialog — open
        // ------------------------------------------------------------------

        _openAIDialog: function (oAIModel) {
            var that = this;

            if (this._aiDialog) {
                this._aiDialog.destroy();
                this._aiDialog = null;
            }

            Fragment.load({
                id:         this.getView().getId(),
                name:       "financial.flux.analysis.fragment.AIAnalysisDialog",
                controller: this
            }).then(function (oDialog) {
                that._aiDialog = oDialog;
                that.getView().addDependent(oDialog);
                oDialog.setModel(oAIModel, "aiResult");
                oDialog.open();
            }).catch(function (err) {
                MessageBox.error("Could not load AI dialog: " + err.message);
            });
        },

        // ------------------------------------------------------------------
        // AI Dialog — close
        // ------------------------------------------------------------------

        onAIDialogClose: function () {
            if (this._aiDialog) {
                this._aiDialog.close();
                this._aiDialog.destroy();
                this._aiDialog = null;
            }
        },

        // ------------------------------------------------------------------
        // AI Dialog — Load Detail (line items)
        // ------------------------------------------------------------------

        onExplainDetailPress: function () {
            if (!this._aiDialog) return;

            var oAIModel = this._aiDialog.getModel("aiResult");
            if (!oAIModel) return;

            // If already loaded, just expand the panel
            var aItems = oAIModel.getProperty("/lineItems") || [];
            if (aItems.length > 0) {
                var oPanel = Fragment.byId(this.getView().getId(), "idDetailPanel");
                if (oPanel) oPanel.setExpanded(true);
                return;
            }

            var sKey         = oAIModel.getProperty("/_key");
            var sGroupBy     = oAIModel.getProperty("/_groupBy");
            var iCurrentYear = oAIModel.getProperty("/_currentYear");
            var iPrevYear    = oAIModel.getProperty("/_previousYear");
            var sCompany     = oAIModel.getProperty("/_company");
            var sSegment     = oAIModel.getProperty("/_segment");
            var sFuncArea    = oAIModel.getProperty("/_functionalArea");

            this._showBusyDialog("Loading line items\u2026");

            var that = this;
            var url  = BASE_URL + "/api/account-line-items"
                + "?group_by="      + encodeURIComponent(sGroupBy)
                + "&key="           + encodeURIComponent(sKey)
                + "&current_year="  + encodeURIComponent(iCurrentYear)
                + "&previous_year=" + encodeURIComponent(iPrevYear);
            if (sCompany)   url += "&company_code="    + encodeURIComponent(sCompany);
            if (sSegment)   url += "&segment="         + encodeURIComponent(sSegment);
            if (sFuncArea)  url += "&functional_area=" + encodeURIComponent(sFuncArea);

            fetch(url)
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    that._closeBusyDialog();
                    if (data.status !== "success") {
                        MessageBox.error("Failed to load line items: " + (data.message || "Unknown error"));
                        return;
                    }
                    oAIModel.setProperty("/lineItems", data.items || []);
                    var oPanel = Fragment.byId(that.getView().getId(), "idDetailPanel");
                    if (oPanel) oPanel.setExpanded(true);
                })
                .catch(function (err) {
                    that._closeBusyDialog();
                    MessageBox.error("Error loading line items: " + err.message);
                });
        },

        // ------------------------------------------------------------------
        // BusyDialog helpers
        // ------------------------------------------------------------------

        _showBusyDialog: function (sText) {
            if (!this._busyDialog) {
                this._busyDialog = new BusyDialog({ title: "Please wait" });
            }
            this._busyDialog.setText(sText || "Loading\u2026");
            this._busyDialog.open();
        },

        _closeBusyDialog: function () {
            if (this._busyDialog) {
                this._busyDialog.close();
            }
        },

        // ------------------------------------------------------------------
        // Numeric format helpers (local, no dependency on formatter module)
        // ------------------------------------------------------------------

        _fmtMoney: function (v) {
            var n = Number(v);
            if (isNaN(n)) return "";
            var abs  = Math.abs(n);
            var sign = n < 0 ? "-" : "";
            if (abs >= 1e6)  return sign + "$" + (abs / 1e6).toFixed(2) + "M";
            if (abs >= 1e3)  return sign + "$" + (abs / 1e3).toFixed(1) + "K";
            return sign + "$" + abs.toFixed(2);
        },

        _fmtDelta: function (v) {
            var n = Number(v);
            if (isNaN(n)) return "";
            var abs  = Math.abs(n);
            var sign = n > 0 ? "+" : (n < 0 ? "-" : "");
            if (abs >= 1e6)  return sign + "$" + (abs / 1e6).toFixed(2) + "M";
            if (abs >= 1e3)  return sign + "$" + (abs / 1e3).toFixed(1) + "K";
            return sign + "$" + abs.toFixed(2);
        },

        _fmtPct: function (v) {
            var n = Number(v);
            if (isNaN(n)) return "N/A";
            return (n > 0 ? "+" : "") + n.toFixed(1) + "%";
        },

        // ------------------------------------------------------------------
        // Drilldown
        // ------------------------------------------------------------------

        _openDrilldown: function (sKey) {
            var that   = this;
            var oModel = this.getView().getModel("variances");

            var sGroupBy     = oModel.getProperty("/selectedGroupBy");
            var iCurrentYear = oModel.getProperty("/_current_year");
            var iPrevYear    = oModel.getProperty("/_previous_year");
            var sCompany     = oModel.getProperty("/_company_code");
            var sSegment     = oModel.getProperty("/_segment");
            var sFuncArea    = oModel.getProperty("/_functional_area");

            if (!iCurrentYear || !iPrevYear) {
                MessageBox.warning("Please select fiscal years before drilling down.");
                return;
            }

            var url = BASE_URL + "/api/group-detail"
                + "?group_by="      + encodeURIComponent(sGroupBy)
                + "&key="           + encodeURIComponent(sKey)
                + "&current_year="  + encodeURIComponent(iCurrentYear)
                + "&previous_year=" + encodeURIComponent(iPrevYear);
            if (sCompany)  url += "&company_code="    + encodeURIComponent(sCompany);
            if (sSegment)  url += "&segment="         + encodeURIComponent(sSegment);
            if (sFuncArea) url += "&functional_area=" + encodeURIComponent(sFuncArea);

            fetch(url)
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.status !== "success") {
                        MessageBox.error("Drilldown failed: " + (data.message || "Unknown error"));
                        return;
                    }
                    that._showDrilldownDialog(sGroupBy, sKey, iCurrentYear, iPrevYear, data.detail || []);
                })
                .catch(function (err) {
                    MessageBox.error("Error loading drilldown: " + err.message);
                });
        },

        _showDrilldownDialog: function (sGroupBy, sKey, iCurrentYear, iPrevYear, aDetail) {
            if (this._drilldownDialog) {
                this._drilldownDialog.destroy();
                this._drilldownDialog = null;
            }

            var oDrillModel = new JSONModel({ detail: aDetail });

            var oTable = new MTable({
                items: {
                    path: "drill>/detail",
                    template: new ColumnListItem({
                        cells: [
                            new MText({ text: "{drill>monthLabel}" }),
                            new MText({
                                text: {
                                    path: "drill>currentAmount",
                                    formatter: _formatMillions
                                }
                            }),
                            new MText({
                                text: {
                                    path: "drill>previousAmount",
                                    formatter: _formatMillions
                                }
                            }),
                            new MText({
                                text: {
                                    path: "drill>delta",
                                    formatter: _formatDelta
                                }
                            })
                        ]
                    })
                },
                columns: [
                    new MColumn({ header: new MText({ text: "Month" }) }),
                    new MColumn({ header: new MText({ text: "FY " + iCurrentYear }), hAlign: "End" }),
                    new MColumn({ header: new MText({ text: "FY " + iPrevYear }), hAlign: "End" }),
                    new MColumn({ header: new MText({ text: "Delta" }), hAlign: "End" })
                ]
            });
            oTable.setModel(oDrillModel, "drill");
            oTable.addStyleClass("fluxDrilldownTable");

            this._drilldownDialog = new Dialog({
                title: sGroupBy + ": " + sKey,
                contentWidth: "44rem",
                content: [oTable],
                beginButton: new Button({
                    text: "Close",
                    press: function () {
                        this._drilldownDialog.close();
                    }.bind(this)
                }),
                afterClose: function () {
                    this._drilldownDialog.destroy();
                    this._drilldownDialog = null;
                }.bind(this)
            });
            this._drilldownDialog.addStyleClass("fluxDrilldownDialog");

            this.getView().addDependent(this._drilldownDialog);
            this._drilldownDialog.open();
        },

        // ------------------------------------------------------------------
        // Internal: load grouped data from backend
        // ------------------------------------------------------------------

        _loadGroupedData: function (sGroupBy, iCurrentYear, iPreviousYear, sCompanyCode, sSegment, sFunctionalArea) {
            var oModel = this.getView().getModel("variances");
            oModel.setProperty("/loading", true);
            oModel.setProperty("/error",   null);

            // Capture and clear the initial-load BusyDialog so it's closed exactly once
            var oInitBusy = this._initBusyDialog || null;
            this._initBusyDialog = null;

            if (!iCurrentYear || !iPreviousYear) {
                oModel.setProperty("/loading",       false);
                oModel.setProperty("/groupedData",   []);
                oModel.setProperty("/filteredData",  []);
                oModel.setProperty("/filteredCount", 0);
                oModel.setProperty("/totalCount",    0);
                if (oInitBusy) { oInitBusy.close(); }
                return;
            }

            var url = BASE_URL + "/api/grouped-analysis"
                + "?group_by="      + encodeURIComponent(sGroupBy || "G/L Account")
                + "&current_year="  + encodeURIComponent(iCurrentYear)
                + "&previous_year=" + encodeURIComponent(iPreviousYear);
            if (sCompanyCode)   url += "&company_code="    + encodeURIComponent(sCompanyCode);
            if (sSegment)       url += "&segment="         + encodeURIComponent(sSegment);
            if (sFunctionalArea) url += "&functional_area=" + encodeURIComponent(sFunctionalArea);

            var that = this;

            fetch(url)
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    oModel.setProperty("/loading", false);
                    // Close initial-load BusyDialog now that data is ready
                    if (oInitBusy) { oInitBusy.close(); }
                    if (data.status === "success") {
                        var aProcessed = (data.results || []).map(_processRow);
                        oModel.setProperty("/groupedData",   aProcessed);
                        oModel.setProperty("/totalCount",    aProcessed.length);
                        that._applyFilter();
                    } else {
                        oModel.setProperty("/error", data.message || "Unknown error");
                        MessageBox.error("Analysis failed: " + (data.message || "Unknown error"));
                    }
                })
                .catch(function (err) {
                    oModel.setProperty("/loading", false);
                    // Close initial-load BusyDialog on error too
                    if (oInitBusy) { oInitBusy.close(); }
                    oModel.setProperty("/error",   err.message);
                    MessageBox.error("Error loading data: " + err.message);
                });
        }
    });
});