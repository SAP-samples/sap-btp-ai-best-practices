sap.ui.define([], function () {
    "use strict";

    // -----------------------------------------------------------------------
    // Number formatters
    // -----------------------------------------------------------------------

    function formatMillions(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return String(v);
        var abs  = Math.abs(n);
        var sign = n < 0 ? "-" : "";
        if (abs >= 1000000) {
            return sign + "$" + (abs / 1000000).toFixed(2) + "M";
        } else if (abs >= 1000) {
            return sign + "$" + (abs / 1000).toFixed(1) + "K";
        } else {
            return sign + "$" + abs.toFixed(2);
        }
    }

    function formatDelta(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return String(v);
        var abs  = Math.abs(n);
        var sign = n > 0 ? "+" : (n < 0 ? "-" : "");
        if (abs >= 1000000) {
            return sign + "$" + (abs / 1000000).toFixed(2) + "M";
        } else if (abs >= 1000) {
            return sign + "$" + (abs / 1000).toFixed(1) + "K";
        } else {
            return sign + "$" + abs.toFixed(2);
        }
    }

    function formatPercent(v) {
        if (v === null || v === undefined || v === "") return "N/A";
        var n = Number(v);
        if (isNaN(n)) return "N/A";
        var sign = n > 0 ? "+" : "";
        return sign + n.toFixed(1) + "%";
    }

    function currencyUSD(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return String(v);
        return new Intl.NumberFormat("en-US", {
            style: "currency",
            currency: "USD",
            maximumFractionDigits: 2
        }).format(n);
    }

    function numberInt(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return String(v);
        return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(n);
    }

    function number2(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return String(v);
        return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(n);
    }

    function percent1(v) {
        if (v === null || v === undefined || v === "") return "";
        var n = Number(v);
        if (isNaN(n)) return String(v);
        return new Intl.NumberFormat("en-US", {
            style: "percent",
            maximumFractionDigits: 1
        }).format(n);
    }

    // -----------------------------------------------------------------------
    // CSS class helpers
    // -----------------------------------------------------------------------

    function valueStateForDelta(v) {
        var n = Number(v);
        if (isNaN(n)) return "None";
        if (n > 0) return "Success";
        if (n < 0) return "Error";
        return "None";
    }

    function valueClassForDelta(v) {
        var n = Number(v);
        if (isNaN(n) || n === 0) return "fluxNeutralValue fluxNumericCell";
        return n > 0 ? "fluxPositiveValue fluxNumericCell" : "fluxNegativeValue fluxNumericCell";
    }

    function valueStateForClassification(cls) {
        switch ((cls || "").toLowerCase()) {
            case "critical": return "Error";
            case "high":     return "Warning";
            case "medium":   return "Information";
            case "low":      return "Success";
            default:         return "None";
        }
    }

    function boolToYesNo(v) {
        return v ? "Yes" : "No";
    }

    return {
        formatMillions:              formatMillions,
        formatDelta:                 formatDelta,
        formatPercent:               formatPercent,
        currencyUSD:                 currencyUSD,
        numberInt:                   numberInt,
        number2:                     number2,
        percent1:                    percent1,
        valueStateForDelta:          valueStateForDelta,
        valueClassForDelta:          valueClassForDelta,
        valueStateForClassification: valueStateForClassification,
        boolToYesNo:                 boolToYesNo
    };
});