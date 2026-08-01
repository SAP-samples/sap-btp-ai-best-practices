/**
 * API Service for Financial Flux Analysis
 * Handles all backend communication with the FastAPI server.
 * Data source: SAP HANA view v_delec_fin
 */
sap.ui.define([], function () {
    "use strict";

    // En desarrollo: Vite proxy redirige /api → localhost:8000
    // En producción (CF): llamadas directas al backend
    var IS_DEV = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    var BASE_URL = IS_DEV ? "" : "https://flux-backend.cfapps.eu10-005.hana.ondemand.com";

    var ApiService = {

        /**
         * Fetch raw financial data with optional filters.
         * @param {object} params - { fiscal_year, fiscal_period, company_code }
         * @returns {Promise}
         */
        getFinancialData: function (params) {
            var url = BASE_URL + "/api/financial-data";
            var queryParams = [];
            if (params.fiscal_year)   queryParams.push("fiscal_year="   + encodeURIComponent(params.fiscal_year));
            if (params.fiscal_period) queryParams.push("fiscal_period=" + encodeURIComponent(params.fiscal_period));
            if (params.company_code)  queryParams.push("company_code="  + encodeURIComponent(params.company_code));
            if (queryParams.length) url += "?" + queryParams.join("&");
            return fetch(url).then(function (r) { return r.json(); });
        },

        /**
         * Fetch distinct fiscal periods and years.
         * @param {object} params - { fiscal_year, company_code }
         * @returns {Promise}
         */
        getAvailablePeriods: function (params) {
            var url = BASE_URL + "/api/available-periods";
            var queryParams = [];
            if (params && params.fiscal_year)  queryParams.push("fiscal_year="  + encodeURIComponent(params.fiscal_year));
            if (params && params.company_code) queryParams.push("company_code=" + encodeURIComponent(params.company_code));
            if (queryParams.length) url += "?" + queryParams.join("&");
            return fetch(url).then(function (r) { return r.json(); });
        },

        /**
         * Fetch distinct company codes.
         * @param {object} params - { fiscal_year }
         * @returns {Promise}
         */
        getCompanyCodes: function (params) {
            var url = BASE_URL + "/api/company-codes";
            var queryParams = [];
            if (params && params.fiscal_year) queryParams.push("fiscal_year=" + encodeURIComponent(params.fiscal_year));
            if (queryParams.length) url += "?" + queryParams.join("&");
            return fetch(url).then(function (r) { return r.json(); });
        },

        /**
         * Fetch period-over-period variance comparison.
         * @param {object} params - { current_period, previous_period, current_year, previous_year, company_code }
         * @returns {Promise}
         */
        getComparison: function (params) {
            var url = BASE_URL + "/api/comparison";
            var queryParams = [];
            if (params.current_period)  queryParams.push("current_period="  + encodeURIComponent(params.current_period));
            if (params.previous_period) queryParams.push("previous_period=" + encodeURIComponent(params.previous_period));
            if (params.current_year)    queryParams.push("current_year="    + encodeURIComponent(params.current_year));
            if (params.previous_year)   queryParams.push("previous_year="   + encodeURIComponent(params.previous_year));
            if (params.company_code)    queryParams.push("company_code="    + encodeURIComponent(params.company_code));
            if (queryParams.length) url += "?" + queryParams.join("&");
            return fetch(url).then(function (r) { return r.json(); });
        },

        /**
         * Fetch grouped aggregation with optional period comparison.
         * group_by must be one of: "G/L Account", "Profit Center", "Cost Center", "Financial Statement Line Item"
         * @param {object} params - { group_by, fiscal_year, fiscal_period, previous_period, company_code }
         * @returns {Promise}
         */
        getGroupedAnalysis: function (params) {
            var url = BASE_URL + "/api/grouped-analysis";
            var queryParams = [];
            if (params.group_by)        queryParams.push("group_by="        + encodeURIComponent(params.group_by));
            if (params.fiscal_year)     queryParams.push("fiscal_year="     + encodeURIComponent(params.fiscal_year));
            if (params.fiscal_period)   queryParams.push("fiscal_period="   + encodeURIComponent(params.fiscal_period));
            if (params.previous_period) queryParams.push("previous_period=" + encodeURIComponent(params.previous_period));
            if (params.company_code)    queryParams.push("company_code="    + encodeURIComponent(params.company_code));
            if (queryParams.length) url += "?" + queryParams.join("&");
            return fetch(url).then(function (r) { return r.json(); });
        },

        /**
         * Run full variance analysis with AI explanation.
         * @param {object} params - { current_period, previous_period, fiscal_year, company_code }
         * @returns {Promise}
         */
        runAnalysis: function (params) {
            return fetch(BASE_URL + "/api/analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(params)
            }).then(function (r) { return r.json(); });
        },

        /**
         * Check API and HANA connectivity.
         * @returns {Promise}
         */
        getHealth: function () {
            return fetch(BASE_URL + "/health").then(function (r) { return r.json(); });
        },

        /**
         * Generate AI explanation for a specific account.
         * @param {object} params - { account_number, group_by, current_year, previous_year, company_code }
         * @returns {Promise}
         */
        getAIExplanation: function (params) {
            return fetch(BASE_URL + "/api/ai-explain", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    account_number: params.account_number,
                    group_by:       params.group_by       || "G/L Account",
                    current_year:   params.current_year,
                    previous_year:  params.previous_year,
                    company_code:   params.company_code   || null
                })
            }).then(function (r) { return r.json(); });
        },

        /**
         * Fetch raw line items for a specific account key.
         * @param {object} params - { group_by, key, current_year, previous_year, company_code, limit }
         * @returns {Promise}
         */
        getAccountLineItems: function (params) {
            var url = BASE_URL + "/api/account-line-items";
            var q = [];
            if (params.group_by)      q.push("group_by="      + encodeURIComponent(params.group_by));
            if (params.key)           q.push("key="           + encodeURIComponent(params.key));
            if (params.current_year)  q.push("current_year="  + encodeURIComponent(params.current_year));
            if (params.previous_year) q.push("previous_year=" + encodeURIComponent(params.previous_year));
            if (params.company_code)  q.push("company_code="  + encodeURIComponent(params.company_code));
            if (params.limit)         q.push("limit="         + encodeURIComponent(params.limit));
            if (q.length) url += "?" + q.join("&");
            return fetch(url).then(function (r) { return r.json(); });
        },

        /**
         * Send a chat message to the AI Chat module.
         * Maintains multi-turn conversation context via conversation_id.
         * @param {object} params - { message, conversation_id, context }
         *   context: { currentYear, previousYear, companyCode }
         * @returns {Promise<{ response, conversation_id, metadata }>}
         */
        sendChatMessage: function (params) {
            return fetch(BASE_URL + "/api/chat/message", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message:         params.message,
                    conversation_id: params.conversation_id || null,
                    context:         params.context         || {}
                })
            }).then(function (r) {
                if (!r.ok) {
                    return r.json().then(function (err) {
                        throw new Error(err.detail || "Chat request failed (" + r.status + ")");
                    });
                }
                return r.json();
            });
        }
    };

    return ApiService;
});