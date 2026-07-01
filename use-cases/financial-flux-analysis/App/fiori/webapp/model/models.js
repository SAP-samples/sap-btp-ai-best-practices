sap.ui.define([
  "sap/ui/model/json/JSONModel",
  "sap/ui/Device"
], function (JSONModel, Device) {
  "use strict";

  function createDeviceModel() {
    var oModel = new JSONModel(Device);
    oModel.setDefaultBindingMode("OneWay");
    return oModel;
  }

  function createUIModel() {
    // UI/Config model: stores materiality, LLM toggle, loading flags, base URL, last messages
    var oData = {
      baseUrl: "https://account-backend.cfapps.eu01-canary.hana.ondemand.com",
      materiality_threshold: 1000,
      run_llm_analysis: true,
      busy: false,
      lastMessage: "",
      lastUpdatedAt: null
    };
    return new JSONModel(oData);
  }

  function createAnalysisModel() {
    // Mirrors /analyze response payload
    var oData = {
      success: null,
      message: "",
      error: null,
      total_variances: 0,
      materiality_threshold: 1000,
      // core results
      document_variances: [], // array of records
      summary: [],            // array of records (grouped by GL & CC)
      classifications: {
        critical: [],
        high: [],
        medium: [],
        low: [],
        counts: {
          critical: 0,
          high: 0,
          medium: 0,
          low: 0
        }
      },
      patterns: {},
      insights_summary: "",
      llm_explanation: "",
      text_analysis: {
        unique_descriptions: 0,
        total_with_text: 0,
        top_descriptions: {} // map description -> count
      },
      drilldown_results: [], // array of drilldown objects
      escalations: []        // array of escalation items
    };
    return new JSONModel(oData);
  }

  function createWBSModel() {
    // Mirrors /wbs-analysis response payload
    var oData = {
      success: null,
      message: "",
      error: null,
      materiality_threshold: 1000,
      wbs_summary: [],
      program_summary: [],
      critical_items: [],
      insights: "",
      llm_explanation: "",
      aligned_data: [],
      key_metrics: {
        total_net_variance: 0,
        total_abs_variance: 0,
        unique_wbs_elements: 0,
        programs_affected: 0
      }
    };
    return new JSONModel(oData);
  }

  return {
    createDeviceModel: createDeviceModel,
    createUIModel: createUIModel,
    createAnalysisModel: createAnalysisModel,
    createWBSModel: createWBSModel
  };
});
