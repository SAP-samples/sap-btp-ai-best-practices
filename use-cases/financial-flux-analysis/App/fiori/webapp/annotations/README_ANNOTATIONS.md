# Annotations and Fiori Elements Plan

Context and decision:
- Requirement asked to implement with Fiori Web Elements (Fiori Elements V4, annotation-driven) and avoid freestyle UI unless strictly necessary.
- FastAPI backend exposes REST JSON endpoints (no OData service, no $metadata).
- Fiori Elements requires OData V2/V4 services plus UI/Analytical annotations.
- Restriction: Do NOT modify backend FastAPI.

Decision:
- Implement initial frontend with SAPUI5 freestyle using Fiori floorplans (DynamicPage, ShellBar) and JSONModel consumption of REST, strictly documenting this choice.
- Provide a future plan to migrate to Fiori Elements by introducing an OData facade without changing FastAPI code (e.g., external gateway or proxy), and include draft annotations showing the intended UI.

Templates reused:
- Floorplans from samples/fiori/DynamicPage and samples/fiori/ShellBar were used as the baseline for page structure and global navigation (ShellBar).
- Naming conventions and project layout follow standard SAPUI5 app structure (webapp/, manifest.json, Component.js, views/controllers, i18n, util).

Streamlit → Fiori mapping:
- Document-Level Variances (main report):
  - Streamlit tables → DynamicPage content with sap.m.Table bound to analysis>/document_variances and analysis>/summary.
  - Filters: materiality_threshold (number input), run_llm_analysis (switch).
  - Actions: Run Analysis (POST /analyze), navigate to WBS.
  - KPIs: total variance, net variance, concentration ratio.
  - Classifications: critical table bound to analysis>/classifications/critical.
  - Insights, Text Analysis (top descriptions), LLM explanation, Drilldown, Escalations → dedicated collapsible panels.
- WBS Analysis:
  - Streamlit WBS sections → summary tables bound to wbs>/wbs_summary and wbs>/program_summary.
  - Critical WBS items, aligned detailed data, insights, LLM explanation, key metrics → panels and tables.
  - Action: Run WBS Analysis (POST /wbs-analysis), navigate back to Document-Level.

Endpoints consumed:
- POST /analyze
  - Request: { materiality_threshold: number, run_llm_analysis: boolean }
  - Response: see backend/README_API.md → data.document_variances[], data.summary[], data.classifications.*, data.patterns, data.insights_summary, data.llm_explanation, data.text_analysis, data.drilldown_results, data.escalations
- POST /wbs-analysis
  - Request: { materiality_threshold: number, run_llm_analysis: boolean }
  - Response: data.wbs_summary[], data.program_summary[], data.critical_items[], data.insights, data.llm_explanation, data.aligned_data[], data.key_metrics
- GET /health, GET /config for status and defaults.

Why freestyle now:
- Without OData and $metadata, Fiori Elements cannot be bootstrapped.
- Using JSONModel directly binds REST results, aligning with the no-backend-change restriction.
- All decision points and mapping are documented to allow future migration to Fiori Elements.

Plan to enable Fiori Elements (OData facade):
- Introduce a non-invasive OData V4 proxy layer that translates FastAPI REST to OData:
  - Option A: Deploy an API Management / API Gateway that exposes an OData endpoint and forwards to FastAPI.
  - Option B: Add a sidecar microservice (Node.js or CAP) that provides OData entities and delegates to FastAPI. This keeps FastAPI unchanged; sidecar lives separately.
- Once the OData facade exists, replace JSONModel with an ODataModel V4 in manifest.json and switch to Fiori Elements templates:
  - List Report for document_variances entity with SelectionFields for materiality_threshold and run_llm_analysis (triggered via action).
  - Analytical List Page (ALP) where KPIs and charts are built from summary and patterns data.
  - Object Page only if detailed single-variance navigation is required.

Draft annotations for future Fiori Elements (illustrative; not active):
- Target entity types (example): DocumentVariance, SummaryItem, DrilldownResult, EscalationItem, WBSSummary, ProgramSummary, WBSAlignedItem.
- Example OData V4 XML annotations (UI and Analytical):

```xml
<!-- UI annotations for DocumentVariance entity -->
<Annotations Target="svc.DocumentVariance">
  <Annotation Term="UI.LineItem">
    <Collection>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="GL_Account"/>
        <PropertyValue Property="Label" String="GL Account"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Cost_Center"/>
        <PropertyValue Property="Label" String="Cost Center"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Driver"/>
        <PropertyValue Property="Label" String="Driver"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Amount_Q3_24"/>
        <PropertyValue Property="Label" String="Amount Q3 24"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Amount_Q3_25"/>
        <PropertyValue Property="Label" String="Amount Q3 25"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Delta"/>
        <PropertyValue Property="Label" String="Delta"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Abs_Delta"/>
        <PropertyValue Property="Label" String="Abs Delta"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Percent_Change"/>
        <PropertyValue Property="Label" String="% Change"/>
      </Record>
    </Collection>
  </Annotation>

  <Annotation Term="UI.SelectionFields">
    <Collection>
      <!-- Facade-side virtual parameters for actions could be represented via function imports or bound actions -->
      <PropertyPath>MaterialityThreshold</PropertyPath>
      <PropertyPath>RunLLMAnalysis</PropertyPath>
    </Collection>
  </Annotation>
</Annotations>

<!-- Analytical annotations for SummaryItem entity -->
<Annotations Target="svc.SummaryItem">
  <Annotation Term="UI.LineItem">
    <Collection>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="GL_Account"/>
        <PropertyValue Property="Label" String="GL Account"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Cost_Center"/>
        <PropertyValue Property="Label" String="Cost Center"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Total_Delta"/>
        <PropertyValue Property="Label" String="Total Delta"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Total_Abs_Delta"/>
        <PropertyValue Property="Label" String="Total Abs Delta"/>
      </Record>
      <Record Type="UI.DataField">
        <PropertyValue Property="Value" Path="Document_Count"/>
        <PropertyValue Property="Label" String="Document Count"/>
      </Record>
    </Collection>
  </Annotation>

  <!-- Example of Chart annotation for ALP (requires proper Measure/Dimension definition in the entity) -->
  <Annotation Term="com.sap.vocabularies.UI.v1.Chart">
    <Record Type="com.sap.vocabularies.UI.v1.ChartDefinitionType">
      <PropertyValue Property="ChartType" EnumMember="com.sap.vocabularies.UI.v1.ChartType/Bar"/>
      <PropertyValue Property="Title" String="Variance by GL & Cost Center"/>
      <PropertyValue Property="Measures">
        <Collection>
          <Record Type="com.sap.vocabularies.UI.v1.MeasureDefinition">
            <PropertyValue Property="Measure" PropertyPath="Total_Delta"/>
            <PropertyValue Property="Role" EnumMember="com.sap.vocabularies.UI.v1.MeasureRoleType/Axis1"/>
            <PropertyValue Property="Title" String="Total Delta"/>
          </Record>
        </Collection>
      </PropertyValue>
      <PropertyValue Property="Dimensions">
        <Collection>
          <Record Type="com.sap.vocabularies.UI.v1.DimensionDefinition">
            <PropertyValue Property="Dimension" PropertyPath="GL_Account"/>
            <PropertyValue Property="Role" EnumMember="com.sap.vocabularies.UI.v1.DimensionRoleType/Category"/>
            <PropertyValue Property="Title" String="GL Account"/>
          </Record>
        </Collection>
      </PropertyValue>
    </Record>
  </Annotation>
</Annotations>

<!-- Similar annotations can be defined for WBS entities (WBSSummary, ProgramSummary, WBSAlignedItem) -->
```

Future manifest changes for Fiori Elements:
- Replace JSON models with sap.ui.model.odata.v4.ODataModel in manifest.json:
  - datasources → set OData service URL of the facade.
  - sap.ui5/models → default model pointing to OData V4.
- Pages to use:
  - Analytical List Page for the main variances report and WBS analysis.
  - Object Page if drill-down to a single item detail is required.
- Actions:
  - Bound action/function import to trigger analysis with parameters (materiality, runLLM).
  - Download could be handled by backend-generated media streams or by client-side export.

Suposiciones:
- OData entity sets will mirror REST payload shapes:
  - DocumentVariance
    - GL_Account, Cost_Center, Driver, Amount_Q3_24, Amount_Q3_25, Delta, Abs_Delta, Percent_Change
  - SummaryItem
    - GL_Account, Cost_Center, Total_Delta, Total_Abs_Delta, Document_Count
  - ClassificationCritical (subset of DocumentVariance)
  - DrilldownResult
    - gl_account, cost_center, driver, amount, drill_path (string or navigation), confidence, requires_escalation, escalation_reason
  - EscalationItem
    - GL_Account, Cost_Center, Driver, Delta, Reasons, Missing_Data
  - WBSSummary
    - WBS_Element, Total_Q3_24, Total_Q3_25, Net_Variance, Total_Abs_Variance, GL_Account_Count
  - ProgramSummary
    - Program, Total_Q3_24, Total_Q3_25, Net_Variance, Total_Abs_Variance, WBS_Count
  - WBSAlignedItem
    - WBS_Element, GL_Account, Cost_Center, Amount_Q3_24, Amount_Q3_25, Delta, Abs_Delta, Percent_Change

Notas:
- Al no usar Fiori Elements todavía, todas las vistas actuales están en XML + controladores UI5 con bindings a JSONModel.
- Los comentarios en views y controllers documentan el origen de datos y las decisiones de diseño.
- Este README sirve como puente para migrar a Fiori Elements sin tocar FastAPI, cumpliendo el estándar enterprise de ser annotation-driven en cuanto sea viable.
