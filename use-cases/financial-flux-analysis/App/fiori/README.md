# R&D Variance Analysis - SAPUI5 (Fiori Floorplans) Frontend

Objetivo
- Reemplazar la app Streamlit (solo como especificación funcional) con un frontend SAPUI5 usando floorplans Fiori, consumiendo exclusivamente la API FastAPI provista.
- Respetar estructura y buenas prácticas de los templates Fiori provistos en samples/ (DynamicPage, ShellBar) y documentar claramente el mapeo funcional y las decisiones.
- Preparar un plan futuro para migración a Fiori Elements V4 (annotation-driven) sin modificar el backend.

Arquitectura y decisión (por qué Freestyle UI5 ahora)
- Fiori Elements V4 requiere OData V2/V4 y anotaciones. El backend es REST JSON (FastAPI) sin $metadata OData y está explícitamente fuera de alcance modificarlo.
- Se implementa Freestyle UI5 con floorplans Fiori (sap.f.DynamicPage, sap.f.ShellBar) y JSONModel para consumir REST.
- Se incluye fe-fiori/webapp/annotations/README_ANNOTATIONS.md con plan y ejemplos de anotaciones UI/Analytical para futura migración a Fiori Elements usando un façade OData externo (APIM/CAP sidecar), sin tocar FastAPI.

Estructura del Frontend (webapp/)
- index.html: bootstrap SAPUI5 desde CDN y carga del componente com.rnd.variance
- manifest.json: routing, modelos JSON (ui, analysis, wbs), rootView App
- Component.js: inicializa router y modelos
- i18n/i18n.properties: títulos, etiquetas, textos
- model/models.js: JSONModels: device, ui (config), analysis, wbs
- services/apiService.js: consumo FastAPI (GET /health, GET /config, POST /analyze, POST /wbs-analysis)
- util/formatter.js: formateo de moneda, número, porcentaje y value states
- view/App.view.xml + controller/App.controller.js: ShellBar + host de router, navegación Variances/WBS
- view/Variances.view.xml + controller/Variances.controller.js: vista principal con filtros, acciones, tablas, KPI y descargas CSV; POST /analyze
- view/WBS.view.xml + controller/WBS.controller.js: análisis WBS con secciones completas; POST /wbs-analysis
- annotations/README_ANNOTATIONS.md: plan de migración a Fiori Elements y borrador de anotaciones

Cómo ejecutar
1) Backend FastAPI
   - Requisitos: Python 3.10+ (ver backend/requirements.txt)
   - Iniciar API:
     cd backend
     uvicorn api:app --reload --host 0.0.0.0 --port 8000
   - Swagger: http://localhost:8000/docs
   - Health:  http://localhost:8000/health

2) Frontend SAPUI5
   - Abrir en el navegador el archivo:
     fe-fiori/webapp/index.html
   - Asegúrese de que el backend esté levantado (la app consume http://localhost:8000 por defecto).
   - Para cambiar el endpoint (p. ej. entorno productivo), editar en webapp/model/models.js la propiedad ui.baseUrl.

Correspondencia funcional Streamlit → Fiori
- Parámetros (sidebar Streamlit):
  - Materiality Threshold ($) → Input numérico en header (ui>/materiality_threshold)
  - Generate AI Explanation → Switch en header (ui>/run_llm_analysis)
- Acciones (botones):
  - Run Analysis → Variances: Button Emphasized → POST /analyze
  - WBS Analysis → WBS: Button Emphasized → POST /wbs-analysis
  - Navegación entre vistas: ShellBar menu items
- Vistas/Secciones Variances:
  - Document-Level Variances → Tabla analysis>/document_variances
  - Summary by GL & Cost Center → Tabla analysis>/summary
  - Classification Critical (Top 10%) → Tabla analysis>/classifications/critical
  - Insights/Patterns → Panel con analysis>/insights_summary
  - Text Analysis (top descriptions) → Panel; normalización a lista realizada en Variances.controller tras POST /analyze
  - LLM Explanation → Panel con FormattedText htmlText
  - Drill-down Results → Tabla analysis>/drilldown_results
  - Escalations → Tabla analysis>/escalations (+ MessageStrip si vacío)
  - Key Metrics → Panel con KPIs: total_variance, net_variance, concentration_ratio
  - Descargas CSV → Botones por sección (documentos, resumen, críticos)
- Vistas/Secciones WBS:
  - Summary by WBS → Tabla wbs>/wbs_summary
  - Summary by Program → Tabla wbs>/program_summary
  - Critical WBS Items → Tabla wbs>/critical_items
  - WBS Insights → Panel wbs>/insights
  - LLM Explanation → Panel con FormattedText htmlText
  - Detailed aligned data → Tabla wbs>/aligned_data
  - Key Metrics → Panel con KPIs de wbs>/key_metrics
  - Descargas CSV → Botones por sección (wbs summary, program summary, critical, aligned)

Consumo de API FastAPI (contratos)
- POST /analyze
  - Request: { "materiality_threshold": number, "run_llm_analysis": boolean }
  - Response: data con:
    - document_variances[] (GL_Account, Cost_Center, Driver, Amount_Q3_24, Amount_Q3_25, Delta, Abs_Delta, Percent_Change)
    - summary[] (GL_Account, Cost_Center, Total_Delta, Total_Abs_Delta, Document_Count)
    - classifications.{critical[], high[], medium[], low[], counts{...}}
    - patterns{...}, insights_summary, llm_explanation
    - text_analysis { unique_descriptions, total_with_text, top_descriptions }
      - Nota: top_descriptions llega como mapa; el controller normaliza a [{Description, Count}]
    - drilldown_results[]
    - escalations[]
- POST /wbs-analysis
  - Request: { "materiality_threshold": number, "run_llm_analysis": boolean }
  - Response: data con:
    - wbs_summary[], program_summary[], critical_items[], insights, llm_explanation
    - aligned_data[], key_metrics { total_net_variance, total_abs_variance, unique_wbs_elements, programs_affected }
- GET /health, GET /config: para status y defaults; baseUrl configurable en ui model.

Suposiciones realizadas
- Se asume que el backend devuelve las claves según backend/README_API.md y que CORS está permitido (api.py habilita allow_origins=["*"]).
- El cálculo/negocio reside 100% en el backend; el frontend no ejecuta cálculos de negocio.
- La exportación CSV se realiza en cliente (front) a partir del payload JSON (alineado a UX de Streamlit con botones de descarga).
- Accesos y tiempos: primeras ejecuciones pueden tardar por el tamaño de los Excel; se muestra BusyIndicator (ui>/busy).

Templates base reutilizados y adaptaciones
- Basado en samples/fiori/DynamicPage y samples/fiori/ShellBar:
  - DynamicPage como contenedor de cada vista, acorde a Fiori floorplans.
  - ShellBar para navegación global entre Variances y WBS.
- Convenciones:
  - Estructura estándar SAPUI5 (webapp/...) con manifest.json, Component.js, views/controllers, i18n, util, services.
  - Nombres de modelos: "ui", "analysis", "wbs".

Plan de migración a Fiori Elements (annotation-driven)
- Ver fe-fiori/webapp/annotations/README_ANNOTATIONS.md:
  - Introducir façade OData V4 (APIM o sidecar CAP/Node) sin modificar FastAPI.
  - Definir entidades OData que mapeen los payloads actuales.
  - Crear UI/Analytical annotations (UI.LineItem, UI.SelectionFields, UI.Chart, etc.).
  - Cambiar manifest para usar ODataModel V4 y plantillas Fiori Elements:
    - Analytical List Page (ALP) para reportes con KPIs y charts.
    - List Report/Object Page para detalle y navegación adicional.
  - Acciones parametrizadas como bound actions / function imports.

Notas de operación
- Cambiar endpoint: editar ui.baseUrl en model/models.js
- Depurar: usar devtools del navegador para ver POST y payloads (analyze, wbs-analysis).
- Seguridad: en producción, ajustar CORS en FastAPI y servir app desde un host/approuter corporativo.

Licencia
- Uso interno, material de referencia y prototipo enterprise-ready para evolución a Fiori Elements.
