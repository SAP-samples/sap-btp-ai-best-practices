# API FastAPI - R&D Variance Analysis

## Descripción

API REST creada con FastAPI para consumir la funcionalidad de análisis de varianzas R&D de la aplicación Streamlit.

## Instalación

Las dependencias ya están incluidas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Ejecución

### Modo desarrollo (con auto-reload)

```bash
cd backend
python api.py
```

O usando uvicorn directamente:

```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Modo producción

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Endpoints Disponibles

### 1. Root / Health Check
```
GET /
```
Retorna información general de la API y endpoints disponibles.

**Respuesta:**
```json
{
  "message": "R&D Variance Analysis API",
  "status": "running",
  "version": "1.0.0",
  "endpoints": {
    "/analyze": "POST - Ejecutar análisis de varianzas a nivel documento",
    "/wbs-analysis": "POST - Ejecutar análisis de varianzas por WBS",
    "/health": "GET - Verificar estado de la API"
  }
}
```

### 2. Health Check
```
GET /health
```
Verifica el estado de la API y disponibilidad de archivos de datos.

**Respuesta:**
```json
{
  "status": "healthy",
  "data_files_available": true,
  "file_24": "/path/to/ACDOCU_YTD Q3 24...",
  "file_25": "/path/to/ACDOCU_YTD Q3 25..."
}
```

### 3. Análisis de Varianzas (Nivel Documento)
```
POST /analyze
```

**Request Body:**
```json
{
  "materiality_threshold": 1000,
  "run_llm_analysis": true
}
```

**Parámetros:**
- `materiality_threshold` (float, default: 1000): Umbral de materialidad en dólares
- `run_llm_analysis` (bool, default: true): Ejecutar análisis con IA

**Respuesta:**
```json
{
  "success": true,
  "message": "Análisis completado exitosamente. 1234 varianzas encontradas.",
  "data": {
    "document_variances": [...],
    "total_variances": 1234,
    "materiality_threshold": 1000,
    "summary": [...],
    "classifications": {
      "critical": [...],
      "high": [...],
      "medium": [...],
      "low": [...],
      "counts": {
        "critical": 10,
        "high": 50,
        "medium": 100,
        "low": 1074
      }
    },
    "patterns": {...},
    "insights_summary": "...",
    "llm_explanation": "...",
    "text_analysis": {...},
    "drilldown_results": [...],
    "escalations": [...]
  }
}
```

### 4. Análisis WBS
```
POST /wbs-analysis
```

**Request Body:**
```json
{
  "materiality_threshold": 1000,
  "run_llm_analysis": true
}
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Análisis WBS completado. 567 combinaciones WBS/GL encontradas.",
  "data": {
    "wbs_summary": [...],
    "program_summary": [...],
    "critical_items": [...],
    "insights": "...",
    "llm_explanation": "...",
    "aligned_data": [...],
    "materiality_threshold": 1000,
    "key_metrics": {
      "total_net_variance": 1234567.89,
      "total_abs_variance": 2345678.90,
      "unique_wbs_elements": 45,
      "programs_affected": 12
    }
  }
}
```

### 5. Configuración
```
GET /config
```
Retorna la configuración actual de la API.

**Respuesta:**
```json
{
  "data_files": {
    "q3_2024": "/path/to/file_24.xlsx",
    "q3_2025": "/path/to/file_25.xlsx"
  },
  "default_settings": {
    "materiality_threshold": 1000,
    "run_llm_analysis": true
  }
}
```

## Documentación Interactiva

Una vez que la API esté corriendo, puedes acceder a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Ejemplos de Uso

### cURL

```bash
# Análisis de varianzas
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"materiality_threshold": 5000, "run_llm_analysis": true}'

# Análisis WBS
curl -X POST http://localhost:8000/wbs-analysis \
  -H "Content-Type: application/json" \
  -d '{"materiality_threshold": 5000, "run_llm_analysis": true}'
```

### Python (requests)

```python
import requests

# Análisis de varianzas
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "materiality_threshold": 5000,
        "run_llm_analysis": True
    }
)

data = response.json()
print(f"Total varianzas: {data['data']['total_variances']}")
```

### JavaScript (fetch)

```javascript
// Análisis de varianzas
fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    materiality_threshold: 5000,
    run_llm_analysis: true
  })
})
  .then(response => response.json())
  .then(data => console.log('Total varianzas:', data.data.total_variances));
```

## CORS

La API está configurada para aceptar peticiones de cualquier origen (`allow_origins=["*"]`). Para producción, se recomienda configurar los orígenes permitidos específicamente:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Notas

- La API procesa archivos Excel grandes, por lo que las primeras peticiones pueden tomar varios segundos
- Se recomienda implementar caché para peticiones repetidas en producción
- Los análisis con IA (`run_llm_analysis=true`) requieren más tiempo de procesamiento
- Asegúrate de tener configuradas las variables de entorno necesarias en el archivo `.env` para el acceso a SAP Gen AI Hub
