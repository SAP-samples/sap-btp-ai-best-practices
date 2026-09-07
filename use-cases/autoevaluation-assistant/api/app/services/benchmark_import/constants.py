"""Stable workbook, safety, and scoring contracts for benchmark imports."""

REQUIRED_COLUMNS = (
    "Data Estrazione",
    "Data Sottomissione",
    "ID Impresa",
    "Fatturato",
    "Nr Dipendenti",
    "Settore Operativo (NACE) 1",
    "Settore Operativo (NACE) 2",
    "Settore Operativo (NACE) 3",
    "Dimensione Azienda",
    "Classe",
    "Forma Giuridica",
    "Presenza Geografica",
    "Quotata",
    "Committente Contratti Pubblici",
    "Adesione Codice di  Autodisciplina",
    "ID Questionario",
    "Stato Questionario",
    "Assessment Score",
    "Dimensione",
    "Dimensione Score",
    "ID Domanda",
    "Sezione",
    "Domanda",
    "Domanda Gestita",
    "Stato Validazione",
    "Livello",
    "ID Risposta",
    "Testo risposta",
    "Valore Risposta",
    "Opzionale",
)
"""Required source columns in their exact workbook order."""

OPTIONAL_COMPANY_NAME_COLUMN = "Nome Impresa"
"""Only forward-compatible source column accepted in addition to the contract."""

ESTRAZIONE_SHEET = "Estrazione"
"""Worksheet containing source benchmark response rows."""

SCORING_VERSION = "assessment-v1"
"""Version label for the shared deterministic maturity score formula."""

MAX_SOURCE_BYTES = 25 * 1024 * 1024
"""Maximum accepted compressed XLSX source size (25 MiB)."""

MAX_ZIP_MEMBERS = 1_000
"""Maximum number of members accepted in one XLSX ZIP container."""

MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
"""Maximum combined uncompressed size accepted from an XLSX container."""

MAX_COMPRESSION_RATIO = 100.0
"""Maximum per-member compression ratio accepted from an XLSX container."""

MAX_SAMPLED_ERRORS = 50
"""Maximum blocking row errors serialized in one validation summary."""

MAX_WARNING_SAMPLES = 5
"""Maximum representative values serialized per aggregate warning."""
