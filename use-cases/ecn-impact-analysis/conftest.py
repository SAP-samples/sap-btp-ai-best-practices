"""Pytest session bootstrap for local SAP credential loading.

Loads gitignored ``api/.env`` at collection time so that:

* opt-in live tests (HANA integration round-trips and, in later phases, the
  SAP Generative AI Hub provider smoke test) can read ``HANA_*`` / ``AICORE_*``
  values, and
* the credential-coupled demo-router modules (``app.routers.agent_demo`` and
  ``app.routers.llm_demo``) can be imported, since ``gen_ai_hub`` eagerly
  constructs a proxy client on import.

No secrets are hardcoded here; values live only in the gitignored ``api/.env``.
Existing process environment variables win (``override=False``) so Cloud
Foundry / CI environments keep precedence over the local file.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load once at import; the repository root is this file's directory.
load_dotenv(Path(__file__).parent / "api" / ".env", override=False)
