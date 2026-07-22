# Configuration

The repository contains examples only:

- `api/.env.example`
- `ui/.env.example`
- `vars.example.yml`
- `api/dox_client/schemas/service_key.json.example`

Never commit `.env`, `vars.yml`, service keys, uploaded documents, or generated run artifacts.

## Required runtime

The extraction flow requires SAP BTP Generative AI Hub credentials represented by the five `AICORE_*` variables. The default recommended profile uses `gemini-2.5-flash` and does not call SAP Document AI.

## Optional integrations

- Set `DOCAI_ENABLED=true` and provide a Document AI service key only when enabling Document AI scenarios.
- Set `S4_INTEGRATION_ENABLED=true` only after configuring either a direct S/4HANA connection or BTP Destination and Connectivity. With the default `false` value, extraction and PR payload review work without creating an S/4HANA business document.

## Security note

The API-key header is suitable for a prototype. For a production deployment, place the UI and API behind an SAP Application Router and XSUAA, and use service bindings instead of client secrets in manifest variables.
