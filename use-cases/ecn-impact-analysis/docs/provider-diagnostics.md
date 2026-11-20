# Provider diagnostics and model configuration

## Purpose

The ECN Impact API uses one text-generation setting, `LLM_MODEL`, whose
default is `gpt-5.4`. When another primary is configured, a provider timeout
triggers one request retry on `gpt-5.4`; the default does not retry itself.

The provider preflight checks OAuth and then resolves an exact running
deployment for the configured `LLM_MODEL`. It supports standard SAP AI Core
service-key URL shapes where the authentication URL is a host and the API base
already ends in `/v2`. The OAuth client credentials are submitted in the
`application/x-www-form-urlencoded` request body, matching the direct AI Core
client contract; the diagnostic check does not assume HTTP Basic client
authentication.

## Inputs and outputs

The preflight reads the existing `AICORE_*` service-key fields plus the three
model names. It returns only:

- a checked timestamp;
- safe availability status;
- configured model names;
- optional HTTP status, provider machine code, and request ID.

It never returns credentials, access tokens, endpoint URLs, headers, prompts,
response bodies, or exception messages.

`GET /api/provider-status` returns the last cached snapshot and never performs
network I/O. `POST /api/provider-status/check` explicitly refreshes the
snapshot.

The API also runs one preflight during startup. When OAuth or the configured
primary deployment is unavailable, deterministic ECN ingestion still commits,
but each new analysis job transitions from `QUEUED` to `BLOCKED` without
creating an attempt or calling the provider. A successful manual status check
atomically returns provider-blocked jobs to `QUEUED` and schedules them through
the normal bounded runtime. Historical `FAILED` jobs are not changed.

Provider generation errors use bounded categories:

- `PROVIDER_AUTH_FAILED`
- `PROVIDER_FORBIDDEN`
- `MODEL_NOT_AVAILABLE`
- `PROVIDER_RATE_LIMITED`
- `PROVIDER_TIMEOUT`
- `PROVIDER_SERVER_ERROR`
- `PROVIDER_RESPONSE_INVALID`

Rate limits and provider 5xx responses use bounded exponential backoff and
jitter. A timeout on the primary text-generation deployment retries the same
request once on `gpt-5.4`; a timeout on that fallback is terminal. A missing
primary deployment does not trigger fallback. Authentication, authorization,
missing deployment, and invalid-response failures are not retried.

## Related files

- `api/app/ecn/llm/diagnostics.py`
- `api/app/ecn/llm/preflight.py`
- `api/app/ecn/llm/provider.py`
- `api/app/routers/provider_status.py`
- `api/app/ecn/services/generation.py`
- `api/.env.example`
- `manifest.yaml`

## Testing

Run the focused tests with:

```bash
PYTHONPATH=api .venv/bin/python -m pytest \
  tests/unit/test_provider_status.py \
  tests/unit/test_impact_provider.py \
  tests/unit/test_analysis_attempts.py \
  tests/unit/test_impact_validation.py -q
```

A live check is opt-in and should print only the serialized safe status model.
Do not print environment variables, response bodies, or exception text.

Run the API from the `api/` directory and let `python-dotenv` load `api/.env`:

```bash
cd api
../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Do not run `source api/.env` or `set -a && source api/.env` first. An `.env`
file is dotenv syntax, not a shell script. Shell expansion can silently change
credentials containing characters meaningful to the shell and cause an OAuth
`401` even though the service-key values in the file are correct. The
application loads the file without shell expansion and keeps credentials out
of diagnostic output.
