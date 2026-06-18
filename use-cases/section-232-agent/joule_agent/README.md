# Section 232 Joule Capability

This folder contains the Joule SAPDAS design-time artifacts for the
Section 232 eligibility assistant. It no longer deploys a Python LangGraph or
A2A service. Joule calls the existing FastAPI backend directly through a BTP
destination.

## Structure

```text
joule_agent/
  da.sapdas.yaml
  section_232_capability/
    capability.sapdas.yaml
    scenarios/
    functions/
```

`da.sapdas.yaml` registers `section_232_capability/`. The capability defines
the `SECTION_232_API` system alias, which resolves to the `Section-232-Agent`
BTP destination.

## Destination

Create or update a BTP HTTP destination named `Section-232-Agent` in the Joule
account/subaccount where this assistant runs. The destination name is
case-sensitive.

```text
URL: https://section-232-agent-api.cfapps.eu10-004.hana.ondemand.com
Proxy Type: Internet
Authentication: NoAuthentication
Additional property: URL.headers.X-API-Key=<current API key>
```

The API key is intentionally not stored in this repository. If the root
deployment script rotates the API key, update the destination manually.

## Capability Surface

The direct SAPDAS capability provides scenarios for:

- API/ruleset overview
- Item search
- Item details and stored eligibility analysis
- Single item analysis submission
- Batch item analysis submission
- Classification job status
- Eligible HTS code lookup
- Direct HTS and metal-weight Section 232 classification
- Section 232 source document listing

## Validate

Run from the repository root:

```bash
joule lint joule_agent
joule compile joule_agent /tmp/section-232-joule-compile
```

`joule compile` may require a valid Joule login, depending on local CLI state.

## Deploy

After logging in to the target Joule tenant:

```bash
joule deploy -c -n "section_232_agent_assistant" joule_agent
joule launch "section_232_agent_assistant"
```

Representative utterances:

- `Find item 5778800`
- `Show details for mm:1001`
- `Start analysis for mm:1001`
- `Check job status <job_id>`
- `Assess HTS 8481.80.30.90 with 1200 grams steel`
- `What Section 232 source documents are loaded?`
