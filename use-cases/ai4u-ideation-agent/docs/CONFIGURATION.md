# Configuration Reference

All configuration is via environment variables. See [`api/.env.example`](../api/.env.example) for a ready-to-copy template.

## Required Services

These must be set for the application to start.

### XSUAA Authentication

| Variable | Description | Example |
|----------|-------------|---------|
| `XSUAA_URL` | XSUAA tenant URL | `https://mysubdomain.authentication.eu10.hana.ondemand.com` |
| `XSUAA_CLIENT_ID` | OAuth2 client ID | From `cf service-key` |
| `XSUAA_CLIENT_SECRET` | OAuth2 client secret | From `cf service-key` |

On Cloud Foundry, these can alternatively be read from `VCAP_SERVICES` (auto-bound).

### SAP AI Core / Gen AI Hub

| Variable | Description | Default |
|----------|-------------|---------|
| `AICORE_AUTH_URL` | AI Core auth endpoint | — |
| `AICORE_CLIENT_ID` | AI Core client ID | — |
| `AICORE_CLIENT_SECRET` | AI Core client secret | — |
| `AICORE_BASE_URL` | AI Core API base URL | — |
| `AICORE_RESOURCE_GROUP` | Resource group name | `default` |

The `sap-ai-sdk-gen` SDK auto-discovers orchestration deployments from these credentials — no separate orchestration URL needed.

### HANA Cloud

| Variable | Description | Default |
|----------|-------------|---------|
| `HANA_HOST` | HANA Cloud hostname | — |
| `HANA_PORT` | HANA Cloud port | `443` |
| `HANA_USER` | Database user | — |
| `HANA_PASSWORD` | Database password | — |
| `HANA_ENCRYPT` | Use TLS | `true` |
| `HANA_SCHEMA` | Schema to use | — |

### Object Store

| Variable | Description | Default |
|----------|-------------|---------|
| `OBJECT_STORE_ACCESS_KEY_ID` | S3 access key | — |
| `OBJECT_STORE_SECRET_ACCESS_KEY` | S3 secret key | — |
| `OBJECT_STORE_BUCKET` | Bucket name | — |
| `OBJECT_STORE_REGION` | AWS region | `eu-central-1` |
| `OBJECT_STORE_HOST` | Custom S3 endpoint (non-AWS) | — |

### Admin Access

| Variable | Description | Example |
|----------|-------------|---------|
| `ADMIN_EMAILS` | Comma-separated admin emails | `admin@example.com,lead@example.com` |

## Optional Features

These are disabled by default and activate when their env vars are set.

### CDC (External User OTP Login)

Enabled when `CDC_API_KEY` is set. Provides email-based OTP login for users without XSUAA accounts.

| Variable | Description | Default |
|----------|-------------|---------|
| `CDC_API_KEY` | SAP CDC API key | — (disabled) |
| `CDC_DATA_CENTER` | CDC data center | `us1.gigya.com` |
| `CDC_USER_KEY` | CDC admin user key | — |
| `CDC_SECRET_KEY` | CDC admin secret | — |
| `CDC_PUBLIC_EMAIL_DOMAINS` | Domains treated as personal email | `gmail.com,hotmail.com,...` |

**Frontend vars** (set on the UI app):

| Variable | Description |
|----------|-------------|
| `VITE_CDC_API_KEY` | Same CDC API key (enables OTP UI) |
| `VITE_CDC_BASE_URL` | CDC accounts URL |

### CSE (Company Search Engine)

SAP-internal company profiling. Enabled when both `CSE_URL` and `CSE_KEY` are set.

| Variable | Description | Default |
|----------|-------------|---------|
| `CSE_URL` | CSE service endpoint | — (disabled) |
| `CSE_KEY` | CSE API key | — |
| `CSE_HTTP_TIMEOUT_SECONDS` | Request timeout | `30` |

### Perplexity Research

AI-powered market research via Perplexity Sonar models deployed on AI Core. Enabled when `IDEATION_PERPLEXITY_SKILL_ENABLED=true`.

| Variable | Description | Default |
|----------|-------------|---------|
| `IDEATION_PERPLEXITY_SKILL_ENABLED` | Enable Perplexity research | `false` |
| `IDEATION_PERPLEXITY_SONAR_PRO_MODEL` | Model name | `sonar-pro` |
| `IDEATION_PERPLEXITY_MAX_TOKENS` | Max response tokens | `1800` |

### LinkedIn Research

Hiring intelligence from public LinkedIn job listings. Enabled when `IDEATION_LINKEDIN_SKILL_ENABLED=true`.

| Variable | Description | Default |
|----------|-------------|---------|
| `IDEATION_LINKEDIN_SKILL_ENABLED` | Enable LinkedIn research | `false` |

## Optional Tuning

Override only if you need different behavior from the defaults.

### LLM Model & Temperature

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_INTERVIEW_TEMPERATURE` | Interview question generation | `0.5` |
| `AGENT_FINAL_TEMPERATURE` | Final recommendation | `0.2` |
| `REIMAGINE_LLM_MODEL` | Model for strategic reframe | `gpt-5.4` |
| `REIMAGINE_TEMPERATURE` | Reimagine creativity | `0.7` |
| `PROVOCATEUR_LLM_MODEL` | Model for provocateur | `gpt-5.4` |
| `PROVOCATEUR_TEMPERATURE` | Provocateur challenge level | `0.4` |
| `CATALOG_MATCH_MODEL` | Model for catalog matching | `gemini-2.5-flash` |

## Application Env Vars

Set in `manifest.yaml` (non-secret, deployment-specific):

| Variable | App | Description |
|----------|-----|-------------|
| `APP_ENV` | API | `production` in CF |
| `ALLOWED_ORIGIN` | API | Frontend URL for CORS |
| `API_BASE_URL` | API | Public API URL |
| `VITE_API_BASE_URL` | UI | Backend URL |
| `VITE_APP_HOST` | UI | Frontend hostname |
| `VITE_XSUAA_URL` | UI | XSUAA endpoint for login redirect |
| `VITE_XSUAA_CLIENT_ID` | UI | XSUAA client ID for PKCE flow |
| `VITE_XSUAA_IDP_ORIGIN` | UI | Identity-provider origin for SAP Login; defaults to `sap.default` when unset |

Set `VITE_XSUAA_IDP_ORIGIN` to your organization's identity-provider origin to direct SAP Login to that provider. Set it explicitly to an empty string (`""` in `manifest.yaml`) to omit the login hint and let XSUAA present its configured identity-provider selection. When the variable is absent, SAP Login uses `sap.default`.

### Version Check (Update Notifications)

Checks for newer versions and shows a notification to all users when an update is available. Enabled by default.

The version check request includes an anonymous installation ID and aggregate usage counts (total conversations, users, submissions — no PII). This allows tracking active installations and version distribution across all deployments. Disable entirely with `UPDATE_CHECK_ENABLED=false`.

| Variable | Description | Default |
|----------|-------------|---------|
| `UPDATE_CHECK_ENABLED` | Enable/disable upstream version checks + anonymous telemetry | `true` |
| `UPDATE_CHECK_URL` | URL to fetch the upstream VERSION file | SAP-hosted telemetry endpoint |
| `INSTALLATION_ID` | Explicit installation identifier (auto-generated UUID if not set) | *(auto)* |

Set `UPDATE_CHECK_ENABLED=false` in air-gapped environments or to opt out of anonymous usage reporting.

The `INSTALLATION_ID` is auto-generated (UUID4) and persisted in HANA across restages. Only override this for the SAP-hosted instance (`sap-hosted-primary`).

## Feature Flag Summary

| Feature | Gate | Default |
|---------|------|---------|
| XSUAA Auth | Always on | Required |
| CDC OTP | `CDC_API_KEY` presence | Disabled |
| CSE | `CSE_URL` + `CSE_KEY` presence | Disabled |
| Perplexity | `IDEATION_PERPLEXITY_SKILL_ENABLED=true` | Disabled |
| LinkedIn | `IDEATION_LINKEDIN_SKILL_ENABLED=true` | Disabled |
| Catalog Matching | Always on (uses AI Core) | Enabled |
| File Upload | Always on (uses Object Store) | Enabled |
| Update Notifications | `UPDATE_CHECK_ENABLED` | Enabled |
| Telemetry Receiver | `INSTALLATION_ID=sap-hosted-primary` | SAP-hosted only |
