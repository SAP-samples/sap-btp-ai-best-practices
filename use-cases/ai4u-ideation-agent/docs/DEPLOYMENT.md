# Deployment Guide

Deploy the AI4U Ideation Agent to SAP BTP Cloud Foundry.

Clone the source as shown in the [README](../README.md#quick-start). Run all deployment commands from `sap-btp-ai-best-practices/use-cases/ai4u-ideation-agent`.

## Prerequisites

### 1. BTP Entitlements

Ensure your subaccount has these service entitlements assigned (BTP Cockpit → Entitlements → Configure Entitlements):

| Service | Plan | Notes |
|---------|------|-------|
| SAP AI Core | `extended` | Required for Gen AI Hub access |
| SAP HANA Cloud | `hana` | |
| Object Store Service | `s3-standard` | |
| Authorization and Trust Management (XSUAA) | `application` | |
| Cloud Foundry Runtime | ≥ 3 GB memory quota | 2 GB API + 1 GB UI |

> ⚠️ Missing entitlements are the most common deployment blocker. The BETA setup wizard will fail at service creation if a plan isn't assigned to your subaccount.

### 2. SAP AI Core (Gen AI Hub)

- Provision an AI Core instance with Gen AI Hub enabled
- Deploy these models: **GPT-5.4** and **Gemini 2.5 Flash**
- Optional: **Sonar Pro** (enables the Perplexity research feature)
- [AI Core setup guide](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/initial-setup)
- [Model deployment tutorial](https://developers.sap.com/tutorials/ai-core-generative-ai)

### 3. SAP HANA Cloud

- Provision a HANA Cloud instance
- You'll need **DBADMIN access** to HANA Cloud (via SQL Console in HANA Cloud Central) to create the application's technical user and schema
- The BETA setup wizard will show you the exact SQL, or see the manual setup section below
- [HANA Cloud setup guide](https://help.sap.com/docs/hana-cloud/sap-hana-cloud-administration-guide/provision-sap-hana-cloud)

### 4. Cloud Foundry CLI

- Install the [CF CLI](https://docs.cloudfoundry.org/cf-cli/install-go-cli.html)
- Use Node.js 22.12+ within Node 22 for local frontend development and builds. Cloud Foundry selects Node 22 using the frontend package's `engines.node` setting.

---

## Quick Deployment (BETA, ~5 minutes)

```bash
# 1. Log in to your CF landscape
cf login -a https://api.cf.<region>.hana.ondemand.com --sso

# 2. Run the BETA setup wizard (creates services, generates configs, deploys)
./setup.sh
```

The BETA setup wizard handles everything interactively: detects your environment, creates XSUAA and Object Store services, extracts credentials, generates all config files, and deploys.

**Subsequent deploys** (code updates):

```bash
git pull
./deploy.sh
```

---

## How It Works

On Cloud Foundry, most credentials are **auto-discovered** from service bindings (VCAP_SERVICES):

| Service | Credentials | Source |
|---------|------------|--------|
| XSUAA | URL, client ID, secret | Auto-discovered from VCAP |
| AI Core | Auth URL, client ID, secret, base URL | Auto-discovered from VCAP |
| Object Store | Access key, secret, bucket, region | Auto-discovered from VCAP |
| HANA Cloud | Host, user, password, schema | **Explicit env vars** (technical DDL user) |

The frontend (`ui/`) gets all `VITE_*` values from the manifest's environment variables at build time. Do **not** create a `ui/.env.production` file — it would override the manifest values.

All database tables auto-create on first access. No migration step needed.

---

## Manual Setup (Reference)

If you prefer manual control instead of the BETA setup wizard (`setup.sh`), follow the steps below.

### 1. Log In and Choose Deployment Names

Provision AI Core and HANA Cloud first, as described in [Prerequisites](#prerequisites). Their service instances must be available in the CF space where you deploy the agent.

```bash
cf login -a https://api.cf.<region>.hana.ondemand.com --sso
cf target -o <your-org> -s <your-space>
```

Choose a unique application prefix, such as `yourcompany-ideation`, and use it consistently for application routes and service names. Replace `<your-prefix>` in the commands below with that prefix. Use your actual CF region, including any suffix such as `eu10-005`, and ensure your chosen routes are available.

### 2. Prepare XSUAA Configuration

```bash
cp xs-security.json.template xs-security.json
```

The tracked `xs-security.json` is a starter for the shared development environment. For your own environment, copy the template as shown above, then replace `((xsappname))` with a unique XSUAA application name, `((app_prefix))` with your chosen prefix, and `((cf_region))` with your CF region. The redirect URI must match the UI route you will use in `manifest.yaml`.

Validate the JSON before creating the service:

```bash
python3 -m json.tool xs-security.json > /dev/null
```

### 3. Create Services and Read the XSUAA Key

```bash
# XSUAA
cf create-service xsuaa application <your-prefix>-xsuaa -c xs-security.json

# Object Store
cf create-service objectstore s3-standard <your-prefix>-objectstore

# Inspect provisioning status
cf service <your-prefix>-xsuaa
cf service <your-prefix>-objectstore
```

Repeat the status checks until each service reports successful creation. Resolve provisioning failures before creating keys or deploying. If you reuse an existing XSUAA instance, apply your redirect configuration with `cf update-service <your-xsuaa-instance> -c xs-security.json` and wait for the update to succeed.

Create and read the XSUAA service key:

```bash
cf create-service-key <your-prefix>-xsuaa <your-prefix>-xsuaa-key
cf service-key <your-prefix>-xsuaa <your-prefix>-xsuaa-key
```

Record `clientid`, `identityzone`, and `url` from the key for the UI configuration. Keep the key's secret values private; the backend obtains its XSUAA credentials from the service binding.

### 4. Create the HANA Technical User

Run this SQL in HANA Cloud Central using an administrator account:

```sql
CREATE USER AI4U_APP PASSWORD "<strong-password>" NO FORCE_FIRST_PASSWORD_CHANGE;
CREATE SCHEMA AI4U_IDEATION;
GRANT ALL ON SCHEMA AI4U_IDEATION TO AI4U_APP;
```

Use your own user and schema names if needed. Record the HANA SQL endpoint, technical username, password, and schema for the production environment file.

### 5. Complete Deployment Configuration

The tracked `manifest.yaml` is a starter for the shared development environment. For your own environment, copy the template and configure your routes and service bindings:

```bash
cp manifest.yaml.template manifest.yaml
(umask 077; cp api/.env.production.example api/.env.production)
```

Fill in `manifest.yaml` using the template's placeholder reference:

| Input | Value |
|-------|-------|
| `((app_prefix))`, `((cf_region))` | The same prefix and region used in `xs-security.json` |
| `((xsuaa_instance))`, `((objectstore_instance))` | The successfully provisioned service names |
| `((aicore_instance))`, `((hana_instance))` | Your actual AI Core and HANA service instance names in the target space |
| `((xsuaa_client_id))`, `((xsuaa_subdomain))` | `clientid` and `identityzone` from the XSUAA key |
| `((cf_region_base))` | Base region without its landscape suffix, such as `eu10` for `eu10-005` |
| `((cdc_api_key))`, `((cdc_base_url))` | Your CDC values, or empty strings when CDC is unused |

Set `VITE_XSUAA_URL` to the actual `url` from the XSUAA key rather than relying on a reconstructed hostname. For a corporate identity provider, configure `VITE_XSUAA_IDP_ORIGIN` as described in the [authentication settings](CONFIGURATION.md#application-env-vars).

When CDC is unused, set both `VITE_CDC_API_KEY` and `VITE_CDC_BASE_URL` to `""` in the YAML. Leave the backend CDC entries commented in `api/.env.production`. When enabling CDC, configure both the UI values and the backend values in [Optional Features](#cdc-external-user-otp-login).

Fill in `HANA_HOST`, `HANA_USER`, `HANA_PASSWORD`, and `HANA_SCHEMA` in `api/.env.production`, retain TLS with `HANA_ENCRYPT=true`, and set `ADMIN_EMAILS` if you need administrator access. Use one `NAME=value` assignment per line, with matching quotes around values when needed. Do not put backend secrets in the UI environment.

`manifest.yaml` and `xs-security.json` remain tracked as starters for the shared development environment. Production credentials in `api/.env.production` are git-ignored. Keep the JSON and YAML syntactically valid and replace every `((...))` placeholder before deployment. This local check rejects unresolved placeholders:

```bash
python3 - <<'PY'
from pathlib import Path
import re

for name in ("xs-security.json", "manifest.yaml", "api/.env.production"):
    if re.search(r"\(\([^\n]*?\)\)", Path(name).read_text()):
        raise SystemExit(f"Resolve all template placeholders in {name}")
print("All template placeholders resolved.")
PY
```

### 6. Deploy

```bash
./deploy.sh
```

The script parses app names from `manifest.yaml`, pushes both apps, sets env vars from `api/.env.production`, and restages.

---

## Optional Features

### CDC (External User OTP Login)

Add to `api/.env.production`:
```env
CDC_API_KEY=<your CDC API key>
CDC_DATA_CENTER=us1.gigya.com
CDC_USER_KEY=<your user key>
CDC_SECRET_KEY=<your secret key>
```

### Perplexity Research

Requires a Sonar Pro model deployed in AI Core. Add to `api/.env.production`:
```env
IDEATION_PERPLEXITY_SKILL_ENABLED=true
```

### LinkedIn Research

Add to `api/.env.production`:
```env
IDEATION_LINKEDIN_SKILL_ENABLED=true
```

---

## Troubleshooting

### App crashes on startup
- Check logs: `cf logs <app-name>-api --recent`
- Verify env vars: `cf env <app-name>-api`

### HANA connection fails
- Verify HANA instance is running in BTP Cockpit
- Check user/password/schema are correct
- Ensure SSL is enabled (`HANA_ENCRYPT=true`)

### AI Core calls fail
- Verify AI Core instance is bound in `manifest.yaml` services section
- Check that Gen AI Hub deployments exist in your resource group
- Confirm model names match: `gpt-5.4`, `gemini-2.5-flash`

### XSUAA token validation fails
- Check XSUAA instance is bound in `manifest.yaml`
- Verify redirect URIs in `xs-security.json` match your app route

### Service creation fails
- Check entitlements are assigned to your subaccount (see Prerequisites)
- Verify you're targeting the correct org/space: `cf target`

### Verify deployment

```bash
# Health check
curl https://<your-api-route>/api/health

# Agent card (A2A protocol)
curl https://<your-api-route>/api/ideation-agent/.well-known/agent-card.json
```

Visit `https://<your-ui-route>` and log in via XSUAA.
