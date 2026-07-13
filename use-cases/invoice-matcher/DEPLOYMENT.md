# Deployment Guide

## Prerequisites

### 1. Cloud Foundry CLI

Install the Cloud Foundry command-line tool:

```bash
# macOS (Homebrew)
brew install cloudfoundry/tap/cf-cli@8

# Windows (Chocolatey)
choco install cloudfoundry-cli

# Linux (Debian/Ubuntu)
wget -q -O - https://packages.cloudfoundry.org/debian/cli.cloudfoundry.org.key | sudo apt-key add -
echo "deb https://packages.cloudfoundry.org/debian stable main" | sudo tee /etc/apt/sources.list.d/cloudfoundry-cli.list
sudo apt-get update && sudo apt-get install cf8-cli
```

Verify installation:

```bash
cf version
```

### 2. SAP BTP Account Access

You need access to an SAP BTP subaccount with:

- A Cloud Foundry environment enabled
- An SAP AI Core service instance provisioned
- A service key created for that AI Core instance

### 3. SAP AI Core Service Key

Obtain your AI Core credentials from the BTP Cockpit:

1. Navigate to your subaccount in the SAP BTP Cockpit
2. Go to **Services > Instances and Subscriptions**
3. Find your **SAP AI Core** service instance
4. Click on the instance, then go to **Service Keys**
5. Create a new service key (or use an existing one)
6. Note down these values from the key:
   - `url` → this is your `AICORE_AUTH_URL`
   - `clientid` → this is your `AICORE_CLIENT_ID`
   - `clientsecret` → this is your `AICORE_CLIENT_SECRET`
   - `serviceurls.AI_API_URL` → this is your `AICORE_BASE_URL`
7. Determine your resource group name (usually `default` unless configured otherwise) → this is your `AICORE_RESOURCE_GROUP`

### 4. Node.js and Python (for local development only)

- Node.js 18+ (includes npm)
- Python 3.10+

---

## Step-by-Step Deployment

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd invoice-matcher
```

### Step 2: Create the Environment File

```bash
cp api/.env.example api/.env
```

Edit `api/.env` and fill in your SAP AI Core credentials:

```
AICORE_AUTH_URL=https://your-subdomain.authentication.eu10.hana.ondemand.com
AICORE_CLIENT_ID=sb-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx!bxxxxxx|aicore!bxxx
AICORE_CLIENT_SECRET=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AICORE_BASE_URL=https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com
AICORE_RESOURCE_GROUP=default
```

### Step 3: Log in to Cloud Foundry

```bash
cf login -a https://api.cf.eu10-005.hana.ondemand.com --sso
```

This opens a browser for SSO authentication. After logging in, select your org and space when prompted:

```
Select an org:
1. my-org

Select a space:
1. dev
2. prod
```

Verify you are targeting the correct space:

```bash
cf target
```

### Step 4: Deploy

Make the deploy script executable and run it:

```bash
chmod +x deploy.sh
./deploy.sh
```

The script will:

1. Read credentials from `api/.env`
2. Validate that all required variables are set
3. Push both applications (API and UI) to Cloud Foundry

Deployment takes approximately 3-5 minutes.

### Step 5: Verify Deployment

Check that both apps are running:

```bash
cf apps
```

You should see:

```
name                        requested state   processes   routes
invoice-matcher-api   started           web:1/1     invoice-matcher-api.cfapps.eu10-005.hana.ondemand.com
invoice-matcher-ui    started           web:1/1     invoice-matcher.cfapps.eu10-005.hana.ondemand.com
```

Open the UI in your browser:

```
https://invoice-matcher.cfapps.eu10-005.hana.ondemand.com
```

### Step 6: Test the Health Endpoint

```bash
curl https://invoice-matcher-api.cfapps.eu10-005.hana.ondemand.com/api/health
```

Expected response:

```json
{"status": "healthy", "timestamp": 1234567890.0, "service": "api", "version": null}
```

---

## Troubleshooting

### App fails to start

Check logs:

```bash
cf logs invoice-matcher-api --recent
cf logs invoice-matcher-ui --recent
```

### AI matching returns errors

- Verify AI Core credentials are correct in the deployed environment:
  ```bash
  cf env invoice-matcher-api
  ```
- Ensure the AI Core service has a deployment for the `gpt-4o` model
- Check that your resource group has access to orchestration scenarios

### CORS errors in browser

Ensure the `ALLOWED_ORIGIN` env var in the API app matches the UI's actual URL. Update `manifest.yaml` if your route differs.

### Redeployment

After making changes, simply run:

```bash
./deploy.sh
```

To redeploy only one app:

```bash
cf push invoice-matcher-api   # API only
cf push invoice-matcher-ui    # UI only
```

---

## Local Development

### Backend

```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API available at http://127.0.0.1:8000 (docs at /docs)

### Frontend

```bash
cd ui
npm install
cp .env.example .env   # Edit VITE_API_BASE_URL if needed
npm run dev
```

UI available at http://localhost:5173
