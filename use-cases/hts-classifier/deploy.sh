#!/bin/bash
#
# This script automates the deployment of the application to Cloud Foundry.
# It generates a secure, random API key and injects it into the deployment
# process, enhancing security and simplifying setup.
#

# --- Main Script ---

# 1. Generate a secure, random API key
# This creates a 32-byte (256-bit) random key and encodes it in hexadecimal.
API_KEY=$(openssl rand -hex 32)
echo "🔑 Generated temporary API key for this deployment."

# 2. Deploy the application using the generated key
# The `cf push` command reads the manifest.yaml and replaces the ((api_key))
# variable with the value we provide here.
#
# Note: You must be logged into Cloud Foundry before running this script.
echo "🚀 Deploying application..."
cf push --var api_key="$API_KEY"

echo "✅ Deployment finished." 