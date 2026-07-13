#!/bin/bash

# Sales Forecast Dashboard - Deployment Script
# This script deploys both the API and UI to Cloud Foundry

set -e

echo "Generating secure API key..."
API_KEY=$(openssl rand -hex 32)

echo "Deploying to Cloud Foundry..."
cf push --var api_key="$API_KEY"

echo ""
echo "Deployment complete!"
echo ""
echo "API Key has been set. Store it securely if needed for debugging:"
echo "API_KEY=$API_KEY"
