#!/bin/bash
# Deployment script for Apex Automotive Services
# Run from new_version/ directory

set -e

echo "Deploying Apex Automotive Services..."

# Generate a secure API key if not provided
if [ -z "$API_KEY" ]; then
    API_KEY=$(openssl rand -hex 32)
    echo "Generated API Key: $API_KEY"
    echo "Save this key for local development!"
fi

# Deploy to Cloud Foundry
echo "Deploying to Cloud Foundry..."
cf push --var api_key="$API_KEY"

echo ""
echo "Deployment complete!"
echo ""
echo "Applications deployed:"
cf apps | grep apex

echo ""
echo "To view logs:"
echo "  cf logs apex-api --recent"
echo "  cf logs apex-ui --recent"
