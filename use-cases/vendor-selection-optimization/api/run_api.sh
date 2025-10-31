#!/bin/bash

# Procurement Assistant API Startup Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
MODE="development"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --production)
            MODE="production"
            shift
            ;;
        --help)
            echo "Usage: ./run_api.sh [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST       Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT       Port to bind to (default: 8000)"
            echo "  --workers N       Number of workers (default: 1)"
            echo "  --production      Run in production mode"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Display startup information
echo -e "${GREEN}Starting Procurement Assistant API${NC}"
echo -e "Mode: ${YELLOW}${MODE}${NC}"
echo -e "Host: ${YELLOW}${HOST}${NC}"
echo -e "Port: ${YELLOW}${PORT}${NC}"

if [ "$MODE" == "production" ]; then
    echo -e "Workers: ${YELLOW}${WORKERS}${NC}"
fi

echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: main.py not found. Please run this script from the api directory.${NC}"
    exit 1
fi

# Change to parent directory to ensure proper imports
cd ..

# Start the server
if [ "$MODE" == "development" ]; then
    echo -e "${GREEN}Starting development server with auto-reload...${NC}"
    echo -e "${YELLOW}API Documentation will be available at: http://localhost:${PORT}/docs${NC}"
    echo ""
    uvicorn api.main:app --reload --host "$HOST" --port "$PORT"
else
    echo -e "${GREEN}Starting production server...${NC}"
    echo -e "${YELLOW}API Documentation will be available at: http://localhost:${PORT}/docs${NC}"
    echo ""
    uvicorn api.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi