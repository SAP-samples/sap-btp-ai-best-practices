#!/usr/bin/env bash
# IBP-Agent.command — double-click to start the local extraction agent
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill anything on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null; sleep 0.5

# Find Python in the virtual environment
PYTHON=""
for candidate in \
    "$DIR/backend/.venv/bin/python" \
    "$HOME/.pyenv/shims/python3" \
    "/usr/local/bin/python3" \
    "/opt/homebrew/bin/python3" \
    "$(which python3 2>/dev/null)"; do
    if [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display alert "Python not found" message "Please run setup.sh first."'
    exit 1
fi

# Install dependencies if needed
"$PYTHON" -c "import flask, flask_cors, requests, dotenv, playwright" 2>/dev/null || \
    "$PYTHON" -m pip install flask flask-cors requests python-dotenv playwright -q

# Start Flask agent in background, log to /tmp/ibp-agent.log
cd "$DIR"
"$PYTHON" local_agent.py > /tmp/ibp-agent.log 2>&1 &
AGENT_PID=$!

# Wait up to 8s for Flask to be ready
echo "Starting IBP Agent (pid $AGENT_PID)..."
for i in $(seq 1 16); do
    sleep 0.5
    if curl -s http://localhost:5001/api/mode > /dev/null 2>&1; then
        echo "IBP Agent is running."
        break
    fi
done

# Open the CF UI (override with CF_UI_URL env var)
open "${CF_UI_URL:-https://<your-cf-ui-host>}"

# Show a notification
osascript -e 'display notification "IBP Agent is running. You can now use Extract Now in the browser." with title "IBP Agent Started"' 2>/dev/null || true

echo "Agent running in background (pid $AGENT_PID). Log: /tmp/ibp-agent.log"
echo "To stop: kill $AGENT_PID"
