#!/usr/bin/env python3
"""Alternative startup script for the Procurement Assistant API"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import and run the app
import uvicorn
from api.main import app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Procurement Assistant API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print(f"Starting Procurement Assistant API")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Documentation: http://localhost:{args.port}/docs")
    print()
    
    if args.reload:
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)