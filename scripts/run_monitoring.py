#!/usr/bin/env python3
"""
Run the Ijon RAG monitoring dashboard.

Usage:
    python scripts/run_monitoring.py [--host HOST] [--port PORT]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.dashboard import run_dashboard


def main():
    parser = argparse.ArgumentParser(description="Run Ijon RAG monitoring dashboard")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    
    args = parser.parse_args()
    
    print(f"Starting monitoring dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    try:
        run_dashboard(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down monitoring dashboard")


if __name__ == "__main__":
    main()