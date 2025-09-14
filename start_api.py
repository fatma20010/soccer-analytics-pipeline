#!/usr/bin/env python3
"""
Startup script for Soccer Analytics API
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import flask_socketio
        import flask_cors
        import cv2
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements with:")
        print("pip install -r requirements-web.txt")
        return False

def main():
    print("🚀 Starting Soccer Analytics API Server...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("web_app.py").exists():
        print("❌ Error: web_app.py not found!")
        print("Please run this script from the soccer-analytics-pipeline directory")
        return 1
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check if src directory exists
    if not Path("src").exists():
        print("❌ Error: src directory not found!")
        print("Please ensure the soccer analytics source code is in the src/ directory")
        return 1
    
    print("✓ All checks passed!")
    print("\n🎯 Starting API server...")
    print("Frontend should connect to: http://localhost:5000")
    print("WebSocket endpoint: ws://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "web_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed with error code: {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
