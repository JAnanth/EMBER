#!/usr/bin/env python3
"""
EMBER Web Application Launcher

Run this script to start the EMBER web application.
Usage: python run_app.py [--port PORT]
"""

import subprocess
import sys
import os
import argparse


def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'qiskit', 'plotly', 'pandas', 'numpy', 'sklearn']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])


def main():
    parser = argparse.ArgumentParser(description='EMBER Web Application Launcher')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the application on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()

    print("=" * 50)
    print("   EMBER - Quantum Lung Cancer Detection System")
    print("=" * 50)
    print()

    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    print("Dependencies OK")
    print()

    # Change to webapp directory
    webapp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp')
    os.chdir(webapp_dir)

    # Build streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(args.port),
        '--server.headless', 'true' if args.no_browser else 'false'
    ]

    print(f"Starting EMBER on port {args.port}...")
    print(f"Access the application at: http://localhost:{args.port}")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down EMBER...")
        sys.exit(0)


if __name__ == '__main__':
    main()
