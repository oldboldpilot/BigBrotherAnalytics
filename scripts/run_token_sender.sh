#!/bin/bash
#
# Token Refresh Sender Launcher
# Runs the token refresh sender using uv
#
# Usage:
#   ./scripts/run_token_sender.sh
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Ensure logs directory exists
mkdir -p logs

# Run with uv
echo "Starting Token Refresh Sender..."
echo "================================"
echo "Press Ctrl+C to stop"
echo ""

exec uv run python scripts/token_refresh_sender.py
