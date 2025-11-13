#!/bin/bash
# BigBrotherAnalytics - Truly Non-Blocking Startup Script
# Launches all services in detached background processes
# Returns immediately without blocking the calling process

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  BigBrotherAnalytics - Non-Blocking Startup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create logs directory
mkdir -p logs

# Function to start process in detached background
start_detached() {
    local name=$1
    local command=$2
    local logfile=$3
    local pidfile=$4

    # Check if already running
    if [ -f "$pidfile" ] && kill -0 $(cat "$pidfile") 2>/dev/null; then
        echo -e "${YELLOW}⚠️  $name already running (PID: $(cat $pidfile))${NC}"
        return 0
    fi

    echo -e "${GREEN}🚀 Starting $name in background...${NC}"

    # Start in subshell, detached, with nohup, redirect output, and disown
    (setsid bash -c "$command" </dev/null >>"$logfile" 2>&1 &)

    # Give it a moment to start
    sleep 0.5

    # Find and save PID
    local pid=$(pgrep -f "$name" | head -1)
    if [ -n "$pid" ]; then
        echo "$pid" > "$pidfile"
        echo -e "   ${GREEN}✓${NC} Started (PID: $pid, log: $logfile)"
    else
        echo -e "   ${RED}✗${NC} Failed to start (check $logfile)"
        return 1
    fi
}

# Start Trading Engine
if [ -f "build/bin/bigbrother" ]; then
    start_detached \
        "bigbrother" \
        "$PROJECT_ROOT/build/bin/bigbrother" \
        "$PROJECT_ROOT/logs/bigbrother.log" \
        "$PROJECT_ROOT/logs/bigbrother.pid"
else
    echo -e "${YELLOW}⚠️  Trading engine not built (run: ninja -C build bigbrother)${NC}"
fi

# Start Dashboard
start_detached \
    "streamlit" \
    "cd $PROJECT_ROOT && uv run streamlit run dashboard/app.py --server.headless true --server.port 8501" \
    "$PROJECT_ROOT/logs/dashboard.log" \
    "$PROJECT_ROOT/logs/dashboard.pid"

echo ""
echo -e "${GREEN}✅ All services started in background!${NC}"
echo ""
echo -e "${BLUE}Access Points:${NC}"
echo "   📊 Dashboard:      http://localhost:8501"
echo "   🤖 Trading Engine: Check logs/bigbrother.log"
echo ""
echo -e "${BLUE}Monitor Logs:${NC}"
echo "   tail -f logs/bigbrother.log"
echo "   tail -f logs/dashboard.log"
echo ""
echo -e "${BLUE}Management:${NC}"
echo "   Stop all:     ./scripts/shutdown.sh"
echo "   Check status: ps aux | grep -E 'bigbrother|streamlit'"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Exit immediately - don't wait for background processes
exit 0
