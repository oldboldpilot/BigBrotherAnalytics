#!/bin/bash
# BigBrotherAnalytics - Non-Blocking System Startup
# Launches trading engine and dashboard in background processes

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  BigBrotherAnalytics - System Startup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Change to project directory
cd "$(dirname "$0")/.." || exit 1

# Check if processes are already running
if pgrep -f "bigbrother" > /dev/null; then
    echo -e "${YELLOW}⚠️  Trading engine already running${NC}"
else
    echo -e "${GREEN}🚀 Starting trading engine in background...${NC}"
    nohup ./build/bin/bigbrother > logs/bigbrother.log 2>&1 &
    ENGINE_PID=$!
    echo "   PID: $ENGINE_PID"
    echo "$ENGINE_PID" > logs/bigbrother.pid
fi

if pgrep -f "streamlit.*dashboard/app.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Dashboard already running${NC}"
else
    echo -e "${GREEN}📊 Starting dashboard in background...${NC}"
    nohup uv run streamlit run dashboard/app.py --server.headless true > logs/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    echo "   PID: $DASHBOARD_PID"
    echo "$DASHBOARD_PID" > logs/dashboard.pid
fi

echo ""
echo -e "${GREEN}✅ System started successfully!${NC}"
echo ""
echo -e "${BLUE}Access Points:${NC}"
echo "   Dashboard: http://localhost:8501"
echo "   Trading Engine: localhost (logs in logs/bigbrother.log)"
echo ""
echo -e "${BLUE}Management:${NC}"
echo "   View engine logs:    tail -f logs/bigbrother.log"
echo "   View dashboard logs: tail -f logs/dashboard.log"
echo "   Shutdown system:     ./scripts/stop_system.sh"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
