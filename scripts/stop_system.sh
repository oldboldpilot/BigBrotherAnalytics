#!/bin/bash
# BigBrotherAnalytics - System Shutdown
# Stops trading engine and dashboard gracefully

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  BigBrotherAnalytics - System Shutdown${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Change to project directory
cd "$(dirname "$0")/.." || exit 1

# Stop trading engine
if [ -f logs/bigbrother.pid ]; then
    PID=$(cat logs/bigbrother.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${BLUE}🛑 Stopping trading engine (PID: $PID)...${NC}"
        kill "$PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${RED}   Force killing engine...${NC}"
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm logs/bigbrother.pid
        echo -e "${GREEN}   ✅ Engine stopped${NC}"
    else
        echo -e "${BLUE}   Engine not running${NC}"
        rm logs/bigbrother.pid
    fi
else
    # Try pkill as fallback
    if pkill -f "^./build/bin/bigbrother" 2>/dev/null; then
        echo -e "${GREEN}✅ Trading engine stopped${NC}"
    else
        echo -e "${BLUE}   Trading engine not running${NC}"
    fi
fi

# Stop dashboard
if [ -f logs/dashboard.pid ]; then
    PID=$(cat logs/dashboard.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${BLUE}🛑 Stopping dashboard (PID: $PID)...${NC}"
        kill "$PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${RED}   Force killing dashboard...${NC}"
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm logs/dashboard.pid
        echo -e "${GREEN}   ✅ Dashboard stopped${NC}"
    else
        echo -e "${BLUE}   Dashboard not running${NC}"
        rm logs/dashboard.pid
    fi
else
    # Try pkill as fallback
    if pkill -f "streamlit.*dashboard/app.py" 2>/dev/null; then
        echo -e "${GREEN}✅ Dashboard stopped${NC}"
    else
        echo -e "${BLUE}   Dashboard not running${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ System shutdown complete${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
