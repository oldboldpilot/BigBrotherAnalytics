#!/bin/bash
# BigBrotherAnalytics - Graceful Shutdown Script

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
echo -e "${BLUE}  BigBrotherAnalytics - Shutdown${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Function to stop a service
stop_service() {
    local name=$1
    local pidfile=$2
    local process_pattern=$3

    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}🛑 Stopping $name (PID: $pid)...${NC}"
            kill -TERM "$pid" 2>/dev/null || true

            # Wait up to 5 seconds for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo -e "   ${GREEN}✓${NC} $name stopped gracefully"
                    rm -f "$pidfile"
                    return 0
                fi
                sleep 0.5
            done

            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "   ${YELLOW}⚠${NC} Forcing $name to stop..."
                kill -KILL "$pid" 2>/dev/null || true
                rm -f "$pidfile"
            fi
        else
            echo -e "${YELLOW}⚠️  $name PID file exists but process not running${NC}"
            rm -f "$pidfile"
        fi
    else
        # Try to find by pattern
        local pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo -e "${YELLOW}🛑 Found $name processes without PID file: $pids${NC}"
            echo "$pids" | xargs kill -TERM 2>/dev/null || true
            sleep 1
            echo -e "   ${GREEN}✓${NC} $name processes stopped"
        else
            echo -e "${GREEN}✓${NC} $name not running"
        fi
    fi
}

# Stop services
stop_service "Dashboard" "logs/dashboard.pid" "streamlit.*dashboard/app.py"
stop_service "Trading Engine" "logs/bigbrother.pid" "build/bin/bigbrother"

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
