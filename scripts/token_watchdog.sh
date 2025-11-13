#!/bin/bash
# Token Watchdog - Restarts bot when token is manually refreshed
# Run this in the background: ./scripts/token_watchdog.sh &

TOKEN_FILE="configs/schwab_tokens.json"
BOT_PID_FILE="/tmp/bigbrother.pid"
LOG_FILE="logs/token_watchdog.log"

echo "[$(date)] Token watchdog started" >> "$LOG_FILE"

# Get initial token modification time
LAST_MOD=$(stat -c %Y "$TOKEN_FILE" 2>/dev/null || echo 0)

while true; do
    sleep 30  # Check every 30 seconds

    CURRENT_MOD=$(stat -c %Y "$TOKEN_FILE" 2>/dev/null || echo 0)

    if [ "$CURRENT_MOD" != "$LAST_MOD" ]; then
        echo "[$(date)] Token file changed - restarting bot" >> "$LOG_FILE"

        # Kill old bot
        pkill -TERM bigbrother
        sleep 2

        # Start new bot
        ./build/bin/bigbrother > logs/bigbrother.log 2>&1 &
        NEW_PID=$!
        echo $NEW_PID > "$BOT_PID_FILE"

        echo "[$(date)] Bot restarted with PID: $NEW_PID" >> "$LOG_FILE"

        LAST_MOD=$CURRENT_MOD
    fi
done
