#!/usr/bin/env python3
"""
Schwab Token Refresh Service

Automatically refreshes Schwab OAuth tokens every 29 minutes.
This ensures the trading engine always has a valid access token.

The schwab-py library handles token refresh automatically,
so we just need to periodically call the API to trigger refresh.

Usage:
    python scripts/token_refresh_service.py
    # Or run in background:
    nohup python scripts/token_refresh_service.py > logs/token_refresh.log 2>&1 &
"""

import schwab
import time
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
TOKEN_FILE = Path('configs/schwab_tokens.json')
APP_KEY = '8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa'
APP_SECRET = 'PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT'
REFRESH_INTERVAL = 29 * 60  # 29 minutes in seconds

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logger.info("Shutdown signal received, stopping token refresh service...")
    running = False
    sys.exit(0)

def refresh_token():
    """Refresh the OAuth token using schwab-py library"""
    try:
        # Load client from token file - schwab-py auto-refreshes if needed
        client = schwab.auth.client_from_token_file(
            str(TOKEN_FILE),
            APP_KEY,
            APP_SECRET
        )

        # Make a simple API call to trigger token refresh if needed
        # This will automatically refresh the token and update the file
        response = client.get_account_numbers()

        if response.status_code == 200:
            logger.info(f"✅ Token refresh successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.debug(f"Account numbers endpoint returned: {response.status_code}")
            return True
        else:
            logger.warning(f"⚠️ Token refresh returned status {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ Token refresh failed: {e}")
        return False

def main():
    """Main service loop"""
    global running

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 70)
    logger.info("Schwab Token Refresh Service Starting")
    logger.info("=" * 70)
    logger.info(f"Token file: {TOKEN_FILE}")
    logger.info(f"Refresh interval: {REFRESH_INTERVAL // 60} minutes")
    logger.info(f"Next refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    iteration = 0

    while running:
        iteration += 1
        logger.info(f"🔄 Token refresh cycle #{iteration}")

        # Attempt token refresh
        success = refresh_token()

        if success:
            logger.info(f"✅ Cycle #{iteration} completed successfully")
        else:
            logger.error(f"❌ Cycle #{iteration} failed - will retry in {REFRESH_INTERVAL // 60} minutes")

        # Calculate next refresh time
        next_refresh = datetime.now().timestamp() + REFRESH_INTERVAL
        next_refresh_str = datetime.fromtimestamp(next_refresh).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"⏰ Next refresh at: {next_refresh_str}")
        logger.info("")

        # Sleep for 29 minutes, checking every second for shutdown signal
        for _ in range(REFRESH_INTERVAL):
            if not running:
                break
            time.sleep(1)

    logger.info("Token refresh service stopped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
