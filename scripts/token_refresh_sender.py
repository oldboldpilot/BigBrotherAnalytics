#!/usr/bin/env python3
"""
Token Refresh Sender - Refreshes Schwab OAuth tokens and sends to C++ trading bot

This service runs in the background and:
1. Reads Schwab OAuth tokens from configs/schwab_tokens.json
2. Refreshes OAuth tokens every 25 minutes using Schwab's refresh_token grant
3. Updates the token file with new tokens
4. Sends updated tokens to the C++ trading bot via Unix domain socket
5. Handles socket disconnections gracefully with retry logic
6. Logs all refresh operations

Compatible with running alongside the dashboard.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-13
"""

import json
import time
import socket
import os
import signal
import sys
from pathlib import Path
import yaml
import requests
import base64
from datetime import datetime
import logging
from typing import Optional, Dict, Any

# Configuration
SOCKET_PATH = "/tmp/bigbrother_token.sock"
TOKEN_FILE = "configs/schwab_tokens.json"
APP_CONFIG_FILE = "configs/schwab_app_config.yaml"
LOG_FILE = "logs/token_refresh_sender.log"
REFRESH_INTERVAL = 25 * 60  # 25 minutes in seconds
MAX_SOCKET_RETRIES = 3
SOCKET_RETRY_DELAY = 5  # seconds


class TokenRefreshSender:
    """Service that refreshes tokens and communicates with C++ bot via socket"""

    def __init__(self):
        self.running = True
        self.socket_path = SOCKET_PATH

        # Setup logging
        self._setup_logging()

        # Load app credentials
        self.config_path = Path(APP_CONFIG_FILE)
        if not self.config_path.exists():
            self.log_error(f"App config file not found: {APP_CONFIG_FILE}")
            raise FileNotFoundError(f"Missing config file: {APP_CONFIG_FILE}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        self.client_id = config['app_key']
        self.client_secret = config['app_secret']

        # Verify token file exists
        self.token_path = Path(TOKEN_FILE)
        if not self.token_path.exists():
            self.log_error(f"Token file not found: {TOKEN_FILE}")
            raise FileNotFoundError(f"Missing token file: {TOKEN_FILE}")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        self.log_info("=" * 70)
        self.log_info("Token Refresh Sender Initialized")
        self.log_info("=" * 70)

    def _setup_logging(self):
        """Setup logging to both file and console"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warn(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def refresh_token(self) -> Optional[Dict[str, Any]]:
        """
        Refresh OAuth token using refresh_token grant

        Returns:
            Dict with access_token, refresh_token, expires_at on success
            None on failure
        """
        try:
            # Load current token
            with open(self.token_path) as f:
                data = json.load(f)

            if 'token' not in data or 'refresh_token' not in data['token']:
                self.log_error("Invalid token file structure - missing refresh_token")
                return None

            refresh_token = data['token']['refresh_token']

            # Make refresh request to Schwab API
            url = "https://api.schwabapi.com/v1/oauth/token"
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }

            # Prepare Basic Authentication header
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')

            headers = {
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            self.log_info("Refreshing OAuth token...")
            response = requests.post(url, data=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                token_data = response.json()

                # Update token file
                data['creation_timestamp'] = int(time.time())
                data['token']['access_token'] = token_data['access_token']
                data['token']['refresh_token'] = token_data.get('refresh_token', refresh_token)
                data['token']['expires_in'] = token_data['expires_in']
                data['token']['expires_at'] = int(time.time()) + token_data['expires_in']

                # Preserve other fields if they exist
                if 'token_type' in token_data:
                    data['token']['token_type'] = token_data['token_type']
                if 'scope' in token_data:
                    data['token']['scope'] = token_data['scope']

                # Write updated tokens to file
                with open(self.token_path, "w") as f:
                    json.dump(data, f, indent=4)

                expires_minutes = token_data['expires_in'] // 60
                self.log_info(f"Token refreshed successfully (expires in {expires_minutes} minutes)")

                return {
                    'access_token': data['token']['access_token'],
                    'refresh_token': data['token']['refresh_token'],
                    'expires_at': data['token']['expires_at']
                }
            else:
                self.log_error(f"Token refresh failed: HTTP {response.status_code}")
                self.log_error(f"Response: {response.text}")
                return None

        except FileNotFoundError:
            self.log_error(f"Token file not found: {self.token_path}")
            return None
        except json.JSONDecodeError as e:
            self.log_error(f"Failed to parse token JSON: {e}")
            return None
        except requests.exceptions.RequestException as e:
            self.log_error(f"Network error during token refresh: {e}")
            return None
        except Exception as e:
            self.log_error(f"Unexpected error during token refresh: {e}")
            return None

    def send_token_to_bot(self, token_data: Dict[str, Any]) -> bool:
        """
        Send updated token to C++ bot via Unix domain socket

        Args:
            token_data: Dict containing access_token, refresh_token, expires_at

        Returns:
            True if token was successfully sent, False otherwise
        """
        for attempt in range(1, MAX_SOCKET_RETRIES + 1):
            try:
                # Create socket client
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)

                # Connect to C++ bot's socket
                sock.connect(self.socket_path)

                # Send token data as JSON
                message = json.dumps(token_data).encode('utf-8')
                sock.sendall(message)

                # Wait for acknowledgment
                ack = sock.recv(1024).decode('utf-8')
                sock.close()

                if ack == "OK":
                    self.log_info("Token successfully sent to trading bot")
                    return True
                else:
                    self.log_warn(f"Bot responded with unexpected acknowledgment: {ack}")
                    return False

            except FileNotFoundError:
                self.log_warn("Bot socket not found - trading bot may not be running")
                return False
            except ConnectionRefusedError:
                self.log_warn(f"Connection refused (attempt {attempt}/{MAX_SOCKET_RETRIES}) - bot may be starting up")
                if attempt < MAX_SOCKET_RETRIES:
                    time.sleep(SOCKET_RETRY_DELAY)
                    continue
                return False
            except socket.timeout:
                self.log_error(f"Socket timeout (attempt {attempt}/{MAX_SOCKET_RETRIES})")
                if attempt < MAX_SOCKET_RETRIES:
                    time.sleep(SOCKET_RETRY_DELAY)
                    continue
                return False
            except Exception as e:
                self.log_error(f"Failed to send token to bot (attempt {attempt}/{MAX_SOCKET_RETRIES}): {e}")
                if attempt < MAX_SOCKET_RETRIES:
                    time.sleep(SOCKET_RETRY_DELAY)
                    continue
                return False

        return False

    def run(self):
        """Main service loop - refreshes tokens every 25 minutes"""
        self.log_info("=" * 70)
        self.log_info("Token Refresh Sender Started")
        self.log_info("=" * 70)
        self.log_info(f"Refresh interval: {REFRESH_INTERVAL//60} minutes")
        self.log_info(f"Socket path: {self.socket_path}")
        self.log_info(f"Token file: {TOKEN_FILE}")
        self.log_info(f"App config: {APP_CONFIG_FILE}")
        self.log_info("")

        iteration = 0

        while self.running:
            iteration += 1
            self.log_info(f"--- Refresh Cycle {iteration} ---")

            # Refresh token
            token_data = self.refresh_token()

            if token_data:
                # Send to bot if running (non-blocking - continues even if bot isn't running)
                self.send_token_to_bot(token_data)
            else:
                self.log_error("Token refresh failed - will retry in next cycle")

            # Sleep for 25 minutes (or until interrupted)
            if self.running:
                next_refresh = datetime.now().timestamp() + REFRESH_INTERVAL
                next_refresh_time = datetime.fromtimestamp(next_refresh).strftime("%Y-%m-%d %H:%M:%S")
                self.log_info(f"Next refresh at {next_refresh_time} ({REFRESH_INTERVAL//60} minutes)")

                # Sleep in small increments to allow for graceful shutdown
                for _ in range(REFRESH_INTERVAL):
                    if not self.running:
                        break
                    time.sleep(1)

        self.log_info("=" * 70)
        self.log_info("Token Refresh Sender Stopped")
        self.log_info("=" * 70)

    def _shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        self.log_info("")
        self.log_info(f"Received shutdown signal ({signum})")
        self.running = False


def main():
    """Main entry point"""
    try:
        service = TokenRefreshSender()
        service.run()
        return 0
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
