#!/usr/bin/env python3
"""
Token Refresh Service - Refreshes Schwab OAuth tokens and pushes to C++ bot via socket

This service runs in the background and:
1. Refreshes OAuth tokens every 25 minutes
2. Sends updated tokens to the C++ trading bot via Unix domain socket
3. Logs all refresh operations

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

# Configuration
SOCKET_PATH = "/tmp/bigbrother_token.sock"
TOKEN_FILE = "configs/schwab_tokens.json"
APP_CONFIG_FILE = "configs/schwab_app_config.yaml"
LOG_FILE = "logs/token_refresh.log"
REFRESH_INTERVAL = 25 * 60  # 25 minutes in seconds

class TokenRefreshService:
    """Service that refreshes tokens and communicates with C++ bot"""

    def __init__(self):
        self.running = True
        self.socket_path = SOCKET_PATH

        # Load app credentials
        with open(APP_CONFIG_FILE) as f:
            config = yaml.safe_load(f)

        self.client_id = config['app_key']
        self.client_secret = config['app_secret']

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        print(log_msg.strip())

        with open(LOG_FILE, "a") as f:
            f.write(log_msg)

    def refresh_token(self):
        """Refresh OAuth token using refresh_token grant"""
        try:
            # Load current token
            with open(TOKEN_FILE) as f:
                data = json.load(f)

            refresh_token = data['token']['refresh_token']

            # Make refresh request
            url = "https://api.schwabapi.com/v1/oauth/token"
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }

            auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers = {
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            self.log("Refreshing OAuth token...")
            response = requests.post(url, data=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                token_data = response.json()

                # Update token file
                data['creation_timestamp'] = int(time.time())
                data['token']['access_token'] = token_data['access_token']
                data['token']['refresh_token'] = token_data.get('refresh_token', refresh_token)
                data['token']['expires_in'] = token_data['expires_in']
                data['token']['expires_at'] = int(time.time()) + token_data['expires_in']

                with open(TOKEN_FILE, "w") as f:
                    json.dump(data, f, indent=4)

                self.log(f"✅ Token refreshed successfully (expires in {token_data['expires_in']//60} minutes)")

                return {
                    'access_token': data['token']['access_token'],
                    'refresh_token': data['token']['refresh_token'],
                    'expires_at': data['token']['expires_at']
                }
            else:
                self.log(f"❌ Token refresh failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.log(f"❌ Exception during token refresh: {e}")
            return None

    def send_token_to_bot(self, token_data):
        """Send updated token to C++ bot via Unix socket"""
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
                self.log("✅ Token successfully sent to trading bot")
                return True
            else:
                self.log(f"⚠️  Bot responded with: {ack}")
                return False

        except FileNotFoundError:
            self.log("⚠️  Bot socket not found - bot may not be running")
            return False
        except Exception as e:
            self.log(f"❌ Failed to send token to bot: {e}")
            return False

    def run(self):
        """Main service loop"""
        self.log("=" * 70)
        self.log("Token Refresh Service Started")
        self.log("=" * 70)
        self.log(f"Refresh interval: {REFRESH_INTERVAL//60} minutes")
        self.log(f"Socket path: {self.socket_path}")
        self.log(f"Token file: {TOKEN_FILE}")

        while self.running:
            # Refresh token
            token_data = self.refresh_token()

            if token_data:
                # Send to bot if running
                self.send_token_to_bot(token_data)

            # Sleep for 25 minutes (or until interrupted)
            self.log(f"Next refresh in {REFRESH_INTERVAL//60} minutes...")

            for _ in range(REFRESH_INTERVAL):
                if not self.running:
                    break
                time.sleep(1)

        self.log("Token Refresh Service stopped")

    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        self.log("Received shutdown signal")
        self.running = False

if __name__ == "__main__":
    service = TokenRefreshService()
    service.run()
