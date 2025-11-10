#!/usr/bin/env python3
"""
Schwab OAuth Token Acquisition - Robust Implementation

Handles Schwab's strict 30-second timeout by running local callback server.
"""

import yaml
import secrets
import hashlib
import base64
import requests
import json
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Global variables for captured data
auth_code_captured = None
stop_server = threading.Event()

class CallbackHandler(BaseHTTPRequestHandler):
    """Handles OAuth callback redirect"""

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_GET(self):
        global auth_code_captured

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if 'code' in params:
            auth_code_captured = params['code'][0]

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            response = '<html><body><h1 style="color:green">Success!</h1><p>Authorization complete. Close this window.</p></body></html>'
            self.wfile.write(response.encode())

            stop_server.set()
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Error</h1></body></html>')

def main():
    print("Schwab OAuth Token Acquisition")
    print("=" * 60)

    # Load credentials
    config_path = Path(__file__).parent.parent / "configs" / "api_keys.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    schwab = config['schwab']
    app_key = schwab['app_secret']  # Schwab uses app_secret as client_id
    app_secret = schwab['client_secret']
    redirect_uri = schwab['callback_url']

    print(f"App Key: {app_key[:20]}...")
    print(f"Redirect: {redirect_uri}")
    print()

    # Generate PKCE
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('=')
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip('=')

    print(f"Code Verifier: {verifier}")
    print()

    # Start callback server
    server = HTTPServer(('127.0.0.1', 8182), CallbackHandler)
    server_thread = threading.Thread(target=lambda: server.serve_forever(), daemon=True)
    server_thread.start()
    print("✅ Callback server started on port 8182")
    print()

    # Build authorization URL
    auth_url = (
        f"https://api.schwabapi.com/v1/oauth/authorize"
        f"?client_id={app_key}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
    )

    print("Opening browser for authorization...")
    print("URL:", auth_url)
    print()

    # Open browser
    try:
        webbrowser.open(auth_url)
    except:
        print("Could not open browser automatically. Please open manually.")

    print("Waiting for callback (max 60 seconds)...")
    stop_server.wait(timeout=60)
    server.shutdown()

    if not auth_code_captured:
        print("❌ No authorization code received")
        return 1

    print(f"✅ Code captured: {auth_code_captured[:30]}...")
    print()

    # Exchange for tokens IMMEDIATELY
    print("Exchanging code for tokens...")

    token_url = "https://api.schwabapi.com/v1/oauth/token"

    # Basic auth header
    auth_str = f"{app_key}:{app_secret}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "authorization_code",
        "code": auth_code_captured,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier
    }

    response = requests.post(token_url, headers=headers, data=data, timeout=30)

    if response.status_code == 200:
        tokens = response.json()

        print("🎉 SUCCESS! Tokens obtained!")
        print()
        print(f"Access Token: {tokens['access_token'][:30]}...")
        print(f"Refresh Token: {tokens['refresh_token'][:30]}...")
        print(f"Expires: {tokens.get('expires_in', 1800)} seconds")
        print()

        # Save tokens
        tokens_file = Path(__file__).parent.parent / "data" / "schwab_tokens.json"
        tokens_file.parent.mkdir(parents=True, exist_ok=True)

        token_data = {
            "access_token": tokens['access_token'],
            "refresh_token": tokens['refresh_token'],
            "token_type": tokens.get('token_type', 'Bearer'),
            "expires_in": tokens.get('expires_in', 1800),
            "obtained_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=tokens.get('expires_in', 1800))).isoformat(),
            "account_id": schwab['account_id']
        }

        with open(tokens_file, 'w') as f:
            json.dump(token_data, f, indent=2)

        print(f"💾 Tokens saved to: {tokens_file}")
        print()
        print("✅ Schwab API is now authenticated and ready!")
        return 0
    else:
        print(f"❌ Token exchange failed: {response.status_code}")
        print(f"Error: {response.text}")
        return 1

if __name__ == "__main__":
    exit(main())
