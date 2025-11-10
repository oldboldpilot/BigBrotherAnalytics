#!/usr/bin/env python3
"""
Schwab OAuth Server - Automatically Captures Authorization Code

This script:
1. Starts a local web server on 127.0.0.1:8182
2. Generates and displays the authorization URL
3. Automatically captures the authorization code when redirected
4. Immediately exchanges it for tokens (within the 30-second limit!)

Usage:
    python3 scripts/schwab_oauth_server.py
"""

import yaml
import secrets
import hashlib
import base64
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import sys

# Global storage for captured code
captured_code = None
code_verifier_global = None
credentials_global = None
server_should_stop = False

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to capture OAuth callback"""

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

    def do_GET(self):
        """Handle GET request from OAuth redirect"""
        global captured_code, server_should_stop

        # Parse URL
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if 'code' in params:
            captured_code = params['code'][0]
            print(f"\n✅ Authorization code captured!")
            print(f"   Code: {captured_code[:30]}...")
            print()

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_response = """
                <html>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: green;">Authorization Successful!</h1>
                    <p>You can close this window now.</p>
                    <p>The token exchange is happening automatically...</p>
                </body>
                </html>
            """
            self.wfile.write(html_response.encode('utf-8'))

            # Signal server to stop
            server_should_stop = True
        else:
            # No code in URL
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Error: No authorization code received</h1></body></html>")

def load_credentials():
    """Load Schwab credentials"""
    config_file = Path(__file__).parent.parent / "configs" / "api_keys.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    schwab = config.get("schwab", {})
    return {
        "app_key": schwab.get("app_secret"),
        "client_secret": schwab.get("client_secret"),
        "callback_url": schwab.get("callback_url", "https://127.0.0.1:8182"),
        "account_id": schwab.get("account_id")
    }

def generate_pkce_pair():
    """Generate PKCE code verifier and challenge"""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

def exchange_for_tokens(auth_code, code_verifier, creds):
    """Exchange code for tokens"""
    token_url = "https://api.schwabapi.com/v1/oauth/token"

    # Basic Authentication
    auth_string = f"{creds['app_key']}:{creds['client_secret']}"
    auth_b64 = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {auth_b64}"
    }

    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": creds['callback_url'],
        "code_verifier": code_verifier
    }

    response = requests.post(token_url, headers=headers, data=data, timeout=30)
    return response

def save_tokens(tokens, creds):
    """Save tokens to file"""
    tokens_file = Path(__file__).parent.parent / "data" / "schwab_tokens.json"
    tokens_file.parent.mkdir(parents=True, exist_ok=True)

    token_data = {
        "access_token": tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "token_type": tokens.get("token_type", "Bearer"),
        "expires_in": tokens.get("expires_in", 1800),
        "scope": tokens.get("scope", ""),
        "account_id": creds['account_id'],
        "obtained_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(seconds=tokens.get("expires_in", 1800))).isoformat()
    }

    with open(tokens_file, 'w') as f:
        json.dump(token_data, f, indent=2)

    return tokens_file

def run_server():
    """Run HTTP server to capture OAuth callback"""
    global server_should_stop
    server = HTTPServer(('127.0.0.1', 8182), OAuthCallbackHandler)
    print("🌐 OAuth callback server started on http://127.0.0.1:8182")
    print()

    # Run server until code is captured
    while not server_should_stop:
        server.handle_request()

    print("🛑 Server stopped")

def main():
    global code_verifier_global, credentials_global

    print("=" * 70)
    print("  Schwab OAuth 2.0 - Automated Flow")
    print("=" * 70)
    print()

    # Load credentials
    print("[1/5] Loading credentials...")
    credentials_global = load_credentials()
    print("✅ Credentials loaded")
    print()

    # Generate PKCE
    print("[2/5] Generating PKCE...")
    code_verifier_global, code_challenge = generate_pkce_pair()
    print(f"✅ Code Verifier: {code_verifier_global[:30]}...")
    print()

    # Generate authorization URL
    auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={credentials_global['app_key']}&redirect_uri={credentials_global['callback_url']}&response_type=code&code_challenge={code_challenge}&code_challenge_method=S256"

    print("[3/5] Starting OAuth callback server...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print()

    print("=" * 70)
    print("  OPEN THIS URL IN YOUR BROWSER NOW:")
    print("=" * 70)
    print()
    print(auth_url)
    print()
    print("=" * 70)
    print()
    print("Waiting for authorization... (server will auto-capture the code)")
    print()

    # Wait for code to be captured
    server_thread.join(timeout=300)  # 5 minute timeout

    if not captured_code:
        print("❌ Timeout: No authorization code received")
        return 1

    # Exchange code for tokens IMMEDIATELY
    print("[4/5] Exchanging code for tokens (IMMEDIATELY)...")
    response = exchange_for_tokens(captured_code, code_verifier_global, credentials_global)

    if response.status_code == 200:
        tokens = response.json()
        print("🎉 SUCCESS! Tokens obtained!")
        print()
        print(f"   Access Token: {tokens['access_token'][:30]}...")
        print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
        print(f"   Expires In: {tokens.get('expires_in', 1800)} seconds")
        print()

        # Save tokens
        print("[5/5] Saving tokens...")
        tokens_file = save_tokens(tokens, credentials_global)
        print(f"💾 Tokens saved to: {tokens_file}")
        print()
        print("=" * 70)
        print("  🎉 SCHWAB API AUTHENTICATED!")
        print("=" * 70)
        print()
        print("✅ Schwab API is ready for:")
        print("   - Market data queries")
        print("   - Order placement (dry-run mode)")
        print("   - Account information")
        print()
        print("Tokens will auto-refresh every 25 minutes.")
        print()
        return 0
    else:
        print(f"❌ Token exchange failed!")
        print(f"   Status: {response.status_code}")
        print(f"   Error: {response.text}")
        return 1

if __name__ == "__main__":
    exit(main())
