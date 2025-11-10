#!/usr/bin/env python3
"""
Exchange Schwab Authorization Code for Tokens

Usage:
    python3 scripts/exchange_schwab_token.py <AUTHORIZATION_CODE>
"""

import sys
import yaml
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta

def load_credentials():
    """Load Schwab credentials"""
    config_file = Path(__file__).parent.parent / "configs" / "api_keys.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    schwab = config.get("schwab", {})
    return {
        "app_key": schwab.get("app_secret"),
        "client_secret": schwab.get("client_secret"),
        "callback_url": schwab.get("callback_url", "https://127.0.0.1:8182")
    }

def load_code_verifier():
    """Load code verifier from temp file"""
    verifier_file = Path("/tmp/schwab_code_verifier.txt")
    if not verifier_file.exists():
        return None
    return verifier_file.read_text().strip()

def exchange_code_for_tokens(auth_code, code_verifier, app_key, client_secret, callback_url):
    """Exchange authorization code for access and refresh tokens"""
    import base64

    token_url = "https://api.schwabapi.com/v1/oauth/token"

    # Schwab requires Basic Authentication (client_id:client_secret base64 encoded)
    auth_string = f"{app_key}:{client_secret}"
    auth_bytes = auth_string.encode('utf-8')
    auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {auth_b64}"
    }

    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": callback_url,
        "code_verifier": code_verifier
    }

    print(f"📡 Requesting tokens from: {token_url}")
    print(f"   Using app_key: {app_key[:20]}...")
    print()

    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=30)

        print(f"Response Status: {response.status_code}")

        if response.status_code == 200:
            tokens = response.json()
            return tokens, None
        else:
            error = {
                "status_code": response.status_code,
                "response": response.text
            }
            return None, error

    except Exception as e:
        return None, {"error": str(e)}

def save_tokens(tokens):
    """Save tokens to file for future use"""
    tokens_file = Path(__file__).parent.parent / "data" / "schwab_tokens.json"
    tokens_file.parent.mkdir(parents=True, exist_ok=True)

    # Add expiry timestamps
    token_data = {
        "access_token": tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "token_type": tokens.get("token_type", "Bearer"),
        "expires_in": tokens.get("expires_in", 1800),  # 30 minutes
        "scope": tokens.get("scope", ""),
        "obtained_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(seconds=tokens.get("expires_in", 1800))).isoformat()
    }

    with open(tokens_file, 'w') as f:
        json.dump(token_data, f, indent=2)

    print(f"💾 Tokens saved to: {tokens_file}")
    return tokens_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/exchange_schwab_token.py <AUTHORIZATION_CODE>")
        print()
        print("Get authorization code by:")
        print("  1. Run: python3 scripts/test_schwab_oauth.py")
        print("  2. Open the URL in browser and authorize")
        print("  3. Copy the 'code' parameter from the redirect URL")
        return 1

    auth_code = sys.argv[1]

    print("=" * 70)
    print("  Schwab Token Exchange")
    print("=" * 70)
    print()

    # Load credentials
    print("[1/4] Loading credentials...")
    creds = load_credentials()
    print("✅ Credentials loaded")

    # Load code verifier
    print()
    print("[2/4] Loading code verifier...")
    code_verifier = load_code_verifier()
    if not code_verifier:
        print("❌ Code verifier not found in /tmp/schwab_code_verifier.txt")
        print("   Run test_schwab_oauth.py first to generate it")
        return 1
    print(f"✅ Code verifier: {code_verifier[:20]}...")

    # Exchange code for tokens
    print()
    print("[3/4] Exchanging authorization code for tokens...")
    tokens, error = exchange_code_for_tokens(
        auth_code,
        code_verifier,
        creds['app_key'],
        creds['client_secret'],
        creds['callback_url']
    )

    if error:
        print(f"❌ Token exchange failed:")
        print(f"   Status: {error.get('status_code', 'N/A')}")
        print(f"   Error: {error.get('error', error.get('response', 'Unknown'))}")
        return 1

    print("✅ Tokens obtained successfully!")
    print()
    print(f"   Access Token: {tokens['access_token'][:20]}...")
    print(f"   Refresh Token: {tokens['refresh_token'][:20]}...")
    print(f"   Token Type: {tokens.get('token_type', 'Bearer')}")
    print(f"   Expires In: {tokens.get('expires_in', 1800)} seconds ({tokens.get('expires_in', 1800)//60} minutes)")

    # Save tokens
    print()
    print("[4/4] Saving tokens...")
    tokens_file = save_tokens(tokens)
    print("✅ Tokens saved")

    print()
    print("=" * 70)
    print("  SUCCESS! Schwab API is now authenticated")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Test market data: Use the schwab_api C++ module")
    print("  2. The tokens will auto-refresh every 25 minutes")
    print("  3. Refresh token is valid for 7 days")
    print()
    print(f"Token file: {tokens_file}")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
