#!/usr/bin/env python3
"""
Complete Schwab OAuth Flow - Streamlined Version

This script:
1. Generates authorization URL
2. Waits for you to paste the authorization code
3. Immediately exchanges it for tokens

Usage:
    python3 scripts/schwab_oauth_complete.py
"""

import yaml
import secrets
import hashlib
import base64
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
    """Exchange code for tokens with Basic Auth"""
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
    """Save tokens to data directory"""
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

def main():
    print("=" * 70)
    print("  Schwab OAuth 2.0 - Complete Flow")
    print("=" * 70)
    print()

    # Load credentials
    creds = load_credentials()
    print("✅ Credentials loaded")
    print()

    # Generate PKCE
    code_verifier, code_challenge = generate_pkce_pair()
    print("✅ PKCE generated")
    print(f"   Code Verifier: {code_verifier}")
    print()

    # Generate authorization URL
    auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={creds['app_key']}&redirect_uri={creds['callback_url']}&response_type=code&code_challenge={code_challenge}&code_challenge_method=S256"

    print("=" * 70)
    print("STEP 1: Open this URL in your browser RIGHT NOW:")
    print("=" * 70)
    print()
    print(auth_url)
    print()
    print("=" * 70)
    print()

    # Wait for authorization code
    print("STEP 2: After authorizing, you'll be redirected to:")
    print(f"        {creds['callback_url']}/?code=XXXXX&session=YYYYY")
    print()
    auth_code = input("Paste the FULL authorization code (everything after code=, before &session): ").strip()

    # Clean up the code (remove %40 encoding if present)
    if '%40' in auth_code:
        auth_code = auth_code.replace('%40', '@')
    # Remove &session if included
    if '&' in auth_code:
        auth_code = auth_code.split('&')[0]

    print()
    print("=" * 70)
    print("STEP 3: Exchanging code for tokens (IMMEDIATELY)...")
    print("=" * 70)
    print()

    # Exchange immediately
    response = exchange_for_tokens(auth_code, code_verifier, creds)

    if response.status_code == 200:
        tokens = response.json()
        print("🎉 SUCCESS! Tokens obtained!")
        print()
        print(f"   Access Token: {tokens['access_token'][:30]}...")
        print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
        print(f"   Expires In: {tokens.get('expires_in', 1800)} seconds")
        print()

        # Save tokens
        tokens_file = save_tokens(tokens, creds)
        print(f"💾 Tokens saved to: {tokens_file}")
        print()
        print("=" * 70)
        print("  Schwab API is now AUTHENTICATED and READY!")
        print("=" * 70)
        print()
        print("You can now:")
        print("  - Fetch market data")
        print("  - Place orders (dry-run mode recommended)")
        print("  - Access account information")
        print()
        print("Tokens will auto-refresh every 25 minutes.")
        print("Refresh token valid for 7 days.")
        print()
        return 0
    else:
        print(f"❌ Token exchange failed!")
        print(f"   Status: {response.status_code}")
        print(f"   Error: {response.text}")
        return 1

if __name__ == "__main__":
    exit(main())
