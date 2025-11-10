#!/usr/bin/env python3
"""
Test Schwab OAuth 2.0 Flow with Real Credentials

This script tests the OAuth flow using actual credentials from configs/api_keys.yaml.
It will generate an authorization URL and help you obtain tokens.

Usage:
    python3 scripts/test_schwab_oauth.py
"""

import yaml
import secrets
import hashlib
import base64
from pathlib import Path

def load_credentials():
    """Load Schwab credentials from configs/api_keys.yaml"""
    config_file = Path(__file__).parent.parent / "configs" / "api_keys.yaml"

    with open(config_file) as f:
        config = yaml.safe_load(f)

    schwab = config.get("schwab", {})

    # Schwab uses app_key (not client_id) for the OAuth parameter
    app_key = schwab.get("app_secret") or schwab.get("client_id")

    return {
        "app_key": app_key,
        "client_secret": schwab.get("client_secret"),
        "callback_url": schwab.get("callback_url", "https://127.0.0.1:8182"),
        "account_id": schwab.get("account_id"),
        "base_url": schwab.get("base_url", "https://api.schwabapi.com/trader/v1")
    }

def generate_pkce_pair():
    """Generate PKCE code verifier and challenge"""
    # Generate code verifier (43-128 characters)
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

    # Generate code challenge (SHA256 hash of verifier)
    challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')

    return code_verifier, code_challenge

def generate_authorization_url(app_key, callback_url, code_challenge):
    """Generate Schwab OAuth authorization URL

    Note: Schwab uses the app_secret as the client_id in OAuth flow
    """
    auth_url = "https://api.schwabapi.com/v1/oauth/authorize"

    # Schwab OAuth 2.0 parameters
    params = [
        f"client_id={app_key}",
        f"redirect_uri={callback_url}",
        "response_type=code",
        f"code_challenge={code_challenge}",
        "code_challenge_method=S256"
    ]

    return f"{auth_url}?{'&'.join(params)}"

def main():
    print("=" * 70)
    print("  Schwab API OAuth 2.0 Test")
    print("=" * 70)
    print()

    # Load credentials
    print("[1/4] Loading credentials from configs/api_keys.yaml...")
    try:
        creds = load_credentials()
        print(f"✅ App Key (client_id): {creds['app_key'][:20]}...")
        print(f"✅ Callback URL: {creds['callback_url']}")
        print(f"✅ Account ID: {creds['account_id']}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return 1

    # Generate PKCE
    print()
    print("[2/4] Generating PKCE code verifier and challenge...")
    code_verifier, code_challenge = generate_pkce_pair()
    print(f"✅ Code Verifier: {code_verifier[:20]}... ({len(code_verifier)} chars)")
    print(f"✅ Code Challenge: {code_challenge[:20]}... ({len(code_challenge)} chars)")

    # Generate auth URL
    print()
    print("[3/4] Generating authorization URL...")
    auth_url = generate_authorization_url(creds['app_key'], creds['callback_url'], code_challenge)
    print(f"✅ Authorization URL generated")

    # Display instructions
    print()
    print("[4/4] Next Steps:")
    print("=" * 70)
    print()
    print("1. Open this URL in your browser:")
    print()
    print(f"   {auth_url}")
    print()
    print("2. Log in to your Schwab account")
    print()
    print("3. Authorize the application")
    print()
    print("4. You'll be redirected to:")
    print(f"   {creds['callback_url']}?code=AUTHORIZATION_CODE")
    print()
    print("5. Copy the authorization code from the URL")
    print()
    print("6. Save these for token exchange:")
    print(f"   Code Verifier: {code_verifier}")
    print()
    print("=" * 70)
    print()
    print("📝 To exchange the code for tokens, use:")
    print("   python3 scripts/exchange_schwab_token.py <AUTH_CODE>")
    print()
    print("⚠️  Security Notes:")
    print("   - Authorization code expires in 30 seconds")
    print("   - Code verifier must match exactly")
    print("   - Tokens will be stored in DuckDB")
    print()

    # Save code verifier to temp file
    verifier_file = Path("/tmp/schwab_code_verifier.txt")
    verifier_file.write_text(code_verifier)
    print(f"💾 Code verifier saved to: {verifier_file}")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
