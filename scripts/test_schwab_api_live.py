#!/usr/bin/env python3
"""
Test Schwab API with Real Tokens

Uses existing tokens from configs/schwab_tokens.json to test live API calls.
"""

import json
import requests
from pathlib import Path
from datetime import datetime

def load_tokens():
    """Load existing Schwab tokens"""
    token_file = Path(__file__).parent.parent / "configs" / "schwab_tokens.json"
    with open(token_file) as f:
        data = json.load(f)
    return data['token']

def test_account_info(access_token, account_id):
    """Test getting account information"""
    url = f"https://api.schwabapi.com/trader/v1/accounts/{account_id}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }

    print(f"Testing account info for account: {account_id}")
    response = requests.get(url, headers=headers, timeout=30)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Account info retrieved successfully!")
        data = response.json()
        print(json.dumps(data, indent=2))
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_quote(access_token, symbol="SPY"):
    """Test getting a quote"""
    url = f"https://api.schwabapi.com/marketdata/v1/{symbol}/quotes"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }

    print(f"\nTesting quote for: {symbol}")
    response = requests.get(url, headers=headers, timeout=30)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✅ Quote for {symbol} retrieved successfully!")
        data = response.json()
        print(json.dumps(data, indent=2))
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def main():
    print("=" * 70)
    print("  Schwab API Live Test")
    print("=" * 70)
    print()

    # Load tokens
    print("Loading tokens from configs/schwab_tokens.json...")
    tokens = load_tokens()

    access_token = tokens['access_token']
    print(f"Access Token: {access_token[:30]}...")
    print(f"Token Type: {tokens.get('token_type', 'Bearer')}")
    print(f"Expires In: {tokens.get('expires_in', 1800)} seconds")
    print()

    # Load account ID from api_keys.yaml
    import yaml
    config_file = Path(__file__).parent.parent / "configs" / "api_keys.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    account_id = config['schwab']['account_id']

    # Test 1: Get quote (market data endpoint)
    print("=" * 70)
    print("TEST 1: Market Data - Get SPY Quote")
    print("=" * 70)
    success1 = test_quote(access_token, "SPY")

    # Test 2: Get account info
    print()
    print("=" * 70)
    print("TEST 2: Account Information")
    print("=" * 70)
    success2 = test_account_info(access_token, account_id)

    print()
    print("=" * 70)
    if success1 or success2:
        print("  ✅ Schwab API is OPERATIONAL!")
    else:
        print("  ❌ API tests failed - may need token refresh")
    print("=" * 70)

    return 0 if (success1 or success2) else 1

if __name__ == "__main__":
    exit(main())
