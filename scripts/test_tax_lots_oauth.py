#!/usr/bin/env python3
"""
Test Tax Lots Endpoint with OAuth

Attempts to fetch tax lots from Schwab's internal API
using our existing OAuth tokens.

This may or may not work - the internal API might require
web session authentication instead of OAuth.

Usage:
    uv run python scripts/test_tax_lots_oauth.py
"""

import schwab
import requests
import json
from pathlib import Path

def test_tax_lots_with_oauth():
    """Test if we can access tax lots endpoint with OAuth"""

    print("=" * 80)
    print("Testing Tax Lots API with OAuth Authentication")
    print("=" * 80)

    # Load Schwab client with OAuth
    token_file = Path('configs/schwab_tokens.json')
    app_key = '8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa'
    app_secret = 'PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT'

    print("\n🔐 Loading OAuth client...")
    client = schwab.auth.client_from_token_file(
        str(token_file),
        app_key,
        app_secret
    )

    # Get account number
    print("📋 Getting account number...")
    accounts_resp = client.get_account_numbers()
    if accounts_resp.status_code != 200:
        print(f"❌ Failed to get accounts: {accounts_resp.status_code}")
        return 1

    account_num = accounts_resp.json()[0]['accountNumber']
    print(f"   Account: {account_num}")

    # Get positions to extract security IDs
    print("\n📊 Getting positions...")
    # Use hashValue instead of account number (as done in sync_schwab_portfolio.py)
    hash_value = accounts_resp.json()[0]['hashValue']
    positions_resp = client.get_account(hash_value, fields=client.Account.Fields.POSITIONS)

    if positions_resp.status_code != 200:
        print(f"❌ Failed to get positions: {positions_resp.status_code}")
        print(f"   Response: {positions_resp.text[:200]}")
        return 1

    account_data = positions_resp.json()
    positions = account_data.get('securitiesAccount', {}).get('positions', [])
    print(f"   Found {len(positions)} positions")

    # Check if positions have security_id or similar field
    if positions:
        print("\n🔍 Inspecting position fields for security_id...")
        sample = positions[0]
        symbol = sample.get('instrument', {}).get('symbol', 'UNKNOWN')
        cusip = sample.get('instrument', {}).get('cusip')

        print(f"   Sample position: {symbol}")
        print(f"   Available fields: {sorted(sample.keys())}")

        # Look for any ID fields
        id_fields = {k: v for k, v in sample.items() if 'id' in k.lower() or k in ['cusip', 'symbol']}
        if id_fields:
            print(f"\n   ID-like fields:")
            for key, val in id_fields.items():
                print(f"      {key}: {val}")

        # Try to call tax lots endpoint using internal API URL
        print("\n🧪 Attempting to call tax lots endpoint...")
        print("   Endpoint: https://ausgateway.schwab.com/api/is.Holdings/V1/Lots")

        # Get OAuth token from client
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        access_token = tokens.get('access_token')

        # Try different approaches
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'schwab-client-ids': str(account_num)
        }

        # Approach 1: Try with CUSIP
        if cusip:
            url = 'https://ausgateway.schwab.com/api/is.Holdings/V1/Lots'
            params = {
                'isLong': True,
                'itemissueid': cusip  # Try CUSIP as security_id
            }

            print(f"\n   Approach 1: Using CUSIP as itemissueid")
            print(f"      CUSIP: {cusip}")

            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                print(f"      Status: {response.status_code}")

                if response.status_code == 200:
                    print("      ✅ SUCCESS! Tax lots endpoint accessible with OAuth")
                    data = response.json()
                    print(f"      Response: {json.dumps(data, indent=2)[:500]}...")

                    # Save response
                    output_file = Path('data/tax_lots_oauth_response.json')
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"      Saved to: {output_file}")

                elif response.status_code == 401:
                    print("      ❌ 401 Unauthorized - OAuth tokens not accepted by internal API")
                    print("         Internal API likely requires web session cookies")
                elif response.status_code == 404:
                    print("      ❌ 404 Not Found - Wrong security_id or endpoint")
                else:
                    print(f"      ❌ Error {response.status_code}")
                    print(f"      Response: {response.text[:200]}")

            except Exception as e:
                print(f"      ❌ Request failed: {e}")

        # Approach 2: Try with symbol
        print(f"\n   Approach 2: Using symbol as itemissueid")
        params2 = {
            'isLong': True,
            'itemissueid': symbol
        }

        try:
            response2 = requests.get(url, headers=headers, params=params2, timeout=10)
            print(f"      Status: {response2.status_code}")

            if response2.status_code == 200:
                print("      ✅ SUCCESS with symbol!")
                print(f"      Response: {response2.json()}")
            else:
                print(f"      ❌ Error {response2.status_code}")

        except Exception as e:
            print(f"      ❌ Request failed: {e}")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
    print("\nConclusion:")
    print("   If OAuth works: We can fetch tax lots directly")
    print("   If OAuth fails: We need web session auth (itsjafer/schwab-api)")
    print()

    return 0

if __name__ == "__main__":
    exit(test_tax_lots_with_oauth())
