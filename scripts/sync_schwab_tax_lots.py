#!/usr/bin/env python3
"""
Sync Tax Lots from Schwab API

Fetches official tax lot data from Schwab's internal API and syncs with
our bot tax lots tracking system.

API Endpoint: https://ausgateway.schwab.com/api/is.Holdings/V1/Lots

Usage:
    uv run python scripts/sync_schwab_tax_lots.py
"""

import schwab
import duckdb
import json
from pathlib import Path
from datetime import datetime

def sync_schwab_tax_lots():
    """Fetch and sync tax lots from Schwab API"""

    print("=" * 80)
    print("Syncing Tax Lots from Schwab API")
    print("=" * 80)

    # Load Schwab client
    token_file = Path('configs/schwab_tokens.json')
    app_key = '8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa'
    app_secret = 'PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT'

    if not token_file.exists():
        print(f"❌ Token file not found: {token_file}")
        return 1

    print(f"\n📡 Loading Schwab client...")
    client = schwab.auth.client_from_token_file(
        str(token_file),
        app_key,
        app_secret
    )

    # Get account numbers
    print("📋 Fetching account numbers...")
    accounts_response = client.get_account_numbers()

    if accounts_response.status_code != 200:
        print(f"❌ Failed to get accounts: {accounts_response.status_code}")
        return 1

    accounts = accounts_response.json()
    account_num = accounts[0]['accountNumber']
    print(f"   Account: {account_num}")

    # Try to fetch tax lots using the internal API endpoint
    # Note: This might require different authentication or headers
    print("\n🔍 Attempting to fetch tax lots from Schwab API...")

    # Method 1: Try using the internal endpoint directly
    # We'll need to inspect the actual API call structure
    # The schwab-py library might not expose this endpoint directly

    # For now, let's check what data we can get from positions
    # and see if it includes lot information
    print("📊 Fetching account positions with fields...")
    positions_response = client.get_account(
        account_num,
        fields='positions'
    )

    if positions_response.status_code != 200:
        print(f"❌ Failed to get positions: {positions_response.status_code}")
        return 1

    account_data = positions_response.json()

    # Save raw response for inspection
    debug_file = Path('data/schwab_positions_debug.json')
    debug_file.parent.mkdir(exist_ok=True)
    with open(debug_file, 'w') as f:
        json.dump(account_data, f, indent=2)
    print(f"   Saved raw response to: {debug_file}")

    # Check if positions contain lot information
    positions = account_data.get('securitiesAccount', {}).get('positions', [])
    print(f"\n📈 Found {len(positions)} positions")

    # Inspect first position to see available fields
    if positions:
        print("\n🔍 Sample position fields:")
        sample = positions[0]
        for key in sorted(sample.keys()):
            print(f"   {key}: {type(sample[key]).__name__}")

        # Check for lot-related fields
        lot_fields = [k for k in sample.keys() if 'lot' in k.lower() or 'cost' in k.lower() or 'basis' in k.lower()]
        if lot_fields:
            print(f"\n✅ Found lot-related fields: {lot_fields}")
            print("\nSample values:")
            for field in lot_fields:
                print(f"   {field}: {sample.get(field)}")
        else:
            print("\n⚠️  No lot-related fields found in standard positions endpoint")
            print("   Tax lot data likely requires different endpoint or authentication")

    print("\n" + "=" * 80)
    print("Tax Lot Sync Investigation Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check data/schwab_positions_debug.json for available fields")
    print("2. If no lot data, we may need to:")
    print("   a. Use Schwab's web session cookies for internal API")
    print("   b. Contact Schwab developer support for tax lot endpoint access")
    print("   c. Use unofficial schwab-api library (itsjafer/schwab-api)")
    print()

    return 0

if __name__ == "__main__":
    exit(sync_schwab_tax_lots())
