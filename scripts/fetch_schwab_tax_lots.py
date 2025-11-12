#!/usr/bin/env python3
"""
Fetch Schwab Tax Lots using itsjafer/schwab-api

Uses the unofficial schwab-api library to fetch tax lot data from
Schwab's internal API endpoint.

Installation:
    uv pip install schwab-api

Note: This requires web session authentication (browser login),
      separate from OAuth tokens used by schwab-py.

Usage:
    uv run python scripts/fetch_schwab_tax_lots.py
"""

import duckdb
from pathlib import Path
from datetime import datetime
import json

def fetch_tax_lots():
    """Fetch tax lots from Schwab using unofficial API"""

    print("=" * 80)
    print("Fetching Tax Lots from Schwab Internal API")
    print("=" * 80)

    # Try to import schwab-api
    try:
        from schwab_api import Schwab
    except ImportError:
        print("\n❌ schwab-api library not installed")
        print("\nInstall with:")
        print("   uv pip install schwab-api")
        print("\nAlternatively, install from source:")
        print("   git clone https://github.com/itsjafer/schwab-api.git")
        print("   cd schwab-api && pip install -e .")
        return 1

    print("\n🔐 Authenticating with Schwab...")
    print("⚠️  This library requires web login (browser-based authentication)")
    print("    You'll need to log in through your browser")
    print()

    # Initialize Schwab API client
    # This will prompt for browser login
    api = Schwab()

    print("📋 Fetching account information...")
    success, accounts = api.account_info_v2()

    if not success:
        print(f"❌ Failed to get account info: {accounts}")
        return 1

    account_id = accounts[0]['accountId']
    print(f"   Account ID: {account_id}")

    print("\n📊 Fetching positions...")
    success, positions_data = api.positions_v2()

    if not success:
        print(f"❌ Failed to get positions: {positions_data}")
        return 1

    positions = positions_data.get('positions', [])
    print(f"   Found {len(positions)} positions")

    # Save raw positions for debugging
    debug_file = Path('data/schwab_positions_v2_debug.json')
    debug_file.parent.mkdir(exist_ok=True)
    with open(debug_file, 'w') as f:
        json.dump(positions_data, f, indent=2)
    print(f"   Saved to: {debug_file}")

    # Fetch tax lots for each position
    print("\n📦 Fetching tax lots for each position...")

    db_path = Path('data/bigbrother.duckdb')
    conn = duckdb.connect(str(db_path))

    # Create table for Schwab tax lots if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schwab_tax_lots (
            lot_id VARCHAR PRIMARY KEY,
            account_id VARCHAR NOT NULL,
            security_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            quantity DOUBLE NOT NULL,
            cost_per_share DOUBLE NOT NULL,
            acquisition_date TIMESTAMP NOT NULL,
            current_price DOUBLE,
            market_value DOUBLE,
            unrealized_pnl DOUBLE,
            term VARCHAR,  -- SHORT_TERM, LONG_TERM
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    total_lots = 0

    for position in positions[:5]:  # Start with first 5 positions for testing
        symbol = position.get('symbol', 'UNKNOWN')
        security_id = position.get('security_id')

        if not security_id:
            print(f"   ⚠️  {symbol}: No security_id found, skipping")
            continue

        print(f"\n   Fetching lots for {symbol} (security_id: {security_id})...")

        success, lot_data = api.get_lot_info_v2(account_id, security_id)

        if not success:
            print(f"      ❌ Failed: {lot_data}")
            continue

        lots = lot_data.get('lots', [])
        print(f"      ✅ Found {len(lots)} tax lots")

        # Save each lot to database
        for lot in lots:
            lot_id = lot.get('lotId', f"{security_id}_{lot.get('acquisitionDate')}")

            conn.execute("""
                INSERT OR REPLACE INTO schwab_tax_lots
                (lot_id, account_id, security_id, symbol, quantity,
                 cost_per_share, acquisition_date, current_price,
                 market_value, unrealized_pnl, term)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                lot_id,
                str(account_id),
                str(security_id),
                symbol,
                lot.get('quantity', 0),
                lot.get('costPerShare', 0),
                lot.get('acquisitionDate'),
                lot.get('currentPrice', 0),
                lot.get('marketValue', 0),
                lot.get('unrealizedPnL', 0),
                lot.get('term')
            ])

            total_lots += 1

            print(f"         Lot {lot_id}: {lot.get('quantity')} @ ${lot.get('costPerShare'):.2f} ({lot.get('acquisitionDate')})")

    conn.commit()

    print("\n" + "=" * 80)
    print(f"✅ Fetched {total_lots} tax lots from Schwab")
    print("=" * 80)
    print("\nData saved to:")
    print(f"   Database: schwab_tax_lots table in {db_path}")
    print(f"   Debug:    {debug_file}")
    print()

    # Show summary
    summary = conn.execute("""
        SELECT symbol, COUNT(*) as num_lots, SUM(quantity) as total_qty,
               SUM(market_value) as total_value
        FROM schwab_tax_lots
        GROUP BY symbol
        ORDER BY total_value DESC
    """).fetchall()

    if summary:
        print("Tax Lots Summary:")
        for row in summary:
            print(f"   {row[0]:10s} | {row[1]:2d} lots | {row[2]:8.2f} shares | ${row[3]:,.2f}")

    conn.close()
    print()

    return 0

if __name__ == "__main__":
    exit(fetch_tax_lots())
