#!/usr/bin/env python3
"""
Add Mock Tax Lots for Testing

Creates sample bot-managed tax lots to test the dashboard display.
Includes both equities and options with various strategies.

Usage:
    uv run python scripts/add_mock_tax_lots.py
"""

import duckdb
from pathlib import Path
from datetime import datetime, timedelta

def add_mock_tax_lots():
    """Add sample tax lots for testing"""

    db_path = Path('data/bigbrother.duckdb')
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    print("=" * 80)
    print("Adding Mock Tax Lots for Testing")
    print("=" * 80)

    conn = duckdb.connect(str(db_path))

    # Get account ID from positions table
    account_id = conn.execute("SELECT DISTINCT account_id FROM positions LIMIT 1").fetchone()
    if account_id:
        account_id = account_id[0]
    else:
        account_id = "69398875"  # Default

    print(f"\nUsing account ID: {account_id}")

    # Mock tax lots data
    mock_lots = [
        # SPY Call Option - Delta Neutral Straddle (opened today)
        {
            'symbol': 'SPY   251219C00679000',
            'asset_type': 'OPTION',
            'quantity': 1,
            'entry_price': 4.50,  # $450 per contract
            'entry_date': datetime.now() - timedelta(hours=2),
            'strategy': 'delta_neutral_straddle',
            'option_type': 'CALL',
            'strike_price': 679.0,
            'expiration_date': '2025-12-19',
            'underlying_symbol': 'SPY',
        },
        # SPY Put Option - Delta Neutral Straddle (same leg as above)
        {
            'symbol': 'SPY   251219P00679000',
            'asset_type': 'OPTION',
            'quantity': 1,
            'entry_price': 4.75,  # $475 per contract
            'entry_date': datetime.now() - timedelta(hours=2),
            'strategy': 'delta_neutral_straddle',
            'option_type': 'PUT',
            'strike_price': 679.0,
            'expiration_date': '2025-12-19',
            'underlying_symbol': 'SPY',
        },
        # QS Equity - Volatility Arbitrage (opened 3 days ago)
        {
            'symbol': 'QS',
            'asset_type': 'EQUITY',
            'quantity': 50,
            'entry_price': 15.82,
            'entry_date': datetime.now() - timedelta(days=3),
            'strategy': 'volatility_arbitrage',
            'option_type': None,
            'strike_price': None,
            'expiration_date': None,
            'underlying_symbol': None,
        },
        # NVDA Call Option - Delta Neutral Strangle (opened last week)
        {
            'symbol': 'NVDA  260115C00150000',
            'asset_type': 'OPTION',
            'quantity': 2,
            'entry_price': 6.25,  # $625 per contract
            'entry_date': datetime.now() - timedelta(days=7),
            'strategy': 'delta_neutral_strangle',
            'option_type': 'CALL',
            'strike_price': 150.0,
            'expiration_date': '2026-01-15',
            'underlying_symbol': 'NVDA',
        },
        # NVDA Put Option - Delta Neutral Strangle (same leg as above)
        {
            'symbol': 'NVDA  260115P00135000',
            'asset_type': 'OPTION',
            'quantity': 2,
            'entry_price': 5.80,  # $580 per contract
            'entry_date': datetime.now() - timedelta(days=7),
            'strategy': 'delta_neutral_strangle',
            'option_type': 'PUT',
            'strike_price': 135.0,
            'expiration_date': '2026-01-15',
            'underlying_symbol': 'NVDA',
        },
    ]

    print(f"\n📊 Adding {len(mock_lots)} mock tax lots...")

    # Get next available ID
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM tax_lots").fetchone()[0]
    next_id = max_id + 1

    for lot in mock_lots:
        conn.execute("""
            INSERT INTO tax_lots (
                id, account_id, symbol, asset_type, quantity, entry_price, entry_date,
                strategy, option_type, strike_price, expiration_date, underlying_symbol,
                is_closed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, false)
        """, [
            next_id,
            account_id,
            lot['symbol'],
            lot['asset_type'],
            lot['quantity'],
            lot['entry_price'],
            lot['entry_date'],
            lot['strategy'],
            lot['option_type'],
            lot['strike_price'],
            lot['expiration_date'],
            lot['underlying_symbol'],
        ])

        asset_desc = lot['asset_type']
        if lot['asset_type'] == 'OPTION':
            asset_desc = f"{lot['option_type']} ${lot['strike_price']}"

        print(f"   ✅ {lot['symbol']:20s} | {asset_desc:15s} | {lot['strategy']}")
        next_id += 1

    conn.commit()

    # Show summary
    print("\n📈 Tax Lot Summary:")
    summary = conn.execute("SELECT * FROM v_tax_lots_summary").fetchall()
    if summary:
        for row in summary:
            print(f"   {row[0]:20s} | {row[1]:8s} | {row[2]:30s} | Lots: {row[3]} | Cost: ${row[5]:,.2f}")
    else:
        print("   No tax lots found")

    conn.close()

    print("\n" + "=" * 80)
    print("✅ Mock Tax Lots Added Successfully!")
    print("=" * 80)
    print("\nView in dashboard:")
    print("  1. Restart dashboard: uv run streamlit run dashboard/app.py")
    print("  2. Navigate to 'Bot Tax Lots' section")
    print()

    return 0

if __name__ == "__main__":
    exit(add_mock_tax_lots())
