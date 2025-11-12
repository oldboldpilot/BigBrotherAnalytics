#!/usr/bin/env python3
"""
Check Which Symbols Bot Can Safely Trade

With LIFO-only selling, bot can ONLY trade securities where it owns 100% of lots.
This script identifies safe symbols by comparing bot lots vs Schwab positions.

Usage:
    uv run python scripts/check_bot_tradable_symbols.py
"""

import schwab
import duckdb
from pathlib import Path
from datetime import datetime

def check_tradable_symbols():
    """Check which symbols bot can safely trade with LIFO"""

    print("=" * 80)
    print("Bot Tradable Symbols Check (LIFO Constraint)")
    print("=" * 80)
    print("\nRule: Bot can ONLY trade securities where it owns 100% of lots")
    print("      (LIFO applies to ALL lots, not just bot lots)")
    print()

    # Load Schwab client
    token_file = Path('configs/schwab_tokens.json')
    client = schwab.auth.client_from_token_file(
        str(token_file),
        '8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa',
        'PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT'
    )

    # Get Schwab positions
    print("📊 Fetching Schwab positions...")
    accounts = client.get_account_numbers().json()
    hash_value = accounts[0]['hashValue']
    positions_resp = client.get_account(hash_value, fields=client.Account.Fields.POSITIONS)

    positions = positions_resp.json()['securitiesAccount']['positions']
    print(f"   Found {len(positions)} total positions")

    # Get bot tax lots
    print("🤖 Fetching bot tax lots...")
    db_path = Path('data/bigbrother.duckdb')
    conn = duckdb.connect(str(db_path))

    bot_lots = conn.execute("""
        SELECT symbol, asset_type, COUNT(*) as num_lots, SUM(quantity) as total_qty
        FROM tax_lots
        WHERE is_closed = false
        GROUP BY symbol, asset_type
    """).fetchall()
    print(f"   Found {len(bot_lots)} bot positions")

    bot_symbols = {lot[0]: {'qty': lot[3], 'asset_type': lot[1]} for lot in bot_lots}

    # Analyze each position
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    safe_symbols = []
    unsafe_symbols = []
    manual_only = []

    for pos in positions:
        symbol = pos['instrument']['symbol']
        schwab_qty = pos.get('longQuantity', 0)
        asset_type = pos['instrument'].get('assetType', 'UNKNOWN')

        # Get bot quantity for this symbol
        bot_qty = bot_symbols.get(symbol, {}).get('qty', 0)

        # Analysis
        if bot_qty == 0:
            # Manual position only
            manual_only.append({
                'symbol': symbol,
                'type': asset_type,
                'schwab_qty': schwab_qty,
                'bot_qty': 0
            })
        elif abs(schwab_qty - bot_qty) < 0.01:  # Allow small floating point diff
            # Bot owns 100% - SAFE to trade
            safe_symbols.append({
                'symbol': symbol,
                'type': asset_type,
                'schwab_qty': schwab_qty,
                'bot_qty': bot_qty
            })
        else:
            # Mixed manual + bot - UNSAFE
            unsafe_symbols.append({
                'symbol': symbol,
                'type': asset_type,
                'schwab_qty': schwab_qty,
                'bot_qty': bot_qty,
                'manual_qty': schwab_qty - bot_qty
            })

    # Report: Safe Symbols
    print("\n✅ SAFE TO TRADE (Bot owns 100%):")
    if safe_symbols:
        for s in safe_symbols:
            print(f"   {s['symbol']:20s} | {s['type']:10s} | Qty: {s['schwab_qty']:8.2f} (100% bot)")
    else:
        print("   None - Bot has no exclusive positions")

    # Report: Unsafe Symbols
    print("\n❌ UNSAFE TO TRADE (Mixed manual + bot):")
    if unsafe_symbols:
        for s in unsafe_symbols:
            print(f"   {s['symbol']:20s} | {s['type']:10s} | Total: {s['schwab_qty']:8.2f} | Bot: {s['bot_qty']:8.2f} | Manual: {s['manual_qty']:8.2f}")
        print("\n   ⚠️  WARNING: LIFO selling would affect manual lots!")
        print("       Bot must NOT trade these symbols")
    else:
        print("   None - No conflicts")

    # Report: Manual Only
    print("\n📋 Manual Only (No bot tracking):")
    if manual_only:
        for s in manual_only[:5]:  # Show first 5
            print(f"   {s['symbol']:20s} | {s['type']:10s} | Qty: {s['schwab_qty']:8.2f}")
        if len(manual_only) > 5:
            print(f"   ... and {len(manual_only) - 5} more")
    else:
        print("   None")

    # Save safe symbols to config
    print("\n" + "=" * 80)
    print("SAVING SAFE SYMBOLS LIST")
    print("=" * 80)

    safe_symbols_list = [s['symbol'] for s in safe_symbols]

    # Create or update whitelist table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_tradable_symbols (
            symbol VARCHAR PRIMARY KEY,
            asset_type VARCHAR NOT NULL,
            last_verified TIMESTAMP NOT NULL,
            reason VARCHAR
        )
    """)

    # Clear old whitelist
    conn.execute("DELETE FROM bot_tradable_symbols")

    # Insert safe symbols
    for s in safe_symbols:
        conn.execute("""
            INSERT INTO bot_tradable_symbols (symbol, asset_type, last_verified, reason)
            VALUES (?, ?, ?, ?)
        """, [s['symbol'], s['type'], datetime.now(), 'Bot owns 100% of lots'])

    conn.commit()

    print(f"✅ Saved {len(safe_symbols)} safe symbols to bot_tradable_symbols table")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"   Safe to trade:     {len(safe_symbols):3d}")
    print(f"   Unsafe (mixed):    {len(unsafe_symbols):3d}")
    print(f"   Manual only:       {len(manual_only):3d}")
    print()
    print("Bot Strategy Recommendation:")
    if len(safe_symbols) > 0:
        print(f"   ✅ Bot can trade {len(safe_symbols)} symbols using LIFO")
        print(f"   ✅ These are stored in bot_tradable_symbols table")
    else:
        print("   ⚠️  No safe symbols - bot should open new positions only")
        print("       Or use completely separate account for bot trading")
    print()

    conn.close()
    return 0

if __name__ == "__main__":
    exit(check_tradable_symbols())
