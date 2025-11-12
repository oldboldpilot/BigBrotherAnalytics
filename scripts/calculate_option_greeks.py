#!/usr/bin/env python3
"""
Calculate Option Greeks for Tax Lots

Uses OpenMP-accelerated C++ trinomial tree model to calculate Greeks
for all option positions in the tax_lots table.

Usage:
    uv run python scripts/calculate_option_greeks.py
"""

import os
import duckdb
from pathlib import Path
from datetime import datetime
import sys

# Set LD_LIBRARY_PATH for OpenMP and libc++ libraries
# Paths from Ansible playbook (playbooks/complete-tier1-setup.yml:881)
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/local/lib/x86_64-unknown-linux-gnu:/usr/lib/llvm-18/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Import the C++ bindings
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
try:
    import bigbrother_options
except ImportError as e:
    print(f"❌ Failed to import bigbrother_options: {e}")
    print("   Build Python bindings: ninja -C build bigbrother_options")
    print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    sys.exit(1)

def calculate_greeks():
    """Calculate Greeks for all option positions"""

    db_path = Path('data/bigbrother.duckdb')
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    print("=" * 80)
    print("Calculating Option Greeks with OpenMP-Accelerated Trinomial Tree")
    print("=" * 80)

    conn = duckdb.connect(str(db_path))

    # Get all open option positions
    options_df = conn.execute("""
        SELECT id, symbol, option_type, strike_price, expiration_date,
               entry_price, quantity, underlying_symbol
        FROM tax_lots
        WHERE asset_type = 'OPTION' AND is_closed = false
    """).fetchdf()

    if len(options_df) == 0:
        print("\n⚠️  No open option positions found")
        return 0

    print(f"\n📊 Found {len(options_df)} open option positions")
    print()

    # For simplicity, we'll use some default values
    # In production, these would come from market data
    risk_free_rate = 0.041  # Current FRED rate (4.1%)

    # We'll need to fetch current underlying prices
    # For now, we'll use placeholder values
    # In production, fetch from Schwab API or market data
    underlying_prices = {
        'SPY': 679.0,
        'NVDA': 142.0,
        'QS': 15.82  # This is equity, not option underlying
    }

    # Calculate Greeks for each option
    greeks_calculated = 0
    for idx, row in options_df.iterrows():
        symbol = row['symbol']
        option_type = row['option_type']
        strike = row['strike_price']
        expiration = row['expiration_date']
        underlying_symbol = row['underlying_symbol']
        lot_id = row['id']

        # Get underlying price
        if underlying_symbol not in underlying_prices:
            print(f"   ⚠️  {symbol}: No price data for {underlying_symbol}")
            continue

        spot_price = underlying_prices[underlying_symbol]

        # Calculate time to expiration
        # Handle both date and datetime formats
        expiration_str = str(expiration).split()[0]  # Get just the date part
        expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
        days_to_expiry = (expiration_date - datetime.now()).days
        time_to_expiry = days_to_expiry / 365.0

        if time_to_expiry <= 0:
            print(f"   ⚠️  {symbol}: Already expired")
            continue

        # Estimate implied volatility (in production, calculate from market prices)
        # Using historical volatility estimates
        implied_vol = 0.30  # 30% default

        # Determine if call or put
        is_call = (option_type == 'CALL')

        print(f"   Calculating Greeks for {symbol}...")
        print(f"      Underlying: ${spot_price:.2f}")
        print(f"      Strike: ${strike:.2f}")
        print(f"      Days to expiry: {days_to_expiry}")
        print(f"      IV: {implied_vol*100:.1f}%")

        try:
            # Calculate Greeks using OpenMP-accelerated C++ trinomial tree
            greeks = bigbrother_options.calculate_greeks(
                spot=spot_price,
                strike=strike,
                volatility=implied_vol,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                is_call=is_call,
                is_american=True,  # American options
                steps=100  # 100-step trinomial tree
            )

            print(f"      ✅ Delta: {greeks.delta:.4f}")
            print(f"         Gamma: {greeks.gamma:.4f}")
            print(f"         Theta: {greeks.theta:.4f} (per day)")
            print(f"         Vega: {greeks.vega:.4f}")
            print(f"         Rho: {greeks.rho:.4f}")

            # Update database
            conn.execute("""
                UPDATE tax_lots
                SET entry_delta = ?,
                    entry_gamma = ?,
                    entry_theta = ?,
                    entry_vega = ?,
                    entry_rho = ?,
                    entry_iv = ?
                WHERE id = ?
            """, [greeks.delta, greeks.gamma, greeks.theta,
                  greeks.vega, greeks.rho, implied_vol, lot_id])

            greeks_calculated += 1

        except Exception as e:
            print(f"      ❌ Error calculating Greeks: {e}")
            continue

    conn.commit()

    print()
    print("=" * 80)
    print(f"✅ Greeks calculated for {greeks_calculated}/{len(options_df)} options")
    print("=" * 80)
    print("\nGreeks Summary:")

    # Show summary
    summary = conn.execute("""
        SELECT symbol, option_type, entry_delta, entry_gamma,
               entry_theta, entry_vega, entry_rho
        FROM v_open_tax_lots
        WHERE asset_type = 'OPTION'
          AND entry_delta IS NOT NULL
        ORDER BY entry_date DESC
    """).fetchdf()

    if len(summary) > 0:
        print(summary.to_string(index=False))
    else:
        print("   No Greeks data available")

    conn.close()
    print()

    return 0

if __name__ == "__main__":
    exit(calculate_greeks())
