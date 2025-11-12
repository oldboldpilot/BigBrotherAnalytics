#!/usr/bin/env python3
"""
Update Live Prices from Schwab API

Fetches REAL-TIME prices directly from Schwab (no 15-min delay like Yahoo).
Uses existing OAuth credentials - no additional cost.

Schwab API Provides:
- Real-time quotes (equities and options)
- Bid/ask spreads
- Volume and day high/low
- Last trade timestamp
- Options chains with Greeks

Usage:
    uv run python scripts/update_schwab_prices.py
    uv run python scripts/update_schwab_prices.py --symbols SPY,AAPL,NVDA
"""

import schwab
import duckdb
from pathlib import Path
from datetime import datetime
import sys

def update_schwab_prices(symbols=None):
    """Fetch real-time prices from Schwab API"""

    print("=" * 80)
    print("Updating Prices from Schwab API (Real-Time)")
    print("=" * 80)

    # Load Schwab client
    token_file = Path('configs/schwab_tokens.json')
    app_key = '8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa'
    app_secret = 'PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT'

    if not token_file.exists():
        print(f"❌ Token file not found: {token_file}")
        return 1

    print("\n🔐 Loading Schwab OAuth client...")
    client = schwab.auth.client_from_token_file(
        str(token_file),
        app_key,
        app_secret
    )

    # Connect to database
    db_path = Path('data/bigbrother.duckdb')
    conn = duckdb.connect(str(db_path))

    # Get symbols from open positions if not provided
    if not symbols:
        print("\n📊 Fetching symbols from open positions...")
        symbols_query = """
            SELECT DISTINCT
                CASE
                    WHEN asset_type = 'OPTION' THEN
                        -- Extract underlying symbol from option symbol
                        TRIM(SUBSTRING(symbol, 1, POSITION(' ' IN symbol)))
                    ELSE symbol
                END as underlying_symbol
            FROM tax_lots
            WHERE is_closed = false
        """
        positions = conn.execute(symbols_query).fetchall()
        symbols = list(set([p[0] for p in positions if p[0]]))

    print(f"   Found {len(symbols)} symbols: {', '.join(symbols)}")

    # Create price_history table if not exists
    print("\n📦 Ensuring price_history table exists...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            bid DOUBLE,
            ask DOUBLE,
            last_price DOUBLE,
            source VARCHAR DEFAULT 'schwab',
            PRIMARY KEY (symbol, timestamp)
        )
    """)

    # Fetch quotes from Schwab
    print("\n💰 Fetching real-time quotes from Schwab...")
    success_count = 0
    failed_symbols = []

    for symbol in symbols:
        try:
            print(f"   {symbol:10s} ... ", end="", flush=True)

            # Get quote from Schwab
            response = client.get_quote(symbol)

            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}")
                failed_symbols.append(symbol)
                continue

            quote_data = response.json()

            # Schwab returns: {symbol: {quote: {...}}}
            if symbol not in quote_data:
                print("❌ No quote data")
                failed_symbols.append(symbol)
                continue

            quote = quote_data[symbol]['quote']

            # Extract price data
            last_price = quote.get('lastPrice', 0)
            bid_price = quote.get('bidPrice', 0)
            ask_price = quote.get('askPrice', 0)
            open_price = quote.get('openPrice', 0)
            high_price = quote.get('highPrice', 0)
            low_price = quote.get('lowPrice', 0)
            volume = quote.get('totalVolume', 0)

            # Get quote timestamp (milliseconds since epoch)
            quote_time_ms = quote.get('quoteTime', 0)
            if quote_time_ms:
                quote_timestamp = datetime.fromtimestamp(quote_time_ms / 1000.0)
            else:
                quote_timestamp = datetime.now()

            # Insert/update price history
            conn.execute("""
                INSERT OR REPLACE INTO price_history
                (symbol, timestamp, open, high, low, close, volume,
                 bid, ask, last_price, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'schwab')
            """, [
                symbol,
                quote_timestamp,
                float(open_price),
                float(high_price),
                float(low_price),
                float(last_price),  # Use last_price as close
                int(volume),
                float(bid_price),
                float(ask_price),
                float(last_price)
            ])

            # Calculate bid-ask spread
            spread = ask_price - bid_price if (bid_price and ask_price) else 0

            print(f"✅ ${last_price:8.2f}  Bid: ${bid_price:8.2f}  Ask: ${ask_price:8.2f}  Spread: ${spread:.4f}")
            success_count += 1

        except Exception as e:
            print(f"❌ Error: {e}")
            failed_symbols.append(symbol)

    conn.commit()

    # Summary
    print("\n" + "=" * 80)
    print("UPDATE SUMMARY")
    print("=" * 80)
    print(f"   ✅ Successfully updated: {success_count}/{len(symbols)}")

    if failed_symbols:
        print(f"   ❌ Failed symbols: {', '.join(failed_symbols)}")

    # Show latest prices
    print("\n📈 Latest Prices in Database (Real-Time from Schwab):")
    latest_prices = conn.execute("""
        SELECT
            symbol,
            timestamp,
            last_price,
            bid,
            ask,
            volume,
            source
        FROM price_history
        WHERE timestamp = (
            SELECT MAX(timestamp)
            FROM price_history p2
            WHERE p2.symbol = price_history.symbol
        )
        AND source = 'schwab'
        ORDER BY symbol
    """).fetchdf()

    print(latest_prices.to_string(index=False))

    print("\n✅ Advantage: Real-time data (no 15-min delay)")
    print("💡 Tip: Use Schwab for prices, Yahoo Finance for news")
    print()

    conn.close()
    return 0

def main():
    """Main entry point"""

    # Parse symbols from command line
    symbols = None
    for arg in sys.argv[1:]:
        if arg.startswith('--symbols='):
            symbols = arg.split('=')[1].split(',')

    return update_schwab_prices(symbols)

if __name__ == "__main__":
    exit(main())
