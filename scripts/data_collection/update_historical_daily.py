#!/usr/bin/env python3
"""
Daily Historical Data Updater

Updates the stock_prices table in bigbrother.duckdb with the latest prices.
Designed to run daily via cron to keep historical data current.

Usage:
    python update_historical_daily.py

Cron example (run daily at 7 PM after market close):
    0 19 * * 1-5 cd /home/muyiwa/Development/BigBrotherAnalytics && uv run python scripts/data_collection/update_historical_daily.py >> logs/data_update.log 2>&1
"""

import yfinance as yf
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Symbols to update (from config)
SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",  # Market ETFs
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Mega-cap tech
    "NVDA", "AMD", "INTC", "TSM",  # Semiconductors
    "TSLA", "F", "GM",  # Automotive
    "JPM", "BAC", "GS", "C",  # Banks
    "XLE", "XLF", "XLK", "XLV",  # Sector ETFs
]

def update_historical_data():
    """Update historical stock prices in the database"""

    print("=" * 80)
    print(f"DAILY HISTORICAL DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    db_path = Path("data/bigbrother.duckdb")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    # Get date range (last 30 days to ensure we fill any gaps)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print(f"Updating data from {start_date.date()} to {end_date.date()}")
    print(f"Symbols to update: {len(SYMBOLS)}")
    print()

    try:
        conn = duckdb.connect(str(db_path))

        success_count = 0
        error_count = 0
        updated_rows = 0

        for i, symbol in enumerate(SYMBOLS, 1):
            try:
                print(f"[{i}/{len(SYMBOLS)}] {symbol:8s} ... ", end="", flush=True)

                # Download recent data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1d")

                if df.empty:
                    print("❌ No data")
                    error_count += 1
                    continue

                # Prepare data
                df = df.reset_index()
                df['symbol'] = symbol

                # Insert or replace (upsert)
                for _, row in df.iterrows():
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO stock_prices
                            (symbol, date, open, high, low, close, volume, dividends, stock_splits)
                            VALUES (?, ?::DATE, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            symbol,
                            row['Date'].strftime('%Y-%m-%d'),
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row['Volume']),
                            float(row.get('Dividends', 0)),
                            float(row.get('Stock Splits', 0))
                        ])
                        updated_rows += 1
                    except Exception as e:
                        print(f"\n   ⚠️  Row insert failed: {e}")

                print(f"✅ {len(df)} days updated")
                success_count += 1

            except Exception as e:
                print(f"❌ Error: {e}")
                error_count += 1

        conn.commit()
        conn.close()

        # Summary
        print()
        print("=" * 80)
        print("UPDATE SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {success_count}/{len(SYMBOLS)}")
        print(f"❌ Errors: {error_count}")
        print(f"📊 Total rows updated: {updated_rows}")

        # Verify data
        conn = duckdb.connect(str(db_path), read_only=True)
        latest_dates = conn.execute("""
            SELECT symbol, MAX(date) as latest_date
            FROM stock_prices
            WHERE symbol IN (?, ?, ?, ?)
            GROUP BY symbol
            ORDER BY symbol
        """, ['SPY', 'QQQ', 'IWM', 'DIA']).fetchall()

        print()
        print("Latest dates in database:")
        for symbol, date in latest_dates:
            print(f"  {symbol}: {date}")

        conn.close()

        print("=" * 80)
        print()

        return success_count > 0

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = update_historical_data()
    sys.exit(0 if success else 1)
