#!/usr/bin/env python3
"""
BigBrotherAnalytics - Free Historical Data Collection

Collects free market data from Yahoo Finance and FRED for backtesting.
No API keys required - 100% free data sources.

Data Sources:
- Yahoo Finance: Stock prices, options chains (10 years free)
- FRED (Federal Reserve): Economic indicators (unlimited free)

Usage:
    uv run python scripts/collect_free_data.py
    uv run python scripts/collect_free_data.py --symbols AAPL,MSFT,GOOGL --years 5
    uv run python scripts/collect_free_data.py --all-sp500

Requirements:
    uv add yfinance pandas pyarrow duckdb fredapi
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import duckdb

def setup_database():
    """Initialize DuckDB database for historical data."""
    db_path = Path("data/bigbrother.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))

    # Create tables if they don't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            dividends DOUBLE,
            stock_splits DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS options_data (
            symbol VARCHAR,
            date DATE,
            strike DOUBLE,
            expiration DATE,
            option_type VARCHAR,
            last_price DOUBLE,
            bid DOUBLE,
            ask DOUBLE,
            volume BIGINT,
            open_interest BIGINT,
            implied_volatility DOUBLE,
            PRIMARY KEY (symbol, date, strike, expiration, option_type)
        )
    """)

    print("✅ Database initialized: data/bigbrother.duckdb")
    return conn

def collect_stock_data(symbols, years=10, conn=None):
    """
    Download stock data from Yahoo Finance (FREE).

    Args:
        symbols: List of stock symbols
        years: Years of history to download
        conn: DuckDB connection

    Returns:
        Number of symbols successfully downloaded
    """
    print(f"\n📈 Collecting {years} years of stock data for {len(symbols)} symbols...")

    parquet_dir = Path("data/historical/stocks")
    parquet_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] Downloading {symbol}...", end=" ")

        try:
            ticker = yf.Ticker(symbol)

            # Get historical data
            df = ticker.history(period=f"{years}y", interval="1d")

            if df.empty:
                print("❌ No data")
                continue

            # Flatten column names
            df = df.reset_index()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Add symbol column
            df['symbol'] = symbol

            # Save to Parquet (compressed, efficient)
            parquet_file = parquet_dir / f"{symbol}_daily.parquet"
            df.to_parquet(parquet_file, compression='snappy', index=False)

            # Insert into DuckDB if connection provided
            if conn:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO stock_prices
                        SELECT
                            symbol,
                            date::DATE,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            dividends,
                            stock_splits
                        FROM read_parquet(?)
                    """, [str(parquet_file)])
                except Exception as e:
                    print(f"⚠️  Parquet saved, DB insert failed: {e}")

            print(f"✅ {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
            success_count += 1

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n✅ Downloaded {success_count}/{len(symbols)} symbols successfully")
    return success_count

def collect_economic_data(conn=None):
    """
    Download economic indicators from FRED (FREE API key required).

    Key indicators for recession detection and trading:
    - Federal Funds Rate
    - 10-Year Treasury Yield
    - 2-Year Treasury Yield
    - Unemployment Rate
    - CPI (inflation)
    - GDP Growth
    """
    print(f"\n📊 Collecting economic indicators from FRED...")

    try:
        from fredapi import Fred
        import yaml
        import os

        # Try to load API key from api_keys.yaml
        api_key = None
        api_keys_file = Path("configs/api_keys.yaml")

        if not api_keys_file.exists():
            api_keys_file = Path("api_keys.yaml")  # Fallback to root

        if api_keys_file.exists():
            with open(api_keys_file) as f:
                keys = yaml.safe_load(f)
                # Support both fred.api_key and fred_api_key formats
                api_key = keys.get('fred', {}).get('api_key') or keys.get('fred_api_key')

        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv('FRED_API_KEY')

        if not api_key:
            print("  ⚠️  No FRED API key found. Skipping economic data.")
            print("  ℹ️  Add key to api_keys.yaml or set FRED_API_KEY env variable")
            print("  ℹ️  Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            return

        fred = Fred(api_key=api_key)

        indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'DGS10': '10-Year Treasury Yield',
            'DGS2': '2-Year Treasury Yield',
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'CPI (Inflation)',
            'GDP': 'GDP'
        }

        for series_id, name in indicators.items():
            print(f"  Downloading {name}...", end=" ")
            try:
                df = fred.get_series(series_id)
                print(f"✅ {len(df)} observations")
                # TODO: Store in DuckDB
            except:
                print(f"⚠️  Requires API key (get free at fred.stlouisfed.org)")

    except ImportError:
        print("  ℹ️  fredapi not installed. Run: uv add fredapi")
        print("  ℹ️  FRED data collection skipped (optional for Tier 1)")

def get_sp500_symbols():
    """Get S&P 500 symbols from Wikipedia (FREE)."""
    print("\n📋 Fetching S&P 500 symbols from Wikipedia...")

    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]

        symbols = sp500_table['Symbol'].tolist()
        # Clean symbols (some have dots that need conversion)
        symbols = [s.replace('.', '-') for s in symbols]

        print(f"✅ Found {len(symbols)} S&P 500 symbols")
        return symbols

    except Exception as e:
        print(f"❌ Failed to fetch S&P 500 list: {e}")
        print("   Using default symbol list instead")
        return DEFAULT_SYMBOLS

# Default symbols (top holdings + indices)
DEFAULT_SYMBOLS = [
    # Indices
    "SPY", "QQQ", "IWM", "DIA",
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Finance
    "JPM", "BAC", "GS", "WFC",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "HD", "MCD",
    # Energy
    "XOM", "CVX"
]

def main():
    """Main data collection workflow."""
    parser = argparse.ArgumentParser(
        description="Collect free historical market data for BigBrotherAnalytics"
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Years of history to download (default: 10)'
    )
    parser.add_argument(
        '--all-sp500',
        action='store_true',
        help='Download all S&P 500 stocks (takes ~2 hours)'
    )
    parser.add_argument(
        '--skip-db',
        action='store_true',
        help='Skip DuckDB insertion (Parquet only)'
    )

    args = parser.parse_args()

    # Banner
    print("╔════════════════════════════════════════════════════════╗")
    print("║   BigBrotherAnalytics - Free Data Collection          ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print(f"📅 Collecting {args.years} years of historical data")
    print(f"💰 Cost: $0 (100% free data sources)")
    print()

    # Setup database
    conn = None if args.skip_db else setup_database()

    # Determine symbols to download
    if args.all_sp500:
        symbols = get_sp500_symbols()
        print(f"⚠️  WARNING: Downloading {len(symbols)} symbols will take 1-2 hours")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
        print(f"Using default {len(symbols)} symbols (indices + top stocks)")

    # Collect stock data
    success = collect_stock_data(symbols, args.years, conn)

    # Collect economic data (optional)
    collect_economic_data(conn)

    # Summary
    print("\n" + "="*60)
    print("📊 Data Collection Summary:")
    print(f"   Stocks downloaded: {success}/{len(symbols)}")
    print(f"   Data location: data/historical/stocks/*.parquet")
    if conn:
        result = conn.execute("SELECT COUNT(DISTINCT symbol) as symbols, COUNT(*) as rows FROM stock_prices").fetchone()
        print(f"   Database: {result[0]} symbols, {result[1]:,} total rows")
        conn.close()
    print()
    print("✅ Data collection complete!")
    print()
    print("Next steps:")
    print("  1. Verify data: uv run python -c \"import duckdb; print(duckdb.connect('data/bigbrother.duckdb').execute('SELECT * FROM stock_prices LIMIT 5').df())\"")
    print("  2. Begin implementation: Implement Black-Scholes pricing")
    print("  3. Backtest first strategy: Iron condor on SPY")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
