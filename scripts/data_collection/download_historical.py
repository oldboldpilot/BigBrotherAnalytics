#!/usr/bin/env python3
"""
Historical Data Downloader

Downloads historical market data from free sources:
- Yahoo Finance: Stock prices, options data
- FRED: Economic indicators

Stores in Parquet format for efficient loading by C++ engine.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from fredapi import Fred
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to download (per config.yaml)
SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",  # Market ETFs
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Mega-cap tech
    "NVDA", "AMD", "INTC", "TSM",  # Semiconductors
    "TSLA", "F", "GM",  # Automotive
    "JPM", "BAC", "GS", "C",  # Banks
    "XLE", "XLF", "XLK", "XLV",  # Sector ETFs
]

# FRED economic indicators
FRED_SERIES = {
    "DGS10": "10-Year Treasury Rate",
    "DFF": "Federal Funds Rate",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "GDP": "Gross Domestic Product",
    "VIXCLS": "CBOE Volatility Index",
}


def download_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Download stock data from Yahoo Finance"""

    try:
        logger.info(f"Downloading {symbol}...")

        ticker = yf.Ticker(symbol)

        # Download daily data
        df = ticker.history(start=start_date, end=end_date, interval="1d")

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Flatten multi-index columns
        df = df.reset_index()

        # Rename columns to match our schema
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        # Add symbol column
        df["symbol"] = symbol

        # Select relevant columns
        df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

        # Convert timestamp to microseconds since epoch
        df["timestamp"] = df["timestamp"].astype(np.int64) // 1000

        logger.info(f"Downloaded {len(df)} bars for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return None


def download_options_data(symbol: str) -> pd.DataFrame | None:
    """Download current options chain from Yahoo Finance"""

    try:
        logger.info(f"Downloading options chain for {symbol}...")

        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            logger.warning(f"No options available for {symbol}")
            return None

        # Get options for next 3 expirations
        all_options = []

        for exp in expirations[:3]:
            opt = ticker.option_chain(exp)

            # Calls
            calls = opt.calls.copy()
            calls["type"] = "call"
            calls["expiration"] = exp

            # Puts
            puts = opt.puts.copy()
            puts["type"] = "put"
            puts["expiration"] = exp

            all_options.extend([calls, puts])

        df = pd.concat(all_options, ignore_index=True)
        df["symbol"] = symbol

        logger.info(f"Downloaded {len(df)} options for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Failed to download options for {symbol}: {e}")
        return None


def download_economic_data(series_id: str, name: str, fred_api_key: str) -> pd.DataFrame | None:
    """Download economic data from FRED"""

    try:
        logger.info(f"Downloading {name} ({series_id})...")

        fred = Fred(api_key=fred_api_key)

        # Get data for last 10 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)

        data = fred.get_series(series_id, start_date, end_date)

        df = pd.DataFrame({
            "series_id": series_id,
            "timestamp": data.index.astype(np.int64) // 1'000'000,  # Convert to microseconds
            "value": data.values
        })

        logger.info(f"Downloaded {len(df)} observations for {series_id}")

        return df

    except Exception as e:
        logger.error(f"Failed to download {series_id}: {e}")
        return None


def main():
    """Main data download process"""

    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║    BigBrotherAnalytics Historical Data Downloader         ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    logger.info("")

    # Date range: 10 years of history
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Symbols: {len(SYMBOLS)}")
    logger.info("")

    # Download stock data
    logger.info("=== Downloading Stock Data ===")

    all_stock_data = []

    for symbol in SYMBOLS:
        df = download_stock_data(symbol, start_date, end_date)

        if df is not None:
            all_stock_data.append(df)

            # Save individual file
            output_file = DATA_DIR / f"{symbol}.parquet"
            df.to_parquet(output_file, engine="pyarrow", compression="snappy")
            logger.info(f"Saved: {output_file}")

    # Combine all stock data
    if all_stock_data:
        combined = pd.concat(all_stock_data, ignore_index=True)
        combined_file = DATA_DIR / "all_stocks.parquet"
        combined.to_parquet(combined_file, engine="pyarrow", compression="snappy")
        logger.info(f"Combined data saved: {combined_file} ({len(combined)} rows)")

    # Download options data
    logger.info("")
    logger.info("=== Downloading Options Data ===")

    for symbol in ["SPY", "QQQ", "AAPL", "NVDA"]:  # Limited set for options
        df = download_options_data(symbol)

        if df is not None:
            output_file = DATA_DIR / f"{symbol}_options.parquet"
            df.to_parquet(output_file, engine="pyarrow", compression="snappy")
            logger.info(f"Saved: {output_file}")

    # Download economic data
    logger.info("")
    logger.info("=== Downloading Economic Data ===")

    fred_api_key = os.getenv("FRED_API_KEY")

    if fred_api_key:
        for series_id, name in FRED_SERIES.items():
            df = download_economic_data(series_id, name, fred_api_key)

            if df is not None:
                output_file = DATA_DIR / f"fred_{series_id}.parquet"
                df.to_parquet(output_file, engine="pyarrow", compression="snappy")
                logger.info(f"Saved: {output_file}")
    else:
        logger.warning("FRED_API_KEY not set, skipping economic data")

    logger.info("")
    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("Download complete!")
    logger.info(f"Data saved to: {DATA_DIR}")
    logger.info("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
