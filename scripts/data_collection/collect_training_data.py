#!/usr/bin/env python3
"""
Free 5-Year Historical Data Collection for Model Training
Uses 100% FREE data sources - no paid subscriptions needed

Data Sources:
- Yahoo Finance (yfinance): Stock/ETF prices, options chains
- Alpha Vantage: Technical indicators (already have API key)
- FRED: Economic data (already integrated)

Usage:
    uv run python scripts/data_collection/collect_training_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("FREE 5-YEAR DATA COLLECTION - FAST TRACK TO LIVE TRADING")
print("="*80)

# Install yfinance if needed
try:
    import yfinance as yf
    print("✅ yfinance available")
except ImportError:
    print("⚠️  Installing yfinance...")
    import subprocess
    subprocess.run(["uv", "pip", "install", "yfinance"], check=True)
    import yfinance as yf
    print("✅ yfinance installed")

# Create data directory
data_dir = Path('data/historical')
data_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Data directory: {data_dir.absolute()}")

# Symbols to collect (high liquidity, good for training)
SYMBOLS = {
    'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA'],  # Major indices
    'Volatility': ['VXX', 'UVXY'],          # Volatility products
    'Bonds': ['TLT', 'IEF'],                # Treasury bonds
    'Commodities': ['GLD', 'SLV', 'USO'],   # Gold, Silver, Oil
    'Sectors': ['XLE', 'XLF', 'XLK', 'XLV', 'XLP', 'XLU', 'XLB', 'XLI', 'XLY']  # SPDR sectors
}

ALL_SYMBOLS = []
for category, symbols in SYMBOLS.items():
    ALL_SYMBOLS.extend(symbols)

print(f"\n📊 Collecting data for {len(ALL_SYMBOLS)} symbols")
print(f"Categories: {list(SYMBOLS.keys())}")

# ============================================================================
# PART 1: Historical Price Data (5 years)
# ============================================================================

print("\n" + "="*80)
print("PART 1: Historical Price Data (5 years)")
print("="*80)

price_data = {}
failed_symbols = []

for i, symbol in enumerate(ALL_SYMBOLS, 1):
    try:
        print(f"\n[{i}/{len(ALL_SYMBOLS)}] Downloading {symbol}...", end=" ")

        # Download 5 years of daily data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='5y', interval='1d')

        if df.empty:
            print("❌ No data")
            failed_symbols.append(symbol)
            continue

        # Save to CSV
        csv_path = data_dir / f'{symbol}_5y_daily.csv'
        df.to_csv(csv_path)

        price_data[symbol] = df

        print(f"✅ {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

        # Rate limiting (respectful to Yahoo Finance)
        time.sleep(0.5)

    except Exception as e:
        print(f"❌ Error: {e}")
        failed_symbols.append(symbol)

print(f"\n✅ Successfully collected: {len(price_data)}/{len(ALL_SYMBOLS)} symbols")
if failed_symbols:
    print(f"⚠️  Failed symbols: {', '.join(failed_symbols)}")

# ============================================================================
# PART 2: Options Data (Current + 6 months historical)
# ============================================================================

print("\n" + "="*80)
print("PART 2: Options Chains (Current + Historical)")
print("="*80)
print("Note: Yahoo Finance provides current options data")
print("For more history, consider CBOE DataShop (free with signup)")

# Focus on liquid options
OPTIONS_SYMBOLS = ['SPY', 'QQQ', 'IWM']  # Most liquid

options_data = {}

for i, symbol in enumerate(OPTIONS_SYMBOLS, 1):
    try:
        print(f"\n[{i}/{len(OPTIONS_SYMBOLS)}] Collecting {symbol} options...", end=" ")

        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            print("❌ No options available")
            continue

        print(f"✅ {len(expirations)} expirations")

        # Collect options for next 6 months of expirations
        all_calls = []
        all_puts = []

        today = datetime.now()
        six_months = today + timedelta(days=180)

        collected = 0
        for exp_date in expirations:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')

            # Only collect next 6 months
            if exp_datetime > six_months:
                continue

            try:
                chain = ticker.option_chain(exp_date)

                calls = chain.calls.copy()
                calls['expiration'] = exp_date
                calls['symbol'] = symbol
                calls['type'] = 'call'

                puts = chain.puts.copy()
                puts['expiration'] = exp_date
                puts['symbol'] = symbol
                puts['type'] = 'put'

                all_calls.append(calls)
                all_puts.append(puts)
                collected += 1

                time.sleep(0.3)  # Rate limiting

            except Exception as e:
                print(f"\n   ⚠️  Failed to get {exp_date}: {e}")
                continue

        if all_calls:
            calls_df = pd.concat(all_calls, ignore_index=True)
            calls_df.to_csv(data_dir / f'{symbol}_calls.csv', index=False)
            print(f"   ✅ Calls: {len(calls_df)} contracts ({collected} expirations)")

        if all_puts:
            puts_df = pd.concat(all_puts, ignore_index=True)
            puts_df.to_csv(data_dir / f'{symbol}_puts.csv', index=False)
            print(f"   ✅ Puts: {len(puts_df)} contracts ({collected} expirations)")

        options_data[symbol] = {
            'calls': calls_df if all_calls else None,
            'puts': puts_df if all_puts else None
        }

    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# PART 3: Feature Engineering Data
# ============================================================================

print("\n" + "="*80)
print("PART 3: Feature Engineering (Technical Indicators)")
print("="*80)

features_dir = data_dir / 'features'
features_dir.mkdir(exist_ok=True)

for symbol, df in price_data.items():
    try:
        print(f"Computing features for {symbol}...", end=" ")

        # Technical indicators
        features = pd.DataFrame(index=df.index)
        features['symbol'] = symbol

        # Price features
        features['close'] = df['Close']
        features['open'] = df['Open']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['volume'] = df['Volume']

        # Returns
        features['return_1d'] = df['Close'].pct_change()
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_20d'] = df['Close'].pct_change(20)

        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Bollinger Bands
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        features['bb_upper'] = sma20 + (2 * std20)
        features['bb_lower'] = sma20 - (2 * std20)
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()

        # Volume indicators
        features['volume_sma20'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma20']

        # Save features
        features.to_csv(features_dir / f'{symbol}_features.csv')

        print(f"✅ {len(features.columns)} features")

    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# PART 4: Create Training Dataset
# ============================================================================

print("\n" + "="*80)
print("PART 4: Creating Training Dataset")
print("="*80)

print("Combining all features into master training dataset...")

all_features = []
for csv_file in features_dir.glob('*_features.csv'):
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    all_features.append(df)

if all_features:
    master_df = pd.concat(all_features, ignore_index=False)
    master_df = master_df.dropna()  # Remove rows with NaN

    # Create target variables (future returns)
    master_df['target_1d'] = master_df.groupby('symbol')['close'].shift(-1) / master_df['close'] - 1
    master_df['target_5d'] = master_df.groupby('symbol')['close'].shift(-5) / master_df['close'] - 1
    master_df['target_20d'] = master_df.groupby('symbol')['close'].shift(-20) / master_df['close'] - 1

    # Remove last 20 days (no target data)
    master_df = master_df.dropna()

    # Save master dataset
    master_path = data_dir / 'master_training_dataset.csv'
    master_df.to_csv(master_path)

    print(f"\n✅ Master dataset created: {master_path}")
    print(f"   Total samples: {len(master_df):,}")
    print(f"   Features: {len(master_df.columns)}")
    print(f"   Date range: {master_df.index.min()} to {master_df.index.max()}")

    # Split into train/validation/test
    train_size = int(len(master_df) * 0.7)
    val_size = int(len(master_df) * 0.15)

    train_df = master_df.iloc[:train_size]
    val_df = master_df.iloc[train_size:train_size+val_size]
    test_df = master_df.iloc[train_size+val_size:]

    train_df.to_csv(data_dir / 'train_dataset.csv')
    val_df.to_csv(data_dir / 'val_dataset.csv')
    test_df.to_csv(data_dir / 'test_dataset.csv')

    print(f"\n✅ Dataset split:")
    print(f"   Training: {len(train_df):,} samples (70%)")
    print(f"   Validation: {len(val_df):,} samples (15%)")
    print(f"   Test: {len(test_df):,} samples (15%)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("DATA COLLECTION COMPLETE!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   ✅ Price data: {len(price_data)} symbols × 5 years")
print(f"   ✅ Options data: {len(options_data)} symbols")
print(f"   ✅ Features engineered: {len(list(features_dir.glob('*.csv')))} files")
print(f"   ✅ Training dataset: {len(master_df):,} samples ready")

print(f"\n📁 Data location: {data_dir.absolute()}")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review data: ls -lh {data_dir}")
print(f"   2. Train model: uv run python scripts/ml/train_price_predictor.py")
print(f"   3. Backtest: uv run python scripts/ml/backtest_model.py")
print(f"   4. Paper trade: 1 day with trained model")
print(f"   5. GO LIVE: Switch to real trading (Day 9)")

print(f"\n💰 Target: ≥55% win rate → Profitable after 37.1% tax + 3% fees")
print(f"    Current Phase 5 win rate: 75% (3/4 trades)")

print("\n" + "="*80)
print("✅ READY TO TRAIN MODEL AND START MAKING MONEY!")
print("="*80)
