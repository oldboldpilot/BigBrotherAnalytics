#!/usr/bin/env python3
"""
Prepare Training Features Using C++ Feature Extractor

This script uses the C++ feature extractor (via Python bindings) to generate
training data. This ensures perfect parity between training and inference.

Single source of truth: C++ feature_extractor.cppm -> Python bindings -> Training data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import duckdb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import feature_extractor_cpp
    print("✅ C++ feature extractor loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import C++ feature extractor: {e}")
    print("   Make sure to build the Python bindings first:")
    print("   ninja -C build feature_extractor_py")
    sys.exit(1)

try:
    import data_loader_cpp
    print("✅ C++ data loader loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import C++ data loader: {e}")
    print("   Make sure to build the Python bindings first:")
    print("   ninja -C build data_loader_py")
    sys.exit(1)

print("="*80)
print("TRAINING DATA PREPARATION - C++ FEATURE EXTRACTOR")
print("="*80)
print()

# ============================================================================
# 1. Load Stock Price Data with Technical Indicators
# ============================================================================
print("[1/5] Loading stock price data...")
conn_training = duckdb.connect('data/training_data.duckdb', read_only=True)

# Load all splits
train_df = conn_training.execute("SELECT * FROM train").df()
val_df = conn_training.execute("SELECT * FROM validation").df()
test_df = conn_training.execute("SELECT * FROM test").df()

# Combine for feature engineering
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)

# Normalize date to remove timezone for comparison with historical data
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

print(f"   Total samples: {len(df):,}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Symbols: {df['symbol'].nunique()} unique")

conn_training.close()

# ============================================================================
# 2. Load Historical Prices for Each Symbol Using C++ Data Loader
# ============================================================================
print("\n[2/5] Loading historical price data using C++ data loader...")
print("   This ensures we use the SAME query as the bot (ORDER BY date DESC)")

# Build cache of historical prices per symbol using C++ data loader
# This uses the EXACT SAME query pattern as the bot: ORDER BY date DESC LIMIT 100
historical_prices_cache = {}
unique_symbols = df['symbol'].unique()

for symbol in unique_symbols:
    try:
        # Use C++ data loader - same query as bot!
        dates, closes, volumes = data_loader_cpp.load_historical_prices(
            'data/bigbrother.duckdb',
            symbol,
            limit=100
        )

        # Store in cache (data is already in MOST RECENT FIRST order)
        historical_prices_cache[symbol] = {
            'dates': dates,
            'closes': closes,
            'volumes': volumes
        }
    except Exception as e:
        print(f"   ⚠️  Warning: Could not load history for {symbol}: {e}")
        continue

print(f"   Loaded historical prices for {len(historical_prices_cache)} symbols")
print(f"   Using C++ data loader (bot's exact query pattern)")

# For risk-free rates, we still need a DuckDB connection
conn_main = duckdb.connect('data/bigbrother.duckdb', read_only=True)

# ============================================================================
# 3. Load Risk-Free Rates
# ============================================================================
print("\n[3/5] Loading risk-free rates...")

# Get latest rates from key-value table
rates_df = conn_main.execute("SELECT * FROM risk_free_rates").df()
rates_dict = dict(zip(rates_df['rate_name'], rates_df['rate_value']))

# Extract the rates we need
fed_funds_rate = rates_dict.get('fed_funds_rate', 4.5)
treasury_10yr = rates_dict.get('10_year_treasury', 4.11)

print(f"   Fed funds rate: {fed_funds_rate:.4f}")
print(f"   Treasury 10Y: {treasury_10yr:.4f}")

conn_main.close()

# ============================================================================
# 4. Extract Features Using C++ Feature Extractor
# ============================================================================
print("\n[4/5] Extracting features using C++ feature extractor...")
print("   This ensures perfect parity with bot inference!")
print()

# Symbol encoding (must match bot's encoding)
symbol_map = {"SPY": 0.0, "QQQ": 1.0, "IWM": 2.0, "DIA": 3.0,
              "AAPL": 4.0, "MSFT": 5.0, "GOOGL": 6.0, "AMZN": 7.0,
              "NVDA": 8.0, "META": 9.0, "TSLA": 10.0}

feature_rows = []
skipped_count = 0

# Process each row
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    symbol = row['symbol']
    date = pd.to_datetime(row['Date'])

    # Get symbol encoding
    symbol_encoded = symbol_map.get(symbol, 0.0)

    # Get historical prices from cache (loaded via C++ data loader)
    # This uses the EXACT SAME query as the bot: ORDER BY date DESC LIMIT 100
    if symbol not in historical_prices_cache:
        skipped_count += 1
        continue

    symbol_data = historical_prices_cache[symbol]
    price_history = symbol_data['closes']  # Already numpy array, MOST RECENT FIRST
    volume_history = symbol_data['volumes']

    if len(price_history) < 21:
        # Not enough history for features
        skipped_count += 1
        continue

    # Extract features using C++ extractor
    try:
        features = feature_extractor_cpp.extract_features_85(
            float(row['close']),  # close
            float(row['open']),   # open
            float(row['high']),   # high
            float(row['low']),    # low
            float(row['volume']), # volume
            float(row['rsi_14']), # rsi_14
            float(row['macd']),   # macd
            float(row['macd_signal']), # macd_signal
            float(row['bb_upper']),    # bb_upper
            float(row['bb_lower']),    # bb_lower
            float(row['bb_position']), # bb_position
            float(row['atr_14']),      # atr_14
            float(row.get('volume_ratio', 1.0)),  # volume_ratio (will be calculated by C++)
            float(row.get('volume_rsi_signal', 0.0)),  # volume_rsi_signal
            float(row.get('yield_volatility', 0.0)),   # yield_volatility
            float(row.get('macd_volume', 0.0)),        # macd_volume
            float(row.get('bb_momentum', 0.0)),        # bb_momentum
            float(row.get('rate_return', 0.0)),        # rate_return
            float(row.get('rsi_bb_signal', 0.0)),      # rsi_bb_signal
            float(row.get('momentum_3d', 0.0)),        # momentum_3d
            float(row.get('recent_win_rate', 0.5)),    # recent_win_rate
            symbol_encoded,  # symbol_encoded
            float(date.dayofweek),     # day_of_week (0=Monday)
            float(date.day),           # day_of_month
            float(date.month),         # month_of_year
            float((date.month - 1) // 3 + 1),  # quarter
            float(date.timetuple().tm_yday),   # day_of_year
            float(row.get('price_direction', 0.0)),      # price_direction
            float(row.get('price_above_ma5', 1.0)),      # price_above_ma5
            float(row.get('price_above_ma20', 1.0)),     # price_above_ma20
            float(row.get('macd_signal_direction', 1.0)), # macd_signal_direction
            float(row.get('volume_trend', 0.0)),         # volume_trend
            int(date.year),   # year
            int(date.month),  # month
            int(date.day),    # day
            price_history,    # price_history (numpy array)
            volume_history,   # volume_history (numpy array)
            fed_funds_rate,   # fed_funds_rate
            treasury_10yr     # treasury_10yr
        )

        # Add target variables
        feature_dict = {f'feature_{i}': features[i] for i in range(85)}
        feature_dict['symbol'] = symbol
        feature_dict['date'] = date
        feature_dict['target_1d'] = row['target_1d']
        feature_dict['target_5d'] = row['target_5d']
        feature_dict['target_20d'] = row['target_20d']

        feature_rows.append(feature_dict)

    except Exception as e:
        print(f"\n⚠️  Error extracting features for {symbol} on {date}: {e}")
        skipped_count += 1
        continue

print()
print(f"   ✅ Extracted features for {len(feature_rows):,} samples")
print(f"   ⚠️  Skipped {skipped_count:,} samples (insufficient history or errors)")

# ============================================================================
# 5. Save Results
# ============================================================================
print("\n[5/5] Saving feature matrix...")

# Convert to DataFrame
features_df = pd.DataFrame(feature_rows)

# Save to parquet (efficient format for ML)
output_path = Path('models/training_data/features_cpp_85.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)
features_df.to_parquet(output_path, compression='snappy', index=False)

print(f"   Saved to: {output_path}")
print(f"   Shape: {features_df.shape}")
print(f"   Features: 85 (from C++ extractor)")
print(f"   Targets: 3 (1d, 5d, 20d price changes)")

# Save metadata
metadata = {
    'created_at': datetime.now().isoformat(),
    'total_samples': len(features_df),
    'total_features': 85,
    'extractor': 'C++ feature_extractor.cppm via Python bindings',
    'symbols': sorted(features_df['symbol'].unique().tolist()),
    'date_range': {
        'start': str(features_df['date'].min()),
        'end': str(features_df['date'].max())
    },
    'target_columns': ['target_1d', 'target_5d', 'target_20d']
}

import json
metadata_path = Path('models/training_data/features_cpp_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   Metadata saved to: {metadata_path}")

print()
print("="*80)
print("✅ FEATURE EXTRACTION COMPLETE")
print("="*80)
print()
print("Single source of truth established:")
print("  C++ feature_extractor.cppm → Python bindings → Training data")
print()
print("Next step: Train model with these C++-extracted features")
print("  uv run python scripts/ml/train_price_predictor_cpp.py")
print()
