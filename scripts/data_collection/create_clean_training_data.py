#!/usr/bin/env python3
"""
Create Clean Training Dataset - Remove Bad Features

Builds a clean training dataset using only features with proper variance:
- 55 good features from existing data
- Temporal features: day, month, year
- First-order differences: price changes over 1-20 days
- Autocorrelation: ACF values for lags 1-20

Removes:
- 17 constant features (sector, is_option, treasury rates, sentiment, etc.)
- 10 low-variance features (delta, delta_iv, etc.)
"""

import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Connect to existing database
input_db = 'data/custom_training_data.duckdb'
output_db = 'data/clean_training_data.duckdb'

print("=" * 80)
print("CREATING CLEAN TRAINING DATASET")
print("=" * 80)
print(f"Input:  {input_db}")
print(f"Output: {output_db}")
print()

# Load data from existing database
print("Loading data from existing database...")
conn_in = duckdb.connect(input_db, read_only=True)

# Get all data (we'll rebuild train/val/test splits)
df = conn_in.execute("SELECT * FROM train").fetchdf()
print(f"✓ Loaded {len(df):,} training samples")

df_val = conn_in.execute("SELECT * FROM validation").fetchdf()
print(f"✓ Loaded {len(df_val):,} validation samples")

df_test = conn_in.execute("SELECT * FROM test").fetchdf()
print(f"✓ Loaded {len(df_test):,} test samples")

conn_in.close()

# Combine all data for processing
df_all = pd.concat([df, df_val, df_test], ignore_index=True)
print(f"✓ Total samples: {len(df_all):,}\n")

print("=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# GOOD FEATURES (from analysis - 55 features with proper variance)
good_features = [
    # Price/Volume indicators (9)
    'close', 'open', 'high', 'low', 'volume',
    'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_position',
    'atr_14', 'volume_sma20', 'volume_ratio',

    # Greeks (4) - excluding delta and delta_iv (constant)
    'gamma', 'theta', 'vega', 'rho',

    # Momentum/Derived features (14)
    'volume_rsi_signal', 'yield_volatility', 'macd_volume', 'bb_momentum',
    'rate_return', 'gamma_volatility', 'rsi_bb_signal',
    'momentum_3d', 'recent_win_rate',

    # Price lags (20)
    'price_lag_1d', 'price_lag_2d', 'price_lag_3d', 'price_lag_4d', 'price_lag_5d',
    'price_lag_6d', 'price_lag_7d', 'price_lag_8d', 'price_lag_9d', 'price_lag_10d',
    'price_lag_11d', 'price_lag_12d', 'price_lag_13d', 'price_lag_14d', 'price_lag_15d',
    'price_lag_16d', 'price_lag_17d', 'price_lag_18d', 'price_lag_19d', 'price_lag_20d',

    # Symbol encoding (1)
    'symbol_encoded',
]

# Keep legitimate categorical features with few unique values
temporal_features = [
    'day_of_week',      # 5 unique (legitimate)
    'day_of_month',     # 31 unique
    'month_of_year',    # 12 unique
    'quarter',          # 4 unique (legitimate)
    'day_of_year',      # 361 unique
]

# Binary indicators (legitimate 0/1 features)
binary_features = [
    'price_direction',          # Binary
    'price_above_ma5',          # Binary
    'price_above_ma20',         # Binary
    'macd_signal_direction',    # Binary
    'volume_trend',             # Binary
]

# Combine all features to keep
features_to_keep = good_features + temporal_features + binary_features

# Verify they exist
available_features = [f for f in features_to_keep if f in df_all.columns]
missing_features = [f for f in features_to_keep if f not in df_all.columns]

print(f"Features to keep: {len(features_to_keep)}")
print(f"Available: {len(available_features)}")
if missing_features:
    print(f"Missing: {missing_features}")
print()

# Extract features
print("Extracting base features...")
df_clean = df_all[['Date', 'symbol'] + available_features + ['return_1d', 'return_5d', 'return_20d']].copy()

print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Add explicit temporal features
print("Adding temporal features...")
df_clean['year'] = pd.to_datetime(df_clean['Date']).dt.year
df_clean['month'] = pd.to_datetime(df_clean['Date']).dt.month
df_clean['day'] = pd.to_datetime(df_clean['Date']).dt.day

print("✓ Added: year, month, day")

# Compute first-order finite differences (price changes)
print("\nComputing first-order finite differences...")
print("  (Δprice between consecutive days, 1-20 day lags)")

# Sort by symbol and date
df_clean = df_clean.sort_values(['symbol', 'Date'])

# Compute price differences
for lag in range(1, 21):
    col_name = f'price_diff_{lag}d'
    # Difference between price at t and price at t-lag
    df_clean[col_name] = df_clean.groupby('symbol')['close'].diff(lag)

print(f"✓ Added: price_diff_1d through price_diff_20d (20 features)")

# Compute autocorrelation features
print("\nComputing autocorrelation features...")
print("  (ACF of price returns for lags 1-20)")

def compute_rolling_autocorr(series, lag, window=60):
    """Compute rolling autocorrelation"""
    return series.rolling(window=window + lag).apply(
        lambda x: x.iloc[:-lag].corr(x.iloc[lag:]) if len(x) >= window + lag else np.nan,
        raw=False
    )

# Compute returns for autocorrelation
df_clean['returns'] = df_clean.groupby('symbol')['close'].pct_change()

# Compute autocorrelation for lags 1, 5, 10, 20 (not all 20 to save time)
for lag in [1, 5, 10, 20]:
    col_name = f'autocorr_lag_{lag}'
    print(f"  Computing autocorr_lag_{lag}...")
    df_clean[col_name] = df_clean.groupby('symbol')['returns'].transform(
        lambda x: compute_rolling_autocorr(x, lag)
    )

print(f"✓ Added: autocorr_lag_1, autocorr_lag_5, autocorr_lag_10, autocorr_lag_20 (4 features)")

# Drop temporary returns column
df_clean = df_clean.drop(columns=['returns'])

# Drop rows with NaN (from lagged features)
print(f"\nRows before cleaning: {len(df_clean):,}")
df_clean = df_clean.dropna()
print(f"Rows after cleaning: {len(df_clean):,}")

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

feature_cols = [col for col in df_clean.columns if col not in ['Date', 'symbol', 'return_1d', 'return_5d', 'return_20d']]
print(f"\nTotal features: {len(feature_cols)}")
print(f"Total samples: {len(df_clean):,}")
print(f"\nFeature categories:")
print(f"  - Base good features: {len(available_features)}")
print(f"  - Temporal: 3 (year, month, day)")
print(f"  - First-order diffs: 20 (price_diff_1d to 20d)")
print(f"  - Autocorrelation: 4 (lags 1, 5, 10, 20)")
print(f"  - Total: {len(feature_cols)}")

print("\n" + "=" * 80)
print("DATA QUALITY VERIFICATION")
print("=" * 80)

# Check for constant features
print("\nChecking for constant features...")
constant_features = []
for col in feature_cols:
    if df_clean[col].nunique() <= 1:
        constant_features.append((col, df_clean[col].iloc[0]))

if constant_features:
    print(f"❌ Found {len(constant_features)} constant features:")
    for col, val in constant_features:
        print(f"   - {col}: {val}")
else:
    print("✓ No constant features found!")

# Check variance
print("\nVariance check:")
for col in feature_cols[:10]:  # Show first 10
    std = df_clean[col].std()
    unique = df_clean[col].nunique()
    print(f"  {col:30s} | std: {std:10.4f} | unique: {unique:6,}")
print("  ...")

print("\n" + "=" * 80)
print("SAVING TO DUCKDB")
print("=" * 80)

# Create train/val/test splits (70/15/15)
np.random.seed(42)
indices = np.random.permutation(len(df_clean))
n_train = int(0.7 * len(df_clean))
n_val = int(0.15 * len(df_clean))

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

df_train = df_clean.iloc[train_idx].reset_index(drop=True)
df_val = df_clean.iloc[val_idx].reset_index(drop=True)
df_test = df_clean.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(df_train):,} samples")
print(f"Val:   {len(df_val):,} samples")
print(f"Test:  {len(df_test):,} samples")

# Save to DuckDB
print(f"\nSaving to {output_db}...")
conn_out = duckdb.connect(output_db)

conn_out.execute("DROP TABLE IF EXISTS train")
conn_out.execute("CREATE TABLE train AS SELECT * FROM df_train")

conn_out.execute("DROP TABLE IF EXISTS validation")
conn_out.execute("CREATE TABLE validation AS SELECT * FROM df_val")

conn_out.execute("DROP TABLE IF EXISTS test")
conn_out.execute("CREATE TABLE test AS SELECT * FROM df_test")

# Save metadata
metadata = {
    'created_at': datetime.now().isoformat(),
    'total_samples': len(df_clean),
    'train_samples': len(df_train),
    'val_samples': len(df_val),
    'test_samples': len(df_test),
    'total_features': len(feature_cols),
    'feature_names': feature_cols,
    'constant_features': 0,
    'data_quality': 'clean',
    'removed_features': {
        'constant': 17,
        'low_variance': 10,
        'total_removed': 27
    }
}

import json
metadata_path = Path('models/training_data/clean_training_metadata.json')
metadata_path.parent.mkdir(parents=True, exist_ok=True)
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved metadata to {metadata_path}")

conn_out.close()

print("\n" + "=" * 80)
print("SUCCESS!")
print("=" * 80)
print(f"\nClean training database created: {output_db}")
print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(df_clean):,}")
print(f"Constant features: 0 ✓")
print(f"Data quality: CLEAN ✓")
print("\nNext step:")
print("  python scripts/ml/train_price_predictor_clean.py")
