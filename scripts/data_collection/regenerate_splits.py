#!/usr/bin/env python3
"""
Regenerate train/val/test splits from feature CSV files
Ensures all 20 symbols are included in the training dataset
"""

from pathlib import Path
import pandas as pd

features_dir = Path('data/historical/features')
output_dir = Path('data/historical')

print("Loading all feature CSV files...")
all_features = []
for csv_file in sorted(features_dir.glob('*_features.csv')):
    symbol = csv_file.stem.replace('_features', '')
    print(f"Loading {symbol}...", end=" ")

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    # Check if 'symbol' column exists, if not add it
    if 'symbol' not in df.columns:
        df['symbol'] = symbol

    print(f"✅ {len(df)} rows")
    all_features.append(df)

print(f"\nCombining {len(all_features)} symbols...")
master_df = pd.concat(all_features, ignore_index=False)

print(f"Total rows before dropna: {len(master_df)}")
print(f"Unique symbols: {sorted(master_df['symbol'].unique())}")
print(f"Symbol count: {len(master_df['symbol'].unique())}")

# Remove rows with NaN (from indicator calculations)
master_df = master_df.dropna()

print(f"Total rows after dropna: {len(master_df)}")
print(f"Symbols remaining: {sorted(master_df['symbol'].unique())}")
print(f"Symbol count: {len(master_df['symbol'].unique())}")

# Create target variables (future returns)
print("\nCreating target variables...")
master_df['target_1d'] = master_df.groupby('symbol')['close'].shift(-1) / master_df['close'] - 1
master_df['target_5d'] = master_df.groupby('symbol')['close'].shift(-5) / master_df['close'] - 1
master_df['target_20d'] = master_df.groupby('symbol')['close'].shift(-20) / master_df['close'] - 1

# Remove last 20 days per symbol (no target data)
master_df = master_df.dropna()

print(f"Total rows after creating targets: {len(master_df)}")
print(f"Symbols with targets: {sorted(master_df['symbol'].unique())}")
print(f"Symbol count: {len(master_df['symbol'].unique())}")

# Save master dataset
master_path = output_dir / 'master_training_dataset.csv'
master_df.to_csv(master_path)
print(f"\n✅ Master dataset: {master_path}")
print(f"   Total samples: {len(master_df):,}")
print(f"   Features: {len(master_df.columns)}")
print(f"   Date range: {master_df.index.min()} to {master_df.index.max()}")

# Split into train/validation/test (stratified by symbol to ensure all symbols in each split)
print("\nPerforming stratified split by symbol...")
train_dfs = []
val_dfs = []
test_dfs = []

for symbol in master_df['symbol'].unique():
    symbol_df = master_df[master_df['symbol'] == symbol].sort_index()

    train_size = int(len(symbol_df) * 0.7)
    val_size = int(len(symbol_df) * 0.15)

    train_dfs.append(symbol_df.iloc[:train_size])
    val_dfs.append(symbol_df.iloc[train_size:train_size+val_size])
    test_dfs.append(symbol_df.iloc[train_size+val_size:])

train_df = pd.concat(train_dfs)
val_df = pd.concat(val_dfs)
test_df = pd.concat(test_dfs)

train_df.to_csv(output_dir / 'train_dataset.csv')
val_df.to_csv(output_dir / 'val_dataset.csv')
test_df.to_csv(output_dir / 'test_dataset.csv')

print(f"\n✅ Dataset split:")
print(f"   Training: {len(train_df):,} samples (70%) - {len(train_df['symbol'].unique())} symbols")
print(f"   Validation: {len(val_df):,} samples (15%) - {len(val_df['symbol'].unique())} symbols")
print(f"   Test: {len(test_df):,} samples (15%) - {len(test_df['symbol'].unique())} symbols")

print(f"\n✅ ALL DONE! Ready to convert to DuckDB and train model.")
