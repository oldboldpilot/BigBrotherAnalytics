#!/usr/bin/env python3
"""
Convert CSV training data to DuckDB (compressed, optimized for training)

Benefits:
- 10-100x faster queries than CSV
- Built-in compression (2-5x smaller)
- SQL interface for feature engineering
- Zero-copy to PyTorch/NumPy

Usage:
    uv run python scripts/data_collection/convert_to_duckdb.py
"""

import sys
from pathlib import Path
import pandas as pd
import duckdb
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("CONVERT TRAINING DATA TO DUCKDB (COMPRESSED)")
print("="*80)

# Paths
data_dir = Path('data/historical')
db_path = Path('data/training_data.duckdb')

# Connect to DuckDB
print(f"\n📁 Creating DuckDB: {db_path}")
conn = duckdb.connect(str(db_path))

# Enable compression
conn.execute("PRAGMA threads=8")  # Use multiple threads
conn.execute("PRAGMA memory_limit='4GB'")

print("✅ Connected to DuckDB")

# ============================================================================
# 1. Load Price Data
# ============================================================================

print("\n" + "="*80)
print("PART 1: Loading Price Data")
print("="*80)

price_files = list(data_dir.glob('*_5y_daily.csv'))
print(f"Found {len(price_files)} price files")

for i, csv_file in enumerate(price_files, 1):
    symbol = csv_file.stem.split('_')[0]
    print(f"[{i}/{len(price_files)}] Loading {symbol}...", end=" ")

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df['symbol'] = symbol
    df = df.reset_index().rename(columns={'index': 'date'})

    # Create or insert into table
    if i == 1:
        conn.execute("""
            CREATE TABLE prices AS SELECT * FROM df
        """)
        print(f"✅ Created table ({len(df)} rows)")
    else:
        conn.execute("INSERT INTO prices SELECT * FROM df")
        print(f"✅ Inserted ({len(df)} rows)")

total_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
print(f"\n✅ Total price records: {total_prices:,}")

# ============================================================================
# 2. Load Options Data
# ============================================================================

print("\n" + "="*80)
print("PART 2: Loading Options Data")
print("="*80)

calls_files = list(data_dir.glob('*_calls.csv'))
puts_files = list(data_dir.glob('*_puts.csv'))

print(f"Found {len(calls_files)} call files, {len(puts_files)} put files")

# Load calls
for i, csv_file in enumerate(calls_files, 1):
    print(f"[{i}/{len(calls_files)}] Loading calls from {csv_file.stem}...", end=" ")

    df = pd.read_csv(csv_file)

    if i == 1:
        conn.execute("CREATE TABLE options_calls AS SELECT * FROM df")
        print(f"✅ Created table ({len(df)} rows)")
    else:
        conn.execute("INSERT INTO options_calls SELECT * FROM df")
        print(f"✅ Inserted ({len(df)} rows)")

# Load puts
for i, csv_file in enumerate(puts_files, 1):
    print(f"[{i}/{len(puts_files)}] Loading puts from {csv_file.stem}...", end=" ")

    df = pd.read_csv(csv_file)

    if i == 1:
        conn.execute("CREATE TABLE options_puts AS SELECT * FROM df")
        print(f"✅ Created table ({len(df)} rows)")
    else:
        conn.execute("INSERT INTO options_puts SELECT * FROM df")
        print(f"✅ Inserted ({len(df)} rows)")

total_calls = conn.execute("SELECT COUNT(*) FROM options_calls").fetchone()[0]
total_puts = conn.execute("SELECT COUNT(*) FROM options_puts").fetchone()[0]

print(f"\n✅ Total options: {total_calls:,} calls + {total_puts:,} puts")

# ============================================================================
# 3. Load Features
# ============================================================================

print("\n" + "="*80)
print("PART 3: Loading Feature Data")
print("="*80)

feature_files = list((data_dir / 'features').glob('*_features.csv'))
print(f"Found {len(feature_files)} feature files")

for i, csv_file in enumerate(feature_files, 1):
    print(f"[{i}/{len(feature_files)}] Loading {csv_file.stem}...", end=" ")

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.reset_index().rename(columns={'index': 'date'})

    if i == 1:
        conn.execute("CREATE TABLE features AS SELECT * FROM df")
        print(f"✅ Created table ({len(df)} rows)")
    else:
        conn.execute("INSERT INTO features SELECT * FROM df")
        print(f"✅ Inserted ({len(df)} rows)")

total_features = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
print(f"\n✅ Total feature records: {total_features:,}")

# ============================================================================
# 4. Load Training Datasets
# ============================================================================

print("\n" + "="*80)
print("PART 4: Loading Training Datasets")
print("="*80)

datasets = ['train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv']
table_names = ['train', 'validation', 'test']

for dataset_file, table_name in zip(datasets, table_names):
    csv_path = data_dir / dataset_file

    if not csv_path.exists():
        print(f"⚠️  {dataset_file} not found, skipping")
        continue

    print(f"Loading {dataset_file}...", end=" ")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.reset_index().rename(columns={'index': 'date'})

    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"✅ {count:,} samples")

# Also create master dataset
master_path = data_dir / 'master_training_dataset.csv'
if master_path.exists():
    print(f"Loading master_training_dataset.csv...", end=" ")
    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    df = df.reset_index().rename(columns={'index': 'date'})
    conn.execute("CREATE TABLE master AS SELECT * FROM df")
    count = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
    print(f"✅ {count:,} samples")

# ============================================================================
# 5. Create Indexes for Fast Queries
# ============================================================================

print("\n" + "="*80)
print("PART 5: Creating Indexes")
print("="*80)

print("Creating indexes for fast queries...")

# Price indexes
conn.execute("CREATE INDEX idx_prices_symbol ON prices(symbol)")
conn.execute("CREATE INDEX idx_prices_date ON prices(date)")
print("✅ Price indexes")

# Options indexes
conn.execute("CREATE INDEX idx_calls_symbol ON options_calls(symbol)")
conn.execute("CREATE INDEX idx_calls_expiration ON options_calls(expiration)")
conn.execute("CREATE INDEX idx_puts_symbol ON options_puts(symbol)")
conn.execute("CREATE INDEX idx_puts_expiration ON options_puts(expiration)")
print("✅ Options indexes")

# Feature indexes
conn.execute("CREATE INDEX idx_features_symbol ON features(symbol)")
conn.execute("CREATE INDEX idx_features_date ON features(date)")
print("✅ Feature indexes")

# Training data indexes
conn.execute("CREATE INDEX idx_train_symbol ON train(symbol)")
conn.execute("CREATE INDEX idx_val_symbol ON validation(symbol)")
conn.execute("CREATE INDEX idx_test_symbol ON test(symbol)")
print("✅ Training data indexes")

# ============================================================================
# 6. Database Statistics
# ============================================================================

print("\n" + "="*80)
print("DATABASE STATISTICS")
print("="*80)

tables = conn.execute("SHOW TABLES").fetchall()
print(f"\nTables created: {len(tables)}")

for table in tables:
    table_name = table[0]
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    size_query = f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))"

    print(f"  - {table_name}: {count:,} rows")

# Get database file size
import os
db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
print(f"\n📊 Database size: {db_size:.1f} MB (compressed)")

# Compare to CSV size
csv_size = sum(f.stat().st_size for f in data_dir.glob('**/*.csv')) / (1024 * 1024)
print(f"📊 CSV size: {csv_size:.1f} MB")
print(f"📊 Compression: {csv_size/db_size:.1f}x smaller")

# ============================================================================
# 7. Vacuum and Optimize
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZING DATABASE")
print("="*80)

print("Running VACUUM to optimize storage...", end=" ")
conn.execute("VACUUM")
print("✅")

print("Running ANALYZE to optimize queries...", end=" ")
conn.execute("ANALYZE")
print("✅")

# Close connection
conn.close()

# Final size check
final_size = os.path.getsize(db_path) / (1024 * 1024)
print(f"\n✅ Final database size: {final_size:.1f} MB")

# ============================================================================
# 8. Create Compressed Backup
# ============================================================================

print("\n" + "="*80)
print("CREATING COMPRESSED BACKUP")
print("="*80)

import gzip
import shutil

backup_path = db_path.with_suffix('.duckdb.gz')
print(f"Compressing to: {backup_path}...", end=" ")

with open(db_path, 'rb') as f_in:
    with gzip.open(backup_path, 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)

backup_size = os.path.getsize(backup_path) / (1024 * 1024)
print(f"✅ {backup_size:.1f} MB")

print(f"Compression ratio: {final_size/backup_size:.1f}x")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("CONVERSION COMPLETE!")
print("="*80)

print(f"\n✅ DuckDB database: {db_path}")
print(f"   Size: {final_size:.1f} MB")
print(f"   Tables: {len(tables)}")
print(f"   Records: {total_prices + total_features + total_calls + total_puts:,}")

print(f"\n✅ Compressed backup: {backup_path}")
print(f"   Size: {backup_size:.1f} MB")
print(f"   Compression: {csv_size/backup_size:.1f}x from CSV")

print(f"\n📊 Storage comparison:")
print(f"   CSV files: {csv_size:.1f} MB")
print(f"   DuckDB: {final_size:.1f} MB ({csv_size/final_size:.1f}x smaller)")
print(f"   DuckDB.gz: {backup_size:.1f} MB ({csv_size/backup_size:.1f}x smaller)")

print(f"\n🚀 BENEFITS:")
print(f"   - 10-100x faster queries than CSV")
print(f"   - SQL interface for feature engineering")
print(f"   - Zero-copy to PyTorch/NumPy tensors")
print(f"   - {csv_size/final_size:.1f}x space savings")

print(f"\n📖 USAGE:")
print(f"   import duckdb")
print(f"   conn = duckdb.connect('{db_path}')")
print(f"   df = conn.execute('SELECT * FROM train').df()")

print(f"\n🔄 NEXT STEPS:")
print(f"   1. Train model: uv run python scripts/ml/train_price_predictor.py")
print(f"   2. Backtest: uv run python scripts/ml/backtest_model.py")
print(f"   3. Paper trade: 1 day")
print(f"   4. GO LIVE! 💰")

print("="*80)
