#!/usr/bin/env python3
"""
Export Test Features for C++ Validation

Exports a single row of features from the clean training dataset
to validate C++ feature extraction matches Python exactly.

Output: CSV with feature_index,feature_name,value
"""

import duckdb
import pandas as pd
import sys
from pathlib import Path

# Feature order (official from clean_training_metadata.json)
FEATURE_ORDER = [
    # [0-4] OHLCV
    "close", "open", "high", "low", "volume",
    # [5-7] Technical
    "rsi_14", "macd", "macd_signal",
    # [8-10] Bollinger
    "bb_upper", "bb_lower", "bb_position",
    # [11-13] Volume/Volatility
    "atr_14", "volume_sma20", "volume_ratio",
    # [14-17] Greeks
    "gamma", "theta", "vega", "rho",
    # [18-24] Interactions
    "volume_rsi_signal", "yield_volatility", "macd_volume",
    "bb_momentum", "rate_return", "gamma_volatility", "rsi_bb_signal",
    # [25-26] Momentum
    "momentum_3d", "recent_win_rate",
    # [27-46] Price lags
    "price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_4d", "price_lag_5d",
    "price_lag_6d", "price_lag_7d", "price_lag_8d", "price_lag_9d", "price_lag_10d",
    "price_lag_11d", "price_lag_12d", "price_lag_13d", "price_lag_14d", "price_lag_15d",
    "price_lag_16d", "price_lag_17d", "price_lag_18d", "price_lag_19d", "price_lag_20d",
    # [47] Symbol
    "symbol_encoded",
    # [48-52] Time
    "day_of_week", "day_of_month", "month_of_year", "quarter", "day_of_year",
    # [53-57] Directional
    "price_direction", "price_above_ma5", "price_above_ma20",
    "macd_signal_direction", "volume_trend",
    # [58-60] Date
    "year", "month", "day",
    # [61-80] Price diffs
    "price_diff_1d", "price_diff_2d", "price_diff_3d", "price_diff_4d", "price_diff_5d",
    "price_diff_6d", "price_diff_7d", "price_diff_8d", "price_diff_9d", "price_diff_10d",
    "price_diff_11d", "price_diff_12d", "price_diff_13d", "price_diff_14d", "price_diff_15d",
    "price_diff_16d", "price_diff_17d", "price_diff_18d", "price_diff_19d", "price_diff_20d",
    # [81-84] Autocorr
    "autocorr_lag_1", "autocorr_lag_5", "autocorr_lag_10", "autocorr_lag_20"
]

def main():
    # Connect to clean training database
    db_path = 'data/clean_training_data.duckdb'

    if not Path(db_path).exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        print("Run: python scripts/data_collection/create_clean_training_data.py", file=sys.stderr)
        sys.exit(1)

    conn = duckdb.connect(db_path, read_only=True)

    # Get a single row from test set (most recent data)
    query = """
        SELECT * FROM test
        ORDER BY Date DESC
        LIMIT 1
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("ERROR: No data found in test set", file=sys.stderr)
        sys.exit(1)

    # Print header
    print("feature_index,feature_name,value")

    # Export features in exact order
    for idx, feature_name in enumerate(FEATURE_ORDER):
        if feature_name not in df.columns:
            print(f"WARNING: Feature '{feature_name}' not found in database", file=sys.stderr)
            value = 0.0
        else:
            value = df[feature_name].iloc[0]

        print(f"{idx},{feature_name},{value:.10f}")

    # Also print metadata for debugging
    print(f"\n# Metadata", file=sys.stderr)
    print(f"# Date: {df['Date'].iloc[0]}", file=sys.stderr)
    print(f"# Symbol: {df['symbol'].iloc[0]}", file=sys.stderr)
    print(f"# Close: {df['close'].iloc[0]:.2f}", file=sys.stderr)
    print(f"# Features exported: {len(FEATURE_ORDER)}", file=sys.stderr)

if __name__ == '__main__':
    main()
