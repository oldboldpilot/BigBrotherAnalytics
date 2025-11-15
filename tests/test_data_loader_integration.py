#!/usr/bin/env python3
"""
Integration test for C++ data loader and feature extractor

This test verifies that:
1. C++ data loader correctly loads historical prices from DuckDB
2. Data is returned in MOST RECENT FIRST order (critical!)
3. Data integrates correctly with C++ feature extractor
4. Complete pipeline: Data loading -> Feature extraction works
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import data_loader_cpp
    print("✅ C++ data loader module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import C++ data loader: {e}")
    print("   Make sure to build the Python bindings first:")
    print("   ninja -C build data_loader_py")
    sys.exit(1)

try:
    import feature_extractor_cpp
    print("✅ C++ feature extractor module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import C++ feature extractor: {e}")
    print("   Make sure to build the Python bindings first:")
    print("   ninja -C build feature_extractor_py")
    sys.exit(1)

print()
print("="*80)
print("DATA LOADER + FEATURE EXTRACTOR INTEGRATION TEST")
print("="*80)
print()

# ============================================================================
# Test 1: Load Historical Prices
# ============================================================================
print("[1/3] Testing C++ data loader...")
print()

try:
    # Load 100 days of SPY historical data
    dates, closes, volumes = data_loader_cpp.load_historical_prices(
        'data/bigbrother.duckdb',
        'SPY',
        limit=100
    )

    print(f"✅ Loaded {len(closes)} historical prices for SPY")
    print(f"   Date range: {dates[-1]} to {dates[0]}")
    print(f"   Most recent price: {closes[0]:.2f} (date: {dates[0]})")
    print(f"   Oldest price: {closes[-1]:.2f} (date: {dates[-1]})")
    print()

    # Verify data ordering (most recent first)
    if len(dates) >= 2:
        # Dates should be in descending order
        from datetime import datetime
        date_0 = datetime.fromisoformat(dates[0].replace('Z', '+00:00'))
        date_1 = datetime.fromisoformat(dates[1].replace('Z', '+00:00'))

        if date_0 >= date_1:
            print(f"✅ Data ordering verified: MOST RECENT FIRST")
            print(f"   dates[0] = {dates[0]} >= dates[1] = {dates[1]}")
        else:
            print(f"❌ ERROR: Data ordering is wrong!")
            print(f"   dates[0] = {dates[0]} < dates[1] = {dates[1]}")
            sys.exit(1)

    print()

except Exception as e:
    print(f"❌ Data loader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 2: Load Training Data
# ============================================================================
print("[2/3] Testing training data loader...")
print()

try:
    # Load training data (just a few rows to verify it works)
    training_data = data_loader_cpp.load_training_data(
        'data/training_data.duckdb',
        table_name='train'
    )

    # Get column names
    columns = list(training_data.keys())

    # Get row count
    if columns:
        row_count = len(training_data[columns[0]])
    else:
        row_count = 0

    print(f"✅ Loaded training data from 'train' table")
    print(f"   Rows: {row_count:,}")
    print(f"   Columns: {len(columns)}")
    print(f"   Sample columns: {', '.join(columns[:10])}")
    print()

except Exception as e:
    print(f"❌ Training data loader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 3: Integration with Feature Extractor
# ============================================================================
print("[3/3] Testing integration with feature extractor...")
print()

try:
    # Use the loaded SPY data to extract features
    # We'll use the most recent day's data

    # Mock some indicator values (in real training, these come from DuckDB)
    test_input = {
        'close': float(closes[0]),
        'open': float(closes[1]),  # Use previous close as open approximation
        'high': float(closes[0]) * 1.01,  # Mock high
        'low': float(closes[0]) * 0.99,   # Mock low
        'volume': float(volumes[0]),
        'rsi_14': 50.0,
        'macd': 1.0,
        'macd_signal': 0.9,
        'bb_upper': float(closes[0]) * 1.05,
        'bb_lower': float(closes[0]) * 0.95,
        'bb_position': 0.5,
        'atr_14': 5.0,
        'volume_ratio': 1.0,
        'volume_rsi_signal': 0.0,
        'yield_volatility': 0.05,
        'macd_volume': 1.0,
        'bb_momentum': 0.01,
        'rate_return': 0.001,
        'rsi_bb_signal': 0.5,
        'momentum_3d': 0.02,
        'recent_win_rate': 0.55,
        'symbol_encoded': 0.0,  # SPY
        'day_of_week': 3.0,
        'day_of_month': 14.0,
        'month_of_year': 11.0,
        'quarter': 4.0,
        'day_of_year': 318.0,
        'price_direction': 1.0,
        'price_above_ma5': 1.0,
        'price_above_ma20': 1.0,
        'macd_signal_direction': 1.0,
        'volume_trend': 1.0,
        'year': 2025,
        'month': 11,
        'day': 14,
        'fed_funds_rate': 4.5,
        'treasury_10yr': 4.11,
    }

    # Extract features using the C++ feature extractor with data from C++ data loader
    features = feature_extractor_cpp.extract_features_85(
        test_input['close'],
        test_input['open'],
        test_input['high'],
        test_input['low'],
        test_input['volume'],
        test_input['rsi_14'],
        test_input['macd'],
        test_input['macd_signal'],
        test_input['bb_upper'],
        test_input['bb_lower'],
        test_input['bb_position'],
        test_input['atr_14'],
        test_input['volume_ratio'],
        test_input['volume_rsi_signal'],
        test_input['yield_volatility'],
        test_input['macd_volume'],
        test_input['bb_momentum'],
        test_input['rate_return'],
        test_input['rsi_bb_signal'],
        test_input['momentum_3d'],
        test_input['recent_win_rate'],
        test_input['symbol_encoded'],
        test_input['day_of_week'],
        test_input['day_of_month'],
        test_input['month_of_year'],
        test_input['quarter'],
        test_input['day_of_year'],
        test_input['price_direction'],
        test_input['price_above_ma5'],
        test_input['price_above_ma20'],
        test_input['macd_signal_direction'],
        test_input['volume_trend'],
        test_input['year'],
        test_input['month'],
        test_input['day'],
        closes.astype(np.float32),   # price_history from C++ data loader
        volumes.astype(np.float32),  # volume_history from C++ data loader
        test_input['fed_funds_rate'],
        test_input['treasury_10yr'],
    )

    print(f"✅ Feature extraction successful with C++ data loader integration")
    print(f"   Features extracted: {len(features)}")
    print(f"   Using price history from C++ data loader: {len(closes)} prices")
    print()

    # Display first 10 features
    print("First 10 features:")
    feature_names = [
        "close", "open", "high", "low", "volume",
        "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"
    ]
    for i in range(10):
        print(f"  [{i:2d}] {feature_names[i]:15s} = {features[i]:12.6f}")

    print()

    # Verify price lags use actual historical prices
    print("Price lags [27-31] (should match actual historical prices):")
    for i in range(27, 32):
        lag_idx = i - 27
        print(f"  [{i:2d}] price_lag_{lag_idx+1}d = {features[i]:12.2f} (actual: {closes[lag_idx]:.2f})")

    # Verify they match
    price_lags_match = all(abs(features[27+i] - closes[i]) < 0.01 for i in range(5))
    if price_lags_match:
        print()
        print(f"✅ Price lags correctly use actual historical prices from data loader")
    else:
        print()
        print(f"❌ Price lags do NOT match historical prices from data loader!")
        sys.exit(1)

    print()

except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print()
print("="*80)
print("✅ INTEGRATION TEST PASSED")
print("="*80)
print()
print("Summary:")
print("  1. ✅ C++ data loader correctly loads historical prices")
print("  2. ✅ Data ordering verified: MOST RECENT FIRST")
print("  3. ✅ C++ feature extractor integrates with C++ data loader")
print("  4. ✅ Price lags use actual historical prices from data loader")
print()
print("Single source of truth established:")
print("  C++ data_loader → C++ feature_extractor → Training data")
print()
print("Ready for complete training pipeline!")
print()

sys.exit(0)
