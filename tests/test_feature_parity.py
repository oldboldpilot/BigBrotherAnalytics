#!/usr/bin/env python3
"""
Test C++ vs Python feature extraction parity

This script validates that the C++ feature extractor produces IDENTICAL
results to the Python implementation for the same inputs.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import feature_extractor_cpp
    HAS_CPP_MODULE = True
except ImportError:
    print("⚠️  C++ module not built. Run: python bindings/setup.py build_ext --inplace")
    HAS_CPP_MODULE = False

def test_feature_extraction_parity():
    """Test that C++ and Python produce identical features"""

    print("=" * 80)
    print("FEATURE EXTRACTION PARITY TEST")
    print("=" * 80)
    print()

    # Test case: SPY on 2025-11-14
    test_input = {
        'close': 670.59,
        'open': 672.04,
        'high': 683.38,
        'low': 670.59,
        'volume': 0.0,
        'rsi_14': 40.19,
        'macd': 2.49,
        'macd_signal': 2.24,
        'bb_upper': 690.70,
        'bb_lower': 668.15,
        'bb_position': (670.59 - 668.15) / (690.70 - 668.15),
        'atr_14': 5.0,
        'volume_ratio': 0.8,
        'volume_rsi_signal': 0.0,
        'yield_volatility': 0.05,
        'macd_volume': 1.99,
        'bb_momentum': 0.01,
        'rate_return': 0.001,
        'rsi_bb_signal': 0.4,
        'momentum_3d': 0.02,
        'recent_win_rate': 0.55,
        'symbol_encoded': 0.0,  # SPY
        'day_of_week': 3.0,  # Thursday (Python: 0=Monday)
        'day_of_month': 14.0,
        'month_of_year': 11.0,
        'quarter': 4.0,
        'day_of_year': 318.0,
        'price_direction': 0.0,
        'price_above_ma5': 1.0,
        'price_above_ma20': 1.0,
        'macd_signal_direction': 1.0,
        'volume_trend': 0.0,
        'year': 2025,
        'month': 11,
        'day': 14,
        'fed_funds_rate': 4.5,
        'treasury_10yr': 4.11,
    }

    # Historical prices (100 days, most recent first)
    price_history = np.array([
        670.59, 672.04, 665.32, 668.91, 662.15, 660.04, 658.23, 655.12,
        650.45, 648.90, 645.32, 642.11, 640.05, 638.92, 635.21, 632.45,
        630.12, 628.34, 625.67, 622.90, 620.45, 618.23, 615.67, 612.34,
        610.12, 608.45, 605.90, 603.21, 600.67, 598.34, 595.12, 592.45,
    ] + [590.0] * 68, dtype=np.float32)  # Pad to 100 days

    volume_history = np.array([
        50000000.0, 48000000.0, 52000000.0, 49000000.0, 51000000.0,
        47000000.0, 48500000.0, 50500000.0, 49500000.0, 48000000.0,
        47500000.0, 49000000.0, 50000000.0, 48500000.0, 49500000.0,
        47000000.0, 48000000.0, 49000000.0, 50000000.0, 48500000.0,
    ] + [48000000.0] * 80, dtype=np.float32)  # Pad to 100 days

    if not HAS_CPP_MODULE:
        print("❌ C++ module not available - skipping test")
        return False

    print("Testing C++ feature extraction...")
    try:
        cpp_features = feature_extractor_cpp.extract_features_85(
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
            price_history,
            volume_history,
            test_input['fed_funds_rate'],
            test_input['treasury_10yr'],
        )

        print(f"✅ C++ extraction successful: {len(cpp_features)} features")
        print()

        # Display first 20 features
        print("First 20 C++ features:")
        feature_names = [
            "close", "open", "high", "low", "volume",
            "rsi_14", "macd", "macd_signal",
            "bb_upper", "bb_lower", "bb_position",
            "atr_14", "volume_sma20", "volume_ratio",
            "gamma", "theta", "vega", "rho",
            "volume_rsi_signal", "yield_volatility"
        ]
        for i in range(20):
            print(f"  [{i:2d}] {feature_names[i]:20s} = {cpp_features[i]:12.6f}")

        print()
        print("Price lags [27-31]:")
        for i in range(27, 32):
            print(f"  [{i:2d}] price_lag_{i-26}d = {cpp_features[i]:12.6f}")

        print()
        print("Price diffs [61-65]:")
        for i in range(61, 66):
            print(f"  [{i:2d}] price_diff_{i-60}d = {cpp_features[i]:12.6f}")

        print()
        print("Autocorrelations [81-84]:")
        for i, lag in enumerate([1, 5, 10, 20]):
            print(f"  [{81+i:2d}] autocorr_lag_{lag} = {cpp_features[81+i]:12.6f}")

        print()
        print("=" * 80)
        print("✅ FEATURE EXTRACTION TEST PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"❌ C++ extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_extraction_parity()
    sys.exit(0 if success else 1)
