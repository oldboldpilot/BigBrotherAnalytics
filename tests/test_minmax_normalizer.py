#!/usr/bin/env python3
"""
Test MinMaxNormalizer - Verify Python implementation matches C++ behavior

Tests:
1. Basic fit/transform
2. Inverse transformation (round-trip)
3. Batch operations
4. Constant feature handling
5. JSON save/load
6. C++ header export
"""

import sys
from pathlib import Path
import numpy as np
import json
import tempfile

# Add scripts/ml to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'ml'))
from minmax_normalizer import MinMaxNormalizer

def test_basic_fit_transform():
    """Test basic fit and transform"""
    print("[Test 1] Basic fit/transform")

    # Create simple dataset
    X_train = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [20.0, 30.0, 40.0]
    ], dtype=np.float32)

    normalizer = MinMaxNormalizer()
    X_normalized = normalizer.fit_transform(X_train)

    # Check expected values
    expected_first = np.array([0.0, 0.0, 0.0])  # Min values → 0
    expected_last = np.array([1.0, 1.0, 1.0])   # Max values → 1

    assert np.allclose(X_normalized[0], expected_first, atol=1e-5), "First row should be all zeros"
    assert np.allclose(X_normalized[2], expected_last, atol=1e-5), "Last row should be all ones"

    print("✅ PASS\n")

def test_inverse_transform():
    """Test inverse transformation (round-trip)"""
    print("[Test 2] Inverse transformation")

    X_train = np.array([
        [100.0, 200.0, 300.0],
        [150.0, 250.0, 350.0],
        [200.0, 300.0, 400.0]
    ], dtype=np.float32)

    normalizer = MinMaxNormalizer()
    X_normalized = normalizer.fit_transform(X_train)
    X_recovered = normalizer.inverse_transform(X_normalized)

    # Check round-trip accuracy
    assert np.allclose(X_train, X_recovered, atol=1e-4), "Round-trip failed"

    print("✅ PASS (round-trip accuracy)\n")

def test_constant_feature():
    """Test handling of constant features"""
    print("[Test 3] Constant feature handling")

    # Feature 1 is constant
    X_train = np.array([
        [10.0, 42.0, 30.0],
        [20.0, 42.0, 40.0],
        [30.0, 42.0, 50.0]
    ], dtype=np.float32)

    normalizer = MinMaxNormalizer()
    X_normalized = normalizer.fit_transform(X_train)

    # Check no NaN or Inf
    assert not np.any(np.isnan(X_normalized)), "NaN detected"
    assert not np.any(np.isinf(X_normalized)), "Inf detected"

    # Constant feature should be 0 (since min=max, and we use range=1.0 fallback)
    assert np.allclose(X_normalized[:, 1], 0.0, atol=1e-5), "Constant feature should be 0"

    print("✅ PASS (handles constant features)\n")

def test_save_load_json():
    """Test JSON save/load"""
    print("[Test 4] JSON save/load")

    X_train = np.random.randn(10, 5).astype(np.float32)

    normalizer = MinMaxNormalizer()
    normalizer.fit(X_train)

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)

    normalizer.save(temp_path)

    # Load and verify
    loaded = MinMaxNormalizer.load(temp_path)

    assert np.allclose(loaded.min_, normalizer.min_), "Min not loaded correctly"
    assert np.allclose(loaded.max_, normalizer.max_), "Max not loaded correctly"
    assert np.allclose(loaded.range_, normalizer.range_), "Range not loaded correctly"

    # Clean up
    temp_path.unlink()

    print("✅ PASS (JSON serialization)\n")

def test_cpp_header_export():
    """Test C++ header export"""
    print("[Test 5] C++ header export")

    X_train = np.array([
        [0.0, 100.0],
        [50.0, 200.0],
        [100.0, 300.0]
    ], dtype=np.float32)

    normalizer = MinMaxNormalizer()
    normalizer.fit(X_train)

    # Export to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.hpp', delete=False) as f:
        temp_path = Path(f.name)

    normalizer.export_cpp_header(temp_path, template_size=2)

    # Read and verify content
    content = temp_path.read_text()

    assert "std::array<float, 2>" in content, "Wrong array size"
    assert "FEATURE_MIN" in content, "Missing FEATURE_MIN"
    assert "FEATURE_MAX" in content, "Missing FEATURE_MAX"
    assert "0.00000000f" in content, "Wrong min value"
    assert "100.00000000f" in content, "Wrong max value for feature 0"

    # Clean up
    temp_path.unlink()

    print("✅ PASS (C++ header export)\n")

def test_normalization_range():
    """Test that normalization produces values in [0, 1]"""
    print("[Test 6] Normalization range check")

    X_train = np.random.randn(100, 10).astype(np.float32) * 100  # Random data

    normalizer = MinMaxNormalizer()
    X_normalized = normalizer.fit_transform(X_train)

    # Check range
    assert np.all(X_normalized >= 0.0), "Values below 0"
    assert np.all(X_normalized <= 1.0), "Values above 1"

    print("✅ PASS (all values in [0, 1])\n")

def main():
    print("\n" + "=" * 70)
    print(" MinMaxNormalizer Test Suite")
    print("=" * 70 + "\n")

    tests = [
        test_basic_fit_transform,
        test_inverse_transform,
        test_constant_feature,
        test_save_load_json,
        test_cpp_header_export,
        test_normalization_range,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {e}\n")

    print("=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 70 + "\n")

    return 0 if passed == len(tests) else 1

if __name__ == '__main__':
    sys.exit(main())
