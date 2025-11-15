# Feature Parity Validation Guide

Quick reference for validating C++ feature extraction matches Python training.

---

## Quick Start

```bash
# 1. Export Python test features
python scripts/ml/export_test_features.py > test_features.csv

# 2. Build validation test
make test_feature_parity

# 3. Run validation
./build/tests/verify_feature_parity test_features.csv
```

---

## Detailed Steps

### Step 1: Verify Training Database Exists

```bash
ls -lh data/clean_training_data.duckdb
```

If missing, create it:
```bash
python scripts/data_collection/create_clean_training_data.py
```

### Step 2: Export Test Features

```bash
python scripts/ml/export_test_features.py > test_features.csv
```

This exports one row from the test set with all 85 features in correct order.

**Expected output:**
```csv
feature_index,feature_name,value
0,close,193.4700000000
1,open,192.3500000000
...
84,autocorr_lag_20,0.1200000000

# Metadata
# Date: 2025-11-13 16:00:00
# Symbol: SPY
# Close: 193.47
# Features exported: 85
```

### Step 3: Build Validation Test

Add to `CMakeLists.txt`:
```cmake
# Feature parity validation test
add_executable(verify_feature_parity
    tests/verify_feature_parity.cpp
)
target_link_libraries(verify_feature_parity
    bigbrother_market_intelligence
)
```

Build:
```bash
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release ..
make verify_feature_parity
```

### Step 4: Run Validation

```bash
./build/tests/verify_feature_parity test_features.csv
```

**Expected output:**
```
Feature Parity Validation Test
Model Version: Production (85 features, INT32 SIMD)

Loaded 85 Python features from test_features.csv

==================================================================================================
FEATURE PARITY VALIDATION RESULTS
==================================================================================================

Index | Feature Name                | Python Value  | C++ Value     | Diff       | Status
--------------------------------------------------------------------------------------------------
[0]   | close                       |    193.470000 |    193.470000 |   0.000000 | ✓
[1]   | open                        |    192.350000 |    192.350000 |   0.000000 | ✓
[2]   | high                        |    195.120000 |    195.120000 |   0.000000 | ✓
...
[84]  | autocorr_lag_20             |      0.120000 |      0.119998 |   0.000002 | ✓
--------------------------------------------------------------------------------------------------
SUMMARY:
  Total features: 85
  Passed: 85 (100.0%)
  Failed: 0 (0.0%)
==================================================================================================

✅ ALL FEATURES MATCH! Feature parity verified.
```

---

## Tolerance Settings

**Default tolerance:** 1e-3 (0.001)

This allows for minor floating-point precision differences between Python and C++.

To adjust tolerance:
```cpp
auto results = compareFeatures(python_features, cpp_features, 1e-4f);  // Stricter
```

---

## Troubleshooting

### Issue: "Database not found"

**Solution:**
```bash
python scripts/data_collection/create_clean_training_data.py
```

### Issue: "No features loaded from CSV"

**Solution:**
Check CSV format:
```bash
head -5 test_features.csv
```

Should show:
```
feature_index,feature_name,value
0,close,193.47
...
```

### Issue: "Feature mismatch detected"

**Root causes:**
1. Feature calculation differs between Python and C++
2. Feature order differs
3. Input data differs

**Debug steps:**
1. Check which features failed:
   ```bash
   ./verify_feature_parity test_features.csv | grep '✗'
   ```

2. Review feature calculation in both codebases:
   - Python: `scripts/ml/prepare_custom_features.py` and `scripts/data_collection/create_clean_training_data.py`
   - C++: `src/market_intelligence/feature_extractor.cppm` (toArray85 function)

3. Compare formulas:
   - See: `docs/feature_parity_analysis.md`

### Issue: "Compilation errors"

**Common fixes:**
```bash
# Update compiler
sudo apt-get install g++-14

# Check C++23 support
g++-14 --version

# Enable modules
export CXX=g++-14
export CXXFLAGS="-std=c++23 -fmodules-ts"
```

---

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/test.yml`:

```yaml
name: Feature Parity Test

on: [push, pull_request]

jobs:
  validate-features:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install duckdb pandas numpy

      - name: Export test features
        run: |
          python scripts/ml/export_test_features.py > test_features.csv

      - name: Build C++ test
        run: |
          mkdir build && cd build
          cmake -DCMAKE_CXX_COMPILER=g++-14 ..
          make verify_feature_parity

      - name: Run validation
        run: |
          ./build/tests/verify_feature_parity test_features.csv
```

---

## Manual Verification (Alternative)

If automated test fails, manually verify critical features:

### 1. Price Diffs

**Python:**
```python
import duckdb
conn = duckdb.connect('data/clean_training_data.duckdb')
df = conn.execute('SELECT price_diff_1d, price_diff_5d, price_diff_20d FROM test LIMIT 1').df()
print(df)
```

**C++:**
Inspect `price_diffs` array in debugger.

### 2. Autocorrelations

**Python:**
```python
df = conn.execute('SELECT autocorr_lag_1, autocorr_lag_5 FROM test LIMIT 1').df()
print(df)
```

**C++:**
Inspect `autocorr_1`, `autocorr_5` values.

### 3. Day of Week

**Python:**
```python
import pandas as pd
date = pd.to_datetime('2025-11-14')  # Thursday
print(date.dayofweek)  # Should be 3 (0=Monday)
```

**C++:**
```cpp
// tm_wday for Thursday = 4
// Converted: (4 - 1) = 3 ✓
```

---

## Feature List Reference

### All 85 Features (in order)

```
[0-4]   close, open, high, low, volume
[5-7]   rsi_14, macd, macd_signal
[8-10]  bb_upper, bb_lower, bb_position
[11-13] atr_14, volume_sma20, volume_ratio
[14-17] gamma, theta, vega, rho
[18-24] volume_rsi_signal, yield_volatility, macd_volume, bb_momentum,
        rate_return, gamma_volatility, rsi_bb_signal
[25-26] momentum_3d, recent_win_rate
[27-46] price_lag_1d through price_lag_20d (20 features)
[47]    symbol_encoded
[48-52] day_of_week, day_of_month, month_of_year, quarter, day_of_year
[53-57] price_direction, price_above_ma5, price_above_ma20,
        macd_signal_direction, volume_trend
[58-60] year, month, day
[61-80] price_diff_1d through price_diff_20d (20 features)
[81-84] autocorr_lag_1, autocorr_lag_5, autocorr_lag_10, autocorr_lag_20
```

Total: **85 features**

---

## Contact

For issues or questions:
- Check: `docs/feature_parity_analysis.md`
- Review: `docs/feature_parity_fixes_summary.md`
- File issue: Include test output and feature index of failures

---

**Last Updated:** 2025-11-14
