# Feature Parity Analysis: Python Training vs C++ Inference

**Date:** 2025-11-14
**Module:** bigbrother.market_intelligence.price_predictor
**Model Version:** Production (85 features, INT32 SIMD)
**Status:** ⚠️ DISCREPANCIES FOUND (NOW FIXED - See FEATURE_PARITY_REPORT.md)

## Executive Summary

This document provides a comprehensive comparison of all 85 features between:
- **Python Training:** `scripts/ml/prepare_custom_features.py` + `scripts/data_collection/create_clean_training_data.py`
- **C++ Inference:** `src/market_intelligence/feature_extractor.cppm` (toArray85 function)

### Critical Findings

🔴 **MAJOR DISCREPANCIES FOUND:**
1. **Feature Order Mismatch:** C++ toArray85() has different ordering than Python metadata
2. **Price Diffs Calculation:** C++ uses `price[i] - price[i+1]`, Python uses `diff(lag)`
3. **Autocorrelation Calculation:** C++ uses price-based ACF, Python uses returns-based ACF
4. **Volume SMA20 Calculation:** C++ correctly calculates from volume_history
5. **Greeks Calculation:** Both use ATR-based volatility, but need to verify exact formulas

---

## Feature-by-Feature Comparison (All 85 Features)

### Official Feature Order (from clean_training_metadata.json)

```
[0-4]   OHLCV: close, open, high, low, volume
[5-7]   Technical: rsi_14, macd, macd_signal
[8-10]  Bollinger: bb_upper, bb_lower, bb_position
[11-13] Volume: atr_14, volume_sma20, volume_ratio
[14-17] Greeks: gamma, theta, vega, rho
[18-24] Interactions: volume_rsi_signal, yield_volatility, macd_volume, bb_momentum, rate_return, gamma_volatility, rsi_bb_signal
[25-26] Momentum: momentum_3d, recent_win_rate
[27-46] Price Lags: price_lag_1d through price_lag_20d
[47]    Symbol: symbol_encoded
[48-52] Time: day_of_week, day_of_month, month_of_year, quarter, day_of_year
[53-57] Directional: price_direction, price_above_ma5, price_above_ma20, macd_signal_direction, volume_trend
[58-60] Date: year, month, day
[61-80] Price Diffs: price_diff_1d through price_diff_20d
[81-84] Autocorr: autocorr_lag_1, autocorr_lag_5, autocorr_lag_10, autocorr_lag_20
```

---

## Detailed Feature Comparison

### ✅ [0-4] OHLCV - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 0 | close | `df['close']` | `close` | ✅ Match |
| 1 | open | `df['open']` | `open` | ✅ Match |
| 2 | high | `df['high']` | `high` | ✅ Match |
| 3 | low | `df['low']` | `low` | ✅ Match |
| 4 | volume | `df['volume']` | `volume` | ✅ Match |

### ✅ [5-7] Technical Indicators - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 5 | rsi_14 | Calculated from training_data.duckdb | `calculateRSI(price_history)` | ✅ Match |
| 6 | macd | Calculated from training_data.duckdb | `calculateMACD(price_history)[0]` | ✅ Match |
| 7 | macd_signal | Calculated from training_data.duckdb | `calculateMACD(price_history)[1]` | ✅ Match |

### ✅ [8-10] Bollinger Bands - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 8 | bb_upper | Calculated from training_data.duckdb | `bb_upper` | ✅ Match |
| 9 | bb_lower | Calculated from training_data.duckdb | `bb_lower` | ✅ Match |
| 10 | bb_position | Calculated from training_data.duckdb | `bb_position` | ✅ Match |

### ✅ [11-13] Volume/Volatility - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 11 | atr_14 | `true_range.rolling(14).mean()` | `calculateATR(price_history)` | ✅ Match |
| 12 | volume_sma20 | `df['Volume'].rolling(20).mean()` | `sum(volume_history[0:20])/20` | ✅ Match |
| 13 | volume_ratio | `volume / volume_sma20` | `volume / volume_sma20` | ✅ Match |

### ⚠️ [14-17] Greeks - NEED VERIFICATION
| Index | Feature | Python Calculation | C++ Calculation | Status |
|-------|---------|-------------------|-----------------|--------|
| 14 | gamma | JAX Black-Scholes: `norm_pdf(d1)/(S*σ*√T)` | `n_prime_d1 / (S * sigma * sqrt_T)` | ⚠️ Verify match |
| 15 | theta | JAX BS: `(-S*N'(d1)*σ/(2√T) - r*K*e^(-rT)*N(d2))/365` | `-(S * n_prime_d1 * sigma) / (2 * sqrt_T) / 365` | ⚠️ Simplified |
| 16 | vega | JAX BS: `S*N'(d1)*√T/100` | `S * n_prime_d1 * sqrt_T / 100` | ✅ Match |
| 17 | rho | JAX BS: `K*T*e^(-rT)*N(d2)/100` | `K * T * exp(-r*T) * 0.5 / 100` | ⚠️ Approximation |

**Notes:**
- Python: Uses JAX-accelerated Black-Scholes with full formula
- C++: Uses simplified Black-Scholes (theta missing second term, rho is approximation)
- Both: Use ATR-based volatility `(ATR/Price) * sqrt(252)`, clipped to [0.05, 2.0]
- Both: Use ATM options (K=S) with 30-day expiry

**⚠️ CRITICAL:** C++ theta is missing the `-r*K*exp(-r*T)*norm_cdf(d2)` term!

### ✅ [18-24] Interaction Features - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 18 | volume_rsi_signal | `volume_ratio * (rsi_14 - 50) / 50` | `volume_ratio * (rsi_14 - 50) / 50` | ✅ Match |
| 19 | yield_volatility | `yield_curve_slope * atr_14` | `yield_curve_slope * atr_14` | ✅ Match |
| 20 | macd_volume | `macd * volume_ratio` | `macd * volume_ratio` | ✅ Match |
| 21 | bb_momentum | `bb_position * return_1d` | `bb_position * return_1d` | ✅ Match |
| 22 | rate_return | `fed_funds_rate * return_20d` | `fed_funds_rate * return_20d` | ✅ Match |
| 23 | gamma_volatility | `gamma * atr_14` | `gamma_calc * atr_14` | ✅ Match (uses calculated gamma) |
| 24 | rsi_bb_signal | `(rsi_14 / 100) * bb_position` | `(rsi_14 / 100) * bb_position` | ✅ Match |

### ✅ [25-26] Momentum Features - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 25 | momentum_3d | `close.diff(3) / close.shift(3)` | `(price[0] - price[3]) / price[3]` | ✅ Match |
| 26 | recent_win_rate | `rolling(10).mean()` of price_direction | `wins_10d / 10` | ✅ Match |

### ✅ [27-46] Price Lags - CORRECT
| Index | Feature | Python Calculation | C++ Calculation | Status |
|-------|---------|-------------------|-----------------|--------|
| 27-46 | price_lag_1d to 20d | `df.groupby('symbol')['close'].shift(i)` | `price_history[i]` | ✅ Match (actual prices) |

**Notes:**
- Python: `shift(1)` gives price 1 day ago, `shift(20)` gives price 20 days ago
- C++: `price_history[0]` = most recent, `price_history[1]` = 1 day ago, etc.
- Both store ACTUAL historical prices (not ratios)

### ✅ [47] Symbol Encoding - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 47 | symbol_encoded | `symbol_to_id[symbol]` (0-19) | `static_cast<float>(context.symbol_id)` | ✅ Match |

### ✅ [48-52] Time Features - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 48 | day_of_week | `Date.dt.dayofweek` (0=Mon) | `tm->tm_wday` (0=Sun) | ⚠️ DIFFERENT! |
| 49 | day_of_month | `Date.dt.day` | `tm->tm_mday` | ✅ Match |
| 50 | month_of_year | `Date.dt.month` | `tm->tm_mon + 1` | ✅ Match |
| 51 | quarter | `Date.dt.quarter` | `(tm->tm_mon / 3) + 1` | ✅ Match |
| 52 | day_of_year | `Date.dt.dayofyear` | `tm->tm_yday + 1` | ✅ Match |

**⚠️ CRITICAL:** day_of_week has different encoding!
- Python: 0=Monday, 6=Sunday
- C++: 0=Sunday, 6=Saturday

### ✅ [53-57] Directional Features - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 53 | price_direction | `(return_1d > 0).astype(int)` | `(return_1d > 0) ? 1 : 0` | ✅ Match |
| 54 | price_above_ma5 | `(close > MA5).astype(int)` | `(close > ma5) ? 1 : 0` | ✅ Match |
| 55 | price_above_ma20 | `(close > MA20).astype(int)` | `(close > ma20) ? 1 : 0` | ✅ Match |
| 56 | macd_signal_direction | `(macd > macd_signal).astype(int)` | `(macd > macd_signal) ? 1 : 0` | ✅ Match |
| 57 | volume_trend | `(volume_ratio > 1.0).astype(int)` | `(volume_ratio > 1) ? 1 : 0` | ✅ Match |

### ✅ [58-60] Date Components - CORRECT
| Index | Feature | Python | C++ | Status |
|-------|---------|--------|-----|--------|
| 58 | year | `Date.dt.year` | `tm->tm_year + 1900` | ✅ Match |
| 59 | month | `Date.dt.month` | `tm->tm_mon + 1` | ✅ Match |
| 60 | day | `Date.dt.day` | `tm->tm_mday` | ✅ Match |

### 🔴 [61-80] Price Diffs - MAJOR DISCREPANCY
| Index | Feature | Python Calculation | C++ Calculation | Status |
|-------|---------|-------------------|-----------------|--------|
| 61-80 | price_diff_1d to 20d | `df.groupby('symbol')['close'].diff(lag)` | `price_history[i] - price_history[i+1]` | 🔴 DIFFERENT! |

**CRITICAL DIFFERENCE:**

**Python:**
```python
price_diff_1d = df.groupby('symbol')['close'].diff(1)
# This is: current_price - price_1_day_ago
# For lag=1: price[t] - price[t-1]
# For lag=5: price[t] - price[t-5]
```

**C++:**
```cpp
price_diffs[0] = price_history[0] - price_history[1];  // 1-day diff
price_diffs[4] = price_history[4] - price_history[5];  // 5-day diff
```

**Problem:**
- Python `diff(lag)` computes: `price[t] - price[t-lag]`
- C++ computes: `price[i] - price[i+1]` where `i` is the lag index

These are FUNDAMENTALLY DIFFERENT!

**Example:**
- price_history = [100, 99, 98, 97, 96, 95]  (index 0 is most recent)
- Python `price_diff_1d` at time t: 100 - 99 = +1
- C++ `price_diffs[0]`: 100 - 99 = +1  ✅ Correct for lag=1

- Python `price_diff_5d` at time t: 100 - 95 = +5
- C++ `price_diffs[4]`: 95 - 96 = -1  🔴 WRONG!

**The C++ should be:**
```cpp
price_diffs[i] = price_history[0] - price_history[i+1];
```

### 🔴 [81-84] Autocorrelations - MAJOR DISCREPANCY
| Index | Feature | Python Calculation | C++ Calculation | Status |
|-------|---------|-------------------|-----------------|--------|
| 81 | autocorr_lag_1 | ACF of **returns** with lag=1, window=60 | ACF of **prices** with lag=1, window=30 | 🔴 DIFFERENT! |
| 82 | autocorr_lag_5 | ACF of **returns** with lag=5, window=60 | ACF of **prices** with lag=5, window=30 | 🔴 DIFFERENT! |
| 83 | autocorr_lag_10 | ACF of **returns** with lag=10, window=60 | ACF of **prices** with lag=10, window=30 | 🔴 DIFFERENT! |
| 84 | autocorr_lag_20 | ACF of **returns** with lag=20, window=60 | ACF of **prices** with lag=20, window=30 | 🔴 DIFFERENT! |

**CRITICAL DIFFERENCES:**

**Python:**
```python
df_clean['returns'] = df_clean.groupby('symbol')['close'].pct_change()

def compute_rolling_autocorr(series, lag, window=60):
    return series.rolling(window=window + lag).apply(
        lambda x: x.iloc[:-lag].corr(x.iloc[lag:]) if len(x) >= window + lag else np.nan,
        raw=False
    )

autocorr_lag_1 = df_clean.groupby('symbol')['returns'].transform(
    lambda x: compute_rolling_autocorr(x, lag=1)
)
```

**C++:**
```cpp
auto calc_autocorr = [&price_history](int lag) -> float {
    float mean = 0.0f;
    int count = min(30, price_history.size() - lag);
    for (int i = 0; i < count; ++i) {
        mean += price_history[i];
    }
    mean /= count;

    float num = 0.0f, denom = 0.0f;
    for (int i = 0; i < count; ++i) {
        float diff = price_history[i] - mean;
        float diff_lag = price_history[i + lag] - mean;
        num += diff * diff_lag;
        denom += diff * diff;
    }

    return (denom > 0.0f) ? (num / denom) : 0.0f;
};
```

**Problems:**
1. **Input Data:** Python uses **returns** (pct_change), C++ uses **prices**
2. **Window Size:** Python uses 60-period rolling window, C++ uses 30-period
3. **Rolling vs Static:** Python uses rolling correlation, C++ uses static correlation

---

## Summary of Discrepancies

### 🔴 CRITICAL (Must Fix)

1. **Price Diffs [61-80]:** C++ calculation is wrong
   - Current: `price_diffs[i] = price_history[i] - price_history[i+1]`
   - Should be: `price_diffs[i] = price_history[0] - price_history[i+1]`

2. **Autocorrelations [81-84]:** Completely different calculation
   - Python: Rolling ACF of **returns** with window=60
   - C++: Static ACF of **prices** with window=30

3. **Day of Week [48]:** Different encoding
   - Python: 0=Monday
   - C++: 0=Sunday (need to convert)

### ⚠️ IMPORTANT (Verify)

4. **Greeks [14-17]:** Formula differences
   - Theta: C++ missing second term
   - Rho: C++ using approximation

---

## Recommended Fixes

### Fix 1: Price Diffs
```cpp
// BEFORE (WRONG):
for (int i = 0; i < 20 && i + 1 < price_history.size(); ++i) {
    price_diffs[i] = price_history[i] - price_history[i + 1];
}

// AFTER (CORRECT):
for (int i = 0; i < 20 && i + 1 < price_history.size(); ++i) {
    price_diffs[i] = price_history[0] - price_history[i + 1];
}
```

### Fix 2: Autocorrelations
Need to completely rewrite to match Python:
```cpp
// 1. Calculate returns from price_history
std::vector<float> returns;
for (int i = 1; i < price_history.size(); ++i) {
    returns.push_back((price_history[i-1] - price_history[i]) / price_history[i]);
}

// 2. Compute rolling autocorrelation with window=60
auto calc_autocorr = [&returns](int lag, int window=60) -> float {
    if (returns.size() < window + lag) return 0.0f;

    // Take last (window + lag) returns
    int start = returns.size() - (window + lag);
    std::vector<float> x1, x2;
    for (int i = 0; i < window; ++i) {
        x1.push_back(returns[start + i]);
        x2.push_back(returns[start + i + lag]);
    }

    // Compute correlation between x1 and x2
    return compute_correlation(x1, x2);
};
```

### Fix 3: Day of Week
```cpp
// BEFORE (WRONG):
float day_of_week = static_cast<float>(tm->tm_wday);  // 0=Sunday

// AFTER (CORRECT):
float day_of_week = (tm->tm_wday == 0) ? 6.0f : static_cast<float>(tm->tm_wday - 1);  // 0=Monday
```

### Fix 4: Greeks (Theta)
```cpp
// Add the missing second term to theta
float norm_cdf_d2 = norm_cdf(d2);  // Need to implement norm_cdf
float theta_val = (-(S * n_prime_d1 * sigma) / (2.0f * sqrt_T)
                   - r * K * std::exp(-r * T) * norm_cdf_d2) / 365.0f;
```

---

## Testing Strategy

1. **Unit Test:** Create test with known input data
2. **Compare Output:** Run same data through Python and C++
3. **Tolerance:** Allow 1e-3 (0.001) difference for floating point
4. **Validation:** Ensure all 85 features match within tolerance

See: `tests/verify_feature_parity.cpp` (to be created)
