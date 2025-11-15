"""
Analyze which features are varying vs constant across symbols
"""
import numpy as np

# Feature vectors from bot logs (after price_lag fix)
spy_features = [670.590027, 683.380005, 683.380005, 672.039978, 103193400.000000, 35.570595, 2.558472, 2.302624, 691.049255, 667.521790, 0.130410, 4.091548, 0.000000, 0.000000, 0.010000, -0.050000, 0.200000, 0.010000, -0.000000, -0.008183, 0.000000, -0.002164, 0.000059, 0.040915, 0.046388, -0.013794, 0.500000, 0.997842, 0.981284, 0.981830, 0.984078, 0.984078, 0.999434, 1.000418, 0.989684, 0.993114, 0.981342, 0.981342, 0.983183, 0.986408, 0.975560, 0.976028, 0.978621, 0.978621, 0.990166, 0.998258, 1.004178, 0.000000, 4.000000, 14.000000, 11.000000, 4.000000, 318.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 2025.000000, 11.000000, 14.000000, -11.340027, 0.380005, 1.559998, 0.000000, 10.470032, 0.659973, -7.270020, 2.340027, -8.100037, 0.000000, 1.280029, 2.229980, -7.559998, 0.330017, 1.820007, 0.000000, 7.989990, 5.489990, 3.960022, -3.489990, 0.776308, 0.426027, 0.180670, 0.349162]

qqq_features = [605.780029, 621.080017, 623.229980, 608.400024, 71071104.000000, 31.623505, 2.820740, 2.538666, 639.458740, 604.418213, 0.038864, 5.728469, 0.000000, 0.000000, 0.010000, -0.050000, 0.200000, 0.010000, -0.000000, -0.011457, 0.000000, -0.000793, -0.000256, 0.057285, 0.012290, -0.023795, 0.200000, 0.995694, 0.975366, 0.974597, 0.972001, 0.972001, 0.993505, 0.990371, 0.971923, 0.978248, 0.958391, 0.958391, 0.962977, 0.967622, 0.952829, 0.957119, 0.964480, 0.964480, 0.981656, 0.992139, 1.000479, 1.000000, 4.000000, 14.000000, 11.000000, 4.000000, 318.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 2025.000000, 11.000000, 14.000000, -12.679993, -0.489990, -1.659973, 0.000000, 13.489990, -1.929993, -11.610046, 4.030029, -12.830017, 0.000000, 3.010010, 3.020020, -9.720032, 2.850037, 4.829956, 0.000000, 10.990051, 6.519958, 5.090027, -5.890015, 0.793172, 0.518811, 0.231926, 0.357928]

iwm_features = [236.449997, 243.639999, 244.240005, 236.789993, 65619800.000000, 31.760414, -2.132446, -1.919202, 251.550674, 238.261276, -0.136294, 2.119229, 0.000000, 0.000000, 0.010000, -0.050000, 0.200000, 0.010000, -0.000000, -0.004238, -0.000000, 0.003832, -0.002168, 0.021192, -0.043288, -0.029669, 0.400000, 0.998564, 0.970489, 0.968105, 0.968938, 0.968938, 0.978643, 0.983774, 0.966364, 0.980307, 0.963372, 0.963372, 0.960281, 0.965693, 0.957908, 0.949751, 0.944666, 0.944666, 0.947961, 0.959541, 0.971686, 2.000000, 4.000000, 14.000000, 11.000000, 4.000000, 318.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2025.000000, 11.000000, 14.000000, -6.850006, -0.600006, 0.210007, 0.000000, 2.419998, 1.259995, -4.329987, 3.479996, -4.240005, 0.000000, -0.789993, 1.379990, -1.989990, -2.120010, -1.339996, 0.000000, 0.870010, 3.009995, 3.080002, -3.650009, 0.456577, 0.267060, 0.037412, -0.266169]

# Feature names from metadata
feature_names = [
    "close", "open", "high", "low", "volume",  # 0-4
    "rsi_14", "macd", "macd_signal",  # 5-7
    "bb_upper", "bb_lower", "bb_position",  # 8-10
    "atr_14", "volume_sma20", "volume_ratio",  # 11-13
    "gamma", "theta", "vega", "rho",  # 14-17
    "volume_rsi_signal", "yield_volatility", "macd_volume",  # 18-20
    "bb_momentum", "rate_return", "gamma_volatility", "rsi_bb_signal",  # 21-24
    "momentum_3d", "recent_win_rate",  # 25-26
    # Price lags 27-46
    "price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_4d", "price_lag_5d",
    "price_lag_6d", "price_lag_7d", "price_lag_8d", "price_lag_9d", "price_lag_10d",
    "price_lag_11d", "price_lag_12d", "price_lag_13d", "price_lag_14d", "price_lag_15d",
    "price_lag_16d", "price_lag_17d", "price_lag_18d", "price_lag_19d", "price_lag_20d",
    "symbol_encoded",  # 47
    "day_of_week", "day_of_month", "month_of_year", "quarter", "day_of_year",  # 48-52
    "price_direction", "price_above_ma5", "price_above_ma20",  # 53-55
    "macd_signal_direction", "volume_trend",  # 56-57
    "year", "month", "day",  # 58-60
    # Price diffs 61-80
    "price_diff_1d", "price_diff_2d", "price_diff_3d", "price_diff_4d", "price_diff_5d",
    "price_diff_6d", "price_diff_7d", "price_diff_8d", "price_diff_9d", "price_diff_10d",
    "price_diff_11d", "price_diff_12d", "price_diff_13d", "price_diff_14d", "price_diff_15d",
    "price_diff_16d", "price_diff_17d", "price_diff_18d", "price_diff_19d", "price_diff_20d",
    "autocorr_lag_1", "autocorr_lag_5", "autocorr_lag_10", "autocorr_lag_20"  # 81-84
]

spy = np.array(spy_features)
qqq = np.array(qqq_features)
iwm = np.array(iwm_features)

print("=" * 100)
print("FEATURE VARIANCE ANALYSIS")
print("=" * 100)
print()

# Check which features are IDENTICAL across all symbols
constant_features = []
varying_features = []

for i in range(85):
    if abs(spy[i] - qqq[i]) < 1e-6 and abs(spy[i] - iwm[i]) < 1e-6:
        constant_features.append((i, feature_names[i], spy[i]))
    else:
        varying_features.append((i, feature_names[i]))

print(f"CONSTANT FEATURES (identical for all symbols): {len(constant_features)}/85")
print("-" * 100)
for idx, name, value in constant_features:
    print(f"  [{idx:2d}] {name:30s} = {value:10.6f}")

print()
print(f"VARYING FEATURES (different by symbol): {len(varying_features)}/85")
print("-" * 100)
for idx, name in varying_features:
    print(f"  [{idx:2d}] {name:30s}  SPY={spy[idx]:10.6f}  QQQ={qqq[idx]:10.6f}  IWM={iwm[idx]:10.6f}")

print()
print("=" * 100)
print("PROBLEM ANALYSIS")
print("=" * 100)
print()

# Check features that are ZEROS or hardcoded
zeros_features = []
for i in range(85):
    if abs(spy[i]) < 1e-6 and abs(qqq[i]) < 1e-6 and abs(iwm[i]) < 1e-6:
        zeros_features.append((i, feature_names[i]))

print(f"FEATURES THAT ARE ALL ZEROS: {len(zeros_features)}")
for idx, name in zeros_features:
    print(f"  [{idx:2d}] {name}")

print()
# Check Greeks (features 14-17)
print("GREEKS (features 14-17):")
print(f"  SPY: gamma={spy[14]:.6f}, theta={spy[15]:.6f}, vega={spy[16]:.6f}, rho={spy[17]:.6f}")
print(f"  QQQ: gamma={qqq[14]:.6f}, theta={qqq[15]:.6f}, vega={qqq[16]:.6f}, rho={qqq[17]:.6f}")
print(f"  IWM: gamma={iwm[14]:.6f}, theta={iwm[15]:.6f}, vega={iwm[16]:.6f}, rho={iwm[17]:.6f}")
print("  ⚠️ Greeks are IDENTICAL → Likely hardcoded estimates, not real options data!")

print()
print("SUMMARY:")
print(f"  - {len(constant_features)} features are CONSTANT across all symbols")
print(f"  - {len(varying_features)} features actually vary")
print(f"  - {len(zeros_features)} features are ALL ZEROS")
print()
print("If the model learned during training that varying features are important,")
print("but at inference time most features are constant/zero, it may produce")
print("constant outputs regardless of the few features that do vary.")
