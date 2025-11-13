#!/usr/bin/env python3
"""
Prepare Custom Features for Price Prediction Model - JAX Accelerated

This script gathers all required features from multiple data sources:
- Stock prices and technical indicators (training_data.duckdb)
- Treasury rates and interest rates (bigbrother.duckdb: risk_free_rates)
- Company sectors (bigbrother.duckdb: company_sectors)
- News sentiment (bigbrother.duckdb: news_articles)
- Options Greeks (calculated from options_calls/puts + Black-Scholes with JAX)
- Time features (extracted from timestamps)

Uses JAX for GPU-accelerated numerical computations

Output: Comprehensive feature matrix ready for model training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import duckdb

# JAX imports for GPU acceleration
import jax
import jax.numpy as jnp
from jax import jit, vmap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("CUSTOM PRICE PREDICTION MODEL - FEATURE PREPARATION (JAX Accelerated)")
print("="*80)

# Check JAX backend
print(f"\n🖥️  JAX Backend: {jax.default_backend()}")
if jax.default_backend() == 'gpu':
    print(f"   GPU devices: {jax.devices()}")
else:
    print(f"   Using CPU (install jaxlib with CUDA for GPU acceleration)")
print()

# ============================================================================
# 1. Load Stock Price Data with Technical Indicators
# ============================================================================
print("\n[1/6] Loading stock price data...")
conn_training = duckdb.connect('data/training_data.duckdb', read_only=True)

# Load all splits
train_df = conn_training.execute("SELECT * FROM train").df()
val_df = conn_training.execute("SELECT * FROM validation").df()
test_df = conn_training.execute("SELECT * FROM test").df()

# Combine for feature engineering (we'll split again later)
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"   Total samples: {len(df):,}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Symbols: {df['symbol'].nunique()} unique")

conn_training.close()

# ============================================================================
# 2. Add Symbol and Sector Encoding
# ============================================================================
print("\n[2/6] Adding symbol and sector encoding...")
conn_main = duckdb.connect('data/bigbrother.duckdb', read_only=True)

# Get sector mapping
sectors_df = conn_main.execute("SELECT symbol, sector_code FROM company_sectors").df()
sector_map = dict(zip(sectors_df['symbol'], sectors_df['sector_code']))

# Create symbol encoding
unique_symbols = sorted(df['symbol'].unique())
symbol_to_id = {symbol: idx for idx, symbol in enumerate(unique_symbols)}

df['symbol_encoded'] = df['symbol'].map(symbol_to_id)

# Map sectors (use -1 for ETFs without sector info)
df['sector_encoded'] = df['symbol'].map(sector_map).fillna(-1).astype(int)

print(f"   Symbol encoding: {len(symbol_to_id)} symbols")
print(f"   Sector mapping: {df['sector_encoded'].nunique()} sectors")
print(f"   Symbols: {dict(list(symbol_to_id.items())[:5])}... (showing first 5)")

# ============================================================================
# 3. Add Instrument Type (Stock vs Option)
# ============================================================================
print("\n[3/6] Adding instrument type...")
# For now, all are stocks/ETFs (value = 0)
# When we add options trading, we'll set is_option = 1
df['is_option'] = 0
print(f"   All instruments are stocks/ETFs (is_option=0)")

# ============================================================================
# 4. Add Time Features
# ============================================================================
print("\n[4/6] Extracting time features...")

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract time features
df['hour_of_day'] = df['Date'].dt.hour
df['minute_of_hour'] = df['Date'].dt.minute
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_month'] = df['Date'].dt.day
df['month_of_year'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['day_of_year'] = df['Date'].dt.dayofyear

# Market hours indicator (9:30 AM - 4:00 PM ET)
# Note: Our data is likely daily close (4:00 PM), so most will be 1
df['is_market_open'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] < 16)).astype(int)

print(f"   Time features extracted: hour, day_of_week, month, etc.")
print(f"   Sample dates: {df['Date'].iloc[0]}, {df['Date'].iloc[len(df)//2]}, {df['Date'].iloc[-1]}")

# ============================================================================
# 5. Add Treasury Rates and Interest Rates
# ============================================================================
print("\n[5/6] Adding treasury rates and interest rates...")

# Get latest rates (we'll use these for all historical data for now)
# In production, you'd join on date to get historical rates
rates_df = conn_main.execute("SELECT * FROM risk_free_rates").df()
rates_dict = dict(zip(rates_df['rate_name'], rates_df['rate_value']))

print(f"   Available rates: {list(rates_dict.keys())}")

# Add rates as columns (broadcast to all rows)
df['fed_funds_rate'] = rates_dict.get('fed_funds_rate', 0.0387)
df['treasury_3mo'] = rates_dict.get('3_month_treasury', 0.0392)
df['treasury_2yr'] = rates_dict.get('2_year_treasury', 0.0355)
df['treasury_5yr'] = rates_dict.get('5_year_treasury', 0.0367)
df['treasury_10yr'] = rates_dict.get('10_year_treasury', 0.0411)

# Calculate yield curve features
df['yield_curve_slope'] = df['treasury_10yr'] - df['treasury_2yr']
df['yield_curve_inversion'] = (df['treasury_2yr'] > df['treasury_10yr']).astype(int)

print(f"   Fed funds rate: {df['fed_funds_rate'].iloc[0]:.4f}")
print(f"   Treasury 10Y: {df['treasury_10yr'].iloc[0]:.4f}")
print(f"   Yield curve slope: {df['yield_curve_slope'].iloc[0]:.4f}")

# ============================================================================
# 6. Add News Sentiment
# ============================================================================
print("\n[6/6] Adding news sentiment...")

# Get sentiment data
news_df = conn_main.execute("""
    SELECT
        symbol,
        DATE_TRUNC('day', published_at) as news_date,
        AVG(sentiment_score) as avg_sentiment,
        COUNT(*) as news_count
    FROM news_articles
    WHERE sentiment_score IS NOT NULL
    GROUP BY symbol, DATE_TRUNC('day', published_at)
""").df()

print(f"   News articles: {conn_main.execute('SELECT COUNT(*) FROM news_articles').fetchone()[0]:,}")
print(f"   Unique symbols with news: {news_df['symbol'].nunique()}")

conn_main.close()

# Prepare for merge
df['trade_date'] = df['Date'].dt.date
news_df['news_date'] = pd.to_datetime(news_df['news_date']).dt.date

# Merge sentiment (left join)
df = df.merge(
    news_df[['symbol', 'news_date', 'avg_sentiment', 'news_count']],
    left_on=['symbol', 'trade_date'],
    right_on=['symbol', 'news_date'],
    how='left'
)

# Fill missing sentiment with 0 (neutral)
df['avg_sentiment'] = df['avg_sentiment'].fillna(0.0)
df['news_count'] = df['news_count'].fillna(0).astype(int)

print(f"   Average sentiment range: [{df['avg_sentiment'].min():.3f}, {df['avg_sentiment'].max():.3f}]")
print(f"   Rows with sentiment data: {(df['news_count'] > 0).sum():,} ({(df['news_count'] > 0).sum()/len(df)*100:.1f}%)")

# Drop temporary columns
df = df.drop(columns=['trade_date', 'news_date'], errors='ignore')

# ============================================================================
# 7. Add Options Greeks (JAX-Accelerated Black-Scholes)
# ============================================================================
print("\n[7/7] Adding options Greeks with JAX acceleration...")

# For stocks/ETFs, we'll use ATM option Greeks as proxy for volatility regime

# JAX-accelerated Black-Scholes Greeks calculation
@jit
def norm_cdf(x):
    """Standard normal CDF using error function"""
    return 0.5 * (1.0 + jnp.tanh(x / jnp.sqrt(2.0) * 0.7978845608))

@jit
def norm_pdf(x):
    """Standard normal PDF"""
    return jnp.exp(-0.5 * x**2) / jnp.sqrt(2.0 * jnp.pi)

@jit
def calculate_greeks_jax(S, K, T, r, sigma):
    """
    Calculate Black-Scholes Greeks using JAX (GPU-accelerated)

    Args:
        S: Stock price
        K: Strike price (ATM = S)
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility

    Returns:
        (delta, gamma, theta, vega, rho, iv)
    """
    # Prevent division by zero
    sigma = jnp.maximum(sigma, 0.01)
    T = jnp.maximum(T, 1/365)

    # d1 and d2 for Black-Scholes
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    # Greeks
    delta = norm_cdf(d1)  # Call delta
    gamma = norm_pdf(d1) / (S * sigma * jnp.sqrt(T))
    theta = (-S * norm_pdf(d1) * sigma / (2 * jnp.sqrt(T))
             - r * K * jnp.exp(-r * T) * norm_cdf(d2)) / 365
    vega = S * norm_pdf(d1) * jnp.sqrt(T) / 100
    rho = K * T * jnp.exp(-r * T) * norm_cdf(d2) / 100

    return delta, gamma, theta, vega, rho, sigma

# Vectorize the function to process all rows at once
calculate_greeks_vectorized = vmap(calculate_greeks_jax)

print("   Calculating Greeks using JAX-accelerated Black-Scholes...")
print(f"   Processing {len(df):,} samples on {jax.default_backend().upper()}...")

# Prepare input arrays
S_array = jnp.array(df['close'].values)
K_array = S_array  # ATM strike
T_array = jnp.full(len(df), 30/365)  # 30 days to expiration
r_array = jnp.array(df['treasury_3mo'].values)
sigma_array = jnp.full(len(df), 0.25)  # Assume 25% IV

# Calculate all Greeks at once (vectorized on GPU if available)
start_greeks = datetime.now()
delta, gamma, theta, vega, rho, iv = calculate_greeks_vectorized(
    S_array, K_array, T_array, r_array, sigma_array
)
greeks_time = (datetime.now() - start_greeks).total_seconds()

# Convert back to numpy and add to dataframe
df['delta'] = np.array(delta)
df['gamma'] = np.array(gamma)
df['theta'] = np.array(theta)
df['vega'] = np.array(vega)
df['rho'] = np.array(rho)
df['implied_volatility'] = np.array(iv)

print(f"   ✅ Greeks calculated in {greeks_time:.3f}s ({len(df)/greeks_time:,.0f} samples/sec)")
print(f"   Delta range: [{df['delta'].min():.3f}, {df['delta'].max():.3f}]")
print(f"   Gamma range: [{df['gamma'].min():.5f}, {df['gamma'].max():.5f}]")

# ============================================================================
# 8. Add Feature Interactions (Non-linear relationships)
# ============================================================================
print("\n[8/8] Creating feature interactions...")

# Interaction 1: Sentiment × Momentum (news amplifying trends)
df['sentiment_momentum'] = df['avg_sentiment'] * df['return_5d']
print(f"   sentiment_momentum: {df['sentiment_momentum'].min():.4f} to {df['sentiment_momentum'].max():.4f}")

# Interaction 2: Volume × RSI (volume confirming overbought/oversold)
df['volume_rsi_signal'] = df['volume_ratio'] * (df['rsi_14'] - 50) / 50  # Normalized RSI
print(f"   volume_rsi_signal: {df['volume_rsi_signal'].min():.4f} to {df['volume_rsi_signal'].max():.4f}")

# Interaction 3: Yield Curve × Volatility (rate environment affecting volatility)
df['yield_volatility'] = df['yield_curve_slope'] * df['atr_14']
print(f"   yield_volatility: {df['yield_volatility'].min():.4f} to {df['yield_volatility'].max():.4f}")

# Interaction 4: Delta × IV (options sensitivity to volatility)
df['delta_iv'] = df['delta'] * df['implied_volatility']
print(f"   delta_iv: {df['delta_iv'].min():.4f} to {df['delta_iv'].max():.4f}")

# Interaction 5: MACD × Volume (momentum confirmed by volume)
df['macd_volume'] = df['macd'] * df['volume_ratio']
print(f"   macd_volume: {df['macd_volume'].min():.4f} to {df['macd_volume'].max():.4f}")

# Interaction 6: Bollinger Position × Momentum (mean reversion vs trend)
df['bb_momentum'] = df['bb_position'] * df['return_1d']
print(f"   bb_momentum: {df['bb_momentum'].min():.4f} to {df['bb_momentum'].max():.4f}")

# Interaction 7: Sentiment × News Count (sentiment strength weighted by volume)
df['sentiment_strength'] = df['avg_sentiment'] * np.log1p(df['news_count'])
print(f"   sentiment_strength: {df['sentiment_strength'].min():.4f} to {df['sentiment_strength'].max():.4f}")

# Interaction 8: Treasury Rate × Return (rate environment affecting returns)
df['rate_return'] = df['fed_funds_rate'] * df['return_20d']
print(f"   rate_return: {df['rate_return'].min():.4f} to {df['rate_return'].max():.4f}")

# Interaction 9: Gamma × ATR (options convexity in volatile markets)
df['gamma_volatility'] = df['gamma'] * df['atr_14']
print(f"   gamma_volatility: {df['gamma_volatility'].min():.4f} to {df['gamma_volatility'].max():.4f}")

# Interaction 10: RSI × BB Position (overbought/oversold confirmation)
df['rsi_bb_signal'] = (df['rsi_14'] / 100) * df['bb_position']
print(f"   rsi_bb_signal: {df['rsi_bb_signal'].min():.4f} to {df['rsi_bb_signal'].max():.4f}")

print(f"   ✅ Created 10 interaction features")

# ============================================================================
# 8.5. Add Price Directionality Features (For Direction Prediction)
# ============================================================================
print("\n[8.5/9] Adding price directionality features...")

# Direction 1: Current price direction (binary: up=1, down=0)
df['price_direction'] = (df['return_1d'] > 0).astype(int)
print(f"   price_direction: {df['price_direction'].min():.0f} to {df['price_direction'].max():.0f}")

# Direction 2: Trend strength (consecutive up/down days)
df['trend_strength'] = df.groupby('symbol')['price_direction'].transform(
    lambda x: x.rolling(window=5, min_periods=1).sum() - 2.5  # -2.5 to 2.5 range
)
print(f"   trend_strength: {df['trend_strength'].min():.4f} to {df['trend_strength'].max():.4f}")

# Direction 3: Price position relative to moving averages
df['price_above_ma5'] = (df['close'] > df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())).astype(int)
df['price_above_ma20'] = (df['close'] > df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())).astype(int)
print(f"   price_above_ma5: Binary (0/1)")
print(f"   price_above_ma20: Binary (0/1)")

# Direction 4: Directional momentum (3-day slope)
df['momentum_3d'] = df.groupby('symbol')['close'].transform(
    lambda x: x.diff(3) / x.shift(3)
)
print(f"   momentum_3d: {df['momentum_3d'].min():.4f} to {df['momentum_3d'].max():.4f}")

# Direction 5: MACD signal (buy=1, sell=0)
df['macd_signal_direction'] = (df['macd'] > df['macd_signal']).astype(int)
print(f"   macd_signal_direction: Binary (0/1)")

# Direction 6: Volume trend (increasing=1, decreasing=0)
df['volume_trend'] = (df['volume_ratio'] > 1.0).astype(int)
print(f"   volume_trend: Binary (0/1)")

# Direction 7: Recent win rate (% of positive days in last 10 days)
df['recent_win_rate'] = df.groupby('symbol')['price_direction'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)
print(f"   recent_win_rate: {df['recent_win_rate'].min():.4f} to {df['recent_win_rate'].max():.4f}")

# Fill NaN values from rolling calculations
directional_features_to_fill = ['trend_strength', 'price_above_ma5', 'price_above_ma20',
                                 'momentum_3d', 'recent_win_rate']
for feat in directional_features_to_fill:
    df[feat] = df[feat].fillna(0)

print(f"   ✅ Created 7 price directionality features")

# ============================================================================
# 8.6. Add Historical Price Sequence (Last 20 Days)
# ============================================================================
print("\n[8.6/9] Adding historical price sequence (last 20 days)...")

# Create lagged price features (last 20 days of closing prices)
# Normalize by current price to make it relative
for i in range(1, 21):
    col_name = f'price_lag_{i}d'
    df[col_name] = df.groupby('symbol')['close'].shift(i) / df['close']  # Relative to current
    print(f"   {col_name}: relative price {i} days ago")

# Fill NaN values (for early dates where we don't have 20 days history)
lag_features = [f'price_lag_{i}d' for i in range(1, 21)]
for feat in lag_features:
    df[feat] = df[feat].fillna(1.0)  # Fill with 1.0 (meaning same price)

print(f"   ✅ Created 20 historical price features (normalized relative to current)")

# ============================================================================
# 9. Summary of Features
# ============================================================================
print("\n" + "="*80)
print("FEATURE SUMMARY")
print("="*80)

# Count feature categories
identification_features = ['symbol_encoded', 'sector_encoded', 'is_option']
time_features = ['hour_of_day', 'minute_of_hour', 'day_of_week', 'day_of_month',
                 'month_of_year', 'quarter', 'day_of_year', 'is_market_open']
treasury_features = ['fed_funds_rate', 'treasury_3mo', 'treasury_2yr',
                     'treasury_5yr', 'treasury_10yr', 'yield_curve_slope',
                     'yield_curve_inversion']
greeks_features = ['delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility']
sentiment_features = ['avg_sentiment', 'news_count']
price_features = ['close', 'open', 'high', 'low', 'volume']
momentum_features = ['return_1d', 'return_5d', 'return_20d', 'rsi_14', 'macd',
                     'macd_signal', 'volume_ratio']
volatility_features = ['atr_14', 'bb_upper', 'bb_lower', 'bb_position']
interaction_features = ['sentiment_momentum', 'volume_rsi_signal', 'yield_volatility',
                       'delta_iv', 'macd_volume', 'bb_momentum', 'sentiment_strength',
                       'rate_return', 'gamma_volatility', 'rsi_bb_signal']
directional_features = ['price_direction', 'trend_strength', 'price_above_ma5',
                       'price_above_ma20', 'momentum_3d', 'macd_signal_direction',
                       'volume_trend', 'recent_win_rate']
historical_price_features = [f'price_lag_{i}d' for i in range(1, 21)]
target_features = ['target_1d', 'target_5d', 'target_20d']

all_feature_cols = (identification_features + time_features + treasury_features +
                    greeks_features + sentiment_features + price_features +
                    momentum_features + volatility_features + interaction_features +
                    directional_features + historical_price_features)

print(f"\nIdentification Features ({len(identification_features)}): {identification_features}")
print(f"Time Features ({len(time_features)}): {time_features[:3]}... (showing first 3)")
print(f"Treasury/Rates Features ({len(treasury_features)}): {treasury_features[:3]}... (showing first 3)")
print(f"Greeks Features ({len(greeks_features)}): {greeks_features}")
print(f"Sentiment Features ({len(sentiment_features)}): {sentiment_features}")
print(f"Price Features ({len(price_features)}): {price_features}")
print(f"Momentum Features ({len(momentum_features)}): {momentum_features[:3]}... (showing first 3)")
print(f"Volatility Features ({len(volatility_features)}): {volatility_features}")
print(f"Interaction Features ({len(interaction_features)}): {interaction_features[:3]}... (showing first 3)")
print(f"Directional Features ({len(directional_features)}): {directional_features[:3]}... (showing first 3)")
print(f"Historical Price Features ({len(historical_price_features)}): {historical_price_features[:3]}... (showing first 3)")
print(f"Target Features ({len(target_features)}): {target_features}")

print(f"\n{'='*80}")
print(f"TOTAL INPUT FEATURES: {len(all_feature_cols)}")
print(f"TOTAL OUTPUT TARGETS: {len(target_features)}")
print(f"{'='*80}")

# ============================================================================
# 9. Save Prepared Dataset
# ============================================================================
print("\n[Saving] Writing prepared dataset...")

# Save to new DuckDB database
output_db = 'data/custom_training_data.duckdb'
conn_output = duckdb.connect(output_db)

# Create table with all features
conn_output.execute("DROP TABLE IF EXISTS features_full")
conn_output.execute(f"""
    CREATE TABLE features_full AS
    SELECT * FROM df
""")

# Split back into train/val/test (70/15/15 temporal split)
n_total = len(df)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)

df_sorted = df.sort_values('Date').reset_index(drop=True)

train_split = df_sorted.iloc[:n_train]
val_split = df_sorted.iloc[n_train:n_train+n_val]
test_split = df_sorted.iloc[n_train+n_val:]

conn_output.execute("DROP TABLE IF EXISTS train")
conn_output.execute("DROP TABLE IF EXISTS validation")
conn_output.execute("DROP TABLE IF EXISTS test")

conn_output.execute("CREATE TABLE train AS SELECT * FROM train_split")
conn_output.execute("CREATE TABLE validation AS SELECT * FROM val_split")
conn_output.execute("CREATE TABLE test AS SELECT * FROM test_split")

# Save feature metadata
feature_metadata = {
    'total_features': len(all_feature_cols),
    'feature_columns': all_feature_cols,
    'target_columns': target_features,
    'train_samples': len(train_split),
    'val_samples': len(val_split),
    'test_samples': len(test_split),
    'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
    'symbols': unique_symbols,
    'created_at': datetime.now().isoformat()
}

import json
with open('models/custom_features_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

conn_output.close()

print(f"   ✅ Saved to: {output_db}")
print(f"   Train: {len(train_split):,} samples")
print(f"   Validation: {len(val_split):,} samples")
print(f"   Test: {len(test_split):,} samples")
print(f"   Metadata: models/custom_features_metadata.json")

print("\n" + "="*80)
print("FEATURE PREPARATION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review feature distributions: df.describe()")
print("2. Train custom model: uv run python scripts/ml/train_custom_price_predictor.py")
print("3. Backtest performance: uv run python scripts/ml/backtest_model.py")
print("="*80)
