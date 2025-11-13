# Custom Price Prediction Model Architecture

## Overview
This document describes the custom neural network model for price prediction that incorporates comprehensive market, instrument-specific, macroeconomic, and sentiment data.

## Model Purpose
Predict future stock/option prices by analyzing:
- Instrument characteristics (symbol, type, sector)
- Market microstructure (Greeks, momentum, sentiment)
- Macroeconomic conditions (treasury rates, interest rates)
- Temporal patterns (time of day, date features)

## Input Features (12 Categories)

### 1. Stock Identification
- **Feature**: `symbol_encoded`
- **Type**: Integer (0-N for N unique symbols)
- **Description**: Numerical mapping of stock ticker symbols
- **Example**: SPY=0, QQQ=1, AAPL=2, etc.
- **Purpose**: Allow model to learn symbol-specific patterns

### 2. Business Sector
- **Feature**: `sector_encoded`
- **Type**: Integer (0-10 for 11 sectors)
- **Description**: GICS sector classification
- **Sectors**:
  - 0: Technology (XLK)
  - 1: Financials (XLF)
  - 2: Healthcare (XLV)
  - 3: Consumer Discretionary (XLY)
  - 4: Industrials (XLI)
  - 5: Energy (XLE)
  - 6: Consumer Staples (XLP)
  - 7: Utilities (XLU)
  - 8: Materials (XLB)
  - 9: Real Estate
  - 10: Communication Services
- **Purpose**: Capture sector-specific behavior and correlations

### 3. Instrument Type
- **Feature**: `is_option`
- **Type**: Binary (0 or 1)
- **Description**:
  - 0 = Stock/ETF
  - 1 = Option contract
- **Purpose**: Differentiate between stock and option pricing dynamics

### 4. Time Features
- **Features**:
  - `hour_of_day` (0-23)
  - `minute_of_hour` (0-59)
  - `day_of_week` (0=Monday, 6=Sunday)
  - `day_of_month` (1-31)
  - `month_of_year` (1-12)
  - `is_market_open` (binary: 0=closed, 1=open)
- **Type**: Integer
- **Description**: Temporal features to capture time-of-day effects
- **Purpose**: Model intraday volatility patterns, end-of-day effects, monthly cycles

### 5. Treasury Rates (Risk-Free Rate Curve)
- **Features**:
  - `treasury_1mo` - 1-month T-bill rate
  - `treasury_3mo` - 3-month T-bill rate
  - `treasury_6mo` - 6-month T-bill rate
  - `treasury_1yr` - 1-year Treasury rate
  - `treasury_2yr` - 2-year Treasury rate
  - `treasury_5yr` - 5-year Treasury rate
  - `treasury_10yr` - 10-year Treasury rate
  - `treasury_30yr` - 30-year Treasury rate
- **Type**: Float (percentage, e.g., 4.5 for 4.5%)
- **Source**: FRED API (Federal Reserve Economic Data)
- **Purpose**: Capture risk-free rate environment affecting discount rates and option pricing

### 6. Yield Curve Features
- **Features**:
  - `yield_curve_slope` - (10yr - 2yr) spread
  - `yield_curve_inversion` - Binary flag if 2yr > 10yr
- **Type**: Float / Binary
- **Purpose**: Detect recession signals and market risk sentiment

### 7. Options Greeks (for options or underlying stock)
- **Features**:
  - `delta` - Price sensitivity to underlying ($\Delta$)
  - `gamma` - Rate of delta change ($\Gamma$)
  - `theta` - Time decay ($\Theta$)
  - `vega` - Volatility sensitivity ($\mathcal{V}$)
  - `rho` - Interest rate sensitivity ($\rho$)
  - `implied_volatility` - IV percentage
- **Type**: Float
- **Description**:
  - For options: actual contract Greeks
  - For stocks: Greeks of ATM call option as proxy for volatility regime
- **Source**: `options_greek_cache` table (calculated by C++ engine)
- **Purpose**: Capture volatility expectations and option-specific risks

### 8. Interest Rate Environment
- **Features**:
  - `fed_funds_rate` - Federal funds effective rate
  - `real_interest_rate` - Nominal rate - inflation
  - `rate_change_1mo` - Change in fed funds over 1 month
- **Type**: Float
- **Source**: FRED API
- **Purpose**: Model cost of capital and monetary policy impact

### 9. Sentiment Score
- **Feature**: `avg_sentiment`
- **Type**: Float (-1.0 to +1.0)
- **Description**: Average sentiment from recent news articles (last 24 hours)
  - -1.0 = Very negative
  - 0.0 = Neutral
  - +1.0 = Very positive
- **Source**: `news_articles` table (sentiment_score column)
- **Purpose**: Capture market psychology and news-driven momentum

### 10. Trading Momentum Indicators
- **Features**:
  - `rsi_14` - Relative Strength Index (0-100)
  - `macd` - MACD line
  - `macd_signal` - MACD signal line
  - `volume_ratio` - Current volume / 20-day average
  - `price_momentum_1d` - 1-day return
  - `price_momentum_5d` - 5-day return
  - `price_momentum_20d` - 20-day return
- **Type**: Float
- **Purpose**: Quantify current market momentum and trend strength

### 11. Volatility Measures
- **Features**:
  - `atr_14` - Average True Range (14-period)
  - `bb_position` - Position within Bollinger Bands (0-1)
  - `historical_volatility_20d` - 20-day realized volatility
- **Type**: Float
- **Purpose**: Measure price stability and potential for large moves

### 12. Price & Volume Features
- **Features**:
  - `close` - Current close price
  - `open` - Open price
  - `high` - Daily high
  - `low` - Daily low
  - `volume` - Trading volume
  - `vwap` - Volume-weighted average price
- **Type**: Float
- **Purpose**: Core price action and liquidity measures

## Output (Target Variable)

### Target
- **Feature**: `target_price`
- **Type**: Float
- **Description**: Future price at prediction horizon
- **Variants**:
  - `target_1d` - Price 1 day ahead
  - `target_5d` - Price 5 days ahead
  - `target_20d` - Price 20 days ahead

## Model Architecture

### Neural Network Design
```
Input Layer: 60+ features (after encoding all categories)
    ↓
Hidden Layer 1: 256 neurons + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2: 128 neurons + ReLU + Dropout(0.3)
    ↓
Hidden Layer 3: 64 neurons + ReLU + Dropout(0.3)
    ↓
Hidden Layer 4: 32 neurons + ReLU + Dropout(0.2)
    ↓
Output Layer: 3 neurons (1d, 5d, 20d predictions)
```

### Total Parameters: ~50,000-70,000

### Loss Function
- **Primary**: Mean Squared Error (MSE)
- **Alternative**: Huber Loss (robust to outliers)

### Optimizer
- **Type**: Adam
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduler)
- **Weight Decay**: 1e-5 (L2 regularization)

## Data Sources

### 1. Stock Price Data
- **Source**: DuckDB `training_data.duckdb` table
- **Columns**: Date, symbol, OHLCV, technical indicators
- **Coverage**: 20 ETFs, 5 years historical

### 2. Treasury Rates
- **Source**: FRED API
- **Script**: `scripts/data_collection/fetch_treasury_rates.py`
- **Update**: Daily

### 3. Options Greeks
- **Source**: DuckDB `options_greek_cache` table
- **Calculation**: C++ engine (`src/options/black_scholes.cppm`)
- **Update**: Real-time during market hours

### 4. News Sentiment
- **Source**: DuckDB `news_articles` table
- **Calculation**: Keyword-based sentiment analyzer
- **Update**: Hourly via `scripts/data_collection/news_ingestion.py`

### 5. Interest Rates
- **Source**: FRED API (FEDFUNDS, T10YIE)
- **Update**: Daily

## Data Processing Pipeline

### 1. Feature Engineering
```python
# Symbol encoding
symbol_to_id = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
df['symbol_encoded'] = df['symbol'].map(symbol_to_id)

# Sector encoding
sector_map = {'XLK': 0, 'XLF': 1, ...}
df['sector_encoded'] = df['symbol'].map(sector_map)

# Time features
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Yield curve
df['yield_curve_slope'] = df['treasury_10yr'] - df['treasury_2yr']
df['yield_curve_inversion'] = (df['treasury_2yr'] > df['treasury_10yr']).astype(int)
```

### 2. Normalization
- **Method**: StandardScaler (zero mean, unit variance)
- **Applied to**: All continuous features
- **Not applied to**: Binary features (is_option, is_market_open)

### 3. Train/Validation/Test Split
- **Train**: 70% (oldest data)
- **Validation**: 15% (middle period)
- **Test**: 15% (most recent data)
- **Method**: Temporal split (no shuffle to prevent lookahead bias)

## Training Configuration

### Hyperparameters
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 512
- **Early Stopping Patience**: 15 epochs
- **Learning Rate Reduction**: Factor 0.5, patience 5 epochs

### Success Metrics
1. **RMSE < 2%** for 1-day predictions
2. **Directional Accuracy > 55%** (profitable threshold)
3. **Sharpe Ratio > 1.5** in backtesting
4. **Max Drawdown < 20%**

### Hardware Requirements
- **GPU**: NVIDIA RTX 4070 SUPER (12GB VRAM)
- **Training Time**: ~5-10 minutes
- **Inference**: <1ms per prediction (ONNX + CUDA)

## Model Outputs

### Files Generated
1. `models/custom_price_predictor.pth` - PyTorch checkpoint
2. `models/custom_price_predictor.onnx` - ONNX for C++ inference
3. `models/custom_price_predictor_info.json` - Training metadata
4. `models/custom_price_predictor_scaler.pkl` - StandardScaler for normalization

### Deployment
- **C++ Integration**: Load ONNX model in `src/market_intelligence/price_predictor.cppm`
- **Inference**: Real-time predictions during trading hours
- **Update Frequency**: Retrain weekly with new data

## Expected Performance

### Target Metrics (based on backtesting)
- **Win Rate**: 55-58%
- **Average Return per Trade**: 0.5-1.5%
- **Annual Return**: 50-150% (with 10% position sizing)
- **Sharpe Ratio**: 1.5-2.5
- **Max Drawdown**: 15-25%

### Profitability Analysis
With 55% win rate:
- Gross wins: 55% × trades
- Gross losses: 45% × trades
- After 37.1% tax + $0.65/contract commission
- **Net positive return**: 10-30% annually (conservative)

## Risk Considerations

### Model Limitations
1. **Black Swan Events**: Cannot predict unprecedented market crashes
2. **Regime Changes**: May underperform during structural market shifts
3. **Liquidity**: Assumes sufficient liquidity for execution
4. **Slippage**: Does not account for large order impact

### Mitigation Strategies
1. **Position Sizing**: Max 10% of capital per trade
2. **Stop Loss**: 5% per position
3. **Diversification**: Trade multiple symbols and sectors
4. **Regular Retraining**: Weekly model updates with new data

## Usage

### Training
```bash
uv run python scripts/ml/train_custom_price_predictor.py
```

### Inference (Python)
```python
import torch
model = torch.load('models/custom_price_predictor.pth')
prediction = model(features)
```

### Inference (C++ with ONNX)
```cpp
auto session = Ort::Session(env, "models/custom_price_predictor.onnx");
auto output = session.Run(..., input_tensor);
```

## Version History

### v1.0 (Current)
- Initial implementation with 60+ features
- 4-layer architecture
- Predicts 1d/5d/20d price movements
- CUDA-accelerated training and inference

---

**Last Updated**: 2025-11-12
**Author**: BigBrother Analytics ML Team
**Status**: In Development
