"""
Verify ML predictions WITH real historical data (simulating fixed bot)
"""
import numpy as np
import joblib
import torch
import torch.nn as nn
import duckdb

# Load model and scaler
class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(85, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.network(x)

model = PricePredictor()
checkpoint = torch.load('models/price_predictor_85feat_best.pth', map_location='cpu', weights_only=True)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

scaler = joblib.load('models/scaler_85feat.pkl')

# Load historical data from database
def load_historical_prices(symbol):
    """Load 100 days of historical prices from database"""
    conn = duckdb.connect('data/bigbrother.duckdb', read_only=True)
    query = f"""
        SELECT date, close, volume
        FROM stock_prices
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT 100
    """
    result = conn.execute(query).fetchall()
    conn.close()

    # Reverse to chronological order (oldest first)
    result = list(reversed(result))

    prices = [row[1] for row in result]
    return np.array(prices, dtype=np.float32)

# Create feature vectors WITH real historical data
def create_bot_features_with_history(symbol, current_close, current_open, current_high, current_low):
    """Simulate what the bot creates WITH real historical data"""
    features = np.zeros(85, dtype=np.float32)

    # Load real historical prices
    historical_prices = load_historical_prices(symbol)
    print(f"  Loaded {len(historical_prices)} historical prices for {symbol}")
    print(f"  Historical prices (last 5): {historical_prices[-5:]}")

    # [0-4] OHLCV
    features[0] = current_close
    features[1] = current_open
    features[2] = current_high
    features[3] = current_low
    features[4] = 0.0  # volume (market closed)

    # [5-10] Technical indicators (simplified - bot calculates these)
    features[5] = 40.0  # rsi_14 (estimated)
    features[6] = 2.0   # macd
    features[7] = 2.0   # macd_signal
    features[8] = current_close * 1.02  # bb_upper (estimated)
    features[9] = current_close * 0.98  # bb_lower (estimated)
    features[10] = 0.5  # bb_position

    # [11-13] Volatility
    features[11] = 2.0   # atr_14
    features[12] = 0.0   # volume_sma20
    features[13] = 1.0   # volume_ratio

    # [14-26] Greeks + interaction features (mostly zeros for now)

    # [27-46] Price lags (20 days) - NOW WITH REAL DATA!
    for lag in range(1, 21):  # 1 to 20 days
        lag_idx = 27 + lag - 1
        if lag < len(historical_prices):
            features[lag_idx] = historical_prices[-(lag + 1)]  # lag days ago
        else:
            features[lag_idx] = current_close  # fallback

    # [47] Symbol encoding
    symbol_map = {"SPY": 0.0, "QQQ": 1.0, "IWM": 2.0}
    features[47] = symbol_map.get(symbol, 0.0)

    # [48-57] Time + directional features
    features[48] = 3  # day_of_week
    features[49] = 14 # day_of_month
    features[50] = 11 # month
    features[51] = 4  # quarter
    features[52] = 318 # day_of_year

    # [58-60] Date components
    features[58] = 2025
    features[59] = 11
    features[60] = 14

    # [61-80] Price diffs (20 days) - NOW COMPUTED FROM REAL DATA!
    for lag in range(1, 21):
        diff_idx = 61 + lag - 1
        if lag < len(historical_prices):
            features[diff_idx] = current_close - historical_prices[-(lag + 1)]
        else:
            features[diff_idx] = 0.0

    # [81-84] Autocorrelations (simplified)
    features[81] = 1.0
    features[82] = 1.0
    features[83] = 1.0
    features[84] = 1.0

    return features

# Test with SPY, QQQ, IWM
test_cases = [
    ("SPY", 670.59, 672.04, 683.38, 670.59),
    ("QQQ", 605.78, 608.40, 623.23, 605.78),
    ("IWM", 236.45, 236.79, 244.24, 236.45)
]

print("=" * 80)
print("VERIFICATION: ML predictions WITH real historical data")
print("=" * 80)
print()

for symbol, close, open_, high, low in test_cases:
    print(f"{symbol}:")
    print(f"  Current OHLC: close={close}, open={open_}, high={high}, low={low}")

    features = create_bot_features_with_history(symbol, close, open_, high, low)

    print(f"  Feature[0] (close): {features[0]:.2f}")
    print(f"  Feature[27] (price_lag_1d): {features[27]:.2f}")
    print(f"  Feature[28] (price_lag_2d): {features[28]:.2f}")
    print(f"  Feature[29] (price_lag_3d): {features[29]:.2f}")
    print(f"  Feature[47] (symbol_enc): {features[47]}")
    print(f"  Feature[61] (price_diff_1d): {features[61]:.2f}")
    print(f"  Feature[62] (price_diff_2d): {features[62]:.2f}")
    print(f"  Feature[63] (price_diff_3d): {features[63]:.2f}")

    # Normalize
    normalized = scaler.transform(features.reshape(1, -1))

    # Predict
    with torch.no_grad():
        input_tensor = torch.from_numpy(normalized).float()
        output = model(input_tensor).numpy()[0]

    print(f"  Predictions: 1d={output[0]*100:.2f}%, 5d={output[1]*100:.2f}%, 20d={output[2]*100:.2f}%")
    print()

print("=" * 80)
print("EXPECTED RESULT:")
print("=" * 80)
print("With real historical data, predictions should be:")
print("- DIFFERENT for each symbol (not identical)")
print("- Price lags should show actual historical prices (not all = current)")
print("- Price diffs should show actual differences (not all = 0)")
print("=" * 80)
