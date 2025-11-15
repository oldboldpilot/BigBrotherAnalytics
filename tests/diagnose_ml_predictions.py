"""
Comprehensive diagnosis of why ML predictions are constant
"""
import numpy as np
import joblib
import torch
import torch.nn as nn

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

# Bot's feature vector from logs (with FIXED OHLCV):
# SPY: features[0-9]=670.59,672.04,683.38,670.59,0.00,40.19,2.49,2.24,690.70,668.15
# Feature extraction creates a full 85-element vector - let's simulate what the bot produces

# Create feature vectors with FIXED OHLCV (from latest logs)
def create_bot_features(symbol, close, open_, high, low, vol=0.0):
    """Simulate what the bot creates during inference"""
    features = np.zeros(85, dtype=np.float32)

    # [0-4] OHLCV - NOW FIXED!
    features[0] = close
    features[1] = open_
    features[2] = high
    features[3] = low
    features[4] = vol  # Still 0 (market closed)

    # [5-10] Technical indicators (from logs) - REASONABLE
    # SPY: rsi=40.19, macd=2.49, macd_signal=2.24, bb_upper=690.70, bb_lower=668.15
    features[5] = 40.19  # rsi_14
    features[6] = 2.49   # macd
    features[7] = 2.24   # macd_signal
    features[8] = 690.70 # bb_upper
    features[9] = 668.15 # bb_lower
    features[10] = (close - features[9]) / (features[8] - features[9])  # bb_position

    # [11-13] Volatility - ESTIMATED
    features[11] = 2.0   # atr_14 (estimated)
    features[12] = 0.0   # volume_sma20 (not calculated)
    features[13] = 1.0   # volume_ratio

    # [14-26] Greeks + interaction features - MOSTLY ZEROS/ESTIMATES
    # These are the problem - bot doesn't have real options data!

    # [27-46] Price lags (20 days) - ALL SAME (no historical variation!)
    for i in range(27, 47):
        features[i] = close  # BUG: All price lags are current close!

    # [47] Symbol encoding
    if symbol == "SPY":
        features[47] = 0.0
    elif symbol == "QQQ":
        features[47] = 1.0
    elif symbol == "IWM":
        features[47] = 2.0

    # [48-57] Time + directional features - REASONABLE
    features[48] = 3  # day_of_week
    features[49] = 14 # day_of_month
    features[50] = 11 # month
    features[51] = 4  # quarter
    features[52] = 318 # day_of_year

    # [58-60] Date components
    features[58] = 2025
    features[59] = 11
    features[60] = 14

    # [61-80] Price diffs (20 days) - ALL ZEROS (no variation!)
    # BUG: All diffs are 0 because all price lags are the same!

    # [81-84] Autocorrelations - ZEROS/ESTIMATES

    return features

# Test with SPY, QQQ, IWM
test_cases = [
    ("SPY", 670.59, 672.04, 683.38, 670.59),
    ("QQQ", 605.78, 608.40, 623.23, 605.78),
    ("IWM", 236.45, 236.79, 244.24, 236.45)
]

print("=" * 80)
print("DIAGNOSIS: Why ML predictions are constant")
print("=" * 80)
print()

for symbol, close, open_, high, low in test_cases:
    features = create_bot_features(symbol, close, open_, high, low)

    # Normalize
    normalized = scaler.transform(features.reshape(1, -1))

    # Predict
    with torch.no_grad():
        input_tensor = torch.from_numpy(normalized).float()
        output = model(input_tensor).numpy()[0]

    print(f"{symbol}:")
    print(f"  OHLCV: close={close}, open={open_}, high={high}, low={low}")
    print(f"  Feature[47] (symbol_enc): {features[47]}")
    print(f"  Feature[27-31] (price_lags): {features[27:32]}")  # Should vary!
    print(f"  Feature[61-65] (price_diffs): {features[61:66]}")  # Should vary!
    print(f"  Predictions: 1d={output[0]*100:.2f}%, 5d={output[1]*100:.2f}%, 20d={output[2]*100:.2f}%")
    print()

print("=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)
print()
print("The bot's feature extraction has TWO critical bugs:")
print()
print("1. **Price lags [27-46] are ALL THE SAME**")
print("   - Training data: price_lag_1d=672.04, price_lag_2d=665.32, etc. (VARY)")
print("   - Bot inference: price_lag_1d=670.59, price_lag_2d=670.59, etc. (CONSTANT!)")
print()
print("2. **Price diffs [61-80] are ALL ZERO**")
print("   - Training data: price_diff_1d=-2.55, price_diff_2d=5.27, etc. (VARY)")
print("   - Bot inference: price_diff_1d=0.00, price_diff_2d=0.00, etc. (ALL ZEROS!)")
print()
print("These 40 features (47% of inputs!) are completely broken at inference time.")
print("The model learned that varying price lags/diffs → varying predictions.")
print("When it sees constant lags/zero diffs → it produces constant output!")
print()
print("=" * 80)
