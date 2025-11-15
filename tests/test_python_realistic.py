"""
Test Python FP32 model with realistic features to establish baseline
"""
import numpy as np
import joblib
import torch
import torch.nn as nn
import sys

# Define model architecture (85 -> 256 -> 128 -> 64 -> 32 -> 3)
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

# Load model and scaler
model = PricePredictor()
checkpoint = torch.load('models/price_predictor_85feat_best.pth', map_location='cpu', weights_only=True)
# Check if it's a direct state dict or wrapped
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
scaler = joblib.load('models/scaler_85feat.pkl')

# Test cases with realistic features (from bot logs)
test_cases = {
    "SPY": {"close": 670.59, "symbol_enc": 0.0},
    "QQQ": {"close": 605.78, "symbol_enc": 1.0},
    "IWM": {"close": 236.45, "symbol_enc": 2.0}
}

print("=" * 80)
print("Python FP32 Model - Realistic Feature Test")
print("=" * 80)
print()

for symbol, data in test_cases.items():
    # Create feature vector (zeros except close and symbol_enc)
    features = np.zeros(85, dtype=np.float32)
    features[0] = data["close"]      # close price at index 0
    features[47] = data["symbol_enc"]  # symbol_enc at index 47

    # Normalize
    normalized = scaler.transform(features.reshape(1, -1))

    print(f"{symbol}:")
    print(f"  Raw: close={data['close']}, symbol_enc={data['symbol_enc']}")
    print(f"  Normalized[0]={normalized[0, 0]:.6f}, Normalized[47]={normalized[0, 47]:.6f}")

    # Predict
    with torch.no_grad():
        input_tensor = torch.from_numpy(normalized).float()
        output = model(input_tensor).numpy()[0]

    print(f"  Predictions: 1d={output[0]*100:.2f}%, 5d={output[1]*100:.2f}%, 20d={output[2]*100:.2f}%")
    print()

print("=" * 80)
print("Expected: SPY, QQQ, IWM should produce DIFFERENT predictions")
print("=" * 80)
