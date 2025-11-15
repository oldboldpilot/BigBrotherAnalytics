#!/usr/bin/env python3
"""
Verify Perfect Parity Between Python and C++ ML Predictions

This test validates that the C++ INT32 SIMD inference engine produces
IDENTICAL results to the Python PyTorch model when using the same features.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import feature_extractor_cpp

# Load trained model
print("=" * 80)
print("C++ vs PYTHON PARITY VALIDATION TEST")
print("=" * 80)
print()

# Load model info
with open('models/price_predictor_cpp_85feat_info.json', 'r') as f:
    model_info = json.load(f)

print(f"Model: {model_info['model_type']}")
print(f"Test RMSE: {model_info['test_rmse']:.6f}")
print(f"Test Accuracy: {model_info['test_accuracy']:.4f}")
print()

# Create test features (simulate real market data)
print("[1/4] Creating test features...")
test_features = {
    'close': 225.99,
    'open': 225.96,
    'high': 227.49,
    'low': 224.34,
    'volume': 31335322.0,
    'rsi_14': 54.67,
    'macd': 0.83,
    'macd_signal': 0.83,
    'bb_upper': 233.51,
    'bb_lower': 216.33,
    'bb_position': 0.57,
    'atr_14': 3.47,
    'volume_ratio': 0.98,
    'volume_rsi_signal': 0.05,
    'yield_volatility': -0.08,
    'macd_volume': 0.20,
    'bb_momentum': 0.07,
    'rate_return': 0.03,
    'rsi_bb_signal': 0.03,
    'momentum_3d': -1.17,
    'recent_win_rate': 0.50,
    'symbol_encoded': 0.0,  # SPY
    'day_of_week': 3.0,
    'day_of_month': 14.0,
    'month_of_year': 11.0,
    'quarter': 4.0,
    'day_of_year': 318.0,
    'price_direction': 1.0,
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
    225.99, 225.96, 227.49, 224.34, 222.50, 220.75, 218.90, 217.25,
    215.80, 214.50, 213.20, 212.00, 210.90, 209.75, 208.60, 207.50,
    206.40, 205.30, 204.20, 203.10, 202.00, 201.00, 200.00, 199.00,
    198.00, 197.00, 196.00, 195.00, 194.00, 193.00, 192.00, 191.00,
] + [190.0] * 68, dtype=np.float32)

volume_history = np.array([
    31335322.0, 30000000.0, 32000000.0, 31000000.0, 29000000.0,
    30500000.0, 31500000.0, 29500000.0, 30000000.0, 31000000.0,
] + [30000000.0] * 90, dtype=np.float32)

print("✅ Test features created")
print()

# Extract features using C++ (same as training)
print("[2/4] Extracting 85 features using C++ feature extractor...")
cpp_features = feature_extractor_cpp.extract_features_85(
    test_features['close'],
    test_features['open'],
    test_features['high'],
    test_features['low'],
    test_features['volume'],
    test_features['rsi_14'],
    test_features['macd'],
    test_features['macd_signal'],
    test_features['bb_upper'],
    test_features['bb_lower'],
    test_features['bb_position'],
    test_features['atr_14'],
    test_features['volume_ratio'],
    test_features['volume_rsi_signal'],
    test_features['yield_volatility'],
    test_features['macd_volume'],
    test_features['bb_momentum'],
    test_features['rate_return'],
    test_features['rsi_bb_signal'],
    test_features['momentum_3d'],
    test_features['recent_win_rate'],
    test_features['symbol_encoded'],
    test_features['day_of_week'],
    test_features['day_of_month'],
    test_features['month_of_year'],
    test_features['quarter'],
    test_features['day_of_year'],
    test_features['price_direction'],
    test_features['price_above_ma5'],
    test_features['price_above_ma20'],
    test_features['macd_signal_direction'],
    test_features['volume_trend'],
    test_features['year'],
    test_features['month'],
    test_features['day'],
    price_history,
    volume_history,
    test_features['fed_funds_rate'],
    test_features['treasury_10yr'],
)

print(f"✅ Extracted {len(cpp_features)} features")
print(f"   First 5: {cpp_features[:5]}")
print()

# Normalize features using scaler params from model
print("[3/4] Normalizing features using model scaler...")
scaler_mean = np.array(model_info['scaler_params']['mean'], dtype=np.float32)
scaler_std = np.array(model_info['scaler_params']['std'], dtype=np.float32)

normalized_features = (cpp_features - scaler_mean) / scaler_std
print(f"✅ Features normalized")
print(f"   Normalized first 5: {normalized_features[:5]}")
print()

# Load Python model and predict
print("[4/4] Running Python PyTorch inference...")

class PricePredictorCPP(torch.nn.Module):
    """Neural network for price prediction with 85 C++-extracted features"""
    def __init__(self, input_size=85):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)

# Load model weights
model = PricePredictorCPP(input_size=85)
model.load_state_dict(torch.load('models/price_predictor_cpp_85feat.pth', map_location='cpu'))
model.eval()

# Run inference
with torch.no_grad():
    input_tensor = torch.from_numpy(normalized_features).unsqueeze(0)
    output = model(input_tensor)
    predictions = output.numpy()[0]

print(f"✅ Python inference complete")
print()

# Display results
print("=" * 80)
print("PYTHON MODEL PREDICTIONS")
print("=" * 80)
print()
print(f"1-day price change:  {predictions[0]:+.6f}%")
print(f"5-day price change:  {predictions[1]:+.6f}%")
print(f"20-day price change: {predictions[2]:+.6f}%")
print()

# Save predictions for C++ comparison
output_data = {
    'features': normalized_features.tolist(),
    'python_predictions': predictions.tolist(),
    'model_info': model_info
}

with open('/tmp/python_predictions.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("✅ Python predictions saved to: /tmp/python_predictions.json")
print()
print("=" * 80)
print("NEXT STEP: Run C++ validation test")
print("=" * 80)
print()
print("The C++ test will:")
print("  1. Load the same features")
print("  2. Run INT32 SIMD inference")
print("  3. Compare with Python predictions")
print("  4. Verify parity (error < 1e-4)")
print()
print("Command:")
print("  ./build/tests/test_cpp_python_parity")
print()
