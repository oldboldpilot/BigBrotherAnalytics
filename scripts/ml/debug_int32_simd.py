#!/usr/bin/env python3
"""
Debug INT32 SIMD constant output bug by comparing Python vs C++ outputs
"""

import torch
import numpy as np
from pathlib import Path

# Load the trained model
model_path = Path("models/price_predictor_85feat.pth")
model = torch.load(model_path, map_location='cpu')
model.eval()

# Load scaler
import pickle
scaler_path = Path("models/scaler_85feat.pkl")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Test features for SPY, QQQ, IWM (from earlier logs)
test_cases = {
    "SPY": {
        "close": 670.59,
        "rsi": 40.19,
        "symbol_enc": 0.0,
        "price_lag1": 670.59,
        "price_diff1": -1.45,
        "autocorr1": 0.9268
    },
    "QQQ": {
        "close": 605.78,
        "rsi": 34.96,
        "symbol_enc": 1.0,
        "price_lag1": 605.78,
        "price_diff1": -2.62,
        "autocorr1": 0.9051
    },
    "IWM": {
        "close": 236.45,
        "rsi": 33.78,
        "symbol_enc": 2.0,
        "price_lag1": 236.45,
        "price_diff1": -0.34,
        "autocorr1": 0.6250
    }
}

print("=" * 80)
print("DEBUG INT32 SIMD BUG")
print("=" * 80)
print()

print("Testing Python FP32 model with same features from C++ logs:")
print()

for symbol, features_subset in test_cases.items():
    # Create a dummy 85-feature vector (we only have 6 features from logs)
    # We need to fill in the rest - let's use zeros for now
    features = np.zeros(85, dtype=np.float32)

    # Fill in known features (index positions from C++ code)
    features[0] = features_subset["close"]  # close
    features[47] = features_subset["symbol_enc"]  # symbol_enc
    features[8] = features_subset["price_lag1"]  # price_lag1
    features[61] = features_subset["price_diff1"]  # price_diff1
    features[82] = features_subset["autocorr1"]  # autocorr1 (assuming position)

    # Normalize
    features_normalized = scaler.transform([features])[0]

    # Predict
    with torch.no_grad():
        output = model(torch.from_numpy(features_normalized).unsqueeze(0))
        predictions = output.squeeze().numpy()

    print(f"{symbol}:")
    print(f"  Features (first 6 only): close={features_subset['close']:.2f}, rsi={features_subset['rsi']:.2f}, symbol_enc={features_subset['symbol_enc']}, price_lag1={features_subset['price_lag1']:.2f}")
    print(f"  Python FP32: 1d={predictions[0]*100:.2f}%, 5d={predictions[1]*100:.2f}%, 20d={predictions[2]*100:.2f}%")
    print(f"  C++ INT32:   1d=-2.21%, 5d=-5.30%, 20d=-3.13% (CONSTANT BUG)")
    print()

print("=" * 80)
print("HYPOTHESIS:")
print("=" * 80)
print("The Python model produces DIFFERENT outputs for different symbol_enc values,")
print("but the C++ INT32 SIMD produces CONSTANT outputs.")
print()
print("Possible causes:")
print("1. Weight loading issue (weights not being read correctly)")
print("2. Quantization scale mismatch")
print("3. Matrix multiplication bug (ignoring inputs)")
print("4. Layer output not being passed to next layer")
print()
print("Next step: Add debug logging to C++ to print:")
print("  - First layer weights (sample)")
print("  - Quantized input values")
print("  - Output of first layer before ReLU")
print("  - Output of first layer after ReLU")
