#!/usr/bin/env python3
"""Display model input features"""

import torch

# Load model checkpoint
checkpoint = torch.load('models/price_predictor_best.pth', map_location='cpu', weights_only=False)

# Display features
print("="*60)
print("MODEL INPUT FEATURES")
print("="*60)
print()

for i, feat in enumerate(checkpoint['feature_cols'], 1):
    print(f"{i:2d}. {feat}")

print()
print("="*60)
print(f"Total: {len(checkpoint['feature_cols'])} features")
print("="*60)
