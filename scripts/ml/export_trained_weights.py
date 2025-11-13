#!/usr/bin/env python3
"""
Export trained model weights to binary format for C++ inference

Usage:
    uv run python scripts/ml/export_trained_weights.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json

print("=" * 80)
print("EXPORTING TRAINED WEIGHTS TO BINARY FORMAT")
print("=" * 80)

# Define model architecture (must match training script)
class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(60, 256),   # network.0
            nn.ReLU(),            # network.1
            nn.Linear(256, 128),  # network.2
            nn.ReLU(),            # network.3
            nn.Linear(128, 64),   # network.4
            nn.ReLU(),            # network.5
            nn.Linear(64, 32),    # network.6
            nn.ReLU(),            # network.7
            nn.Linear(32, 3)      # network.8
        )

    def forward(self, x):
        return self.network(x)

# Load trained model
model_path = Path('models/price_predictor_60feat_best.pth')
if not model_path.exists():
    print(f"ERROR: Model not found: {model_path}")
    sys.exit(1)

print(f"\nLoading model from: {model_path}")
checkpoint = torch.load(model_path, weights_only=False)

model = PricePredictor()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded successfully")
print(f"  Epoch: {checkpoint['epoch'] + 1}")
print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

# Get state dict
state_dict = model.state_dict()

print(f"\nState dict keys:")
for key in sorted(state_dict.keys()):
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")

# Layer mapping for C++ compatibility
# PyTorch indices: 0, 2, 4, 6, 8 (Linear layers in Sequential)
# C++ indices: 0, 3, 6, 9, 12 (matching original architecture)
layer_mapping = {
    'network.0': 'network_0',   # 60 -> 256
    'network.2': 'network_3',   # 256 -> 128
    'network.4': 'network_6',   # 128 -> 64
    'network.6': 'network_9',   # 64 -> 32
    'network.8': 'network_12'   # 32 -> 3
}

# Create weights directory
weights_dir = Path('models/weights')
weights_dir.mkdir(exist_ok=True)

print(f"\n" + "=" * 80)
print("EXPORTING WEIGHTS")
print("=" * 80)

exported_files = []

for pytorch_layer, cpp_layer in layer_mapping.items():
    weight_key = f'{pytorch_layer}.weight'
    bias_key = f'{pytorch_layer}.bias'

    # Get weight and bias tensors
    weight_data = state_dict[weight_key].cpu().numpy().astype(np.float32)
    bias_data = state_dict[bias_key].cpu().numpy().astype(np.float32)

    # Save as binary files
    weight_file = weights_dir / f'{cpp_layer}_weight.bin'
    bias_file = weights_dir / f'{cpp_layer}_bias.bin'

    weight_data.tofile(weight_file)
    bias_data.tofile(bias_file)

    exported_files.append(str(weight_file))
    exported_files.append(str(bias_file))

    print(f"\n{cpp_layer}:")
    print(f"  Weight: {weight_data.shape} -> {weight_file}")
    print(f"  Bias: {bias_data.shape} -> {bias_file}")
    print(f"  Weight range: [{weight_data.min():.6f}, {weight_data.max():.6f}]")
    print(f"  Bias range: [{bias_data.min():.6f}, {bias_data.max():.6f}]")

print(f"\n" + "=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)

print(f"\nExported {len(exported_files)} files:")
for f in sorted(exported_files):
    file_size = Path(f).stat().st_size
    print(f"  {f} ({file_size:,} bytes)")

total_size = sum(Path(f).stat().st_size for f in exported_files)
print(f"\nTotal size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")

# Verify files match expected format
print(f"\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

# Expected layer sizes
expected_sizes = {
    'network_0': (256, 60),   # weight shape
    'network_3': (128, 256),
    'network_6': (64, 128),
    'network_9': (32, 64),
    'network_12': (3, 32)
}

all_verified = True
for cpp_layer, expected_shape in expected_sizes.items():
    weight_file = weights_dir / f'{cpp_layer}_weight.bin'
    bias_file = weights_dir / f'{cpp_layer}_bias.bin'

    # Read back and verify
    weight_loaded = np.fromfile(weight_file, dtype=np.float32)
    bias_loaded = np.fromfile(bias_file, dtype=np.float32)

    expected_weight_size = expected_shape[0] * expected_shape[1]
    expected_bias_size = expected_shape[0]

    weight_ok = len(weight_loaded) == expected_weight_size
    bias_ok = len(bias_loaded) == expected_bias_size

    if weight_ok and bias_ok:
        print(f"{cpp_layer}: OK")
        print(f"  Weight: {len(weight_loaded)} elements (expected {expected_weight_size})")
        print(f"  Bias: {len(bias_loaded)} elements (expected {expected_bias_size})")
    else:
        print(f"{cpp_layer}: FAILED")
        if not weight_ok:
            print(f"  Weight: {len(weight_loaded)} != {expected_weight_size}")
        if not bias_ok:
            print(f"  Bias: {len(bias_loaded)} != {expected_bias_size}")
        all_verified = False

if all_verified:
    print(f"\nAll weight files verified successfully")
    print(f"Ready for C++ integration")
else:
    print(f"\nERROR: Some weight files failed verification")
    sys.exit(1)

print("\n" + "=" * 80)
