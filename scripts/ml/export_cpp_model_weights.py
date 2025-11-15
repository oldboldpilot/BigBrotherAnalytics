#!/usr/bin/env python3
"""
Export C++ 85-Feature Model Weights to Binary Format

Converts the trained PyTorch price_predictor_cpp_85feat.pth model to binary
weight files for C++ INT32 SIMD inference.

Architecture: 85 → 256 → 128 → 64 → 32 → 3

Output format (little-endian float32):
  - layer1_weight.bin: [85 x 256]
  - layer1_bias.bin: [256]
  - layer2_weight.bin: [256 x 128]
  - layer2_bias.bin: [128]
  - layer3_weight.bin: [128 x 64]
  - layer3_bias.bin: [64]
  - layer4_weight.bin: [64 x 32]
  - layer4_bias.bin: [32]
  - layer5_weight.bin: [32 x 3]
  - layer5_bias.bin: [3]

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""

import sys
from pathlib import Path
import numpy as np
import torch

print("=" * 80)
print("EXPORT C++ 85-FEATURE MODEL WEIGHTS TO BINARY FORMAT")
print("=" * 80)
print()

# Load trained model
model_path = Path('models/price_predictor_cpp_85feat.pth')

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("   Train model first: uv run python scripts/ml/train_price_predictor_cpp.py")
    sys.exit(1)

print(f"📦 Loading model: {model_path}")

# Load with weights_only for security
try:
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
except TypeError:
    # Fallback for older PyTorch versions
    state_dict = torch.load(model_path, map_location='cpu')

print(f"   Model loaded successfully")
print()

# Inspect model structure
print("🔍 Model structure:")
for key in state_dict.keys():
    shape = state_dict[key].shape
    print(f"   {key}: {list(shape)}")
print()

# Expected architecture: 85 → 256 → 128 → 64 → 32 → 3
# Indices: 0=Linear1, 1=ReLU, 2=Dropout, 3=Linear2, 4=ReLU, 5=Dropout,
#          6=Linear3, 7=ReLU, 8=Linear4, 9=ReLU, 10=Linear5
expected_layers = [
    ('network.0.weight', (256, 85)),   # Layer 1 weight
    ('network.0.bias', (256,)),         # Layer 1 bias
    ('network.3.weight', (128, 256)),   # Layer 2 weight (after ReLU + Dropout at 1,2)
    ('network.3.bias', (128,)),
    ('network.6.weight', (64, 128)),    # Layer 3 weight (after ReLU + Dropout at 4,5)
    ('network.6.bias', (64,)),
    ('network.8.weight', (32, 64)),     # Layer 4 weight (after ReLU at 7)
    ('network.8.bias', (32,)),
    ('network.10.weight', (3, 32)),     # Layer 5 weight (after ReLU at 9)
    ('network.10.bias', (3,)),
]

# Validate architecture
print("✅ Validating architecture...")
for expected_key, expected_shape in expected_layers:
    if expected_key not in state_dict:
        print(f"❌ Missing layer: {expected_key}")
        sys.exit(1)
    actual_shape = tuple(state_dict[expected_key].shape)
    if actual_shape != expected_shape:
        print(f"❌ Shape mismatch for {expected_key}:")
        print(f"   Expected: {expected_shape}")
        print(f"   Actual:   {actual_shape}")
        sys.exit(1)

print("   Architecture validated ✓")
print()

# Create output directory
output_dir = Path('models/weights')
output_dir.mkdir(exist_ok=True)

print(f"📝 Exporting weights to: {output_dir}")
print()

# Export each layer
layer_mapping = [
    ('network.0.weight', 'layer1_weight.bin', (256, 85)),
    ('network.0.bias', 'layer1_bias.bin', (256,)),
    ('network.3.weight', 'layer2_weight.bin', (128, 256)),
    ('network.3.bias', 'layer2_bias.bin', (128,)),
    ('network.6.weight', 'layer3_weight.bin', (64, 128)),
    ('network.6.bias', 'layer3_bias.bin', (64,)),
    ('network.8.weight', 'layer4_weight.bin', (32, 64)),
    ('network.8.bias', 'layer4_bias.bin', (32,)),
    ('network.10.weight', 'layer5_weight.bin', (3, 32)),
    ('network.10.bias', 'layer5_bias.bin', (3,)),
]

total_params = 0

for pytorch_key, filename, expected_shape in layer_mapping:
    tensor = state_dict[pytorch_key]

    # Convert to numpy (float32, little-endian)
    weights = tensor.cpu().numpy().astype(np.float32)

    # Verify shape
    if weights.shape != expected_shape:
        print(f"❌ Unexpected shape for {pytorch_key}: {weights.shape} vs {expected_shape}")
        sys.exit(1)

    # Write binary file
    output_path = output_dir / filename
    weights.tofile(output_path)

    # Calculate size
    param_count = np.prod(weights.shape)
    file_size_kb = output_path.stat().st_size / 1024

    print(f"   ✅ {filename}")
    print(f"      Shape: {list(weights.shape)}")
    print(f"      Parameters: {param_count:,}")
    print(f"      File size: {file_size_kb:.2f} KB")

    total_params += param_count

print()
print("=" * 80)
print("EXPORT SUMMARY")
print("=" * 80)
print()
print(f"Total parameters: {total_params:,}")
print(f"Output directory: {output_dir.absolute()}")
print()
print("Files created:")
for _, filename, _ in layer_mapping:
    print(f"  - {filename}")
print()
print("✅ Weights exported successfully!")
print()
print("Next steps:")
print("  1. Build C++ test: cmake --build build")
print("  2. Run parity test: ./build/tests/test_cpp_python_parity")
print()
