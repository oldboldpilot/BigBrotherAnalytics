#!/usr/bin/env python3
"""
Export PyTorch Model Weights to Binary Format

Converts trained PyTorch model to binary weight files for C++ SIMD inference.

Architecture: 80 → 256 → 128 → 64 → 32 → 3

Output format (little-endian float32):
  - layer1_weight.bin: [80 x 256]
  - layer1_bias.bin: [256]
  - layer2_weight.bin: [256 x 128]
  - layer2_bias.bin: [128]
  - layer3_weight.bin: [128 x 64]
  - layer3_bias.bin: [64]
  - layer4_weight.bin: [64 x 32]
  - layer4_bias.bin: [32]
  - layer5_weight.bin: [32 x 3]
  - layer5_bias.bin: [3]

Usage:
    python scripts/ml/export_weights_to_binary.py
"""

import sys
from pathlib import Path
import numpy as np

print("="*80)
print("EXPORT PYTORCH WEIGHTS TO BINARY FORMAT")
print("="*80)
print()

# Check if PyTorch is available
try:
    import torch
except ImportError:
    print("❌ PyTorch not found!")
    print("   Install with: pip install torch")
    sys.exit(1)

# Load trained model
model_path = Path('models/custom_price_predictor_best.pth')

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("   Train model first: uv run python scripts/ml/train_custom_price_predictor.py")
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

# Expected architecture: 80 → 256 → 128 → 64 → 32 → 3
expected_layers = [
    ('network.0.weight', (256, 80)),   # Layer 1 weight
    ('network.0.bias', (256,)),         # Layer 1 bias
    ('network.3.weight', (128, 256)),   # Layer 2 weight (after ReLU + Dropout)
    ('network.3.bias', (128,)),
    ('network.6.weight', (64, 128)),    # Layer 3 weight
    ('network.6.bias', (64,)),
    ('network.9.weight', (32, 64)),     # Layer 4 weight
    ('network.9.bias', (32,)),
    ('network.12.weight', (3, 32)),     # Layer 5 weight (output)
    ('network.12.bias', (3,)),
]

# Verify architecture
print("✅ Verifying architecture...")
for key, expected_shape in expected_layers:
    if key not in state_dict:
        print(f"❌ Missing layer: {key}")
        sys.exit(1)

    actual_shape = tuple(state_dict[key].shape)
    if actual_shape != expected_shape:
        print(f"❌ Shape mismatch for {key}")
        print(f"   Expected: {expected_shape}")
        print(f"   Actual: {actual_shape}")
        sys.exit(1)

print("   Architecture matches: 80 → 256 → 128 → 64 → 32 → 3")
print()

# Create output directory
output_dir = Path('models/weights')
output_dir.mkdir(exist_ok=True)

print(f"💾 Exporting weights to: {output_dir}")
print()

# Export layer by layer
def export_layer(layer_num, weight_key, bias_key, state_dict, output_dir):
    """Export single layer weights and biases to binary files"""

    # Extract tensors
    weight = state_dict[weight_key].numpy()  # [out, in]
    bias = state_dict[bias_key].numpy()      # [out]

    # Save as little-endian float32
    weight_path = output_dir / f'layer{layer_num}_weight.bin'
    bias_path = output_dir / f'layer{layer_num}_bias.bin'

    weight.astype(np.float32).tofile(weight_path)
    bias.astype(np.float32).tofile(bias_path)

    print(f"   Layer {layer_num}: {weight.shape[1]} → {weight.shape[0]}")
    print(f"      Weight: {weight_path.name} ({weight.nbytes:,} bytes)")
    print(f"      Bias: {bias_path.name} ({bias.nbytes:,} bytes)")

    return weight.nbytes + bias.nbytes

# Export all layers
total_bytes = 0

total_bytes += export_layer(1, 'network.0.weight', 'network.0.bias', state_dict, output_dir)
total_bytes += export_layer(2, 'network.3.weight', 'network.3.bias', state_dict, output_dir)
total_bytes += export_layer(3, 'network.6.weight', 'network.6.bias', state_dict, output_dir)
total_bytes += export_layer(4, 'network.9.weight', 'network.9.bias', state_dict, output_dir)
total_bytes += export_layer(5, 'network.12.weight', 'network.12.bias', state_dict, output_dir)

print()
print(f"✅ Export complete!")
print(f"   Total size: {total_bytes:,} bytes ({total_bytes / 1024:.2f} KB)")
print()

# Verify binary files can be loaded
print("🔍 Verifying binary files...")

def verify_layer(layer_num, in_size, out_size, output_dir):
    """Verify binary files can be read correctly"""

    weight_path = output_dir / f'layer{layer_num}_weight.bin'
    bias_path = output_dir / f'layer{layer_num}_bias.bin'

    # Read binary files
    weight = np.fromfile(weight_path, dtype=np.float32)
    bias = np.fromfile(bias_path, dtype=np.float32)

    # Verify shapes
    expected_weight_size = in_size * out_size
    expected_bias_size = out_size

    if len(weight) != expected_weight_size:
        print(f"❌ Layer {layer_num} weight size mismatch")
        print(f"   Expected: {expected_weight_size}, Got: {len(weight)}")
        return False

    if len(bias) != expected_bias_size:
        print(f"❌ Layer {layer_num} bias size mismatch")
        print(f"   Expected: {expected_bias_size}, Got: {len(bias)}")
        return False

    return True

if verify_layer(1, 80, 256, output_dir) and \
   verify_layer(2, 256, 128, output_dir) and \
   verify_layer(3, 128, 64, output_dir) and \
   verify_layer(4, 64, 32, output_dir) and \
   verify_layer(5, 32, 3, output_dir):
    print("   ✅ All binary files verified")
else:
    print("   ❌ Verification failed")
    sys.exit(1)

print()
print("="*80)
print("SUCCESS! Weights exported for C++ SIMD inference")
print("="*80)
print()
print("Usage in C++:")
print()
print("  import bigbrother.ml.neural_net_simd;")
print()
print("  auto net = NeuralNet::create()")
print("                .loadWeights(\"models/weights/\");")
print()
print("  std::array<float, 80> input = { /* normalized features */ };")
print("  auto output = net.predict(input);")
print()
print(f"  // output[0] = 1-day prediction")
print(f"  // output[1] = 5-day prediction")
print(f"  // output[2] = 20-day prediction")
print()
