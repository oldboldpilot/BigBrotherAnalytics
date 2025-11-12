#!/usr/bin/env python3
"""
Export trained PyTorch price predictor model to ONNX format
for deployment in C++ trading engine

ONNX allows C++ inference via ONNX Runtime without Python dependency
"""

import sys
from pathlib import Path
import torch
import numpy as np

print("="*80)
print("EXPORTING PYTORCH MODEL TO ONNX FORMAT")
print("="*80)

# Check if model exists
model_path = Path('models/price_predictor_best.pth')
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("   Run: uv run python scripts/ml/train_price_predictor.py")
    sys.exit(1)

# Load checkpoint
print(f"\n📦 Loading checkpoint: {model_path}")
checkpoint = torch.load(model_path, weights_only=False)

# Get model architecture info
feature_cols = checkpoint['feature_cols']
input_size = len(feature_cols)
output_size = 3  # 1d, 5d, 20d predictions

print(f"   Input features: {input_size}")
print(f"   Output predictions: {output_size}")

# Recreate model architecture (same as training)
import torch.nn as nn

class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=3, dropout=0.3):
        super(PricePredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Create model and load weights
print(f"\n🧠 Recreating model architecture...")
model = PricePredictor(input_size=input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   Model loaded successfully")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dummy input for ONNX export
dummy_input = torch.randn(1, input_size)

# Export to ONNX
onnx_path = Path('models/price_predictor.onnx')
print(f"\n📤 Exporting to ONNX: {onnx_path}")

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=14,  # ONNX opset version
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ ONNX model exported: {onnx_path}")
print(f"   File size: {onnx_path.stat().st_size / 1024:.1f} KB")

# Verify ONNX model
print(f"\n🔍 Verifying ONNX model...")
import onnx
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print(f"✅ ONNX model is valid")

# Test ONNX inference
print(f"\n🧪 Testing ONNX inference...")
import onnxruntime as ort

session = ort.InferenceSession(str(onnx_path))
test_input = np.random.randn(1, input_size).astype(np.float32)
onnx_outputs = session.run(None, {'input': test_input})
print(f"✅ ONNX inference successful")
print(f"   Output shape: {onnx_outputs[0].shape}")
print(f"   Sample prediction: {onnx_outputs[0][0]}")

# Save feature column names for C++ reference
feature_list_path = Path('models/price_predictor_features.txt')
with open(feature_list_path, 'w') as f:
    for feature in feature_cols:
        f.write(f"{feature}\n")
print(f"\n📁 Feature list saved: {feature_list_path}")

# Create model info file for C++
info_path = Path('models/price_predictor_onnx_info.txt')
with open(info_path, 'w') as f:
    f.write(f"model_path: {onnx_path.name}\n")
    f.write(f"input_size: {input_size}\n")
    f.write(f"output_size: {output_size}\n")
    f.write(f"input_name: input\n")
    f.write(f"output_name: output\n")
    f.write(f"opset_version: 14\n")

print(f"📁 Model info saved: {info_path}")

print("\n" + "="*80)
print("✅ MODEL EXPORT COMPLETE!")
print("="*80)
print(f"\n📦 Files created:")
print(f"   1. {onnx_path} - ONNX model for C++ inference")
print(f"   2. {feature_list_path} - Feature column names")
print(f"   3. {info_path} - Model metadata")
print(f"\n🚀 NEXT STEPS:")
print(f"   1. C++ code can now load {onnx_path.name} using ONNX Runtime")
print(f"   2. Pass {input_size} features in same order as training")
print(f"   3. Get 3 outputs: [1d_change, 5d_change, 20d_change]")
print(f"   4. Integrate predictions into trading strategy")
print("="*80)
