#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format for C++ inference
Supports both standard (17 features) and custom (42 features) models
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Standard model architecture (17 features)
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

# Custom model architecture (42 features)
class CustomPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], output_size=3, dropout=0.3):
        super(CustomPricePredictor, self).__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            dropout_rate = dropout if i < 2 else dropout * 0.7
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Determine which model to use
use_custom = '--custom' in sys.argv or Path('models/custom_price_predictor_best.pth').exists()

if use_custom:
    print("="*80)
    print("Converting CUSTOM model (42 features) to ONNX")
    print("="*80)
    checkpoint_path = 'models/custom_price_predictor_best.pth'
else:
    print("="*80)
    print("Converting STANDARD model (17 features) to ONNX")
    print("="*80)
    checkpoint_path = 'models/price_predictor_best.pth'

# Load the trained model
print(f"\nLoading PyTorch model from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Get input size from feature_cols
input_size = len(checkpoint['feature_cols'])
print(f"Input features: {input_size}")

# Create model and load weights
if use_custom:
    print(f"Model type: CustomPricePredictor [256, 128, 64, 32]")
    model = CustomPricePredictor(input_size=input_size)
else:
    print(f"Model type: PricePredictor [128, 64, 32]")
    model = PricePredictor(input_size=input_size)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dummy input for ONNX export
dummy_input = torch.randn(1, input_size)

# Export to ONNX
print("\nExporting to ONNX...")
onnx_path = 'models/price_predictor.onnx'
print(f"Output path: {onnx_path}")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Model exported to {onnx_path}")

# Verify the model
import onnxruntime as ort
print("Verifying ONNX model...")
session = ort.InferenceSession(onnx_path)

# Test inference
test_input = dummy_input.numpy()
outputs = session.run(None, {'input': test_input})
print(f"ONNX output shape: {outputs[0].shape}")
print(f"ONNX output sample: {outputs[0][0]}")

# Compare with PyTorch
with torch.no_grad():
    pytorch_out = model(dummy_input).numpy()
print(f"PyTorch output: {pytorch_out[0]}")

# Check if they match
import numpy as np
if np.allclose(outputs[0], pytorch_out, rtol=1e-3, atol=1e-5):
    print("✅ ONNX model matches PyTorch model!")
else:
    print("⚠️  ONNX model differs from PyTorch model")
    print(f"Max difference: {np.max(np.abs(outputs[0] - pytorch_out))}")

print("\n✅ Conversion complete!")
print(f"C++ bot will load model from: {onnx_path}")
