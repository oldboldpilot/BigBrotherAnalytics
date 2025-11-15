#!/usr/bin/env python3
"""
Test INT32 Quantization Impact on PyTorch Model

This script simulates what the C++ INT32 SIMD engine does:
1. Quantize weights to INT32
2. Quantize inputs to INT32
3. Perform INT32 matmul
4. Dequantize results

This will help us understand if the prediction mismatch is due to quantization.
"""

import torch
import numpy as np
import json

def quantize_to_int32(tensor):
    """Simulate C++ INT32 quantization"""
    MAX_INT32_QUANT = (1 << 30) - 1  # 2^30 - 1

    max_abs = torch.max(torch.abs(tensor)).item()
    scale = max_abs / MAX_INT32_QUANT

    if scale == 0.0:
        scale = 1.0

    inv_scale = 1.0 / scale
    quantized = torch.round(tensor * inv_scale).to(torch.int32)

    return quantized, scale

def dequantize_from_int32(quantized, scale):
    """Convert INT32 back to FP32"""
    return quantized.float() * scale

def int32_linear(input_fp32, weight_fp32, bias_fp32):
    """
    Simulate C++ INT32 SIMD linear layer:
    1. Quantize weights to INT32
    2. Quantize input to INT32
    3. Perform INT32 matmul
    4. Dequantize and add bias
    """
    # Quantize weights (out_features x in_features)
    weight_quantized, weight_scale = quantize_to_int32(weight_fp32)

    # Quantize input (batch_size x in_features)
    input_quantized, input_scale = quantize_to_int32(input_fp32)

    # INT32 matrix multiplication
    # output[i] = sum(weight[i,j] * input[j]) * weight_scale * input_scale + bias[i]
    output_int32 = torch.matmul(input_quantized.float(), weight_quantized.t().float())

    # Dequantize
    combined_scale = weight_scale * input_scale
    output = output_int32 * combined_scale + bias_fp32

    return output

print("=" * 80)
print("INT32 QUANTIZATION IMPACT TEST")
print("=" * 80)
print()

# Load model
print("[1/4] Loading PyTorch model...")
class PricePredictorCPP(torch.nn.Module):
    def __init__(self, input_size=85):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.network(x)

model = PricePredictorCPP(input_size=85)
model.load_state_dict(torch.load('models/price_predictor_cpp_85feat.pth', map_location='cpu'))
model.eval()
print("✅ Model loaded\n")

# Load test features
print("[2/4] Loading test features from Python predictions...")
with open('/tmp/python_predictions.json', 'r') as f:
    data = json.load(f)

normalized_features = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
python_predictions = data['python_predictions']

print(f"✅ Features loaded: shape {normalized_features.shape}\n")

# Run FP32 inference (original)
print("[3/4] Running FP32 inference...")
with torch.no_grad():
    fp32_output = model(normalized_features).numpy()[0]

print("FP32 predictions:")
print(f"  1-day:  {fp32_output[0]:+.6f}%")
print(f"  5-day:  {fp32_output[1]:+.6f}%")
print(f"  20-day: {fp32_output[2]:+.6f}%")
print()

# Run INT32 quantized inference (simulating C++)
print("[4/4] Running INT32 quantized inference (simulating C++)...")

# Extract weights and biases
# Layer indices: 0=Linear1, 3=Linear2, 6=Linear3, 8=Linear4, 10=Linear5
layers = [
    (model.network[0].weight.data, model.network[0].bias.data),   # Layer 1: 85->256
    (model.network[3].weight.data, model.network[3].bias.data),   # Layer 2: 256->128
    (model.network[6].weight.data, model.network[6].bias.data),   # Layer 3: 128->64
    (model.network[8].weight.data, model.network[8].bias.data),   # Layer 4: 64->32
    (model.network[10].weight.data, model.network[10].bias.data), # Layer 5: 32->3
]

with torch.no_grad():
    current = normalized_features

    for i, (weight, bias) in enumerate(layers):
        # INT32 linear layer
        current = int32_linear(current, weight, bias)

        # Apply ReLU (except last layer)
        if i < len(layers) - 1:
            current = torch.relu(current)

        print(f"  After layer {i+1}: output[0:3] = {current[0,:3].numpy()}")

    int32_output = current.numpy()[0]

print()
print("INT32 quantized predictions:")
print(f"  1-day:  {int32_output[0]:+.6f}%")
print(f"  5-day:  {int32_output[1]:+.6f}%")
print(f"  20-day: {int32_output[2]:+.6f}%")
print()

# Compare
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()
print(f"                  FP32           INT32          Error")
print(f"  1-day:    {fp32_output[0]:+12.6f}%   {int32_output[0]:+12.6f}%   {abs(fp32_output[0] - int32_output[0]):12.6f}%")
print(f"  5-day:    {fp32_output[1]:+12.6f}%   {int32_output[1]:+12.6f}%   {abs(fp32_output[1] - int32_output[1]):12.6f}%")
print(f"  20-day:   {fp32_output[2]:+12.6f}%   {int32_output[2]:+12.6f}%   {abs(fp32_output[2] - int32_output[2]):12.6f}%")
print()

max_error = max(abs(fp32_output[i] - int32_output[i]) for i in range(3))
print(f"Max error: {max_error:.6f}%")
print()

if max_error < 0.0001:
    print("✅ INT32 quantization has minimal impact (< 0.0001%)")
elif max_error < 0.01:
    print("⚠️  INT32 quantization has small impact (< 0.01%)")
elif max_error < 0.1:
    print("⚠️  INT32 quantization has moderate impact (< 0.1%)")
else:
    print("❌ INT32 quantization has LARGE impact (>= 0.1%)")
    print()
    print("This explains the C++ parity mismatch!")
    print("The model was trained with FP32, but C++ uses INT32 quantized inference.")
    print()
    print("Solutions:")
    print("  1. Use FP32 C++ inference for parity validation")
    print("  2. Use quantization-aware training (QAT)")
    print("  3. Accept quantization error and update threshold")
print()
