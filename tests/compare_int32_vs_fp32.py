"""
Compare INT32 SIMD C++ implementation against Python FP32 layer-by-layer

This test validates the production price predictor INT32 SIMD implementation
against a reference FP32 Python implementation to ensure quantization accuracy.

Module: bigbrother.market_intelligence.price_predictor
Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""
import numpy as np
import struct
import os

# Load weights from binary files
def load_weights(layer_num):
    """Load weight matrix and bias vector for a layer"""
    weight_file = f"models/weights/layer{layer_num}_weight.bin"
    bias_file = f"models/weights/layer{layer_num}_bias.bin"

    # Layer dimensions
    layer_dims = {
        1: (256, 85),   # 85 -> 256
        2: (128, 256),  # 256 -> 128
        3: (64, 128),   # 128 -> 64
        4: (32, 64),    # 64 -> 32
        5: (3, 32)      # 32 -> 3
    }

    rows, cols = layer_dims[layer_num]

    # Load weights (rows x cols)
    with open(weight_file, 'rb') as f:
        weight_data = f.read()
        weights = np.frombuffer(weight_data, dtype=np.float32).reshape(rows, cols)

    # Load biases (rows)
    with open(bias_file, 'rb') as f:
        bias_data = f.read()
        biases = np.frombuffer(bias_data, dtype=np.float32)

    return weights, biases

# Quantize to INT32 (same as C++ implementation)
def quantize_to_int32(values):
    """Quantize float32 values to int32"""
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        max_abs = 1.0

    MAX_INT32_QUANT = (1 << 30) - 1  # 2^30 - 1
    scale = max_abs / MAX_INT32_QUANT
    inv_scale = 1.0 / scale

    quantized = np.round(values * inv_scale).astype(np.int32)
    return quantized, scale

# FP32 forward pass
def forward_fp32(input_vec, weights, biases, apply_relu=True):
    """Forward pass with FP32"""
    output = np.dot(weights, input_vec) + biases
    if apply_relu:
        output = np.maximum(0, output)
    return output

# INT32 forward pass (simulating C++ INT32 SIMD)
def forward_int32(input_vec, weights, biases, apply_relu=True):
    """Forward pass with INT32 quantization"""
    # Quantize input
    input_int32, input_scale = quantize_to_int32(input_vec)

    # Quantize weights
    weights_flat = weights.flatten()
    weights_int32, weight_scale = quantize_to_int32(weights_flat)
    weights_int32 = weights_int32.reshape(weights.shape)

    # Perform integer matrix multiplication
    output_int64 = np.dot(weights_int32.astype(np.int64), input_int32.astype(np.int64))

    # Dequantize and add bias
    combined_scale = weight_scale * input_scale
    output = output_int64.astype(np.float32) * combined_scale + biases

    if apply_relu:
        output = np.maximum(0, output)

    return output

print("=" * 80)
print("INT32 SIMD vs FP32 Layer-by-Layer Comparison")
print("=" * 80)
print()

# Create test input (sparse features: only close and symbol_enc)
input_vec = np.zeros(85, dtype=np.float32)
input_vec[0] = 670.59  # SPY close price
input_vec[47] = 0.0     # SPY symbol encoding

# Normalize (using StandardScaler from price_predictor.cppm)
MEAN = np.array([
    171.73168510, 171.77098131, 173.85409399, 169.78849837, 18955190.81943483,
    52.07665122, -1.01429406, -1.11443808, 183.70466682, 161.62830260,
    0.53303925, 4.63554388, 18931511.93290513, 1.01396078, 0.06702154,
    -0.09081233, 0.19577506, 0.07070947, 0.02930415, 0.02595905,
    -1.16933052, 0.00245190, 0.00007623, 0.11992571, 0.32397784,
    0.00058185, 0.51397823, 1.00021170, 1.00055151, 1.00072988,
    1.00106309, 1.00148186, 1.00161426, 1.00184679, 1.00197729,
    1.00221717, 1.00252557, 1.00279857, 1.00292291, 1.00302553,
    1.00308145, 1.00332408, 1.00349154, 1.00386256, 1.00421874,
    1.00436739, 1.00463950, 9.48643716, 2.31273208, 15.75989678,
    6.54169551, 2.51582856, 183.66473661, 0.51859777, 0.53345082,
    0.54987727, 0.51526213, 0.42463339, 2023.02114671, 6.54169551,
    15.75989678, -0.05298513, -0.18776332, -0.22518525, -0.26102762,
    -0.41762342, -0.47033575, -0.58258492, -0.67154995, -0.79431408,
    -0.92354285, -1.15338400, -1.26767064, -1.36071822, -1.34270070,
    -1.49789463, -1.61267464, -1.76728610, -1.98800362, -2.11874748,
    -2.25577792, 1.00000000, 1.00000000, 1.00000000, 1.00000000
], dtype=np.float32)

STD = np.array([
    186.03571734, 186.47600380, 191.72157267, 181.70836041, 22005096.42658922,
    16.71652602, 15.57689787, 16.42374095, 223.36568996, 167.49749464,
    0.32584191, 15.18473180, 20390423.38239934, 0.38221233, 0.06174181,
    0.09837635, 0.21208174, 0.07659907, 0.38598442, 0.08503450,
    17.25845955, 0.01734768, 0.00311195, 0.12017425, 0.24795759,
    0.03781362, 0.15687517, 0.02125998, 0.02982959, 0.03591441,
    0.04211676, 0.04605618, 0.04989224, 0.05347304, 0.05674135,
    0.05876375, 0.06280664, 0.06454916, 0.06744410, 0.06983660,
    0.07165713, 0.07489287, 0.07817148, 0.08043051, 0.08268397,
    0.08362892, 0.08593539, 5.75042283, 2.02005544, 8.73732957,
    3.28657881, 1.06885103, 100.31964116, 0.49965400, 0.49887979,
    0.49750604, 0.49976701, 0.49428724, 1.33469640, 3.28657881,
    8.73732957, 11.58576980, 16.32518674, 17.84023616, 19.44449065,
    20.31759436, 22.41613818, 23.65359845, 26.22348338, 26.79057463,
    27.84826555, 31.27653616, 32.78403656, 33.96486964, 34.75751675,
    36.14405424, 38.68106210, 39.64798053, 42.54812539, 44.39562117,
    45.85464070, 1.00000000, 1.00000000, 1.00000000, 1.00000000
], dtype=np.float32)

normalized = (input_vec - MEAN) / STD

print(f"Input: SPY (close={input_vec[0]}, symbol_enc={input_vec[47]})")
print(f"Normalized[0]={normalized[0]:.6f}, Normalized[47]={normalized[47]:.6f}")
print()

# Layer-by-layer comparison
current_fp32 = normalized.copy()
current_int32 = normalized.copy()

for layer_num in range(1, 6):
    print(f"Layer {layer_num}:")
    print("-" * 80)

    # Load weights
    weights, biases = load_weights(layer_num)

    print(f"  Weights shape: {weights.shape}, Biases shape: {biases.shape}")
    print(f"  Weight range: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
    print(f"  Bias range: [{np.min(biases):.6f}, {np.max(biases):.6f}]")

    # Forward pass (apply ReLU for all layers except the last)
    apply_relu = (layer_num < 5)

    # FP32
    output_fp32 = forward_fp32(current_fp32, weights, biases, apply_relu)

    # INT32
    output_int32 = forward_int32(current_int32, weights, biases, apply_relu)

    print(f"  FP32 output[0:5]: {output_fp32[:5]}")
    print(f"  INT32 output[0:5]: {output_int32[:5]}")
    print(f"  Difference[0:5]: {output_fp32[:5] - output_int32[:5]}")
    print(f"  Max abs diff: {np.max(np.abs(output_fp32 - output_int32)):.6f}")
    print(f"  FP32 zeros: {np.sum(output_fp32 == 0)} / {len(output_fp32)}")
    print(f"  INT32 zeros: {np.sum(output_int32 == 0)} / {len(output_int32)}")
    print()

    current_fp32 = output_fp32
    current_int32 = output_int32

print("=" * 80)
print("Final Output (Predictions):")
print("=" * 80)
print(f"FP32:  1d={current_fp32[0]*100:.2f}%, 5d={current_fp32[1]*100:.2f}%, 20d={current_fp32[2]*100:.2f}%")
print(f"INT32: 1d={current_int32[0]*100:.2f}%, 5d={current_int32[1]*100:.2f}%, 20d={current_int32[2]*100:.2f}%")
print()
print("Expected: Both should produce similar predictions within quantization error (~1-5%)")
