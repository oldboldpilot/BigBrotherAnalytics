#!/usr/bin/env python3
"""
Offline Weight Quantization for BigBrother ML Models

Pre-quantizes FP32 neural network weights to INT8 and INT16 formats,
saving them to binary files for direct loading without runtime conversion.

This eliminates runtime quantization overhead and improves inference performance.

Usage:
    python scripts/ml/quantize_weights_offline.py

Output files:
    models/weights/price_predictor_int8.bin
    models/weights/price_predictor_int16.bin
"""

import struct
import numpy as np
import torch
from pathlib import Path


def compute_quantization_params_int8(data: np.ndarray) -> tuple[float, float]:
    """
    Compute symmetric quantization parameters for INT8.

    Maps FP32 range [-max_abs, +max_abs] to INT8 [-127, +127]

    Returns:
        (scale, inv_scale) where scale = max_abs / 127.0
    """
    max_abs = np.abs(data).max()
    if max_abs < 1e-8:
        max_abs = 1.0

    scale = max_abs / 127.0
    inv_scale = 127.0 / max_abs

    return scale, inv_scale


def compute_quantization_params_int16(data: np.ndarray) -> tuple[float, float]:
    """
    Compute symmetric quantization parameters for INT16.

    Maps FP32 range [-max_abs, +max_abs] to INT16 [-32767, +32767]

    Returns:
        (scale, inv_scale) where scale = max_abs / 32767.0
    """
    max_abs = np.abs(data).max()
    if max_abs < 1e-8:
        max_abs = 1.0

    scale = max_abs / 32767.0
    inv_scale = 32767.0 / max_abs

    return scale, inv_scale


def quantize_int8(data: np.ndarray, inv_scale: float) -> np.ndarray:
    """Quantize FP32 data to INT8."""
    quantized = np.round(data * inv_scale)
    quantized = np.clip(quantized, -127, 127)
    return quantized.astype(np.int8)


def quantize_int16(data: np.ndarray, inv_scale: float) -> np.ndarray:
    """Quantize FP32 data to INT16."""
    quantized = np.round(data * inv_scale)
    quantized = np.clip(quantized, -32767, 32767)
    return quantized.astype(np.int16)


def save_quantized_weights_int8(
    model_path: Path,
    output_path: Path
) -> None:
    """
    Load FP32 PyTorch model and save INT8 quantized weights.

    File format:
        [uint32] magic number (0x51494E54 = 'QINT')
        [uint32] version (1)
        [uint32] precision (8 for INT8)
        [uint32] num_layers

        For each layer:
            [uint32] weight_rows
            [uint32] weight_cols
            [float32] weight_scale
            [int8 × rows × cols] quantized weights
            [uint32] bias_size
            [float32 × bias_size] biases (kept as FP32)
    """
    print(f"Loading FP32 model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Extract layers (network.0, network.2, network.4, network.6, network.8)
    layers = []
    layer_idx = 0
    while True:
        weight_key = f'network.{layer_idx}.weight'
        bias_key = f'network.{layer_idx}.bias'

        if weight_key not in state_dict:
            break

        weight = state_dict[weight_key].numpy()
        bias = state_dict[bias_key].numpy()

        layers.append((weight, bias))
        layer_idx += 2  # Skip odd indices (ReLU layers)

    print(f"Found {len(layers)} layers")

    # Quantize and save
    print(f"Quantizing to INT8 and saving to {output_path}...")

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('I', 0x51494E54))  # Magic: 'QINT'
        f.write(struct.pack('I', 1))           # Version
        f.write(struct.pack('I', 8))           # Precision: INT8
        f.write(struct.pack('I', len(layers))) # Num layers

        total_fp32_bytes = 0
        total_int8_bytes = 0

        for idx, (weight, bias) in enumerate(layers):
            rows, cols = weight.shape

            # Quantize weights
            scale, inv_scale = compute_quantization_params_int8(weight)
            weight_int8 = quantize_int8(weight, inv_scale)

            # Write layer data
            f.write(struct.pack('I', rows))
            f.write(struct.pack('I', cols))
            f.write(struct.pack('f', scale))
            f.write(weight_int8.tobytes())

            # Biases stay as FP32
            f.write(struct.pack('I', len(bias)))
            f.write(bias.astype(np.float32).tobytes())

            # Statistics
            fp32_bytes = weight.nbytes + bias.nbytes
            int8_bytes = weight_int8.nbytes + bias.nbytes
            total_fp32_bytes += fp32_bytes
            total_int8_bytes += int8_bytes

            print(f"  Layer {idx+1}: {rows}×{cols} weights, scale={scale:.6f}")

    # Summary
    savings_pct = 100 * (1 - total_int8_bytes / total_fp32_bytes)
    print(f"\nINT8 Quantization Summary:")
    print(f"  FP32 size: {total_fp32_bytes / 1024:.1f} KB")
    print(f"  INT8 size: {total_int8_bytes / 1024:.1f} KB")
    print(f"  Savings: {savings_pct:.1f}%")


def save_quantized_weights_int16(
    model_path: Path,
    output_path: Path
) -> None:
    """
    Load FP32 PyTorch model and save INT16 quantized weights.

    File format: Same as INT8 but with precision=16
    """
    print(f"Loading FP32 model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Extract layers (network.0, network.2, network.4, network.6, network.8)
    layers = []
    layer_idx = 0
    while True:
        weight_key = f'network.{layer_idx}.weight'
        bias_key = f'network.{layer_idx}.bias'

        if weight_key not in state_dict:
            break

        weight = state_dict[weight_key].numpy()
        bias = state_dict[bias_key].numpy()

        layers.append((weight, bias))
        layer_idx += 2  # Skip odd indices (ReLU layers)

    print(f"Found {len(layers)} layers")

    # Quantize and save
    print(f"Quantizing to INT16 and saving to {output_path}...")

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('I', 0x51494E54))  # Magic: 'QINT'
        f.write(struct.pack('I', 1))           # Version
        f.write(struct.pack('I', 16))          # Precision: INT16
        f.write(struct.pack('I', len(layers))) # Num layers

        total_fp32_bytes = 0
        total_int16_bytes = 0

        for idx, (weight, bias) in enumerate(layers):
            rows, cols = weight.shape

            # Quantize weights
            scale, inv_scale = compute_quantization_params_int16(weight)
            weight_int16 = quantize_int16(weight, inv_scale)

            # Write layer data
            f.write(struct.pack('I', rows))
            f.write(struct.pack('I', cols))
            f.write(struct.pack('f', scale))
            f.write(weight_int16.tobytes())

            # Biases stay as FP32
            f.write(struct.pack('I', len(bias)))
            f.write(bias.astype(np.float32).tobytes())

            # Statistics
            fp32_bytes = weight.nbytes + bias.nbytes
            int16_bytes = weight_int16.nbytes + bias.nbytes
            total_fp32_bytes += fp32_bytes
            total_int16_bytes += int16_bytes

            print(f"  Layer {idx+1}: {rows}×{cols} weights, scale={scale:.6f}")

    # Summary
    savings_pct = 100 * (1 - total_int16_bytes / total_fp32_bytes)
    print(f"\nINT16 Quantization Summary:")
    print(f"  FP32 size: {total_fp32_bytes / 1024:.1f} KB")
    print(f"  INT16 size: {total_int16_bytes / 1024:.1f} KB")
    print(f"  Savings: {savings_pct:.1f}%")


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "models" / "price_predictor_85feat_best.pth"
    output_dir = base_dir / "models" / "weights"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using scripts/ml/train_price_predictor.py")
        return 1

    # Quantize to INT8
    int8_output = output_dir / "price_predictor_int8.bin"
    save_quantized_weights_int8(model_path, int8_output)
    print(f"✓ Saved INT8 weights to {int8_output}\n")

    # Quantize to INT16
    int16_output = output_dir / "price_predictor_int16.bin"
    save_quantized_weights_int16(model_path, int16_output)
    print(f"✓ Saved INT16 weights to {int16_output}\n")

    print("=" * 60)
    print("Pre-quantization complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Build the updated neural network engines:")
    print("   SKIP_CLANG_TIDY=1 cmake -G Ninja -B build && ninja -C build")
    print("\n2. Run benchmarks to see improved performance:")
    print("   ./build/bin/benchmark_int8_quantization")
    print("\nExpected improvements:")
    print("  - Eliminates runtime quantization overhead")
    print("  - Faster model initialization")
    print("  - Better inference performance")

    return 0


if __name__ == '__main__':
    exit(main())
