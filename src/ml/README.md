# Machine Learning Library - SIMD Neural Network

High-performance CPU-based ML inference using pure C++ with SIMD intrinsics.

## Quick Start

### 1. Export Weights from PyTorch

```bash
# Train model (if not already trained)
uv run python scripts/ml/train_custom_price_predictor.py

# Export weights to binary format
python scripts/ml/export_weights_to_binary.py
```

This creates `models/weights/layer{1-5}_{weight,bias}.bin`

### 2. Build and Test

```bash
# Build
cmake --build build --target test_neural_net_simd

# Run test
./build/bin/test_neural_net_simd
```

### 3. Use in Code

```cpp
import bigbrother.ml.neural_net_simd;

using namespace bigbrother::ml;

// Create and load weights
auto net = NeuralNet::create()
              .loadWeights("models/weights/");

// Prepare input (80 normalized features)
std::array<float, 80> input = { /* ... */ };

// Run inference
auto output = net.predict(input);

// Results
float day_1_change = output[0];   // 1-day price change %
float day_5_change = output[1];   // 5-day price change %
float day_20_change = output[2];  // 20-day price change %
```

## Files

- **neural_net_simd.cppm** - SIMD-optimized neural network (800 lines)
- **neural_net_mkl.cppm** - Intel MKL version (reference)
- **README.md** - This file

## Performance

| Implementation | Inference Time | Dependencies |
|---------------|----------------|--------------|
| SIMD (AVX-512) | ~0.05 ms | None |
| SIMD (AVX-2) | ~0.08 ms | None |
| SIMD (SSE) | ~0.15 ms | None |
| MKL | ~0.10 ms | Intel MKL |
| ONNX Runtime | ~0.20 ms | ONNX Runtime |

## Architecture

```
80 features → 256 → 128 → 64 → 32 → 3 predictions
```

See [SIMD_NEURAL_NETWORK.md](../../docs/SIMD_NEURAL_NETWORK.md) for detailed documentation.

## CPU Detection

Automatic fallback:
1. Check AVX-512 → Use if available
2. Check AVX-2 → Use if available
3. Fallback to SSE → Always available on x86-64

Single binary runs optimally on all CPUs.
