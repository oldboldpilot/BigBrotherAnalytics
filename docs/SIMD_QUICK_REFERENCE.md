# SIMD Neural Network - Quick Reference

## One-Minute Setup

```bash
# 1. Export weights
python scripts/ml/export_weights_to_binary.py

# 2. Build
cmake --build build --target test_neural_net_simd

# 3. Test
./build/bin/test_neural_net_simd
```

## Basic Usage

```cpp
import bigbrother.ml.neural_net_simd;
using namespace bigbrother::ml;

// Create & load
auto net = NeuralNet::create().loadWeights("models/weights/");

// Predict
std::array<float, 80> input = { /* 80 normalized features */ };
auto output = net.predict(input);  // [3] = {1d, 5d, 20d}
```

## Performance

| CPU | Time | Throughput |
|-----|------|------------|
| AVX-512 | 0.05 ms | 20,000/s |
| AVX-2 | 0.08 ms | 12,500/s |
| SSE | 0.15 ms | 6,600/s |

## File Structure

```
src/ml/neural_net_simd.cppm          # Implementation (800 lines)
scripts/ml/export_weights_to_binary.py   # Weight export tool
tests/test_neural_net_simd.cpp       # Test program
docs/SIMD_NEURAL_NETWORK.md          # Full documentation
```

## Common Tasks

### Export Weights
```bash
python scripts/ml/export_weights_to_binary.py
```

### Check CPU Support
```cpp
auto net = NeuralNet::create();
printf("Using: %s\n", net.getInstructionSetName());
```

### Memory Usage
```cpp
printf("Memory: %.2f KB\n", net.getMemoryUsage() / 1024.0);
// Output: ~350 KB
```

## Troubleshooting

**Problem:** Weights not found
```bash
# Solution: Export weights first
python scripts/ml/export_weights_to_binary.py
```

**Problem:** Slow performance
```bash
# Solution: Build with optimization
cmake -DCMAKE_BUILD_TYPE=Release ..
```

**Problem:** Compilation errors
```bash
# Solution: Ensure Clang 21+ with C++23 support
clang++ --version  # Must be 21.0.0+
```

## See Also

- [Full Documentation](SIMD_NEURAL_NETWORK.md)
- [Implementation Summary](SIMD_IMPLEMENTATION_SUMMARY.md)
- [ML Library README](../src/ml/README.md)
