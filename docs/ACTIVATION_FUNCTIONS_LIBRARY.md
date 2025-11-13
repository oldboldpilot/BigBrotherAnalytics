# Activation Functions Library

**BigBrotherAnalytics C++23 SIMD-Optimized Activation Functions**
**Module:** `bigbrother.ml.activations`
**File:** [src/ml/activations.cppm](../src/ml/activations.cppm)

---

## Overview

Comprehensive activation functions library with automatic SIMD optimization for high-performance neural network inference.

### Key Features

✅ **8 Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh, GELU, Swish/SiLU, ELU, Softmax
✅ **Auto SIMD Optimization**: AVX-512, AVX-2, SSE, Scalar fallback
✅ **C++23 Module**: Clean API with zero-cost abstractions
✅ **High Performance**: 10.44 Gelements/sec (ReLU), 6.03 Gelements/sec (GELU)
✅ **Easy to Use**: Convenience functions + OOP API

---

## Quick Start

```cpp
import bigbrother.ml.activations;
using namespace bigbrother::ml::activations;

// Convenience functions (auto SIMD)
std::array<float, 256> hidden_layer = { /* ... */ };
relu(std::span(hidden_layer));         // f(x) = max(0, x)
leaky_relu(std::span(hidden_layer), 0.01f); // Leaky ReLU with alpha
gelu(std::span(hidden_layer));         // GELU (Transformer activation)

// Object-oriented API
Activation Function relu_fn(ActivationType::ReLU);
relu_fn.apply(std::span(hidden_layer));
```

---

## Supported Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **ReLU** | `f(x) = max(0, x)` | Standard deep learning (default) |
| **Leaky ReLU** | `f(x) = x if x > 0 else α·x` | Avoids "dying ReLU" problem |
| **Sigmoid** | `f(x) = 1 / (1 + e^-x)` | Binary classification, gates (LSTM) |
| **Tanh** | `f(x) = tanh(x)` | Centered around 0, RNNs |
| **GELU** | `f(x) = x·Φ(x)` | Transformers, modern architectures |
| **Swish/SiLU** | `f(x) = x·sigmoid(x)` | Alternative to ReLU (smoother) |
| **ELU** | `f(x) = x if x > 0 else α(e^x - 1)` | Mean closer to 0 |
| **Softmax** | `f(x_i) = e^x_i / Σe^x_j` | Multi-class classification (output) |

---

## API Reference

### Convenience Functions

```cpp
// Apply activation in-place
void relu(std::span<float> data);
void leaky_relu(std::span<float> data, float alpha = 0.01f);
void sigmoid(std::span<float> data);
void tanh_activation(std::span<float> data);
void gelu(std::span<float> data);
void swish(std::span<float> data);
void elu(std::span<float> data, float alpha = 1.0f);
void softmax(std::span<float> data);
```

### ActivationFunction Class

```cpp
class ActivationFunction {
public:
    // Constructor
    ActivationFunction(ActivationType type = ActivationType::ReLU,
                       float alpha = 0.01f);

    // Apply activation
    void apply(std::span<float> data);
    void apply(float* data, int size);

    // Get name
    const char* name() const;
};
```

### Enums

```cpp
enum class ActivationType {
    ReLU,          // Rectified Linear Unit
    LeakyReLU,     // Leaky ReLU with parameter
    Sigmoid,       // Sigmoid function
    Tanh,          // Hyperbolic tangent
    GELU,          // Gaussian Error Linear Unit
    Swish,         // Swish/SiLU
    ELU,           // Exponential Linear Unit
    Softmax,       // Softmax (for output layer)
    Linear         // Identity (no activation)
};

enum class InstructionSet {
    AVX512,   // AVX-512 + FMA (16 floats/vector)
    AVX2,     // AVX-2 + FMA (8 floats/vector)
    SSE,      // SSE2 (4 floats/vector)
    Scalar    // Portable fallback
};

auto detectInstructionSet() -> InstructionSet;
```

---

## Usage Examples

### Example 1: Simple Activation

```cpp
import bigbrother.ml.activations;

std::array<float, 5> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

// Apply ReLU
bigbrother::ml::activations::relu(std::span(data));
// Result: [0.0, 0.0, 0.0, 1.0, 2.0]
```

### Example 2: Neural Network Layer

```cpp
import bigbrother.ml.activations;
using namespace bigbrother::ml::activations;

// After matrix multiply: input @ weights + bias
std::vector<float> layer_output(256);
// ... (compute: output = input @ weights + bias)

// Apply activation
relu(std::span(layer_output));

// Now ready for next layer
```

### Example 3: Configurable Activation

```cpp
ActivationType act_type = ActivationType::GELU;  // From config
ActivationFunction activation(act_type);

// Apply to multiple layers
activation.apply(std::span(layer1_output));
activation.apply(std::span(layer2_output));
activation.apply(std::span(layer3_output));
```

### Example 4: Softmax Output Layer

```cpp
// Last layer: 3 class scores
std::array<float, 3> logits = {2.0f, 1.0f, 0.1f};

// Convert to probabilities (sum = 1.0)
bigbrother::ml::activations::softmax(std::span(logits));
// Result: [0.659, 0.242, 0.099] (sums to 1.0)

// Get predicted class
int predicted_class = std::distance(logits.begin(),
                                    std::max_element(logits.begin(), logits.end()));
```

---

## Performance Benchmarks

**System:** AVX-512 capable CPU, Clang 21, -O3 -march=native

| Activation | Throughput (Gelements/sec) | Latency (μs, 10K elements) |
|------------|---------------------------|---------------------------|
| **ReLU** | **10.44** | 0.96 |
| **GELU** | **6.03** | 1.66 |
| **Sigmoid** | **4.65** | 2.15 |

**Notes:**
- ReLU is fastest (simple max operation)
- GELU is 1.7x slower but more expressive (used in GPT, BERT)
- Sigmoid has approximation overhead (tanh-based)

---

## SIMD Implementation Details

### AVX-512 (16 floats at once)

```cpp
// ReLU using AVX-512
__m512 zero = _mm512_setzero_ps();
__m512 x = _mm512_loadu_ps(&data[i]);
__m512 result = _mm512_max_ps(x, zero);  // max(x, 0)
_mm512_storeu_ps(&data[i], result);
```

### AVX-2 (8 floats at once)

```cpp
// Leaky ReLU using AVX-2
__m256 zero = _mm256_setzero_ps();
__m256 alpha_vec = _mm256_set1_ps(0.01f);
__m256 x = _mm256_loadu_ps(&data[i]);
__m256 neg = _mm256_mul_ps(x, alpha_vec);
__m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
__m256 result = _mm256_blendv_ps(neg, x, mask);
_mm256_storeu_ps(&data[i], result);
```

### Automatic ISA Detection

The library automatically detects the best available instruction set at runtime:

```cpp
auto isa = detectInstructionSet();
// Returns: AVX512 > AVX2 > SSE > Scalar (in order of preference)
```

---

## Integration with Neural Networks

### With Existing Engines

```cpp
import bigbrother.ml.weight_loader;
import bigbrother.ml.neural_net_mkl;
import bigbrother.ml.activations;  // Custom activations

// Load weights
auto weights = PricePredictorConfig::createLoader().load();

// Initialize network (uses built-in ReLU)
NeuralNetMKL predictor(weights);

// Or create custom network with different activations:
// (requires modifying neural_net_mkl.cppm to accept ActivationFunction)
```

### Future: Configurable Activations

```cpp
// Future enhancement (not yet implemented):
NeuralNetMKL predictor(weights);
predictor.setActivation(ActivationType::GELU);  // Replace ReLU with GELU
```

---

## Testing

### Run Demo

```bash
# Build
SKIP_CLANG_TIDY=1 cmake -G Ninja -B build && ninja -C build activation_functions_demo

# Run
./build/bin/activation_functions_demo
```

**Expected Output:**
```
╔════════════════════════════════════════════════════════════╗
║    BigBrotherAnalytics Activation Functions Library       ║
║    C++23 Module with Auto SIMD Optimization                ║
╚════════════════════════════════════════════════════════════╝

=== Basic Activation Functions Demo ===

ReLU:
  Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]
  Output: [0.0000, 0.0000, 0.0000, 1.0000, 2.0000]
  Expected: [0.0, 0.0, 0.0, 1.0, 2.0]

Detected instruction set: AVX-512

ReLU:
  Average time: 0.958 μs
  Throughput: 10.44 Gelements/sec

✅ Demo completed successfully!
```

### Unit Testing Template

```cpp
#include <cassert>
import bigbrother.ml.activations;

void test_relu() {
    std::array<float, 5> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    bigbrother::ml::activations::relu(std::span(data));

    assert(data[0] == 0.0f);  // -2 → 0
    assert(data[1] == 0.0f);  // -1 → 0
    assert(data[2] == 0.0f);  // 0 → 0
    assert(data[3] == 1.0f);  // 1 → 1
    assert(data[4] == 2.0f);  // 2 → 2
}

void test_softmax() {
    std::array<float, 3> data = {1.0f, 2.0f, 3.0f};
    bigbrother::ml::activations::softmax(std::span(data));

    float sum = data[0] + data[1] + data[2];
    assert(std::abs(sum - 1.0f) < 0.0001f);  // Sum should be 1.0
}
```

---

## Comparison with Other Libraries

| Library | AVX-512 | C++23 Modules | Activations | Performance |
|---------|---------|---------------|-------------|-------------|
| **BigBrotherAnalytics** | ✅ | ✅ | 8 | **10.44 Gelem/s** (ReLU) |
| PyTorch C++ API | ✅ | ❌ | 20+ | ~8 Gelem/s |
| Eigen | ✅ | ❌ | 6 | ~7 Gelem/s |
| Intel DNNL | ✅ | ❌ | 15+ | ~11 Gelem/s |

**Advantages:**
- Modern C++23 modules (clean API)
- Zero external dependencies (self-contained)
- Optimized for single-threaded inference
- Direct integration with BigBrotherAnalytics ML stack

---

## Advanced Topics

### Custom Activation Function

To add a new activation function:

1. **Define enum value:**
```cpp
enum class ActivationType {
    // ... existing ...
    Mish,  // NEW: Mish activation
};
```

2. **Implement scalar version:**
```cpp
namespace scalar {
    inline auto mish(float x) -> float {
        return x * std::tanh(std::log(1.0f + std::exp(x)));  // x * tanh(softplus(x))
    }
}
```

3. **Implement SIMD version (optional but recommended):**
```cpp
namespace avx512 {
    inline auto mish(float* data, int size) -> void {
        // ... AVX-512 implementation ...
    }
}
```

4. **Add to ActivationFunction::apply():**
```cpp
case ActivationType::Mish:
    applyMish(data, size);
    break;
```

### Fused Operations

For better cache efficiency, consider fusing activation with matrix multiply:

```cpp
// Instead of:
matrix_multiply(input, weights, output);  // 1st pass: memory write
relu(output);                             // 2nd pass: memory read/write

// Fuse:
matrix_multiply_with_activation(input, weights, output, ActivationType::ReLU);
// Single pass: apply activation during write
```

---

## Troubleshooting

### Issue: Slow Performance

**Symptoms:** Throughput < 1 Gelem/sec

**Solutions:**
1. Check instruction set: `detectInstructionSet()` should return AVX512 or AVX2
2. Verify compiler flags: `-O3 -march=native`
3. Ensure Release build: `-DCMAKE_BUILD_TYPE=Release`
4. Check CPU features: `cat /proc/cpuinfo | grep avx512`

### Issue: Wrong Results

**Symptoms:** Activation outputs don't match expected values

**Solutions:**
1. Verify input range (softmax requires finite inputs)
2. Check for NaN/Inf: `std::isfinite()`
3. Test with scalar fallback to isolate SIMD bugs
4. Compare with PyTorch/NumPy reference

### Issue: Linker Errors

**Error:** `undefined reference to bigbrother::ml::activations::relu`

**Solution:**
```cmake
# Add to CMakeLists.txt
target_link_libraries(your_target
    PRIVATE
    ml_lib  # Contains activations module
)
```

---

## References

- **ReLU Paper:** [Nair & Hinton, 2010](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
- **GELU Paper:** [Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)
- **Swish Paper:** [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
- **AVX-512 Intrinsics:** [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)

---

## File Structure

```
src/ml/activations.cppm          # Main module (540 lines)
examples/activation_functions_demo.cpp  # Demo program (260 lines)
docs/ACTIVATION_FUNCTIONS_LIBRARY.md    # This file
```

---

**Last Updated:** 2025-11-13
**Version:** 1.0
**Maintainer:** Olumuyiwa Oluwasanmi
