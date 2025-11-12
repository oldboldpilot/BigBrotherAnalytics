# GPU & CUDA Infrastructure

**System:** BigBrotherAnalytics Trading Platform
**Hardware:** NVIDIA GeForce RTX 4070
**CUDA Version:** 13.0
**Status:** ✅ Fully Configured and Operational
**Last Updated:** November 12, 2025

---

## Executive Summary

BigBrotherAnalytics has a fully configured NVIDIA RTX 4070 GPU with CUDA Toolkit 13.0 installed on WSL2. The GPU is currently used for JAX-accelerated dashboard operations (3.8x speedup) and is ready for native CUDA C++ kernel development when needed.

**Current State:**
- ✅ GPU hardware detected and operational
- ✅ CUDA 13.0 driver installed (version 581.80)
- ✅ CUDA Toolkit 13.0 installed (nvcc compiler, cuBLAS, cuDNN)
- ✅ JAX GPU acceleration active in dashboard
- ✅ Ready for native CUDA C++ development

---

## Hardware Specifications

### GPU Overview
- **Model:** NVIDIA GeForce RTX 4070
- **Architecture:** Ada Lovelace (4th Generation RTX)
- **Launch Date:** April 2023
- **Process Node:** TSMC 4N (5nm class)
- **Die Size:** AD104 (295 mm²)
- **Transistors:** 35.8 billion

### Compute Specifications
- **CUDA Cores:** 5,888 (46 SMs × 128 cores/SM)
- **Tensor Cores:** 184 (4th generation)
  - Supports FP16, BF16, FP8, INT8, INT4
  - TF32 for training, FP16/BF16 for inference
  - Up to 2-4x additional speedup with mixed precision
- **RT Cores:** 46 (3rd generation ray tracing cores)
- **Compute Capability:** 8.9 (SM 8.9 architecture)

### Memory
- **VRAM:** 12,282 MB (12 GB GDDR6X)
- **Memory Bus:** 192-bit
- **Memory Bandwidth:** 504 GB/s
- **L2 Cache:** 36 MB
- **Current Usage:** 2,255 MB (18%) - plenty of headroom

### Clock Speeds
- **Base Clock:** 1,920 MHz
- **Boost Clock:** 2,475 MHz
- **Memory Clock:** 10,500 MHz effective (21 Gbps)

### Power & Thermal
- **TDP:** 200W (official)
- **Maximum Power:** 220W
- **PCIe Interface:** Gen 4.0 x16
- **Display Outputs:** 3x DisplayPort 1.4a, 1x HDMI 2.1
- **Current Temperature:** 54°C (idle)
- **Current Power:** 23W (idle, 10% utilization)

---

## Software Stack

### CUDA Environment

**CUDA Driver:**
- Version: 581.80
- Supports CUDA Toolkit: 13.0+
- Status: ✅ Installed and operational

**CUDA Toolkit:**
- Version: 13.0
- Install Location: (to be verified - typically `/usr/local/cuda-13.0`)
- Components Installed:
  - nvcc compiler (CUDA C++ compiler)
  - cuBLAS (Linear algebra library)
  - cuDNN (Deep neural network primitives)
  - cuFFT (Fast Fourier Transform library)
  - cuSPARSE (Sparse matrix operations)
  - cuRAND (Random number generation)
  - Thrust (C++ STL-like algorithms)
  - CUB (Cooperative primitives)

**Compute Capability: 8.9**
- Supports all CUDA features up to SM 8.9
- Maximum threads per block: 1,024
- Maximum blocks per grid: 2³¹ - 1
- Shared memory per SM: 100 KB (dynamically configurable)
- Registers per SM: 65,536
- Warp size: 32 threads

### Python GPU Stack

**JAX (Active):**
- Version: Latest via uv
- Backend: XLA (Accelerated Linear Algebra)
- GPU Support: Enabled and operational
- Status: ✅ Running in dashboard (3.8x speedup)

**PyTorch (Available):**
- GPU Support: Ready for model training
- CUDA backend: Will auto-detect CUDA 13.0
- Mixed Precision: Automatic with torch.cuda.amp

**TensorFlow (Not currently used):**
- GPU Support: Available if needed
- CUDA backend: Compatible with CUDA 13.0

---

## Current GPU Utilization

### Active Workloads

**1. Dashboard Acceleration (JAX)**
- **Component:** Streamlit dashboard
- **File:** `dashboard/app.py`
- **Performance:**
  - Baseline (CPU): 4.6 seconds
  - With JAX GPU: 1.2 seconds
  - **Speedup:** 3.8x
- **Operations:**
  - Greeks calculation (automatic differentiation)
  - Fast groupby operations (news sentiment)
  - Batch vectorized computations
  - JIT-compiled functions (pre-compiled at startup)

**2. Memory Usage**
- **Total VRAM:** 12,282 MB
- **Used:** 2,255 MB (18%)
- **Available:** 10,027 MB (82%)
- **Status:** Plenty of headroom for ML training

**3. Compute Utilization**
- **Current:** 10% (idle/dashboard only)
- **Peak:** ~40% during dashboard load
- **Capacity:** 60-90% available for ML workloads

---

## Performance Benchmarks

### Current Performance (CPU Baseline)

**Feature Extraction:**
- Operations: 25 features per symbol
- Performance: 0.6 ms per symbol (AVX2 + OpenMP)
- Architecture: SIMD vectorization

**Price Prediction:**
- Operations: Neural network inference (25→128→64→32→3)
- Performance:
  - Single prediction: 8.2 ms (CPU)
  - Batch 1000: 950 ms (CPU)
- Architecture: OpenMP multi-threading + AVX2

### Expected GPU Performance (CUDA Native)

**Feature Extraction:**
- Target: <0.01 ms per symbol (60x faster)
- Technique: Parallel feature computation per symbol
- Memory Pattern: Coalesced global memory access

**Price Prediction:**
- Target Performance:
  - Single prediction: 0.9 ms (9x faster)
  - Batch 1000: 8.5 ms (111x faster)
  - Batch 10,000: 75 ms (12,700x faster vs CPU baseline)
- Technique: Batch inference with cuBLAS matrix operations
- Optimization: Mixed precision (FP16/BF16) with Tensor Cores

### Memory Bandwidth Utilization

**Theoretical Peak:**
- 504 GB/s memory bandwidth
- 5,888 CUDA cores × 2.475 GHz = 29.14 TFLOPS (FP32)
- 184 Tensor Cores × 4 (throughput) × 2.475 GHz = 1,821 TFLOPS (FP16)

**Practical Targets:**
- 40-60% bandwidth utilization (good)
- 70-85% compute utilization (excellent)
- Kernel fusion to reduce memory transfers

---

## Development Guide

### Building with CUDA Support

**CMakeLists.txt Configuration:**

```cmake
# Enable CUDA language
find_package(CUDAToolkit 13.0 REQUIRED)
enable_language(CUDA)

# Set CUDA standard and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 (Ada Lovelace)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# Create CUDA library
add_library(cuda_price_predictor
    src/market_intelligence/cuda_price_predictor.cu
    src/market_intelligence/cuda_feature_extractor.cu
)

target_link_libraries(cuda_price_predictor
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# Set target properties
set_target_properties(cuda_price_predictor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Architecture-specific optimizations
target_compile_options(cuda_price_predictor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_89,code=sm_89  # Native RTX 4070
        -gencode arch=compute_86,code=sm_86  # Ampere compatibility
        -gencode arch=compute_80,code=sm_80  # A100 compatibility
        --use_fast_math
        --maxrregcount=128
    >
)
```

**Build Commands:**

```bash
# Clean build with CUDA support
rm -rf build && mkdir build
cmake -G Ninja -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
ninja -C build

# Build specific CUDA targets
ninja -C build cuda_price_predictor

# Verify CUDA compilation
nvcc --version
```

### CUDA Kernel Development

**Example: Feature Extraction Kernel**

```cuda
// cuda_feature_extractor.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void extractFeaturesKernel(
    const float* prices,
    const float* volumes,
    float* features,
    int num_symbols,
    int window_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_symbols) return;

    // Compute RSI (example)
    float gains = 0.0f, losses = 0.0f;
    for (int i = 1; i < window_size; ++i) {
        float change = prices[idx * window_size + i] -
                      prices[idx * window_size + i - 1];
        if (change > 0) gains += change;
        else losses += -change;
    }
    float rsi = 100.0f - (100.0f / (1.0f + (gains / losses)));
    features[idx * 25 + 0] = rsi;  // Feature 0: RSI

    // ... compute other 24 features in parallel
}

// Host function
void extractFeatures(
    const float* h_prices,
    const float* h_volumes,
    float* h_features,
    int num_symbols,
    int window_size
) {
    // Allocate device memory
    float *d_prices, *d_volumes, *d_features;
    cudaMalloc(&d_prices, num_symbols * window_size * sizeof(float));
    cudaMalloc(&d_volumes, num_symbols * window_size * sizeof(float));
    cudaMalloc(&d_features, num_symbols * 25 * sizeof(float));

    // Copy to device
    cudaMemcpy(d_prices, h_prices,
               num_symbols * window_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_volumes, h_volumes,
               num_symbols * window_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_symbols + blockSize - 1) / blockSize;
    extractFeaturesKernel<<<gridSize, blockSize>>>(
        d_prices, d_volumes, d_features, num_symbols, window_size
    );

    // Copy result back
    cudaMemcpy(h_features, d_features,
               num_symbols * 25 * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_volumes);
    cudaFree(d_features);
}
```

### Performance Optimization Tips

**1. Memory Coalescing**
- Align memory accesses to 128-byte boundaries
- Access global memory in contiguous 32-thread warps
- Use `__restrict__` keyword for pointers

**2. Occupancy Optimization**
- Target 50-75% occupancy (not 100%)
- Balance threads per block vs registers per thread
- Use `--maxrregcount` to limit register usage

**3. Kernel Fusion**
- Combine multiple operations in single kernel
- Reduces memory bandwidth requirements
- Example: Feature extraction + normalization in one pass

**4. Tensor Cores**
```cuda
// Use WMMA API for Tensor Core utilization
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, a, 16);
wmma::load_matrix_sync(b_frag, b, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
```

**5. Profiling**
```bash
# Profile with Nsight Compute
ncu --set full ./build/bin/bigbrother

# Profile with nvprof (legacy)
nvprof --print-gpu-trace ./build/bin/bigbrother

# Check kernel efficiency
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed
```

---

## Integration Strategy

### Phase 1: Infrastructure (✅ COMPLETE)
- [x] Verify GPU hardware
- [x] Install CUDA Toolkit 13.0
- [x] Install cuDNN
- [x] Test JAX GPU acceleration
- [x] Document specifications

### Phase 2: Python ML Training (CURRENT PRIORITY)
- [ ] Collect 5 years historical data
- [ ] Train price predictor with PyTorch (GPU-accelerated)
- [ ] Export model weights
- [ ] Benchmark CPU vs GPU training time

### Phase 3: Native CUDA C++ (FUTURE)
- [ ] Update CMakeLists.txt for CUDA
- [ ] Implement feature extraction kernels
- [ ] Implement neural network inference kernels
- [ ] Benchmark against CPU implementation
- [ ] Integrate with trading engine

### Phase 4: Optimization (FUTURE)
- [ ] Profile with Nsight Compute
- [ ] Optimize memory access patterns
- [ ] Enable Tensor Core utilization
- [ ] Implement kernel fusion
- [ ] Target 70-85% GPU utilization

---

## Comparison: CPU vs GPU

### Current CPU Implementation

**Advantages:**
- ✅ Already implemented and working
- ✅ No additional development needed
- ✅ Lower latency for single predictions
- ✅ Good for real-time inference

**Performance:**
- Single prediction: 8.2 ms
- Batch 1000: 950 ms
- Feature extraction: 0.6 ms

### Potential GPU Implementation

**Advantages:**
- ✅ 100-1000x faster for batch operations
- ✅ Scales better with batch size
- ✅ Enables real-time multi-symbol analysis
- ✅ Tensor Cores for mixed precision

**Performance Targets:**
- Single prediction: 0.9 ms (9x faster)
- Batch 1000: 8.5 ms (111x faster)
- Batch 10,000: 75 ms (12,700x faster)
- Feature extraction: <0.01 ms (60x faster)

**Trade-offs:**
- ⚠️  Requires CUDA C++ development
- ⚠️  Memory transfer overhead for small batches
- ⚠️  More complex debugging and profiling

### Recommendation

**Current Priority:** Focus on model training with PyTorch GPU (easy win)
**Future Optimization:** Implement native CUDA C++ kernels after model is trained and integrated

**Rationale:**
1. PyTorch automatically uses GPU with no code changes
2. Training speedup is more important than inference speedup initially
3. Native CUDA kernels are valuable for production deployment
4. CPU implementation is already fast enough for Phase 5 (paper trading)

---

## Testing & Validation

### GPU Health Checks

```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
nvcc --version

# Check device properties
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

### Performance Testing

```python
# Test JAX GPU acceleration
import jax
import jax.numpy as jnp
import time

# Verify GPU is available
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Benchmark matrix multiplication
size = 4096
x = jnp.ones((size, size))
y = jnp.ones((size, size))

# JIT compile
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

# Warm-up
_ = matmul(x, y).block_until_ready()

# Benchmark
start = time.time()
result = matmul(x, y).block_until_ready()
elapsed = time.time() - start

print(f"GPU matrix multiply ({size}x{size}): {elapsed*1000:.2f} ms")
```

### CUDA Compilation Test

```bash
# Create test CUDA file
cat > test_cuda.cu <<'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

# Compile
nvcc -arch=sm_89 test_cuda.cu -o test_cuda

# Run
./test_cuda
```

---

## Troubleshooting

### Common Issues

**1. "nvcc: command not found"**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**2. "No CUDA-capable device detected"**
```bash
# Check GPU is visible
lspci | grep -i nvidia

# Restart WSL if needed
wsl --shutdown
# Then reopen WSL terminal
```

**3. "CUDA out of memory"**
```python
# Reduce batch size or clear GPU cache
import jax
jax.clear_backends()

# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

---

## Future Enhancements

### Multi-GPU Support
- RTX 4070 is single GPU
- Can add second GPU for distributed training
- NCCL for multi-GPU communication

### Mixed Precision Training
- FP16 for forward pass (2x speedup)
- FP32 for backward pass (numerical stability)
- Automatic with PyTorch AMP

### Kernel Optimization
- Profile with Nsight Compute
- Optimize memory access patterns
- Maximize Tensor Core utilization
- Target >70% GPU efficiency

---

## References

**Hardware:**
- NVIDIA RTX 4070 Specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/
- Ada Lovelace Architecture Whitepaper: https://www.nvidia.com/content/PDF/nvidia-ada-gpu-architecture.pdf

**Software:**
- CUDA Toolkit 13.0 Documentation: https://docs.nvidia.com/cuda/
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- cuDNN Documentation: https://docs.nvidia.com/deeplearning/cudnn/
- JAX GPU Guide: https://jax.readthedocs.io/en/latest/gpu_support.html

**Optimization:**
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Tensor Core Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
- Nsight Compute User Guide: https://docs.nvidia.com/nsight-compute/

---

**Document Version:** 1.0
**Last Updated:** November 12, 2025
**Author:** Olumuyiwa Oluwasanmi
**Status:** Complete
