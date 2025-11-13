/**
 * INT8 Quantization Module
 *
 * Provides symmetric quantization for neural network weights and activations.
 * Uses AVX-512 for high-performance quantized matrix operations.
 *
 * Without VNNI (avx512_vnni), we use standard AVX-512 integer instructions
 * which still provides 2-3× speedup over FP32.
 *
 * @module bigbrother.ml.quantization
 */

module;

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <span>
#include <array>
#include <vector>
#include <limits>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

export module bigbrother.ml.quantization;

export namespace bigbrother::ml {
namespace quantization {

/**
 * Quantization parameters for symmetric quantization
 * Maps FP32 range [-scale, +scale] to INT8 range [-127, +127]
 */
struct QuantizationParams {
    float scale;           // Scale factor: fp32_value = int8_value * scale
    float inv_scale;       // Inverse: int8_value = fp32_value / inv_scale
    int8_t zero_point{0};  // Symmetric quantization always uses 0

    QuantizationParams() : scale(1.0f), inv_scale(1.0f) {}

    explicit QuantizationParams(float s)
        : scale(s)
        , inv_scale(1.0f / s)
    {}
};

/**
 * Compute quantization parameters from FP32 data
 * Uses symmetric quantization: [-max_abs, +max_abs] → [-127, +127]
 */
inline auto computeQuantizationParams(std::span<const float> data) -> QuantizationParams {
    float max_abs = 0.0f;
    for (float val : data) {
        max_abs = std::max(max_abs, std::abs(val));
    }

    // Avoid division by zero
    if (max_abs < 1e-8f) {
        max_abs = 1.0f;
    }

    // Map [-max_abs, +max_abs] to [-127, +127]
    float scale = max_abs / 127.0f;
    return QuantizationParams(scale);
}

/**
 * Quantize FP32 array to INT8
 * output[i] = clamp(round(input[i] / scale), -127, 127)
 */
inline auto quantize(std::span<const float> input,
                     std::span<int8_t> output,
                     const QuantizationParams& params) -> void
{
    const float inv_scale = params.inv_scale;
    const int size = static_cast<int>(input.size());

#ifdef __AVX512F__
    // AVX-512: Process 16 floats at a time
    const __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
    const __m512i min_val = _mm512_set1_epi32(-127);
    const __m512i max_val = _mm512_set1_epi32(127);

    int i = 0;
    for (; i + 16 <= size; i += 16) {
        // Load 16 floats
        __m512 fp32_vals = _mm512_loadu_ps(&input[i]);

        // Scale: fp32 / scale
        __m512 scaled = _mm512_mul_ps(fp32_vals, inv_scale_vec);

        // Round to nearest integer
        __m512i int32_vals = _mm512_cvtps_epi32(scaled);  // Default rounding

        // Clamp to [-127, 127]
        int32_vals = _mm512_max_epi32(int32_vals, min_val);
        int32_vals = _mm512_min_epi32(int32_vals, max_val);

        // Convert int32 → int8 (with saturation)
        // Extract and store as int8
        alignas(64) std::array<int32_t, 16> int32_arr;
        _mm512_store_si512(int32_arr.data(), int32_vals);

        for (int j = 0; j < 16; ++j) {
            output[i + j] = static_cast<int8_t>(int32_arr[j]);
        }
    }

    // Handle remainder
    for (; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        quantized = std::clamp(quantized, -127, 127);
        output[i] = static_cast<int8_t>(quantized);
    }
#else
    // Scalar fallback
    for (int i = 0; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        quantized = std::clamp(quantized, -127, 127);
        output[i] = static_cast<int8_t>(quantized);
    }
#endif
}

/**
 * Dequantize INT8 array to FP32
 * output[i] = input[i] * scale
 */
inline auto dequantize(std::span<const int8_t> input,
                       std::span<float> output,
                       const QuantizationParams& params) -> void
{
    const float scale = params.scale;
    const int size = static_cast<int>(input.size());

#ifdef __AVX512F__
    // AVX-512: Process 16 int8 values at a time
    const __m512 scale_vec = _mm512_set1_ps(scale);

    int i = 0;
    for (; i + 16 <= size; i += 16) {
        // Load 16 int8 values
        alignas(64) std::array<int8_t, 16> int8_arr;
        for (int j = 0; j < 16; ++j) {
            int8_arr[j] = input[i + j];
        }

        // Convert int8 → int32 → fp32
        alignas(64) std::array<int32_t, 16> int32_arr;
        for (int j = 0; j < 16; ++j) {
            int32_arr[j] = static_cast<int32_t>(int8_arr[j]);
        }

        __m512i int32_vals = _mm512_load_si512(int32_arr.data());
        __m512 fp32_vals = _mm512_cvtepi32_ps(int32_vals);

        // Scale: int8 * scale
        fp32_vals = _mm512_mul_ps(fp32_vals, scale_vec);

        // Store
        _mm512_storeu_ps(&output[i], fp32_vals);
    }

    // Handle remainder
    for (; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
#else
    // Scalar fallback
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
#endif
}

/**
 * INT8 Matrix-Vector Multiply: y = A @ x (quantized)
 *
 * Without VNNI, we manually compute dot products using:
 * - _mm512_cvtepi8_epi16: int8 → int16 (avoids overflow)
 * - _mm512_mullo_epi16: int16 multiply
 * - _mm512_add_epi32: accumulate to int32
 *
 * Performance: ~2-3× faster than FP32 SIMD (without VNNI)
 *              ~4-5× faster with VNNI (not available on this CPU)
 *
 * @param A Matrix (rows × cols), row-major, INT8
 * @param x Vector (cols), INT8
 * @param y Output vector (rows), INT32 (accumulated, needs dequantization)
 * @param rows Number of rows in A
 * @param cols Number of columns in A
 */
inline auto matmul_int8(const int8_t* A,
                        const int8_t* x,
                        int32_t* y,
                        int rows,
                        int cols) -> void
{
#ifdef __AVX512F__
    for (int r = 0; r < rows; ++r) {
        __m512i acc = _mm512_setzero_si512();  // Accumulator (int32)

        int c = 0;
        // Process 32 int8 elements at a time
        for (; c + 32 <= cols; c += 32) {
            // Load 32 int8 values from matrix row
            __m256i a_int8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&A[r * cols + c]));

            // Load 32 int8 values from vector
            __m256i x_int8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x[c]));

            // Convert int8 → int16 (32 int8 → 32 int16)
            __m512i a_int16 = _mm512_cvtepi8_epi16(a_int8);
            __m512i x_int16 = _mm512_cvtepi8_epi16(x_int8);

            // Multiply int16 × int16 → int16 (32 elements)
            __m512i prod = _mm512_mullo_epi16(a_int16, x_int16);

            // Convert int16 → int32 and accumulate
            // Lower 16 int16 values → 16 int32
            __m256i prod_lo = _mm512_castsi512_si256(prod);
            __m512i prod_lo_int32 = _mm512_cvtepi16_epi32(prod_lo);
            acc = _mm512_add_epi32(acc, prod_lo_int32);

            // Upper 16 int16 values → 16 int32
            __m256i prod_hi = _mm512_extracti64x4_epi64(prod, 1);
            __m512i prod_hi_int32 = _mm512_cvtepi16_epi32(prod_hi);
            acc = _mm512_add_epi32(acc, prod_hi_int32);
        }

        // Horizontal sum of 16 int32 values in acc
        alignas(64) std::array<int32_t, 16> acc_arr;
        _mm512_store_si512(acc_arr.data(), acc);

        int32_t sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += acc_arr[i];
        }

        // Handle remainder
        for (; c < cols; ++c) {
            sum += static_cast<int32_t>(A[r * cols + c]) * static_cast<int32_t>(x[c]);
        }

        y[r] = sum;
    }
#else
    // Scalar fallback
    for (int r = 0; r < rows; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < cols; ++c) {
            sum += static_cast<int32_t>(A[r * cols + c]) * static_cast<int32_t>(x[c]);
        }
        y[r] = sum;
    }
#endif
}

/**
 * INT8 Matrix-Vector Multiply with bias and dequantization
 * y = (A @ x) * scale + bias (all in FP32)
 *
 * @param A_int8 Matrix (rows × cols), INT8
 * @param x_int8 Input vector (cols), INT8
 * @param bias Bias vector (rows), FP32
 * @param y Output vector (rows), FP32
 * @param rows Number of rows
 * @param cols Number of columns
 * @param scale_A Scale for matrix A
 * @param scale_x Scale for vector x
 */
inline auto matmul_int8_dequantize(const int8_t* A_int8,
                                   const int8_t* x_int8,
                                   const float* bias,
                                   float* y,
                                   int rows,
                                   int cols,
                                   float scale_A,
                                   float scale_x) -> void
{
    // Temporary buffer for INT32 accumulation
    std::vector<int32_t> y_int32(rows);

    // Quantized matmul: y_int32 = A_int8 @ x_int8
    matmul_int8(A_int8, x_int8, y_int32.data(), rows, cols);

    // Dequantize and add bias
    const float output_scale = scale_A * scale_x;

#ifdef __AVX512F__
    const __m512 scale_vec = _mm512_set1_ps(output_scale);

    int i = 0;
    for (; i + 16 <= rows; i += 16) {
        // Load 16 int32 values
        __m512i int32_vals = _mm512_loadu_si512(&y_int32[i]);

        // Convert to FP32
        __m512 fp32_vals = _mm512_cvtepi32_ps(int32_vals);

        // Scale
        fp32_vals = _mm512_mul_ps(fp32_vals, scale_vec);

        // Add bias
        __m512 bias_vals = _mm512_loadu_ps(&bias[i]);
        fp32_vals = _mm512_add_ps(fp32_vals, bias_vals);

        // Store
        _mm512_storeu_ps(&y[i], fp32_vals);
    }

    // Remainder
    for (; i < rows; ++i) {
        y[i] = static_cast<float>(y_int32[i]) * output_scale + bias[i];
    }
#else
    for (int i = 0; i < rows; ++i) {
        y[i] = static_cast<float>(y_int32[i]) * output_scale + bias[i];
    }
#endif
}

// ============================================================================
// INT16 Quantization (Higher Precision, 2× memory of INT8)
// ============================================================================

/**
 * Compute INT16 quantization parameters
 * Maps [-max_abs, +max_abs] → [-32767, +32767]
 */
inline auto computeQuantizationParams16(std::span<const float> data) -> QuantizationParams {
    float max_abs = 0.0f;
    for (float val : data) {
        max_abs = std::max(max_abs, std::abs(val));
    }

    if (max_abs < 1e-8f) {
        max_abs = 1.0f;
    }

    float scale = max_abs / 32767.0f;
    return QuantizationParams(scale);
}

/**
 * Quantize FP32 to INT16
 */
inline auto quantize16(std::span<const float> input,
                       std::span<int16_t> output,
                       const QuantizationParams& params) -> void
{
    const float inv_scale = params.inv_scale;
    const int size = static_cast<int>(input.size());

#ifdef __AVX512F__
    const __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
    const __m512i min_val = _mm512_set1_epi32(-32767);
    const __m512i max_val = _mm512_set1_epi32(32767);

    int i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 fp32_vals = _mm512_loadu_ps(&input[i]);
        __m512 scaled = _mm512_mul_ps(fp32_vals, inv_scale_vec);
        __m512i int32_vals = _mm512_cvtps_epi32(scaled);
        int32_vals = _mm512_max_epi32(int32_vals, min_val);
        int32_vals = _mm512_min_epi32(int32_vals, max_val);

        alignas(64) std::array<int32_t, 16> int32_arr;
        _mm512_store_si512(int32_arr.data(), int32_vals);

        for (int j = 0; j < 16; ++j) {
            output[i + j] = static_cast<int16_t>(int32_arr[j]);
        }
    }

    for (; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        quantized = std::clamp(quantized, -32767, 32767);
        output[i] = static_cast<int16_t>(quantized);
    }
#else
    for (int i = 0; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        int32_t quantized = static_cast<int32_t>(std::round(scaled));
        quantized = std::clamp(quantized, -32767, 32767);
        output[i] = static_cast<int16_t>(quantized);
    }
#endif
}

/**
 * Dequantize INT16 to FP32
 */
inline auto dequantize16(std::span<const int16_t> input,
                         std::span<float> output,
                         const QuantizationParams& params) -> void
{
    const float scale = params.scale;
    const int size = static_cast<int>(input.size());

#ifdef __AVX512F__
    const __m512 scale_vec = _mm512_set1_ps(scale);

    int i = 0;
    for (; i + 16 <= size; i += 16) {
        alignas(64) std::array<int32_t, 16> int32_arr;
        for (int j = 0; j < 16; ++j) {
            int32_arr[j] = static_cast<int32_t>(input[i + j]);
        }

        __m512i int32_vals = _mm512_load_si512(int32_arr.data());
        __m512 fp32_vals = _mm512_cvtepi32_ps(int32_vals);
        fp32_vals = _mm512_mul_ps(fp32_vals, scale_vec);
        _mm512_storeu_ps(&output[i], fp32_vals);
    }

    for (; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
#else
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
#endif
}

/**
 * INT16 Matrix-Vector Multiply
 */
inline auto matmul_int16(const int16_t* A,
                         const int16_t* x,
                         int32_t* y,
                         int rows,
                         int cols) -> void
{
#ifdef __AVX512F__
    for (int r = 0; r < rows; ++r) {
        __m512i acc = _mm512_setzero_si512();

        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m256i a_int16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&A[r * cols + c]));
            __m256i x_int16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x[c]));

            __m512i a_int32 = _mm512_cvtepi16_epi32(a_int16);
            __m512i x_int32 = _mm512_cvtepi16_epi32(x_int16);

            __m512i prod = _mm512_mullo_epi32(a_int32, x_int32);
            acc = _mm512_add_epi32(acc, prod);
        }

        alignas(64) std::array<int32_t, 16> acc_arr;
        _mm512_store_si512(acc_arr.data(), acc);

        int32_t sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += acc_arr[i];
        }

        for (; c < cols; ++c) {
            sum += static_cast<int32_t>(A[r * cols + c]) * static_cast<int32_t>(x[c]);
        }

        y[r] = sum;
    }
#else
    for (int r = 0; r < rows; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < cols; ++c) {
            sum += static_cast<int32_t>(A[r * cols + c]) * static_cast<int32_t>(x[c]);
        }
        y[r] = sum;
    }
#endif
}

/**
 * INT16 matmul with dequantization
 */
inline auto matmul_int16_dequantize(const int16_t* A_int16,
                                    const int16_t* x_int16,
                                    const float* bias,
                                    float* y,
                                    int rows,
                                    int cols,
                                    float scale_A,
                                    float scale_x) -> void
{
    std::vector<int32_t> y_int32(rows);
    matmul_int16(A_int16, x_int16, y_int32.data(), rows, cols);

    const float output_scale = scale_A * scale_x;

#ifdef __AVX512F__
    const __m512 scale_vec = _mm512_set1_ps(output_scale);

    int i = 0;
    for (; i + 16 <= rows; i += 16) {
        __m512i int32_vals = _mm512_loadu_si512(&y_int32[i]);
        __m512 fp32_vals = _mm512_cvtepi32_ps(int32_vals);
        fp32_vals = _mm512_mul_ps(fp32_vals, scale_vec);
        __m512 bias_vals = _mm512_loadu_ps(&bias[i]);
        fp32_vals = _mm512_add_ps(fp32_vals, bias_vals);
        _mm512_storeu_ps(&y[i], fp32_vals);
    }

    for (; i < rows; ++i) {
        y[i] = static_cast<float>(y_int32[i]) * output_scale + bias[i];
    }
#else
    for (int i = 0; i < rows; ++i) {
        y[i] = static_cast<float>(y_int32[i]) * output_scale + bias[i];
    }
#endif
}

} // namespace quantization
} // namespace bigbrother::ml
