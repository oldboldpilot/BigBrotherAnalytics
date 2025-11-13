/**
 * Activation Functions Library (C++23 Module)
 *
 * Provides optimized activation functions for neural networks with:
 * - Scalar implementations for portability
 * - SIMD-optimized versions (AVX-512, AVX-2, SSE)
 * - Support for common activations: ReLU, Leaky ReLU, Sigmoid, Tanh, GELU, Swish, ELU, Softmax
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 */

module;

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <span>
#include <array>

export module bigbrother.ml.activations;

export namespace bigbrother::ml {
namespace activations {

/**
 * Activation function types
 */
enum class ActivationType {
    ReLU,          // Rectified Linear Unit: f(x) = max(0, x)
    LeakyReLU,     // Leaky ReLU: f(x) = x if x > 0 else alpha*x
    Sigmoid,       // Sigmoid: f(x) = 1 / (1 + e^-x)
    Tanh,          // Hyperbolic tangent: f(x) = tanh(x)
    GELU,          // Gaussian Error Linear Unit (approximation)
    Swish,         // Swish/SiLU: f(x) = x * sigmoid(x)
    ELU,           // Exponential Linear Unit: f(x) = x if x > 0 else alpha*(e^x - 1)
    Softmax,       // Softmax: f(x_i) = e^x_i / sum(e^x_j)
    Linear         // Linear (identity): f(x) = x
};

/**
 * CPU Instruction Set Detection
 */
enum class InstructionSet {
    AVX512,
    AVX2,
    SSE,
    Scalar
};

/**
 * Detect available instruction set
 */
inline auto detectInstructionSet() -> InstructionSet {
#ifdef __AVX512F__
    return InstructionSet::AVX512;
#elif __AVX2__
    return InstructionSet::AVX2;
#elif __SSE2__
    return InstructionSet::SSE;
#else
    return InstructionSet::Scalar;
#endif
}

/**
 * Scalar Activation Functions (Portable, No SIMD)
 */
namespace scalar {

    inline auto relu(float x) -> float {
        return std::max(0.0f, x);
    }

    inline auto leaky_relu(float x, float alpha = 0.01f) -> float {
        return x > 0.0f ? x : alpha * x;
    }

    inline auto sigmoid(float x) -> float {
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline auto tanh_activation(float x) -> float {
        return std::tanh(x);
    }

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    inline auto gelu(float x) -> float {
        constexpr float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
        constexpr float coeff = 0.044715f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        return 0.5f * x * (1.0f + std::tanh(inner));
    }

    inline auto swish(float x) -> float {
        return x * sigmoid(x);
    }

    inline auto elu(float x, float alpha = 1.0f) -> float {
        return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
    }

} // namespace scalar

/**
 * AVX-512 SIMD Activation Functions (16 floats at once)
 */
namespace avx512 {

#ifdef __AVX512F__
    inline auto relu(float* data, int size) -> void {
        __m512 zero = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= size; i += 16) {
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 result = _mm512_max_ps(x, zero);
            _mm512_storeu_ps(&data[i], result);
        }
        // Handle remainder with scalar
        for (; i < size; ++i) {
            data[i] = scalar::relu(data[i]);
        }
    }

    inline auto leaky_relu(float* data, int size, float alpha = 0.01f) -> void {
        __m512 zero = _mm512_setzero_ps();
        __m512 alpha_vec = _mm512_set1_ps(alpha);
        int i = 0;
        for (; i + 16 <= size; i += 16) {
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 neg = _mm512_mul_ps(x, alpha_vec);
            __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
            __m512 result = _mm512_mask_blend_ps(mask, neg, x);
            _mm512_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::leaky_relu(data[i], alpha);
        }
    }

    // Fast sigmoid approximation: 1 / (1 + e^-x) ≈ 0.5 + 0.5 * tanh(0.5 * x)
    inline auto sigmoid(float* data, int size) -> void {
        __m512 half = _mm512_set1_ps(0.5f);
        int i = 0;
        for (; i + 16 <= size; i += 16) {
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 x_half = _mm512_mul_ps(x, half);

            // Fast tanh approximation using polynomial
            __m512 x2 = _mm512_mul_ps(x_half, x_half);
            __m512 x4 = _mm512_mul_ps(x2, x2);

            // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
            __m512 c27 = _mm512_set1_ps(27.0f);
            __m512 c9 = _mm512_set1_ps(9.0f);
            __m512 num = _mm512_fmadd_ps(x2, x_half, _mm512_mul_ps(c27, x_half));
            __m512 den = _mm512_fmadd_ps(x2, c9, c27);
            __m512 tanh_approx = _mm512_div_ps(num, den);

            __m512 result = _mm512_fmadd_ps(tanh_approx, half, half);
            _mm512_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::sigmoid(data[i]);
        }
    }

    inline auto gelu(float* data, int size) -> void {
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;
        __m512 half = _mm512_set1_ps(0.5f);
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 sqrt_2pi = _mm512_set1_ps(sqrt_2_over_pi);
        __m512 coeff_vec = _mm512_set1_ps(coeff);

        int i = 0;
        for (; i + 16 <= size; i += 16) {
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 x_sq = _mm512_mul_ps(x, x);
            __m512 x_cu = _mm512_mul_ps(x_sq, x);

            // sqrt_2_over_pi * (x + 0.044715 * x^3)
            __m512 inner = _mm512_fmadd_ps(coeff_vec, x_cu, x);
            inner = _mm512_mul_ps(sqrt_2pi, inner);

            // Fast tanh approximation
            __m512 tanh_val = _mm512_max_ps(_mm512_set1_ps(-1.0f),
                              _mm512_min_ps(_mm512_set1_ps(1.0f), inner));

            // 0.5 * x * (1 + tanh)
            __m512 result = _mm512_mul_ps(half, x);
            result = _mm512_mul_ps(result, _mm512_add_ps(one, tanh_val));

            _mm512_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::gelu(data[i]);
        }
    }

#endif // __AVX512F__

} // namespace avx512

/**
 * AVX-2 SIMD Activation Functions (8 floats at once)
 */
namespace avx2 {

#ifdef __AVX2__
    inline auto relu(float* data, int size) -> void {
        __m256 zero = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::relu(data[i]);
        }
    }

    inline auto leaky_relu(float* data, int size, float alpha = 0.01f) -> void {
        __m256 zero = _mm256_setzero_ps();
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        int i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 neg = _mm256_mul_ps(x, alpha_vec);
            __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 result = _mm256_blendv_ps(neg, x, mask);
            _mm256_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::leaky_relu(data[i], alpha);
        }
    }

    inline auto sigmoid(float* data, int size) -> void {
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        int i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 x_half = _mm256_mul_ps(x, half);

            // Fast tanh approximation
            __m256 x2 = _mm256_mul_ps(x_half, x_half);
            __m256 c27 = _mm256_set1_ps(27.0f);
            __m256 c9 = _mm256_set1_ps(9.0f);
            __m256 num = _mm256_fmadd_ps(x2, x_half, _mm256_mul_ps(c27, x_half));
            __m256 den = _mm256_fmadd_ps(x2, c9, c27);
            __m256 tanh_approx = _mm256_div_ps(num, den);

            __m256 result = _mm256_fmadd_ps(tanh_approx, half, half);
            _mm256_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::sigmoid(data[i]);
        }
    }

    inline auto gelu(float* data, int size) -> void {
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 sqrt_2pi = _mm256_set1_ps(sqrt_2_over_pi);
        __m256 coeff_vec = _mm256_set1_ps(coeff);

        int i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 x_sq = _mm256_mul_ps(x, x);
            __m256 x_cu = _mm256_mul_ps(x_sq, x);

            __m256 inner = _mm256_fmadd_ps(coeff_vec, x_cu, x);
            inner = _mm256_mul_ps(sqrt_2pi, inner);

            // Clamp for tanh approximation
            __m256 tanh_val = _mm256_max_ps(_mm256_set1_ps(-1.0f),
                              _mm256_min_ps(_mm256_set1_ps(1.0f), inner));

            __m256 result = _mm256_mul_ps(half, x);
            result = _mm256_mul_ps(result, _mm256_add_ps(one, tanh_val));

            _mm256_storeu_ps(&data[i], result);
        }
        for (; i < size; ++i) {
            data[i] = scalar::gelu(data[i]);
        }
    }

#endif // __AVX2__

} // namespace avx2

/**
 * Unified Activation Function Interface
 * Automatically selects optimal implementation based on CPU capabilities
 */
class ActivationFunction {
private:
    InstructionSet isa_;
    ActivationType type_;
    float alpha_; // For Leaky ReLU and ELU

public:
    ActivationFunction(ActivationType type = ActivationType::ReLU, float alpha = 0.01f)
        : isa_(detectInstructionSet()), type_(type), alpha_(alpha) {}

    /**
     * Apply activation function in-place to array
     */
    auto apply(std::span<float> data) -> void {
        apply(data.data(), data.size());
    }

    /**
     * Apply activation function in-place to raw pointer
     */
    auto apply(float* data, int size) -> void {
        switch (type_) {
            case ActivationType::ReLU:
                applyReLU(data, size);
                break;
            case ActivationType::LeakyReLU:
                applyLeakyReLU(data, size);
                break;
            case ActivationType::Sigmoid:
                applySigmoid(data, size);
                break;
            case ActivationType::Tanh:
                applyTanh(data, size);
                break;
            case ActivationType::GELU:
                applyGELU(data, size);
                break;
            case ActivationType::Swish:
                applySwish(data, size);
                break;
            case ActivationType::ELU:
                applyELU(data, size);
                break;
            case ActivationType::Softmax:
                applySoftmax(data, size);
                break;
            case ActivationType::Linear:
                // No-op for linear activation
                break;
        }
    }

    /**
     * Get activation type name
     */
    [[nodiscard]] auto name() const -> const char* {
        switch (type_) {
            case ActivationType::ReLU: return "ReLU";
            case ActivationType::LeakyReLU: return "Leaky ReLU";
            case ActivationType::Sigmoid: return "Sigmoid";
            case ActivationType::Tanh: return "Tanh";
            case ActivationType::GELU: return "GELU";
            case ActivationType::Swish: return "Swish";
            case ActivationType::ELU: return "ELU";
            case ActivationType::Softmax: return "Softmax";
            case ActivationType::Linear: return "Linear";
            default: return "Unknown";
        }
    }

private:
    auto applyReLU(float* data, int size) -> void {
#ifdef __AVX512F__
        if (isa_ == InstructionSet::AVX512) {
            avx512::relu(data, size);
            return;
        }
#endif
#ifdef __AVX2__
        if (isa_ == InstructionSet::AVX2) {
            avx2::relu(data, size);
            return;
        }
#endif
        // Scalar fallback
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::relu(data[i]);
        }
    }

    auto applyLeakyReLU(float* data, int size) -> void {
#ifdef __AVX512F__
        if (isa_ == InstructionSet::AVX512) {
            avx512::leaky_relu(data, size, alpha_);
            return;
        }
#endif
#ifdef __AVX2__
        if (isa_ == InstructionSet::AVX2) {
            avx2::leaky_relu(data, size, alpha_);
            return;
        }
#endif
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::leaky_relu(data[i], alpha_);
        }
    }

    auto applySigmoid(float* data, int size) -> void {
#ifdef __AVX512F__
        if (isa_ == InstructionSet::AVX512) {
            avx512::sigmoid(data, size);
            return;
        }
#endif
#ifdef __AVX2__
        if (isa_ == InstructionSet::AVX2) {
            avx2::sigmoid(data, size);
            return;
        }
#endif
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::sigmoid(data[i]);
        }
    }

    auto applyTanh(float* data, int size) -> void {
        // Use standard library tanh (often vectorized by compiler)
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::tanh_activation(data[i]);
        }
    }

    auto applyGELU(float* data, int size) -> void {
#ifdef __AVX512F__
        if (isa_ == InstructionSet::AVX512) {
            avx512::gelu(data, size);
            return;
        }
#endif
#ifdef __AVX2__
        if (isa_ == InstructionSet::AVX2) {
            avx2::gelu(data, size);
            return;
        }
#endif
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::gelu(data[i]);
        }
    }

    auto applySwish(float* data, int size) -> void {
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::swish(data[i]);
        }
    }

    auto applyELU(float* data, int size) -> void {
        for (int i = 0; i < size; ++i) {
            data[i] = scalar::elu(data[i], alpha_);
        }
    }

    auto applySoftmax(float* data, int size) -> void {
        // Find max for numerical stability
        float max_val = data[0];
        for (int i = 1; i < size; ++i) {
            max_val = std::max(max_val, data[i]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            data[i] = std::exp(data[i] - max_val);
            sum += data[i];
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; ++i) {
            data[i] *= inv_sum;
        }
    }
};

/**
 * Convenience functions for common activations
 */
inline auto relu(std::span<float> data) -> void {
    ActivationFunction(ActivationType::ReLU).apply(data);
}

inline auto leaky_relu(std::span<float> data, float alpha = 0.01f) -> void {
    ActivationFunction(ActivationType::LeakyReLU, alpha).apply(data);
}

inline auto sigmoid(std::span<float> data) -> void {
    ActivationFunction(ActivationType::Sigmoid).apply(data);
}

inline auto tanh_activation(std::span<float> data) -> void {
    ActivationFunction(ActivationType::Tanh).apply(data);
}

inline auto gelu(std::span<float> data) -> void {
    ActivationFunction(ActivationType::GELU).apply(data);
}

inline auto swish(std::span<float> data) -> void {
    ActivationFunction(ActivationType::Swish).apply(data);
}

inline auto elu(std::span<float> data, float alpha = 1.0f) -> void {
    ActivationFunction(ActivationType::ELU, alpha).apply(data);
}

inline auto softmax(std::span<float> data) -> void {
    ActivationFunction(ActivationType::Softmax).apply(data);
}

} // namespace activations
} // namespace bigbrother::ml
