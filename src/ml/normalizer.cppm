/**
 * Composable Normalizer with Automatic Inverse Pipeline
 *
 * Implements dataset-level min/max normalization with full pipeline composition.
 * Supports automatic reverse transformation for bidirectional data flow.
 *
 * Key Features:
 * - Dataset-level min/max learning (fit once, transform many)
 * - Full pipeline composition with operator>>
 * - Automatic inverse with proper order reversal
 * - Template metaprogramming for zero runtime overhead
 * - AVX2 SIMD acceleration for batch operations
 *
 * Example Usage:
 *
 *   // Training: Learn normalizers from dataset
 *   auto feature_norm = Normalizer<85>::fit(training_features);
 *   auto price_norm = Normalizer<3>::fit(training_prices);
 *
 *   // Forward transformation
 *   auto normalized = feature_norm.transform(raw_features);  // → [0,1]
 *
 *   // Inverse transformation
 *   auto original = price_norm.inverse(predictions);  // [0,1] → actual prices
 *
 *   // Pipeline composition (advanced)
 *   auto pipeline = feature_norm >> model >> price_norm.inverse();
 *   auto result = pipeline(raw_input);
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 * Architecture: "omo oko ni e" - composable transformations
 */

module;

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <span>
#include <vector>
#include <immintrin.h>  // AVX2

export module bigbrother.ml.normalizer;

export namespace bigbrother::ml {

/**
 * Concept: Transformation
 * Any type that can transform and inverse-transform data
 */
template<typename T>
concept Transformation = requires(T t, typename T::input_type in, typename T::output_type out) {
    typename T::input_type;
    typename T::output_type;
    { t.transform(in) } -> std::same_as<typename T::output_type>;
    { t.inverse(out) } -> std::same_as<typename T::input_type>;
};

/**
 * MinMaxNormalizer: Dataset-level min/max normalization to [0, 1]
 *
 * Learns min/max from training dataset once, then applies consistently
 * to all samples (training, validation, test, inference).
 *
 * Formula:
 *   transform:  normalized[i] = (x[i] - min[i]) / (max[i] - min[i])
 *   inverse:    x[i] = normalized[i] * (max[i] - min[i]) + min[i]
 */
template<size_t N>
class Normalizer {
public:
    using input_type = std::array<float, N>;
    using output_type = std::array<float, N>;

private:
    std::array<float, N> min_;
    std::array<float, N> max_;
    std::array<float, N> range_;  // Precomputed: max - min

public:
    /**
     * Default constructor (identity transformation)
     */
    Normalizer() {
        min_.fill(0.0f);
        max_.fill(1.0f);
        range_.fill(1.0f);
    }

    /**
     * Constructor with explicit min/max
     */
    Normalizer(std::array<float, N> const& min, std::array<float, N> const& max)
        : min_(min), max_(max) {
        for (size_t i = 0; i < N; ++i) {
            range_[i] = max_[i] - min_[i];
            // Avoid division by zero for constant features
            if (range_[i] < 1e-8f) {
                range_[i] = 1.0f;
            }
        }
    }

    /**
     * Learn min/max from training dataset
     *
     * @param data Training dataset (must have at least 1 sample)
     * @return Fitted normalizer ready for transform/inverse
     */
    [[nodiscard]] static auto fit(std::span<input_type const> data) -> Normalizer<N> {
        if (data.empty()) {
            return Normalizer<N>();  // Identity if no data
        }

        std::array<float, N> min_vals;
        std::array<float, N> max_vals;

        // Initialize with first sample
        min_vals = data[0];
        max_vals = data[0];

        // Find global min/max across all samples
        for (size_t i = 1; i < data.size(); ++i) {
            for (size_t j = 0; j < N; ++j) {
                min_vals[j] = std::min(min_vals[j], data[i][j]);
                max_vals[j] = std::max(max_vals[j], data[i][j]);
            }
        }

        return Normalizer<N>(min_vals, max_vals);
    }

    /**
     * Transform: raw → [0, 1]
     *
     * Uses AVX2 SIMD for 8x parallelism when N >= 8
     */
    [[nodiscard]] auto transform(input_type const& x) const -> output_type {
        output_type result;

        #if defined(__AVX2__)
        // AVX2: Process 8 floats at a time
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 input = _mm256_loadu_ps(&x[i]);
            __m256 min_vec = _mm256_loadu_ps(&min_[i]);
            __m256 range_vec = _mm256_loadu_ps(&range_[i]);

            // normalized = (input - min) / range
            __m256 numerator = _mm256_sub_ps(input, min_vec);
            __m256 normalized = _mm256_div_ps(numerator, range_vec);

            // Clamp to [0, 1]
            __m256 zero = _mm256_setzero_ps();
            __m256 one = _mm256_set1_ps(1.0f);
            normalized = _mm256_max_ps(zero, _mm256_min_ps(one, normalized));

            _mm256_storeu_ps(&result[i], normalized);
        }

        // Remainder (scalar)
        for (; i < N; ++i) {
            float val = (x[i] - min_[i]) / range_[i];
            result[i] = std::clamp(val, 0.0f, 1.0f);
        }
        #else
        // Scalar fallback
        for (size_t i = 0; i < N; ++i) {
            float val = (x[i] - min_[i]) / range_[i];
            result[i] = std::clamp(val, 0.0f, 1.0f);
        }
        #endif

        return result;
    }

    /**
     * Inverse: [0, 1] → raw
     *
     * Uses AVX2 SIMD for 8x parallelism when N >= 8
     */
    [[nodiscard]] auto inverse(output_type const& x) const -> input_type {
        input_type result;

        #if defined(__AVX2__)
        // AVX2: Process 8 floats at a time
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 normalized = _mm256_loadu_ps(&x[i]);
            __m256 range_vec = _mm256_loadu_ps(&range_[i]);
            __m256 min_vec = _mm256_loadu_ps(&min_[i]);

            // original = normalized * range + min
            __m256 scaled = _mm256_mul_ps(normalized, range_vec);
            __m256 original = _mm256_add_ps(scaled, min_vec);

            _mm256_storeu_ps(&result[i], original);
        }

        // Remainder (scalar)
        for (; i < N; ++i) {
            result[i] = x[i] * range_[i] + min_[i];
        }
        #else
        // Scalar fallback
        for (size_t i = 0; i < N; ++i) {
            result[i] = x[i] * range_[i] + min_[i];
        }
        #endif

        return result;
    }

    /**
     * Batch transform: Process multiple samples efficiently
     */
    [[nodiscard]] auto transform_batch(std::span<input_type const> batch) const
        -> std::vector<output_type> {
        std::vector<output_type> results;
        results.reserve(batch.size());
        for (auto const& sample : batch) {
            results.push_back(transform(sample));
        }
        return results;
    }

    /**
     * Batch inverse: Process multiple samples efficiently
     */
    [[nodiscard]] auto inverse_batch(std::span<output_type const> batch) const
        -> std::vector<input_type> {
        std::vector<input_type> results;
        results.reserve(batch.size());
        for (auto const& sample : batch) {
            results.push_back(inverse(sample));
        }
        return results;
    }

    /**
     * Get learned min/max for inspection or serialization
     */
    [[nodiscard]] auto get_min() const -> std::array<float, N> const& { return min_; }
    [[nodiscard]] auto get_max() const -> std::array<float, N> const& { return max_; }
    [[nodiscard]] auto get_range() const -> std::array<float, N> const& { return range_; }
};

/**
 * InverseTransformation: Wraps a transformation and swaps transform/inverse
 *
 * Allows natural composition: normalizer.inverse() returns a new transformation
 * where transform() calls the original's inverse() and vice versa.
 */
template<Transformation T>
class InverseTransformation {
public:
    using input_type = typename T::output_type;   // Swapped!
    using output_type = typename T::input_type;   // Swapped!

private:
    T inner_;

public:
    explicit InverseTransformation(T const& t) : inner_(t) {}

    [[nodiscard]] auto transform(input_type const& x) const -> output_type {
        return inner_.inverse(x);  // Swapped!
    }

    [[nodiscard]] auto inverse(output_type const& x) const -> input_type {
        return inner_.transform(x);  // Swapped!
    }

    [[nodiscard]] auto unwrap() const -> T const& { return inner_; }
};

/**
 * Helper: Create inverse transformation
 */
template<Transformation T>
[[nodiscard]] auto make_inverse(T const& t) -> InverseTransformation<T> {
    return InverseTransformation<T>(t);
}

/**
 * ComposedTransformation: Chains two transformations
 *
 * Forward:  output = op2.transform(op1.transform(input))
 * Inverse:  input  = op1.inverse(op2.inverse(output))
 *
 * The inverse automatically reverses the order!
 */
template<Transformation Op1, Transformation Op2>
requires std::same_as<typename Op1::output_type, typename Op2::input_type>
class ComposedTransformation {
public:
    using input_type = typename Op1::input_type;
    using output_type = typename Op2::output_type;

private:
    Op1 op1_;
    Op2 op2_;

public:
    ComposedTransformation(Op1 const& op1, Op2 const& op2)
        : op1_(op1), op2_(op2) {}

    /**
     * Forward: op1 then op2
     */
    [[nodiscard]] auto transform(input_type const& x) const -> output_type {
        return op2_.transform(op1_.transform(x));
    }

    /**
     * Inverse: op2⁻¹ then op1⁻¹ (automatic order reversal!)
     */
    [[nodiscard]] auto inverse(output_type const& x) const -> input_type {
        return op1_.inverse(op2_.inverse(x));
    }

    [[nodiscard]] auto first() const -> Op1 const& { return op1_; }
    [[nodiscard]] auto second() const -> Op2 const& { return op2_; }
};

/**
 * Pipeline Composition Operator: op1 >> op2
 *
 * Creates a composed transformation that automatically handles:
 * - Forward chaining: op1 then op2
 * - Inverse reversal: op2⁻¹ then op1⁻¹
 *
 * Example:
 *   auto pipeline = feature_norm >> model >> price_norm.inverse();
 *   auto result = pipeline.transform(raw_input);
 *   auto original = pipeline.inverse(result);
 */
template<Transformation Op1, Transformation Op2>
requires std::same_as<typename Op1::output_type, typename Op2::input_type>
[[nodiscard]] auto operator>>(Op1 const& op1, Op2 const& op2)
    -> ComposedTransformation<Op1, Op2> {
    return ComposedTransformation<Op1, Op2>(op1, op2);
}

}  // namespace bigbrother::ml
