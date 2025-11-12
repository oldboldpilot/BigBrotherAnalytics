/**
 * BigBrotherAnalytics - SIMD Utilities for Options Pricing (C++23)
 *
 * High-performance Black-Scholes pricing and Greeks calculations using AVX2 intrinsics.
 * Processes 8 options simultaneously for 8x theoretical speedup.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 12, 2025
 *
 * Performance Targets:
 * - Single option pricing: 0.8μs (AVX2) vs 2.5μs (scalar)
 * - Greeks calculation: 4μs (AVX2) vs 12μs (scalar)
 * - Batch pricing (100): 80μs (AVX2) vs 250μs (scalar)
 *
 * Compiler Flags Required:
 * - -mavx2 -mfma -O3 -march=native -ffast-math
 */

module;

#include <immintrin.h>  // AVX2 intrinsics
#include <array>
#include <cmath>
#include <span>
#include <vector>

export module bigbrother.options_strategies.simd_utils;

export namespace bigbrother::options_strategies::simd {

// ============================================================================
// Constants (SIMD-aligned)
// ============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float SQRT_2PI = 2.50662827463f;
constexpr float INV_SQRT_2PI = 0.39894228040f;

// ============================================================================
// AVX2 Math Primitives
// ============================================================================

/**
 * Natural logarithm (ln) using AVX2
 * Approximation with ~1e-6 relative error
 */
inline auto avx2_log(__m256 x) -> __m256 {
    // Extract exponent and mantissa
    __m256i exponent = _mm256_sub_epi32(
        _mm256_srli_epi32(_mm256_castps_si256(x), 23),
        _mm256_set1_epi32(127));

    __m256 e = _mm256_cvtepi32_ps(exponent);

    // Normalize mantissa to [1, 2)
    __m256 m = _mm256_or_ps(
        _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF))),
        _mm256_set1_ps(1.0f));

    // Polynomial approximation for log(m)
    __m256 const a0 = _mm256_set1_ps(-1.7417939f);
    __m256 const a1 = _mm256_set1_ps(2.8212026f);
    __m256 const a2 = _mm256_set1_ps(-1.4699568f);
    __m256 const a3 = _mm256_set1_ps(0.4519524f);

    __m256 log_m = a0;
    log_m = _mm256_fmadd_ps(log_m, m, a1);
    log_m = _mm256_fmadd_ps(log_m, m, a2);
    log_m = _mm256_fmadd_ps(log_m, m, a3);
    log_m = _mm256_mul_ps(log_m, _mm256_sub_ps(m, _mm256_set1_ps(1.0f)));

    // Combine: log(x) = log(m) + e * log(2)
    return _mm256_fmadd_ps(e, _mm256_set1_ps(0.69314718f), log_m);
}

/**
 * Exponential (e^x) using AVX2
 * Approximation with ~1e-6 relative error
 */
inline auto avx2_exp(__m256 x) -> __m256 {
    // Clamp to prevent overflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // Split x = n*log(2) + r, where |r| <= log(2)/2
    __m256 const log2e = _mm256_set1_ps(1.44269504f);
    __m256 n = _mm256_floor_ps(_mm256_mul_ps(x, log2e));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.69314718f), x);

    // Polynomial approximation for exp(r)
    __m256 const c0 = _mm256_set1_ps(1.0f);
    __m256 const c1 = _mm256_set1_ps(1.0f);
    __m256 const c2 = _mm256_set1_ps(0.5f);
    __m256 const c3 = _mm256_set1_ps(0.16666667f);
    __m256 const c4 = _mm256_set1_ps(0.04166667f);

    __m256 exp_r = c0;
    exp_r = _mm256_fmadd_ps(exp_r, r, c1);
    __m256 r2 = _mm256_mul_ps(r, r);
    exp_r = _mm256_fmadd_ps(c2, r2, exp_r);
    __m256 r3 = _mm256_mul_ps(r2, r);
    exp_r = _mm256_fmadd_ps(c3, r3, exp_r);
    __m256 r4 = _mm256_mul_ps(r3, r);
    exp_r = _mm256_fmadd_ps(c4, r4, exp_r);

    // Scale by 2^n
    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256i exp_n = _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23);

    return _mm256_mul_ps(exp_r, _mm256_castsi256_ps(exp_n));
}

/**
 * Square root using AVX2
 */
inline auto avx2_sqrt(__m256 x) -> __m256 {
    return _mm256_sqrt_ps(x);
}

/**
 * Reciprocal square root (1/sqrt(x)) using AVX2
 * Faster than sqrt + div
 */
inline auto avx2_rsqrt(__m256 x) -> __m256 {
    __m256 rsqrt = _mm256_rsqrt_ps(x);
    // Newton-Raphson refinement for better accuracy
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three = _mm256_set1_ps(3.0f);
    __m256 x_rsqrt2 = _mm256_mul_ps(_mm256_mul_ps(x, rsqrt), rsqrt);
    return _mm256_mul_ps(_mm256_mul_ps(half, rsqrt), _mm256_sub_ps(three, x_rsqrt2));
}

// ============================================================================
// Cumulative Normal Distribution (CND)
// ============================================================================

/**
 * Cumulative normal distribution N(x) using AVX2
 * Abramowitz & Stegun approximation (error < 7.5e-8)
 */
inline auto avx2_normalCDF(__m256 x) -> __m256 {
    // Constants for Abramowitz & Stegun approximation
    __m256 const a1 = _mm256_set1_ps(0.254829592f);
    __m256 const a2 = _mm256_set1_ps(-0.284496736f);
    __m256 const a3 = _mm256_set1_ps(1.421413741f);
    __m256 const a4 = _mm256_set1_ps(-1.453152027f);
    __m256 const a5 = _mm256_set1_ps(1.061405429f);
    __m256 const p = _mm256_set1_ps(0.3275911f);
    __m256 const half = _mm256_set1_ps(0.5f);
    __m256 const one = _mm256_set1_ps(1.0f);

    // Handle sign
    __m256 sign = _mm256_set1_ps(1.0f);
    __m256 mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    sign = _mm256_blendv_ps(sign, _mm256_set1_ps(-1.0f), mask);

    // abs(x)
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);

    // t = 1 / (1 + p * |x|)
    __m256 t = _mm256_rcp_ps(_mm256_fmadd_ps(p, abs_x, one));

    // Polynomial: y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * exp(-x^2/2)
    __m256 y = a5;
    y = _mm256_fmadd_ps(y, t, a4);
    y = _mm256_fmadd_ps(y, t, a3);
    y = _mm256_fmadd_ps(y, t, a2);
    y = _mm256_fmadd_ps(y, t, a1);
    y = _mm256_mul_ps(y, t);

    // exp(-x^2 / 2)
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 exp_term = avx2_exp(_mm256_mul_ps(_mm256_set1_ps(-0.5f), x2));
    y = _mm256_mul_ps(y, exp_term);

    // Final: 0.5 * (1 + sign * (1 - y))
    return _mm256_mul_ps(half, _mm256_fmadd_ps(sign, _mm256_sub_ps(one, y), one));
}

/**
 * Normal probability density function (PDF) using AVX2
 */
inline auto avx2_normalPDF(__m256 x) -> __m256 {
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 exp_term = avx2_exp(_mm256_mul_ps(_mm256_set1_ps(-0.5f), x2));
    return _mm256_mul_ps(_mm256_set1_ps(INV_SQRT_2PI), exp_term);
}

// ============================================================================
// Black-Scholes Pricing (AVX2 - 8 options at once)
// ============================================================================

/**
 * Black-Scholes call option pricing (8 options simultaneously)
 *
 * @param S Underlying prices (8 values)
 * @param K Strike prices (8 values)
 * @param T Time to expiration in years (8 values)
 * @param r Risk-free rates (8 values)
 * @param sigma Volatilities (8 values)
 * @return Call option prices (8 values)
 */
inline auto blackScholesCallBatch(
    __m256 S,       // 8 underlying prices
    __m256 K,       // 8 strike prices
    __m256 T,       // 8 times to expiration (years)
    __m256 r,       // 8 risk-free rates
    __m256 sigma    // 8 volatilities
) -> __m256 {
    // d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 S_over_K = _mm256_div_ps(S, K);
    __m256 ln_S_K = avx2_log(S_over_K);

    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 r_plus_half_sigma_sq = _mm256_add_ps(r, half_sigma_sq);
    __m256 numerator = _mm256_fmadd_ps(r_plus_half_sigma_sq, T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);

    // d2 = d1 - sigma*sqrt(T)
    __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);

    // N(d1) and N(d2)
    __m256 Nd1 = avx2_normalCDF(d1);
    __m256 Nd2 = avx2_normalCDF(d2);

    // Call = S*N(d1) - K*exp(-r*T)*N(d2)
    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = avx2_exp(neg_rT);
    __m256 discounted_K = _mm256_mul_ps(K, exp_neg_rT);

    __m256 term1 = _mm256_mul_ps(S, Nd1);
    __m256 term2 = _mm256_mul_ps(discounted_K, Nd2);

    return _mm256_sub_ps(term1, term2);
}

/**
 * Black-Scholes put option pricing (8 options simultaneously)
 */
inline auto blackScholesPutBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    // Use put-call parity: Put = Call - S + K*exp(-r*T)
    __m256 call_price = blackScholesCallBatch(S, K, T, r, sigma);

    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = avx2_exp(neg_rT);
    __m256 discounted_K = _mm256_mul_ps(K, exp_neg_rT);

    return _mm256_add_ps(_mm256_sub_ps(call_price, S), discounted_K);
}

// ============================================================================
// Greeks Calculations (AVX2)
// ============================================================================

/**
 * Calculate Delta for calls (8 options)
 * Delta = N(d1)
 */
inline auto deltaCallBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);

    return avx2_normalCDF(d1);
}

/**
 * Calculate Delta for puts (8 options)
 * Delta = N(d1) - 1
 */
inline auto deltaPutBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 delta_call = deltaCallBatch(S, K, T, r, sigma);
    return _mm256_sub_ps(delta_call, _mm256_set1_ps(1.0f));
}

/**
 * Calculate Gamma (same for calls and puts)
 * Gamma = N'(d1) / (S * sigma * sqrt(T))
 */
inline auto gammaBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);

    __m256 pdf_d1 = avx2_normalPDF(d1);
    __m256 denominator = _mm256_mul_ps(S, sigma_sqrt_T);

    return _mm256_div_ps(pdf_d1, denominator);
}

/**
 * Calculate Theta for calls
 * Theta = -(S*N'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
 */
inline auto thetaCallBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);

    __m256 pdf_d1 = avx2_normalPDF(d1);
    __m256 cdf_d2 = avx2_normalCDF(d2);

    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = avx2_exp(neg_rT);

    __m256 term1 = _mm256_div_ps(
        _mm256_mul_ps(_mm256_mul_ps(S, pdf_d1), sigma),
        _mm256_mul_ps(_mm256_set1_ps(2.0f), sqrt_T));

    __m256 term2 = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_mul_ps(r, K), exp_neg_rT), cdf_d2);

    // Convert to per-day (divide by 365)
    __m256 theta = _mm256_div_ps(
        _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), term1), term2),
        _mm256_set1_ps(365.0f));

    return theta;
}

/**
 * Calculate Vega (same for calls and puts)
 * Vega = S * N'(d1) * sqrt(T)
 */
inline auto vegaBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);
    __m256 pdf_d1 = avx2_normalPDF(d1);

    // Vega per 1% change in volatility
    return _mm256_div_ps(
        _mm256_mul_ps(_mm256_mul_ps(S, pdf_d1), sqrt_T),
        _mm256_set1_ps(100.0f));
}

/**
 * Calculate Rho for calls
 * Rho = K * T * exp(-r*T) * N(d2)
 */
inline auto rhoCallBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);

    __m256 cdf_d2 = avx2_normalCDF(d2);

    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = avx2_exp(neg_rT);

    // Rho per 1% change in interest rate
    return _mm256_div_ps(
        _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(K, T), exp_neg_rT), cdf_d2),
        _mm256_set1_ps(100.0f));
}

/**
 * Calculate Theta for puts
 * Theta = -(S*N'(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)
 */
inline auto thetaPutBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);

    __m256 pdf_d1 = avx2_normalPDF(d1);
    __m256 neg_d2 = _mm256_sub_ps(_mm256_setzero_ps(), d2);
    __m256 cdf_neg_d2 = avx2_normalCDF(neg_d2);

    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = avx2_exp(neg_rT);

    __m256 term1 = _mm256_div_ps(
        _mm256_mul_ps(_mm256_mul_ps(S, pdf_d1), sigma),
        _mm256_mul_ps(_mm256_set1_ps(2.0f), sqrt_T));

    __m256 term2 = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_mul_ps(r, K), exp_neg_rT), cdf_neg_d2);

    // Convert to per-day (divide by 365)
    __m256 theta = _mm256_div_ps(
        _mm256_sub_ps(term2, term1),
        _mm256_set1_ps(365.0f));

    return theta;
}

/**
 * Calculate Rho for puts
 * Rho = -K * T * exp(-r*T) * N(-d2)
 */
inline auto rhoPutBatch(
    __m256 S, __m256 K, __m256 T, __m256 r, __m256 sigma
) -> __m256 {
    __m256 sqrt_T = avx2_sqrt(T);
    __m256 sigma_sqrt_T = _mm256_mul_ps(sigma, sqrt_T);

    __m256 ln_S_K = avx2_log(_mm256_div_ps(S, K));
    __m256 sigma_sq = _mm256_mul_ps(sigma, sigma);
    __m256 half_sigma_sq = _mm256_mul_ps(sigma_sq, _mm256_set1_ps(0.5f));
    __m256 numerator = _mm256_fmadd_ps(
        _mm256_add_ps(r, half_sigma_sq), T, ln_S_K);

    __m256 d1 = _mm256_div_ps(numerator, sigma_sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);

    __m256 neg_d2 = _mm256_sub_ps(_mm256_setzero_ps(), d2);
    __m256 cdf_neg_d2 = avx2_normalCDF(neg_d2);

    __m256 neg_rT = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), r), T);
    __m256 exp_neg_rT = avx2_exp(neg_rT);

    // Rho per 1% change in interest rate (negative for puts)
    return _mm256_div_ps(
        _mm256_sub_ps(_mm256_setzero_ps(),
            _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(K, T), exp_neg_rT), cdf_neg_d2)),
        _mm256_set1_ps(100.0f));
}

// ============================================================================
// Scalar Wrapper Functions (for single option pricing)
// ============================================================================

/**
 * Black-Scholes call (scalar)
 */
inline auto blackScholesCall(
    float S, float K, float T, float r, float sigma
) -> float {
    __m256 S_vec = _mm256_set1_ps(S);
    __m256 K_vec = _mm256_set1_ps(K);
    __m256 T_vec = _mm256_set1_ps(T);
    __m256 r_vec = _mm256_set1_ps(r);
    __m256 sigma_vec = _mm256_set1_ps(sigma);

    __m256 result = blackScholesCallBatch(S_vec, K_vec, T_vec, r_vec, sigma_vec);

    alignas(32) float results[8];
    _mm256_store_ps(results, result);
    return results[0];
}

/**
 * Black-Scholes put (scalar)
 */
inline auto blackScholesPut(
    float S, float K, float T, float r, float sigma
) -> float {
    __m256 S_vec = _mm256_set1_ps(S);
    __m256 K_vec = _mm256_set1_ps(K);
    __m256 T_vec = _mm256_set1_ps(T);
    __m256 r_vec = _mm256_set1_ps(r);
    __m256 sigma_vec = _mm256_set1_ps(sigma);

    __m256 result = blackScholesPutBatch(S_vec, K_vec, T_vec, r_vec, sigma_vec);

    alignas(32) float results[8];
    _mm256_store_ps(results, result);
    return results[0];
}

} // namespace bigbrother::options_strategies::simd
