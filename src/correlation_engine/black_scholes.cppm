/**
 * Black-Scholes Options Pricing Model - C++23 Module
 *
 * Implementation of the Black-Scholes-Merton model for European options pricing.
 * Performance target: < 1μs per option pricing
 *
 * References:
 * - Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
 * - Hull: "Options, Futures, and Other Derivatives" (11th ed)
 */

// Global module fragment for standard library
module;

#include <cmath>
#include <numbers>

// Module declaration
export module bigbrother.pricing.black_scholes;

export namespace bigbrother::options {

/**
 * Black-Scholes Pricing Model
 *
 * Implements European option pricing with dividends.
 * Uses C++23 trailing return syntax and [[nodiscard]] attributes.
 */
class BlackScholesModel {
public:
    /**
     * Calculate European call option price
     *
     * @param S Stock price
     * @param K Strike price
     * @param r Risk-free rate (annual, e.g., 0.05 for 5%)
     * @param T Time to expiration (years, e.g., 0.25 for 3 months)
     * @param sigma Volatility (annual, e.g., 0.30 for 30%)
     * @param q Dividend yield (annual, e.g., 0.02 for 2%)
     * @return Call option price
     */
    [[nodiscard]] static auto callPrice(
        double S,
        double K,
        double r,
        double T,
        double sigma,
        double q = 0.0
    ) -> double {
        if (T <= 0.0) return std::max(S - K, 0.0);  // Expired option
        if (sigma <= 0.0) return std::max(S - K, 0.0);  // Zero vol edge case

        auto const d1 = calculateD1(S, K, r, T, sigma, q);
        auto const d2 = d1 - sigma * std::sqrt(T);

        // C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        auto const call = S * std::exp(-q * T) * normalCDF(d1) -
                          K * std::exp(-r * T) * normalCDF(d2);

        return call;
    }

    /**
     * Calculate European put option price
     *
     * Uses put-call parity for efficiency:
     * P = C - S*e^(-qT) + K*e^(-rT)
     */
    [[nodiscard]] static auto putPrice(
        double S,
        double K,
        double r,
        double T,
        double sigma,
        double q = 0.0
    ) -> double {
        if (T <= 0.0) return std::max(K - S, 0.0);  // Expired option

        auto const call = callPrice(S, K, r, T, sigma, q);

        // Put-call parity
        auto const put = call - S * std::exp(-q * T) + K * std::exp(-r * T);

        return put;
    }

    /**
     * Calculate d1 parameter for Black-Scholes
     *
     * d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
     */
    [[nodiscard]] static auto calculateD1(
        double S,
        double K,
        double r,
        double T,
        double sigma,
        double q = 0.0
    ) -> double {
        auto const numerator = std::log(S / K) +
                               (r - q + 0.5 * sigma * sigma) * T;
        auto const denominator = sigma * std::sqrt(T);

        return numerator / denominator;
    }

    /**
     * Standard normal cumulative distribution function
     *
     * Uses error function for accuracy:
     * N(x) = 0.5 * [1 + erf(x/√2)]
     *
     * Accuracy: < 10^-7 (sufficient for options pricing)
     */
    [[nodiscard]] static auto normalCDF(double x) -> double {
        // Using std::erf from <cmath>
        // N(x) = Φ(x) = 0.5 * [1 + erf(x / sqrt(2))]
        return 0.5 * (1.0 + std::erf(x / std::numbers::sqrt2));
    }

    /**
     * Standard normal probability density function
     *
     * N'(x) = (1/√(2π)) * e^(-x²/2)
     *
     * Used for Greeks calculations
     */
    [[nodiscard]] static auto normalPDF(double x) -> double {
        constexpr double inv_sqrt_2pi = 0.3989422804014327;  // 1/√(2π)
        return inv_sqrt_2pi * std::exp(-0.5 * x * x);
    }

    /**
     * Validate pricing parameters
     *
     * @return true if parameters are valid for pricing
     */
    [[nodiscard]] static auto validateParams(
        double S,
        double K,
        double r,
        double T,
        double sigma
    ) -> bool {
        return S > 0.0 && K > 0.0 && T >= 0.0 && sigma >= 0.0;
    }
};

} // export namespace bigbrother::options
