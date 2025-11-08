/**
 * BigBrotherAnalytics - Risk-Free Rate Constants Module (C++23)
 *
 * Auto-generated: 2025-11-07 17:32:56.528381
 * Risk-free rate from FRED as of 2025-11-06
 * Data source: https://fred.stlouisfed.org/series/DGS10
 *
 * Following C++ Core Guidelines:
 * - Con.1: By default, make objects immutable
 * - F.4: Use constexpr for compile-time evaluation
 * - ES.10: Declare one name per declaration
 */

// Global module fragment
module;

// Module declaration
export module bigbrother.utils.risk_free_rate;

export namespace bigbrother::constants {

// ============================================================================
// Risk-Free Rates from U.S. Treasury
// ============================================================================

// 10-Year Treasury Rate (updated 2025-11-06)
constexpr double RISK_FREE_RATE = 0.041100;  // 4.110%

// For different option expirations, use appropriate tenor:
constexpr double RISK_FREE_RATE_3M = 0.041100;   // Update with DTB3
constexpr double RISK_FREE_RATE_1Y = 0.041100;   // Update with DGS1
constexpr double RISK_FREE_RATE_10Y = 0.041100;  // DGS10

// Last updated
constexpr const char* RATE_DATE = "2025-11-06";

// ============================================================================
// Helper Functions (C++ Core Guidelines: F.4 Use constexpr)
// ============================================================================

/**
 * Get risk-free rate for given time to expiration
 * 
 * F.15: Prefer simple and conventional ways
 * F.4: constexpr for compile-time evaluation
 * F.6: noexcept - no exceptions
 */
[[nodiscard]] constexpr auto getRiskFreeRate(double years_to_expiration) noexcept -> double {
    if (years_to_expiration <= 0.25) {
        return RISK_FREE_RATE_3M;
    } else if (years_to_expiration <= 1.0) {
        return RISK_FREE_RATE_1Y;
    } else {
        return RISK_FREE_RATE_10Y;
    }
}

/**
 * Get rate update date
 * 
 * F.6: noexcept - no exceptions
 */
[[nodiscard]] constexpr auto getRateDate() noexcept -> const char* {
    return RATE_DATE;
}

} // export namespace bigbrother::constants
