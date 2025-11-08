/**
 * Trinomial Tree Options Pricing Model - C++23 Module
 *
 * DEFAULT pricing method for BigBrotherAnalytics.
 *
 * Advantages over Black-Scholes:
 * - Handles American options (early exercise)
 * - Better convergence than binomial (fewer steps needed)
 * - More stable for barrier options
 * - Handles discrete dividends naturally
 *
 * Advantages over Binomial:
 * - Faster convergence (N trinomial ≈ N² binomial accuracy)
 * - More stable numerical behavior
 * - Better for path-dependent options
 *
 * Performance Target: < 10μs for 100 steps, < 100μs for 1000 steps
 *
 * Following C++ Core Guidelines:
 * - F.4: constexpr for compile-time evaluation
 * - F.6: noexcept where applicable
 * - F.16: Pass by value (cheap) or const& (expensive)
 * - F.20: Return values via std::expected
 * - P.5: Prefer compile-time to run-time checking
 * - R.1: RAII for resource management
 */

// Global module fragment
module;

#include <vector>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <expected>

// Module declaration
export module bigbrother.pricing.trinomial_tree;

export namespace bigbrother::options {

/**
 * Trinomial Tree Pricer - Fluent Builder API
 *
 * Following PRD requirement for fluent composable APIs.
 * Builder pattern for ergonomic options pricing.
 *
 * Usage (Fluent API):
 *   auto price = TrinomialPricer::forStock(100.0)
 *       .strike(105.0)
 *       .expiration(0.25)  // 3 months
 *       .volatility(0.30)
 *       .riskFreeRate(0.04)
 *       .asCall()
 *       .american()
 *       .withSteps(200)
 *       .price();
 *
 * Or using params (traditional):
 *   auto price = TrinomialTreeModel::price(S, K, r, T, sigma, q, is_call, is_american, steps);
 */

/**
 * Trinomial Tree Pricing Builder (Fluent API)
 *
 * F.20: Each method returns *this for chaining
 * C.2: Private data with public fluent interface
 */
class TrinomialPricer {
public:
    /**
     * Start fluent chain with stock price
     * F.20: Return builder by value
     */
    [[nodiscard]] static auto forStock(double S) -> TrinomialPricer {
        TrinomialPricer builder;
        builder.S_ = S;
        return builder;
    }

    /**
     * Set strike price (fluent)
     */
    [[nodiscard]] auto strike(double K) -> TrinomialPricer& {
        K_ = K;
        return *this;
    }

    /**
     * Set time to expiration in years (fluent)
     */
    [[nodiscard]] auto expiration(double T) -> TrinomialPricer& {
        T_ = T;
        return *this;
    }

    /**
     * Set volatility (fluent)
     */
    [[nodiscard]] auto volatility(double sigma) -> TrinomialPricer& {
        sigma_ = sigma;
        return *this;
    }

    /**
     * Set risk-free rate (fluent)
     */
    [[nodiscard]] auto riskFreeRate(double r) -> TrinomialPricer& {
        r_ = r;
        return *this;
    }

    /**
     * Set dividend yield (fluent)
     */
    [[nodiscard]] auto dividendYield(double q) -> TrinomialPricer& {
        q_ = q;
        return *this;
    }

    /**
     * Set as call option (fluent)
     */
    [[nodiscard]] auto asCall() -> TrinomialPricer& {
        is_call_ = true;
        return *this;
    }

    /**
     * Set as put option (fluent)
     */
    [[nodiscard]] auto asPut() -> TrinomialPricer& {
        is_call_ = false;
        return *this;
    }

    /**
     * Set American exercise (fluent)
     */
    [[nodiscard]] auto american() -> TrinomialPricer& {
        is_american_ = true;
        return *this;
    }

    /**
     * Set European exercise (fluent)
     */
    [[nodiscard]] auto european() -> TrinomialPricer& {
        is_american_ = false;
        return *this;
    }

    /**
     * Set number of time steps (fluent)
     */
    [[nodiscard]] auto withSteps(int steps) -> TrinomialPricer& {
        steps_ = steps;
        return *this;
    }

    /**
     * Execute pricing (terminal method)
     * F.20: Return expected<double> for error handling
     */
    [[nodiscard]] auto price() const -> std::expected<double, std::string>;

    /**
     * Calculate Greeks (terminal method)
     */
    [[nodiscard]] auto greeks() const -> std::expected<Greeks, std::string>;

private:
    double S_{100.0};
    double K_{100.0};
    double r_{0.041};  // Default from current FRED rate
    double T_{0.25};
    double sigma_{0.30};
    double q_{0.0};
    bool is_call_{true};
    bool is_american_{true};
    int steps_{100};
};

/**
 * Traditional Trinomial Tree Model (for backward compatibility)
 */
class TrinomialTreeModel {
public:
    /**
     * Price European or American option using trinomial tree
     *
     * F.20: Return std::expected for error handling
     * F.16: Pass params by const& (potentially expensive)
     *
     * @param S Stock price
     * @param K Strike price
     * @param r Risk-free rate
     * @param T Time to expiration (years)
     * @param sigma Volatility (annual)
     * @param q Dividend yield
     * @param is_call true for call, false for put
     * @param is_american true for American, false for European
     * @param steps Number of time steps (default: 100)
     * @return Option price or error
     */
    [[nodiscard]] static auto price(
        double S,
        double K,
        double r,
        double T,
        double sigma,
        double q,
        bool is_call,
        bool is_american = true,
        int steps = 100
    ) -> std::expected<double, std::string> {

        // F.6: Input validation
        if (S <= 0.0) return std::unexpected("Stock price must be positive");
        if (K <= 0.0) return std::unexpected("Strike must be positive");
        if (T < 0.0) return std::unexpected("Time cannot be negative");
        if (sigma < 0.0) return std::unexpected("Volatility cannot be negative");
        if (steps < 1) return std::unexpected("Steps must be at least 1");

        // Handle edge case: expired option
        if (T == 0.0) {
            return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        }

        // Calculate time step
        double const dt = T / static_cast<double>(steps);

        // Trinomial parameters (Boyle parameterization)
        constexpr double lambda = std::numbers::sqrt3 / std::numbers::sqrt2;  // √(3/2)

        double const up = std::exp(lambda * sigma * std::sqrt(dt));
        double const down = 1.0 / up;  // down = e^(-λσ√Δt)

        // Risk-neutral probabilities
        double const dx = std::exp((r - q) * dt);
        double const dx_sq = dx * dx;

        double const p_u = ((dx_sq - dx * down) / (up - down) -
                           (dx - down) / (up - down)) / (up + down - 2 * down);
        double const p_d = ((up * dx - dx_sq) / (up - down) -
                           (up - dx) / (up - down)) / (up + down - 2 * down);
        double const p_m = 1.0 - p_u - p_d;

        // Validate probabilities (P.5: Compile-time checking where possible)
        if (p_u < 0.0 || p_u > 1.0 || p_d < 0.0 || p_d > 1.0 || p_m < 0.0 || p_m > 1.0) {
            return std::unexpected("Invalid trinomial probabilities - reduce time step");
        }

        // Build price tree (R.1: RAII with std::vector)
        // Using vector instead of raw arrays for safety
        std::vector<double> prices(2 * steps + 1);

        // Initialize terminal stock prices
        for (int i = -steps; i <= steps; ++i) {
            int const idx = i + steps;
            double const power = static_cast<double>(i);
            prices[idx] = S * std::pow(up, power);
        }

        // Initialize option values at expiration
        std::vector<double> values(2 * steps + 1);

        for (int i = -steps; i <= steps; ++i) {
            int const idx = i + steps;
            double const stock_price = prices[idx];

            if (is_call) {
                values[idx] = std::max(0.0, stock_price - K);
            } else {
                values[idx] = std::max(0.0, K - stock_price);
            }
        }

        // Backward induction through tree
        double const discount = std::exp(-r * dt);

        for (int step = steps - 1; step >= 0; --step) {
            for (int i = -step; i <= step; ++i) {
                int const idx = i + steps;

                // Calculate continuation value
                // Value = e^(-r*dt) * [p_u*V_up + p_m*V_mid + p_d*V_down]
                double const continuation_value = discount * (
                    p_u * values[idx + 1] +
                    p_m * values[idx] +
                    p_d * values[idx - 1]
                );

                // For American options, check early exercise
                if (is_american) {
                    double const stock_price = prices[idx];
                    double const intrinsic_value = is_call ?
                        std::max(0.0, stock_price - K) :
                        std::max(0.0, K - stock_price);

                    values[idx] = std::max(continuation_value, intrinsic_value);
                } else {
                    values[idx] = continuation_value;
                }
            }
        }

        return values[steps];  // Option value at root (current time)
    }

    /**
     * Calculate Greeks using trinomial tree
     *
     * F.20: Return struct by value (move semantics)
     */
    [[nodiscard]] static auto greeks(
        double S,
        double K,
        double r,
        double T,
        double sigma,
        double q,
        bool is_call,
        bool is_american = true,
        int steps = 100
    ) -> std::expected<Greeks, std::string> {

        // Price at current spot
        auto price_result = price(S, K, r, T, sigma, q, is_call, is_american, steps);
        if (!price_result) return std::unexpected(price_result.error());

        double const V = *price_result;

        // Delta: ∂V/∂S (using finite difference)
        double const dS = S * 0.01;  // 1% bump
        auto price_up = price(S + dS, K, r, T, sigma, q, is_call, is_american, steps);
        auto price_down = price(S - dS, K, r, T, sigma, q, is_call, is_american, steps);

        if (!price_up || !price_down) {
            return std::unexpected("Failed to calculate delta");
        }

        double const delta = (*price_up - *price_down) / (2.0 * dS);

        // Gamma: ∂²V/∂S²
        double const gamma = (*price_up - 2.0 * V + *price_down) / (dS * dS);

        // Theta: ∂V/∂t (use smaller time step)
        double const dT = std::min(T, 1.0 / 365.0);  // 1 day
        auto price_later = price(S, K, r, T - dT, sigma, q, is_call, is_american, steps);

        if (!price_later) {
            return std::unexpected("Failed to calculate theta");
        }

        double const theta = (*price_later - V) / dT * (1.0 / 365.0);  // Per day

        // Vega: ∂V/∂σ (per 1% change)
        double const dsigma = 0.01;
        auto price_vol_up = price(S, K, r, T, sigma + dsigma, q, is_call, is_american, steps);

        if (!price_vol_up) {
            return std::unexpected("Failed to calculate vega");
        }

        double const vega = (*price_vol_up - V) / dsigma / 100.0;  // Per 1%

        // Rho: ∂V/∂r (per 1% change)
        double const dr = 0.01;
        auto price_rate_up = price(S, K, r + dr, T, sigma, q, is_call, is_american, steps);

        if (!price_rate_up) {
            return std::unexpected("Failed to calculate rho");
        }

        double const rho = (*price_rate_up - V) / dr / 100.0;  // Per 1%

        return Greeks{delta, gamma, theta, vega, rho};
    }

    /**
     * Optimize number of steps for target accuracy
     *
     * F.20: Return optimal step count
     * F.4: Can be constexpr in future
     */
    [[nodiscard]] static auto optimizeSteps(
        double T,
        double target_accuracy = 0.01
    ) noexcept -> int {
        // Rule of thumb: steps = max(50, 50 * T)
        // More steps for longer expiration
        int steps = static_cast<int>(50.0 * std::max(1.0, T));

        // Clamp to reasonable range
        return std::clamp(steps, 50, 1000);
    }
};

/**
 * Greeks structure
 *
 * C.1: struct for passive data
 * Defined here to avoid circular dependency
 */
struct Greeks {
    double delta{0.0};
    double gamma{0.0};
    double theta{0.0};
    double vega{0.0};
    double rho{0.0};

    [[nodiscard]] constexpr auto isValid() const noexcept -> bool {
        return !std::isnan(delta) && !std::isnan(gamma) &&
               !std::isnan(theta) && !std::isnan(vega) &&
               !std::isnan(rho);
    }
};

} // export namespace bigbrother::options
