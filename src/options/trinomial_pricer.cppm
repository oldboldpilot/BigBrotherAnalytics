/**
 * @file trinomial_pricer.cppm
 * @brief Trinomial Tree Option Pricing with Greeks Calculation
 *
 * Implements trinomial tree model for American and European options pricing
 * with OpenMP parallelization for performance optimization.
 *
 * Greeks calculated:
 * - Delta: ∂V/∂S (rate of change of option value with respect to underlying price)
 * - Gamma: ∂²V/∂S² (rate of change of delta)
 * - Theta: ∂V/∂t (time decay)
 * - Vega: ∂V/∂σ (sensitivity to volatility)
 * - Rho: ∂V/∂r (sensitivity to risk-free rate)
 *
 * Trinomial tree advantages over binomial:
 * - Better convergence for American options
 * - More accurate Greeks
 * - Handles early exercise more naturally
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 */

module;

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

// SIMD intrinsics for vectorization
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h> // AVX/AVX2
    #define SIMD_AVAILABLE 1
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define SIMD_AVAILABLE 1
#else
    #define SIMD_AVAILABLE 0
#endif

export module options:trinomial_pricer;

export namespace options {

/**
 * @brief Option type enumeration
 */
enum class OptionType { CALL, PUT };

/**
 * @brief Option style enumeration
 */
enum class OptionStyle {
    EUROPEAN, // Can only be exercised at expiration
    AMERICAN  // Can be exercised at any time
};

/**
 * @brief Greeks structure containing all option sensitivities
 */
struct Greeks {
    double delta; // ∂V/∂S
    double gamma; // ∂²V/∂S²
    double theta; // ∂V/∂t (per day)
    double vega;  // ∂V/∂σ (per 1% change in volatility)
    double rho;   // ∂V/∂r (per 1% change in interest rate)
};

/**
 * @brief Option pricing result with price and Greeks
 */
struct PricingResult {
    double price;
    double implied_volatility;
    Greeks greeks;
};

/**
 * @brief Trinomial Tree Option Pricer
 *
 * Uses trinomial tree model with OpenMP parallelization for computing
 * option prices and Greeks. Supports both American and European options.
 */
class TrinomialPricer {
  public:
    /**
     * @brief Construct a trinomial pricer
     *
     * @param steps Number of time steps in the tree (more steps = more accuracy)
     */
    explicit TrinomialPricer(int steps = 100) : steps_(steps) {}

    /**
     * @brief Calculate option price and Greeks
     *
     * @param spot Current underlying price
     * @param strike Strike price
     * @param time_to_expiry Time to expiration (in years)
     * @param volatility Annualized volatility (e.g., 0.25 = 25%)
     * @param risk_free_rate Risk-free rate (e.g., 0.05 = 5%)
     * @param option_type CALL or PUT
     * @param option_style EUROPEAN or AMERICAN
     * @return PricingResult containing price and Greeks
     */
    [[nodiscard]] auto price(double spot, double strike, double time_to_expiry, double volatility,
                             double risk_free_rate, OptionType option_type,
                             OptionStyle option_style = OptionStyle::AMERICAN) const
        -> PricingResult {
        PricingResult result;
        result.price = calculate_price(spot, strike, time_to_expiry, volatility, risk_free_rate,
                                       option_type, option_style);
        result.implied_volatility = volatility; // Store input volatility
        result.greeks = calculate_greeks(spot, strike, time_to_expiry, volatility, risk_free_rate,
                                         option_type, option_style);
        return result;
    }

    /**
     * @brief Calculate option price only (no Greeks)
     *
     * @param spot Current underlying price
     * @param strike Strike price
     * @param time_to_expiry Time to expiration (in years)
     * @param volatility Annualized volatility
     * @param risk_free_rate Risk-free rate
     * @param option_type CALL or PUT
     * @param option_style EUROPEAN or AMERICAN
     * @return Option price
     */
    [[nodiscard]] auto calculate_price(double spot, double strike, double time_to_expiry,
                                       double volatility, double risk_free_rate,
                                       OptionType option_type,
                                       OptionStyle option_style = OptionStyle::AMERICAN) const
        -> double {
        // Time step
        double dt = time_to_expiry / steps_;

        // Trinomial tree parameters
        double u = std::exp(volatility * std::sqrt(2.0 * dt)); // Up factor
        double d = 1.0 / u;                                    // Down factor (middle stays at spot)

        // Risk-neutral probabilities
        double dx = volatility * std::sqrt(dt);
        double nu = risk_free_rate - 0.5 * volatility * volatility;
        double pu =
            0.5 * ((volatility * volatility * dt + nu * nu * dt * dt) / (dx * dx) + nu * dt / dx);
        double pd =
            0.5 * ((volatility * volatility * dt + nu * nu * dt * dt) / (dx * dx) - nu * dt / dx);
        double pm = 1.0 - pu - pd; // Middle probability

        // Discount factor
        double disc = std::exp(-risk_free_rate * dt);

        // Build tree: option_values[step][node]
        // At step n, we have 2*n+1 possible states
        std::vector<std::vector<double>> option_values(steps_ + 1);
        for (int i = 0; i <= steps_; ++i) {
            option_values[i].resize(2 * i + 1);
        }

        // Terminal payoffs at expiration (step = steps_)
        int n_terminal = 2 * steps_ + 1;
#pragma omp parallel for if (n_terminal > 100)
        for (int j = 0; j < n_terminal; ++j) {
            int power = j - steps_; // Range: -steps_ to +steps_
            double stock_price = spot * std::pow(u, power);
            double payoff = 0.0;

            if (option_type == OptionType::CALL) {
                payoff = std::max(stock_price - strike, 0.0);
            } else {
                payoff = std::max(strike - stock_price, 0.0);
            }

            option_values[steps_][j] = payoff;
        }

        // Backward induction through the tree
        for (int i = steps_ - 1; i >= 0; --i) {
            int n_nodes = 2 * i + 1;

#if SIMD_AVAILABLE && (defined(__AVX__) || defined(__AVX2__))
            // SIMD-optimized path for larger node counts
            if (n_nodes >= 4 && option_style == OptionStyle::EUROPEAN) {
                // For European options, we can vectorize the continuation value calculation
                __m256d vec_pu = _mm256_set1_pd(pu * disc);
                __m256d vec_pm = _mm256_set1_pd(pm * disc);
                __m256d vec_pd = _mm256_set1_pd(pd * disc);

                int j = 0;
                // Process 4 nodes at a time
                for (; j + 3 < n_nodes; j += 4) {
                    // Load next level values
                    __m256d vec_up = _mm256_loadu_pd(&option_values[i + 1][j + 2]);
                    __m256d vec_mid = _mm256_loadu_pd(&option_values[i + 1][j + 1]);
                    __m256d vec_down = _mm256_loadu_pd(&option_values[i + 1][j]);

                    // Calculate: hold_value = pu*up + pm*mid + pd*down
                    __m256d vec_result = _mm256_mul_pd(vec_pu, vec_up);
                    vec_result = _mm256_fmadd_pd(vec_pm, vec_mid, vec_result);
                    vec_result = _mm256_fmadd_pd(vec_pd, vec_down, vec_result);

                    // Store result
                    _mm256_storeu_pd(&option_values[i][j], vec_result);
                }

                // Handle remaining nodes (tail)
                for (; j < n_nodes; ++j) {
                    double hold_value =
                        disc * (pu * option_values[i + 1][j + 2] +
                                pm * option_values[i + 1][j + 1] + pd * option_values[i + 1][j]);
                    option_values[i][j] = hold_value;
                }
            } else {
    // American options or small node counts - use OpenMP parallelization
    #pragma omp parallel for schedule(static) if (n_nodes > 100)
                for (int j = 0; j < n_nodes; ++j) {
                    // Expected value from holding the option
                    double hold_value = disc * (pu * option_values[i + 1][j + 2] + // Up
                                                pm * option_values[i + 1][j + 1] + // Middle
                                                pd * option_values[i + 1][j]       // Down
                                               );

                    if (option_style == OptionStyle::AMERICAN) {
                        // Early exercise value
                        int power = j - i;
                        double stock_price = spot * std::pow(u, power);
                        double exercise_value = 0.0;

                        if (option_type == OptionType::CALL) {
                            exercise_value = std::max(stock_price - strike, 0.0);
                        } else {
                            exercise_value = std::max(strike - stock_price, 0.0);
                        }

                        // Take maximum of holding vs exercising
                        option_values[i][j] = std::max(hold_value, exercise_value);
                    } else {
                        // European: can't exercise early
                        option_values[i][j] = hold_value;
                    }
                }
            }
#else
    // Fallback: OpenMP only (no SIMD)
    #pragma omp parallel for schedule(static) if (n_nodes > 100)
            for (int j = 0; j < n_nodes; ++j) {
                // Expected value from holding the option
                double hold_value = disc * (pu * option_values[i + 1][j + 2] + // Up
                                            pm * option_values[i + 1][j + 1] + // Middle
                                            pd * option_values[i + 1][j]       // Down
                                           );

                if (option_style == OptionStyle::AMERICAN) {
                    // Early exercise value
                    int power = j - i;
                    double stock_price = spot * std::pow(u, power);
                    double exercise_value = 0.0;

                    if (option_type == OptionType::CALL) {
                        exercise_value = std::max(stock_price - strike, 0.0);
                    } else {
                        exercise_value = std::max(strike - stock_price, 0.0);
                    }

                    // Take maximum of holding vs exercising
                    option_values[i][j] = std::max(hold_value, exercise_value);
                } else {
                    // European: can't exercise early
                    option_values[i][j] = hold_value;
                }
            }
#endif
        }

        return option_values[0][0];
    }

    /**
     * @brief Calculate all Greeks using finite differences
     *
     * @param spot Current underlying price
     * @param strike Strike price
     * @param time_to_expiry Time to expiration (in years)
     * @param volatility Annualized volatility
     * @param risk_free_rate Risk-free rate
     * @param option_type CALL or PUT
     * @param option_style EUROPEAN or AMERICAN
     * @return Greeks structure
     */
    [[nodiscard]] auto calculate_greeks(double spot, double strike, double time_to_expiry,
                                        double volatility, double risk_free_rate,
                                        OptionType option_type,
                                        OptionStyle option_style = OptionStyle::AMERICAN) const
        -> Greeks {
        Greeks greeks;

        // Base option price
        double V = calculate_price(spot, strike, time_to_expiry, volatility, risk_free_rate,
                                   option_type, option_style);

        // Perturbation sizes
        double dS = spot * 0.01; // 1% change in spot
        double dt = 1.0 / 365.0; // 1 day change
        double dv = 0.01;        // 1% change in volatility
        double dr = 0.01;        // 1% change in interest rate

        // Calculate prices with perturbations (can be parallelized)
        double V_up, V_down, V_up_up, V_down_down, V_t_minus, V_v_up, V_r_up;

#pragma omp parallel sections
        {
#pragma omp section
            {
                // For Delta and Gamma
                V_up = calculate_price(spot + dS, strike, time_to_expiry, volatility,
                                       risk_free_rate, option_type, option_style);
            }

#pragma omp section
            {
                V_down = calculate_price(spot - dS, strike, time_to_expiry, volatility,
                                         risk_free_rate, option_type, option_style);
            }

#pragma omp section
            {
                V_up_up = calculate_price(spot + 2 * dS, strike, time_to_expiry, volatility,
                                          risk_free_rate, option_type, option_style);
            }

#pragma omp section
            {
                V_down_down = calculate_price(spot - 2 * dS, strike, time_to_expiry, volatility,
                                              risk_free_rate, option_type, option_style);
            }

#pragma omp section
            {
                // For Theta (1 day earlier)
                if (time_to_expiry > dt) {
                    V_t_minus = calculate_price(spot, strike, time_to_expiry - dt, volatility,
                                                risk_free_rate, option_type, option_style);
                } else {
                    V_t_minus = V; // Can't go negative time
                }
            }

#pragma omp section
            {
                // For Vega
                V_v_up = calculate_price(spot, strike, time_to_expiry, volatility + dv,
                                         risk_free_rate, option_type, option_style);
            }

#pragma omp section
            {
                // For Rho
                V_r_up = calculate_price(spot, strike, time_to_expiry, volatility,
                                         risk_free_rate + dr, option_type, option_style);
            }
        }

        // Delta: ∂V/∂S (central difference)
        greeks.delta = (V_up - V_down) / (2.0 * dS);

        // Gamma: ∂²V/∂S² (second-order central difference)
        greeks.gamma = (V_up - 2.0 * V + V_down) / (dS * dS);

        // Theta: ∂V/∂t (per day, negative because time decay)
        greeks.theta = (V_t_minus - V) / 1.0; // Per day

        // Vega: ∂V/∂σ (per 1% change in volatility)
        greeks.vega = (V_v_up - V) / dv;

        // Rho: ∂V/∂r (per 1% change in interest rate)
        greeks.rho = (V_r_up - V) / dr;

        return greeks;
    }

    /**
     * @brief Set number of time steps
     * @param steps Number of steps (more = more accurate but slower)
     */
    auto set_steps(int steps) -> void { steps_ = steps; }

    /**
     * @brief Get number of time steps
     * @return Current number of steps
     */
    [[nodiscard]] auto get_steps() const -> int { return steps_; }

  private:
    int steps_; // Number of time steps in the tree
};

} // namespace options
