/**
 * BigBrotherAnalytics - Stress Testing Engine Module (C++23)
 *
 * Fluent API for portfolio stress testing with SIMD acceleration.
 * Tests portfolio resilience under extreme market conditions.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - AVX2/AVX-512 SIMD vectorization
 * - NO std::format (it's buggy)
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>
#include <immintrin.h>  // AVX/AVX-512 intrinsics
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.risk.stress_testing;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using bigbrother::utils::Logger;

// ============================================================================
// Stress Scenario Types
// ============================================================================

enum class StressScenario {
    MarketCrash,        // -20% to -40% broad market decline
    VolatilitySpike,    // 2x to 5x volatility increase
    SectorRotation,     // Sector-specific shocks
    InterestRateShock,  // +/-200bp rate change
    CreditCrunch,       // Credit spread widening
    BlackSwan,          // Extreme tail event (-50%)
    FlashCrash,         // Rapid intraday crash
    Custom              // User-defined scenario
};

// ============================================================================
// Position Data for Stress Testing
// ============================================================================

struct StressPosition {
    std::string symbol;
    double quantity{0.0};
    double current_price{0.0};
    double beta{1.0};            // Market sensitivity
    double sector_exposure{1.0};  // Sector sensitivity
    double duration{0.0};         // Interest rate sensitivity
    double delta{0.0};            // For options
    double vega{0.0};             // Volatility sensitivity
};

// ============================================================================
// Stress Test Result
// ============================================================================

struct StressTestResult {
    StressScenario scenario;
    double initial_value{0.0};
    double stressed_value{0.0};
    double pnl{0.0};
    double pnl_percentage{0.0};
    std::unordered_map<std::string, double> position_impacts;  // Symbol -> P&L

    [[nodiscard]] auto getSeverity() const noexcept -> char const* {
        double loss_pct = std::abs(pnl_percentage);
        if (loss_pct > 0.30) return "CRITICAL";
        if (loss_pct > 0.20) return "HIGH";
        if (loss_pct > 0.10) return "MEDIUM";
        return "LOW";
    }

    [[nodiscard]] auto isPortfolioViable() const noexcept -> bool {
        // Portfolio considered viable if loss < 50%
        return pnl_percentage > -0.50;
    }
};

// ============================================================================
// Stress Testing Engine - Fluent API with SIMD
// ============================================================================

class StressTestingEngine {
public:
    // Factory method
    [[nodiscard]] static auto create() noexcept -> StressTestingEngine {
        return StressTestingEngine{};
    }

    // Fluent API - Add positions
    [[nodiscard]] auto addPosition(StressPosition position) noexcept -> StressTestingEngine& {
        std::lock_guard lock{mutex_};
        positions_.push_back(std::move(position));
        return *this;
    }

    [[nodiscard]] auto clearPositions() noexcept -> StressTestingEngine& {
        std::lock_guard lock{mutex_};
        positions_.clear();
        return *this;
    }

    // Run stress test
    [[nodiscard]] auto runStressTest(StressScenario scenario) const noexcept
        -> Result<StressTestResult> {

        std::lock_guard lock{mutex_};

        if (positions_.empty()) {
            return makeError<StressTestResult>(ErrorCode::InvalidParameter,
                                              "No positions to stress test");
        }

        switch (scenario) {
            case StressScenario::MarketCrash:
                return stressMarketCrash();
            case StressScenario::VolatilitySpike:
                return stressVolatilitySpike();
            case StressScenario::SectorRotation:
                return stressSectorRotation();
            case StressScenario::InterestRateShock:
                return stressInterestRateShock();
            case StressScenario::CreditCrunch:
                return stressCreditCrunch();
            case StressScenario::BlackSwan:
                return stressBlackSwan();
            case StressScenario::FlashCrash:
                return stressFlashCrash();
            default:
                return stressMarketCrash();
        }
    }

    // Run multiple scenarios
    [[nodiscard]] auto runAllScenarios() const noexcept
        -> std::vector<StressTestResult> {

        std::vector<StressTestResult> results;

        auto scenarios = {
            StressScenario::MarketCrash,
            StressScenario::VolatilitySpike,
            StressScenario::SectorRotation,
            StressScenario::InterestRateShock,
            StressScenario::BlackSwan,
            StressScenario::FlashCrash
        };

        for (auto scenario : scenarios) {
            if (auto result = runStressTest(scenario); result) {
                results.push_back(*result);
            }
        }

        return results;
    }

    // Query methods
    [[nodiscard]] auto getPositionCount() const noexcept -> size_t {
        std::lock_guard lock{mutex_};
        return positions_.size();
    }

    [[nodiscard]] auto getTotalValue() const noexcept -> double {
        std::lock_guard lock{mutex_};
        return calculatePortfolioValue();
    }

private:
    StressTestingEngine() = default;

    mutable std::mutex mutex_;
    std::vector<StressPosition> positions_;

    // ========================================================================
    // Portfolio Value Calculation (SIMD-accelerated)
    // ========================================================================

    [[nodiscard]] auto calculatePortfolioValue() const noexcept -> double {
        size_t n = positions_.size();
        double total = 0.0;

#ifdef __AVX2__
        // AVX2 vectorized calculation (4 doubles at a time)
        size_t vec_size = n / 4 * 4;  // Round down to multiple of 4
        __m256d sum_vec = _mm256_setzero_pd();

        for (size_t i = 0; i < vec_size; i += 4) {
            __m256d qty = _mm256_set_pd(
                positions_[i+3].quantity,
                positions_[i+2].quantity,
                positions_[i+1].quantity,
                positions_[i].quantity
            );
            __m256d price = _mm256_set_pd(
                positions_[i+3].current_price,
                positions_[i+2].current_price,
                positions_[i+1].current_price,
                positions_[i].current_price
            );
            __m256d value = _mm256_mul_pd(qty, price);
            sum_vec = _mm256_add_pd(sum_vec, value);
        }

        // Horizontal sum of AVX2 vector
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        total = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (size_t i = vec_size; i < n; ++i) {
            total += positions_[i].quantity * positions_[i].current_price;
        }
#else
        // Fallback: scalar calculation
        for (auto const& pos : positions_) {
            total += pos.quantity * pos.current_price;
        }
#endif

        return total;
    }

    // ========================================================================
    // Market Crash Scenario (-30% broad market, SIMD-accelerated)
    // ========================================================================

    [[nodiscard]] auto stressMarketCrash() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();
        double market_shock = -0.30;  // -30% market decline

        StressTestResult result;
        result.scenario = StressScenario::MarketCrash;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;
        result.pnl = 0.0;

        size_t n = positions_.size();

#ifdef __AVX2__
        // AVX2 vectorized stress calculation
        __m256d market_shock_vec = _mm256_set1_pd(market_shock);
        __m256d one_vec = _mm256_set1_pd(1.0);

        size_t vec_size = n / 4 * 4;
        __m256d stressed_sum = _mm256_setzero_pd();

        for (size_t i = 0; i < vec_size; i += 4) {
            // Load position data
            __m256d qty = _mm256_set_pd(
                positions_[i+3].quantity,
                positions_[i+2].quantity,
                positions_[i+1].quantity,
                positions_[i].quantity
            );
            __m256d price = _mm256_set_pd(
                positions_[i+3].current_price,
                positions_[i+2].current_price,
                positions_[i+1].current_price,
                positions_[i].current_price
            );
            __m256d beta = _mm256_set_pd(
                positions_[i+3].beta,
                positions_[i+2].beta,
                positions_[i+1].beta,
                positions_[i].beta
            );

            // shock_factor = 1 + (market_shock * beta)
            __m256d shock_factor = _mm256_fmadd_pd(market_shock_vec, beta, one_vec);

            // stressed_price = price * shock_factor
            __m256d stressed_price = _mm256_mul_pd(price, shock_factor);

            // stressed_value = qty * stressed_price
            __m256d stressed_value = _mm256_mul_pd(qty, stressed_price);
            stressed_sum = _mm256_add_pd(stressed_sum, stressed_value);

            // Store individual position impacts (scalar loop within vector processing)
            double stressed_prices[4];
            _mm256_storeu_pd(stressed_prices, stressed_price);

            for (int j = 0; j < 4; ++j) {
                double pnl = positions_[i+j].quantity * (stressed_prices[j] - positions_[i+j].current_price);
                result.position_impacts[positions_[i+j].symbol] = pnl;
            }
        }

        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, stressed_sum);
        result.stressed_value = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (size_t i = vec_size; i < n; ++i) {
            double shock_factor = 1.0 + (market_shock * positions_[i].beta);
            double stressed_price = positions_[i].current_price * shock_factor;
            double stressed_value = positions_[i].quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = positions_[i].quantity * (stressed_price - positions_[i].current_price);
            result.position_impacts[positions_[i].symbol] = pnl;
        }
#else
        // Fallback: scalar calculation
        for (auto const& pos : positions_) {
            double shock_factor = 1.0 + (market_shock * pos.beta);
            double stressed_price = pos.current_price * shock_factor;
            double stressed_value = pos.quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = pos.quantity * (stressed_price - pos.current_price);
            result.position_impacts[pos.symbol] = pnl;
        }
#endif

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().info(
            "Market Crash Stress Test: Initial ${:.2f} -> Stressed ${:.2f}, P&L ${:.2f} ({:.1f}%)",
            result.initial_value, result.stressed_value, result.pnl, result.pnl_percentage * 100);

        return result;
    }

    // ========================================================================
    // Volatility Spike Scenario (3x vol increase)
    // ========================================================================

    [[nodiscard]] auto stressVolatilitySpike() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();
        double vol_multiplier = 3.0;  // 3x volatility increase

        StressTestResult result;
        result.scenario = StressScenario::VolatilitySpike;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;

        // Volatility spike primarily affects vega (options)
        for (auto const& pos : positions_) {
            // Assume current volatility is 20%, spike to 60%
            double vol_change = 0.40;  // +40% volatility points
            double vega_impact = pos.vega * vol_change * pos.quantity;

            double stressed_value = pos.quantity * pos.current_price + vega_impact;
            result.stressed_value += stressed_value;

            result.position_impacts[pos.symbol] = vega_impact;
        }

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().info(
            "Volatility Spike Stress Test: P&L ${:.2f} ({:.1f}%)",
            result.pnl, result.pnl_percentage * 100);

        return result;
    }

    // ========================================================================
    // Sector Rotation Scenario
    // ========================================================================

    [[nodiscard]] auto stressSectorRotation() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();

        StressTestResult result;
        result.scenario = StressScenario::SectorRotation;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;

        // Assume worst-case: concentrated sectors down 15%, others flat
        double sector_shock = -0.15;

        for (auto const& pos : positions_) {
            double shock_factor = 1.0 + (sector_shock * pos.sector_exposure);
            double stressed_price = pos.current_price * shock_factor;
            double stressed_value = pos.quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = pos.quantity * (stressed_price - pos.current_price);
            result.position_impacts[pos.symbol] = pnl;
        }

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().info(
            "Sector Rotation Stress Test: P&L ${:.2f} ({:.1f}%)",
            result.pnl, result.pnl_percentage * 100);

        return result;
    }

    // ========================================================================
    // Interest Rate Shock Scenario (+200bp)
    // ========================================================================

    [[nodiscard]] auto stressInterestRateShock() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();
        double rate_change = 0.02;  // +200 basis points

        StressTestResult result;
        result.scenario = StressScenario::InterestRateShock;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;

        for (auto const& pos : positions_) {
            // Duration-based impact: DV01 * rate_change
            double price_impact = -pos.duration * rate_change * pos.current_price;
            double stressed_price = pos.current_price + price_impact;
            double stressed_value = pos.quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = pos.quantity * price_impact;
            result.position_impacts[pos.symbol] = pnl;
        }

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().info(
            "Interest Rate Shock Stress Test: P&L ${:.2f} ({:.1f}%)",
            result.pnl, result.pnl_percentage * 100);

        return result;
    }

    // ========================================================================
    // Credit Crunch Scenario (credit spread widening)
    // ========================================================================

    [[nodiscard]] auto stressCreditCrunch() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();
        double spread_widening = 0.05;  // 500bp spread widening

        StressTestResult result;
        result.scenario = StressScenario::CreditCrunch;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;

        for (auto const& pos : positions_) {
            // Impact proportional to duration and credit sensitivity
            double price_impact = -pos.duration * spread_widening * pos.current_price * 0.5;
            double stressed_price = pos.current_price + price_impact;
            double stressed_value = pos.quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = pos.quantity * price_impact;
            result.position_impacts[pos.symbol] = pnl;
        }

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().info(
            "Credit Crunch Stress Test: P&L ${:.2f} ({:.1f}%)",
            result.pnl, result.pnl_percentage * 100);

        return result;
    }

    // ========================================================================
    // Black Swan Scenario (-50% extreme crash)
    // ========================================================================

    [[nodiscard]] auto stressBlackSwan() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();
        double extreme_shock = -0.50;  // -50% crash

        StressTestResult result;
        result.scenario = StressScenario::BlackSwan;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;

        for (auto const& pos : positions_) {
            double shock_factor = 1.0 + (extreme_shock * pos.beta);
            double stressed_price = pos.current_price * shock_factor;
            double stressed_value = pos.quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = pos.quantity * (stressed_price - pos.current_price);
            result.position_impacts[pos.symbol] = pnl;
        }

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().warn(
            "Black Swan Stress Test: CRITICAL - P&L ${:.2f} ({:.1f}%)",
            result.pnl, result.pnl_percentage * 100);

        return result;
    }

    // ========================================================================
    // Flash Crash Scenario (-15% rapid intraday crash)
    // ========================================================================

    [[nodiscard]] auto stressFlashCrash() const noexcept
        -> Result<StressTestResult> {

        double initial_value = calculatePortfolioValue();
        double flash_shock = -0.15;  // -15% rapid decline

        StressTestResult result;
        result.scenario = StressScenario::FlashCrash;
        result.initial_value = initial_value;
        result.stressed_value = 0.0;

        for (auto const& pos : positions_) {
            // Flash crashes affect high-beta stocks more
            double amplified_shock = flash_shock * (1.0 + std::abs(pos.beta - 1.0));
            double shock_factor = 1.0 + amplified_shock;
            double stressed_price = pos.current_price * shock_factor;
            double stressed_value = pos.quantity * stressed_price;
            result.stressed_value += stressed_value;

            double pnl = pos.quantity * (stressed_price - pos.current_price);
            result.position_impacts[pos.symbol] = pnl;
        }

        result.pnl = result.stressed_value - result.initial_value;
        result.pnl_percentage = result.pnl / result.initial_value;

        Logger::getInstance().info(
            "Flash Crash Stress Test: P&L ${:.2f} ({:.1f}%)",
            result.pnl, result.pnl_percentage * 100);

        return result;
    }
};

} // namespace bigbrother::risk
