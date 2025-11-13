/**
 * BigBrotherAnalytics - Value at Risk (VaR) Calculator Module (C++23)
 *
 * Fluent API for VaR calculation with Intel MKL acceleration.
 * Implements Parametric, Historical, and Monte Carlo VaR methods.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - Intel MKL vector math for performance
 * - NO std::format (it's buggy)
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>
#include <string>
#include <vector>

// Intel MKL headers
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>

// Module declaration
export module bigbrother.risk.var_calculator;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using bigbrother::utils::Logger;

// ============================================================================
// VaR Methods
// ============================================================================

enum class VaRMethod {
    Parametric, // Normal distribution assumption
    Historical, // Historical simulation
    MonteCarlo, // Monte Carlo simulation
    Hybrid      // Combines multiple methods
};

// ============================================================================
// VaR Configuration
// ============================================================================

struct VaRConfig {
    VaRMethod method{VaRMethod::Parametric};
    double confidence_level{0.95}; // 95% confidence
    int holding_period{1};         // Days
    int lookback_period{252};      // Trading days
    int simulations{10000};        // For Monte Carlo
    bool use_ewma{false};          // Exponentially weighted moving average
    double ewma_lambda{0.94};      // EWMA decay factor
};

// ============================================================================
// VaR Result
// ============================================================================

struct VaRResult {
    double var_amount{0.0};         // VaR in dollars
    double var_percentage{0.0};     // VaR as % of portfolio
    double expected_shortfall{0.0}; // CVaR/ES
    double volatility{0.0};         // Portfolio volatility
    VaRMethod method_used;
    double confidence_level{0.0};
    int holding_period{0};

    [[nodiscard]] auto isValid() const noexcept -> bool {
        return var_amount > 0.0 && var_percentage > 0.0;
    }

    [[nodiscard]] auto getRiskLevel() const noexcept -> char const* {
        if (var_percentage > 0.10)
            return "HIGH";
        if (var_percentage > 0.05)
            return "MEDIUM";
        return "LOW";
    }
};

// ============================================================================
// VaR Calculator - Fluent API with MKL Acceleration
// ============================================================================

class VaRCalculator {
  public:
    // Public constructor for pybind11 shared_ptr holder
    VaRCalculator() = default;

    // Destructor
    ~VaRCalculator() = default;

    // Factory method
    [[nodiscard]] static auto create() noexcept -> VaRCalculator { return VaRCalculator{}; }

    // Fluent API - Configuration
    [[nodiscard]] auto withMethod(VaRMethod method) noexcept -> VaRCalculator& {
        config_.method = method;
        return *this;
    }

    [[nodiscard]] auto withConfidenceLevel(double level) noexcept -> VaRCalculator& {
        config_.confidence_level = std::clamp(level, 0.90, 0.99);
        return *this;
    }

    [[nodiscard]] auto withHoldingPeriod(int days) noexcept -> VaRCalculator& {
        config_.holding_period = std::max(1, days);
        return *this;
    }

    [[nodiscard]] auto withLookbackPeriod(int days) noexcept -> VaRCalculator& {
        config_.lookback_period = std::max(30, days);
        return *this;
    }

    [[nodiscard]] auto withSimulations(int count) noexcept -> VaRCalculator& {
        config_.simulations = std::max(1000, count);
        return *this;
    }

    [[nodiscard]] auto withEWMA(bool enable, double lambda = 0.94) noexcept -> VaRCalculator& {
        config_.use_ewma = enable;
        config_.ewma_lambda = lambda;
        return *this;
    }

    // Set historical returns data
    [[nodiscard]] auto withReturns(std::vector<double> returns) noexcept -> VaRCalculator& {
        std::lock_guard lock{mutex_};
        returns_ = std::move(returns);
        return *this;
    }

    // Calculate VaR
    [[nodiscard]] auto calculate(double portfolio_value) const noexcept -> Result<VaRResult> {
        std::lock_guard lock{mutex_};

        if (portfolio_value <= 0.0) {
            return makeError<VaRResult>(ErrorCode::InvalidParameter,
                                        "Portfolio value must be positive");
        }

        if (returns_.empty()) {
            return makeError<VaRResult>(ErrorCode::InvalidParameter, "No returns data provided");
        }

        switch (config_.method) {
            case VaRMethod::Parametric:
                return calculateParametricVaR(portfolio_value);
            case VaRMethod::Historical:
                return calculateHistoricalVaR(portfolio_value);
            case VaRMethod::MonteCarlo:
                return calculateMonteCarloVaR(portfolio_value);
            case VaRMethod::Hybrid:
                return calculateHybridVaR(portfolio_value);
            default:
                return calculateParametricVaR(portfolio_value);
        }
    }

    // Query methods
    [[nodiscard]] auto getConfig() const noexcept -> VaRConfig { return config_; }

    [[nodiscard]] auto getDataSize() const noexcept -> size_t {
        std::lock_guard lock{mutex_};
        return returns_.size();
    }

  private:
    // Move constructor - mutex cannot be moved, so we default-construct a new one
    VaRCalculator(VaRCalculator&& other) noexcept
        : config_(std::move(other.config_)), returns_(std::move(other.returns_)) {
        // mutex_ is default-constructed
    }

    // Move assignment - mutex cannot be moved
    auto operator=(VaRCalculator&& other) noexcept -> VaRCalculator& {
        if (this != &other) {
            config_ = std::move(other.config_);
            returns_ = std::move(other.returns_);
            // mutex_ remains as-is
        }
        return *this;
    }

    // Explicitly delete copy operations
    VaRCalculator(VaRCalculator const&) = delete;
    auto operator=(VaRCalculator const&) -> VaRCalculator& = delete;

    mutable std::mutex mutex_;
    VaRConfig config_;
    std::vector<double> returns_;

    // ========================================================================
    // Parametric VaR (Normal Distribution Assumption) with MKL
    // ========================================================================

    [[nodiscard]] auto calculateParametricVaR(double portfolio_value) const noexcept
        -> Result<VaRResult> {

        if (returns_.empty()) {
            return makeError<VaRResult>(ErrorCode::InvalidParameter, "No returns data");
        }

        // Calculate mean and standard deviation using MKL
        double mean = 0.0;
        double variance = 0.0;

        if (config_.use_ewma) {
            // Exponentially weighted moving average
            variance = calculateEWMAVariance();
            mean = calculateEWMAMean();
        } else {
            // Standard mean and variance using MKL
            int n = static_cast<int>(returns_.size());

            // MKL vector mean
            mean = cblas_dasum(n, returns_.data(), 1) / n;

            // MKL vector variance (using vdSub and vdSqr)
            std::vector<double> centered(n);
            std::vector<double> squared(n);

            // Subtract mean: centered = returns - mean
            vdLinearFrac(n, returns_.data(), returns_.data(), 1.0, -mean, 0.0, 0.0,
                         centered.data());

            // Square: squared = centered^2
            vdSqr(n, centered.data(), squared.data());

            // Sum and divide by n-1 for sample variance
            variance = cblas_dasum(n, squared.data(), 1) / (n - 1);
        }

        double std_dev = std::sqrt(variance);

        // Get z-score for confidence level using inverse normal CDF
        double z_score = getZScore(config_.confidence_level);

        // Scale for holding period (sqrt rule)
        double scaled_std = std_dev * std::sqrt(static_cast<double>(config_.holding_period));

        // VaR = portfolio_value * (mean - z * std_dev)
        double var_percentage = -(mean - z_score * scaled_std);
        double var_amount = portfolio_value * var_percentage;

        // Expected Shortfall (CVaR) for normal distribution
        // ES = portfolio_value * std_dev * phi(z) / (1 - confidence_level)
        double pdf_at_z = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * z_score * z_score);
        double es_percentage = scaled_std * pdf_at_z / (1.0 - config_.confidence_level);
        double expected_shortfall = portfolio_value * es_percentage;

        VaRResult result{.var_amount = var_amount,
                         .var_percentage = var_percentage,
                         .expected_shortfall = expected_shortfall,
                         .volatility = scaled_std,
                         .method_used = VaRMethod::Parametric,
                         .confidence_level = config_.confidence_level,
                         .holding_period = config_.holding_period};

        Logger::getInstance().debug("Parametric VaR: ${:.2f} ({:.2f}%), ES: ${:.2f}, Vol: {:.2f}%",
                                    var_amount, var_percentage * 100, expected_shortfall,
                                    scaled_std * 100);

        return result;
    }

    // ========================================================================
    // Historical VaR with MKL-accelerated sorting
    // ========================================================================

    [[nodiscard]] auto calculateHistoricalVaR(double portfolio_value) const noexcept
        -> Result<VaRResult> {

        if (returns_.empty()) {
            return makeError<VaRResult>(ErrorCode::InvalidParameter, "No returns data");
        }

        // Copy returns for sorting
        std::vector<double> sorted_returns = returns_;
        int n = static_cast<int>(sorted_returns.size());

        // Sort using std::sort (compiler will use SIMD when possible)
        std::sort(sorted_returns.begin(), sorted_returns.end());

        // Calculate percentile index
        double percentile = 1.0 - config_.confidence_level;
        int index = static_cast<int>(percentile * n);
        index = std::clamp(index, 0, n - 1);

        // VaR is the return at the percentile
        double var_return = -sorted_returns[index]; // Negative because losses are negative
        double var_percentage = var_return * std::sqrt(static_cast<double>(config_.holding_period));
        double var_amount = portfolio_value * var_percentage;

        // Expected Shortfall: average of all returns worse than VaR
        double es_sum = 0.0;
        int es_count = 0;
        for (int i = 0; i <= index; ++i) {
            es_sum += -sorted_returns[i];
            ++es_count;
        }
        double es_percentage = (es_count > 0) ? (es_sum / es_count) : var_percentage;
        es_percentage *= std::sqrt(static_cast<double>(config_.holding_period));
        double expected_shortfall = portfolio_value * es_percentage;

        // Calculate volatility using MKL
        double mean = cblas_dasum(n, returns_.data(), 1) / n;
        std::vector<double> centered(n);
        std::vector<double> squared(n);
        vdLinearFrac(n, returns_.data(), returns_.data(), 1.0, -mean, 0.0, 0.0, centered.data());
        vdSqr(n, centered.data(), squared.data());
        double variance = cblas_dasum(n, squared.data(), 1) / (n - 1);
        double volatility = std::sqrt(variance);

        VaRResult result{.var_amount = var_amount,
                         .var_percentage = var_percentage,
                         .expected_shortfall = expected_shortfall,
                         .volatility = volatility,
                         .method_used = VaRMethod::Historical,
                         .confidence_level = config_.confidence_level,
                         .holding_period = config_.holding_period};

        Logger::getInstance().debug("Historical VaR: ${:.2f} ({:.2f}%), ES: ${:.2f}", var_amount,
                                    var_percentage * 100, expected_shortfall);

        return result;
    }

    // ========================================================================
    // Monte Carlo VaR with MKL VSL (Vector Statistics Library)
    // ========================================================================

    [[nodiscard]] auto calculateMonteCarloVaR(double portfolio_value) const noexcept
        -> Result<VaRResult> {

        if (returns_.empty()) {
            return makeError<VaRResult>(ErrorCode::InvalidParameter, "No returns data");
        }

        // Calculate mean and std dev from historical data
        int n = static_cast<int>(returns_.size());
        double mean = cblas_dasum(n, returns_.data(), 1) / n;

        std::vector<double> centered(n);
        std::vector<double> squared(n);
        vdLinearFrac(n, returns_.data(), returns_.data(), 1.0, -mean, 0.0, 0.0, centered.data());
        vdSqr(n, centered.data(), squared.data());
        double variance = cblas_dasum(n, squared.data(), 1) / (n - 1);
        double std_dev = std::sqrt(variance);

        // Generate random returns using MKL VSL
        std::vector<double> simulated_returns(config_.simulations);

        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MT19937, static_cast<unsigned int>(std::random_device{}()));

        // Generate normal random numbers
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, config_.simulations,
                      simulated_returns.data(), mean, std_dev);

        vslDeleteStream(&stream);

        // Scale for holding period
        double scale_factor = std::sqrt(static_cast<double>(config_.holding_period));
        cblas_dscal(config_.simulations, scale_factor, simulated_returns.data(), 1);

        // Sort to find percentile
        std::sort(simulated_returns.begin(), simulated_returns.end());

        double percentile = 1.0 - config_.confidence_level;
        int index = static_cast<int>(percentile * config_.simulations);
        index = std::clamp(index, 0, config_.simulations - 1);

        double var_return = -simulated_returns[index];
        double var_percentage = var_return;
        double var_amount = portfolio_value * var_percentage;

        // Expected Shortfall
        double es_sum = 0.0;
        for (int i = 0; i <= index; ++i) {
            es_sum += -simulated_returns[i];
        }
        double es_percentage = es_sum / (index + 1);
        double expected_shortfall = portfolio_value * es_percentage;

        VaRResult result{.var_amount = var_amount,
                         .var_percentage = var_percentage,
                         .expected_shortfall = expected_shortfall,
                         .volatility = std_dev * scale_factor,
                         .method_used = VaRMethod::MonteCarlo,
                         .confidence_level = config_.confidence_level,
                         .holding_period = config_.holding_period};

        Logger::getInstance().debug("Monte Carlo VaR ({} sims): ${:.2f} ({:.2f}%), ES: ${:.2f}",
                                    config_.simulations, var_amount, var_percentage * 100,
                                    expected_shortfall);

        return result;
    }

    // ========================================================================
    // Hybrid VaR (Average of multiple methods)
    // ========================================================================

    [[nodiscard]] auto calculateHybridVaR(double portfolio_value) const noexcept
        -> Result<VaRResult> {

        auto parametric = calculateParametricVaR(portfolio_value);
        auto historical = calculateHistoricalVaR(portfolio_value);
        auto monte_carlo = calculateMonteCarloVaR(portfolio_value);

        if (!parametric || !historical || !monte_carlo) {
            return makeError<VaRResult>(ErrorCode::RuntimeError, "Hybrid VaR calculation failed");
        }

        // Average the results
        VaRResult result{
            .var_amount =
                (parametric->var_amount + historical->var_amount + monte_carlo->var_amount) / 3.0,
            .var_percentage = (parametric->var_percentage + historical->var_percentage +
                               monte_carlo->var_percentage) /
                              3.0,
            .expected_shortfall = (parametric->expected_shortfall + historical->expected_shortfall +
                                   monte_carlo->expected_shortfall) /
                                  3.0,
            .volatility =
                (parametric->volatility + historical->volatility + monte_carlo->volatility) / 3.0,
            .method_used = VaRMethod::Hybrid,
            .confidence_level = config_.confidence_level,
            .holding_period = config_.holding_period};

        Logger::getInstance().info("Hybrid VaR: ${:.2f} ({:.2f}%), ES: ${:.2f}", result.var_amount,
                                   result.var_percentage * 100, result.expected_shortfall);

        return result;
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    [[nodiscard]] auto calculateEWMAVariance() const noexcept -> double {
        if (returns_.empty())
            return 0.0;

        double lambda = config_.ewma_lambda;
        double variance = 0.0;
        double weight_sum = 0.0;

        // EWMA: variance = sum(lambda^i * r_i^2)
        for (size_t i = 0; i < returns_.size(); ++i) {
            double weight = std::pow(lambda, static_cast<double>(i));
            variance += weight * returns_[i] * returns_[i];
            weight_sum += weight;
        }

        return (weight_sum > 0.0) ? (variance / weight_sum) : 0.0;
    }

    [[nodiscard]] auto calculateEWMAMean() const noexcept -> double {
        if (returns_.empty())
            return 0.0;

        double lambda = config_.ewma_lambda;
        double mean = 0.0;
        double weight_sum = 0.0;

        for (size_t i = 0; i < returns_.size(); ++i) {
            double weight = std::pow(lambda, static_cast<double>(i));
            mean += weight * returns_[i];
            weight_sum += weight;
        }

        return (weight_sum > 0.0) ? (mean / weight_sum) : 0.0;
    }

    [[nodiscard]] static auto getZScore(double confidence_level) noexcept -> double {
        // Approximation of inverse normal CDF for common confidence levels
        if (confidence_level >= 0.99)
            return 2.326;
        if (confidence_level >= 0.975)
            return 1.960;
        if (confidence_level >= 0.95)
            return 1.645;
        if (confidence_level >= 0.90)
            return 1.282;
        return 1.645; // Default to 95%
    }
};

} // namespace bigbrother::risk
