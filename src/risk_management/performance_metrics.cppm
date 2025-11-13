/**
 * BigBrotherAnalytics - Performance Metrics Calculator Module (C++23)
 *
 * Fluent API for calculating portfolio and strategy performance metrics.
 * Includes Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, etc.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - SIMD vectorization for calculations
 * - NO std::format (it's buggy)
 */

// Global module fragment
module;

#include <algorithm>
#include <cmath>
#include <immintrin.h> // AVX/AVX-512 intrinsics
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.risk.performance_metrics;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using bigbrother::utils::Logger;

// ============================================================================
// Performance Period
// ============================================================================

enum class PerformancePeriod { Daily, Weekly, Monthly, Quarterly, Annual };

// ============================================================================
// Performance Metrics Result
// ============================================================================

struct PerformanceMetrics {
    // Return metrics
    double total_return{0.0};
    double annualized_return{0.0};
    double average_return{0.0};

    // Risk metrics
    double volatility{0.0};          // Annualized standard deviation
    double downside_deviation{0.0};  // Downside volatility
    double max_drawdown{0.0};        // Maximum peak-to-trough decline
    double max_drawdown_duration{0}; // Days in drawdown

    // Risk-adjusted returns
    double sharpe_ratio{0.0};  // (Return - RFR) / Volatility
    double sortino_ratio{0.0}; // (Return - RFR) / Downside Dev
    double calmar_ratio{0.0};  // Annual Return / Max Drawdown
    double omega_ratio{0.0};   // Gains / Losses

    // Win/Loss statistics
    double win_rate{0.0};      // % of winning periods
    double profit_factor{0.0}; // Gross profit / Gross loss
    double expectancy{0.0};    // Average win/loss per trade

    // Distribution metrics
    double skewness{0.0}; // Return distribution skew
    double kurtosis{0.0}; // Return distribution tail risk

    [[nodiscard]] auto isHealthy() const noexcept -> bool {
        return sharpe_ratio > 1.0 && max_drawdown > -0.25;
    }

    [[nodiscard]] auto getRating() const noexcept -> char const* {
        if (sharpe_ratio > 2.0)
            return "EXCELLENT";
        if (sharpe_ratio > 1.5)
            return "GOOD";
        if (sharpe_ratio > 1.0)
            return "FAIR";
        if (sharpe_ratio > 0.5)
            return "POOR";
        return "VERY POOR";
    }
};

// ============================================================================
// Performance Metrics Calculator - Fluent API
// ============================================================================

class PerformanceMetricsCalculator {
  public:
    // Factory method
    [[nodiscard]] static auto create() noexcept -> PerformanceMetricsCalculator {
        return PerformanceMetricsCalculator{};
    }

    // Fluent API - Configure calculator
    [[nodiscard]] auto withReturns(std::vector<double> returns) noexcept
        -> PerformanceMetricsCalculator& {
        std::lock_guard lock{mutex_};
        returns_ = std::move(returns);
        return *this;
    }

    [[nodiscard]] auto withRiskFreeRate(double rfr) noexcept -> PerformanceMetricsCalculator& {
        risk_free_rate_ = rfr;
        return *this;
    }

    [[nodiscard]] auto withPeriod(PerformancePeriod period) noexcept
        -> PerformanceMetricsCalculator& {
        period_ = period;
        return *this;
    }

    [[nodiscard]] auto withTargetReturn(double target) noexcept -> PerformanceMetricsCalculator& {
        target_return_ = target;
        return *this;
    }

    // Calculate all metrics
    [[nodiscard]] auto calculate() const noexcept -> Result<PerformanceMetrics> {
        std::lock_guard lock{mutex_};

        if (returns_.empty()) {
            return makeError<PerformanceMetrics>(ErrorCode::InvalidParameter,
                                                 "No returns data provided");
        }

        if (returns_.size() < 2) {
            return makeError<PerformanceMetrics>(ErrorCode::InvalidParameter,
                                                 "Need at least 2 return observations");
        }

        PerformanceMetrics metrics;

        // Basic return metrics
        metrics.total_return = calculateTotalReturn();
        metrics.average_return = calculateMean(returns_);
        metrics.annualized_return = annualizeReturn(metrics.average_return);

        // Risk metrics
        metrics.volatility = calculateVolatility();
        metrics.downside_deviation = calculateDownsideDeviation();
        metrics.max_drawdown = calculateMaxDrawdown();

        // Risk-adjusted returns
        metrics.sharpe_ratio = calculateSharpeRatio(metrics.annualized_return, metrics.volatility);
        metrics.sortino_ratio =
            calculateSortinoRatio(metrics.annualized_return, metrics.downside_deviation);
        metrics.calmar_ratio = (metrics.max_drawdown != 0.0)
                                   ? metrics.annualized_return / std::abs(metrics.max_drawdown)
                                   : 0.0;
        metrics.omega_ratio = calculateOmegaRatio();

        // Win/Loss statistics
        metrics.win_rate = calculateWinRate();
        metrics.profit_factor = calculateProfitFactor();
        metrics.expectancy = metrics.average_return;

        // Distribution metrics
        metrics.skewness = calculateSkewness();
        metrics.kurtosis = calculateKurtosis();

        Logger::getInstance().info(
            "Performance Metrics: Sharpe={:.2f}, Sortino={:.2f}, MaxDD={:.1f}%, "
            "Win Rate={:.1f}%",
            metrics.sharpe_ratio, metrics.sortino_ratio, metrics.max_drawdown * 100,
            metrics.win_rate * 100);

        return metrics;
    }

    // Utility: Calculate metrics from equity curve
    [[nodiscard]] static auto fromEquityCurve(std::vector<double> const& equity,
                                              double rfr = 0.0) noexcept
        -> Result<PerformanceMetrics> {

        if (equity.size() < 2) {
            return makeError<PerformanceMetrics>(ErrorCode::InvalidParameter,
                                                 "Need at least 2 equity values");
        }

        // Convert equity curve to returns
        std::vector<double> returns;
        returns.reserve(equity.size() - 1);

        for (size_t i = 1; i < equity.size(); ++i) {
            if (equity[i - 1] != 0.0) {
                returns.push_back((equity[i] - equity[i - 1]) / equity[i - 1]);
            }
        }

        return PerformanceMetricsCalculator::create()
            .withReturns(std::move(returns))
            .withRiskFreeRate(rfr)
            .calculate();
    }

  public:
    // Public constructor for pybind11 shared_ptr holder
    PerformanceMetricsCalculator() = default;

  private:
    // Move constructor - mutex cannot be moved, so we default-construct a new one
    PerformanceMetricsCalculator(PerformanceMetricsCalculator&& other) noexcept
        : returns_(std::move(other.returns_)), risk_free_rate_(std::move(other.risk_free_rate_)),
          target_return_(std::move(other.target_return_)), period_(std::move(other.period_)) {
        // mutex_ is default-constructed
    }

    // Move assignment - mutex cannot be moved
    auto operator=(PerformanceMetricsCalculator&& other) noexcept -> PerformanceMetricsCalculator& {
        if (this != &other) {
            returns_ = std::move(other.returns_);
            risk_free_rate_ = std::move(other.risk_free_rate_);
            target_return_ = std::move(other.target_return_);
            period_ = std::move(other.period_);
            // mutex_ remains as-is
        }
        return *this;
    }

    // Destructor - complete Rule of Five
    ~PerformanceMetricsCalculator() = default;

    // Explicitly delete copy operations
    PerformanceMetricsCalculator(PerformanceMetricsCalculator const&) = delete;
    auto operator=(PerformanceMetricsCalculator const&) -> PerformanceMetricsCalculator& = delete;

    mutable std::mutex mutex_;
    std::vector<double> returns_;
    double risk_free_rate_{0.0};
    double target_return_{0.0};
    PerformancePeriod period_{PerformancePeriod::Daily};

    // ========================================================================
    // Period Scaling Factors
    // ========================================================================

    [[nodiscard]] auto getAnnualizationFactor() const noexcept -> double {
        switch (period_) {
            case PerformancePeriod::Daily:
                return 252.0;
            case PerformancePeriod::Weekly:
                return 52.0;
            case PerformancePeriod::Monthly:
                return 12.0;
            case PerformancePeriod::Quarterly:
                return 4.0;
            case PerformancePeriod::Annual:
                return 1.0;
            default:
                return 252.0;
        }
    }

    // ========================================================================
    // Basic Statistics (SIMD-accelerated)
    // ========================================================================

    [[nodiscard]] auto calculateMean(std::vector<double> const& data) const noexcept -> double {

        if (data.empty())
            return 0.0;

        size_t n = data.size();
        double sum = 0.0;

#ifdef __AVX2__
        // AVX2 vectorized sum (4 doubles at a time)
        size_t vec_size = n / 4 * 4;
        __m256d sum_vec = _mm256_setzero_pd();

        for (size_t i = 0; i < vec_size; i += 4) {
            __m256d values = _mm256_loadu_pd(&data[i]);
            sum_vec = _mm256_add_pd(sum_vec, values);
        }

        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (size_t i = vec_size; i < n; ++i) {
            sum += data[i];
        }
#else
        // Fallback: scalar sum
        sum = std::accumulate(data.begin(), data.end(), 0.0);
#endif

        return sum / static_cast<double>(n);
    }

    [[nodiscard]] auto calculateVariance(std::vector<double> const& data,
                                         double mean) const noexcept -> double {

        if (data.empty())
            return 0.0;

        size_t n = data.size();
        double sum_sq_diff = 0.0;

#ifdef __AVX2__
        // AVX2 vectorized variance
        __m256d mean_vec = _mm256_set1_pd(mean);
        __m256d sum_vec = _mm256_setzero_pd();

        size_t vec_size = n / 4 * 4;

        for (size_t i = 0; i < vec_size; i += 4) {
            __m256d values = _mm256_loadu_pd(&data[i]);
            __m256d diff = _mm256_sub_pd(values, mean_vec);
            __m256d sq_diff = _mm256_mul_pd(diff, diff);
            sum_vec = _mm256_add_pd(sum_vec, sq_diff);
        }

        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum_sq_diff = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (size_t i = vec_size; i < n; ++i) {
            double diff = data[i] - mean;
            sum_sq_diff += diff * diff;
        }
#else
        // Fallback: scalar variance
        for (auto val : data) {
            double diff = val - mean;
            sum_sq_diff += diff * diff;
        }
#endif

        return sum_sq_diff / static_cast<double>(n - 1);
    }

    // ========================================================================
    // Return Metrics
    // ========================================================================

    [[nodiscard]] auto calculateTotalReturn() const noexcept -> double {
        double cumulative = 1.0;
        for (double ret : returns_) {
            cumulative *= (1.0 + ret);
        }
        return cumulative - 1.0;
    }

    [[nodiscard]] auto annualizeReturn(double avg_return) const noexcept -> double {
        double factor = getAnnualizationFactor();
        return std::pow(1.0 + avg_return, factor) - 1.0;
    }

    // ========================================================================
    // Risk Metrics
    // ========================================================================

    [[nodiscard]] auto calculateVolatility() const noexcept -> double {
        double mean = calculateMean(returns_);
        double variance = calculateVariance(returns_, mean);
        double std_dev = std::sqrt(variance);

        // Annualize volatility
        return std_dev * std::sqrt(getAnnualizationFactor());
    }

    [[nodiscard]] auto calculateDownsideDeviation() const noexcept -> double {
        // Calculate standard deviation of returns below target
        std::vector<double> downside_returns;

        for (double ret : returns_) {
            if (ret < target_return_) {
                downside_returns.push_back(ret - target_return_);
            }
        }

        if (downside_returns.empty())
            return 0.0;

        double mean = 0.0; // Target is already subtracted
        double variance = calculateVariance(downside_returns, mean);
        double downside_dev = std::sqrt(variance);

        // Annualize
        return downside_dev * std::sqrt(getAnnualizationFactor());
    }

    [[nodiscard]] auto calculateMaxDrawdown() const noexcept -> double {
        if (returns_.empty())
            return 0.0;

        double peak = 1.0;
        double max_dd = 0.0;
        double cumulative = 1.0;

        for (double ret : returns_) {
            cumulative *= (1.0 + ret);

            if (cumulative > peak) {
                peak = cumulative;
            }

            double drawdown = (cumulative - peak) / peak;
            if (drawdown < max_dd) {
                max_dd = drawdown;
            }
        }

        return max_dd;
    }

    // ========================================================================
    // Risk-Adjusted Returns
    // ========================================================================

    [[nodiscard]] auto calculateSharpeRatio(double annual_return, double volatility) const noexcept
        -> double {
        if (volatility == 0.0)
            return 0.0;
        return (annual_return - risk_free_rate_) / volatility;
    }

    [[nodiscard]] auto calculateSortinoRatio(double annual_return,
                                             double downside_dev) const noexcept -> double {
        if (downside_dev == 0.0)
            return 0.0;
        return (annual_return - risk_free_rate_) / downside_dev;
    }

    [[nodiscard]] auto calculateOmegaRatio() const noexcept -> double {
        double gains = 0.0;
        double losses = 0.0;

        for (double ret : returns_) {
            if (ret > target_return_) {
                gains += (ret - target_return_);
            } else {
                losses += (target_return_ - ret);
            }
        }

        return (losses > 0.0) ? gains / losses : 0.0;
    }

    // ========================================================================
    // Win/Loss Statistics
    // ========================================================================

    [[nodiscard]] auto calculateWinRate() const noexcept -> double {
        if (returns_.empty())
            return 0.0;

        size_t wins = 0;
        for (double ret : returns_) {
            if (ret > 0.0)
                ++wins;
        }

        return static_cast<double>(wins) / static_cast<double>(returns_.size());
    }

    [[nodiscard]] auto calculateProfitFactor() const noexcept -> double {
        double gross_profit = 0.0;
        double gross_loss = 0.0;

        for (double ret : returns_) {
            if (ret > 0.0) {
                gross_profit += ret;
            } else {
                gross_loss += std::abs(ret);
            }
        }

        return (gross_loss > 0.0) ? gross_profit / gross_loss : 0.0;
    }

    // ========================================================================
    // Distribution Metrics
    // ========================================================================

    [[nodiscard]] auto calculateSkewness() const noexcept -> double {
        double mean = calculateMean(returns_);
        double variance = calculateVariance(returns_, mean);
        double std_dev = std::sqrt(variance);

        if (std_dev == 0.0)
            return 0.0;

        double sum_cubed = 0.0;
        for (double ret : returns_) {
            double z_score = (ret - mean) / std_dev;
            sum_cubed += z_score * z_score * z_score;
        }

        size_t n = returns_.size();
        return sum_cubed / static_cast<double>(n);
    }

    [[nodiscard]] auto calculateKurtosis() const noexcept -> double {
        double mean = calculateMean(returns_);
        double variance = calculateVariance(returns_, mean);
        double std_dev = std::sqrt(variance);

        if (std_dev == 0.0)
            return 0.0;

        double sum_fourth = 0.0;
        for (double ret : returns_) {
            double z_score = (ret - mean) / std_dev;
            sum_fourth += z_score * z_score * z_score * z_score;
        }

        size_t n = returns_.size();
        return (sum_fourth / static_cast<double>(n)) - 3.0; // Excess kurtosis
    }
};

} // namespace bigbrother::risk
