/**
 * BigBrotherAnalytics - Correlation Analysis Module (C++23)
 *
 * High-performance statistical correlation analysis for trading signals.
 * Discovers relationships between securities across multiple timeframes.
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - Fluent API for analysis workflows
 * - Concepts for type safety
 * - std::expected for error handling
 * - OpenMP parallelization for performance
 *
 * Performance Targets:
 * - Single correlation: < 10 microseconds
 * - 1000x1000 matrix: < 10 seconds
 * - Near-linear MPI scaling
 */

// Global module fragment
module;

#include <vector>
#include <span>
#include <map>
#include <unordered_map>
#include <memory>
#include <concepts>
#include <ranges>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <numbers>
#include <optional>
#include <functional>
#include <expected>

#ifdef _OPENMP
#include <omp.h>
#endif

// Module declaration
export module bigbrother.correlation;

// Import dependencies
import bigbrother.utils.types;

export namespace bigbrother::correlation {

using namespace bigbrother::types;

// ============================================================================
// Core Types
// ============================================================================

/**
 * Correlation Type
 */
enum class CorrelationType {
    Pearson,    // Linear correlation
    Spearman,   // Rank correlation (non-linear)
    Kendall,    // Tau correlation (ordinal)
    Distance    // Distance correlation
};

/**
 * Correlation Result
 * C.1: Struct for passive data
 */
struct CorrelationResult {
    std::string symbol1;
    std::string symbol2;
    double correlation{0.0};     // Correlation coefficient
    double p_value{0.0};         // Statistical significance
    int sample_size{0};          // Number of data points
    int lag{0};                  // Time lag in periods (0 = contemporaneous)
    CorrelationType type{CorrelationType::Pearson};

    [[nodiscard]] constexpr auto isSignificant(double alpha = 0.05) const noexcept -> bool {
        return p_value < alpha;
    }

    [[nodiscard]] constexpr auto isStrong() const noexcept -> bool {
        return std::abs(correlation) > 0.7;
    }

    [[nodiscard]] constexpr auto isModerate() const noexcept -> bool {
        return std::abs(correlation) > 0.4 && std::abs(correlation) <= 0.7;
    }

    [[nodiscard]] constexpr auto isWeak() const noexcept -> bool {
        return std::abs(correlation) <= 0.4;
    }
};

/**
 * Time Series Data
 * C.1: Struct for passive data
 */
struct TimeSeries {
    std::string symbol;
    std::vector<double> values;
    std::vector<Timestamp> timestamps;

    [[nodiscard]] auto size() const noexcept -> size_t {
        return values.size();
    }

    [[nodiscard]] auto validate() const noexcept -> Result<void> {
        if (values.empty()) {
            return makeError<void>(ErrorCode::InvalidParameter, "Empty time series");
        }
        if (values.size() != timestamps.size()) {
            return makeError<void>(ErrorCode::InvalidParameter, "Size mismatch");
        }
        return {};
    }
};

/**
 * Custom hash for std::pair<std::string, std::string>
 */
struct PairHash {
    auto operator()(std::pair<std::string, std::string> const& p) const noexcept -> size_t {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * Correlation Matrix (NxN pairwise correlations)
 */
class CorrelationMatrix {
public:
    CorrelationMatrix() = default;
    explicit CorrelationMatrix(std::vector<std::string> symbols)
        : symbols_{std::move(symbols)} {}

    /**
     * Set correlation value
     * F.20: Return void for simple operations
     */
    auto set(std::string const& symbol1, std::string const& symbol2, double correlation) -> void {
        matrix_[{symbol1, symbol2}] = correlation;
        matrix_[{symbol2, symbol1}] = correlation;  // Symmetric
    }

    /**
     * Get correlation value
     * F.20: Return by value for cheap types
     */
    [[nodiscard]] auto get(std::string const& symbol1, std::string const& symbol2) const -> double {
        if (auto it = matrix_.find({symbol1, symbol2}); it != matrix_.end()) {
            return it->second;
        }
        return 0.0;
    }

    /**
     * Get all symbols
     */
    [[nodiscard]] auto getSymbols() const noexcept -> std::vector<std::string> const& {
        return symbols_;
    }

    /**
     * Get matrix size
     */
    [[nodiscard]] auto size() const noexcept -> size_t {
        return symbols_.size();
    }

    /**
     * Find highly correlated pairs
     */
    [[nodiscard]] auto findHighlyCorrelated(double threshold = 0.7) const
        -> std::vector<CorrelationResult>;

private:
    std::vector<std::string> symbols_;
    // Use unordered_map with custom hash for better performance
    std::unordered_map<std::pair<std::string, std::string>, double, PairHash> matrix_;
};

// ============================================================================
// Correlation Calculator - Core Algorithms
// ============================================================================

/**
 * Correlation Calculator
 *
 * Core correlation computation with trailing return types
 */
class CorrelationCalculator {
public:
    /**
     * Calculate Pearson correlation
     * ρ = Cov(X,Y) / (σ_X * σ_Y)
     */
    [[nodiscard]] static auto pearson(
        std::span<double const> x,
        std::span<double const> y
    ) noexcept -> Result<double>;

    /**
     * Calculate Spearman rank correlation
     */
    [[nodiscard]] static auto spearman(
        std::span<double const> x,
        std::span<double const> y
    ) noexcept -> Result<double>;

    /**
     * Calculate time-lagged cross-correlation
     *
     * @param x Leading series
     * @param y Lagging series
     * @param max_lag Maximum lag to test
     * @return Vector of correlations at each lag
     */
    [[nodiscard]] static auto crossCorrelation(
        std::span<double const> x,
        std::span<double const> y,
        int max_lag = 30
    ) noexcept -> Result<std::vector<double>>;

    /**
     * Find optimal lag with maximum correlation
     */
    [[nodiscard]] static auto findOptimalLag(
        std::span<double const> x,
        std::span<double const> y,
        int max_lag = 30
    ) noexcept -> Result<std::pair<int, double>>;

    /**
     * Calculate rolling correlation
     */
    [[nodiscard]] static auto rollingCorrelation(
        std::span<double const> x,
        std::span<double const> y,
        size_t window_size = 20
    ) noexcept -> Result<std::vector<double>>;

    /**
     * Calculate correlation matrix (OpenMP parallelized)
     */
    [[nodiscard]] static auto correlationMatrix(
        std::vector<TimeSeries> const& series,
        CorrelationType type = CorrelationType::Pearson
    ) noexcept -> Result<CorrelationMatrix>;

    /**
     * Calculate p-value for correlation
     */
    [[nodiscard]] static constexpr auto calculatePValue(
        double correlation,
        int sample_size
    ) noexcept -> double;
};

// ============================================================================
// Fluent API for Correlation Analysis
// ============================================================================

/**
 * Correlation Analyzer - Fluent API
 *
 * Example Usage:
 *
 *   // Simple correlation
 *   auto corr = CorrelationAnalyzer()
 *       .addSeries("NVDA", nvda_prices)
 *       .addSeries("AMD", amd_prices)
 *       .usePearson()
 *       .calculate();
 *
 *   // Time-lagged analysis
 *   auto lagged = CorrelationAnalyzer()
 *       .addSeries("NVDA", nvda_prices)
 *       .addSeries("AMD", amd_prices)
 *       .withLags(0, 30)
 *       .parallel()
 *       .calculate();
 *
 *   // Full matrix
 *   auto matrix = CorrelationAnalyzer()
 *       .addSeries("SPY", spy_data)
 *       .addSeries("QQQ", qqq_data)
 *       .addSeries("IWM", iwm_data)
 *       .usePearson()
 *       .parallel()
 *       .calculateMatrix();
 */
class CorrelationAnalyzer {
public:
    CorrelationAnalyzer() = default;

    /**
     * Add time series for analysis
     * F.16: Pass by value and move
     */
    [[nodiscard]] auto addSeries(std::string symbol, std::vector<double> values) -> CorrelationAnalyzer& {
        TimeSeries series{
            .symbol = std::move(symbol),
            .values = std::move(values),
            .timestamps = {}
        };
        series_.push_back(std::move(series));
        return *this;
    }

    /**
     * Add time series with timestamps
     */
    [[nodiscard]] auto addSeries(TimeSeries series) -> CorrelationAnalyzer& {
        series_.push_back(std::move(series));
        return *this;
    }

    /**
     * Set correlation type
     */
    [[nodiscard]] auto usePearson() noexcept -> CorrelationAnalyzer& {
        type_ = CorrelationType::Pearson;
        return *this;
    }

    [[nodiscard]] auto useSpearman() noexcept -> CorrelationAnalyzer& {
        type_ = CorrelationType::Spearman;
        return *this;
    }

    /**
     * Enable time-lagged analysis
     */
    [[nodiscard]] auto withLags(int min_lag, int max_lag) noexcept -> CorrelationAnalyzer& {
        min_lag_ = min_lag;
        max_lag_ = max_lag;
        use_lags_ = true;
        return *this;
    }

    /**
     * Enable parallel computation
     */
    [[nodiscard]] auto parallel() noexcept -> CorrelationAnalyzer& {
        use_parallel_ = true;
        return *this;
    }

    /**
     * Set rolling window size
     */
    [[nodiscard]] auto rollingWindow(size_t window) noexcept -> CorrelationAnalyzer& {
        window_size_ = window;
        use_rolling_ = true;
        return *this;
    }

    /**
     * Calculate pairwise correlation (for 2 series)
     * Terminal operation
     */
    [[nodiscard]] auto calculate() -> Result<CorrelationResult> {
        if (series_.size() != 2) {
            return makeError<CorrelationResult>(
                ErrorCode::InvalidParameter,
                "Need exactly 2 series for pairwise correlation"
            );
        }

        auto const& s1 = series_[0];
        auto const& s2 = series_[1];

        auto corr_result = CorrelationCalculator::pearson(s1.values, s2.values);
        if (!corr_result) {
            return std::unexpected(corr_result.error());
        }

        CorrelationResult result{
            .symbol1 = s1.symbol,
            .symbol2 = s2.symbol,
            .correlation = *corr_result,
            .p_value = 0.001,  // Stub
            .sample_size = static_cast<int>(s1.values.size()),
            .lag = 0,
            .type = type_
        };

        return result;
    }

    /**
     * Calculate correlation matrix
     * Terminal operation
     */
    [[nodiscard]] auto calculateMatrix() -> Result<CorrelationMatrix> {
        return CorrelationCalculator::correlationMatrix(series_, type_);
    }

private:
    std::vector<TimeSeries> series_;
    CorrelationType type_{CorrelationType::Pearson};
    bool use_parallel_{false};
    bool use_lags_{false};
    int min_lag_{0};
    int max_lag_{30};
    bool use_rolling_{false};
    size_t window_size_{20};
};

} // export namespace bigbrother::correlation

// ============================================================================
// Implementation Section (module-private)
// ============================================================================

module :private;

namespace bigbrother::correlation {

// ============================================================================
// CorrelationCalculator Implementation
// ============================================================================

auto CorrelationCalculator::pearson(
    std::span<double const> x,
    std::span<double const> y
) noexcept -> Result<double> {

    // Validate inputs
    if (x.empty() || y.empty()) {
        return makeError<double>(ErrorCode::InvalidParameter, "Empty input arrays");
    }

    if (x.size() != y.size()) {
        return makeError<double>(ErrorCode::InvalidParameter, "Array size mismatch");
    }

    if (x.size() < 2) {
        return makeError<double>(ErrorCode::InvalidParameter, "Need at least 2 data points");
    }

    size_t const n = x.size();

    // Calculate means
    double const mean_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
    double const mean_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);

    // Calculate sums (single pass for cache efficiency)
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    double sum_yy = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double const dx = x[i] - mean_x;
        double const dy = y[i] - mean_y;

        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    // Check for zero variance
    if (sum_xx == 0.0 || sum_yy == 0.0) {
        return 0.0;
    }

    // Calculate correlation
    double const correlation = sum_xy / std::sqrt(sum_xx * sum_yy);

    // Clamp to valid range [-1, +1]
    return std::clamp(correlation, -1.0, 1.0);
}

auto CorrelationCalculator::spearman(
    std::span<double const> x,
    std::span<double const> y
) noexcept -> Result<double> {

    if (x.empty() || y.empty() || x.size() != y.size()) {
        return makeError<double>(ErrorCode::InvalidParameter, "Invalid inputs");
    }

    // Convert to ranks and calculate Pearson on ranks
    // Stub for now
    return pearson(x, y);
}

auto CorrelationCalculator::crossCorrelation(
    std::span<double const> x,
    std::span<double const> y,
    int max_lag
) noexcept -> Result<std::vector<double>> {

    std::vector<double> correlations;
    correlations.reserve(max_lag + 1);

    for (int lag = 0; lag <= max_lag; ++lag) {
        if (lag >= static_cast<int>(y.size())) {
            break;
        }

        // Calculate correlation with lag
        auto x_lagged = x.subspan(0, x.size() - lag);
        auto y_lagged = y.subspan(lag);

        auto corr = pearson(x_lagged, y_lagged);
        if (corr) {
            correlations.push_back(*corr);
        } else {
            correlations.push_back(0.0);
        }
    }

    return correlations;
}

auto CorrelationCalculator::findOptimalLag(
    std::span<double const> x,
    std::span<double const> y,
    int max_lag
) noexcept -> Result<std::pair<int, double>> {

    auto correlations = crossCorrelation(x, y, max_lag);
    if (!correlations) {
        return std::unexpected(correlations.error());
    }

    auto const& corrs = *correlations;
    if (corrs.empty()) {
        return std::pair{0, 0.0};
    }

    // Find max absolute correlation
    auto max_it = std::ranges::max_element(corrs,
        [](double a, double b) -> bool { return std::abs(a) < std::abs(b); });

    int const optimal_lag = static_cast<int>(std::distance(corrs.begin(), max_it));
    double const max_corr = *max_it;

    return std::pair{optimal_lag, max_corr};
}

auto CorrelationCalculator::rollingCorrelation(
    std::span<double const> x,
    std::span<double const> y,
    size_t window_size
) noexcept -> Result<std::vector<double>> {

    if (x.size() < window_size || y.size() < window_size) {
        return makeError<std::vector<double>>(
            ErrorCode::InvalidParameter,
            "Series too short for window size"
        );
    }

    std::vector<double> rolling_corrs;
    rolling_corrs.reserve(x.size() - window_size + 1);

    for (size_t i = 0; i + window_size <= x.size(); ++i) {
        auto x_window = x.subspan(i, window_size);
        auto y_window = y.subspan(i, window_size);

        auto corr = pearson(x_window, y_window);
        if (corr) {
            rolling_corrs.push_back(*corr);
        } else {
            rolling_corrs.push_back(0.0);
        }
    }

    return rolling_corrs;
}

auto CorrelationCalculator::correlationMatrix(
    std::vector<TimeSeries> const& series,
    CorrelationType type
) noexcept -> Result<CorrelationMatrix> {

    if (series.empty()) {
        return makeError<CorrelationMatrix>(
            ErrorCode::InvalidParameter,
            "Empty series vector"
        );
    }

    // Extract symbols
    std::vector<std::string> symbols;
    symbols.reserve(series.size());
    for (auto const& s : series) {
        symbols.push_back(s.symbol);
    }

    CorrelationMatrix matrix{symbols};

    size_t const n = series.size();

    // Calculate all pairwise correlations (OpenMP parallelized)
    #pragma omp parallel for schedule(dynamic) if(n > 10)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            if (i == j) {
                #pragma omp critical
                matrix.set(series[i].symbol, series[j].symbol, 1.0);
                continue;
            }

            auto corr = pearson(series[i].values, series[j].values);
            if (corr) {
                #pragma omp critical
                matrix.set(series[i].symbol, series[j].symbol, *corr);
            }
        }
    }

    return matrix;
}

constexpr auto CorrelationCalculator::calculatePValue(
    double correlation,
    int sample_size
) noexcept -> double {
    // Stub - will implement t-test for correlation significance
    return 0.001;
}

// ============================================================================
// CorrelationMatrix Implementation
// ============================================================================

auto CorrelationMatrix::findHighlyCorrelated(double threshold) const
    -> std::vector<CorrelationResult> {

    std::vector<CorrelationResult> results;

    for (size_t i = 0; i < symbols_.size(); ++i) {
        for (size_t j = i + 1; j < symbols_.size(); ++j) {
            double const corr = get(symbols_[i], symbols_[j]);

            if (std::abs(corr) >= threshold) {
                results.push_back(CorrelationResult{
                    .symbol1 = symbols_[i],
                    .symbol2 = symbols_[j],
                    .correlation = corr,
                    .p_value = 0.001,  // Stub
                    .sample_size = 100,  // Stub
                    .lag = 0,
                    .type = CorrelationType::Pearson
                });
            }
        }
    }

    return results;
}

} // namespace bigbrother::correlation
