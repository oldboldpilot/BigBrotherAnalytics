#include "correlation.hpp"
#include "../utils/logger.hpp"
#include "../utils/timer.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <numbers>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bigbrother::correlation {

using std::numbers::sqrt2;
using std::numbers::pi;

/**
 * Pearson Correlation Implementation
 *
 * Measures linear relationship between two variables.
 *
 * Formula:
 *   ρ(X,Y) = Cov(X,Y) / (σ_X * σ_Y)
 *          = Σ((X_i - μ_X)(Y_i - μ_Y)) / √(Σ(X_i - μ_X)² * Σ(Y_i - μ_Y)²)
 *
 * Properties:
 *   - Range: [-1, +1]
 *   - ρ = +1: Perfect positive linear relationship
 *   - ρ = 0: No linear relationship
 *   - ρ = -1: Perfect negative linear relationship
 *
 * Performance: O(n) time, O(1) space
 */

[[nodiscard]] auto CorrelationCalculator::pearson(
    std::span<double const> x,
    std::span<double const> y
) noexcept -> Result<double> {

    PROFILE_SCOPE("CorrelationCalculator::pearson");

    // Validate inputs
    if (x.empty() || y.empty()) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Empty input arrays"
        );
    }

    if (x.size() != y.size()) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Array size mismatch"
        );
    }

    if (x.size() < 2) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Need at least 2 data points"
        );
    }

    size_t const n = x.size();

    // Calculate means
    double const mean_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
    double const mean_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);

    // Calculate sums
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    double sum_yy = 0.0;

    // Single pass for all sums (cache-friendly)
    for (size_t i = 0; i < n; ++i) {
        double const dx = x[i] - mean_x;
        double const dy = y[i] - mean_y;

        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    // Check for zero variance (constant series)
    if (sum_xx == 0.0 || sum_yy == 0.0) {
        return 0.0;  // Correlation undefined, return 0
    }

    // Calculate correlation
    double const correlation = sum_xy / std::sqrt(sum_xx * sum_yy);

    // Clamp to valid range (handle floating point errors)
    double const clamped = std::clamp(correlation, -1.0, 1.0);

    return clamped;
}

/**
 * Calculate p-value for correlation
 *
 * Tests null hypothesis: ρ = 0
 *
 * Test statistic:
 *   t = ρ * √((n-2) / (1-ρ²))
 *
 * Follows Student's t-distribution with (n-2) degrees of freedom.
 *
 * For large n, approximates normal distribution.
 */
[[nodiscard]] constexpr auto CorrelationCalculator::calculatePValue(
    double correlation,
    int sample_size
) noexcept -> double {

    if (sample_size < 3) {
        return 1.0;  // Not enough data for significance
    }

    // Calculate t-statistic
    double const r = std::abs(correlation);
    double const df = static_cast<double>(sample_size - 2);

    // Check for perfect correlation
    if (r >= 1.0 - 1e-10) {
        return 0.0;  // Perfectly correlated
    }

    double const t = r * std::sqrt(df / (1.0 - r * r));

    // For large samples, use normal approximation
    // For small samples, would need Student's t-distribution
    // Using simplified Fisher transformation for now

    // Fisher's z-transformation
    double const z = 0.5 * std::log((1.0 + r) / (1.0 - r));
    double const se = 1.0 / std::sqrt(df);
    double const z_score = std::abs(z / se);

    // Convert to p-value (two-tailed test)
    // Using complementary error function approximation
    double const p = std::erfc(z_score / sqrt2);

    return p;
}

/**
 * Rolling Correlation
 *
 * Calculates correlation over sliding windows.
 * Useful for detecting regime changes.
 */
[[nodiscard]] auto CorrelationCalculator::rollingCorrelation(
    std::span<double const> x,
    std::span<double const> y,
    size_t window_size
) noexcept -> Result<std::vector<double>> {

    PROFILE_SCOPE("CorrelationCalculator::rollingCorrelation");

    if (x.size() != y.size()) {
        return makeError<std::vector<double>>(
            ErrorCode::InvalidParameter,
            "Array size mismatch"
        );
    }

    if (window_size < 2 || window_size > x.size()) {
        return makeError<std::vector<double>>(
            ErrorCode::InvalidParameter,
            "Invalid window size"
        );
    }

    size_t const n = x.size();
    size_t const num_windows = n - window_size + 1;

    std::vector<double> correlations;
    correlations.reserve(num_windows);

    // Calculate correlation for each window
    for (size_t i = 0; i < num_windows; ++i) {
        auto x_window = x.subspan(i, window_size);
        auto y_window = y.subspan(i, window_size);

        auto corr_result = pearson(x_window, y_window);

        if (corr_result) {
            correlations.push_back(*corr_result);
        } else {
            correlations.push_back(0.0);  // Use 0 for errors
        }
    }

    return correlations;
}

/**
 * Correlation Matrix
 *
 * Computes all pairwise correlations.
 * Parallelized with OpenMP for speed.
 *
 * Complexity: O(n² * m) where n = # symbols, m = # data points
 * With OpenMP: O(n² * m / p) where p = # threads
 */
[[nodiscard]] auto CorrelationCalculator::correlationMatrix(
    std::vector<TimeSeries> const& series,
    CorrelationType type
) noexcept -> Result<CorrelationMatrix> {

    PROFILE_SCOPE("CorrelationCalculator::correlationMatrix");

    if (series.empty()) {
        return makeError<CorrelationMatrix>(
            ErrorCode::InvalidParameter,
            "Empty series vector"
        );
    }

    // Validate all series
    for (auto const& ts : series) {
        if (auto validation = ts.validate(); !validation) {
            return std::unexpected(validation.error());
        }
    }

    // Extract symbols
    std::vector<std::string> symbols;
    symbols.reserve(series.size());
    for (auto const& ts : series) {
        symbols.push_back(ts.symbol);
    }

    CorrelationMatrix matrix{symbols};

    size_t const n = series.size();

    LOG_INFO("Calculating {}x{} correlation matrix ({} type)",
             n, n,
             type == CorrelationType::Pearson ? "Pearson" : "Spearman");

    auto const start_time = utils::Timer::timepoint();

    // Parallel computation of upper triangle
    // (matrix is symmetric, so we only compute half)
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            if (i == j) {
                // Diagonal: perfect self-correlation
                matrix.set(series[i].symbol, series[j].symbol, 1.0);
            } else {
                // Calculate correlation
                Result<double> corr_result;

                if (type == CorrelationType::Pearson) {
                    corr_result = pearson(
                        std::span(series[i].values),
                        std::span(series[j].values)
                    );
                } else if (type == CorrelationType::Spearman) {
                    corr_result = spearman(
                        std::span(series[i].values),
                        std::span(series[j].values)
                    );
                } else {
                    corr_result = makeError<double>(
                        ErrorCode::InvalidParameter,
                        "Unsupported correlation type"
                    );
                }

                double correlation = 0.0;
                if (corr_result) {
                    correlation = *corr_result;
                } else {
                    LOG_WARN("Failed to calculate correlation between {} and {}: {}",
                            series[i].symbol, series[j].symbol,
                            corr_result.error().message);
                }

                // Set both (i,j) and (j,i) since matrix is symmetric
                #pragma omp critical
                {
                    matrix.set(series[i].symbol, series[j].symbol, correlation);
                    matrix.set(series[j].symbol, series[i].symbol, correlation);
                }
            }
        }

        // Progress logging
        if (i % 10 == 0) {
            #pragma omp critical
            {
                LOG_DEBUG("Correlation matrix progress: {}/{}", i + 1, n);
            }
        }
    }

    auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        utils::Timer::timepoint() - start_time
    ).count();

    LOG_INFO("Correlation matrix calculated in {} ms ({} correlations)",
             elapsed, (n * (n - 1)) / 2);

    return matrix;
}

// CorrelationMatrix implementation
CorrelationMatrix::CorrelationMatrix(std::vector<std::string> symbols)
    : symbols_{std::move(symbols)} {}

auto CorrelationMatrix::set(
    std::string const& symbol1,
    std::string const& symbol2,
    double correlation
) -> void {
    matrix_[{symbol1, symbol2}] = correlation;
}

[[nodiscard]] auto CorrelationMatrix::get(
    std::string const& symbol1,
    std::string const& symbol2
) const -> double {
    auto it = matrix_.find({symbol1, symbol2});
    if (it != matrix_.end()) {
        return it->second;
    }
    return 0.0;  // Default to 0 if not found
}

[[nodiscard]] auto CorrelationMatrix::getSymbols() const noexcept
    -> std::vector<std::string> const& {
    return symbols_;
}

[[nodiscard]] auto CorrelationMatrix::size() const noexcept -> size_t {
    return symbols_.size();
}

[[nodiscard]] auto CorrelationMatrix::findHighlyCorrelated(double threshold) const
    -> std::vector<CorrelationResult> {

    std::vector<CorrelationResult> results;

    for (size_t i = 0; i < symbols_.size(); ++i) {
        for (size_t j = i + 1; j < symbols_.size(); ++j) {
            double const corr = get(symbols_[i], symbols_[j]);

            if (std::abs(corr) >= threshold) {
                CorrelationResult result{
                    .symbol1 = symbols_[i],
                    .symbol2 = symbols_[j],
                    .correlation = corr,
                    .p_value = 0.0,  // TODO: calculate from sample size
                    .sample_size = 0,
                    .lag = 0,
                    .type = CorrelationType::Pearson
                };
                results.push_back(result);
            }
        }
    }

    // Sort by absolute correlation (strongest first)
    std::ranges::sort(results, [](auto const& a, auto const& b) {
        return std::abs(a.correlation) > std::abs(b.correlation);
    });

    return results;
}

[[nodiscard]] auto CorrelationMatrix::toCSV() const -> std::string {
    std::ostringstream oss;

    // Header row
    oss << "Symbol";
    for (auto const& symbol : symbols_) {
        oss << "," << symbol;
    }
    oss << "\n";

    // Data rows
    for (auto const& row_symbol : symbols_) {
        oss << row_symbol;
        for (auto const& col_symbol : symbols_) {
            oss << "," << get(row_symbol, col_symbol);
        }
        oss << "\n";
    }

    return oss.str();
}

} // namespace bigbrother::correlation
