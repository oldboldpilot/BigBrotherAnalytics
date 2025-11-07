#include "correlation.hpp"
#include "../utils/logger.hpp"
#include "../utils/timer.hpp"

#include <algorithm>
#include <ranges>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bigbrother::correlation {

/**
 * Time-Lagged Cross-Correlation
 *
 * CRITICAL FOR ALGORITHMIC TRADING
 *
 * Identifies leading/lagging relationships between securities.
 * For example:
 * - NVDA earnings announcement → AMD stock movement (15-min lag)
 * - Fed rate decision → Bank stocks (30-min lag)
 * - Oil prices → Airline stocks (1-day lag)
 *
 * This is MORE VALUABLE than contemporaneous correlation because
 * it tells us WHEN the relationship manifests, not just IF it exists.
 *
 * Formula:
 *   C_xy(τ) = Correlation(X_t, Y_{t+τ})
 *
 * Where τ is the time lag in periods.
 *
 * Example:
 *   If C_xy(5) = 0.85, it means:
 *   "Changes in X predict changes in Y with a 5-period lag"
 *
 * Trading Strategy:
 *   1. Detect X moves significantly
 *   2. Predict Y will move similarly in τ periods
 *   3. Trade Y immediately to capitalize on predictable move
 */

[[nodiscard]] auto CorrelationCalculator::crossCorrelation(
    std::span<double const> x,
    std::span<double const> y,
    int max_lag
) noexcept -> Result<std::vector<double>> {

    PROFILE_SCOPE("CorrelationCalculator::crossCorrelation");

    // Validate inputs
    if (x.size() != y.size()) {
        return makeError<std::vector<double>>(
            ErrorCode::InvalidParameter,
            "Array size mismatch"
        );
    }

    if (max_lag < 0 || static_cast<size_t>(max_lag) >= x.size()) {
        return makeError<std::vector<double>>(
            ErrorCode::InvalidParameter,
            "Invalid max_lag"
        );
    }

    size_t const n = x.size();
    std::vector<double> correlations;
    correlations.reserve(max_lag + 1);

    LOG_DEBUG("Calculating cross-correlation with max_lag={}", max_lag);

    // Calculate correlation at each lag
    // lag=0 means contemporaneous (no lag)
    // lag=k means Y is shifted forward by k periods
    for (int lag = 0; lag <= max_lag; ++lag) {
        // Effective sample size after shifting
        size_t const effective_n = n - static_cast<size_t>(lag);

        if (effective_n < 2) {
            correlations.push_back(0.0);
            continue;
        }

        // X: use first effective_n points
        // Y: use points from lag to end (shifted by lag)
        auto x_subset = x.subspan(0, effective_n);
        auto y_subset = y.subspan(static_cast<size_t>(lag), effective_n);

        auto corr_result = pearson(x_subset, y_subset);

        if (corr_result) {
            correlations.push_back(*corr_result);
        } else {
            LOG_WARN("Failed to calculate correlation at lag {}", lag);
            correlations.push_back(0.0);
        }
    }

    return correlations;
}

/**
 * Find Optimal Lag
 *
 * Finds the time lag that maximizes correlation.
 *
 * This is the KEY insight for trading:
 * - Lag = 0: Trade immediately (no predictive power)
 * - Lag > 0: We have time to trade before Y moves
 * - Larger |correlation| = stronger predictive relationship
 *
 * Returns: (optimal_lag, correlation_at_lag)
 */
[[nodiscard]] auto CorrelationCalculator::findOptimalLag(
    std::span<double const> x,
    std::span<double const> y,
    int max_lag
) noexcept -> Result<std::pair<int, double>> {

    PROFILE_SCOPE("CorrelationCalculator::findOptimalLag");

    // Get all lag correlations
    auto corr_result = crossCorrelation(x, y, max_lag);

    if (!corr_result) {
        return std::unexpected(corr_result.error());
    }

    auto const& correlations = *corr_result;

    // Find lag with maximum absolute correlation
    int optimal_lag = 0;
    double max_abs_corr = 0.0;

    for (int lag = 0; lag <= max_lag; ++lag) {
        double const abs_corr = std::abs(correlations[static_cast<size_t>(lag)]);

        if (abs_corr > max_abs_corr) {
            max_abs_corr = abs_corr;
            optimal_lag = lag;
        }
    }

    double const optimal_corr = correlations[static_cast<size_t>(optimal_lag)];

    LOG_INFO("Optimal lag: {} periods (correlation: {:.3f})",
             optimal_lag, optimal_corr);

    return std::make_pair(optimal_lag, optimal_corr);
}

/**
 * Spearman Rank Correlation
 *
 * Non-parametric correlation that works with:
 * - Non-linear monotonic relationships
 * - Ordinal data
 * - Outlier-resistant
 *
 * Algorithm:
 * 1. Convert values to ranks
 * 2. Calculate Pearson correlation on ranks
 *
 * Equivalent to Pearson on ranked data.
 */
[[nodiscard]] auto CorrelationCalculator::spearman(
    std::span<double const> x,
    std::span<double const> y
) noexcept -> Result<double> {

    PROFILE_SCOPE("CorrelationCalculator::spearman");

    if (x.size() != y.size() || x.empty()) {
        return makeError<double>(
            ErrorCode::InvalidParameter,
            "Invalid input arrays"
        );
    }

    size_t const n = x.size();

    // Convert to ranks
    auto rankify = [](std::span<double const> values) -> std::vector<double> {
        std::vector<std::pair<double, size_t>> indexed;
        indexed.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            indexed.emplace_back(values[i], i);
        }

        // Sort by value
        std::ranges::sort(indexed, [](auto const& a, auto const& b) {
            return a.first < b.first;
        });

        // Assign ranks (handle ties by averaging)
        std::vector<double> ranks(values.size());

        for (size_t i = 0; i < indexed.size(); ) {
            // Find range of tied values
            size_t j = i;
            while (j < indexed.size() && indexed[j].first == indexed[i].first) {
                ++j;
            }

            // Average rank for ties
            double const avg_rank = static_cast<double>(i + j - 1) / 2.0 + 1.0;

            // Assign average rank to all tied values
            for (size_t k = i; k < j; ++k) {
                ranks[indexed[k].second] = avg_rank;
            }

            i = j;
        }

        return ranks;
    };

    auto x_ranks = rankify(x);
    auto y_ranks = rankify(y);

    // Calculate Pearson correlation on ranks
    return pearson(
        std::span(x_ranks),
        std::span(y_ranks)
    );
}

/**
 * Correlation Signal Generator
 *
 * Converts correlation analysis into actionable trading signals
 */
[[nodiscard]] auto CorrelationSignalGenerator::generateSignals(
    std::vector<CorrelationResult> const& correlations,
    double min_correlation,
    double min_confidence
) -> std::vector<CorrelationSignal> {

    PROFILE_SCOPE("CorrelationSignalGenerator::generateSignals");

    std::vector<CorrelationSignal> signals;

    for (auto const& corr : correlations) {
        // Skip weak correlations
        if (std::abs(corr.correlation) < min_correlation) {
            continue;
        }

        // Skip non-significant correlations
        if (!corr.isSignificant()) {
            continue;
        }

        // Calculate confidence based on:
        // 1. Correlation strength
        // 2. Statistical significance (p-value)
        // 3. Sample size
        double confidence = std::abs(corr.correlation);

        // Adjust for p-value
        confidence *= (1.0 - corr.p_value);

        // Adjust for sample size (more data = more confidence)
        if (corr.sample_size > 100) {
            confidence *= 1.0;
        } else if (corr.sample_size > 50) {
            confidence *= 0.9;
        } else {
            confidence *= 0.7;
        }

        if (confidence < min_confidence) {
            continue;
        }

        // Generate signal
        CorrelationSignal signal;

        // If lag > 0, symbol1 leads symbol2
        if (corr.lag > 0) {
            signal.leading_symbol = corr.symbol1;
            signal.lagging_symbol = corr.symbol2;
        } else {
            // For lag=0 (contemporaneous), use alphabetical order
            signal.leading_symbol = corr.symbol1;
            signal.lagging_symbol = corr.symbol2;
        }

        signal.optimal_lag = corr.lag;
        signal.correlation = corr.correlation;
        signal.confidence = confidence;

        // Generate rationale
        std::ostringstream rationale;
        rationale << "Strong "
                  << (corr.correlation > 0 ? "positive" : "negative")
                  << " correlation (" << std::fixed << std::setprecision(2)
                  << corr.correlation << ") ";

        if (corr.lag > 0) {
            rationale << "with " << corr.lag << "-period lag. "
                     << signal.leading_symbol << " leads "
                     << signal.lagging_symbol << " by " << corr.lag << " periods.";
        } else {
            rationale << "(contemporaneous). Symbols move together.";
        }

        signal.rationale = rationale.str();

        signals.push_back(signal);
    }

    // Sort by confidence (highest first)
    std::ranges::sort(signals, [](auto const& a, auto const& b) {
        return a.confidence > b.confidence;
    });

    LOG_INFO("Generated {} correlation signals from {} correlations",
             signals.size(), correlations.size());

    return signals;
}

/**
 * Detect Correlation Breakdown
 *
 * When historical correlation breaks down, it may signal:
 * 1. Mean reversion opportunity (correlation will restore)
 * 2. Regime change (new trading environment)
 *
 * Example:
 * - Tech stocks historically correlated 0.85
 * - Recent correlation drops to 0.40
 * - Signal: Either mean reversion trade OR avoid pair trading
 */
[[nodiscard]] auto CorrelationSignalGenerator::detectCorrelationBreakdown(
    std::string symbol1,
    std::string symbol2,
    double historical_correlation,
    double recent_correlation
) -> std::optional<CorrelationSignal> {

    // Calculate breakdown magnitude
    double const breakdown = std::abs(historical_correlation - recent_correlation);

    // Require significant breakdown (> 0.30 change)
    if (breakdown < 0.30) {
        return std::nullopt;
    }

    CorrelationSignal signal;
    signal.leading_symbol = symbol1;
    signal.lagging_symbol = symbol2;
    signal.optimal_lag = 0;
    signal.correlation = recent_correlation;
    signal.confidence = breakdown;  // Higher breakdown = higher confidence

    std::ostringstream rationale;
    rationale << "Correlation breakdown detected. "
              << "Historical: " << std::fixed << std::setprecision(2)
              << historical_correlation << ", "
              << "Recent: " << recent_correlation << ". "
              << "Breakdown magnitude: " << breakdown << ". ";

    if (std::abs(recent_correlation) < std::abs(historical_correlation)) {
        rationale << "Correlation weakening - may signal regime change or mean reversion opportunity.";
    } else {
        rationale << "Correlation strengthening - may signal new trading relationship.";
    }

    signal.rationale = rationale.str();

    LOG_WARN("Correlation breakdown: {} vs {} (Δ={:.2f})",
             symbol1, symbol2, breakdown);

    return signal;
}

} // namespace bigbrother::correlation
