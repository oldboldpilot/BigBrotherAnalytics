#pragma once

#include "../utils/types.hpp"
#include "../utils/math.hpp"
#include <vector>
#include <span>
#include <unordered_map>
#include <memory>
#include <concepts>
#include <ranges>

namespace bigbrother::correlation {

using namespace types;

// Hash function for std::pair (for unordered_map keys)
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        // Combine hashes using shift and XOR
        return hash1 ^ (hash2 << 1);
    }
};

/**
 * Correlation Engine
 *
 * High-performance statistical correlation analysis for trading signals.
 * Discovers relationships between securities across multiple timeframes.
 *
 * Key Features:
 * - Pearson correlation (linear relationships)
 * - Spearman correlation (rank-based, non-linear)
 * - Time-lagged cross-correlation (identify leading/lagging indicators)
 * - Rolling window correlations (detect regime changes)
 * - MPI parallelization for massive datasets (32+ cores)
 * - OpenMP multi-threading within nodes
 * - Intel MKL optimization for linear algebra
 *
 * Performance Targets (per PRD):
 * - 1000x1000 correlation matrix: < 10 seconds
 * - Near-linear scaling with core count
 * - Time-lagged correlations (0-30 day lags): < 30 seconds
 *
 * Use Cases:
 * - Find correlated securities for pairs trading
 * - Identify leading indicators (lag analysis)
 * - Detect sector rotation patterns
 * - Options strategy correlation (NVDA earnings → AMD)
 * - Economic indicator relationships
 */

/**
 * Correlation Type
 */
enum class CorrelationType {
    Pearson,        // Linear correlation (-1 to +1)
    Spearman,       // Rank correlation (non-linear)
    Kendall,        // Tau correlation (ordinal)
    Distance        // Distance correlation (non-linear)
};

/**
 * Correlation Result
 */
struct CorrelationResult {
    std::string symbol1;
    std::string symbol2;
    double correlation;         // Correlation coefficient
    double p_value;             // Statistical significance
    int sample_size;            // Number of data points
    int lag;                    // Time lag in periods (0 = contemporaneous)
    CorrelationType type;

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
 * Correlation Matrix
 *
 * NxN matrix of pairwise correlations
 */
class CorrelationMatrix {
public:
    CorrelationMatrix() = default;
    explicit CorrelationMatrix(std::vector<std::string> symbols);

    /**
     * Set correlation value
     */
    auto set(std::string const& symbol1, std::string const& symbol2, double correlation)
        -> void;

    /**
     * Get correlation value
     */
    [[nodiscard]] auto get(std::string const& symbol1, std::string const& symbol2) const
        -> double;

    /**
     * Get all symbols
     */
    [[nodiscard]] auto getSymbols() const noexcept -> std::vector<std::string> const&;

    /**
     * Get matrix size
     */
    [[nodiscard]] auto size() const noexcept -> size_t;

    /**
     * Find highly correlated pairs (|correlation| > threshold)
     */
    [[nodiscard]] auto findHighlyCorrelated(double threshold = 0.7) const
        -> std::vector<CorrelationResult>;

    /**
     * Export to CSV
     */
    [[nodiscard]] auto toCSV() const -> std::string;

private:
    std::vector<std::string> symbols_;
    std::unordered_map<std::pair<std::string, std::string>, double, PairHash> matrix_;
};

/**
 * Time Series Data for Correlation
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
 * Correlation Calculator
 *
 * Core correlation computation engine
 */
class CorrelationCalculator {
public:
    /**
     * Calculate Pearson correlation
     *
     * Measures linear relationship between two variables.
     * ρ = Cov(X,Y) / (σ_X * σ_Y)
     *
     * @param x First time series
     * @param y Second time series
     * @return Correlation coefficient [-1, +1]
     */
    [[nodiscard]] static auto pearson(
        std::span<double const> x,
        std::span<double const> y
    ) noexcept -> Result<double>;

    /**
     * Calculate Spearman rank correlation
     *
     * Non-parametric measure of monotonic relationship.
     * Works with non-linear relationships.
     *
     * @param x First time series
     * @param y Second time series
     * @return Correlation coefficient [-1, +1]
     */
    [[nodiscard]] static auto spearman(
        std::span<double const> x,
        std::span<double const> y
    ) noexcept -> Result<double>;

    /**
     * Calculate time-lagged cross-correlation
     *
     * CRITICAL FOR TRADING: Identifies leading/lagging relationships.
     * Example: NVDA earnings announcement → AMD stock movement (15-min lag)
     *
     * @param x Leading series
     * @param y Lagging series
     * @param max_lag Maximum lag to test (in periods)
     * @return Vector of correlations at each lag [0, max_lag]
     */
    [[nodiscard]] static auto crossCorrelation(
        std::span<double const> x,
        std::span<double const> y,
        int max_lag = 30
    ) noexcept -> Result<std::vector<double>>;

    /**
     * Find optimal lag with maximum correlation
     *
     * @return (optimal_lag, correlation_at_lag)
     */
    [[nodiscard]] static auto findOptimalLag(
        std::span<double const> x,
        std::span<double const> y,
        int max_lag = 30
    ) noexcept -> Result<std::pair<int, double>>;

    /**
     * Calculate rolling correlation
     *
     * Detects regime changes in correlation structure.
     *
     * @param x First time series
     * @param y Second time series
     * @param window_size Rolling window size
     * @return Vector of correlations over time
     */
    [[nodiscard]] static auto rollingCorrelation(
        std::span<double const> x,
        std::span<double const> y,
        size_t window_size = 20
    ) noexcept -> Result<std::vector<double>>;

    /**
     * Calculate correlation matrix for multiple series
     *
     * Computes all pairwise correlations.
     * Parallelized with OpenMP for speed.
     *
     * @param series Vector of time series
     * @param type Correlation type (Pearson, Spearman)
     * @return NxN correlation matrix
     */
    [[nodiscard]] static auto correlationMatrix(
        std::vector<TimeSeries> const& series,
        CorrelationType type = CorrelationType::Pearson
    ) noexcept -> Result<CorrelationMatrix>;

    /**
     * Calculate p-value for correlation
     *
     * Tests null hypothesis that correlation = 0
     *
     * @param correlation Correlation coefficient
     * @param sample_size Number of samples
     * @return p-value
     */
    [[nodiscard]] static constexpr auto calculatePValue(
        double correlation,
        int sample_size
    ) noexcept -> double;
};

/**
 * Parallel Correlation Engine
 *
 * MPI + OpenMP parallelized correlation computation
 * for massive datasets (10+ years, 500+ symbols)
 */
class ParallelCorrelationEngine {
public:
    ParallelCorrelationEngine();
    ~ParallelCorrelationEngine();

    // Delete copy, allow move
    ParallelCorrelationEngine(ParallelCorrelationEngine const&) = delete;
    auto operator=(ParallelCorrelationEngine const&) = delete;
    ParallelCorrelationEngine(ParallelCorrelationEngine&&) noexcept;
    auto operator=(ParallelCorrelationEngine&&) noexcept -> ParallelCorrelationEngine&;

    /**
     * Initialize MPI (if available)
     */
    [[nodiscard]] auto initialize(int* argc, char*** argv) -> Result<void>;

    /**
     * Finalize MPI
     */
    auto finalize() -> void;

    /**
     * Calculate correlation matrix in parallel
     *
     * Distributes computation across MPI ranks and OpenMP threads.
     * Near-linear scaling with core count.
     *
     * @param series Time series data
     * @param type Correlation type
     * @return Correlation matrix
     */
    [[nodiscard]] auto calculateMatrix(
        std::vector<TimeSeries> const& series,
        CorrelationType type = CorrelationType::Pearson
    ) -> Result<CorrelationMatrix>;

    /**
     * Calculate time-lagged correlations in parallel
     *
     * For each pair of series, find optimal lag.
     * Extremely computationally intensive - benefits greatly from parallelization.
     *
     * @param series Time series data
     * @param max_lag Maximum lag to test
     * @return Vector of optimal lags for each pair
     */
    [[nodiscard]] auto calculateLaggedCorrelations(
        std::vector<TimeSeries> const& series,
        int max_lag = 30
    ) -> Result<std::vector<CorrelationResult>>;

    /**
     * Get MPI rank and size
     */
    [[nodiscard]] auto getMPIInfo() const noexcept
        -> std::pair<int, int>;  // (rank, size)

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Correlation Signal Generator
 *
 * Generates trading signals from correlation analysis
 */
class CorrelationSignalGenerator {
public:
    struct CorrelationSignal {
        std::string leading_symbol;     // Leading indicator
        std::string lagging_symbol;     // Symbol to trade
        int optimal_lag;                // Time lag in periods
        double correlation;             // Correlation strength
        double confidence;              // Signal confidence [0, 1]
        std::string rationale;          // Human-readable explanation

        [[nodiscard]] auto isActionable() const noexcept -> bool {
            return std::abs(correlation) > 0.6 && confidence > 0.7;
        }
    };

    /**
     * Generate signals from correlation analysis
     *
     * @param correlations Correlation results
     * @param min_correlation Minimum correlation threshold
     * @param min_confidence Minimum confidence threshold
     * @return Trading signals
     */
    [[nodiscard]] static auto generateSignals(
        std::vector<CorrelationResult> const& correlations,
        double min_correlation = 0.6,
        double min_confidence = 0.7
    ) -> std::vector<CorrelationSignal>;

    /**
     * Detect correlation breakdown (regime change)
     *
     * When historical correlation suddenly breaks, it may signal
     * a trading opportunity (mean reversion or trend continuation).
     *
     * @param symbol1 First symbol
     * @param symbol2 Second symbol
     * @param historical_correlation Historical correlation
     * @param recent_correlation Recent correlation
     * @return Signal if breakdown detected
     */
    [[nodiscard]] static auto detectCorrelationBreakdown(
        std::string symbol1,
        std::string symbol2,
        double historical_correlation,
        double recent_correlation
    ) -> std::optional<CorrelationSignal>;
};

/**
 * Sector Correlation Analyzer
 *
 * Analyzes correlations within and across sectors
 */
class SectorCorrelationAnalyzer {
public:
    /**
     * Calculate intra-sector correlation
     *
     * How correlated are stocks within the same sector?
     *
     * @param sector_symbols Symbols in the sector
     * @param prices Price data for all symbols
     * @return Average intra-sector correlation
     */
    [[nodiscard]] static auto calculateIntraSectorCorrelation(
        std::vector<std::string> const& sector_symbols,
        std::unordered_map<std::string, TimeSeries> const& prices
    ) -> Result<double>;

    /**
     * Detect sector rotation
     *
     * Identify when capital is moving from one sector to another
     * based on changing correlation patterns.
     *
     * @param sector1_symbols First sector symbols
     * @param sector2_symbols Second sector symbols
     * @param prices Price data
     * @return Rotation signal if detected
     */
    [[nodiscard]] static auto detectSectorRotation(
        std::vector<std::string> const& sector1_symbols,
        std::vector<std::string> const& sector2_symbols,
        std::unordered_map<std::string, TimeSeries> const& prices
    ) -> std::optional<CorrelationSignalGenerator::CorrelationSignal>;
};

} // namespace bigbrother::correlation
