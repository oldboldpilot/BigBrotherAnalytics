#pragma once

#include "correlation.hpp"
#include <optional>
#include <initializer_list>

namespace bigbrother::correlation {

/**
 * Fluent API for Correlation Analysis
 *
 * Provides intuitive, chainable interface for correlation operations.
 *
 * Example Usage:
 *
 *   // Calculate simple correlation
 *   auto corr = CorrelationAnalyzer()
 *       .between("AAPL", "MSFT")
 *       .using(prices_data)
 *       .pearson()
 *       .calculate();
 *
 *   // Find time-lagged relationships
 *   auto lag_analysis = CorrelationAnalyzer()
 *       .between("NVDA", "AMD")
 *       .using(prices_data)
 *       .laggedUpTo(30)  // Test 0-30 day lags
 *       .findOptimalLag();
 *
 *   // Build correlation matrix for portfolio
 *   auto matrix = CorrelationAnalyzer()
 *       .forSymbols({"AAPL", "MSFT", "GOOGL", "AMZN"})
 *       .using(prices_data)
 *       .buildMatrix();
 *
 *   // Find trading opportunities
 *   auto signals = CorrelationAnalyzer()
 *       .forSymbols(tech_stocks)
 *       .using(prices_data)
 *       .laggedUpTo(15)
 *       .generateSignals();
 */

class CorrelationAnalyzer {
public:
    CorrelationAnalyzer() = default;

    // Symbol selection
    [[nodiscard]] auto between(std::string symbol1, std::string symbol2) noexcept
        -> CorrelationAnalyzer& {
        symbol1_ = std::move(symbol1);
        symbol2_ = std::move(symbol2);
        return *this;
    }

    [[nodiscard]] auto forSymbols(std::vector<std::string> symbols) noexcept
        -> CorrelationAnalyzer& {
        symbols_ = std::move(symbols);
        return *this;
    }

    template<typename... Symbols>
    [[nodiscard]] auto forSymbols(Symbols&&... symbols) noexcept
        -> CorrelationAnalyzer& {
        symbols_.clear();
        (symbols_.push_back(std::forward<Symbols>(symbols)), ...);
        return *this;
    }

    // Data source
    [[nodiscard]] auto using_(std::map<std::string, TimeSeries> data) noexcept
        -> CorrelationAnalyzer& {
        data_ = std::move(data);
        return *this;
    }

    [[nodiscard]] auto withTimeSeries(TimeSeries ts1, TimeSeries ts2) noexcept
        -> CorrelationAnalyzer& {
        ts1_ = std::move(ts1);
        ts2_ = std::move(ts2);
        return *this;
    }

    // Correlation type
    [[nodiscard]] auto pearson() noexcept -> CorrelationAnalyzer& {
        type_ = CorrelationType::Pearson;
        return *this;
    }

    [[nodiscard]] auto spearman() noexcept -> CorrelationAnalyzer& {
        type_ = CorrelationType::Spearman;
        return *this;
    }

    // Time lag analysis
    [[nodiscard]] auto laggedUpTo(int max_lag) noexcept -> CorrelationAnalyzer& {
        max_lag_ = max_lag;
        use_lagged_ = true;
        return *this;
    }

    [[nodiscard]] auto rolling(size_t window_size) noexcept -> CorrelationAnalyzer& {
        window_size_ = window_size;
        use_rolling_ = true;
        return *this;
    }

    // Terminal operations

    /**
     * Calculate simple correlation
     */
    [[nodiscard]] auto calculate() const -> Result<double> {
        if (!ts1_ || !ts2_) {
            // Try to get from data map
            if (!symbol1_.empty() && !symbol2_.empty() && !data_.empty()) {
                auto it1 = data_.find(symbol1_);
                auto it2 = data_.find(symbol2_);

                if (it1 == data_.end() || it2 == data_.end()) {
                    return makeError<double>(
                        ErrorCode::DataNotAvailable,
                        "Time series not found in data"
                    );
                }

                ts1_ = it1->second;
                ts2_ = it2->second;
            } else {
                return makeError<double>(
                    ErrorCode::InvalidParameter,
                    "Time series not specified. Use withTimeSeries() or using_()"
                );
            }
        }

        if (type_ == CorrelationType::Pearson) {
            return CorrelationCalculator::pearson(
                std::span(ts1_->values),
                std::span(ts2_->values)
            );
        } else {
            return CorrelationCalculator::spearman(
                std::span(ts1_->values),
                std::span(ts2_->values)
            );
        }
    }

    /**
     * Find optimal time lag
     */
    [[nodiscard]] auto findOptimalLag() const
        -> Result<std::pair<int, double>> {

        if (!ts1_ || !ts2_) {
            return makeError<std::pair<int, double>>(
                ErrorCode::InvalidParameter,
                "Time series not specified"
            );
        }

        return CorrelationCalculator::findOptimalLag(
            std::span(ts1_->values),
            std::span(ts2_->values),
            max_lag_.value_or(30)
        );
    }

    /**
     * Calculate rolling correlation
     */
    [[nodiscard]] auto calculateRolling() const -> Result<std::vector<double>> {
        if (!ts1_ || !ts2_) {
            return makeError<std::vector<double>>(
                ErrorCode::InvalidParameter,
                "Time series not specified"
            );
        }

        return CorrelationCalculator::rollingCorrelation(
            std::span(ts1_->values),
            std::span(ts2_->values),
            window_size_.value_or(20)
        );
    }

    /**
     * Build correlation matrix
     */
    [[nodiscard]] auto buildMatrix() const -> Result<CorrelationMatrix> {
        if (symbols_.empty()) {
            return makeError<CorrelationMatrix>(
                ErrorCode::InvalidParameter,
                "No symbols specified. Use forSymbols()"
            );
        }

        if (data_.empty()) {
            return makeError<CorrelationMatrix>(
                ErrorCode::InvalidParameter,
                "No data specified. Use using_()"
            );
        }

        // Build time series vector
        std::vector<TimeSeries> series;
        series.reserve(symbols_.size());

        for (auto const& symbol : symbols_) {
            auto it = data_.find(symbol);
            if (it != data_.end()) {
                series.push_back(it->second);
            } else {
                LOG_WARN("Symbol {} not found in data, skipping", symbol);
            }
        }

        return CorrelationCalculator::correlationMatrix(series, type_);
    }

    /**
     * Generate trading signals from correlation analysis
     */
    [[nodiscard]] auto generateSignals(
        double min_correlation = 0.6,
        double min_confidence = 0.7
    ) const -> Result<std::vector<CorrelationSignalGenerator::CorrelationSignal>> {

        // First, find all lagged correlations
        if (symbols_.size() < 2) {
            return makeError<std::vector<CorrelationSignalGenerator::CorrelationSignal>>(
                ErrorCode::InvalidParameter,
                "Need at least 2 symbols for signal generation"
            );
        }

        std::vector<CorrelationResult> all_correlations;

        // Calculate for all pairs
        for (size_t i = 0; i < symbols_.size(); ++i) {
            for (size_t j = i + 1; j < symbols_.size(); ++j) {
                auto const& sym1 = symbols_[i];
                auto const& sym2 = symbols_[j];

                auto it1 = data_.find(sym1);
                auto it2 = data_.find(sym2);

                if (it1 == data_.end() || it2 == data_.end()) {
                    continue;
                }

                // Find optimal lag
                auto lag_result = CorrelationCalculator::findOptimalLag(
                    std::span(it1->second.values),
                    std::span(it2->second.values),
                    max_lag_.value_or(30)
                );

                if (lag_result) {
                    auto const [optimal_lag, correlation] = *lag_result;

                    // Calculate p-value
                    int const sample_size = static_cast<int>(it1->second.values.size());
                    double const p_value = CorrelationCalculator::calculatePValue(
                        correlation,
                        sample_size
                    );

                    CorrelationResult result{
                        .symbol1 = sym1,
                        .symbol2 = sym2,
                        .correlation = correlation,
                        .p_value = p_value,
                        .sample_size = sample_size,
                        .lag = optimal_lag,
                        .type = type_
                    };

                    all_correlations.push_back(result);
                }
            }
        }

        // Generate signals
        auto signals = CorrelationSignalGenerator::generateSignals(
            all_correlations,
            min_correlation,
            min_confidence
        );

        return signals;
    }

private:
    mutable std::optional<TimeSeries> ts1_;
    mutable std::optional<TimeSeries> ts2_;
    std::string symbol1_;
    std::string symbol2_;
    std::vector<std::string> symbols_;
    std::map<std::string, TimeSeries> data_;
    CorrelationType type_{CorrelationType::Pearson};
    bool use_lagged_{false};
    bool use_rolling_{false};
    std::optional<int> max_lag_;
    std::optional<size_t> window_size_;
};

/**
 * Convenience functions for quick correlation analysis
 */

// Calculate Pearson correlation between two symbols
[[nodiscard]] inline auto calculateCorrelation(
    std::string const& symbol1,
    std::string const& symbol2,
    std::map<std::string, TimeSeries> const& data
) -> Result<double> {
    return CorrelationAnalyzer()
        .between(symbol1, symbol2)
        .using_(data)
        .pearson()
        .calculate();
}

// Find optimal lag between two symbols
[[nodiscard]] inline auto findOptimalLag(
    std::string const& symbol1,
    std::string const& symbol2,
    std::map<std::string, TimeSeries> const& data,
    int max_lag = 30
) -> Result<std::pair<int, double>> {
    return CorrelationAnalyzer()
        .between(symbol1, symbol2)
        .using_(data)
        .laggedUpTo(max_lag)
        .findOptimalLag();
}

// Build correlation matrix for portfolio
[[nodiscard]] inline auto buildCorrelationMatrix(
    std::vector<std::string> const& symbols,
    std::map<std::string, TimeSeries> const& data
) -> Result<CorrelationMatrix> {
    return CorrelationAnalyzer()
        .forSymbols(symbols)
        .using_(data)
        .pearson()
        .buildMatrix();
}

} // namespace bigbrother::correlation
