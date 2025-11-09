/**
 * BigBrotherAnalytics - Explainability Module (C++23)
 *
 * Provides decision logging, feature importance analysis, and trade analysis
 * for regulatory compliance and strategy understanding.
 *
 * Following C++ Core Guidelines:
 * - I.2: Avoid non-const global variables
 * - F.51: Prefer default arguments over overloading
 * - Trailing return type syntax
 */

// Global module fragment
module;

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <map>
#include <algorithm>

// Module declaration
export module bigbrother.explainability;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::explainability {

using namespace bigbrother::types;
using namespace bigbrother::utils;

// ============================================================================
// Decision Logging
// ============================================================================

/**
 * Trade Decision Record
 * 
 * C.1: Use struct for passive data
 */
struct DecisionRecord {
    Timestamp timestamp{0};
    std::string symbol;
    std::string strategy_name;
    std::string decision;  // "BUY", "SELL", "HOLD"
    double confidence{0.0};
    std::map<std::string, double> features;
    std::string rationale;
};

/**
 * Decision Logger
 * 
 * Logs all trading decisions for audit and analysis
 * C.2: Use class when invariants exist
 */
class DecisionLogger {
public:
    /**
     * Get singleton instance
     * F.51: Default arguments
     */
    [[nodiscard]] static auto getInstance() -> DecisionLogger&;
    
    /**
     * Log a trading decision
     * 
     * F.15: Prefer conventional ways
     * F.20: Return by value
     */
    auto logDecision(DecisionRecord record) -> Result<void>;
    
    /**
     * Retrieve decision history
     * 
     * F.20: Return by value
     */
    [[nodiscard]] auto getHistory(
        std::string const& symbol,
        Timestamp start_time = 0,
        Timestamp end_time = 0
    ) -> Result<std::vector<DecisionRecord>>;
    
    // C.21: Rule of Five
    DecisionLogger(DecisionLogger const&) = delete;
    auto operator=(DecisionLogger const&) -> DecisionLogger& = delete;
    DecisionLogger(DecisionLogger&&) = delete;
    auto operator=(DecisionLogger&&) -> DecisionLogger& = delete;
    ~DecisionLogger() = default;
    
private:
    DecisionLogger() = default;
    
    std::vector<DecisionRecord> history_;
};

// ============================================================================
// Feature Importance Analysis
// ============================================================================

/**
 * Feature Importance Score
 */
struct FeatureImportance {
    std::string feature_name;
    double importance_score{0.0};  // 0.0 to 1.0
    double correlation{0.0};        // -1.0 to 1.0
};

/**
 * Feature Importance Analyzer
 * 
 * Analyzes which features contribute most to trading decisions
 */
class FeatureAnalyzer {
public:
    /**
     * Analyze feature importance for strategy
     * 
     * F.20: Return by value
     */
    [[nodiscard]] static auto analyzeStrategy(
        std::string const& strategy_name,
        std::vector<DecisionRecord> const& decisions
    ) -> Result<std::vector<FeatureImportance>>;
    
    /**
     * Get top N most important features
     */
    [[nodiscard]] static auto getTopFeatures(
        std::vector<FeatureImportance> const& features,
        size_t n = 10
    ) -> std::vector<FeatureImportance>;
};

// ============================================================================
// Trade Analysis
// ============================================================================

/**
 * Trade Performance Metrics
 */
struct TradeMetrics {
    size_t total_trades{0};
    size_t winning_trades{0};
    size_t losing_trades{0};
    double win_rate{0.0};
    double avg_profit{0.0};
    double avg_loss{0.0};
    double profit_factor{0.0};
    double sharpe_ratio{0.0};
    double max_drawdown{0.0};
    
    /**
     * Calculate win rate
     * F.4: constexpr
     */
    [[nodiscard]] constexpr auto calculateWinRate() const noexcept -> double {
        return total_trades > 0 
            ? static_cast<double>(winning_trades) / static_cast<double>(total_trades)
            : 0.0;
    }
};

/**
 * Trade Analyzer
 * 
 * Analyzes trade performance and generates reports
 */
class TradeAnalyzer {
public:
    /**
     * Analyze trades for symbol
     */
    [[nodiscard]] static auto analyzeTrades(
        std::string const& symbol,
        Timestamp start_time = 0,
        Timestamp end_time = 0
    ) -> Result<TradeMetrics>;
    
    /**
     * Generate performance report
     */
    [[nodiscard]] static auto generateReport(
        std::vector<DecisionRecord> const& decisions
    ) -> Result<std::string>;
};

// ============================================================================
// Module Implementation (inline for stubs)
// ============================================================================

inline auto DecisionLogger::getInstance() -> DecisionLogger& {
    static DecisionLogger instance;
    return instance;
}

inline auto DecisionLogger::logDecision(DecisionRecord record) -> Result<void> {
    Logger::getInstance().info("Decision logged for {}", record.symbol);
    history_.push_back(std::move(record));
    return {};
}

inline auto DecisionLogger::getHistory(
    std::string const& symbol,
    Timestamp start_time,
    Timestamp end_time
) -> Result<std::vector<DecisionRecord>> {
    std::vector<DecisionRecord> filtered;
    
    for (auto const& record : history_) {
        if (record.symbol == symbol) {
            if ((start_time == 0 || record.timestamp >= start_time) &&
                (end_time == 0 || record.timestamp <= end_time)) {
                filtered.push_back(record);
            }
        }
    }
    
    return filtered;
}

inline auto FeatureAnalyzer::analyzeStrategy(
    std::string const& strategy_name,
    std::vector<DecisionRecord> const& decisions
) -> Result<std::vector<FeatureImportance>> {
    // Stub implementation
    std::vector<FeatureImportance> result;
    Logger::getInstance().info("Feature analysis for strategy: {}", strategy_name);
    return result;
}

inline auto FeatureAnalyzer::getTopFeatures(
    std::vector<FeatureImportance> const& features,
    size_t n
) -> std::vector<FeatureImportance> {
    auto sorted = features;
    std::sort(sorted.begin(), sorted.end(),
        [](auto const& a, auto const& b) -> bool { return a.importance_score > b.importance_score; });
    
    if (sorted.size() > n) {
        sorted.resize(n);
    }
    
    return sorted;
}

inline auto TradeAnalyzer::analyzeTrades(
    std::string const& symbol,
    Timestamp start_time,
    Timestamp end_time
) -> Result<TradeMetrics> {
    // Stub implementation
    TradeMetrics metrics;
    Logger::getInstance().info("Trade analysis for symbol: {}", symbol);
    return metrics;
}

inline auto TradeAnalyzer::generateReport(
    std::vector<DecisionRecord> const& decisions
) -> Result<std::string> {
    // Stub implementation
    return std::string("Performance Report\n==================\nStub implementation");
}

} // export namespace bigbrother::explainability
