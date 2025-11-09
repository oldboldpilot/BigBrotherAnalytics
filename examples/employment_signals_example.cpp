/**
 * BigBrotherAnalytics - Employment Signals Integration Example
 *
 * Demonstrates how to populate StrategyContext with employment signals
 * and use them in trading strategies.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

import bigbrother.strategy;
import bigbrother.employment.signals;
import bigbrother.utils.logger;

using namespace bigbrother::strategy;
using namespace bigbrother::employment;

/**
 * Example 1: Populate StrategyContext with Employment Signals
 */
auto populateContextWithEmploymentData() -> StrategyContext {
    StrategyContext context;

    // Set basic context data
    context.account_value = 100000.0;
    context.available_capital = 50000.0;
    context.current_time = 1699564800; // Nov 9, 2025

    // Initialize employment signal generator
    EmploymentSignalGenerator generator("scripts", "data/bigbrother.duckdb");

    // Generate employment signals from BLS data
    context.employment_signals = generator.generateSignals();

    // Generate sector rotation recommendations
    context.rotation_signals = generator.generateRotationSignals();

    // Check for recession warning (jobless claims spike)
    context.jobless_claims_alert = generator.checkJoblessClaimsSpike();

    Logger::getInstance().info("Context populated with {} employment signals, {} rotation signals",
                               context.employment_signals.size(), context.rotation_signals.size());

    if (context.hasRecessionWarning()) {
        Logger::getInstance().warn("RECESSION WARNING: Jobless claims spike detected!");
    }

    return context;
}

/**
 * Example 2: Strategy Using Employment Signals - Sector Rotation
 */
class EmploymentDrivenRotationStrategy : public BaseStrategy {
  public:
    EmploymentDrivenRotationStrategy()
        : BaseStrategy("Employment-Driven Rotation",
                       "Rotates into sectors with strong employment trends") {}

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> override {

        std::vector<TradingSignal> signals;

        // Check for recession warning - reduce activity
        if (context.hasRecessionWarning()) {
            Logger::getInstance().warn(
                "{}: Recession warning active - generating defensive signals only", getName());
            // Could generate signals to close positions or hedge here
            return signals;
        }

        // Get aggregate employment health
        double employment_health = context.getAggregateEmploymentScore();
        Logger::getInstance().info("{}: Aggregate employment health score: {:.2f}", getName(),
                                   employment_health);

        // Process rotation signals
        for (auto const& rotation : context.rotation_signals) {
            // Only act on strong signals
            if (!rotation.isStrongSignal())
                continue;

            if (rotation.action == SectorRotationSignal::Action::Overweight) {
                // Generate BUY signal for this sector ETF
                TradingSignal signal;
                signal.symbol = rotation.sector_etf;
                signal.strategy_name = getName();
                signal.type = SignalType::Buy;
                signal.confidence = std::abs(rotation.composite_score);
                signal.expected_return = rotation.composite_score * 200.0; // $200 per point
                signal.max_risk = 100.0;
                signal.win_probability = rotation.composite_score;
                signal.timestamp = context.current_time;
                signal.rationale =
                    "Sector rotation OVERWEIGHT: " + rotation.sector_name +
                    " (employment score: " + std::to_string(rotation.employment_score) + ")";

                signals.push_back(signal);

                Logger::getInstance().info(
                    "{}: BUY {} ({}), confidence={:.2f}, employment_score={:.2f}", getName(),
                    rotation.sector_etf, rotation.sector_name, signal.confidence,
                    rotation.employment_score);
            } else if (rotation.action == SectorRotationSignal::Action::Underweight) {
                // Generate SELL signal for this sector ETF
                TradingSignal signal;
                signal.symbol = rotation.sector_etf;
                signal.strategy_name = getName();
                signal.type = SignalType::Sell;
                signal.confidence = std::abs(rotation.composite_score);
                signal.expected_return = std::abs(rotation.composite_score) * 150.0;
                signal.max_risk = 80.0;
                signal.win_probability = std::abs(rotation.composite_score);
                signal.timestamp = context.current_time;
                signal.rationale =
                    "Sector rotation UNDERWEIGHT: " + rotation.sector_name +
                    " (employment score: " + std::to_string(rotation.employment_score) + ")";

                signals.push_back(signal);

                Logger::getInstance().info(
                    "{}: SELL {} ({}), confidence={:.2f}, employment_score={:.2f}", getName(),
                    rotation.sector_etf, rotation.sector_name, signal.confidence,
                    rotation.employment_score);
            }
        }

        return signals;
    }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {
            {"min_composite_score", "0.70"},
            {"employment_weight", "0.60"},
        };
    }
};

/**
 * Example 3: Strategy Using Employment as Risk Filter
 */
class EmploymentFilteredVolatilityStrategy : public BaseStrategy {
  public:
    EmploymentFilteredVolatilityStrategy()
        : BaseStrategy("Employment-Filtered Volatility",
                       "Volatility strategy with employment-based risk filtering") {}

    [[nodiscard]] auto generateSignals(StrategyContext const& context)
        -> std::vector<TradingSignal> override {

        std::vector<TradingSignal> signals;

        // Get strongest employment signals
        auto top_employment_signals = context.getStrongestEmploymentSignals(5);

        Logger::getInstance().info("{}: Analyzing {} top employment signals", getName(),
                                   top_employment_signals.size());

        // Count bearish signals as risk indicator
        int bearish_count = 0;
        for (auto const& emp_signal : top_employment_signals) {
            if (emp_signal.bearish) {
                bearish_count++;
                Logger::getInstance().info("{}: Bearish employment signal - {} (strength: {:.2f})",
                                           getName(), emp_signal.sector_name,
                                           emp_signal.signal_strength);
            }
        }

        // Calculate risk adjustment factor based on employment
        double risk_adjustment = 1.0;
        if (bearish_count >= 3) {
            risk_adjustment = 0.5; // Reduce position sizes by 50%
            Logger::getInstance().warn("{}: High bearish employment signals - reducing risk by 50%",
                                       getName());
        } else if (bearish_count >= 2) {
            risk_adjustment = 0.75; // Reduce position sizes by 25%
        }

        // Example: Generate volatility signals (simplified)
        // In practice, would iterate through options_chains
        for (auto const& [symbol, quote] : context.current_quotes) {
            // Check sector-specific employment for this symbol
            // (In practice, would map symbol to sector)
            auto sector_name = "Information Technology"; // Example mapping
            auto sector_signals = context.getEmploymentSignalsForSector(sector_name);

            bool sector_is_bearish = false;
            for (auto const& sig : sector_signals) {
                if (sig.bearish && sig.signal_strength < -0.5) {
                    sector_is_bearish = true;
                    break;
                }
            }

            // Skip opportunities in bearish sectors
            if (sector_is_bearish) {
                Logger::getInstance().info("{}: Skipping {} - sector {} has bearish employment",
                                           getName(), symbol, sector_name);
                continue;
            }

            // Generate signal with risk adjustment
            TradingSignal signal;
            signal.symbol = symbol;
            signal.strategy_name = getName();
            signal.type = SignalType::Buy;
            signal.confidence = 0.75;
            signal.expected_return = 150.0 * risk_adjustment;
            signal.max_risk = 100.0 * risk_adjustment; // Adjusted for employment risk
            signal.win_probability = 0.65;
            signal.timestamp = context.current_time;
            signal.rationale = "Volatility opportunity with employment filter " +
                               "(risk_adj=" + std::to_string(risk_adjustment) + ")";

            signals.push_back(signal);
        }

        return signals;
    }

    [[nodiscard]] auto getParameters() const
        -> std::unordered_map<std::string, std::string> override {
        return {
            {"bearish_threshold", "3"},
            {"risk_reduction_factor", "0.5"},
        };
    }
};

/**
 * Example 4: Demonstrate Helper Methods
 */
auto demonstrateHelperMethods(StrategyContext const& context) -> void {
    Logger::getInstance().info("\n=== StrategyContext Employment Helper Methods Demo ===\n");

    // 1. Check for recession warning
    if (context.hasRecessionWarning()) {
        auto const& alert = context.jobless_claims_alert.value();
        Logger::getInstance().warn(
            "Recession Warning Active:\n  Confidence: {:.2f}\n  Rationale: {}", alert.confidence,
            alert.rationale);
    } else {
        Logger::getInstance().info("No recession warning active");
    }

    // 2. Get aggregate employment score
    double aggregate = context.getAggregateEmploymentScore();
    Logger::getInstance().info("\nAggregate Employment Score: {:.2f}", aggregate);

    if (aggregate > 0.5) {
        Logger::getInstance().info("  → Strong employment growth across sectors");
    } else if (aggregate < -0.5) {
        Logger::getInstance().warn("  → Weak employment across sectors");
    } else {
        Logger::getInstance().info("  → Mixed employment signals");
    }

    // 3. Get strongest employment signals
    auto strongest = context.getStrongestEmploymentSignals(3);
    Logger::getInstance().info("\nTop 3 Strongest Employment Signals:");
    for (size_t i = 0; i < strongest.size(); ++i) {
        auto const& sig = strongest[i];
        Logger::getInstance().info(
            "  {}. {} - Strength: {:.2f}, Confidence: {:.2f}, Change: {:.1f}%", i + 1,
            sig.sector_name, sig.signal_strength, sig.confidence, sig.employment_change);
    }

    // 4. Get sector-specific signals
    std::vector<std::string> sectors_to_check = {"Information Technology", "Financials",
                                                 "Health Care"};

    Logger::getInstance().info("\nSector-Specific Employment Analysis:");
    for (auto const& sector : sectors_to_check) {
        auto signals = context.getEmploymentSignalsForSector(sector);
        auto rotation = context.getRotationSignalForSector(sector);

        Logger::getInstance().info("\n  {}: ", sector);
        Logger::getInstance().info("    Employment Signals: {}", signals.size());

        if (rotation.has_value()) {
            std::string action_str;
            switch (rotation->action) {
                case SectorRotationSignal::Action::Overweight:
                    action_str = "OVERWEIGHT";
                    break;
                case SectorRotationSignal::Action::Underweight:
                    action_str = "UNDERWEIGHT";
                    break;
                default:
                    action_str = "NEUTRAL";
            }

            Logger::getInstance().info("    Rotation: {} (composite: {:.2f}, ETF: {})", action_str,
                                       rotation->composite_score, rotation->sector_etf);
        } else {
            Logger::getInstance().info("    Rotation: No signal available");
        }
    }

    Logger::getInstance().info("\n=== End Demo ===\n");
}

/**
 * Main Example Function
 */
auto main() -> int {
    Logger::getInstance().info("Employment Signals Integration Example\n");

    // 1. Populate context with employment data
    auto context = populateContextWithEmploymentData();

    // 2. Demonstrate helper methods
    demonstrateHelperMethods(context);

    // 3. Create and test employment-driven rotation strategy
    EmploymentDrivenRotationStrategy rotation_strategy;
    auto rotation_signals = rotation_strategy.generateSignals(context);

    Logger::getInstance().info("\nEmployment-Driven Rotation Strategy generated {} signals",
                               rotation_signals.size());

    for (auto const& signal : rotation_signals) {
        Logger::getInstance().info("  Signal: {} {} (confidence: {:.2f})\n    {}",
                                   (signal.type == SignalType::Buy ? "BUY" : "SELL"), signal.symbol,
                                   signal.confidence, signal.rationale);
    }

    // 4. Create and test employment-filtered volatility strategy
    EmploymentFilteredVolatilityStrategy filtered_strategy;

    // Add some mock quotes for demonstration
    context.current_quotes["AAPL"] = Quote{}; // Simplified
    context.current_quotes["MSFT"] = Quote{};
    context.current_quotes["JPM"] = Quote{};

    auto filtered_signals = filtered_strategy.generateSignals(context);

    Logger::getInstance().info("\nEmployment-Filtered Volatility Strategy generated {} signals",
                               filtered_signals.size());

    for (auto const& signal : filtered_signals) {
        Logger::getInstance().info("  Signal: {} (max_risk: ${:.2f}, expected_return: ${:.2f})",
                                   signal.symbol, signal.max_risk, signal.expected_return);
    }

    Logger::getInstance().info("\nExample completed successfully!");
    return 0;
}
