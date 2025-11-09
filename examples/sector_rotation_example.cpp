/**
 * BigBrotherAnalytics - Sector Rotation Strategy Example
 *
 * Demonstrates usage of the SectorRotationStrategy with:
 * - Custom configuration
 * - Signal generation
 * - Risk management integration
 * - Position sizing
 * - Rebalancing logic
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>

import bigbrother.strategies;
import bigbrother.strategy;
import bigbrother.risk_management;
import bigbrother.utils.logger;

using namespace bigbrother::strategies;
using namespace bigbrother::strategy;
using namespace bigbrother::risk;
using namespace bigbrother::utils;

// Helper function to print signals
void printSignals(std::vector<TradingSignal> const& signals) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Generated Trading Signals\n";
    std::cout << std::string(80, '=') << "\n\n";

    if (signals.empty()) {
        std::cout << "No signals generated.\n";
        return;
    }

    std::cout << std::left
              << std::setw(8) << "Symbol"
              << std::setw(12) << "Action"
              << std::setw(12) << "Confidence"
              << std::setw(15) << "Position Size"
              << std::setw(12) << "Max Risk"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto const& signal : signals) {
        std::string action = (signal.type == SignalType::Buy) ? "OVERWEIGHT" : "UNDERWEIGHT";
        double position_size = (signal.type == SignalType::Buy)
            ? signal.expected_return / 0.15  // Reverse calculate from expected return
            : 0.0;

        std::cout << std::left
                  << std::setw(8) << signal.symbol
                  << std::setw(12) << action
                  << std::setw(12) << std::fixed << std::setprecision(2) << signal.confidence
                  << std::setw(15) << "$" + std::to_string(static_cast<int>(position_size))
                  << std::setw(12) << "$" + std::to_string(static_cast<int>(signal.max_risk))
                  << "\n";

        std::cout << "  Rationale: " << signal.rationale << "\n\n";
    }
}

// Example 1: Basic usage with default configuration
void example1_basic_usage() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Example 1: Basic Sector Rotation Strategy\n";
    std::cout << std::string(80, '=') << "\n";

    // Create strategy with default configuration
    auto strategy = createSectorRotationStrategy();

    // Set up strategy context
    StrategyContext context{
        .current_quotes = {},
        .options_chains = {},
        .current_positions = {},
        .account_value = 30000.0,
        .available_capital = 10000.0,
        .current_time = static_cast<Timestamp>(std::time(nullptr))
    };

    // Generate signals
    auto signals = strategy->generateSignals(context);

    // Print results
    printSignals(signals);
}

// Example 2: Custom configuration for aggressive rotation
void example2_custom_config() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Example 2: Aggressive Rotation Configuration\n";
    std::cout << std::string(80, '=') << "\n";

    // Configure for aggressive rotation
    SectorRotationStrategy::Config config{
        .min_composite_score = 0.50,      // Lower threshold
        .rotation_threshold = 0.60,       // More sensitive rotation
        .employment_weight = 0.70,        // Higher employment weight
        .sentiment_weight = 0.20,
        .momentum_weight = 0.10,
        .top_n_overweight = 4,            // Overweight 4 sectors
        .bottom_n_underweight = 3,        // Underweight 3 sectors
        .max_sector_allocation = 0.30,    // Allow higher concentration
        .min_sector_allocation = 0.03,
        .rebalance_frequency_days = 14,   // Rebalance bi-weekly
        .db_path = "data/bigbrother.duckdb",
        .scripts_path = "scripts"
    };

    auto strategy = createSectorRotationStrategy(std::move(config));

    StrategyContext context{
        .account_value = 50000.0,
        .available_capital = 20000.0,
        .current_time = static_cast<Timestamp>(std::time(nullptr))
    };

    auto signals = strategy->generateSignals(context);
    printSignals(signals);

    std::cout << "\nConfiguration Details:\n";
    std::cout << "  Top N Overweight: 4 sectors\n";
    std::cout << "  Bottom N Underweight: 3 sectors\n";
    std::cout << "  Employment Weight: 70%\n";
    std::cout << "  Max Sector Allocation: 30%\n";
}

// Example 3: Integration with RiskManager
void example3_risk_integration() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Example 3: Risk Management Integration\n";
    std::cout << std::string(80, '=') << "\n";

    // Create strategy
    auto strategy = createSectorRotationStrategy();

    // Create risk manager
    RiskManager risk_manager{RiskLimits::forThirtyKAccount()};

    StrategyContext context{
        .account_value = 30000.0,
        .available_capital = 10000.0,
        .current_time = static_cast<Timestamp>(std::time(nullptr))
    };

    // Generate signals
    auto signals = strategy->generateSignals(context);

    std::cout << "\nRisk Assessment:\n";
    std::cout << std::string(80, '-') << "\n";

    // Validate each signal against risk limits
    for (auto const& signal : signals) {
        if (signal.type == SignalType::Buy) {
            double position_size = signal.expected_return / 0.15;

            auto trade_risk = risk_manager.assessTrade(
                signal.symbol,
                position_size,
                100.0,  // entry_price (placeholder)
                90.0,   // stop_price (10% stop loss)
                115.0,  // target_price (15% target)
                signal.win_probability
            );

            if (trade_risk && trade_risk->approved) {
                std::cout << "✓ " << signal.symbol << " - APPROVED\n";
                std::cout << "  Position Size: $" << std::fixed << std::setprecision(2)
                          << position_size << "\n";
                std::cout << "  Max Risk: $" << trade_risk->max_loss << "\n";
                std::cout << "  Expected Return: $" << trade_risk->expected_return << "\n";
                std::cout << "  Risk/Reward: " << trade_risk->risk_reward_ratio << "\n";
            } else if (trade_risk) {
                std::cout << "✗ " << signal.symbol << " - REJECTED\n";
                std::cout << "  Reason: " << trade_risk->rejection_reason << "\n";
            }
        }
    }
}

// Example 4: Strategy performance tracking
void example4_performance_tracking() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Example 4: Strategy Performance Tracking\n";
    std::cout << std::string(80, '=') << "\n";

    // Create strategy manager
    StrategyManager manager;

    // Add sector rotation strategy
    manager.addStrategy(createSectorRotationStrategy());

    StrategyContext context{
        .account_value = 30000.0,
        .available_capital = 10000.0,
        .current_time = static_cast<Timestamp>(std::time(nullptr))
    };

    // Generate signals
    auto signals = manager.generateSignals(context);

    std::cout << "\nStrategy Manager Results:\n";
    std::cout << "  Total Signals Generated: " << signals.size() << "\n\n";

    // Get performance metrics
    auto performance = manager.getPerformance("Sector Rotation (Multi-Signal)");

    if (performance) {
        std::cout << "Performance Metrics:\n";
        std::cout << "  Strategy: " << performance->name << "\n";
        std::cout << "  Signals Generated: " << performance->signals_generated << "\n";
        std::cout << "  Trades Executed: " << performance->trades_executed << "\n";
        std::cout << "  Total P&L: $" << std::fixed << std::setprecision(2)
                  << performance->total_pnl << "\n";
        std::cout << "  Win Rate: " << (performance->win_rate * 100.0) << "%\n";
        std::cout << "  Sharpe Ratio: " << performance->sharpe_ratio << "\n";
        std::cout << "  Active: " << (performance->active ? "Yes" : "No") << "\n";
    }
}

// Example 5: Multi-strategy comparison
void example5_multi_strategy_comparison() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Example 5: Multi-Strategy Comparison\n";
    std::cout << std::string(80, '=') << "\n";

    StrategyManager manager;

    // Add multiple strategies
    manager.addStrategy(createSectorRotationStrategy());
    manager.addStrategy(createStraddleStrategy());
    manager.addStrategy(createVolatilityArbStrategy());

    StrategyContext context{
        .account_value = 30000.0,
        .available_capital = 10000.0,
        .current_time = static_cast<Timestamp>(std::time(nullptr))
    };

    auto signals = manager.generateSignals(context);

    std::cout << "\nCombined Strategy Signals:\n";
    std::cout << "  Total Signals: " << signals.size() << "\n\n";

    // Group by strategy
    std::cout << "Signals by Strategy:\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto const& signal : signals) {
        std::cout << "  [" << signal.strategy_name << "] "
                  << signal.symbol
                  << " - Confidence: " << std::fixed << std::setprecision(2)
                  << signal.confidence << "\n";
    }

    // Print all performance metrics
    auto all_performance = manager.getAllPerformance();

    std::cout << "\n\nStrategy Performance Summary:\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto const& perf : all_performance) {
        std::cout << perf.name << ":\n";
        std::cout << "  Signals: " << perf.signals_generated
                  << " | Active: " << (perf.active ? "Yes" : "No") << "\n";
    }
}

// Main function
int main() {
    try {
        // Initialize logger
        Logger::getInstance().setLevel(Logger::Level::INFO);

        std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   BigBrotherAnalytics - Sector Rotation Strategy Examples         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";

        // Run examples
        example1_basic_usage();
        example2_custom_config();
        example3_risk_integration();
        example4_performance_tracking();
        example5_multi_strategy_comparison();

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "All examples completed successfully!\n";
        std::cout << std::string(80, '=') << "\n\n";

        return 0;

    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
