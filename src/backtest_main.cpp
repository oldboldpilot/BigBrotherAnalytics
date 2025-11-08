/**
 * BigBrotherAnalytics - Backtesting Application
 *
 * Standalone application for backtesting trading strategies.
 *
 * Usage:
 *   ./backtest --strategy straddle --start 2020-01-01 --end 2024-01-01
 *   ./backtest --all-strategies --start 2020-01-01 --end 2024-01-01
 *   ./backtest --config configs/backtest.yaml
 */

#include "utils/logger.hpp"
#include "utils/config.hpp"
#include "utils/timer.hpp"
#include "backtesting/backtest_engine.hpp"
#include "trading_decision/strategy_straddle.hpp"
#include "trading_decision/strategy_volatility_arb.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace bigbrother;

/**
 * Print backtest results
 */
auto printResults(backtest::BacktestMetrics const& metrics) -> void {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              BACKTEST RESULTS                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "Returns:\n";
    std::cout << "  Total Return:      $" << metrics.total_return << " ("
              << metrics.total_return_percent << "%)\n";
    std::cout << "  Annualized Return: " << metrics.annualized_return << "%\n";
    std::cout << "  CAGR:              " << metrics.cagr << "%\n";
    std::cout << "\n";

    std::cout << "Risk-Adjusted:\n";
    std::cout << "  Sharpe Ratio:      " << metrics.sharpe_ratio << "\n";
    std::cout << "  Sortino Ratio:     " << metrics.sortino_ratio << "\n";
    std::cout << "  Max Drawdown:      " << metrics.max_drawdown_percent * 100.0 << "%\n";
    std::cout << "\n";

    std::cout << "Trade Statistics:\n";
    std::cout << "  Total Trades:      " << metrics.total_trades << "\n";
    std::cout << "  Winning Trades:    " << metrics.winning_trades << "\n";
    std::cout << "  Losing Trades:     " << metrics.losing_trades << "\n";
    std::cout << "  Win Rate:          " << metrics.win_rate * 100.0 << "%\n";
    std::cout << "\n";

    std::cout << "P&L Analysis:\n";
    std::cout << "  Average Win:       $" << metrics.avg_win << "\n";
    std::cout << "  Average Loss:      $" << metrics.avg_loss << "\n";
    std::cout << "  Profit Factor:     " << metrics.profit_factor << "\n";
    std::cout << "  Expectancy:        $" << metrics.expectancy << " per trade\n";
    std::cout << "\n";

    // Success criteria check
    std::cout << "Success Criteria (per PRD):\n";
    std::cout << "  Win Rate > 60%:    " << (metrics.win_rate >= 0.60 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  Sharpe > 2.0:      " << (metrics.sharpe_ratio >= 2.0 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "  Max DD < 15%:      " << (metrics.max_drawdown_percent <= 0.15 ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "\n";

    if (metrics.passesThresholds()) {
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✓ STRATEGY PASSES ALL SUCCESS CRITERIA                   ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    } else {
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✗ STRATEGY DOES NOT MEET SUCCESS CRITERIA                ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    }

    std::cout << "\n";
}

auto main(int argc, char* argv[]) -> int {
    // Initialize logger
    utils::Logger::getInstance().initialize("logs/backtest.log", utils::LogLevel::INFO, true);

    LOG_INFO("╔════════════════════════════════════════════════════════════╗");
    LOG_INFO("║         BigBrotherAnalytics Backtesting Engine            ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // Parse command line arguments
    std::string start_date = "2020-01-01";
    std::string end_date = "2024-01-01";
    std::string strategy_name = "straddle";
    std::string data_path = "data/historical/";
    bool all_strategies = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--start" && i + 1 < argc) {
            start_date = argv[++i];
        } else if (arg == "--end" && i + 1 < argc) {
            end_date = argv[++i];
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy_name = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--all-strategies") {
            all_strategies = true;
        } else if (arg == "--help") {
            std::cout << R"(
BigBrotherAnalytics Backtesting Engine

Usage:
  backtest [OPTIONS]

Options:
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  --strategy NAME       Strategy name (straddle, strangle, vol_arb, mean_reversion)
  --all-strategies      Test all strategies
  --data PATH           Path to historical data
  --help                Show this help

Examples:
  # Backtest straddle strategy
  ./backtest --strategy straddle --start 2020-01-01 --end 2024-01-01

  # Backtest all strategies
  ./backtest --all-strategies --start 2020-01-01 --end 2024-01-01

  # Custom data path
  ./backtest --strategy vol_arb --data /mnt/data/historical/
)" << std::endl;
            return 0;
        }
    }

    LOG_INFO("Backtest Configuration:");
    LOG_INFO("  Start Date: {}", start_date);
    LOG_INFO("  End Date:   {}", end_date);
    LOG_INFO("  Strategy:   {}", all_strategies ? "ALL" : strategy_name);
    LOG_INFO("  Data Path:  {}", data_path);
    LOG_INFO("");

    // Symbols to backtest
    std::vector<std::string> symbols = {
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AMD", "TSLA", "META"
    };

    // Run backtest
    try {
        backtest::BacktestRunner runner;
        runner.from(start_date)
              .to(end_date)
              .withCapital(30'000.0)
              .commission(0.0)
              .slippage(2.0)
              .forSymbols(symbols)
              .loadData(data_path);

        // Add strategies
        if (all_strategies || strategy_name == "straddle") {
            runner.addStrategy<strategy::DeltaNeutralStraddleStrategy>();
        }

        if (all_strategies || strategy_name == "strangle") {
            runner.addStrategy<strategy::DeltaNeutralStrangleStrategy>();
        }

        if (all_strategies || strategy_name == "vol_arb") {
            runner.addStrategy<strategy::VolatilityArbitrageStrategy>();
        }

        if (all_strategies || strategy_name == "mean_reversion") {
            runner.addStrategy<strategy::MeanReversionStrategy>();
        }

        LOG_INFO("Running backtest...");

        utils::Timer timer;
        auto result = runner.run();

        auto const elapsed = timer.elapsedSeconds();

        LOG_INFO("Backtest completed in {:.2f} seconds", elapsed);

        // Print results
        printResults(result);

        // Export results
        runner.exportTrades("data/backtest_results/trades.csv");
        runner.exportMetrics("data/backtest_results/metrics.csv");

        LOG_INFO("Results exported to data/backtest_results/");

        // Return success/failure based on thresholds
        return result.passesThresholds() ? 0 : 1;

    } catch (std::exception const& e) {
        LOG_ERROR("Backtest error: {}", e.what());
        return 1;
    }
}
