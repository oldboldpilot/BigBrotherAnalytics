/**
 * BigBrotherAnalytics - Main Trading Application
 *
 * AI-powered algorithmic options day trading platform.
 *
 * Architecture:
 * - C++23 high-performance core (95% of codebase)
 * - Microsecond-level latency for critical paths
 * - MPI + OpenMP parallelization
 * - Real-time data streaming
 * - Comprehensive risk management
 *
 * Features:
 * - Options pricing (< 100μs per option)
 * - Correlation analysis with time-lag detection
 * - Multiple trading strategies (straddle, strangle, vol arb)
 * - Kelly Criterion position sizing
 * - Monte Carlo pre-trade validation
 * - Schwab API integration (OAuth 2.0, WebSocket)
 * - DuckDB for data storage and analytics
 * - Full explainability for all trades
 *
 * Usage:
 *   ./bigbrother --config configs/config.yaml
 *   ./bigbrother --config configs/config.yaml --paper-trading
 *   ./bigbrother --backtest --start 2020-01-01 --end 2024-01-01
 */

import bigbrother.utils.logger;
import bigbrother.utils.config;
import bigbrother.utils.database;
import bigbrother.utils.timer;
import bigbrother.options.pricing;
import bigbrother.correlation;
import bigbrother.risk_management;
import bigbrother.schwab_api;
import bigbrother.strategy;
import bigbrother.strategies;

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace bigbrother;

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

/**
 * Signal handler for graceful shutdown
 */
auto signalHandler(int signal) -> void {
    if (signal == SIGINT || signal == SIGTERM) {
        utils::Logger::getInstance().info(
            "Shutdown signal received, closing positions and exiting...");
        g_running.store(false);
    }
}

/**
 * Trading Engine
 *
 * Main orchestration class that coordinates all systems
 */
class TradingEngine {
  public:
    TradingEngine()
        : risk_limits_{risk::RiskLimits::forThirtyKAccount()}, risk_manager_{risk_limits_},
          paper_trading_{false} {}

    [[nodiscard]] auto initialize(std::string const& config_file) -> bool {
        utils::Logger::getInstance().info(
            "╔══════════════════════════════════════════════════════════╗");
        utils::Logger::getInstance().info(
            "║        BigBrotherAnalytics Trading Engine v1.0          ║");
        utils::Logger::getInstance().info(
            "╚══════════════════════════════════════════════════════════╝");
        utils::Logger::getInstance().info("");

        // Load configuration
        utils::Logger::getInstance().info("Loading configuration from: {}", config_file);

        if (!config_.load(config_file)) {
            utils::Logger::getInstance().error("Failed to load configuration");
            return false;
        }

        // Initialize logger
        auto const log_file = config_.get<std::string>("logging.file", "logs/bigbrother.log");
        auto const log_level_str = config_.get<std::string>("logging.level", "info");

        utils::LogLevel log_level = utils::LogLevel::INFO;
        if (log_level_str == "trace")
            log_level = utils::LogLevel::TRACE;
        else if (log_level_str == "debug")
            log_level = utils::LogLevel::DEBUG;
        else if (log_level_str == "warn")
            log_level = utils::LogLevel::WARN;
        else if (log_level_str == "error")
            log_level = utils::LogLevel::ERROR;

        logger_.initialize(log_file, log_level, true);

        utils::Logger::getInstance().info("Logger initialized: {} (level: {})", log_file,
                                          log_level_str);

        // Initialize database
        auto const db_path = config_.get<std::string>("database.path", "data/bigbrother.duckdb");

        database_ = std::make_unique<utils::Database>(db_path);
        utils::Logger::getInstance().info("Database initialized: {}", db_path);

        // Get trading mode
        paper_trading_ = config_.get<bool>("trading.paper_trading", true);

        if (paper_trading_) {
            utils::Logger::getInstance().warn(
                "═══════════════════════════════════════════════════════");
            utils::Logger::getInstance().warn(
                "    PAPER TRADING MODE - NO REAL MONEY AT RISK       ");
            utils::Logger::getInstance().warn(
                "═══════════════════════════════════════════════════════");
        } else {
            utils::Logger::getInstance().critical(
                "═══════════════════════════════════════════════════════");
            utils::Logger::getInstance().critical(
                "    LIVE TRADING MODE - REAL MONEY AT RISK           ");
            utils::Logger::getInstance().critical(
                "    Account Value: $30,000                           ");
            utils::Logger::getInstance().critical(
                "    Max Daily Loss: $900 (3%)                        ");
            utils::Logger::getInstance().critical(
                "═══════════════════════════════════════════════════════");
        }

        // Initialize Schwab client
        utils::Logger::getInstance().info("Initializing Schwab API client...");

        auto const client_id = config_.get<std::string>("schwab.client_id", "");
        auto const client_secret = config_.get<std::string>("schwab.client_secret", "");
        auto const redirect_uri =
            config_.get<std::string>("schwab.redirect_uri", "https://localhost:8080/callback");

        if (client_id.empty() || client_secret.empty()) {
            utils::Logger::getInstance().error("Schwab API credentials not found in config");
            return false;
        }

        schwab::OAuth2Config oauth_config;
        oauth_config.client_id = client_id;
        oauth_config.client_secret = client_secret;
        oauth_config.redirect_uri = redirect_uri;

        schwab_client_ = std::make_unique<schwab::SchwabClient>(oauth_config);

        // Initialize strategy manager
        utils::Logger::getInstance().info("Initializing trading strategies...");

        strategy_manager_ = std::make_unique<strategy::StrategyManager>();

        // Add default strategies
        strategy_manager_->addStrategy(strategies::createStraddleStrategy());
        strategy_manager_->addStrategy(strategies::createStrangleStrategy());
        strategy_manager_->addStrategy(strategies::createVolatilityArbStrategy());

        utils::Logger::getInstance().info("Strategies registered:");
        for (auto const* strat : strategy_manager_->getStrategies()) {
            utils::Logger::getInstance().info("  - {}: {}", strat->getName(), "Strategy");
        }

        utils::Logger::getInstance().info("");
        utils::Logger::getInstance().info("Initialization complete!");
        utils::Logger::getInstance().info("");

        return true;
    }

    [[nodiscard]] auto run() -> int {
        utils::Logger::getInstance().info("Starting trading engine...");

        utils::Timer engine_timer;

        while (g_running.load()) {
            try {
                // Main trading loop
                runTradingCycle();

                // Sleep for configured interval
                auto const cycle_interval_ms = config_.get<int>("trading.cycle_interval_ms",
                                                                60000 // Default: 1 minute
                );

                std::this_thread::sleep_for(std::chrono::milliseconds(cycle_interval_ms));

            } catch (std::exception const& e) {
                utils::Logger::getInstance().error("Error in trading cycle: {}", e.what());

                // Continue running after error
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }

        utils::Logger::getInstance().info("Trading engine stopped after {:.2f} seconds",
                                          engine_timer.elapsedSeconds());

        return 0;
    }

    auto shutdown() -> void {
        utils::Logger::getInstance().info("Shutting down trading engine...");

        // Close all positions if configured
        auto const close_on_exit = config_.get<bool>("trading.close_positions_on_exit", true);

        if (close_on_exit && !paper_trading_) {
            utils::Logger::getInstance().warn("Closing all positions before shutdown...");
            // Emergency stop - close all positions (stub)
            utils::Logger::getInstance().critical("Emergency stop initiated");
        }

        // Flush logs
        logger_.flush();

        // Close database
        if (database_) {
            // Database cleanup (auto-handled by RAII)
        }

        // Print final statistics
        utils::Profiler::getInstance().printStats();

        utils::Logger::getInstance().info("Shutdown complete");
    }

  private:
    auto runTradingCycle() -> void {
        // PROFILE_SCOPE("TradingEngine::runTradingCycle");  // Profiling disabled for now

        utils::Logger::getInstance().debug("═══ Trading Cycle Start ═══");

        // 1. Build strategy context
        auto context = buildContext();

        // 2. Generate signals from all strategies
        auto signals = strategy_manager_->generateSignals(context);

        utils::Logger::getInstance().info("Generated {} trading signals", signals.size());

        // 3. Execute approved signals (stub - full implementation with Schwab API later)
        if (!signals.empty()) {
            // Filter signals by confidence
            std::vector<strategy::TradingSignal> filtered_signals;
            for (auto const& signal : signals) {
                if (signal.confidence >= 0.60) {
                    filtered_signals.push_back(signal);
                }
            }

            auto execution_result = strategy::StrategyExecutor(*strategy_manager_)
                                        .withContext(context)
                                        .withRiskManager(risk_manager_)
                                        .withSchwabClient(*schwab_client_)
                                        .minConfidence(0.60)
                                        .maxSignals(10)
                                        .execute();

            if (execution_result) {
                utils::Logger::getInstance().info("Executed {} trades", execution_result->size());

                for (auto const& order_id : *execution_result) {
                    utils::Logger::getInstance().info("  Order placed: {}", order_id);
                }
            } else {
                utils::Logger::getInstance().error("Trade execution failed: {}",
                                                   execution_result.error().message);
            }
        }

        // 4. Update positions and risk
        updatePositions();

        // 5. Check stop losses
        checkStopLosses();

        // 6. Check daily loss limit
        auto portfolio_risk = risk_manager_.getPortfolioRisk();
        if (portfolio_risk && portfolio_risk->daily_loss_remaining <= 0.0) {
            utils::Logger::getInstance().critical(
                "═══════════════════════════════════════════════════════");
            utils::Logger::getInstance().critical(
                "   DAILY LOSS LIMIT REACHED ($900)                    ");
            utils::Logger::getInstance().critical(
                "   TRADING HALTED FOR TODAY                           ");
            utils::Logger::getInstance().critical(
                "═══════════════════════════════════════════════════════");

            // Stop trading for today
            g_running.store(false);
        }

        utils::Logger::getInstance().debug("═══ Trading Cycle End ═══");
    }

    [[nodiscard]] auto buildContext() -> strategy::StrategyContext {
        strategy::StrategyContext context;

        context.current_time = utils::Timer::now();
        context.account_value = 30'000.0;     // TODO: Get from Schwab API
        context.available_capital = 20'000.0; // TODO: Get from Schwab API

        // Get current positions
        context.current_positions = {}; // TODO: Get from Schwab API

        // TODO: Load market data, options chains, correlation data
        // TODO: Get current quotes from Schwab API
        // TODO: Get options chains from Schwab API

        return context;
    }

    auto updatePositions() -> void {
        // TODO: Update positions from Schwab API
        // TODO: Calculate real-time P&L
        // TODO: Update risk metrics
    }

    auto checkStopLosses() -> void {
        // TODO: Check stop losses for all positions
        // TODO: Execute stop orders if triggered
    }

    utils::Logger& logger_{utils::Logger::getInstance()};
    utils::Config& config_{utils::Config::getInstance()};
    std::unique_ptr<utils::Database> database_;

    risk::RiskLimits risk_limits_;
    risk::RiskManager risk_manager_;

    std::unique_ptr<schwab::SchwabClient> schwab_client_;
    std::unique_ptr<strategy::StrategyManager> strategy_manager_;

    bool paper_trading_;
};

/**
 * Main Entry Point
 */
auto main(int argc, char* argv[]) -> int {
    // Register signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Parse command line arguments
    std::string config_file = "configs/config.yaml";
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--config") {
            if (i + 1 < argc) {
                config_file = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "BigBrotherAnalytics v1.0.0" << std::endl;
            return 0;
        }
    }

    if (show_help) {
        std::cout << R"(
BigBrotherAnalytics - AI-Powered Algorithmic Trading Platform

Usage:
  bigbrother [OPTIONS]

Options:
  --config FILE     Configuration file (default: configs/config.yaml)
  --help, -h        Show this help message
  --version, -v     Show version information

Examples:
  # Run with default config
  ./bigbrother

  # Run with custom config
  ./bigbrother --config configs/production.yaml

  # Paper trading mode (set in config file)
  ./bigbrother --config configs/paper_trading.yaml

For more information, see docs/README.md
)" << std::endl;
        return 0;
    }

    // Create and initialize trading engine
    TradingEngine engine;

    if (!engine.initialize(config_file)) {
        std::cerr << "Failed to initialize trading engine" << std::endl;
        return 1;
    }

    // Run trading engine
    int const exit_code = engine.run();

    // Shutdown
    engine.shutdown();

    return exit_code;
}
