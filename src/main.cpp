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
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 */

// Standard library includes MUST come before module imports to avoid libc++ header conflicts
// NOTE: Minimal header set to avoid transitive chrono includes that conflict with C++23 modules
#include <atomic>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <ctime>  // For time_t, needed for token parsing

// Core string types - MUST be included before modules for ABI compatibility
#include <string>
#include <string_view>

// Module imports
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
import bigbrother.employment.signals;

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

        // Note: Config::load() is a stub - config values are hardcoded in config.cppm
        // This avoids C++23 module/YAML-cpp linker issues
        // TODO: Implement proper YAML loading after module system stabilizes

        // Load risk limits from config
        risk_limits_.account_value = config_.get<double>("risk.account_value", 30000.0);
        risk_limits_.max_daily_loss = config_.get<double>("risk.max_daily_loss", 900.0);
        risk_limits_.max_position_size = config_.get<double>("risk.max_position_size", 1500.0);
        risk_limits_.max_concurrent_positions =
            config_.get<int>("risk.max_concurrent_positions", 10);
        risk_limits_.max_portfolio_heat = config_.get<double>("risk.max_portfolio_heat", 0.15);
        risk_limits_.max_correlation_exposure =
            config_.get<double>("risk.max_correlation_exposure", 0.30);
        risk_limits_.require_stop_loss = config_.get<bool>("risk.require_stop_loss", true);

        // Update risk manager with configured limits using fluent API
        risk_manager_.withLimits(risk_limits_);

        utils::Logger::getInstance().info("Risk Limits configured:");
        utils::Logger::getInstance().info("  Account Value: ${:.2f}", risk_limits_.account_value);
        utils::Logger::getInstance().info(
            "  Max Daily Loss: ${:.2f} ({:.1f}%)", risk_limits_.max_daily_loss,
            (risk_limits_.max_daily_loss / risk_limits_.account_value) * 100.0);
        utils::Logger::getInstance().info(
            "  Max Position Size: ${:.2f} ({:.1f}%)", risk_limits_.max_position_size,
            (risk_limits_.max_position_size / risk_limits_.account_value) * 100.0);
        utils::Logger::getInstance().info("  Max Concurrent Positions: {}",
                                          risk_limits_.max_concurrent_positions);
        utils::Logger::getInstance().info("  Max Portfolio Heat: {:.1f}%",
                                          risk_limits_.max_portfolio_heat * 100.0);
        utils::Logger::getInstance().info("  Require Stop Loss: {}",
                                          risk_limits_.require_stop_loss ? "Yes" : "No");

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
        if (auto connect_result = database_->connect(); !connect_result) {
            utils::Logger::getInstance().error("Failed to connect to database {}: {}", db_path,
                                               connect_result.error().message);
            database_.reset();
        } else {
            utils::Logger::getInstance().info("Database connected: {}", db_path);
        }

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

        // Load OAuth token from file
        auto const token_file =
            config_.get<std::string>("schwab.token_file", "configs/schwab_tokens.json");
        utils::Logger::getInstance().info("Loading OAuth token from: {}", token_file);

        // Simple JSON parsing without nlohmann (to avoid module conflicts)
        auto extract_json_string_value = [](std::string const& json,
                                            std::string const& key) -> std::string {
            auto key_pos = json.find("\"" + key + "\"");
            if (key_pos == std::string::npos)
                return "";

            auto colon_pos = json.find(":", key_pos);
            if (colon_pos == std::string::npos)
                return "";

            auto quote1_pos = json.find("\"", colon_pos);
            if (quote1_pos == std::string::npos)
                return "";

            auto quote2_pos = json.find("\"", quote1_pos + 1);
            if (quote2_pos == std::string::npos)
                return "";

            return json.substr(quote1_pos + 1, quote2_pos - quote1_pos - 1);
        };

        try {
            std::ifstream token_stream(token_file);
            if (!token_stream) {
                utils::Logger::getInstance().error("Failed to open token file: {}", token_file);
                utils::Logger::getInstance().warn(
                    "Continuing without OAuth token - API calls will fail with 401");
            } else {
                std::string token_json((std::istreambuf_iterator<char>(token_stream)),
                                       std::istreambuf_iterator<char>());

                auto access_token = extract_json_string_value(token_json, "access_token");
                if (!access_token.empty()) {
                    oauth_config.access_token = access_token;
                    utils::Logger::getInstance().info("Loaded access_token from file");
                }

                auto refresh_token = extract_json_string_value(token_json, "refresh_token");
                if (!refresh_token.empty()) {
                    oauth_config.refresh_token = refresh_token;
                    utils::Logger::getInstance().info("Loaded refresh_token from file");
                }

                auto expires_at_str = extract_json_string_value(token_json, "expires_at");
                if (!expires_at_str.empty()) {
                    try {
                        // Parse as Unix timestamp (integer)
                        auto expires_timestamp = std::stoll(expires_at_str);
                        oauth_config.token_expiry =
                            std::chrono::system_clock::from_time_t(expires_timestamp);

                        // Calculate time remaining
                        auto now = std::chrono::system_clock::now();
                        auto time_remaining = std::chrono::duration_cast<std::chrono::minutes>(
                            oauth_config.token_expiry - now);

                        utils::Logger::getInstance().info(
                            "Token expires in {} minutes (timestamp: {})", time_remaining.count(),
                            expires_timestamp);
                    } catch (std::exception const& e) {
                        utils::Logger::getInstance().warn("Failed to parse expires_at: {}",
                                                          e.what());
                    }
                }
            }
        } catch (std::exception const& e) {
            utils::Logger::getInstance().error("Failed to parse token file: {}", e.what());
            utils::Logger::getInstance().warn(
                "Continuing without OAuth token - API calls will fail with 401");
        }

        schwab_client_ = std::make_unique<schwab::SchwabClient>(oauth_config);

        // Initialize strategy manager
        utils::Logger::getInstance().info("Initializing trading strategies...");

        strategy_manager_ = std::make_unique<strategy::StrategyManager>();

        // Add default strategies
        // TEMPORARY: ML strategy disabled due to quote lookup crash (corrupted symbols from JSON parsing)
        // strategy_manager_->addStrategy(
        //     strategies::createMLPredictorStrategy()); // AI-powered predictions
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

        // 5. Calculate and update daily return for VaR/Sharpe calculations
        auto const daily_return = calculateDailyReturn();
        risk_manager_.updateReturnHistory(daily_return);

        // 6. Get portfolio risk metrics (includes real-time VaR and Sharpe)
        auto portfolio_risk = risk_manager_.getPortfolioRisk();
        if (portfolio_risk) {
            utils::Logger::getInstance().info(
                "Risk Metrics - VaR(95%): {:.2f}%, Sharpe: {:.2f}, Daily P&L: ${:.2f}",
                portfolio_risk->var_95 * 100.0, portfolio_risk->sharpe_ratio,
                portfolio_risk->daily_pnl);

            // Check VaR breach (halt trading if risk too high)
            if (risk_manager_.isVaRBreached(-0.03)) { // -3% daily VaR threshold
                utils::Logger::getInstance().critical(
                    "═══════════════════════════════════════════════════════");
                utils::Logger::getInstance().critical(
                    "   VaR BREACH DETECTED - RISK TOO HIGH                ");
                utils::Logger::getInstance().critical(
                    "   Current VaR: {:.2f}% < -3.0% threshold            ",
                    portfolio_risk->var_95 * 100.0);
                utils::Logger::getInstance().critical(
                    "   TRADING HALTED FOR RISK MANAGEMENT                 ");
                utils::Logger::getInstance().critical(
                    "═══════════════════════════════════════════════════════");

                // Halt trading until risk normalizes
                g_running.store(false);
            }

            // Check Sharpe ratio (warning only, don't halt)
            if (!risk_manager_.isSharpeAcceptable(1.0)) {
                utils::Logger::getInstance().warn(
                    "⚠️  Sharpe ratio below target: {:.2f} < 1.0 (poor risk-adjusted returns)",
                    portfolio_risk->sharpe_ratio);
            }

            // Check daily loss limit
            if (portfolio_risk->daily_loss_remaining <= 0.0) {
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
        }

        // 7. Check stop losses
        checkStopLosses();

        utils::Logger::getInstance().debug("═══ Trading Cycle End ═══");
    }

    [[nodiscard]] auto buildContext() -> strategy::StrategyContext {
        strategy::StrategyContext context;

        context.current_time = utils::Timer::now();

        // Get account information from Schwab API
        auto balance_result = schwab_client_->account().getBalance();
        if (balance_result) {
            auto const& balance = *balance_result;
            context.account_value = balance.total_value;
            context.available_capital = balance.buying_power;

            utils::Logger::getInstance().debug(
                "Account balance: ${:.2f} total, ${:.2f} buying power", balance.total_value,
                balance.buying_power);
        } else {
            utils::Logger::getInstance().error("Failed to get account balance: {}",
                                               balance_result.error().message);
            // Use safe defaults
            context.account_value = 30'000.0;
            context.available_capital = 20'000.0;
        }

        // Get current positions from Schwab API
        auto positions_result = schwab_client_->account().getPositions();
        if (positions_result) {
            context.current_positions = *positions_result;
            utils::Logger::getInstance().debug("Retrieved {} current positions",
                                               context.current_positions.size());
        } else {
            utils::Logger::getInstance().error("Failed to get positions: {}",
                                               positions_result.error().message);
            context.current_positions = {};
        }

        // Get quotes for commonly traded symbols (sector ETFs)
        std::vector<std::string> symbols = {
            "SPY", "QQQ", "IWM",                                          // Market indices
            "XLE", "XLF", "XLV", "XLI", "XLK", "XLY", "XLP", "XLU", "XLB" // Sector ETFs
        };

        for (auto const& symbol : symbols) {
            auto quote_result = schwab_client_->marketData().getQuote(symbol);
            if (quote_result) {
                context.current_quotes[symbol] = *quote_result;
            } else {
                utils::Logger::getInstance().warn("Failed to get quote for {}: {}", symbol,
                                                  quote_result.error().message);
            }
        }

        utils::Logger::getInstance().debug("Retrieved {} quotes", context.current_quotes.size());

        // Load employment signals from BLS data
        loadEmploymentSignals(context);

        // Get options chains for high-conviction symbols (liquid ETFs)
        // Focus on SPY and QQQ for now - highly liquid with tight spreads
        std::vector<std::string> options_symbols = {"SPY", "QQQ"};

        for (auto const& symbol : options_symbols) {
            auto request = schwab::OptionsChainRequest::forSymbol(symbol);
            request.contract_type = "ALL";   // Both calls and puts
            request.days_to_expiration = 45; // ~45 DTE for reasonable theta decay

            auto chain_result = schwab_client_->marketData().getOptionChain(request);
            if (chain_result) {
                context.options_chains[symbol] = *chain_result;
                utils::Logger::getInstance().info(
                    "Fetched options chain for {}: {} calls, {} puts (underlying: ${:.2f})", symbol,
                    chain_result->calls.size(), chain_result->puts.size(),
                    chain_result->underlying_price);
            } else {
                utils::Logger::getInstance().warn("Failed to get options chain for {}: {}", symbol,
                                                  chain_result.error().message);
            }
        }

        utils::Logger::getInstance().debug("Retrieved {} options chains",
                                           context.options_chains.size());

        // TODO: Load correlation data

        return context;
    }

    /**
     * Load employment signals from BLS data via Python backend
     *
     * Populates the StrategyContext with:
     * - employment_signals: Individual sector employment trends
     * - rotation_signals: Sector rotation recommendations
     * - jobless_claims_alert: Recession warning (if applicable)
     */
    auto loadEmploymentSignals(strategy::StrategyContext& context) -> void {
        utils::Logger::getInstance().debug("Loading employment signals from DuckDB...");

        try {
            // Initialize employment signal generator
            auto const scripts_path = config_.get<std::string>("paths.scripts", "scripts");
            auto const db_path =
                config_.get<std::string>("database.path", "data/bigbrother.duckdb");

            employment::EmploymentSignalGenerator signal_generator{scripts_path, db_path};

            // 1. Generate employment signals for all sectors
            utils::Logger::getInstance().debug("Generating sector employment signals...");
            auto employment_signals = signal_generator.generateSignals();

            utils::Logger::getInstance().info("Loaded {} employment signals",
                                              employment_signals.size());

            // Log actionable employment signals
            int actionable_count = 0;
            for (auto const& signal : employment_signals) {
                if (signal.isActionable()) {
                    actionable_count++;
                    utils::Logger::getInstance().info(
                        "  Employment Signal: {} | {} | Confidence: {:.1f}% | Change: {:+.2f}% | "
                        "Strength: {:+.2f}",
                        signal.sector_name, signal.bullish ? "BULLISH" : "BEARISH",
                        signal.confidence * 100.0, signal.employment_change,
                        signal.signal_strength);
                    utils::Logger::getInstance().debug("    Rationale: {}", signal.rationale);
                }
            }

            utils::Logger::getInstance().info("Actionable employment signals: {}/{}",
                                              actionable_count, employment_signals.size());

            // Add signals to context
            context.employment_signals = std::move(employment_signals);

            // 2. Generate sector rotation signals
            utils::Logger::getInstance().debug("Generating sector rotation signals...");
            auto rotation_signals = signal_generator.generateRotationSignals();

            utils::Logger::getInstance().info("Loaded {} rotation signals",
                                              rotation_signals.size());

            // Log rotation recommendations
            int overweight_count = 0;
            int underweight_count = 0;
            for (auto const& rotation : rotation_signals) {
                if (rotation.isStrongSignal()) {
                    std::string action_str;
                    if (rotation.action == employment::SectorRotationSignal::Action::Overweight) {
                        action_str = "OVERWEIGHT";
                        overweight_count++;
                    } else if (rotation.action ==
                               employment::SectorRotationSignal::Action::Underweight) {
                        action_str = "UNDERWEIGHT";
                        underweight_count++;
                    } else {
                        action_str = "NEUTRAL";
                    }

                    utils::Logger::getInstance().info("  Rotation Signal: {} ({}) | {} | Score: "
                                                      "{:+.2f} | Target Allocation: {:.1f}%",
                                                      rotation.sector_name, rotation.sector_etf,
                                                      action_str, rotation.composite_score,
                                                      rotation.target_allocation);
                }
            }

            utils::Logger::getInstance().info(
                "Sector rotation recommendations: {} Overweight, {} Underweight", overweight_count,
                underweight_count);

            // Add rotation signals to context
            context.rotation_signals = std::move(rotation_signals);

            // 3. Check for jobless claims spike (recession warning)
            utils::Logger::getInstance().debug("Checking for jobless claims spike...");
            auto jobless_claims_alert = signal_generator.checkJoblessClaimsSpike();

            if (jobless_claims_alert) {
                utils::Logger::getInstance().critical(
                    "═══════════════════════════════════════════════════════");
                utils::Logger::getInstance().critical(
                    "  JOBLESS CLAIMS SPIKE DETECTED - RECESSION WARNING");
                utils::Logger::getInstance().critical("  Confidence: {:.1f}% | Change: {:+.2f}%",
                                                      jobless_claims_alert->confidence * 100.0,
                                                      jobless_claims_alert->employment_change);
                utils::Logger::getInstance().critical("  Rationale: {}",
                                                      jobless_claims_alert->rationale);
                utils::Logger::getInstance().critical(
                    "═══════════════════════════════════════════════════════");

                // Add alert to context
                context.jobless_claims_alert = jobless_claims_alert;
            } else {
                utils::Logger::getInstance().debug("No jobless claims spike detected");
            }

            // 4. Calculate and log aggregate employment health
            auto aggregate_score = context.getAggregateEmploymentScore();
            std::string health_status;
            if (aggregate_score > 0.3) {
                health_status = "STRONG";
            } else if (aggregate_score > 0.0) {
                health_status = "IMPROVING";
            } else if (aggregate_score > -0.3) {
                health_status = "WEAKENING";
            } else {
                health_status = "WEAK";
            }

            utils::Logger::getInstance().info("Aggregate employment health: {} (score: {:+.2f})",
                                              health_status, aggregate_score);

            // 5. Log strongest employment signals for quick reference
            auto strongest_signals = context.getStrongestEmploymentSignals(3);
            if (!strongest_signals.empty()) {
                utils::Logger::getInstance().info("Top employment signals:");
                for (auto const& signal : strongest_signals) {
                    utils::Logger::getInstance().info(
                        "  - {} | Strength: {:+.2f} | {:.1f}% confidence", signal.sector_name,
                        signal.signal_strength, signal.confidence * 100.0);
                }
            }

        } catch (std::exception const& e) {
            utils::Logger::getInstance().error("Failed to load employment signals: {}", e.what());
            utils::Logger::getInstance().warn(
                "Continuing without employment signals - strategies will use price data only");

            // Initialize empty signal vectors to prevent null reference issues
            context.employment_signals = {};
            context.rotation_signals = {};
            context.jobless_claims_alert = std::nullopt;
        }
    }

    // Calculate daily return for VaR/Sharpe tracking
    [[nodiscard]] auto calculateDailyReturn() const noexcept -> double {
        // Get account value from Schwab API
        auto balance_result = schwab_client_->account().getBalance();
        if (!balance_result) {
            return 0.0; // Return neutral if can't get balance
        }

        auto const current_value = balance_result->total_value;

        // Compare to yesterday's value (stored in config or database)
        auto const previous_value = config_.get<double>("account.previous_value", current_value);

        if (previous_value < 1e-6) {
            return 0.0; // Avoid division by zero
        }

        // Daily return = (current - previous) / previous
        return (current_value - previous_value) / previous_value;
    }

    auto updatePositions() -> void {
        utils::Logger::getInstance().debug("Updating positions and P&L...");

        // Get latest positions from Schwab API
        auto positions_result = schwab_client_->account().getPositions();
        if (!positions_result) {
            utils::Logger::getInstance().error("Failed to update positions: {}",
                                               positions_result.error().message);
            return;
        }

        auto const& positions = *positions_result;

        // Calculate total P&L
        double total_unrealized_pnl = 0.0;
        double total_realized_pnl = 0.0;
        int bot_managed_count = 0;

        for (auto const& position : positions) {
            total_unrealized_pnl += position.unrealized_pnl;
            total_realized_pnl += position.realized_pnl;

            if (position.is_bot_managed) {
                bot_managed_count++;

                utils::Logger::getInstance().debug(
                    "  Bot Position: {} | Qty: {} | Price: ${:.2f} | P&L: ${:.2f}", position.symbol,
                    position.quantity, position.current_price, position.unrealized_pnl);
            }
        }

        utils::Logger::getInstance().info("Positions: {} total ({} bot-managed) | Unrealized P&L: "
                                          "${:.2f} | Realized P&L: ${:.2f}",
                                          positions.size(), bot_managed_count, total_unrealized_pnl,
                                          total_realized_pnl);

        // Update positions in DuckDB for historical tracking
        if (database_) {
            try {
                std::string const create_table_sql = R"(
                    CREATE TABLE IF NOT EXISTS positions_history (
                        timestamp TIMESTAMP,
                        symbol VARCHAR,
                        quantity DOUBLE,
                        average_price DOUBLE,
                        current_price DOUBLE,
                        unrealized_pnl DOUBLE,
                        is_bot_managed BOOLEAN,
                        strategy VARCHAR
                    )
                )";

                if (auto ensure_table = database_->execute(create_table_sql); !ensure_table) {
                    utils::Logger::getInstance().error(
                        "Failed to ensure positions_history table: {}",
                        ensure_table.error().message);
                    return;
                }

                auto escapeSqlString = [](std::string const& value) -> std::string {
                    std::string escaped;
                    escaped.reserve(value.size());
                    for (auto const ch : value) {
                        if (ch == '\'') {
                            escaped += "''";
                        } else {
                            escaped.push_back(ch);
                        }
                    }
                    return escaped;
                };

                for (auto const& pos : positions) {
                    if (!pos.is_bot_managed) {
                        continue;
                    }

                    auto const strategy =
                        pos.bot_strategy.empty() ? std::string{"BOT"} : pos.bot_strategy;
                    auto const escaped_symbol = escapeSqlString(pos.symbol);
                    auto const escaped_strategy = escapeSqlString(strategy);

                    std::ostringstream insert_sql;
                    insert_sql.setf(std::ios::fixed);
                    insert_sql.precision(4);
                    insert_sql << "INSERT INTO positions_history (timestamp, symbol, quantity, "
                                  "average_price, "
                               << "current_price, unrealized_pnl, is_bot_managed, strategy) VALUES "
                                  "(CURRENT_TIMESTAMP, '"
                               << escaped_symbol << "', " << pos.quantity << ", "
                               << pos.average_price << ", " << pos.current_price << ", "
                               << pos.unrealized_pnl << ", " << (pos.is_bot_managed ? 1 : 0)
                               << ", '" << escaped_strategy << "')";

                    if (auto insert_result = database_->execute(insert_sql.str()); !insert_result) {
                        utils::Logger::getInstance().error("Failed to persist position {}: {}",
                                                           pos.symbol,
                                                           insert_result.error().message);
                    }
                }

                utils::Logger::getInstance().debug("Positions saved to database");

            } catch (std::exception const& e) {
                utils::Logger::getInstance().error("Failed to save positions to database: {}",
                                                   e.what());
            }
        }
    }

    auto checkStopLosses() -> void {
        utils::Logger::getInstance().debug("Checking stop losses...");

        // Get current positions
        auto positions_result = schwab_client_->account().getPositions();
        if (!positions_result) {
            utils::Logger::getInstance().error("Failed to get positions for stop loss check: {}",
                                               positions_result.error().message);
            return;
        }

        auto const& positions = *positions_result;

        for (auto const& position : positions) {
            // Only manage bot-managed positions
            if (!position.is_bot_managed) {
                continue;
            }

            // Calculate position loss percentage
            double position_value = position.quantity * position.average_price;
            double loss_pct = 0.0;

            if (position_value > 0.0) {
                loss_pct = (position.unrealized_pnl / position_value) * 100.0;
            }

            // Stop loss trigger: 10% loss on any single position
            constexpr double STOP_LOSS_PCT = -10.0;

            if (loss_pct <= STOP_LOSS_PCT) {
                utils::Logger::getInstance().critical(
                    "═══════════════════════════════════════════════════════");
                utils::Logger::getInstance().critical("  STOP LOSS TRIGGERED: {} ({:.1f}%)",
                                                      position.symbol, loss_pct);
                utils::Logger::getInstance().critical("  Position: {} @ ${:.2f}, Current: ${:.2f}",
                                                      position.quantity, position.average_price,
                                                      position.current_price);
                utils::Logger::getInstance().critical("  Loss: ${:.2f}", position.unrealized_pnl);
                utils::Logger::getInstance().critical(
                    "═══════════════════════════════════════════════════════");

                // Place market order to close position
                schwab::Order close_order;
                close_order.symbol = position.symbol;
                close_order.type = schwab::OrderType::Market;
                close_order.duration = schwab::OrderDuration::Day;
                close_order.quantity = position.quantity; // Close entire position

                utils::Logger::getInstance().warn("Placing STOP LOSS order to close {} position...",
                                                  position.symbol);

                auto order_result = schwab_client_->orders().placeOrder(close_order);

                if (order_result) {
                    utils::Logger::getInstance().critical("✓ STOP LOSS executed: {} (Order ID: {})",
                                                          position.symbol, *order_result);
                } else {
                    utils::Logger::getInstance().error("✗ STOP LOSS FAILED for {}: {}",
                                                       position.symbol,
                                                       order_result.error().message);

                    // This is critical - alert immediately
                    utils::Logger::getInstance().critical(
                        "MANUAL INTERVENTION REQUIRED: Stop loss order failed for {}",
                        position.symbol);
                }
            }
        }
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
