/**
 * BigBrotherAnalytics - Alert Manager Integration Example
 *
 * This file demonstrates how to integrate AlertManager into various
 * parts of the BigBrotherAnalytics C++ codebase.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 10, 2025
 * Phase 4, Week 3: Custom Alerts System
 *
 * INTEGRATION POINTS:
 * 1. main.cpp - System lifecycle alerts
 * 2. orders_manager.cppm - Trading alerts
 * 3. schwab_api.cppm - System health alerts
 * 4. Correlation engine - Data alerts
 * 5. Performance monitoring - Performance alerts
 */

import bigbrother.alert_manager;
import bigbrother.utils.database;
import bigbrother.utils.logger;

using namespace bigbrother;

// ============================================================================
// 1. MAIN.CPP INTEGRATION
// ============================================================================

/**
 * Add to TradingEngine::initialize() after database initialization:
 */
void integrateAlertsInMain() {
    // Initialize alert manager
    auto& alert_mgr = alerts::AlertManager::getInstance();
    alert_mgr.initialize(database_);

    utils::Logger::getInstance().info("Alert manager initialized");

    // System startup alert is automatically sent by AlertManager
}

/**
 * Add to TradingEngine::run() at the start:
 */
void alertSystemStart() {
    auto& alert_mgr = alerts::AlertManager::getInstance();
    // Already sent by initialize(), but can manually trigger if needed
}

/**
 * Add to graceful shutdown (signal handler or destructor):
 */
void alertSystemShutdown() {
    auto& alert_mgr = alerts::AlertManager::getInstance();
    alert_mgr.alertSystemShutdown();
}

// ============================================================================
// 2. ORDERS MANAGER INTEGRATION
// ============================================================================

/**
 * In orders_manager.cppm, add alert calls:
 */
void orderManagerIntegration() {
    auto& alert_mgr = alerts::AlertManager::getInstance();

    // Example 1: Order execution failed
    // Add after order submission fails:
    /*
    if (!order_result.success) {
        alert_mgr.alertOrderFailed(symbol, order_result.error_message);
    }
    */

    // Example 2: Position opened
    // Add after successfully opening position:
    /*
    if (position_opened) {
        alert_mgr.alertPositionOpened(symbol, position_size);
    }
    */

    // Example 3: Position closed with P&L
    // Add after closing position:
    /*
    double pnl = calculatePnL(position);
    alert_mgr.alertPositionClosed(symbol, pnl);

    // Check P&L threshold
    if (std::abs(pnl) >= 100.0) {
        alert_mgr.alertPnLThreshold(pnl, symbol);
    }
    */

    // Example 4: Stop-loss triggered
    // Add in stop-loss monitoring code:
    /*
    if (current_price <= stop_loss_price) {
        double loss = calculateLoss(position);
        alert_mgr.alertStopLossTriggered(symbol, loss);
    }
    */

    // Example 5: Order placement slow
    // Add in order execution timing:
    /*
    auto start = std::chrono::high_resolution_clock::now();
    placeOrder(order);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (duration_ms > 200.0) {  // 200ms threshold
        alert_mgr.alertOrderPlacementSlow(duration_ms);
    }
    */
}

// ============================================================================
// 3. SCHWAB API INTEGRATION
// ============================================================================

/**
 * In schwab_api.cppm, add health monitoring:
 */
void schwabApiIntegration() {
    auto& alert_mgr = alerts::AlertManager::getInstance();

    // Example 1: API disconnected
    // Add in connection monitoring:
    /*
    if (!isConnected()) {
        alert_mgr.alertSchwabApiDisconnected("Connection lost to Schwab API");
    }
    */

    // Example 2: API reconnected
    // Add after successful reconnection:
    /*
    if (reconnected) {
        alert_mgr.alertSchwabApiReconnected();
    }
    */

    // Example 3: API error rate monitoring
    // Add in error tracking:
    /*
    static int consecutive_errors = 0;
    if (api_call_failed) {
        consecutive_errors++;
        if (consecutive_errors >= 3) {
            alert_mgr.alertSchwabApiError("Multiple consecutive API failures", consecutive_errors);
        }
    } else {
        consecutive_errors = 0;
    }
    */

    // Example 4: Circuit breaker opened
    // Add in circuit breaker logic:
    /*
    if (circuit_breaker.shouldOpen()) {
        alert_mgr.alertCircuitBreakerOpened("Error rate exceeded threshold");
    }

    if (circuit_breaker.shouldClose()) {
        alert_mgr.alertCircuitBreakerClosed();
    }
    */
}

// ============================================================================
// 4. DATA COLLECTION INTEGRATION
// ============================================================================

/**
 * In employment_signals.cppm or data collection modules:
 */
void dataCollectionIntegration() {
    auto& alert_mgr = alerts::AlertManager::getInstance();

    // Example 1: Employment data updated
    // Add after successful data collection:
    /*
    if (new_employment_data_available) {
        alert_mgr.alertEmploymentDataUpdated(records_count);
    }
    */

    // Example 2: Jobless claims updated
    // Add after jobless claims update:
    /*
    if (new_jobless_claims) {
        alert_mgr.alertJoblessClaimsUpdated(records_count);

        // Check for spike
        double spike_percent = calculateSpike(current, four_week_avg);
        if (spike_percent > 10.0) {
            alert_mgr.alertJoblessClaimsSpike(spike_percent);
        }
    }
    */

    // Example 3: Correlation matrix updated
    // Add after correlation recalculation:
    /*
    if (correlations_recalculated) {
        alert_mgr.alertCorrelationMatrixUpdated(correlation_count);
    }
    */

    // Example 4: Data staleness check
    // Add in data freshness monitoring:
    /*
    auto days_old = calculateDataAge(last_update_timestamp);
    if (days_old > 7) {
        alert_mgr.alertDataStaleness("employment_data", days_old);
    }
    */

    // Example 5: Data collection failed
    // Add in error handling:
    /*
    try {
        collectData();
    } catch (std::exception const& e) {
        alert_mgr.alertDataCollectionFailed("employment_data", e.what());
    }
    */
}

// ============================================================================
// 5. SIGNAL GENERATION INTEGRATION
// ============================================================================

/**
 * In strategy.cppm or signal generation:
 */
void signalGenerationIntegration() {
    auto& alert_mgr = alerts::AlertManager::getInstance();

    // Example 1: New signal generated
    // Add after signal generation:
    /*
    if (new_signal_generated) {
        alert_mgr.alertNewSignal("sector_rotation", symbol, action);  // "BUY", "SELL", etc.
    }
    */

    // Example 2: Signal generation performance
    // Add timing monitoring:
    /*
    auto start = std::chrono::high_resolution_clock::now();
    auto signals = generateSignals();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (duration_ms > 500.0) {  // 500ms threshold
        alert_mgr.alertSignalGenerationSlow(duration_ms);
    }
    */
}

// ============================================================================
// 6. PERFORMANCE MONITORING INTEGRATION
// ============================================================================

/**
 * Add performance monitoring throughout codebase:
 */
void performanceMonitoringIntegration() {
    auto& alert_mgr = alerts::AlertManager::getInstance();

    // Example 1: Options pricing performance
    // Add in options pricing module:
    /*
    auto start = std::chrono::high_resolution_clock::now();
    auto price = calculateOptionPrice(option);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration<double, std::micro>(end - start).count();

    if (duration_us > 100.0) {  // 100μs threshold
        alert_mgr.alertOptionsPricingSlow(duration_us);
    }
    */

    // Example 2: Memory usage monitoring
    // Add in periodic resource monitoring:
    /*
    double memory_usage_percent = getMemoryUsage();
    if (memory_usage_percent > 80.0) {
        alert_mgr.alertHighMemoryUsage(memory_usage_percent);
    }
    */

    // Example 3: Database query timeout
    // Add in database wrapper:
    /*
    auto start = std::chrono::high_resolution_clock::now();
    auto result = executeQuery(query);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (duration_ms > 1000.0) {  // 1 second threshold
        alert_mgr.alertDatabaseQueryTimeout(query, duration_ms);
    }
    */

    // Example 4: Latency spike detection
    // Add in latency monitoring:
    /*
    double baseline_ms = 50.0;  // Historical baseline
    double current_ms = measureLatency();

    if (current_ms > baseline_ms * 2.0) {  // 2x multiplier
        alert_mgr.alertLatencySpike("order_execution", current_ms, baseline_ms);
    }
    */
}

// ============================================================================
// 7. RISK MANAGEMENT INTEGRATION
// ============================================================================

/**
 * In risk_management.cppm:
 */
void riskManagementIntegration() {
    auto& alert_mgr = alerts::AlertManager::getInstance();

    // Example 1: Daily loss limit approaching
    // Add in risk monitoring:
    /*
    double daily_loss = calculateDailyLoss();
    double max_daily_loss = 900.0;  // $900 for $30k account

    if (daily_loss >= max_daily_loss * 0.8) {  // 80% of limit
        alert_mgr.sendTradingAlert(
            "daily_loss_warning",
            alerts::AlertSeverity::WARNING,
            std::format("Daily loss approaching limit: ${:.2f} / ${:.2f}", daily_loss, max_daily_loss),
            std::format("{{\"current_loss\": {:.2f}, \"limit\": {:.2f}}}", daily_loss, max_daily_loss)
        );
    }
    */

    // Example 2: Win/loss streak tracking
    // Add in trade tracking:
    /*
    static int consecutive_wins = 0;
    static int consecutive_losses = 0;

    if (trade_profitable) {
        consecutive_wins++;
        consecutive_losses = 0;
        if (consecutive_wins >= 5) {
            alert_mgr.alertWinStreak(consecutive_wins);
        }
    } else {
        consecutive_losses++;
        consecutive_wins = 0;
        if (consecutive_losses >= 3) {
            alert_mgr.alertLossStreak(consecutive_losses);
        }
    }
    */
}

// ============================================================================
// 8. MAIN FUNCTION EXAMPLE
// ============================================================================

/**
 * Complete example of integrating AlertManager in main.cpp:
 */
/*
auto TradingEngine::initialize(std::string const& config_file) -> bool {
    // ... existing initialization code ...

    // Initialize database
    auto const db_path = config_.get<std::string>("database.path", "data/bigbrother.duckdb");
    database_ = std::make_unique<utils::Database>(db_path);

    if (auto connect_result = database_->connect(); !connect_result) {
        utils::Logger::getInstance().error("Failed to connect to database {}: {}", db_path,
                                           connect_result.error().message);
        return false;
    }

    // Initialize alert manager (AFTER database)
    auto& alert_mgr = alerts::AlertManager::getInstance();
    alert_mgr.initialize(database_);

    utils::Logger::getInstance().info("Alert manager initialized");

    // ... rest of initialization ...

    return true;
}

auto TradingEngine::~TradingEngine() {
    // Send shutdown alert before cleanup
    auto& alert_mgr = alerts::AlertManager::getInstance();
    alert_mgr.alertSystemShutdown();
}
*/

// ============================================================================
// NOTES FOR INTEGRATION
// ============================================================================

/**
 * KEY POINTS:
 *
 * 1. Initialize AlertManager AFTER database connection:
 *    auto& alert_mgr = alerts::AlertManager::getInstance();
 *    alert_mgr.initialize(database_);
 *
 * 2. AlertManager is a singleton - use getInstance() anywhere:
 *    alerts::AlertManager::getInstance().alertOrderFailed(symbol, reason);
 *
 * 3. Alerts are written to database, not sent directly:
 *    - C++ writes to alerts table
 *    - Python alert_processor.py polls table and delivers
 *    - Run alert processor as background service
 *
 * 4. Alert throttling is automatic:
 *    - Same alert type won't spam (5 minute default window)
 *    - Can be configured or disabled
 *
 * 5. Severity levels guide delivery:
 *    - INFO: Slack only (no email)
 *    - WARNING: Email + Slack
 *    - ERROR: Email + Slack
 *    - CRITICAL: All channels (Email + Slack + SMS if configured)
 *
 * 6. Context is JSON string for structured data:
 *    auto context = std::format("{{\"symbol\": \"{}\", \"pnl\": {:.2f}}}", symbol, pnl);
 *
 * 7. Add import at top of file:
 *    import bigbrother.alert_manager;
 *
 * 8. Start alert processor before running trading engine:
 *    uv run python scripts/monitoring/alert_processor.py --daemon
 */
