/**
 * BigBrotherAnalytics - Alert Manager Module (C++23)
 *
 * Comprehensive alerting system for trading events, system health, and performance.
 * Integrates with database-backed alert queue for Python processor delivery.
 *
 * Features:
 * - Trading alerts (P&L, stop-loss, signals, orders)
 * - Data alerts (updates, staleness, quality)
 * - System health alerts (circuit breaker, API, database)
 * - Performance alerts (latency, resource usage)
 * - Alert throttling to prevent spam
 * - Severity-based routing (INFO, WARNING, ERROR, CRITICAL)
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 10, 2025
 * Phase 4, Week 3: Custom Alerts System
 */

// ============================================================================
// 1. GLOBAL MODULE FRAGMENT
// ============================================================================
module;

#include <chrono>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

// ============================================================================
// 2. MODULE DECLARATION
// ============================================================================
export module bigbrother.alert_manager;

import bigbrother.utils.database;
import bigbrother.utils.logger;

// ============================================================================
// 3. EXPORTED INTERFACE
// ============================================================================
export namespace bigbrother::alerts {

/**
 * Alert Severity Levels
 */
enum class AlertSeverity {
    INFO,      // Informational (new signal, data update)
    WARNING,   // Needs attention (high P&L, stale data)
    ERROR,     // Immediate action (circuit open, failed order)
    CRITICAL   // System failure (database down, API disconnected)
};

/**
 * Alert Types
 */
enum class AlertType {
    TRADING,      // Trading-related alerts
    DATA,         // Data collection and quality alerts
    SYSTEM,       // System health alerts
    PERFORMANCE   // Performance and latency alerts
};

/**
 * Alert Manager
 *
 * Manages alert generation, throttling, and database storage.
 * Alerts are written to database and picked up by Python processor for delivery.
 *
 * Usage:
 *   auto& alert_mgr = AlertManager::getInstance();
 *   alert_mgr.initialize(db);
 *
 *   // Send alerts
 *   alert_mgr.sendTradingAlert("stop_loss_triggered", AlertSeverity::ERROR,
 *                              "Stop-loss triggered for AAPL", "{'symbol': 'AAPL', 'loss': 150.0}");
 *
 *   alert_mgr.sendSystemAlert("circuit_breaker_opened", AlertSeverity::CRITICAL,
 *                             "Circuit breaker opened - trading halted");
 */
class AlertManager {
  public:
    /**
     * Get singleton instance
     */
    [[nodiscard]] static auto getInstance() -> AlertManager&;

    // Non-copyable, non-movable (singleton)
    AlertManager(AlertManager const&) = delete;
    auto operator=(AlertManager const&) -> AlertManager& = delete;
    AlertManager(AlertManager&&) = delete;
    auto operator=(AlertManager&&) -> AlertManager& = delete;

    /**
     * Initialize alert manager with database connection
     */
    auto initialize(std::shared_ptr<utils::Database> db) -> void;

    /**
     * Enable/disable alert throttling
     */
    auto setThrottlingEnabled(bool enabled) -> void;

    /**
     * Set default throttle window (seconds)
     */
    auto setDefaultThrottleWindow(int seconds) -> void;

    // ========================================================================
    // TRADING ALERTS
    // ========================================================================

    /**
     * Send generic trading alert
     */
    auto sendTradingAlert(std::string_view subtype, AlertSeverity severity,
                          std::string_view message, std::string_view context = "") -> void;

    /**
     * P&L threshold alert
     */
    auto alertPnLThreshold(double pnl_change, std::string_view symbol = "") -> void;

    /**
     * Stop-loss triggered alert
     */
    auto alertStopLossTriggered(std::string_view symbol, double loss_amount) -> void;

    /**
     * New signal generated alert
     */
    auto alertNewSignal(std::string_view signal_type, std::string_view symbol,
                        std::string_view action) -> void;

    /**
     * Order execution failed alert
     */
    auto alertOrderFailed(std::string_view symbol, std::string_view reason) -> void;

    /**
     * Position opened/closed alerts
     */
    auto alertPositionOpened(std::string_view symbol, double position_size) -> void;
    auto alertPositionClosed(std::string_view symbol, double pnl) -> void;

    /**
     * Win/loss streak alert
     */
    auto alertWinStreak(int streak_count) -> void;
    auto alertLossStreak(int streak_count) -> void;

    // ========================================================================
    // DATA ALERTS
    // ========================================================================

    /**
     * Send generic data alert
     */
    auto sendDataAlert(std::string_view subtype, AlertSeverity severity,
                       std::string_view message, std::string_view context = "") -> void;

    /**
     * Employment data updated alert
     */
    auto alertEmploymentDataUpdated(int records_updated) -> void;

    /**
     * Jobless claims updated alert
     */
    auto alertJoblessClaimsUpdated(int records_updated) -> void;

    /**
     * Jobless claims spike detected alert
     */
    auto alertJoblessClaimsSpike(double spike_percent) -> void;

    /**
     * Correlation matrix updated alert
     */
    auto alertCorrelationMatrixUpdated(int correlations_count) -> void;

    /**
     * Data staleness warning alert
     */
    auto alertDataStaleness(std::string_view data_type, int days_old) -> void;

    /**
     * Data collection failure alert
     */
    auto alertDataCollectionFailed(std::string_view data_type, std::string_view error) -> void;

    // ========================================================================
    // SYSTEM HEALTH ALERTS
    // ========================================================================

    /**
     * Send generic system alert
     */
    auto sendSystemAlert(std::string_view subtype, AlertSeverity severity,
                         std::string_view message, std::string_view context = "") -> void;

    /**
     * Circuit breaker alerts
     */
    auto alertCircuitBreakerOpened(std::string_view reason) -> void;
    auto alertCircuitBreakerClosed() -> void;

    /**
     * Schwab API alerts
     */
    auto alertSchwabApiDisconnected(std::string_view reason) -> void;
    auto alertSchwabApiReconnected() -> void;
    auto alertSchwabApiError(std::string_view error, int error_count = 1) -> void;

    /**
     * Database alerts
     */
    auto alertDatabaseConnectionLost(std::string_view error) -> void;
    auto alertDatabaseQueryTimeout(std::string_view query, double duration_ms) -> void;

    /**
     * System lifecycle alerts
     */
    auto alertSystemStartup() -> void;
    auto alertSystemShutdown() -> void;
    auto alertUnexpectedCrash(std::string_view error) -> void;

    // ========================================================================
    // PERFORMANCE ALERTS
    // ========================================================================

    /**
     * Send generic performance alert
     */
    auto sendPerformanceAlert(std::string_view subtype, AlertSeverity severity,
                              std::string_view message, std::string_view context = "") -> void;

    /**
     * Latency threshold alerts
     */
    auto alertSignalGenerationSlow(double duration_ms) -> void;
    auto alertOrderPlacementSlow(double duration_ms) -> void;
    auto alertOptionsPricingSlow(double duration_us) -> void;

    /**
     * Resource usage alerts
     */
    auto alertHighMemoryUsage(double usage_percent) -> void;
    auto alertHighCpuUsage(double usage_percent) -> void;
    auto alertLowDiskSpace(double free_percent) -> void;

    /**
     * Performance degradation alerts
     */
    auto alertLatencySpike(std::string_view operation, double current_ms,
                           double baseline_ms) -> void;

  private:
    AlertManager() = default;
    ~AlertManager() = default;

    // Core alert sending method
    auto sendAlert(AlertType type, std::string_view subtype, AlertSeverity severity,
                   std::string_view message, std::string_view context,
                   std::string_view source) -> void;

    // Throttling
    auto shouldThrottle(std::string_view throttle_key) -> bool;
    auto updateThrottle(std::string_view throttle_key) -> void;

    // Helpers
    [[nodiscard]] static auto severityToString(AlertSeverity severity) -> std::string;
    [[nodiscard]] static auto typeToString(AlertType type) -> std::string;

    // State
    std::shared_ptr<utils::Database> database_;
    bool initialized_{false};
    bool throttling_enabled_{true};
    int default_throttle_window_seconds_{300}; // 5 minutes

    // Throttle tracking (in-memory, also uses database)
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> throttle_map_;
    std::mutex throttle_mutex_;
};

} // namespace bigbrother::alerts

// ============================================================================
// 4. PRIVATE IMPLEMENTATION
// ============================================================================
module :private;

namespace bigbrother::alerts {

[[nodiscard]] auto AlertManager::getInstance() -> AlertManager& {
    static AlertManager instance;
    return instance;
}

auto AlertManager::initialize(std::shared_ptr<utils::Database> db) -> void {
    database_ = db;
    initialized_ = true;

    utils::Logger::getInstance().info("AlertManager initialized");

    // Send startup alert
    alertSystemStartup();
}

auto AlertManager::setThrottlingEnabled(bool enabled) -> void {
    throttling_enabled_ = enabled;
}

auto AlertManager::setDefaultThrottleWindow(int seconds) -> void {
    default_throttle_window_seconds_ = seconds;
}

// ============================================================================
// CORE ALERT SENDING
// ============================================================================

auto AlertManager::sendAlert(AlertType type, std::string_view subtype, AlertSeverity severity,
                              std::string_view message, std::string_view context,
                              std::string_view source) -> void {
    if (!initialized_ || !database_) {
        utils::Logger::getInstance().warn("AlertManager not initialized, cannot send alert");
        return;
    }

    // Generate throttle key
    auto throttle_key = std::format("{}:{}:{}", typeToString(type), subtype, severityToString(severity));

    // Check throttling
    if (throttling_enabled_ && shouldThrottle(throttle_key)) {
        utils::Logger::getInstance().debug("Alert throttled: {}", throttle_key);
        return;
    }

    // Log the alert
    utils::Logger::getInstance().info("[ALERT] [{}] {}", severityToString(severity), message);

    try {
        // Insert into alerts table
        auto query = std::format(
            "INSERT INTO alerts (alert_type, alert_subtype, severity, message, context, source, "
            "throttle_key) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}')",
            typeToString(type), subtype, severityToString(severity), message, context, source,
            throttle_key);

        database_->execute(query);

        // Update throttle
        updateThrottle(throttle_key);

    } catch (std::exception const& e) {
        utils::Logger::getInstance().error("Failed to insert alert into database: {}", e.what());
    }
}

// ============================================================================
// THROTTLING
// ============================================================================

auto AlertManager::shouldThrottle(std::string_view throttle_key) -> bool {
    std::lock_guard<std::mutex> lock(throttle_mutex_);

    auto now = std::chrono::steady_clock::now();
    auto key_str = std::string(throttle_key);

    // Check in-memory map
    if (throttle_map_.contains(key_str)) {
        auto last_sent = throttle_map_[key_str];
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_sent).count();

        if (elapsed < default_throttle_window_seconds_) {
            return true; // Throttled
        }
    }

    return false;
}

auto AlertManager::updateThrottle(std::string_view throttle_key) -> void {
    std::lock_guard<std::mutex> lock(throttle_mutex_);

    auto now = std::chrono::steady_clock::now();
    throttle_map_[std::string(throttle_key)] = now;
}

// ============================================================================
// TRADING ALERTS
// ============================================================================

auto AlertManager::sendTradingAlert(std::string_view subtype, AlertSeverity severity,
                                    std::string_view message, std::string_view context) -> void {
    sendAlert(AlertType::TRADING, subtype, severity, message, context, "trading_engine");
}

auto AlertManager::alertPnLThreshold(double pnl_change, std::string_view symbol) -> void {
    auto severity = (pnl_change > 0) ? AlertSeverity::INFO : AlertSeverity::WARNING;
    auto message = std::format("P&L threshold reached: ${:.2f} on {}", pnl_change,
                               symbol.empty() ? "portfolio" : std::string(symbol));
    auto context = std::format("{{\"pnl\": {:.2f}, \"symbol\": \"{}\"}}", pnl_change, symbol);
    sendTradingAlert("pnl_threshold", severity, message, context);
}

auto AlertManager::alertStopLossTriggered(std::string_view symbol, double loss_amount) -> void {
    auto message = std::format("Stop-loss triggered for {} - Loss: ${:.2f}", symbol, loss_amount);
    auto context = std::format("{{\"symbol\": \"{}\", \"loss\": {:.2f}}}", symbol, loss_amount);
    sendTradingAlert("stop_loss_triggered", AlertSeverity::ERROR, message, context);
}

auto AlertManager::alertNewSignal(std::string_view signal_type, std::string_view symbol,
                                  std::string_view action) -> void {
    auto message = std::format("New {} signal: {} on {}", signal_type, action, symbol);
    auto context = std::format("{{\"type\": \"{}\", \"symbol\": \"{}\", \"action\": \"{}\"}}",
                               signal_type, symbol, action);
    sendTradingAlert("new_signal", AlertSeverity::INFO, message, context);
}

auto AlertManager::alertOrderFailed(std::string_view symbol, std::string_view reason) -> void {
    auto message = std::format("Order execution failed for {}: {}", symbol, reason);
    auto context = std::format("{{\"symbol\": \"{}\", \"reason\": \"{}\"}}", symbol, reason);
    sendTradingAlert("order_failed", AlertSeverity::ERROR, message, context);
}

auto AlertManager::alertPositionOpened(std::string_view symbol, double position_size) -> void {
    auto message = std::format("Position opened: {} (size: ${:.2f})", symbol, position_size);
    auto context =
        std::format("{{\"symbol\": \"{}\", \"size\": {:.2f}}}", symbol, position_size);
    sendTradingAlert("position_opened", AlertSeverity::INFO, message, context);
}

auto AlertManager::alertPositionClosed(std::string_view symbol, double pnl) -> void {
    auto severity = (pnl >= 0) ? AlertSeverity::INFO : AlertSeverity::WARNING;
    auto message = std::format("Position closed: {} - P&L: ${:.2f}", symbol, pnl);
    auto context = std::format("{{\"symbol\": \"{}\", \"pnl\": {:.2f}}}", symbol, pnl);
    sendTradingAlert("position_closed", severity, message, context);
}

auto AlertManager::alertWinStreak(int streak_count) -> void {
    auto message = std::format("Win streak achieved: {} consecutive winning trades", streak_count);
    auto context = std::format("{{\"streak\": {}}}", streak_count);
    sendTradingAlert("win_streak", AlertSeverity::INFO, message, context);
}

auto AlertManager::alertLossStreak(int streak_count) -> void {
    auto message = std::format("Loss streak detected: {} consecutive losing trades", streak_count);
    auto context = std::format("{{\"streak\": {}}}", streak_count);
    sendTradingAlert("loss_streak", AlertSeverity::WARNING, message, context);
}

// ============================================================================
// DATA ALERTS
// ============================================================================

auto AlertManager::sendDataAlert(std::string_view subtype, AlertSeverity severity,
                                 std::string_view message, std::string_view context) -> void {
    sendAlert(AlertType::DATA, subtype, severity, message, context, "data_collection");
}

auto AlertManager::alertEmploymentDataUpdated(int records_updated) -> void {
    auto message = std::format("Employment data updated - {} records", records_updated);
    auto context = std::format("{{\"records\": {}}}", records_updated);
    sendDataAlert("employment_updated", AlertSeverity::INFO, message, context);
}

auto AlertManager::alertJoblessClaimsUpdated(int records_updated) -> void {
    auto message = std::format("Jobless claims data updated - {} records", records_updated);
    auto context = std::format("{{\"records\": {}}}", records_updated);
    sendDataAlert("jobless_claims_updated", AlertSeverity::INFO, message, context);
}

auto AlertManager::alertJoblessClaimsSpike(double spike_percent) -> void {
    auto message =
        std::format("Jobless claims spike detected: {:.1f}% above 4-week average", spike_percent);
    auto context = std::format("{{\"spike_percent\": {:.1f}}}", spike_percent);
    sendDataAlert("jobless_spike", AlertSeverity::WARNING, message, context);
}

auto AlertManager::alertCorrelationMatrixUpdated(int correlations_count) -> void {
    auto message = std::format("Correlation matrix updated - {} correlations", correlations_count);
    auto context = std::format("{{\"count\": {}}}", correlations_count);
    sendDataAlert("correlation_updated", AlertSeverity::INFO, message, context);
}

auto AlertManager::alertDataStaleness(std::string_view data_type, int days_old) -> void {
    auto message = std::format("{} data is stale - {} days old", data_type, days_old);
    auto context = std::format("{{\"type\": \"{}\", \"days_old\": {}}}", data_type, days_old);
    sendDataAlert("data_stale", AlertSeverity::WARNING, message, context);
}

auto AlertManager::alertDataCollectionFailed(std::string_view data_type,
                                             std::string_view error) -> void {
    auto message = std::format("{} data collection failed: {}", data_type, error);
    auto context = std::format("{{\"type\": \"{}\", \"error\": \"{}\"}}", data_type, error);
    sendDataAlert("collection_failed", AlertSeverity::ERROR, message, context);
}

// ============================================================================
// SYSTEM HEALTH ALERTS
// ============================================================================

auto AlertManager::sendSystemAlert(std::string_view subtype, AlertSeverity severity,
                                   std::string_view message, std::string_view context) -> void {
    sendAlert(AlertType::SYSTEM, subtype, severity, message, context, "system");
}

auto AlertManager::alertCircuitBreakerOpened(std::string_view reason) -> void {
    auto message = std::format("Circuit breaker OPENED - Trading halted: {}", reason);
    auto context = std::format("{{\"reason\": \"{}\"}}", reason);
    sendSystemAlert("circuit_breaker_opened", AlertSeverity::CRITICAL, message, context);
}

auto AlertManager::alertCircuitBreakerClosed() -> void {
    sendSystemAlert("circuit_breaker_closed", AlertSeverity::INFO,
                    "Circuit breaker closed - Trading resumed", "{}");
}

auto AlertManager::alertSchwabApiDisconnected(std::string_view reason) -> void {
    auto message = std::format("Schwab API DISCONNECTED: {}", reason);
    auto context = std::format("{{\"reason\": \"{}\"}}", reason);
    sendSystemAlert("schwab_api_disconnected", AlertSeverity::CRITICAL, message, context);
}

auto AlertManager::alertSchwabApiReconnected() -> void {
    sendSystemAlert("schwab_api_reconnected", AlertSeverity::INFO, "Schwab API reconnected", "{}");
}

auto AlertManager::alertSchwabApiError(std::string_view error, int error_count) -> void {
    auto severity = (error_count >= 3) ? AlertSeverity::ERROR : AlertSeverity::WARNING;
    auto message = std::format("Schwab API error (count: {}): {}", error_count, error);
    auto context = std::format("{{\"error\": \"{}\", \"count\": {}}}", error, error_count);
    sendSystemAlert("schwab_api_error", severity, message, context);
}

auto AlertManager::alertDatabaseConnectionLost(std::string_view error) -> void {
    auto message = std::format("Database connection LOST: {}", error);
    auto context = std::format("{{\"error\": \"{}\"}}", error);
    sendSystemAlert("database_connection_lost", AlertSeverity::CRITICAL, message, context);
}

auto AlertManager::alertDatabaseQueryTimeout(std::string_view query, double duration_ms) -> void {
    auto message = std::format("Database query timeout: {:.2f}ms", duration_ms);
    auto context = std::format("{{\"query\": \"{}\", \"duration_ms\": {:.2f}}}", query, duration_ms);
    sendSystemAlert("database_query_timeout", AlertSeverity::WARNING, message, context);
}

auto AlertManager::alertSystemStartup() -> void {
    sendSystemAlert("system_startup", AlertSeverity::INFO, "BigBrotherAnalytics system started",
                    "{}");
}

auto AlertManager::alertSystemShutdown() -> void {
    sendSystemAlert("system_shutdown", AlertSeverity::INFO,
                    "BigBrotherAnalytics system shutdown (graceful)", "{}");
}

auto AlertManager::alertUnexpectedCrash(std::string_view error) -> void {
    auto message = std::format("System CRASHED unexpectedly: {}", error);
    auto context = std::format("{{\"error\": \"{}\"}}", error);
    sendSystemAlert("unexpected_crash", AlertSeverity::CRITICAL, message, context);
}

// ============================================================================
// PERFORMANCE ALERTS
// ============================================================================

auto AlertManager::sendPerformanceAlert(std::string_view subtype, AlertSeverity severity,
                                        std::string_view message,
                                        std::string_view context) -> void {
    sendAlert(AlertType::PERFORMANCE, subtype, severity, message, context, "performance_monitor");
}

auto AlertManager::alertSignalGenerationSlow(double duration_ms) -> void {
    auto message = std::format("Signal generation slow: {:.2f}ms (threshold: 500ms)", duration_ms);
    auto context = std::format("{{\"duration_ms\": {:.2f}}}", duration_ms);
    sendPerformanceAlert("signal_generation_slow", AlertSeverity::WARNING, message, context);
}

auto AlertManager::alertOrderPlacementSlow(double duration_ms) -> void {
    auto message = std::format("Order placement slow: {:.2f}ms (threshold: 200ms)", duration_ms);
    auto context = std::format("{{\"duration_ms\": {:.2f}}}", duration_ms);
    sendPerformanceAlert("order_placement_slow", AlertSeverity::WARNING, message, context);
}

auto AlertManager::alertOptionsPricingSlow(double duration_us) -> void {
    auto message = std::format("Options pricing slow: {:.2f}μs (threshold: 100μs)", duration_us);
    auto context = std::format("{{\"duration_us\": {:.2f}}}", duration_us);
    sendPerformanceAlert("options_pricing_slow", AlertSeverity::WARNING, message, context);
}

auto AlertManager::alertHighMemoryUsage(double usage_percent) -> void {
    auto severity = (usage_percent >= 90.0) ? AlertSeverity::ERROR : AlertSeverity::WARNING;
    auto message = std::format("High memory usage: {:.1f}%", usage_percent);
    auto context = std::format("{{\"usage_percent\": {:.1f}}}", usage_percent);
    sendPerformanceAlert("high_memory_usage", severity, message, context);
}

auto AlertManager::alertHighCpuUsage(double usage_percent) -> void {
    auto severity = (usage_percent >= 95.0) ? AlertSeverity::ERROR : AlertSeverity::WARNING;
    auto message = std::format("High CPU usage: {:.1f}%", usage_percent);
    auto context = std::format("{{\"usage_percent\": {:.1f}}}", usage_percent);
    sendPerformanceAlert("high_cpu_usage", severity, message, context);
}

auto AlertManager::alertLowDiskSpace(double free_percent) -> void {
    auto severity = (free_percent <= 5.0) ? AlertSeverity::CRITICAL : AlertSeverity::WARNING;
    auto message = std::format("Low disk space: {:.1f}% free", free_percent);
    auto context = std::format("{{\"free_percent\": {:.1f}}}", free_percent);
    sendPerformanceAlert("low_disk_space", severity, message, context);
}

auto AlertManager::alertLatencySpike(std::string_view operation, double current_ms,
                                     double baseline_ms) -> void {
    auto multiplier = current_ms / baseline_ms;
    auto message = std::format("{} latency spike: {:.2f}ms ({:.1f}x baseline)", operation,
                               current_ms, multiplier);
    auto context = std::format(
        "{{\"operation\": \"{}\", \"current_ms\": {:.2f}, \"baseline_ms\": {:.2f}, \"multiplier\": "
        "{:.1f}}}",
        operation, current_ms, baseline_ms, multiplier);
    sendPerformanceAlert("latency_spike", AlertSeverity::WARNING, message, context);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

[[nodiscard]] auto AlertManager::severityToString(AlertSeverity severity) -> std::string {
    switch (severity) {
        case AlertSeverity::INFO:
            return "INFO";
        case AlertSeverity::WARNING:
            return "WARNING";
        case AlertSeverity::ERROR:
            return "ERROR";
        case AlertSeverity::CRITICAL:
            return "CRITICAL";
        default:
            return "UNKNOWN";
    }
}

[[nodiscard]] auto AlertManager::typeToString(AlertType type) -> std::string {
    switch (type) {
        case AlertType::TRADING:
            return "trading";
        case AlertType::DATA:
            return "data";
        case AlertType::SYSTEM:
            return "system";
        case AlertType::PERFORMANCE:
            return "performance";
        default:
            return "unknown";
    }
}

} // namespace bigbrother::alerts
