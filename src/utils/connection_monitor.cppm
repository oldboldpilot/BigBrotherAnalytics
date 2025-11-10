/**
 * BigBrotherAnalytics - Connection Monitor & Health Check Module (C++23)
 *
 * Network connection monitoring with:
 * - Heartbeat checks for API connectivity
 * - Automatic reconnection logic
 * - Connection state tracking
 * - Health metrics collection
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase: 4 (Week 2 - Error Handling & Retry Logic)
 */

// Global module fragment
module;

#include <atomic>
#include <chrono>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// Module declaration
export module bigbrother.utils.connection_monitor;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::utils {

using namespace bigbrother::types;

// ============================================================================
// Connection State
// ============================================================================

/**
 * Connection health status
 */
enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Degraded,       // Connected but experiencing issues
    Reconnecting
};

/**
 * Connection health metrics
 */
struct ConnectionHealth {
    ConnectionState state{ConnectionState::Disconnected};
    std::chrono::steady_clock::time_point last_successful_check;
    std::chrono::steady_clock::time_point last_failure_check;
    int consecutive_successes{0};
    int consecutive_failures{0};
    int total_checks{0};
    int total_successes{0};
    int total_failures{0};
    std::chrono::milliseconds average_latency{0};
    std::string last_error_message;

    [[nodiscard]] auto getUptime() const -> std::chrono::duration<double> {
        auto now = std::chrono::steady_clock::now();
        return now - last_successful_check;
    }

    [[nodiscard]] auto getSuccessRate() const noexcept -> double {
        if (total_checks == 0) return 0.0;
        return static_cast<double>(total_successes) / total_checks;
    }

    [[nodiscard]] auto isHealthy() const noexcept -> bool {
        return state == ConnectionState::Connected &&
               consecutive_failures == 0 &&
               getSuccessRate() >= 0.95;
    }
};

// ============================================================================
// Heartbeat Configuration
// ============================================================================

/**
 * Configuration for heartbeat monitoring
 */
struct HeartbeatConfig {
    std::chrono::seconds interval{30};              // Check every 30 seconds
    std::chrono::seconds timeout{10};               // Timeout after 10 seconds
    int failure_threshold{3};                       // Fail after 3 consecutive failures
    int degraded_threshold{2};                      // Degrade after 2 failures
    std::chrono::seconds reconnect_delay{5};        // Wait 5 seconds before reconnecting
    int max_reconnect_attempts{5};                  // Max reconnect attempts before giving up
    bool auto_reconnect{true};                      // Automatically attempt reconnection

    [[nodiscard]] static auto defaultConfig() noexcept -> HeartbeatConfig {
        return HeartbeatConfig{};
    }

    [[nodiscard]] static auto aggressiveConfig() noexcept -> HeartbeatConfig {
        return HeartbeatConfig{
            .interval = std::chrono::seconds(10),
            .timeout = std::chrono::seconds(5),
            .failure_threshold = 2,
            .degraded_threshold = 1,
            .reconnect_delay = std::chrono::seconds(2),
            .max_reconnect_attempts = 10
        };
    }
};

// ============================================================================
// Connection Monitor
// ============================================================================

/**
 * Monitor connection health with automatic reconnection
 */
class ConnectionMonitor {
public:
    using HealthCheckFunc = std::function<Result<void>()>;
    using ReconnectFunc = std::function<Result<void>()>;

    explicit ConnectionMonitor(std::string name,
                             HealthCheckFunc health_check,
                             ReconnectFunc reconnect_func = nullptr,
                             HeartbeatConfig config = HeartbeatConfig::defaultConfig())
        : name_{std::move(name)},
          health_check_{std::move(health_check)},
          reconnect_func_{std::move(reconnect_func)},
          config_{config},
          running_{false},
          health_{} {

        health_.state = ConnectionState::Disconnected;
    }

    // Rule of Five - deleted due to std::thread member
    ConnectionMonitor(ConnectionMonitor const&) = delete;
    auto operator=(ConnectionMonitor const&) -> ConnectionMonitor& = delete;
    ConnectionMonitor(ConnectionMonitor&&) noexcept = delete;
    auto operator=(ConnectionMonitor&&) noexcept -> ConnectionMonitor& = delete;

    ~ConnectionMonitor() {
        stop();
    }

    /**
     * Start monitoring
     */
    auto start() -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        if (running_) {
            Logger::getInstance().warn("{}: Connection monitor already running", name_);
            return;
        }

        running_ = true;
        monitor_thread_ = std::thread(&ConnectionMonitor::monitorLoop, this);

        Logger::getInstance().info("{}: Connection monitor started (interval: {}s)",
                                 name_, config_.interval.count());
    }

    /**
     * Stop monitoring
     */
    auto stop() -> void {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) return;
            running_ = false;
        }

        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }

        Logger::getInstance().info("{}: Connection monitor stopped", name_);
    }

    /**
     * Perform immediate health check
     */
    [[nodiscard]] auto checkNow() -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);
        return performHealthCheck();
    }

    /**
     * Get current connection health
     */
    [[nodiscard]] auto getHealth() const -> ConnectionHealth {
        std::lock_guard<std::mutex> lock(mutex_);
        return health_;
    }

    /**
     * Get connection state
     */
    [[nodiscard]] auto getState() const -> ConnectionState {
        std::lock_guard<std::mutex> lock(mutex_);
        return health_.state;
    }

    /**
     * Check if connection is healthy
     */
    [[nodiscard]] auto isHealthy() const -> bool {
        std::lock_guard<std::mutex> lock(mutex_);
        return health_.isHealthy();
    }

    /**
     * Force reconnection attempt
     */
    auto reconnect() -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);
        return attemptReconnection();
    }

private:
    /**
     * Main monitoring loop
     */
    auto monitorLoop() -> void {
        Logger::getInstance().info("{}: Monitor loop started", name_);

        while (running_) {
            {
                std::lock_guard<std::mutex> lock(mutex_);

                // Perform health check
                auto check_result = performHealthCheck();

                if (check_result) {
                    // Health check succeeded
                    health_.consecutive_successes++;
                    health_.consecutive_failures = 0;
                    health_.total_successes++;
                    health_.last_successful_check = std::chrono::steady_clock::now();

                    // Transition to connected state if we were degraded/reconnecting
                    if (health_.state != ConnectionState::Connected) {
                        Logger::getInstance().info("{}: Connection restored", name_);
                        health_.state = ConnectionState::Connected;
                    }

                } else {
                    // Health check failed
                    health_.consecutive_failures++;
                    health_.consecutive_successes = 0;
                    health_.total_failures++;
                    health_.last_failure_check = std::chrono::steady_clock::now();
                    health_.last_error_message = check_result.error().message;

                    Logger::getInstance().warn(
                        "{}: Health check failed (consecutive: {}): {}",
                        name_, health_.consecutive_failures, check_result.error().message);

                    // Update connection state based on failure count
                    if (health_.consecutive_failures >= config_.failure_threshold) {
                        if (health_.state != ConnectionState::Disconnected) {
                            Logger::getInstance().error(
                                "{}: Connection lost after {} consecutive failures",
                                name_, health_.consecutive_failures);
                            health_.state = ConnectionState::Disconnected;
                        }

                        // Attempt auto-reconnection if enabled
                        if (config_.auto_reconnect && reconnect_func_) {
                            attemptReconnection();
                        }

                    } else if (health_.consecutive_failures >= config_.degraded_threshold) {
                        if (health_.state != ConnectionState::Degraded) {
                            Logger::getInstance().warn(
                                "{}: Connection degraded (failures: {})",
                                name_, health_.consecutive_failures);
                            health_.state = ConnectionState::Degraded;
                        }
                    }
                }

                health_.total_checks++;
            }

            // Sleep for configured interval
            std::this_thread::sleep_for(config_.interval);
        }

        Logger::getInstance().info("{}: Monitor loop stopped", name_);
    }

    /**
     * Perform health check
     */
    [[nodiscard]] auto performHealthCheck() -> Result<void> {
        if (!health_check_) {
            return makeError<void>(ErrorCode::NotImplemented,
                                 "Health check function not provided");
        }

        auto start_time = std::chrono::steady_clock::now();

        try {
            auto result = health_check_();
            auto end_time = std::chrono::steady_clock::now();

            // Update latency metrics
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Calculate rolling average latency
            if (health_.total_checks > 0) {
                health_.average_latency =
                    (health_.average_latency * health_.total_checks + latency) /
                    (health_.total_checks + 1);
            } else {
                health_.average_latency = latency;
            }

            return result;

        } catch (std::exception const& e) {
            return makeError<void>(ErrorCode::NetworkError,
                                 std::string("Health check exception: ") + e.what());
        }
    }

    /**
     * Attempt to reconnect
     */
    [[nodiscard]] auto attemptReconnection() -> Result<void> {
        if (!reconnect_func_) {
            Logger::getInstance().warn("{}: No reconnect function provided", name_);
            return makeError<void>(ErrorCode::NotImplemented, "Reconnect not implemented");
        }

        Logger::getInstance().info("{}: Attempting reconnection...", name_);
        health_.state = ConnectionState::Reconnecting;

        for (int attempt = 1; attempt <= config_.max_reconnect_attempts; ++attempt) {
            Logger::getInstance().info("{}: Reconnect attempt {}/{}",
                                     name_, attempt, config_.max_reconnect_attempts);

            try {
                auto result = reconnect_func_();

                if (result) {
                    Logger::getInstance().info("{}: Reconnection successful", name_);
                    health_.state = ConnectionState::Connected;
                    health_.consecutive_failures = 0;
                    health_.last_successful_check = std::chrono::steady_clock::now();
                    return result;
                }

                Logger::getInstance().warn("{}: Reconnect attempt {} failed: {}",
                                         name_, attempt, result.error().message);

            } catch (std::exception const& e) {
                Logger::getInstance().error("{}: Reconnect attempt {} threw exception: {}",
                                          name_, attempt, e.what());
            }

            // Wait before next attempt (unless it's the last one)
            if (attempt < config_.max_reconnect_attempts) {
                std::this_thread::sleep_for(config_.reconnect_delay);
            }
        }

        Logger::getInstance().error("{}: Reconnection failed after {} attempts",
                                  name_, config_.max_reconnect_attempts);
        health_.state = ConnectionState::Disconnected;

        return makeError<void>(ErrorCode::NetworkError,
                             "Reconnection failed after " +
                             std::to_string(config_.max_reconnect_attempts) + " attempts");
    }

    std::string name_;
    HealthCheckFunc health_check_;
    ReconnectFunc reconnect_func_;
    HeartbeatConfig config_;
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    mutable std::mutex mutex_;
    ConnectionHealth health_;
};

// ============================================================================
// Connection Pool Manager
// ============================================================================

/**
 * Manage multiple connection monitors
 */
class ConnectionPoolManager {
public:
    /**
     * Register a new connection monitor
     */
    auto registerMonitor(std::string const& name,
                        std::unique_ptr<ConnectionMonitor> monitor) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        monitors_[name] = std::move(monitor);
        Logger::getInstance().info("Registered connection monitor: {}", name);
    }

    /**
     * Start all monitors
     */
    auto startAll() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [name, monitor] : monitors_) {
            monitor->start();
        }
        Logger::getInstance().info("Started {} connection monitors", monitors_.size());
    }

    /**
     * Stop all monitors
     */
    auto stopAll() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [name, monitor] : monitors_) {
            monitor->stop();
        }
        Logger::getInstance().info("Stopped {} connection monitors", monitors_.size());
    }

    /**
     * Get monitor by name
     */
    [[nodiscard]] auto getMonitor(std::string const& name) -> ConnectionMonitor* {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = monitors_.find(name);
        return (it != monitors_.end()) ? it->second.get() : nullptr;
    }

    /**
     * Get health summary for all monitors
     */
    [[nodiscard]] auto getHealthSummary() const -> std::vector<std::pair<std::string, ConnectionHealth>> {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, ConnectionHealth>> summary;
        summary.reserve(monitors_.size());

        for (auto const& [name, monitor] : monitors_) {
            summary.emplace_back(name, monitor->getHealth());
        }

        return summary;
    }

    /**
     * Check if all monitors are healthy
     */
    [[nodiscard]] auto areAllHealthy() const -> bool {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto const& [name, monitor] : monitors_) {
            if (!monitor->isHealthy()) {
                return false;
            }
        }
        return true;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ConnectionMonitor>> monitors_;
};

} // namespace bigbrother::utils
