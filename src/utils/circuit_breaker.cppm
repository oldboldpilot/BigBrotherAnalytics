/**
 * BigBrotherAnalytics - Circuit Breaker Pattern Implementation (C++23)
 *
 * Implements the circuit breaker pattern to prevent cascading failures
 * in distributed systems. Protects critical external calls (Schwab API,
 * Database, BLS/FRED APIs) by opening the circuit after consecutive failures.
 *
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Circuit is open, requests fail fast (return cached data)
 * - HALF_OPEN: Testing recovery, allow limited requests
 *
 * Following C++ Core Guidelines:
 * - R.1: RAII for resource management
 * - C.21: Rule of Five
 * - Thread-safe implementation with std::mutex
 * - Trailing return types throughout
 */

// Global module fragment
module;

#include <atomic>
#include <chrono>
#include <expected>
#include <functional>
#include <mutex>
#include <string>

// Module declaration
export module bigbrother.circuit_breaker;

// Import dependencies
import bigbrother.utils.logger;
import bigbrother.utils.types;

export namespace bigbrother::circuit_breaker {

using namespace bigbrother::utils;
using namespace bigbrother::types;

/**
 * Circuit Breaker State
 */
enum class CircuitState {
    CLOSED,   // Normal operation
    OPEN,     // Circuit is open, fail fast
    HALF_OPEN // Testing recovery
};

/**
 * Circuit Breaker Statistics
 */
struct CircuitStats {
    CircuitState state{CircuitState::CLOSED};
    int total_calls{0};
    int success_count{0};
    int failure_count{0};
    int consecutive_failures{0};
    std::chrono::system_clock::time_point last_failure_time;
    std::chrono::system_clock::time_point last_success_time;
    std::chrono::system_clock::time_point circuit_opened_at;
    std::string last_error;

    [[nodiscard]] auto getSuccessRate() const noexcept -> double {
        if (total_calls == 0)
            return 1.0;
        return static_cast<double>(success_count) / static_cast<double>(total_calls);
    }
};

/**
 * Circuit Breaker Configuration
 */
struct CircuitConfig {
    int failure_threshold{5};                   // Open after N consecutive failures
    std::chrono::seconds timeout{60};           // Time before trying HALF_OPEN
    std::chrono::seconds half_open_timeout{30}; // Time in HALF_OPEN before CLOSED
    int half_open_max_calls{3};                 // Max calls in HALF_OPEN state
    bool enable_logging{true};                  // Log state transitions
    std::string name{"CircuitBreaker"};         // Name for logging

    [[nodiscard]] auto validate() const noexcept -> bool {
        return failure_threshold > 0 && timeout.count() > 0 && half_open_timeout.count() > 0 &&
               half_open_max_calls > 0;
    }
};

/**
 * Circuit Breaker
 *
 * Thread-safe implementation of the circuit breaker pattern.
 * Prevents cascading failures by opening circuit after threshold failures.
 *
 * Usage:
 *   CircuitBreaker breaker{CircuitConfig{.failure_threshold = 5, .timeout = 60s}};
 *
 *   auto result = breaker.call<Quote>([&]() -> std::expected<Quote, std::string> {
 *       return schwab_api.getQuote("SPY");
 *   });
 *
 *   if (!result) {
 *       // Circuit open or call failed
 *       if (breaker.isOpen()) {
 *           // Use cached data
 *       }
 *   }
 */
class CircuitBreaker {
  public:
    /**
     * Constructor
     *
     * @param config Circuit breaker configuration
     */
    explicit CircuitBreaker(CircuitConfig config = CircuitConfig{})
        : config_{std::move(config)}, state_{CircuitState::CLOSED}, failure_count_{0},
          half_open_calls_{0} {

        if (!config_.validate()) {
            Logger::getInstance().error("Invalid circuit breaker configuration");
            throw std::invalid_argument("Invalid circuit breaker configuration");
        }

        Logger::getInstance().info("Circuit breaker '{}' initialized: threshold={}, timeout={}s",
                                   config_.name, config_.failure_threshold,
                                   config_.timeout.count());
    }

    // C.21: Rule of Five - non-copyable due to mutex and atomic
    CircuitBreaker(CircuitBreaker const&) = delete;
    auto operator=(CircuitBreaker const&) -> CircuitBreaker& = delete;
    CircuitBreaker(CircuitBreaker&&) noexcept = delete;
    auto operator=(CircuitBreaker&&) noexcept -> CircuitBreaker& = delete;
    ~CircuitBreaker() = default;

    /**
     * Execute function with circuit breaker protection
     *
     * @tparam T Return type of the function
     * @param func Function to execute
     * @return Result of the function or error if circuit is open
     */
    template <typename T>
    [[nodiscard]] auto call(std::function<Result<T>()> func) -> Result<T> {

        // Check current state
        auto current_state = updateState();

        // If circuit is OPEN, fail fast
        if (current_state == CircuitState::OPEN) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.total_calls++;

            if (config_.enable_logging) {
                Logger::getInstance().warn("Circuit breaker '{}' is OPEN - failing fast",
                                           config_.name);
            }

            return std::unexpected(Error::make(ErrorCode::CircuitBreakerOpen,
                                               "Circuit breaker is OPEN - service unavailable"));
        }

        // If circuit is HALF_OPEN, limit calls
        if (current_state == CircuitState::HALF_OPEN) {
            std::lock_guard<std::mutex> lock(mutex_);

            if (half_open_calls_ >= config_.half_open_max_calls) {
                if (config_.enable_logging) {
                    Logger::getInstance().warn(
                        "Circuit breaker '{}' is HALF_OPEN - max calls reached", config_.name);
                }
                return std::unexpected(
                    Error::make(ErrorCode::CircuitBreakerOpen,
                                "Circuit breaker is HALF_OPEN - limited availability"));
            }

            half_open_calls_++;
        }

        // Execute the function
        auto result = func();

        // Record result
        if (result) {
            recordSuccess();
        } else {
            recordFailure(result.error().message);
        }

        return result;
    }

    /**
     * Get current circuit state
     */
    [[nodiscard]] auto getState() const noexcept -> CircuitState {
        return state_.load(std::memory_order_acquire);
    }

    /**
     * Check if circuit is open
     */
    [[nodiscard]] auto isOpen() const noexcept -> bool { return getState() == CircuitState::OPEN; }

    /**
     * Check if circuit is closed
     */
    [[nodiscard]] auto isClosed() const noexcept -> bool {
        return getState() == CircuitState::CLOSED;
    }

    /**
     * Check if circuit is half-open
     */
    [[nodiscard]] auto isHalfOpen() const noexcept -> bool {
        return getState() == CircuitState::HALF_OPEN;
    }

    /**
     * Manually reset circuit to CLOSED state
     * Use with caution - typically for manual intervention
     */
    auto reset() -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        state_.store(CircuitState::CLOSED, std::memory_order_release);
        failure_count_ = 0;
        half_open_calls_ = 0;
        stats_.consecutive_failures = 0;

        if (config_.enable_logging) {
            Logger::getInstance().info("Circuit breaker '{}' manually RESET to CLOSED",
                                       config_.name);
        }
    }

    /**
     * Get circuit breaker statistics
     */
    [[nodiscard]] auto getStats() const -> CircuitStats {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    /**
     * Get configuration
     */
    [[nodiscard]] auto getConfig() const noexcept -> CircuitConfig const& { return config_; }

    /**
     * Update configuration (thread-safe)
     */
    auto updateConfig(CircuitConfig new_config) -> void {
        if (!new_config.validate()) {
            Logger::getInstance().error("Invalid circuit breaker configuration - not updating");
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        config_ = std::move(new_config);

        Logger::getInstance().info("Circuit breaker '{}' configuration updated", config_.name);
    }

  private:
    /**
     * Update circuit state based on time and conditions
     * Returns current state after update
     */
    [[nodiscard]] auto updateState() -> CircuitState {
        auto current_state = state_.load(std::memory_order_acquire);
        auto now = std::chrono::system_clock::now();

        if (current_state == CircuitState::OPEN) {
            std::lock_guard<std::mutex> lock(mutex_);

            // Check if timeout has elapsed
            auto time_in_open = now - stats_.circuit_opened_at;

            if (time_in_open >= config_.timeout) {
                // Transition to HALF_OPEN
                transitionTo(CircuitState::HALF_OPEN);
                half_open_calls_ = 0;

                if (config_.enable_logging) {
                    Logger::getInstance().info(
                        "Circuit breaker '{}' transitioning OPEN -> HALF_OPEN (timeout elapsed)",
                        config_.name);
                }

                return CircuitState::HALF_OPEN;
            }
        }

        return current_state;
    }

    /**
     * Record successful call
     */
    auto recordSuccess() -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        stats_.total_calls++;
        stats_.success_count++;
        stats_.consecutive_failures = 0;
        stats_.last_success_time = std::chrono::system_clock::now();

        auto current_state = state_.load(std::memory_order_acquire);

        // If HALF_OPEN and successful, transition to CLOSED
        if (current_state == CircuitState::HALF_OPEN) {
            transitionTo(CircuitState::CLOSED);
            failure_count_ = 0;
            half_open_calls_ = 0;

            if (config_.enable_logging) {
                Logger::getInstance().info(
                    "Circuit breaker '{}' transitioning HALF_OPEN -> CLOSED (success)",
                    config_.name);
            }
        }
    }

    /**
     * Record failed call
     */
    auto recordFailure(std::string const& error) -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        stats_.total_calls++;
        stats_.failure_count++;
        stats_.consecutive_failures++;
        stats_.last_failure_time = std::chrono::system_clock::now();
        stats_.last_error = error;

        failure_count_++;

        auto current_state = state_.load(std::memory_order_acquire);

        // If CLOSED and reached threshold, transition to OPEN
        if (current_state == CircuitState::CLOSED &&
            stats_.consecutive_failures >= config_.failure_threshold) {

            transitionTo(CircuitState::OPEN);
            stats_.circuit_opened_at = std::chrono::system_clock::now();

            if (config_.enable_logging) {
                Logger::getInstance().error(
                    "Circuit breaker '{}' transitioning CLOSED -> OPEN ({} consecutive failures)",
                    config_.name, stats_.consecutive_failures);
            }
        }

        // If HALF_OPEN and failure, transition back to OPEN
        if (current_state == CircuitState::HALF_OPEN) {
            transitionTo(CircuitState::OPEN);
            stats_.circuit_opened_at = std::chrono::system_clock::now();
            half_open_calls_ = 0;

            if (config_.enable_logging) {
                Logger::getInstance().error("Circuit breaker '{}' transitioning HALF_OPEN -> OPEN "
                                            "(failure during recovery)",
                                            config_.name);
            }
        }
    }

    /**
     * Transition to new state (internal - must hold lock)
     */
    auto transitionTo(CircuitState new_state) -> void {
        auto old_state = state_.exchange(new_state, std::memory_order_release);

        stats_.state = new_state;

        if (config_.enable_logging && old_state != new_state) {
            Logger::getInstance().info("Circuit breaker '{}' state: {} -> {}", config_.name,
                                       stateToString(old_state), stateToString(new_state));
        }
    }

    /**
     * Convert state to string for logging
     */
    [[nodiscard]] static auto stateToString(CircuitState state) noexcept -> std::string {
        switch (state) {
            case CircuitState::CLOSED:
                return "CLOSED";
            case CircuitState::OPEN:
                return "OPEN";
            case CircuitState::HALF_OPEN:
                return "HALF_OPEN";
            default:
                return "UNKNOWN";
        }
    }

    // Configuration
    CircuitConfig config_;

    // State (atomic for lock-free reads)
    std::atomic<CircuitState> state_;

    // Counters
    int failure_count_;
    int half_open_calls_;

    // Statistics (protected by mutex)
    CircuitStats stats_;

    // Thread safety
    mutable std::mutex mutex_;
};

/**
 * Circuit Breaker Manager
 *
 * Manages multiple circuit breakers for different services.
 * Provides centralized access and monitoring.
 */
class CircuitBreakerManager {
  public:
    CircuitBreakerManager() = default;

    // C.21: Non-copyable, non-movable due to map of unique_ptr
    CircuitBreakerManager(CircuitBreakerManager const&) = delete;
    auto operator=(CircuitBreakerManager const&) -> CircuitBreakerManager& = delete;
    CircuitBreakerManager(CircuitBreakerManager&&) noexcept = delete;
    auto operator=(CircuitBreakerManager&&) noexcept -> CircuitBreakerManager& = delete;
    ~CircuitBreakerManager() = default;

    /**
     * Register a new circuit breaker
     */
    auto registerBreaker(std::string name, CircuitConfig config) -> CircuitBreaker& {
        std::lock_guard<std::mutex> lock(mutex_);

        config.name = name;
        auto breaker = std::make_unique<CircuitBreaker>(config);
        auto& ref = *breaker;

        breakers_[name] = std::move(breaker);

        Logger::getInstance().info("Registered circuit breaker: {}", name);

        return ref;
    }

    /**
     * Get circuit breaker by name
     */
    [[nodiscard]] auto getBreaker(std::string const& name) -> CircuitBreaker* {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = breakers_.find(name);
        if (it == breakers_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    /**
     * Get all circuit breaker statistics
     */
    [[nodiscard]] auto getAllStats() const -> std::vector<std::pair<std::string, CircuitStats>> {

        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::pair<std::string, CircuitStats>> all_stats;
        all_stats.reserve(breakers_.size());

        for (auto const& [name, breaker] : breakers_) {
            all_stats.emplace_back(name, breaker->getStats());
        }

        return all_stats;
    }

    /**
     * Reset all circuit breakers
     */
    auto resetAll() -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& [name, breaker] : breakers_) {
            breaker->reset();
        }

        Logger::getInstance().info("Reset all circuit breakers");
    }

    /**
     * Get count of open circuits
     */
    [[nodiscard]] auto getOpenCount() const -> int {
        std::lock_guard<std::mutex> lock(mutex_);

        int count = 0;
        for (auto const& [name, breaker] : breakers_) {
            if (breaker->isOpen()) {
                count++;
            }
        }

        return count;
    }

  private:
    std::unordered_map<std::string, std::unique_ptr<CircuitBreaker>> breakers_;
    mutable std::mutex mutex_;
};

} // namespace bigbrother::circuit_breaker
