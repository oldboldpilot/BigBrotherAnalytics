/**
 * BigBrotherAnalytics - Retry Logic & Error Handling Module (C++23)
 *
 * Robust retry mechanisms with exponential backoff for:
 * - Network operations (Schwab API calls)
 * - Database operations (DuckDB queries)
 * - Transient error recovery
 *
 * Features:
 * - Exponential backoff with jitter
 * - Error categorization (transient vs permanent)
 * - Comprehensive retry metrics
 * - Thread-safe retry state management
 * - Circuit breaker pattern for repeated failures
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
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.utils.retry;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::utils {

using namespace bigbrother::types;

// ============================================================================
// Error Classification
// ============================================================================

/**
 * Error category for retry decision making
 */
enum class ErrorCategory {
    Transient,   // Retry: network timeout, rate limit, temporary unavailability
    Permanent,   // Don't retry: invalid credentials, bad parameters
    Unknown      // Retry cautiously with low max attempts
};

/**
 * Retry policy configuration
 */
struct RetryPolicy {
    int max_attempts{3};
    std::chrono::milliseconds initial_backoff{100};
    std::chrono::milliseconds max_backoff{10000};
    double backoff_multiplier{2.0};
    bool add_jitter{true};
    double jitter_factor{0.1};  // ±10% jitter

    [[nodiscard]] static auto defaultPolicy() noexcept -> RetryPolicy {
        return RetryPolicy{};
    }

    [[nodiscard]] static auto aggressivePolicy() noexcept -> RetryPolicy {
        return RetryPolicy{
            .max_attempts = 5,
            .initial_backoff = std::chrono::milliseconds(50),
            .max_backoff = std::chrono::milliseconds(5000),
            .backoff_multiplier = 1.5
        };
    }

    [[nodiscard]] static auto conservativePolicy() noexcept -> RetryPolicy {
        return RetryPolicy{
            .max_attempts = 2,
            .initial_backoff = std::chrono::milliseconds(500),
            .max_backoff = std::chrono::milliseconds(30000),
            .backoff_multiplier = 3.0
        };
    }
};

/**
 * Retry statistics for monitoring
 */
struct RetryStats {
    std::atomic<int> total_attempts{0};
    std::atomic<int> total_successes{0};
    std::atomic<int> total_failures{0};
    std::atomic<int> total_retries{0};
    std::chrono::milliseconds total_delay{0};
    std::string operation_name;

    auto recordAttempt() -> void { total_attempts++; }
    auto recordSuccess() -> void { total_successes++; }
    auto recordFailure() -> void { total_failures++; }
    auto recordRetry() -> void { total_retries++; }
    auto addDelay(std::chrono::milliseconds delay) -> void { total_delay += delay; }

    [[nodiscard]] auto getSuccessRate() const noexcept -> double {
        int attempts = total_attempts.load();
        if (attempts == 0) return 0.0;
        return static_cast<double>(total_successes.load()) / attempts;
    }

    [[nodiscard]] auto getAverageDelay() const noexcept -> std::chrono::milliseconds {
        int retries = total_retries.load();
        if (retries == 0) return std::chrono::milliseconds(0);
        return total_delay / retries;
    }
};

// ============================================================================
// Error Categorizer
// ============================================================================

/**
 * Categorize errors for retry decision making
 */
class ErrorCategorizer {
public:
    [[nodiscard]] static auto categorize(Error const& error) noexcept -> ErrorCategory {
        switch (error.code) {
            // Transient errors - always retry
            case ErrorCode::NetworkError:
            case ErrorCode::Timeout:
            case ErrorCode::ServiceUnavailable:
            case ErrorCode::RateLimitExceeded:
            case ErrorCode::DatabaseLocked:
                return ErrorCategory::Transient;

            // Permanent errors - never retry
            case ErrorCode::AuthenticationFailed:
            case ErrorCode::InvalidParameter:
            case ErrorCode::NotFound:
            case ErrorCode::OrderRejected:
            case ErrorCode::InsufficientFunds:
                return ErrorCategory::Permanent;

            // Unknown/other - retry cautiously
            default:
                return ErrorCategory::Unknown;
        }
    }

    [[nodiscard]] static auto isTransient(Error const& error) noexcept -> bool {
        return categorize(error) == ErrorCategory::Transient;
    }

    [[nodiscard]] static auto isPermanent(Error const& error) noexcept -> bool {
        return categorize(error) == ErrorCategory::Permanent;
    }

    /**
     * Analyze error message for transient indicators
     */
    [[nodiscard]] static auto analyzeMessage(std::string const& message) noexcept -> ErrorCategory {
        // Check for common transient error patterns
        std::vector<std::string_view> transient_patterns = {
            "timeout", "timed out", "connection", "network", "503", "504",
            "unavailable", "temporarily", "rate limit", "too many requests",
            "429", "locked", "busy"
        };

        std::string lower_message = message;
        std::transform(lower_message.begin(), lower_message.end(),
                      lower_message.begin(), ::tolower);

        for (auto const& pattern : transient_patterns) {
            if (lower_message.find(pattern) != std::string::npos) {
                return ErrorCategory::Transient;
            }
        }

        // Check for permanent error patterns
        std::vector<std::string_view> permanent_patterns = {
            "invalid", "unauthorized", "forbidden", "not found",
            "bad request", "401", "403", "404", "400"
        };

        for (auto const& pattern : permanent_patterns) {
            if (lower_message.find(pattern) != std::string::npos) {
                return ErrorCategory::Permanent;
            }
        }

        return ErrorCategory::Unknown;
    }
};

// ============================================================================
// Backoff Calculator
// ============================================================================

/**
 * Calculate backoff delays with exponential backoff and jitter
 */
class BackoffCalculator {
public:
    BackoffCalculator() : rng_(std::random_device{}()), dist_(-1.0, 1.0) {}

    [[nodiscard]] auto calculateDelay(int attempt, RetryPolicy const& policy) -> std::chrono::milliseconds {
        if (attempt <= 0) return std::chrono::milliseconds(0);

        // Exponential backoff: initial * (multiplier ^ (attempt - 1))
        double base_delay_ms = policy.initial_backoff.count() *
                              std::pow(policy.backoff_multiplier, attempt - 1);

        // Cap at max backoff
        if (base_delay_ms > policy.max_backoff.count()) {
            base_delay_ms = policy.max_backoff.count();
        }

        // Add jitter if enabled
        if (policy.add_jitter) {
            std::lock_guard<std::mutex> lock(mutex_);
            double jitter = dist_(rng_) * policy.jitter_factor;
            base_delay_ms *= (1.0 + jitter);
        }

        return std::chrono::milliseconds(static_cast<int64_t>(base_delay_ms));
    }

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
    std::mutex mutex_;
};

// ============================================================================
// Circuit Breaker
// ============================================================================

/**
 * Circuit breaker to prevent repeated calls to failing services
 */
class CircuitBreaker {
public:
    enum class State { Closed, Open, HalfOpen };

    explicit CircuitBreaker(int failure_threshold = 5,
                          std::chrono::seconds reset_timeout = std::chrono::seconds(60))
        : failure_threshold_{failure_threshold},
          reset_timeout_{reset_timeout},
          state_{State::Closed},
          consecutive_failures_{0} {}

    [[nodiscard]] auto isOpen() const -> bool {
        std::lock_guard<std::mutex> lock(mutex_);

        if (state_ == State::Open) {
            // Check if we should transition to half-open
            auto now = std::chrono::steady_clock::now();
            if (now - last_failure_time_ >= reset_timeout_) {
                state_ = State::HalfOpen;
                Logger::getInstance().info("Circuit breaker transitioning to half-open state");
                return false;
            }
            return true;
        }

        return false;
    }

    auto recordSuccess() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        consecutive_failures_ = 0;

        if (state_ == State::HalfOpen) {
            state_ = State::Closed;
            Logger::getInstance().info("Circuit breaker closed after successful recovery");
        }
    }

    auto recordFailure() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        consecutive_failures_++;
        last_failure_time_ = std::chrono::steady_clock::now();

        if (consecutive_failures_ >= failure_threshold_ && state_ != State::Open) {
            state_ = State::Open;
            Logger::getInstance().error(
                "Circuit breaker opened after {} consecutive failures (will retry in {}s)",
                consecutive_failures_, reset_timeout_.count());
        }
    }

    [[nodiscard]] auto getState() const -> State {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

private:
    int failure_threshold_;
    std::chrono::seconds reset_timeout_;
    mutable State state_;
    int consecutive_failures_;
    std::chrono::steady_clock::time_point last_failure_time_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Retry Engine
// ============================================================================

/**
 * Main retry engine with comprehensive error handling
 */
class RetryEngine {
public:
    RetryEngine() = default;

    /**
     * Execute operation with automatic retry on transient failures
     *
     * @param operation Function to execute (must return Result<T>)
     * @param operation_name Name for logging and metrics
     * @param policy Retry policy configuration
     * @return Result from operation (success or final error)
     */
    template<typename T, typename Func>
    [[nodiscard]] auto execute(Func&& operation,
                              std::string const& operation_name,
                              RetryPolicy const& policy = RetryPolicy::defaultPolicy())
        -> Result<T> {

        auto& stats = getOrCreateStats(operation_name);
        auto& breaker = getOrCreateCircuitBreaker(operation_name);

        // Check circuit breaker
        if (breaker.isOpen()) {
            Logger::getInstance().error("{}: Circuit breaker is OPEN, failing fast", operation_name);
            return makeError<T>(ErrorCode::ServiceUnavailable,
                              "Service circuit breaker is open - too many recent failures");
        }

        BackoffCalculator backoff_calc;

        for (int attempt = 1; attempt <= policy.max_attempts; ++attempt) {
            stats.recordAttempt();

            Logger::getInstance().debug("{}: Attempt {}/{}",
                                      operation_name, attempt, policy.max_attempts);

            // Execute operation
            auto result = operation();

            // Success case
            if (result) {
                Logger::getInstance().debug("{}: Success on attempt {}", operation_name, attempt);
                stats.recordSuccess();
                breaker.recordSuccess();

                if (attempt > 1) {
                    Logger::getInstance().info("{}: Recovered after {} attempts",
                                             operation_name, attempt);
                }

                return result;
            }

            // Failure case - analyze error
            auto const& error = result.error();
            auto category = ErrorCategorizer::categorize(error);

            // Also check error message for additional context
            auto message_category = ErrorCategorizer::analyzeMessage(error.message);
            if (message_category == ErrorCategory::Transient) {
                category = ErrorCategory::Transient;
            }

            Logger::getInstance().warn(
                "{}: Attempt {}/{} failed: {} (category: {})",
                operation_name, attempt, policy.max_attempts, error.message,
                category == ErrorCategory::Transient ? "transient" :
                category == ErrorCategory::Permanent ? "permanent" : "unknown");

            // Permanent errors - don't retry
            if (category == ErrorCategory::Permanent) {
                Logger::getInstance().error(
                    "{}: Permanent error detected, not retrying: {}",
                    operation_name, error.message);
                stats.recordFailure();
                breaker.recordFailure();
                return result;
            }

            // Last attempt - return failure
            if (attempt >= policy.max_attempts) {
                Logger::getInstance().error(
                    "{}: Max retry attempts ({}) exceeded, final error: {}",
                    operation_name, policy.max_attempts, error.message);
                stats.recordFailure();
                breaker.recordFailure();
                return result;
            }

            // Calculate backoff delay
            auto delay = backoff_calc.calculateDelay(attempt, policy);
            stats.recordRetry();
            stats.addDelay(delay);

            Logger::getInstance().info(
                "{}: Retrying in {} ms (attempt {}/{}, error: {})",
                operation_name, delay.count(), attempt + 1, policy.max_attempts,
                error.message);

            // Wait before retry
            std::this_thread::sleep_for(delay);
        }

        // Should never reach here, but handle gracefully
        stats.recordFailure();
        breaker.recordFailure();
        return makeError<T>(ErrorCode::Unknown, "Retry loop exited unexpectedly");
    }

    /**
     * Get statistics for an operation
     */
    [[nodiscard]] auto getStats(std::string const& operation_name) const
        -> std::optional<RetryStats> {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(operation_name);
        if (it != stats_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * Get all statistics
     */
    [[nodiscard]] auto getAllStats() const -> std::vector<RetryStats> {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<RetryStats> all_stats;
        all_stats.reserve(stats_.size());
        for (auto const& [name, stats] : stats_) {
            all_stats.push_back(stats);
        }
        return all_stats;
    }

    /**
     * Reset all statistics
     */
    auto resetStats() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
    }

    /**
     * Get circuit breaker state
     */
    [[nodiscard]] auto getCircuitBreakerState(std::string const& operation_name) const
        -> std::optional<CircuitBreaker::State> {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = circuit_breakers_.find(operation_name);
        if (it != circuit_breakers_.end()) {
            return it->second.getState();
        }
        return std::nullopt;
    }

private:
    auto getOrCreateStats(std::string const& operation_name) -> RetryStats& {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(operation_name);
        if (it == stats_.end()) {
            RetryStats new_stats;
            new_stats.operation_name = operation_name;
            it = stats_.emplace(operation_name, std::move(new_stats)).first;
        }
        return it->second;
    }

    auto getOrCreateCircuitBreaker(std::string const& operation_name) -> CircuitBreaker& {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = circuit_breakers_.find(operation_name);
        if (it == circuit_breakers_.end()) {
            it = circuit_breakers_.emplace(operation_name, CircuitBreaker{}).first;
        }
        return it->second;
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, RetryStats> stats_;
    std::unordered_map<std::string, CircuitBreaker> circuit_breakers_;
};

// ============================================================================
// Global Retry Engine Instance
// ============================================================================

/**
 * Get global retry engine instance
 */
[[nodiscard]] inline auto getRetryEngine() -> RetryEngine& {
    static RetryEngine engine;
    return engine;
}

// ============================================================================
// Convenience Retry Functions
// ============================================================================

/**
 * Retry wrapper for network operations (Schwab API)
 */
template<typename T, typename Func>
[[nodiscard]] inline auto retryNetworkOperation(Func&& operation,
                                               std::string const& operation_name)
    -> Result<T> {
    return getRetryEngine().execute<T>(
        std::forward<Func>(operation),
        operation_name,
        RetryPolicy::aggressivePolicy()
    );
}

/**
 * Retry wrapper for database operations
 */
template<typename T, typename Func>
[[nodiscard]] inline auto retryDatabaseOperation(Func&& operation,
                                                std::string const& operation_name)
    -> Result<T> {
    return getRetryEngine().execute<T>(
        std::forward<Func>(operation),
        operation_name,
        RetryPolicy::conservativePolicy()
    );
}

/**
 * Retry wrapper with custom policy
 */
template<typename T, typename Func>
[[nodiscard]] inline auto retryWithPolicy(Func&& operation,
                                         std::string const& operation_name,
                                         RetryPolicy const& policy)
    -> Result<T> {
    return getRetryEngine().execute<T>(
        std::forward<Func>(operation),
        operation_name,
        policy
    );
}

} // namespace bigbrother::utils
