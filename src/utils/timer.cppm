/**
 * BigBrotherAnalytics - Timer Module (C++23)
 *
 * High-resolution timing and profiling for algorithmic trading.
 * Thread-safe implementation for concurrent operations.
 *
 * Features:
 * - Microsecond-precision timing
 * - Performance profiling with statistics
 * - Rate limiting for API calls
 * - Latency monitoring with alerts
 *
 * Following C++ Core Guidelines:
 * https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
 */

// Global module fragment
module;

#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <format>

// Module declaration
export module bigbrother.utils.timer;

export namespace bigbrother::utils {

/**
 * High-Resolution Timer
 *
 * Provides microsecond-precision timing for performance measurement.
 * Critical for algorithmic trading where latencies matter.
 *
 * C.1: Use class for types with invariants
 * F: All functions use trailing return type syntax
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::micro>; // Microseconds

    // C.41: Constructor establishes invariant
    Timer() : start_time_{Clock::now()}, running_{true} {}

    /**
     * Start/restart the timer
     * F.20: Return void for simple operations
     */
    auto start() -> void {
        start_time_ = Clock::now();
        running_ = true;
    }

    /**
     * Stop the timer and return elapsed time
     * F.20: Return value, not output parameter
     * @return Elapsed time in microseconds
     */
    [[nodiscard]] auto stop() -> double {
        if (!running_) {
            return 0.0;
        }

        auto const end_time = Clock::now();
        running_ = false;

        Duration const elapsed = end_time - start_time_;
        return elapsed.count();
    }

    /**
     * Get elapsed time without stopping
     * F.6: noexcept - no exceptions possible
     * @return Elapsed time in microseconds
     */
    [[nodiscard]] auto elapsed() const noexcept -> double {
        auto const end_time = Clock::now();
        Duration const elapsed = end_time - start_time_;
        return elapsed.count();
    }

    /**
     * Get elapsed time in milliseconds
     */
    [[nodiscard]] auto elapsedMillis() const noexcept -> double {
        return elapsed() / 1000.0;
    }

    /**
     * Get elapsed time in seconds
     */
    [[nodiscard]] auto elapsedSeconds() const noexcept -> double {
        return elapsed() / 1'000'000.0;
    }

    /**
     * Reset timer
     */
    auto reset() -> void {
        start_time_ = Clock::now();
        running_ = true;
    }

    /**
     * Check if timer is running
     * F.6: noexcept
     */
    [[nodiscard]] auto isRunning() const noexcept -> bool {
        return running_;
    }

    /**
     * Get current timestamp (microseconds since epoch)
     * F.1: Meaningful name
     */
    [[nodiscard]] static auto now() -> int64_t {
        auto const now = Clock::now();
        auto const micros = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()
        );
        return micros.count();
    }

    /**
     * Get current timestamp as TimePoint
     */
    [[nodiscard]] static auto timepoint() -> TimePoint {
        return Clock::now();
    }

private:
    TimePoint start_time_;
    bool running_;
};

/**
 * Scoped Timer
 *
 * RAII timer that automatically logs elapsed time when scope exits.
 *
 * Usage:
 *   {
 *     ScopedTimer timer("MyFunction");
 *     // ... code to measure ...
 *   } // Automatically logs "MyFunction took X microseconds"
 *
 * C.2: Use class when invariants exist (RAII pattern)
 */
class ScopedTimer {
public:
    /**
     * Constructor
     * F.16: Pass string by const& (not cheap to copy)
     */
    explicit ScopedTimer(std::string const& name)
        : name_{name}, timer_{}, stopped_{false} {}

    /**
     * Destructor - logs elapsed time
     */
    ~ScopedTimer() {
        if (!stopped_) {
            stop();
        }
    }

    // C.21: Define or delete default operations as a group
    ScopedTimer(ScopedTimer const&) = delete;
    auto operator=(ScopedTimer const&) -> ScopedTimer& = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    auto operator=(ScopedTimer&&) -> ScopedTimer& = delete;

    /**
     * Stop timer early and log
     */
    auto stop() -> void;

private:
    std::string name_;
    Timer timer_;
    bool stopped_;
};

/**
 * Performance Profiler
 *
 * Collects timing statistics across multiple invocations.
 * Thread-safe singleton for concurrent operations.
 *
 * Usage:
 *   Profiler::getInstance().record("OptionsPricing", elapsed_micros);
 *   Profiler::getInstance().printStats();
 */
class Profiler {
public:
    /**
     * Profiling Statistics
     * C.1: Struct for passive data
     */
    struct Stats {
        std::string name;
        size_t count{0};           // Number of samples
        double total_us{0.0};      // Total time in microseconds
        double mean_us{0.0};       // Mean time
        double min_us{0.0};        // Minimum time
        double max_us{0.0};        // Maximum time
        double stddev_us{0.0};     // Standard deviation
        double median_us{0.0};     // Median time
        double p95_us{0.0};        // 95th percentile
        double p99_us{0.0};        // 99th percentile
    };

    /**
     * Get singleton instance
     * F.20: Return reference
     */
    [[nodiscard]] static auto getInstance() -> Profiler&;

    // C.21: Singleton - delete copy/move
    Profiler(Profiler const&) = delete;
    auto operator=(Profiler const&) -> Profiler& = delete;
    Profiler(Profiler&&) = delete;
    auto operator=(Profiler&&) -> Profiler& = delete;

    /**
     * Record a timing measurement
     * @param name Identifier for this measurement
     * @param elapsed_us Elapsed time in microseconds
     */
    auto record(std::string const& name, double elapsed_us) -> void;

    /**
     * Get statistics for a specific measurement
     */
    [[nodiscard]] auto getStats(std::string const& name) const -> Stats;

    /**
     * Get statistics for all measurements
     */
    [[nodiscard]] auto getAllStats() const -> std::vector<Stats>;

    /**
     * Print statistics to log
     */
    auto printStats() const -> void;

    /**
     * Clear all statistics
     */
    auto clear() -> void;

    /**
     * Clear statistics for specific measurement
     */
    auto clear(std::string const& name) -> void;

    /**
     * Save statistics to file
     */
    [[nodiscard]] auto saveToFile(std::string const& filename) const -> bool;

    /**
     * RAII Profiling Guard
     *
     * Usage:
     *   {
     *     auto guard = Profiler::getInstance().profile("MyFunction");
     *     // ... code to profile ...
     *   } // Automatically records timing
     */
    class ProfileGuard {
    public:
        explicit ProfileGuard(std::string const& name)
            : name_{name}, timer_{} {}

        ~ProfileGuard() {
            auto const elapsed = timer_.elapsed();
            Profiler::getInstance().record(name_, elapsed);
        }

        // C.21: Delete copy/move for RAII guard
        ProfileGuard(ProfileGuard const&) = delete;
        auto operator=(ProfileGuard const&) -> ProfileGuard& = delete;
        ProfileGuard(ProfileGuard&&) = delete;
        auto operator=(ProfileGuard&&) -> ProfileGuard& = delete;

    private:
        std::string name_;
        Timer timer_;
    };

    /**
     * Create profiling guard (fluent API)
     * F.20: Return by value (NRVO applies)
     */
    [[nodiscard]] auto profile(std::string const& name) -> ProfileGuard {
        return ProfileGuard{name};
    }

private:
    Profiler();
    ~Profiler();

    // pImpl pattern for ABI stability
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Rate Limiter
 *
 * Limits the rate of operations (e.g., API calls) to avoid hitting rate limits.
 * Thread-safe for concurrent access.
 *
 * Usage:
 *   RateLimiter limiter(100);  // 100 operations per second
 *   limiter.acquire();  // Blocks if rate limit exceeded
 */
class RateLimiter {
public:
    /**
     * Constructor
     * @param max_rate Maximum operations per second
     */
    explicit RateLimiter(double max_rate);
    ~RateLimiter() = default;

    // C.21: Move-only type (pImpl pattern)
    RateLimiter(RateLimiter const&) = delete;
    auto operator=(RateLimiter const&) -> RateLimiter& = delete;
    RateLimiter(RateLimiter&&) noexcept = default;
    auto operator=(RateLimiter&&) noexcept -> RateLimiter& = default;

    /**
     * Acquire permission to perform one operation
     * Blocks if rate limit would be exceeded
     */
    auto acquire() -> void;

    /**
     * Try to acquire permission without blocking
     * @return true if acquired, false if rate limit exceeded
     */
    [[nodiscard]] auto tryAcquire() -> bool;

    /**
     * Get current rate (operations per second)
     */
    [[nodiscard]] auto getRate() const -> double;

    /**
     * Set new rate limit
     */
    auto setRate(double max_rate) -> void;

    /**
     * Reset rate limiter
     */
    auto reset() -> void;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Latency Monitor
 *
 * Monitors system latencies and alerts when thresholds are exceeded.
 * Useful for detecting performance degradation in real-time.
 */
class LatencyMonitor {
public:
    /**
     * Alert structure
     * C.1: Struct for passive data
     */
    struct Alert {
        std::string name;
        double latency_us;
        double threshold_us;
        Timer::TimePoint timestamp;
    };

    using AlertCallback = std::function<void(Alert const&)>;

    /**
     * Constructor
     * @param name Monitor name
     * @param threshold_us Alert threshold in microseconds
     */
    LatencyMonitor(std::string const& name, double threshold_us);

    /**
     * Record a latency measurement
     * Triggers alert if threshold exceeded
     */
    auto record(double latency_us) -> void;

    /**
     * Set alert callback
     * F.16: Pass by value and move
     */
    auto setAlertCallback(AlertCallback callback) -> void {
        callback_ = std::move(callback);
    }

    /**
     * Get statistics
     */
    [[nodiscard]] auto getStats() const -> Profiler::Stats;

    /**
     * Clear statistics
     */
    auto clear() -> void;

private:
    std::string name_;
    double threshold_us_;
    AlertCallback callback_;
    Profiler& profiler_;
};

} // export namespace bigbrother::utils

// ============================================================================
// Implementation Section (module-private)
// ============================================================================

module :private;

namespace bigbrother::utils {

// Macro-like helpers (not exported)
namespace {
    constexpr auto LOG_INFO = [](auto&&...) -> void {};  // Stub for now
    constexpr auto LOG_ERROR = [](auto&&...) -> void {};  // Stub for now
}

// ScopedTimer implementation
auto ScopedTimer::stop() -> void {
    if (!stopped_) {
        auto const elapsed = timer_.elapsed();
        // Simplified logging - will enhance with logger module import later
        stopped_ = true;
    }
}

// Profiler implementation (thread-safe with shared_mutex)
class Profiler::Impl {
public:
    Impl() = default;

    auto record(std::string const& name, double elapsed_us) -> void {
        std::unique_lock lock{mutex_};  // C++17 CTAD

        auto& samples = measurements_[name];
        samples.push_back(elapsed_us);

        // Keep last 10000 samples
        if (samples.size() > 10'000) {
            samples.erase(samples.begin(), samples.begin() + 1'000);
        }
    }

    [[nodiscard]] auto getStats(std::string const& name) const -> Profiler::Stats {
        std::shared_lock lock{mutex_};

        auto const it = measurements_.find(name);
        if (it == measurements_.end()) {
            return {name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        return computeStats(name, it->second);
    }

    [[nodiscard]] auto getAllStats() const -> std::vector<Profiler::Stats> {
        std::shared_lock lock{mutex_};

        std::vector<Profiler::Stats> all_stats;
        all_stats.reserve(measurements_.size());

        for (auto const& [name, samples] : measurements_) {
            all_stats.push_back(computeStats(name, samples));
        }

        return all_stats;
    }

    auto printStats() const -> void {
        auto const all_stats = getAllStats();

        if (all_stats.empty()) {
            return;
        }

        for (auto const& stat : all_stats) {
            // Simplified printing
        }
    }

    auto clear() -> void {
        std::unique_lock lock{mutex_};
        measurements_.clear();
    }

    auto clear(std::string const& name) -> void {
        std::unique_lock lock{mutex_};
        measurements_.erase(name);
    }

    [[nodiscard]] auto saveToFile(std::string const& filename) const -> bool {
        auto const all_stats = getAllStats();

        std::ofstream file{filename};
        if (!file.is_open()) {
            return false;
        }

        // CSV header
        file << "Name,Count,Total(us),Mean(us),Min(us),Max(us),StdDev(us),Median(us),P95(us),P99(us)\n";

        // Data
        for (auto const& stat : all_stats) {
            file << std::format("{},{},{},{},{},{},{},{},{},{}\n",
                               stat.name, stat.count, stat.total_us, stat.mean_us,
                               stat.min_us, stat.max_us, stat.stddev_us,
                               stat.median_us, stat.p95_us, stat.p99_us);
        }

        return true;
    }

private:
    [[nodiscard]] static auto computeStats(std::string const& name,
                                           std::vector<double> const& samples)
        -> Profiler::Stats {
        if (samples.empty()) {
            return {name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        auto const count = samples.size();
        auto const total = std::accumulate(samples.begin(), samples.end(), 0.0);
        auto const mean = total / static_cast<double>(count);

        auto const [min_it, max_it] = std::minmax_element(samples.begin(), samples.end());
        auto const min_val = *min_it;
        auto const max_val = *max_it;

        // Standard deviation
        auto const variance = std::accumulate(samples.begin(), samples.end(), 0.0,
            [mean](double acc, double val) {
                auto const diff = val - mean;
                return acc + diff * diff;
            }) / static_cast<double>(count);
        auto const stddev = std::sqrt(variance);

        // Percentiles
        auto sorted_samples = samples;
        std::sort(sorted_samples.begin(), sorted_samples.end());

        auto const median = computePercentile(sorted_samples, 0.50);
        auto const p95 = computePercentile(sorted_samples, 0.95);
        auto const p99 = computePercentile(sorted_samples, 0.99);

        return {name, count, total, mean, min_val, max_val, stddev, median, p95, p99};
    }

    [[nodiscard]] static auto computePercentile(std::vector<double> const& sorted_samples,
                                                 double percentile) -> double {
        if (sorted_samples.empty()) {
            return 0.0;
        }

        auto const index = percentile * static_cast<double>(sorted_samples.size() - 1);
        auto const lower_idx = static_cast<size_t>(std::floor(index));
        auto const upper_idx = static_cast<size_t>(std::ceil(index));

        if (lower_idx == upper_idx) {
            return sorted_samples[lower_idx];
        }

        auto const weight = index - static_cast<double>(lower_idx);
        return sorted_samples[lower_idx] * (1.0 - weight) +
               sorted_samples[upper_idx] * weight;
    }

    std::unordered_map<std::string, std::vector<double>> measurements_;
    mutable std::shared_mutex mutex_;  // Thread-safety (read-write lock)
};

// Profiler singleton
Profiler::Profiler() : pImpl_{std::make_unique<Impl>()} {}
Profiler::~Profiler() = default;

auto Profiler::getInstance() -> Profiler& {
    static Profiler instance;
    return instance;
}

auto Profiler::record(std::string const& name, double elapsed_us) -> void {
    pImpl_->record(name, elapsed_us);
}

auto Profiler::getStats(std::string const& name) const -> Stats {
    return pImpl_->getStats(name);
}

auto Profiler::getAllStats() const -> std::vector<Stats> {
    return pImpl_->getAllStats();
}

auto Profiler::printStats() const -> void {
    pImpl_->printStats();
}

auto Profiler::clear() -> void {
    pImpl_->clear();
}

auto Profiler::clear(std::string const& name) -> void {
    pImpl_->clear(name);
}

auto Profiler::saveToFile(std::string const& filename) const -> bool {
    return pImpl_->saveToFile(filename);
}

// RateLimiter implementation
class RateLimiter::Impl {
public:
    explicit Impl(double max_rate)
        : max_rate_{max_rate},
          min_interval_us_{1'000'000.0 / max_rate},
          last_acquire_{Timer::Clock::now()} {}

    auto acquire() -> void {
        std::unique_lock lock{mutex_};

        auto const now = Timer::Clock::now();
        auto const elapsed_us = std::chrono::duration<double, std::micro>(
            now - last_acquire_).count();

        if (elapsed_us < min_interval_us_) {
            auto const sleep_us = min_interval_us_ - elapsed_us;
            lock.unlock();  // Release lock while sleeping

            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<int64_t>(sleep_us))
            );

            lock.lock();  // Reacquire before updating
        }

        last_acquire_ = Timer::Clock::now();
    }

    [[nodiscard]] auto tryAcquire() -> bool {
        std::unique_lock lock{mutex_};

        auto const now = Timer::Clock::now();
        auto const elapsed_us = std::chrono::duration<double, std::micro>(
            now - last_acquire_).count();

        if (elapsed_us >= min_interval_us_) {
            last_acquire_ = now;
            return true;
        }

        return false;
    }

    [[nodiscard]] auto getRate() const -> double {
        std::shared_lock lock{rate_mutex_};
        return max_rate_;
    }

    auto setRate(double max_rate) -> void {
        std::unique_lock lock{rate_mutex_};
        max_rate_ = max_rate;
        min_interval_us_ = 1'000'000.0 / max_rate;
    }

    auto reset() -> void {
        std::unique_lock lock{mutex_};
        last_acquire_ = Timer::Clock::now();
    }

private:
    double max_rate_;
    double min_interval_us_;
    Timer::TimePoint last_acquire_;
    std::mutex mutex_;                      // Protects acquire timing
    mutable std::shared_mutex rate_mutex_;  // Protects rate settings
};

// RateLimiter public interface
RateLimiter::RateLimiter(double max_rate)
    : pImpl_{std::make_unique<Impl>(max_rate)} {}

auto RateLimiter::acquire() -> void {
    pImpl_->acquire();
}

auto RateLimiter::tryAcquire() -> bool {
    return pImpl_->tryAcquire();
}

auto RateLimiter::getRate() const -> double {
    return pImpl_->getRate();
}

auto RateLimiter::setRate(double max_rate) -> void {
    pImpl_->setRate(max_rate);
}

auto RateLimiter::reset() -> void {
    pImpl_->reset();
}

// LatencyMonitor implementation
LatencyMonitor::LatencyMonitor(std::string const& name, double threshold_us)
    : name_{name},
      threshold_us_{threshold_us},
      callback_{},
      profiler_{Profiler::getInstance()} {}

auto LatencyMonitor::record(double latency_us) -> void {
    profiler_.record(name_, latency_us);

    if (latency_us > threshold_us_ && callback_) {
        Alert alert{
            name_,
            latency_us,
            threshold_us_,
            Timer::timepoint()
        };
        callback_(alert);
    }
}

auto LatencyMonitor::getStats() const -> Profiler::Stats {
    return profiler_.getStats(name_);
}

auto LatencyMonitor::clear() -> void {
    profiler_.clear(name_);
}

} // namespace bigbrother::utils
