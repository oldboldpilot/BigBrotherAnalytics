#pragma once

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <memory>

namespace bigbrother {
namespace utils {

/**
 * High-Resolution Timer
 *
 * Provides microsecond-precision timing for performance measurement.
 * Critical for algorithmic trading where latencies matter.
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::micro>; // Microseconds

    Timer() : start_time_(Clock::now()), running_(true) {}

    /**
     * Start/restart the timer
     */
    void start() {
        start_time_ = Clock::now();
        running_ = true;
    }

    /**
     * Stop the timer and return elapsed time
     * @return Elapsed time in microseconds
     */
    double stop() {
        if (!running_) {
            return 0.0;
        }

        auto end_time = Clock::now();
        running_ = false;

        Duration elapsed = end_time - start_time_;
        return elapsed.count();
    }

    /**
     * Get elapsed time without stopping
     * @return Elapsed time in microseconds
     */
    double elapsed() const {
        auto end_time = Clock::now();
        Duration elapsed = end_time - start_time_;
        return elapsed.count();
    }

    /**
     * Get elapsed time in milliseconds
     */
    double elapsedMillis() const {
        return elapsed() / 1000.0;
    }

    /**
     * Get elapsed time in seconds
     */
    double elapsedSeconds() const {
        return elapsed() / 1000000.0;
    }

    /**
     * Reset timer
     */
    void reset() {
        start_time_ = Clock::now();
        running_ = true;
    }

    /**
     * Check if timer is running
     */
    bool isRunning() const {
        return running_;
    }

    /**
     * Get current timestamp (microseconds since epoch)
     */
    static int64_t now() {
        auto now = Clock::now();
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()
        );
        return micros.count();
    }

    /**
     * Get current timestamp as TimePoint
     */
    static TimePoint timepoint() {
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
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

    // Delete copy/move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /**
     * Stop timer early and log
     */
    void stop();

private:
    std::string name_;
    Timer timer_;
    bool stopped_;
};

/**
 * Performance Profiler
 *
 * Collects timing statistics across multiple invocations.
 * Thread-safe.
 *
 * Usage:
 *   Profiler::getInstance().record("OptionsPricing", elapsed_micros);
 *   Profiler::getInstance().printStats();
 */
class Profiler {
public:
    struct Stats {
        std::string name;
        size_t count;           // Number of samples
        double total_us;        // Total time in microseconds
        double mean_us;         // Mean time
        double min_us;          // Minimum time
        double max_us;          // Maximum time
        double stddev_us;       // Standard deviation
        double median_us;       // Median time
        double p95_us;          // 95th percentile
        double p99_us;          // 99th percentile
    };

    static Profiler& getInstance();

    // Delete copy/move
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    /**
     * Record a timing measurement
     * @param name Identifier for this measurement
     * @param elapsed_us Elapsed time in microseconds
     */
    void record(const std::string& name, double elapsed_us);

    /**
     * Get statistics for a specific measurement
     */
    Stats getStats(const std::string& name) const;

    /**
     * Get statistics for all measurements
     */
    std::vector<Stats> getAllStats() const;

    /**
     * Print statistics to log
     */
    void printStats() const;

    /**
     * Clear all statistics
     */
    void clear();

    /**
     * Clear statistics for specific measurement
     */
    void clear(const std::string& name);

    /**
     * Save statistics to file
     */
    bool saveToFile(const std::string& filename) const;

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
        explicit ProfileGuard(const std::string& name);
        ~ProfileGuard();

        ProfileGuard(const ProfileGuard&) = delete;
        ProfileGuard& operator=(const ProfileGuard&) = delete;

    private:
        std::string name_;
        Timer timer_;
    };

    ProfileGuard profile(const std::string& name) {
        return ProfileGuard(name);
    }

private:
    Profiler();
    ~Profiler();

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Rate Limiter
 *
 * Limits the rate of operations (e.g., API calls) to avoid hitting rate limits.
 * Thread-safe.
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

    /**
     * Acquire permission to perform one operation
     * Blocks if rate limit would be exceeded
     */
    void acquire();

    /**
     * Try to acquire permission without blocking
     * @return true if acquired, false if rate limit exceeded
     */
    bool tryAcquire();

    /**
     * Get current rate (operations per second)
     */
    double getRate() const;

    /**
     * Set new rate limit
     */
    void setRate(double max_rate);

    /**
     * Reset rate limiter
     */
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Latency Monitor
 *
 * Monitors system latencies and alerts when thresholds are exceeded.
 * Useful for detecting performance degradation in real-time.
 */
class LatencyMonitor {
public:
    struct Alert {
        std::string name;
        double latency_us;
        double threshold_us;
        Timer::TimePoint timestamp;
    };

    using AlertCallback = std::function<void(const Alert&)>;

    /**
     * Constructor
     * @param name Monitor name
     * @param threshold_us Alert threshold in microseconds
     */
    LatencyMonitor(const std::string& name, double threshold_us);

    /**
     * Record a latency measurement
     * Triggers alert if threshold exceeded
     */
    void record(double latency_us);

    /**
     * Set alert callback
     */
    void setAlertCallback(AlertCallback callback);

    /**
     * Get statistics
     */
    Profiler::Stats getStats() const;

    /**
     * Clear statistics
     */
    void clear();

private:
    std::string name_;
    double threshold_us_;
    AlertCallback callback_;
    Profiler& profiler_;
};

// Convenience macros for profiling
#define PROFILE_SCOPE(name) \
    auto __profiler_guard_##__LINE__ = ::bigbrother::utils::Profiler::getInstance().profile(name)

#define PROFILE_FUNCTION() \
    PROFILE_SCOPE(__FUNCTION__)

} // namespace utils
} // namespace bigbrother
