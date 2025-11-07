#include "timer.hpp"
#include "logger.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <format>
#include <thread>
#include <mutex>
#include <shared_mutex>

namespace bigbrother::utils {

/**
 * Thread-Safe Timer Implementation
 *
 * Now includes mutex protection for:
 * - Concurrent profiling from multiple strategies
 * - Parallel correlation calculations
 * - Multi-threaded market data processing
 */

// ScopedTimer implementation
ScopedTimer::ScopedTimer(std::string const& name)
    : name_{name}, timer_{}, stopped_{false} {}

ScopedTimer::~ScopedTimer() {
    if (!stopped_) {
        stop();
    }
}

auto ScopedTimer::stop() -> void {
    if (!stopped_) {
        auto const elapsed = timer_.elapsed();
        LOG_INFO("{} took {:.2f} μs ({:.4f} ms)", name_, elapsed, elapsed / 1000.0);
        stopped_ = true;
    }
}

// Profiler implementation (thread-safe with shared_mutex for read/write efficiency)
class Profiler::Impl {
public:
    Impl() = default;

    auto record(std::string const& name, double elapsed_us) -> void {
        std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive lock for write

        auto& samples = measurements_[name];
        samples.push_back(elapsed_us);

        // Keep last 10000 samples
        if (samples.size() > 10000) {
            samples.erase(samples.begin(), samples.begin() + 1000);
        }
    }

    [[nodiscard]] auto getStats(std::string const& name) const -> Profiler::Stats {
        std::shared_lock<std::shared_mutex> lock(mutex_);  // Shared lock for read

        auto it = measurements_.find(name);
        if (it == measurements_.end()) {
            return {name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        return computeStats(name, it->second);
    }

    [[nodiscard]] auto getAllStats() const -> std::vector<Profiler::Stats> {
        std::shared_lock<std::shared_mutex> lock(mutex_);  // Shared lock for read

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
            LOG_INFO("No profiling data collected");
            return;
        }

        LOG_INFO("Performance Profiling Statistics");
        LOG_INFO("═══════════════════════════════════════════════════════");

        for (auto const& stat : all_stats) {
            LOG_INFO("{:30} | Count: {:8} | Mean: {:10.2f}μs | P95: {:10.2f}μs",
                     stat.name, stat.count, stat.mean_us, stat.p95_us);
        }
    }

    auto clear() -> void {
        std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive lock
        measurements_.clear();
    }

    auto clear(std::string const& name) -> void {
        std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive lock
        measurements_.erase(name);
    }

    [[nodiscard]] auto saveToFile(std::string const& filename) const -> bool {
        auto const all_stats = getAllStats();  // Already thread-safe

        std::ofstream file{filename};
        if (!file.is_open()) {
            LOG_ERROR("Failed to open file: {}", filename);
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

        LOG_INFO("Profiler statistics saved to: {}", filename);
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

    std::map<std::string, std::vector<double>> measurements_;
    mutable std::shared_mutex mutex_;  // Thread-safety (read-write lock)
};

// Profiler singleton
Profiler::Profiler() : pImpl{std::make_unique<Impl>()} {}
Profiler::~Profiler() = default;

auto Profiler::getInstance() -> Profiler& {
    static Profiler instance;
    return instance;
}

auto Profiler::record(std::string const& name, double elapsed_us) -> void {
    pImpl->record(name, elapsed_us);
}

[[nodiscard]] auto Profiler::getStats(std::string const& name) const -> Stats {
    return pImpl->getStats(name);
}

[[nodiscard]] auto Profiler::getAllStats() const -> std::vector<Stats> {
    return pImpl->getAllStats();
}

auto Profiler::printStats() const -> void {
    pImpl->printStats();
}

auto Profiler::clear() -> void {
    pImpl->clear();
}

auto Profiler::clear(std::string const& name) -> void {
    pImpl->clear(name);
}

[[nodiscard]] auto Profiler::saveToFile(std::string const& filename) const -> bool {
    return pImpl->saveToFile(filename);
}

// ProfileGuard implementation
Profiler::ProfileGuard::ProfileGuard(std::string const& name)
    : name_{name}, timer_{} {}

Profiler::ProfileGuard::~ProfileGuard() {
    auto const elapsed = timer_.elapsed();
    Profiler::getInstance().record(name_, elapsed);
}

// RateLimiter implementation (thread-safe for concurrent API calls)
class RateLimiter::Impl {
public:
    explicit Impl(double max_rate)
        : max_rate_{max_rate},
          min_interval_us_{1'000'000.0 / max_rate},
          last_acquire_{Timer::Clock::now()} {}

    auto acquire() -> void {
        std::unique_lock<std::mutex> lock(mutex_);

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
        std::unique_lock<std::mutex> lock(mutex_);

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
        std::shared_lock<std::shared_mutex> lock(rate_mutex_);
        return max_rate_;
    }

    auto setRate(double max_rate) -> void {
        std::unique_lock<std::shared_mutex> lock(rate_mutex_);
        max_rate_ = max_rate;
        min_interval_us_ = 1'000'000.0 / max_rate;
    }

    auto reset() -> void {
        std::unique_lock<std::mutex> lock(mutex_);
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
    : pImpl{std::make_unique<Impl>(max_rate)} {}

auto RateLimiter::acquire() -> void {
    pImpl->acquire();
}

[[nodiscard]] auto RateLimiter::tryAcquire() -> bool {
    return pImpl->tryAcquire();
}

[[nodiscard]] auto RateLimiter::getRate() const -> double {
    return pImpl->getRate();
}

auto RateLimiter::setRate(double max_rate) -> void {
    pImpl->setRate(max_rate);
}

auto RateLimiter::reset() -> void {
    pImpl->reset();
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

auto LatencyMonitor::setAlertCallback(AlertCallback callback) -> void {
    callback_ = std::move(callback);
}

[[nodiscard]] auto LatencyMonitor::getStats() const -> Profiler::Stats {
    return profiler_.getStats(name_);
}

auto LatencyMonitor::clear() -> void {
    profiler_.clear(name_);
}

} // namespace bigbrother::utils
