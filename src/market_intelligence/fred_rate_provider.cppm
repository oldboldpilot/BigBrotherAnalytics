/**
 * BigBrotherAnalytics - FRED Rate Provider (Singleton)
 *
 * Thread-safe singleton that provides live risk-free rates from FRED API.
 * Used by options pricing engine, Greeks calculation, and trading engine.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase 5+: Live Risk-Free Rate Integration
 *
 * Usage:
 *   auto& provider = FREDRateProvider::getInstance();
 *   provider.initialize(api_key);
 *   double rf_rate = provider.getRiskFreeRate();  // For options pricing
 *   provider.startAutoRefresh(3600);  // Auto-refresh every hour
 */

module;

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

export module bigbrother.market_intelligence.fred_rate_provider;

import bigbrother.market_intelligence.fred_rates;
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Re-export std::map for visibility in exported interface
export {
    using std::map;
}

export namespace bigbrother::market_intelligence {

using namespace bigbrother::types;
using namespace bigbrother::utils;

/**
 * Thread-safe singleton for accessing FRED risk-free rates
 *
 * Provides automatic refresh, caching, and fallback mechanisms
 */
class FREDRateProvider {
  public:
    /**
     * Get singleton instance
     */
    [[nodiscard]] static auto getInstance() -> FREDRateProvider& {
        static FREDRateProvider instance;
        return instance;
    }

    // Delete copy/move operations (singleton)
    FREDRateProvider(FREDRateProvider const&) = delete;
    auto operator=(FREDRateProvider const&) -> FREDRateProvider& = delete;
    FREDRateProvider(FREDRateProvider&&) = delete;
    auto operator=(FREDRateProvider&&) -> FREDRateProvider& = delete;

    /**
     * Initialize provider with API key
     *
     * @param api_key FRED API key
     * @param default_series Default rate series (default: 3-month Treasury)
     * @return True if initialization successful
     */
    [[nodiscard]] auto initialize(
        std::string const& api_key,
        RateSeries default_series = RateSeries::ThreeMonthTreasury) -> bool {

        std::lock_guard<std::mutex> lock(mutex_);

        try {
            FREDConfig config;
            config.api_key = api_key;
            config.timeout_seconds = 10;
            config.max_observations = 5;

            fetcher_ = std::make_unique<FREDRatesFetcher>(std::move(config));
            default_series_ = default_series;
            initialized_ = true;

            Logger::getInstance().info("FRED Rate Provider initialized with series: {}",
                                      FREDRatesFetcher::getSeriesId(default_series));

            // Fetch initial rates (ignore result, logged internally)
            static_cast<void>(refreshRates());

            return true;

        } catch (std::exception const& e) {
            Logger::getInstance().error("Failed to initialize FRED Rate Provider: {}", e.what());
            initialized_ = false;
            return false;
        }
    }

    /**
     * Get risk-free rate for options pricing
     *
     * @param series Rate series to use (default: configured default)
     * @return Risk-free rate as decimal (e.g., 0.05 for 5%)
     *
     * Thread-safe. Returns cached value if fresh, fetches new if stale.
     * Falls back to 4% default if fetch fails.
     */
    [[nodiscard]] auto getRiskFreeRate(
        std::optional<RateSeries> series = std::nullopt) -> double {

        std::lock_guard<std::mutex> lock(mutex_);

        if (!initialized_) {
            Logger::getInstance().warn("FRED Rate Provider not initialized, using default 4%");
            return 0.04;
        }

        auto const target_series = series.value_or(default_series_);

        // Check if we have a fresh cached rate
        if (auto cached = getCachedRateLocked(target_series)) {
            return *cached;
        }

        // Cache miss or stale - fetch new rate
        try {
            auto result = fetcher_->fetchLatestRate(target_series);

            if (result) {
                rate_cache_[target_series] = result->rate_value;
                last_refresh_ = std::chrono::steady_clock::now();

                Logger::getInstance().info("Fetched fresh FRED rate: {} = {:.3f}%",
                                          result->series_id, result->rate_value * 100.0);

                return result->rate_value;
            }

        } catch (std::exception const& e) {
            Logger::getInstance().error("Failed to fetch FRED rate: {}", e.what());
        }

        // Fallback to default
        Logger::getInstance().warn("Using default risk-free rate: 4%");
        return 0.04;
    }

    /**
     * Get all available rates
     *
     * @return Map of series to rate values
     */
    [[nodiscard]] auto getAllRates() -> std::map<RateSeries, double> {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!initialized_) {
            Logger::getInstance().warn("FRED Rate Provider not initialized");
            return {};
        }

        // Refresh if stale (ignore result, logged internally)
        if (isCacheStale()) {
            static_cast<void>(refreshRatesLocked());
        }

        return rate_cache_;
    }

    /**
     * Manually refresh all rates
     *
     * @return True if refresh successful
     */
    [[nodiscard]] auto refreshRates() -> bool {
        std::lock_guard<std::mutex> lock(mutex_);
        return refreshRatesLocked();
    }

    /**
     * Start automatic background refresh
     *
     * @param interval_seconds Refresh interval in seconds (default: 3600 = 1 hour)
     */
    auto startAutoRefresh(int interval_seconds = 3600) -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        if (auto_refresh_running_) {
            Logger::getInstance().warn("Auto-refresh already running");
            return;
        }

        auto_refresh_running_ = true;
        refresh_interval_ = std::chrono::seconds(interval_seconds);

        // Start background thread
        refresh_thread_ = std::thread([this, interval_seconds]() {
            Logger::getInstance().info("FRED auto-refresh started (interval: {}s)", interval_seconds);

            while (auto_refresh_running_) {
                std::this_thread::sleep_for(refresh_interval_);

                if (auto_refresh_running_) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    static_cast<void>(refreshRatesLocked());  // Ignore result, logged internally
                }
            }

            Logger::getInstance().info("FRED auto-refresh stopped");
        });
    }

    /**
     * Stop automatic refresh
     */
    auto stopAutoRefresh() -> void {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto_refresh_running_ = false;
        }

        if (refresh_thread_.joinable()) {
            refresh_thread_.join();
        }
    }

    /**
     * Check if provider is initialized
     */
    [[nodiscard]] auto isInitialized() const noexcept -> bool {
        return initialized_;
    }

    /**
     * Get time since last refresh
     */
    [[nodiscard]] auto getTimeSinceRefresh() const -> std::chrono::seconds {
        std::lock_guard<std::mutex> lock(mutex_);
        auto const now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - last_refresh_);
    }

  private:
    FREDRateProvider() = default;

    ~FREDRateProvider() {
        stopAutoRefresh();
    }

    /**
     * Check if cache is stale (> 1 hour old)
     */
    [[nodiscard]] auto isCacheStale() const -> bool {
        auto const now = std::chrono::steady_clock::now();
        auto const age = std::chrono::duration_cast<std::chrono::seconds>(now - last_refresh_);
        return age > cache_ttl_;
    }

    /**
     * Get cached rate (locked version)
     */
    [[nodiscard]] auto getCachedRateLocked(RateSeries series) const -> std::optional<double> {
        if (isCacheStale()) {
            return std::nullopt;
        }

        if (auto it = rate_cache_.find(series); it != rate_cache_.end()) {
            return it->second;
        }

        return std::nullopt;
    }

    /**
     * Refresh all rates (locked version)
     */
    [[nodiscard]] auto refreshRatesLocked() -> bool {
        if (!initialized_ || !fetcher_) {
            return false;
        }

        try {
            auto rates = fetcher_->fetchAllRates();

            for (auto const& [series, rate_data] : rates) {
                rate_cache_[series] = rate_data.rate_value;
            }

            last_refresh_ = std::chrono::steady_clock::now();

            Logger::getInstance().info("FRED rates refreshed: {} series updated", rates.size());
            return true;

        } catch (std::exception const& e) {
            Logger::getInstance().error("Failed to refresh FRED rates: {}", e.what());
            return false;
        }
    }

    std::unique_ptr<FREDRatesFetcher> fetcher_;
    std::map<RateSeries, double> rate_cache_;
    RateSeries default_series_{RateSeries::ThreeMonthTreasury};

    std::chrono::steady_clock::time_point last_refresh_;
    std::chrono::seconds cache_ttl_{3600};  // 1 hour
    std::chrono::seconds refresh_interval_{3600};

    std::atomic<bool> initialized_{false};
    std::atomic<bool> auto_refresh_running_{false};
    std::thread refresh_thread_;

    mutable std::mutex mutex_;
};

}  // namespace bigbrother::market_intelligence
