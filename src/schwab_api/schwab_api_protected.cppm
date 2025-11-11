/**
 * BigBrotherAnalytics - Schwab API with Circuit Breaker Protection (C++23)
 *
 * Wraps Schwab API calls with circuit breaker protection to prevent
 * cascading failures. Provides graceful degradation with cached data
 * when circuit is open.
 *
 * Following C++ Core Guidelines:
 * - R.1: RAII for resource management
 * - C.21: Rule of Five
 * - Thread-safe implementation
 */

// Global module fragment
module;

#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.schwab_api.protected;

// Import dependencies
import bigbrother.schwab_api;
import bigbrother.circuit_breaker;
import bigbrother.utils.logger;

export namespace bigbrother::schwab::protected_api {

using namespace bigbrother::schwab;
using namespace bigbrother::circuit_breaker;
using namespace bigbrother::utils;

/**
 * Protected Schwab API Client with Circuit Breaker
 *
 * Wraps all Schwab API calls with circuit breaker protection.
 * Provides fallback to cached data when circuit is open.
 */
class ProtectedSchwabClient {
  public:
    explicit ProtectedSchwabClient(OAuth2Config config)
        : client_{std::make_unique<SchwabClient>(std::move(config))} {

        // Initialize circuit breakers for different API endpoints
        initializeCircuitBreakers();

        Logger::getInstance().info("Protected Schwab API client initialized with circuit breakers");
    }

    // C.21: Rule of Five - non-copyable due to unique_ptr
    ProtectedSchwabClient(ProtectedSchwabClient const&) = delete;
    auto operator=(ProtectedSchwabClient const&) -> ProtectedSchwabClient& = delete;
    ProtectedSchwabClient(ProtectedSchwabClient&&) noexcept = default;
    auto operator=(ProtectedSchwabClient&&) noexcept -> ProtectedSchwabClient& = default;
    ~ProtectedSchwabClient() = default;

    /**
     * Get Quote with Circuit Breaker Protection
     *
     * Falls back to cached quote if circuit is open.
     */
    [[nodiscard]] auto getQuote(std::string const& symbol) -> Result<Quote> {
        auto* breaker = getBreaker("market_data");

        return breaker->call<Quote>([&]() -> std::expected<Quote, std::string> {
            auto result = client_->marketData().getQuote(symbol);

            if (result) {
                // Cache successful result
                cacheQuote(symbol, *result);
            }

            return result;
        });
    }

    /**
     * Get Quote with Fallback
     *
     * Returns cached quote if circuit is open or call fails.
     */
    [[nodiscard]] auto getQuoteWithFallback(std::string const& symbol) -> Result<Quote> {
        auto result = getQuote(symbol);

        if (!result) {
            // Try to get cached data
            auto cached = getCachedQuote(symbol);
            if (cached) {
                Logger::getInstance().warn("Using cached quote for {} (circuit breaker active)",
                                           symbol);
                return *cached;
            }

            // No cached data available
            return std::unexpected(result.error());
        }

        return result;
    }

    /**
     * Get Quotes with Circuit Breaker Protection
     */
    [[nodiscard]] auto getQuotes(std::vector<std::string> const& symbols)
        -> Result<std::vector<Quote>> {

        auto* breaker = getBreaker("market_data");

        return breaker->call<std::vector<Quote>>([&]() {
            auto result = client_->marketData().getQuotes(symbols);

            if (result) {
                // Cache all successful results
                for (auto const& quote : *result) {
                    cacheQuote(quote.symbol, quote);
                }
            }

            return result;
        });
    }

    /**
     * Get Option Chain with Circuit Breaker Protection
     */
    [[nodiscard]] auto getOptionChain(OptionsChainRequest const& request)
        -> Result<OptionsChainData> {

        auto* breaker = getBreaker("market_data");

        return breaker->call<OptionsChainData>([&]() {
            auto result = client_->marketData().getOptionChain(request);

            if (result) {
                // Cache successful result
                cacheOptionChain(request.symbol, *result);
            }

            return result;
        });
    }

    /**
     * Get Option Chain with Fallback
     */
    [[nodiscard]] auto getOptionChainWithFallback(OptionsChainRequest const& request)
        -> Result<OptionsChainData> {

        auto result = getOptionChain(request);

        if (!result) {
            // Try to get cached data
            auto cached = getCachedOptionChain(request.symbol);
            if (cached) {
                Logger::getInstance().warn(
                    "Using cached option chain for {} (circuit breaker active)", request.symbol);
                return *cached;
            }

            return std::unexpected(result.error());
        }

        return result;
    }

    /**
     * Place Order with Circuit Breaker Protection
     *
     * IMPORTANT: Orders should NOT use cached data fallback.
     * If circuit is open, order placement should fail immediately.
     */
    [[nodiscard]] auto placeOrder(Order order) -> Result<std::string> {
        auto* breaker = getBreaker("orders");

        return breaker->call<std::string>([&]() -> Result<std::string> { return client_->orders().placeOrder(order); });
    }

    /**
     * Cancel Order with Circuit Breaker Protection
     */
    [[nodiscard]] auto cancelOrder(std::string const& order_id) -> Result<void> {
        auto* breaker = getBreaker("orders");

        return breaker->call<void>([&]() -> Result<void> { return client_->orders().cancelOrder(order_id); });
    }

    /**
     * Get Account Balance with Circuit Breaker Protection
     */
    [[nodiscard]] auto getBalance() -> Result<AccountBalance> {
        auto* breaker = getBreaker("account");

        return breaker->call<AccountBalance>([&]() { return client_->account().getBalance(); });
    }

    /**
     * Get Account Balance with Fallback
     */
    [[nodiscard]] auto getBalanceWithFallback() -> Result<AccountBalance> {
        auto result = getBalance();

        if (!result) {
            // Try to get cached balance
            auto cached = getCachedBalance();
            if (cached) {
                Logger::getInstance().warn("Using cached account balance (circuit breaker active)");
                return *cached;
            }

            return std::unexpected(result.error());
        }

        // Cache successful result
        cacheBalance(*result);
        return result;
    }

    /**
     * Get Positions with Circuit Breaker Protection
     */
    [[nodiscard]] auto getPositions() -> Result<std::vector<AccountPosition>> {
        auto* breaker = getBreaker("account");

        return breaker->call<std::vector<AccountPosition>>(
            [&]() { return client_->account().getPositions(); });
    }

    /**
     * Get Positions with Fallback
     */
    [[nodiscard]] auto getPositionsWithFallback() -> Result<std::vector<AccountPosition>> {
        auto result = getPositions();

        if (!result) {
            // Try to get cached positions
            auto cached = getCachedPositions();
            if (cached) {
                Logger::getInstance().warn("Using cached positions (circuit breaker active)");
                return *cached;
            }

            return std::unexpected(result.error());
        }

        // Cache successful result
        cachePositions(*result);
        return result;
    }

    /**
     * Get Historical Data with Circuit Breaker Protection
     */
    [[nodiscard]] auto getHistoricalData(std::string const& symbol,
                                         std::string const& period_type = "day",
                                         std::string const& frequency_type = "minute",
                                         int frequency = 1) -> Result<HistoricalData> {

        auto* breaker = getBreaker("market_data");

        return breaker->call<HistoricalData>([&]() {
            return client_->marketData().getHistoricalData(symbol, period_type, frequency_type,
                                                           frequency);
        });
    }

    /**
     * Get Circuit Breaker Statistics
     */
    [[nodiscard]] auto getCircuitBreakerStats() const
        -> std::vector<std::pair<std::string, CircuitStats>> {
        return breaker_manager_.getAllStats();
    }

    /**
     * Check if any circuit is open
     */
    [[nodiscard]] auto hasOpenCircuit() const -> bool {
        return breaker_manager_.getOpenCount() > 0;
    }

    /**
     * Manually reset all circuit breakers
     * Use with caution - typically for manual intervention
     */
    auto resetAllCircuits() -> void { breaker_manager_.resetAll(); }

    /**
     * Reset specific circuit breaker
     */
    auto resetCircuit(std::string const& name) -> void {
        auto* breaker = getBreaker(name);
        if (breaker) {
            breaker->reset();
            Logger::getInstance().info("Reset circuit breaker: {}", name);
        }
    }

    /**
     * Access underlying client (for non-protected operations)
     */
    [[nodiscard]] auto client() -> SchwabClient& { return *client_; }

  private:
    /**
     * Initialize circuit breakers for different API endpoints
     */
    auto initializeCircuitBreakers() -> void {
        // Market Data circuit breaker (higher threshold, market data is less critical)
        breaker_manager_.registerBreaker("market_data", CircuitConfig{
                                                            .failure_threshold = 5,
                                                            .timeout = std::chrono::seconds(60),
                                                            .half_open_timeout = std::chrono::seconds(30),
                                                            .half_open_max_calls = 3,
                                                            .enable_logging = true,
                                                            .name = "market_data",
                                                        });

        // Orders circuit breaker (lower threshold, orders are critical)
        breaker_manager_.registerBreaker("orders", CircuitConfig{
                                                       .failure_threshold = 3,
                                                       .timeout = std::chrono::seconds(120),
                                                       .half_open_timeout = std::chrono::seconds(60),
                                                       .half_open_max_calls = 1,
                                                       .enable_logging = true,
                                                       .name = "orders",
                                                   });

        // Account circuit breaker (medium threshold)
        breaker_manager_.registerBreaker("account", CircuitConfig{
                                                        .failure_threshold = 4,
                                                        .timeout = std::chrono::seconds(90),
                                                        .half_open_timeout = std::chrono::seconds(45),
                                                        .half_open_max_calls = 2,
                                                        .enable_logging = true,
                                                        .name = "account",
                                                    });
    }

    /**
     * Get circuit breaker by name
     */
    [[nodiscard]] auto getBreaker(std::string const& name) const -> CircuitBreaker* {
        return const_cast<CircuitBreakerManager&>(breaker_manager_).getBreaker(name);
    }

    // ========================================================================
    // Cache Management
    // ========================================================================

    auto cacheQuote(std::string const& symbol, Quote const& quote) -> void {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        quote_cache_[symbol] = quote;
    }

    [[nodiscard]] auto getCachedQuote(std::string const& symbol) const -> std::optional<Quote> {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = quote_cache_.find(symbol);
        if (it != quote_cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    auto cacheOptionChain(std::string const& symbol, OptionsChainData const& chain) -> void {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        chain_cache_[symbol] = chain;
    }

    [[nodiscard]] auto getCachedOptionChain(std::string const& symbol) const
        -> std::optional<OptionsChainData> {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = chain_cache_.find(symbol);
        if (it != chain_cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    auto cacheBalance(AccountBalance const& balance) -> void {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        balance_cache_ = balance;
    }

    [[nodiscard]] auto getCachedBalance() const -> std::optional<AccountBalance> {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (balance_cache_.total_value > 0) {
            return balance_cache_;
        }
        return std::nullopt;
    }

    auto cachePositions(std::vector<AccountPosition> const& positions) -> void {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        positions_cache_ = positions;
    }

    [[nodiscard]] auto getCachedPositions() const -> std::optional<std::vector<AccountPosition>> {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (!positions_cache_.empty()) {
            return positions_cache_;
        }
        return std::nullopt;
    }

    // ========================================================================
    // Member Variables
    // ========================================================================

    std::unique_ptr<SchwabClient> client_;
    mutable CircuitBreakerManager breaker_manager_;

    // Cache for fallback data
    std::unordered_map<std::string, Quote> quote_cache_;
    std::unordered_map<std::string, OptionsChainData> chain_cache_;
    AccountBalance balance_cache_;
    std::vector<AccountPosition> positions_cache_;

    mutable std::mutex cache_mutex_;
};

} // namespace bigbrother::schwab::protected_api
