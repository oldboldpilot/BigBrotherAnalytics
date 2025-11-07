#pragma once

#include "../utils/types.hpp"
#include "../correlation_engine/options_pricing.hpp"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <atomic>

namespace bigbrother::schwab {

using namespace types;

/**
 * Schwab API Client
 *
 * Complete C++23 client for Charles Schwab Trading API.
 * Implements OAuth 2.0 authentication with automatic token refresh.
 *
 * API Documentation:
 * https://developer.schwab.com/products/trader-api--individual
 *
 * Features:
 * - OAuth 2.0 authentication (automatic token refresh every 25 minutes)
 * - Market data (quotes, bars, options chains)
 * - Order placement and management
 * - Account information and positions
 * - Real-time WebSocket streaming
 * - Thread-safe operations
 * - Rate limiting (120 calls/minute per Schwab limits)
 *
 * Reference Implementation:
 * https://github.com/oldboldpilot/SchwabFirstAPI
 */

/**
 * OAuth 2.0 Configuration
 */
struct OAuth2Config {
    std::string client_id;          // App Key from Schwab
    std::string client_secret;      // App Secret from Schwab
    std::string redirect_uri;       // Callback URL (https://localhost:8080/callback)
    std::string auth_code;          // Authorization code (from initial OAuth flow)
    std::string refresh_token;      // Refresh token (persisted)
    std::string access_token;       // Access token (30-min lifetime)
    std::chrono::system_clock::time_point token_expiry;

    [[nodiscard]] auto isAccessTokenExpired() const noexcept -> bool {
        auto const now = std::chrono::system_clock::now();
        // Refresh 5 minutes before actual expiry for safety
        auto const safe_expiry = token_expiry - std::chrono::minutes(5);
        return now >= safe_expiry;
    }

    [[nodiscard]] auto validate() const noexcept -> Result<void>;
};

/**
 * Options Chain Request Parameters
 */
struct OptionsChainRequest {
    std::string symbol;                 // Underlying symbol
    std::string contract_type;          // "CALL", "PUT", or "ALL"
    std::string strategy;               // "SINGLE", "ANALYTICAL", "COVERED", "VERTICAL", "CALENDAR", etc.
    double strike_from{0.0};           // Min strike price
    double strike_to{0.0};             // Max strike price
    int days_to_expiration{0};         // Days to expiration filter
    std::string expiration_month;      // "ALL", "JAN", "FEB", etc.
    std::string option_type;           // "S" (Standard), "NS" (Non-standard)

    [[nodiscard]] static auto forSymbol(std::string symbol) -> OptionsChainRequest {
        return {
            .symbol = std::move(symbol),
            .contract_type = "ALL",
            .strategy = "SINGLE"
        };
    }
};

/**
 * Options Chain Response
 */
struct OptionsChainData {
    std::string symbol;
    std::string status;
    std::vector<options::OptionContract> calls;
    std::vector<options::OptionContract> puts;

    struct OptionQuote {
        options::OptionContract contract;
        Quote quote;
        options::Greeks greeks;
        double implied_volatility;
        int open_interest;
        int volume;
    };

    std::vector<OptionQuote> all_options;

    [[nodiscard]] auto findContract(
        options::OptionType type,
        Price strike,
        Timestamp expiration
    ) const noexcept -> std::optional<OptionQuote>;
};

/**
 * Order Request
 */
struct OrderRequest {
    std::string account_id;
    std::string symbol;
    Quantity quantity;
    OrderType order_type;
    OrderSide side;
    Price limit_price{0.0};         // For limit orders
    Price stop_price{0.0};          // For stop orders
    TimeInForce time_in_force{TimeInForce::Day};
    std::string instruction;        // "BUY", "SELL", "BUY_TO_OPEN", "SELL_TO_CLOSE", etc.
    std::string asset_type;         // "EQUITY", "OPTION"

    [[nodiscard]] static auto buyStock(
        std::string account_id,
        std::string symbol,
        Quantity quantity
    ) -> OrderRequest;

    [[nodiscard]] static auto sellStock(
        std::string account_id,
        std::string symbol,
        Quantity quantity
    ) -> OrderRequest;

    [[nodiscard]] static auto buyToOpen(
        std::string account_id,
        std::string option_symbol,
        int contracts
    ) -> OrderRequest;

    [[nodiscard]] auto toJson() const -> std::string;
};

/**
 * Account Information
 */
struct AccountInfo {
    std::string account_id;
    std::string type;               // "CASH", "MARGIN"
    double current_balance;
    double available_funds;
    double buying_power;
    double day_trading_buying_power;
    std::vector<Position> positions;

    [[nodiscard]] auto canTrade(double required_capital) const noexcept -> bool {
        return available_funds >= required_capital;
    }
};

/**
 * WebSocket Message Types
 */
enum class StreamMessageType {
    Quote,
    Trade,
    Book,           // Level 2 market depth
    Timesale,
    Chart,
    OptionQuote,
    Heartbeat,
    Error
};

/**
 * WebSocket Stream Handler
 */
class StreamHandler {
public:
    using QuoteCallback = std::function<void(Quote const&)>;
    using TradeCallback = std::function<void(Trade const&)>;
    using ErrorCallback = std::function<void(std::string const&)>;

    virtual ~StreamHandler() = default;

    virtual auto onQuote(Quote const& quote) -> void = 0;
    virtual auto onTrade(Trade const& trade) -> void = 0;
    virtual auto onError(std::string const& error) -> void = 0;
};

/**
 * Token Manager
 *
 * Manages OAuth 2.0 tokens with automatic refresh.
 * Thread-safe operations.
 */
class TokenManager {
public:
    explicit TokenManager(OAuth2Config config);
    ~TokenManager();

    // Delete copy, allow move
    TokenManager(TokenManager const&) = delete;
    auto operator=(TokenManager const&) = delete;
    TokenManager(TokenManager&&) noexcept;
    auto operator=(TokenManager&&) noexcept -> TokenManager&;

    /**
     * Get current access token (refreshes if needed)
     * Thread-safe
     */
    [[nodiscard]] auto getAccessToken() -> Result<std::string>;

    /**
     * Refresh access token using refresh token
     */
    [[nodiscard]] auto refreshAccessToken() -> Result<void>;

    /**
     * Initial OAuth flow (interactive, for first-time setup)
     */
    [[nodiscard]] static auto initialOAuthFlow(
        std::string const& client_id,
        std::string const& redirect_uri
    ) -> Result<OAuth2Config>;

    /**
     * Exchange authorization code for tokens
     */
    [[nodiscard]] auto exchangeAuthCode(std::string const& auth_code)
        -> Result<void>;

    /**
     * Save tokens to encrypted file
     */
    [[nodiscard]] auto saveTokens(std::string const& file_path) const
        -> Result<void>;

    /**
     * Load tokens from encrypted file
     */
    [[nodiscard]] static auto loadTokens(std::string const& file_path)
        -> Result<OAuth2Config>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Market Data Client
 *
 * Retrieves market data from Schwab API.
 */
class MarketDataClient {
public:
    explicit MarketDataClient(std::shared_ptr<TokenManager> token_manager);
    ~MarketDataClient();

    // Delete copy, allow move
    MarketDataClient(MarketDataClient const&) = delete;
    auto operator=(MarketDataClient const&) = delete;
    MarketDataClient(MarketDataClient&&) noexcept;
    auto operator=(MarketDataClient&&) noexcept -> MarketDataClient&;

    /**
     * Get real-time quote for symbol
     */
    [[nodiscard]] auto getQuote(std::string const& symbol) const
        -> Result<Quote>;

    /**
     * Get quotes for multiple symbols (batch)
     */
    [[nodiscard]] auto getQuotes(std::vector<std::string> const& symbols) const
        -> Result<std::vector<Quote>>;

    /**
     * Get historical bars
     */
    [[nodiscard]] auto getBars(
        std::string const& symbol,
        std::string const& period_type,    // "day", "month", "year"
        int period,                         // Number of periods
        std::string const& frequency_type,  // "minute", "daily", "weekly"
        int frequency                       // Frequency amount
    ) const -> Result<std::vector<Bar>>;

    /**
     * Get options chain
     */
    [[nodiscard]] auto getOptionsChain(OptionsChainRequest const& request) const
        -> Result<OptionsChainData>;

    /**
     * Get option quote
     */
    [[nodiscard]] auto getOptionQuote(std::string const& option_symbol) const
        -> Result<Quote>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Trading Client
 *
 * Places and manages orders.
 */
class TradingClient {
public:
    explicit TradingClient(std::shared_ptr<TokenManager> token_manager);
    ~TradingClient();

    // Delete copy, allow move
    TradingClient(TradingClient const&) = delete;
    auto operator=(TradingClient const&) = delete;
    TradingClient(TradingClient&&) noexcept;
    auto operator=(TradingClient&&) noexcept -> TradingClient&;

    /**
     * Place order
     */
    [[nodiscard]] auto placeOrder(OrderRequest const& order) const
        -> Result<std::string>;  // Returns order ID

    /**
     * Cancel order
     */
    [[nodiscard]] auto cancelOrder(
        std::string const& account_id,
        std::string const& order_id
    ) const -> Result<void>;

    /**
     * Get order status
     */
    [[nodiscard]] auto getOrderStatus(
        std::string const& account_id,
        std::string const& order_id
    ) const -> Result<Order>;

    /**
     * Get all orders for account
     */
    [[nodiscard]] auto getOrders(
        std::string const& account_id,
        int max_results = 100
    ) const -> Result<std::vector<Order>>;

    /**
     * Replace order (modify existing order)
     */
    [[nodiscard]] auto replaceOrder(
        std::string const& account_id,
        std::string const& order_id,
        OrderRequest const& new_order
    ) const -> Result<void>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Account Client
 *
 * Account information and positions.
 */
class AccountClient {
public:
    explicit AccountClient(std::shared_ptr<TokenManager> token_manager);
    ~AccountClient();

    // Delete copy, allow move
    AccountClient(AccountClient const&) = delete;
    auto operator=(AccountClient const&) = delete;
    AccountClient(AccountClient&&) noexcept;
    auto operator=(AccountClient&&) noexcept -> AccountClient&;

    /**
     * Get account information
     */
    [[nodiscard]] auto getAccount(std::string const& account_id) const
        -> Result<AccountInfo>;

    /**
     * Get all accounts
     */
    [[nodiscard]] auto getAccounts() const
        -> Result<std::vector<AccountInfo>>;

    /**
     * Get positions for account
     */
    [[nodiscard]] auto getPositions(std::string const& account_id) const
        -> Result<std::vector<Position>>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * WebSocket Streaming Client
 *
 * Real-time data streaming via WebSocket.
 */
class StreamingClient {
public:
    explicit StreamingClient(std::shared_ptr<TokenManager> token_manager);
    ~StreamingClient();

    // Delete copy, allow move
    StreamingClient(StreamingClient const&) = delete;
    auto operator=(StreamingClient const&) = delete;
    StreamingClient(StreamingClient&&) noexcept;
    auto operator=(StreamingClient&&) noexcept -> StreamingClient&;

    /**
     * Connect to WebSocket stream
     */
    [[nodiscard]] auto connect() -> Result<void>;

    /**
     * Disconnect from stream
     */
    auto disconnect() -> void;

    /**
     * Subscribe to quotes
     */
    [[nodiscard]] auto subscribeQuotes(std::vector<std::string> const& symbols)
        -> Result<void>;

    /**
     * Subscribe to trades
     */
    [[nodiscard]] auto subscribeTrades(std::vector<std::string> const& symbols)
        -> Result<void>;

    /**
     * Subscribe to options quotes
     */
    [[nodiscard]] auto subscribeOptions(std::vector<std::string> const& symbols)
        -> Result<void>;

    /**
     * Unsubscribe from symbols
     */
    [[nodiscard]] auto unsubscribe(std::vector<std::string> const& symbols)
        -> Result<void>;

    /**
     * Set message handler
     */
    auto setHandler(std::shared_ptr<StreamHandler> handler) -> void;

    /**
     * Check if connected
     */
    [[nodiscard]] auto isConnected() const noexcept -> bool;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Unified Schwab Client
 *
 * Main entry point combining all API functionality.
 */
class SchwabClient {
public:
    explicit SchwabClient(OAuth2Config config);
    ~SchwabClient();

    // Delete copy, allow move
    SchwabClient(SchwabClient const&) = delete;
    auto operator=(SchwabClient const&) = delete;
    SchwabClient(SchwabClient&&) noexcept;
    auto operator=(SchwabClient&&) noexcept -> SchwabClient&;

    /**
     * Initialize client (authenticate, connect)
     */
    [[nodiscard]] auto initialize() -> Result<void>;

    /**
     * Get market data client
     */
    [[nodiscard]] auto marketData() -> MarketDataClient&;

    /**
     * Get trading client
     */
    [[nodiscard]] auto trading() -> TradingClient&;

    /**
     * Get account client
     */
    [[nodiscard]] auto account() -> AccountClient&;

    /**
     * Get streaming client
     */
    [[nodiscard]] auto streaming() -> StreamingClient&;

    /**
     * Get token manager
     */
    [[nodiscard]] auto tokens() -> TokenManager&;

private:
    std::shared_ptr<TokenManager> token_manager_;
    std::unique_ptr<MarketDataClient> market_data_;
    std::unique_ptr<TradingClient> trading_;
    std::unique_ptr<AccountClient> account_;
    std::unique_ptr<StreamingClient> streaming_;
};

} // namespace bigbrother::schwab
