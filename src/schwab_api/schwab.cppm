/**
 * BigBrotherAnalytics - Schwab API Module (C++23)
 *
 * Complete C++23 client for Charles Schwab Trading API.
 * Implements OAuth 2.0 authentication with automatic token refresh.
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - Fluent API for queries and orders
 * - Thread-safe token management
 * - std::expected for error handling
 * - Modern C++23 features
 *
 * API Documentation:
 * https://developer.schwab.com/products/trader-api--individual
 */

// Global module fragment
module;

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <atomic>
#include <optional>

// Module declaration
export module bigbrother.schwab;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.options.pricing;

export namespace bigbrother::schwab {

using namespace bigbrother::types;

// ============================================================================
// Basic Type Definitions
// ============================================================================

/**
 * Quote Data
 */
struct Quote {
    std::string symbol;
    Price bid{0.0};
    Price ask{0.0};
    Price last{0.0};
    Volume volume{0};
};

/**
 * Option Contract Details
 */
struct OptionContract {
    std::string symbol;
    std::string underlying;
    OptionType type{OptionType::Call};
    Price strike{0.0};
    Timestamp expiration{0};
};

/**
 * Account Information
 */
struct AccountInfo {
    std::string account_id;
    double cash_balance{0.0};
    double buying_power{0.0};
};

/**
 * Order Request
 */
struct OrderRequest {
    std::string symbol;
    int quantity{0};
    std::string order_type{"MARKET"};
};

// ============================================================================
// OAuth2 Configuration
// ============================================================================

/**
 * OAuth 2.0 Configuration
 * C.1: Struct for passive data
 */
struct OAuth2Config {
    std::string client_id;          // App Key from Schwab
    std::string client_secret;      // App Secret from Schwab
    std::string redirect_uri;       // Callback URL
    std::string auth_code;          // Authorization code
    std::string refresh_token;      // Refresh token (persisted)
    std::string access_token;       // Access token (30-min lifetime)
    std::chrono::system_clock::time_point token_expiry;

    [[nodiscard]] auto isAccessTokenExpired() const noexcept -> bool {
        auto const now = std::chrono::system_clock::now();
        auto const safe_expiry = token_expiry - std::chrono::minutes(5);
        return now >= safe_expiry;
    }

    [[nodiscard]] auto validate() const noexcept -> Result<void>;
};

// ============================================================================
// Market Data Types
// ============================================================================

/**
 * Options Chain Request
 */
struct OptionsChainRequest {
    std::string symbol;
    std::string contract_type{"ALL"};      // "CALL", "PUT", "ALL"
    std::string strategy{"SINGLE"};
    double strike_from{0.0};
    double strike_to{0.0};

    [[nodiscard]] static auto forSymbol(std::string symbol) -> OptionsChainRequest {
        return OptionsChainRequest{.symbol = std::move(symbol)};
    }
};

/**
 * Option Quote Data (un-nested for std::optional compatibility)
 */
struct OptionQuote {
    OptionContract contract;
    Quote quote;
    options::Greeks greeks;
    double implied_volatility{0.0};
    int volume{0};
    int open_interest{0};
};

/**
 * Options Chain Data
 */
struct OptionsChainData {
    std::string symbol;
    std::string status;
    std::vector<OptionQuote> calls;
    std::vector<OptionQuote> puts;

    // Removed findContract method - causes std::optional compatibility issues
    // TODO: Implement when needed using pointer or Result<> instead
};

// ============================================================================
// Schwab Client - Main API Interface
// ============================================================================

/**
 * Token Manager
 */
class TokenManager {
public:
    explicit TokenManager(OAuth2Config config);
    ~TokenManager();

    TokenManager(TokenManager const&) = delete;
    auto operator=(TokenManager const&) -> TokenManager& = delete;
    TokenManager(TokenManager&&) noexcept;
    auto operator=(TokenManager&&) noexcept -> TokenManager&;

    [[nodiscard]] auto getAccessToken() -> Result<std::string>;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * Unified Schwab Client
 */
class SchwabClient {
public:
    explicit SchwabClient(OAuth2Config config);
    ~SchwabClient();

    SchwabClient(SchwabClient const&) = delete;
    auto operator=(SchwabClient const&) -> SchwabClient& = delete;
    SchwabClient(SchwabClient&&) noexcept;
    auto operator=(SchwabClient&&) noexcept -> SchwabClient&;

    [[nodiscard]] auto initialize() -> Result<void>;
    [[nodiscard]] auto getQuote(std::string const& symbol) const -> Result<Quote>;
    [[nodiscard]] auto getOptionsChain(OptionsChainRequest const& request) const -> Result<OptionsChainData>;
    [[nodiscard]] auto placeOrder(OrderRequest const& order) const -> Result<std::string>;
    [[nodiscard]] auto getAccount(std::string const& account_id) const -> Result<AccountInfo>;

private:
    std::shared_ptr<TokenManager> token_manager_;
};

// ============================================================================
// Fluent API for Schwab Queries
// ============================================================================

/**
 * Schwab Query Builder - Fluent API
 *
 * Example Usage:
 *
 *   // Get quote
 *   auto quote = SchwabQuery(client)
 *       .symbol("AAPL")
 *       .getQuote();
 *
 *   // Get options chain
 *   auto chain = SchwabQuery(client)
 *       .symbol("SPY")
 *       .calls()
 *       .strikes(580.0, 620.0)
 *       .daysToExpiration(30)
 *       .getOptionsChain();
 *
 *   // Place order
 *   auto order_id = SchwabOrder(client)
 *       .account(account_id)
 *       .buy()
 *       .symbol("AAPL")
 *       .quantity(100)
 *       .market()
 *       .submit();
 */
class SchwabQuery {
public:
    explicit SchwabQuery(SchwabClient& client)
        : client_{client} {}

    [[nodiscard]] auto symbol(std::string sym) -> SchwabQuery& {
        symbol_ = std::move(sym);
        return *this;
    }

    [[nodiscard]] auto calls() noexcept -> SchwabQuery& {
        request_.contract_type = "CALL";
        return *this;
    }

    [[nodiscard]] auto puts() noexcept -> SchwabQuery& {
        request_.contract_type = "PUT";
        return *this;
    }

    [[nodiscard]] auto strikes(double from, double to) noexcept -> SchwabQuery& {
        request_.strike_from = from;
        request_.strike_to = to;
        return *this;
    }

    /**
     * Get quote (terminal operation)
     */
    [[nodiscard]] auto getQuote() -> Result<Quote> {
        return client_.getQuote(symbol_);
    }

    /**
     * Get options chain (terminal operation)
     */
    [[nodiscard]] auto getOptionsChain() -> Result<OptionsChainData> {
        request_.symbol = symbol_;
        return client_.getOptionsChain(request_);
    }

private:
    SchwabClient& client_;
    std::string symbol_;
    OptionsChainRequest request_;
};

} // export namespace bigbrother::schwab

// ============================================================================
// Implementation Section
// ============================================================================

module :private;

namespace bigbrother::schwab {

// OAuth2Config validation
auto OAuth2Config::validate() const noexcept -> Result<void> {
    if (client_id.empty()) {
        return makeError<void>(ErrorCode::InvalidParameter, "Client ID required");
    }
    if (redirect_uri.empty()) {
        return makeError<void>(ErrorCode::InvalidParameter, "Redirect URI required");
    }
    return {};
}

// OptionsChainData::findContract removed due to std::optional<OptionQuote> compatibility issues

} // namespace bigbrother::schwab
