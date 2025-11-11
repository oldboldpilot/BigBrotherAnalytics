/**
 * BigBrotherAnalytics - Unified Market Data Types (C++23)
 *
 * Common data structures for market data from multiple sources
 * (Schwab, Yahoo Finance, AlphaVantage, NewsAPI).
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase 5+: Market Intelligence System
 *
 * Following C++ Core Guidelines:
 * - I.2: Avoid non-const global variables
 * - C.1: Use struct for passive data
 * - Trailing return type syntax
 */

// Global module fragment
module;

#include <chrono>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.market_intelligence.types;

export namespace bigbrother::market_intelligence {

// ============================================================================
// Market Data Sources
// ============================================================================

enum class DataSource { SCHWAB, YAHOO_FINANCE, NEWSAPI, ALPHAVANTAGE, UNKNOWN };

[[nodiscard]] auto to_string(DataSource source) -> std::string {
    switch (source) {
        case DataSource::SCHWAB:
            return "schwab";
        case DataSource::YAHOO_FINANCE:
            return "yahoo_finance";
        case DataSource::NEWSAPI:
            return "newsapi";
        case DataSource::ALPHAVANTAGE:
            return "alphavantage";
        case DataSource::UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

// ============================================================================
// Unified Quote Structure
// ============================================================================

/**
 * Unified quote structure for all market data sources
 * C.1: Use struct for passive data
 */
struct Quote {
    std::string symbol;
    double last_price{0.0};
    double bid{0.0};
    double ask{0.0};
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    int64_t volume{0};
    double change{0.0};
    double change_percent{0.0};
    std::chrono::system_clock::time_point timestamp;
    DataSource source{DataSource::UNKNOWN};

    // Additional fields
    double bid_size{0.0};
    double ask_size{0.0};
    std::string exchange;

    // Derived calculations
    [[nodiscard]] auto spread() const -> double { return ask - bid; }

    [[nodiscard]] auto mid_price() const -> double { return (bid + ask) / 2.0; }
};

// ============================================================================
// OHLCV (Candlestick) Data
// ============================================================================

struct OHLCV {
    std::chrono::system_clock::time_point timestamp;
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    int64_t volume{0};
    double adjusted_close{0.0}; // Adjusted for splits/dividends
};

// ============================================================================
// Position Data
// ============================================================================

enum class PositionType { EQUITY, OPTION, FUTURE, FOREX, CRYPTO, UNKNOWN };

struct Position {
    std::string symbol;
    PositionType type{PositionType::UNKNOWN};
    double quantity{0.0};
    double average_price{0.0};
    double current_price{0.0};
    double market_value{0.0};
    double cost_basis{0.0};
    double unrealized_pnl{0.0};
    double unrealized_pnl_percent{0.0};
    std::string account_hash;

    // Option-specific fields
    std::string underlying_symbol;
    double strike{0.0};
    std::string expiration_date;
    bool is_call{true};
};

// ============================================================================
// News Article
// ============================================================================

struct NewsArticle {
    std::string article_id;
    std::string symbol;
    std::string title;
    std::string description;
    std::string content;
    std::string url;
    std::string source_name;
    std::string source_id;
    std::string author;
    std::chrono::system_clock::time_point published_at;
    std::chrono::system_clock::time_point fetched_at;

    // Sentiment analysis
    double sentiment_score{0.0}; // -1.0 to 1.0
    std::string sentiment_label; // "positive", "negative", "neutral"
    std::vector<std::string> positive_keywords;
    std::vector<std::string> negative_keywords;

    // Metadata
    std::string image_url;
    std::string category;
    std::string language{"en"};
    std::vector<std::string> topics;

    DataSource source{DataSource::UNKNOWN};
};

// ============================================================================
// Order Types (for future trading)
// ============================================================================

enum class OrderAction { BUY, SELL, BUY_TO_COVER, SELL_SHORT };

enum class OrderType { MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP };

enum class OrderDuration {
    DAY,
    GTC, // Good Till Canceled
    GTD, // Good Till Date
    IOC, // Immediate or Cancel
    FOK  // Fill or Kill
};

enum class OrderStatus {
    PENDING,
    ACCEPTED,
    WORKING,
    FILLED,
    PARTIAL_FILL,
    CANCELED,
    REJECTED,
    EXPIRED
};

struct Order {
    std::string order_id;
    std::string symbol;
    OrderAction action;
    OrderType type;
    OrderDuration duration{OrderDuration::DAY};
    double quantity{0.0};
    double limit_price{0.0};
    double stop_price{0.0};
    OrderStatus status{OrderStatus::PENDING};

    std::chrono::system_clock::time_point placed_at;
    std::chrono::system_clock::time_point updated_at;

    double filled_quantity{0.0};
    double average_fill_price{0.0};
    std::string account_hash;
};

// ============================================================================
// Account Information
// ============================================================================

struct Account {
    std::string account_number;
    std::string account_hash;
    std::string account_type; // "CASH", "MARGIN", "IRA", etc.
    double total_value{0.0};
    double cash_balance{0.0};
    double buying_power{0.0};
    double margin_balance{0.0};
    double day_trading_buying_power{0.0};

    // P&L
    double total_pnl{0.0};
    double day_pnl{0.0};
    double day_pnl_percent{0.0};
};

// ============================================================================
// Options Chain
// ============================================================================

struct OptionContract {
    std::string symbol;
    std::string underlying_symbol;
    double strike{0.0};
    std::string expiration_date;
    bool is_call{true};

    double bid{0.0};
    double ask{0.0};
    double last{0.0};
    double mark{0.0};
    int64_t volume{0};
    int64_t open_interest{0};

    // Greeks
    double delta{0.0};
    double gamma{0.0};
    double theta{0.0};
    double vega{0.0};
    double rho{0.0};

    double implied_volatility{0.0};
    double intrinsic_value{0.0};
    double time_value{0.0};
};

struct OptionsChain {
    std::string underlying_symbol;
    double underlying_price{0.0};
    std::vector<std::string> expiration_dates;
    std::vector<OptionContract> calls;
    std::vector<OptionContract> puts;
    std::chrono::system_clock::time_point timestamp;
};

// ============================================================================
// Historical Period
// ============================================================================

enum class Period {
    ONE_DAY,
    FIVE_DAY,
    ONE_MONTH,
    THREE_MONTH,
    SIX_MONTH,
    ONE_YEAR,
    TWO_YEAR,
    FIVE_YEAR,
    TEN_YEAR,
    YTD,
    MAX
};

[[nodiscard]] auto to_string(Period period) -> std::string {
    switch (period) {
        case Period::ONE_DAY:
            return "1d";
        case Period::FIVE_DAY:
            return "5d";
        case Period::ONE_MONTH:
            return "1mo";
        case Period::THREE_MONTH:
            return "3mo";
        case Period::SIX_MONTH:
            return "6mo";
        case Period::ONE_YEAR:
            return "1y";
        case Period::TWO_YEAR:
            return "2y";
        case Period::FIVE_YEAR:
            return "5y";
        case Period::TEN_YEAR:
            return "10y";
        case Period::YTD:
            return "ytd";
        case Period::MAX:
            return "max";
    }
    return "1d";
}

enum class Interval {
    ONE_MINUTE,
    FIVE_MINUTE,
    FIFTEEN_MINUTE,
    THIRTY_MINUTE,
    ONE_HOUR,
    ONE_DAY,
    ONE_WEEK,
    ONE_MONTH
};

[[nodiscard]] auto to_string(Interval interval) -> std::string {
    switch (interval) {
        case Interval::ONE_MINUTE:
            return "1m";
        case Interval::FIVE_MINUTE:
            return "5m";
        case Interval::FIFTEEN_MINUTE:
            return "15m";
        case Interval::THIRTY_MINUTE:
            return "30m";
        case Interval::ONE_HOUR:
            return "1h";
        case Interval::ONE_DAY:
            return "1d";
        case Interval::ONE_WEEK:
            return "1wk";
        case Interval::ONE_MONTH:
            return "1mo";
    }
    return "1d";
}

} // namespace bigbrother::market_intelligence
