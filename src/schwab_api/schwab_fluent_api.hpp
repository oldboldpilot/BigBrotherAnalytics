#pragma once

#include "schwab_client.hpp"
#include <optional>

namespace bigbrother::schwab {

/**
 * Fluent API for Schwab Trading
 *
 * Provides intuitive, chainable interface for trading operations.
 *
 * Example Usage:
 *
 *   // Get a quote
 *   auto quote = SchwabQuery(client)
 *       .forSymbol("AAPL")
 *       .getQuote();
 *
 *   // Get options chain
 *   auto chain = SchwabQuery(client)
 *       .forSymbol("SPY")
 *       .calls()
 *       .daysToExpiration(30)
 *       .getOptionsChain();
 *
 *   // Place a stock order
 *   auto order_id = SchwabOrder(client)
 *       .buy("AAPL")
 *       .quantity(100)
 *       .market()
 *       .submit();
 *
 *   // Place an options order
 *   auto option_order = SchwabOrder(client)
 *       .buyToOpen("SPY_012025C450")
 *       .contracts(1)
 *       .limit(5.25)
 *       .submit();
 *
 *   // Stream real-time quotes
 *   SchwabStream(client)
 *       .subscribe("AAPL", "MSFT", "GOOGL")
 *       .onQuote([](Quote const& q) {
 *           std::cout << q.symbol << ": " << q.last << std::endl;
 *       })
 *       .start();
 */

class SchwabQuery {
public:
    explicit SchwabQuery(SchwabClient& client)
        : client_{client} {}

    // Symbol selection
    [[nodiscard]] auto forSymbol(std::string symbol) noexcept -> SchwabQuery& {
        symbol_ = std::move(symbol);
        return *this;
    }

    // Options chain filters
    [[nodiscard]] auto calls() noexcept -> SchwabQuery& {
        contract_type_ = "CALL";
        return *this;
    }

    [[nodiscard]] auto puts() noexcept -> SchwabQuery& {
        contract_type_ = "PUT";
        return *this;
    }

    [[nodiscard]] auto allOptions() noexcept -> SchwabQuery& {
        contract_type_ = "ALL";
        return *this;
    }

    [[nodiscard]] auto daysToExpiration(int days) noexcept -> SchwabQuery& {
        days_to_expiration_ = days;
        return *this;
    }

    [[nodiscard]] auto strikeRange(double from, double to) noexcept -> SchwabQuery& {
        strike_from_ = from;
        strike_to_ = to;
        return *this;
    }

    // Historical data parameters
    [[nodiscard]] auto daily() noexcept -> SchwabQuery& {
        frequency_type_ = "daily";
        return *this;
    }

    [[nodiscard]] auto weekly() noexcept -> SchwabQuery& {
        frequency_type_ = "weekly";
        return *this;
    }

    [[nodiscard]] auto intraday(int minutes = 5) noexcept -> SchwabQuery& {
        frequency_type_ = "minute";
        frequency_ = minutes;
        return *this;
    }

    [[nodiscard]] auto period(int periods) noexcept -> SchwabQuery& {
        period_ = periods;
        return *this;
    }

    // Terminal operations

    /**
     * Get current quote
     */
    [[nodiscard]] auto getQuote() const -> Result<Quote> {
        if (symbol_.empty()) {
            return makeError<Quote>(
                ErrorCode::InvalidParameter,
                "Symbol not specified. Use forSymbol()"
            );
        }
        return client_.marketData().getQuote(symbol_);
    }

    /**
     * Get options chain
     */
    [[nodiscard]] auto getOptionsChain() const -> Result<OptionsChainData> {
        if (symbol_.empty()) {
            return makeError<OptionsChainData>(
                ErrorCode::InvalidParameter,
                "Symbol not specified"
            );
        }

        OptionsChainRequest request = OptionsChainRequest::forSymbol(symbol_);
        request.contract_type = contract_type_.value_or("ALL");
        request.days_to_expiration = days_to_expiration_.value_or(0);
        request.strike_from = strike_from_.value_or(0.0);
        request.strike_to = strike_to_.value_or(0.0);

        return client_.marketData().getOptionsChain(request);
    }

    /**
     * Get historical bars
     */
    [[nodiscard]] auto getBars() const -> Result<std::vector<Bar>> {
        if (symbol_.empty()) {
            return makeError<std::vector<Bar>>(
                ErrorCode::InvalidParameter,
                "Symbol not specified"
            );
        }

        return client_.marketData().getBars(
            symbol_,
            "day",  // period type
            period_.value_or(10),
            frequency_type_.value_or("daily"),
            frequency_.value_or(1)
        );
    }

private:
    SchwabClient& client_;
    std::string symbol_;

    // Options filters
    std::optional<std::string> contract_type_;
    std::optional<int> days_to_expiration_;
    std::optional<double> strike_from_;
    std::optional<double> strike_to_;

    // Historical data
    std::optional<std::string> frequency_type_;
    std::optional<int> frequency_;
    std::optional<int> period_;
};

/**
 * Schwab Order Builder
 */
class SchwabOrder {
public:
    explicit SchwabOrder(SchwabClient& client)
        : client_{client} {}

    // Order type selection
    [[nodiscard]] auto buy(std::string symbol) noexcept -> SchwabOrder& {
        symbol_ = std::move(symbol);
        side_ = OrderSide::Buy;
        asset_type_ = "EQUITY";
        return *this;
    }

    [[nodiscard]] auto sell(std::string symbol) noexcept -> SchwabOrder& {
        symbol_ = std::move(symbol);
        side_ = OrderSide::Sell;
        asset_type_ = "EQUITY";
        return *this;
    }

    [[nodiscard]] auto buyToOpen(std::string option_symbol) noexcept -> SchwabOrder& {
        symbol_ = std::move(option_symbol);
        side_ = OrderSide::Buy;
        asset_type_ = "OPTION";
        instruction_ = "BUY_TO_OPEN";
        return *this;
    }

    [[nodiscard]] auto sellToClose(std::string option_symbol) noexcept -> SchwabOrder& {
        symbol_ = std::move(option_symbol);
        side_ = OrderSide::Sell;
        asset_type_ = "OPTION";
        instruction_ = "SELL_TO_CLOSE";
        return *this;
    }

    [[nodiscard]] auto sellToOpen(std::string option_symbol) noexcept -> SchwabOrder& {
        symbol_ = std::move(option_symbol);
        side_ = OrderSide::Sell;
        asset_type_ = "OPTION";
        instruction_ = "SELL_TO_OPEN";
        return *this;
    }

    [[nodiscard]] auto buyToClose(std::string option_symbol) noexcept -> SchwabOrder& {
        symbol_ = std::move(option_symbol);
        side_ = OrderSide::Buy;
        asset_type_ = "OPTION";
        instruction_ = "BUY_TO_CLOSE";
        return *this;
    }

    // Quantity
    [[nodiscard]] auto quantity(Quantity qty) noexcept -> SchwabOrder& {
        quantity_ = qty;
        return *this;
    }

    [[nodiscard]] auto contracts(int num) noexcept -> SchwabOrder& {
        quantity_ = num * 100;  // Each contract = 100 shares
        return *this;
    }

    // Order types
    [[nodiscard]] auto market() noexcept -> SchwabOrder& {
        order_type_ = OrderType::Market;
        return *this;
    }

    [[nodiscard]] auto limit(Price price) noexcept -> SchwabOrder& {
        order_type_ = OrderType::Limit;
        limit_price_ = price;
        return *this;
    }

    [[nodiscard]] auto stop(Price price) noexcept -> SchwabOrder& {
        order_type_ = OrderType::Stop;
        stop_price_ = price;
        return *this;
    }

    [[nodiscard]] auto stopLimit(Price stop, Price limit) noexcept -> SchwabOrder& {
        order_type_ = OrderType::StopLimit;
        stop_price_ = stop;
        limit_price_ = limit;
        return *this;
    }

    // Time in force
    [[nodiscard]] auto day() noexcept -> SchwabOrder& {
        tif_ = TimeInForce::Day;
        return *this;
    }

    [[nodiscard]] auto gtc() noexcept -> SchwabOrder& {
        tif_ = TimeInForce::GTC;
        return *this;
    }

    [[nodiscard]] auto ioc() noexcept -> SchwabOrder& {
        tif_ = TimeInForce::IOC;
        return *this;
    }

    // Account
    [[nodiscard]] auto forAccount(std::string account_id) noexcept -> SchwabOrder& {
        account_id_ = std::move(account_id);
        return *this;
    }

    // Terminal operation: submit order
    [[nodiscard]] auto submit() const -> Result<std::string> {
        if (symbol_.empty()) {
            return makeError<std::string>(
                ErrorCode::InvalidParameter,
                "Symbol not specified"
            );
        }

        if (!account_id_) {
            return makeError<std::string>(
                ErrorCode::InvalidParameter,
                "Account ID not specified. Use forAccount()"
            );
        }

        OrderRequest order;
        order.account_id = *account_id_;
        order.symbol = symbol_;
        order.quantity = quantity_.value_or(0);
        order.order_type = order_type_.value_or(OrderType::Market);
        order.side = side_;
        order.limit_price = limit_price_.value_or(0.0);
        order.stop_price = stop_price_.value_or(0.0);
        order.time_in_force = tif_.value_or(TimeInForce::Day);
        order.asset_type = asset_type_;
        order.instruction = instruction_.value_or(
            side_ == OrderSide::Buy ? "BUY" : "SELL"
        );

        return client_.trading().placeOrder(order);
    }

    // Cancel order
    [[nodiscard]] auto cancel(std::string const& order_id) const -> Result<void> {
        if (!account_id_) {
            return makeError<void>(
                ErrorCode::InvalidParameter,
                "Account ID not specified"
            );
        }

        return client_.trading().cancelOrder(*account_id_, order_id);
    }

private:
    SchwabClient& client_;
    std::string symbol_;
    std::string asset_type_{"EQUITY"};
    OrderSide side_{OrderSide::Buy};

    std::optional<std::string> account_id_;
    std::optional<std::string> instruction_;
    std::optional<Quantity> quantity_;
    std::optional<OrderType> order_type_;
    std::optional<Price> limit_price_;
    std::optional<Price> stop_price_;
    std::optional<TimeInForce> tif_;
};

/**
 * Schwab Streaming API
 */
class SchwabStream {
public:
    explicit SchwabStream(SchwabClient& client)
        : client_{client} {}

    // Subscribe to symbols
    template<typename... Symbols>
    [[nodiscard]] auto subscribe(Symbols&&... symbols) -> SchwabStream& {
        (symbols_.push_back(std::forward<Symbols>(symbols)), ...);
        return *this;
    }

    // Callbacks
    [[nodiscard]] auto onQuote(StreamHandler::QuoteCallback callback) noexcept
        -> SchwabStream& {
        quote_callback_ = std::move(callback);
        return *this;
    }

    [[nodiscard]] auto onTrade(StreamHandler::TradeCallback callback) noexcept
        -> SchwabStream& {
        trade_callback_ = std::move(callback);
        return *this;
    }

    [[nodiscard]] auto onError(StreamHandler::ErrorCallback callback) noexcept
        -> SchwabStream& {
        error_callback_ = std::move(callback);
        return *this;
    }

    // Start streaming
    [[nodiscard]] auto start() -> Result<void> {
        if (symbols_.empty()) {
            return makeError<void>(
                ErrorCode::InvalidParameter,
                "No symbols specified for streaming"
            );
        }

        // Create handler
        auto handler = std::make_shared<CallbackHandler>(
            quote_callback_,
            trade_callback_,
            error_callback_
        );

        client_.streaming().setHandler(handler);

        // Connect
        if (auto result = client_.streaming().connect(); !result) {
            return std::unexpected(result.error());
        }

        // Subscribe
        return client_.streaming().subscribeQuotes(symbols_);
    }

    // Stop streaming
    auto stop() -> void {
        client_.streaming().disconnect();
    }

private:
    class CallbackHandler : public StreamHandler {
    public:
        CallbackHandler(
            StreamHandler::QuoteCallback quote_cb,
            StreamHandler::TradeCallback trade_cb,
            StreamHandler::ErrorCallback error_cb
        ) : quote_cb_{std::move(quote_cb)},
            trade_cb_{std::move(trade_cb)},
            error_cb_{std::move(error_cb)} {}

        auto onQuote(Quote const& quote) -> void override {
            if (quote_cb_) {
                quote_cb_(quote);
            }
        }

        auto onTrade(Trade const& trade) -> void override {
            if (trade_cb_) {
                trade_cb_(trade);
            }
        }

        auto onError(std::string const& error) -> void override {
            if (error_cb_) {
                error_cb_(error);
            }
        }

    private:
        StreamHandler::QuoteCallback quote_cb_;
        StreamHandler::TradeCallback trade_cb_;
        StreamHandler::ErrorCallback error_cb_;
    };

    SchwabClient& client_;
    std::vector<std::string> symbols_;
    StreamHandler::QuoteCallback quote_callback_;
    StreamHandler::TradeCallback trade_callback_;
    StreamHandler::ErrorCallback error_callback_;
};

/**
 * Convenience functions for quick operations
 */

// Get quote
[[nodiscard]] inline auto getQuote(
    SchwabClient& client,
    std::string const& symbol
) -> Result<Quote> {
    return SchwabQuery(client).forSymbol(symbol).getQuote();
}

// Place market buy order
[[nodiscard]] inline auto buyMarket(
    SchwabClient& client,
    std::string const& account_id,
    std::string const& symbol,
    Quantity quantity
) -> Result<std::string> {
    return SchwabOrder(client)
        .buy(symbol)
        .quantity(quantity)
        .market()
        .day()
        .forAccount(account_id)
        .submit();
}

// Place limit buy order
[[nodiscard]] inline auto buyLimit(
    SchwabClient& client,
    std::string const& account_id,
    std::string const& symbol,
    Quantity quantity,
    Price limit_price
) -> Result<std::string> {
    return SchwabOrder(client)
        .buy(symbol)
        .quantity(quantity)
        .limit(limit_price)
        .day()
        .forAccount(account_id)
        .submit();
}

// Get options chain
[[nodiscard]] inline auto getOptionsChain(
    SchwabClient& client,
    std::string const& symbol,
    int days_to_expiration = 30
) -> Result<OptionsChainData> {
    return SchwabQuery(client)
        .forSymbol(symbol)
        .allOptions()
        .daysToExpiration(days_to_expiration)
        .getOptionsChain();
}

} // namespace bigbrother::schwab
