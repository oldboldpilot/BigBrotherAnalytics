/**
 * BigBrotherAnalytics - Market Data Python Bindings
 *
 * pybind11 bindings for C++23 market data modules (Schwab + Yahoo Finance).
 * Provides Python access to fluent API with GIL-free execution.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase 5+: Market Intelligence System
 *
 * Tagged: PYTHON_BINDINGS
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

// Import C++23 modules
import bigbrother.market_intelligence.types;
import bigbrother.market_intelligence.yahoo_finance;
import bigbrother.market_intelligence.schwab_api;

namespace py = pybind11;

using namespace bigbrother::market_intelligence;

// Tagged: PYTHON_BINDINGS
PYBIND11_MODULE(market_data_py, m) {
    m.doc() = "BigBrotherAnalytics Market Data - Python Bindings (Schwab + Yahoo Finance)";

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<DataSource>(m, "DataSource")
        .value("SCHWAB", DataSource::SCHWAB)
        .value("YAHOO_FINANCE", DataSource::YAHOO_FINANCE)
        .value("NEWSAPI", DataSource::NEWSAPI)
        .value("ALPHAVANTAGE", DataSource::ALPHAVANTAGE)
        .value("UNKNOWN", DataSource::UNKNOWN)
        .export_values();

    py::enum_<PositionType>(m, "PositionType")
        .value("EQUITY", PositionType::EQUITY)
        .value("OPTION", PositionType::OPTION)
        .value("FUTURE", PositionType::FUTURE)
        .value("FOREX", PositionType::FOREX)
        .value("CRYPTO", PositionType::CRYPTO)
        .value("UNKNOWN", PositionType::UNKNOWN)
        .export_values();

    py::enum_<OrderAction>(m, "OrderAction")
        .value("BUY", OrderAction::BUY)
        .value("SELL", OrderAction::SELL)
        .value("BUY_TO_COVER", OrderAction::BUY_TO_COVER)
        .value("SELL_SHORT", OrderAction::SELL_SHORT)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::MARKET)
        .value("LIMIT", OrderType::LIMIT)
        .value("STOP", OrderType::STOP)
        .value("STOP_LIMIT", OrderType::STOP_LIMIT)
        .value("TRAILING_STOP", OrderType::TRAILING_STOP)
        .export_values();

    py::enum_<Period>(m, "Period")
        .value("ONE_DAY", Period::ONE_DAY)
        .value("FIVE_DAY", Period::FIVE_DAY)
        .value("ONE_MONTH", Period::ONE_MONTH)
        .value("THREE_MONTH", Period::THREE_MONTH)
        .value("SIX_MONTH", Period::SIX_MONTH)
        .value("ONE_YEAR", Period::ONE_YEAR)
        .value("TWO_YEAR", Period::TWO_YEAR)
        .value("FIVE_YEAR", Period::FIVE_YEAR)
        .value("TEN_YEAR", Period::TEN_YEAR)
        .value("YTD", Period::YTD)
        .value("MAX", Period::MAX)
        .export_values();

    py::enum_<Interval>(m, "Interval")
        .value("ONE_MINUTE", Interval::ONE_MINUTE)
        .value("FIVE_MINUTE", Interval::FIVE_MINUTE)
        .value("FIFTEEN_MINUTE", Interval::FIFTEEN_MINUTE)
        .value("THIRTY_MINUTE", Interval::THIRTY_MINUTE)
        .value("ONE_HOUR", Interval::ONE_HOUR)
        .value("ONE_DAY", Interval::ONE_DAY)
        .value("ONE_WEEK", Interval::ONE_WEEK)
        .value("ONE_MONTH", Interval::ONE_MONTH)
        .export_values();

    // ========================================================================
    // Data Structures
    // ========================================================================

    py::class_<Quote>(m, "Quote")
        .def(py::init<>())
        .def_readwrite("symbol", &Quote::symbol)
        .def_readwrite("last_price", &Quote::last_price)
        .def_readwrite("bid", &Quote::bid)
        .def_readwrite("ask", &Quote::ask)
        .def_readwrite("open", &Quote::open)
        .def_readwrite("high", &Quote::high)
        .def_readwrite("low", &Quote::low)
        .def_readwrite("close", &Quote::close)
        .def_readwrite("volume", &Quote::volume)
        .def_readwrite("change", &Quote::change)
        .def_readwrite("change_percent", &Quote::change_percent)
        .def_readwrite("timestamp", &Quote::timestamp)
        .def_readwrite("source", &Quote::source)
        .def_readwrite("bid_size", &Quote::bid_size)
        .def_readwrite("ask_size", &Quote::ask_size)
        .def_readwrite("exchange", &Quote::exchange)
        .def("spread", &Quote::spread)
        .def("mid_price", &Quote::mid_price)
        .def("__repr__", [](Quote const& q) {
            return "<Quote " + q.symbol + " $" + std::to_string(q.last_price) +
                   " bid=" + std::to_string(q.bid) + " ask=" + std::to_string(q.ask) + ">";
        });

    py::class_<OHLCV>(m, "OHLCV")
        .def(py::init<>())
        .def_readwrite("timestamp", &OHLCV::timestamp)
        .def_readwrite("open", &OHLCV::open)
        .def_readwrite("high", &OHLCV::high)
        .def_readwrite("low", &OHLCV::low)
        .def_readwrite("close", &OHLCV::close)
        .def_readwrite("volume", &OHLCV::volume)
        .def_readwrite("adjusted_close", &OHLCV::adjusted_close);

    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("symbol", &Position::symbol)
        .def_readwrite("type", &Position::type)
        .def_readwrite("quantity", &Position::quantity)
        .def_readwrite("average_price", &Position::average_price)
        .def_readwrite("current_price", &Position::current_price)
        .def_readwrite("market_value", &Position::market_value)
        .def_readwrite("cost_basis", &Position::cost_basis)
        .def_readwrite("unrealized_pnl", &Position::unrealized_pnl)
        .def_readwrite("unrealized_pnl_percent", &Position::unrealized_pnl_percent)
        .def_readwrite("account_hash", &Position::account_hash)
        .def("__repr__", [](Position const& p) {
            return "<Position " + p.symbol + " qty=" + std::to_string(p.quantity) +
                   " value=$" + std::to_string(p.market_value) + ">";
        });

    py::class_<Account>(m, "Account")
        .def(py::init<>())
        .def_readwrite("account_number", &Account::account_number)
        .def_readwrite("account_hash", &Account::account_hash)
        .def_readwrite("account_type", &Account::account_type)
        .def_readwrite("total_value", &Account::total_value)
        .def_readwrite("cash_balance", &Account::cash_balance)
        .def_readwrite("buying_power", &Account::buying_power);

    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id", &Order::order_id)
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("action", &Order::action)
        .def_readwrite("type", &Order::type)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("limit_price", &Order::limit_price);

    // ========================================================================
    // Yahoo Finance Collector (Fluent API)
    // ========================================================================

    py::class_<YahooConfig>(m, "YahooConfig")
        .def(py::init<>())
        .def_readwrite("base_url", &YahooConfig::base_url)
        .def_readwrite("timeout_seconds", &YahooConfig::timeout_seconds)
        .def_readwrite("max_retries", &YahooConfig::max_retries)
        .def_readwrite("enable_circuit_breaker", &YahooConfig::enable_circuit_breaker);

    py::class_<YahooFinanceCollector>(m, "YahooFinanceCollector")
        .def(py::init<>())
        .def(py::init<YahooConfig>())
        // Fluent API
        .def("for_symbol", &YahooFinanceCollector::forSymbol,
            "Configure single symbol",
            py::return_value_policy::reference,
            py::arg("symbol"))
        .def("for_symbols", &YahooFinanceCollector::forSymbols,
            "Configure multiple symbols",
            py::return_value_policy::reference,
            py::arg("symbols"))
        .def("with_timeout", &YahooFinanceCollector::withTimeout,
            "Set timeout in seconds",
            py::return_value_policy::reference,
            py::arg("seconds"))
        .def("with_sentiment", &YahooFinanceCollector::withSentiment,
            "Enable sentiment analysis",
            py::return_value_policy::reference,
            py::arg("enable") = true)
        .def("with_parallel", &YahooFinanceCollector::withParallel,
            "Enable parallel fetching",
            py::return_value_policy::reference,
            py::arg("enable") = true)
        // Data fetching (GIL-free)
        .def("get_quotes", [](YahooFinanceCollector& collector) {
            py::gil_scoped_release release;  // GIL-FREE
            return collector.getQuotes();
        }, "Fetch quotes for configured symbols (GIL-free)")
        .def("get_quote", [](YahooFinanceCollector& collector) {
            py::gil_scoped_release release;  // GIL-FREE
            return collector.getQuote();
        }, "Fetch single quote (GIL-free)")
        .def("get_news", [](YahooFinanceCollector& collector) {
            py::gil_scoped_release release;  // GIL-FREE
            return collector.getNews();
        }, "Fetch news for configured symbols (GIL-free)")
        .def("get_history", [](YahooFinanceCollector& collector, Period period, Interval interval) {
            py::gil_scoped_release release;  // GIL-FREE
            return collector.getHistory(period, interval);
        }, "Fetch historical data (GIL-free)",
           py::arg("period") = Period::ONE_MONTH,
           py::arg("interval") = Interval::ONE_DAY);
        // Circuit breaker (temporarily disabled)
        // .def("is_circuit_breaker_open", &YahooFinanceCollector::isCircuitBreakerOpen)
        // .def("reset_circuit_breaker", &YahooFinanceCollector::resetCircuitBreaker);

    // ========================================================================
    // Schwab API Client (Fluent API)
    // ========================================================================

    py::class_<SchwabConfig>(m, "SchwabConfig")
        .def(py::init<>())
        .def_readwrite("app_key", &SchwabConfig::app_key)
        .def_readwrite("app_secret", &SchwabConfig::app_secret)
        .def_readwrite("token_file", &SchwabConfig::token_file)
        .def_readwrite("base_url", &SchwabConfig::base_url)
        .def_readwrite("timeout_seconds", &SchwabConfig::timeout_seconds);

    py::class_<SchwabAPIClient>(m, "SchwabAPIClient")
        .def(py::init<SchwabConfig>())
        // Fluent API
        .def("for_symbol", &SchwabAPIClient::forSymbol,
            "Configure single symbol",
            py::return_value_policy::reference,
            py::arg("symbol"))
        .def("for_symbols", &SchwabAPIClient::forSymbols,
            "Configure multiple symbols",
            py::return_value_policy::reference,
            py::arg("symbols"))
        .def("for_account", &SchwabAPIClient::forAccount,
            "Configure account",
            py::return_value_policy::reference,
            py::arg("account_hash"))
        .def("with_timeout", &SchwabAPIClient::withTimeout,
            "Set timeout in seconds",
            py::return_value_policy::reference,
            py::arg("seconds"))
        .def("with_parallel", &SchwabAPIClient::withParallel,
            "Enable parallel fetching",
            py::return_value_policy::reference,
            py::arg("enable") = true)
        // Account & positions (GIL-free)
        .def("get_accounts", [](SchwabAPIClient& client) {
            py::gil_scoped_release release;  // GIL-FREE
            return client.getAccounts();
        }, "Fetch all accounts (GIL-free)")
        .def("get_positions", [](SchwabAPIClient& client) {
            py::gil_scoped_release release;  // GIL-FREE
            return client.getPositions();
        }, "Fetch positions for configured account (GIL-free)")
        // Market data (GIL-free)
        .def("get_quotes", [](SchwabAPIClient& client) {
            py::gil_scoped_release release;  // GIL-FREE
            return client.getQuotes();
        }, "Fetch quotes for configured symbols (GIL-free)")
        .def("get_quote", [](SchwabAPIClient& client) {
            py::gil_scoped_release release;  // GIL-FREE
            return client.getQuote();
        }, "Fetch single quote (GIL-free)")
        // Orders (fluent)
        .def("buy", &SchwabAPIClient::buy,
            "Configure buy order",
            py::return_value_policy::reference,
            py::arg("symbol"))
        .def("sell", &SchwabAPIClient::sell,
            "Configure sell order",
            py::return_value_policy::reference,
            py::arg("symbol"))
        .def("quantity", &SchwabAPIClient::quantity,
            "Set order quantity",
            py::return_value_policy::reference,
            py::arg("qty"))
        .def("limit_price", &SchwabAPIClient::limitPrice,
            "Set limit price",
            py::return_value_policy::reference,
            py::arg("price"))
        .def("market_order", &SchwabAPIClient::marketOrder,
            "Set as market order",
            py::return_value_policy::reference)
        .def("place_order", [](SchwabAPIClient& client) {
            py::gil_scoped_release release;  // GIL-FREE
            return client.placeOrder();
        }, "Place configured order (GIL-free)");
        // Circuit breaker (temporarily disabled)
        // .def("is_circuit_breaker_open", &SchwabAPIClient::isCircuitBreakerOpen)
        // .def("reset_circuit_breaker", &SchwabAPIClient::resetCircuitBreaker);

    // ========================================================================
    // Utility Functions
    // ========================================================================

    m.def("to_string", py::overload_cast<DataSource>(to_string),
          "Convert DataSource to string");
    m.def("to_string", py::overload_cast<Period>(to_string),
          "Convert Period to string");
    m.def("to_string", py::overload_cast<Interval>(to_string),
          "Convert Interval to string");
}
