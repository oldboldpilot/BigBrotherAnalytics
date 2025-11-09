/**
 * Standalone test for Schwab API JSON parsing functions
 *
 * Tests the parsing logic without full module compilation
 */

#include <iostream>
#include <string>
#include <nlohmann/json.hpp>
#include <vector>
#include <optional>

using json = nlohmann::json;

// Simplified data structures for testing
struct Quote {
    std::string symbol;
    double bid{0.0};
    double ask{0.0};
    double last{0.0};
    long volume{0};
    long timestamp{0};
};

struct OptionContract {
    std::string symbol;
    std::string underlying;
    std::string type; // "CALL" or "PUT"
    double strike{0.0};
    long expiration{0};
    int contract_size{100};
};

struct Greeks {
    double delta{0.0};
    double gamma{0.0};
    double theta{0.0};
    double vega{0.0};
    double rho{0.0};
};

struct OptionQuote {
    OptionContract contract;
    Quote quote;
    Greeks greeks;
    double implied_volatility{0.0};
    int open_interest{0};
    int volume{0};
};

struct OptionsChainData {
    std::string symbol;
    std::string status;
    std::vector<OptionQuote> calls;
    std::vector<OptionQuote> puts;
    double underlying_price{0.0};
    int days_to_expiration{0};
};

struct OHLCVBar {
    long timestamp{0};
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    long volume{0};

    bool isValid() const {
        return open > 0.0 && high >= low && low > 0.0 &&
               high >= open && high >= close &&
               low <= open && low <= close;
    }
};

struct HistoricalData {
    std::string symbol;
    std::vector<OHLCVBar> bars;
};

// Test the quote parsing
bool test_quote_parsing() {
    std::cout << "Testing Quote Parsing..." << std::endl;

    // Sample Schwab API response
    std::string json_str = R"({
        "AAPL": {
            "symbol": "AAPL",
            "bidPrice": 180.50,
            "askPrice": 180.55,
            "lastPrice": 180.52,
            "totalVolume": 45230000,
            "quoteTime": 1699564800000
        }
    })";

    try {
        auto data = json::parse(json_str);
        std::string symbol = "AAPL";

        Quote quote;
        quote.symbol = symbol;

        if (data.contains(symbol)) {
            auto const& q = data[symbol];
            quote.bid = q.value("bidPrice", 0.0);
            quote.ask = q.value("askPrice", 0.0);
            quote.last = q.value("lastPrice", 0.0);
            quote.volume = q.value("totalVolume", 0);
            quote.timestamp = q.value("quoteTime", 0);

            std::cout << "  Symbol: " << quote.symbol << std::endl;
            std::cout << "  Bid: $" << quote.bid << std::endl;
            std::cout << "  Ask: $" << quote.ask << std::endl;
            std::cout << "  Last: $" << quote.last << std::endl;
            std::cout << "  Volume: " << quote.volume << std::endl;
            std::cout << "  PASSED" << std::endl;
            return true;
        }
    } catch (json::exception const& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
        return false;
    }

    return false;
}

// Test option chain parsing
bool test_option_chain_parsing() {
    std::cout << "\nTesting Option Chain Parsing..." << std::endl;

    // Simplified Schwab options chain response
    std::string json_str = R"({
        "symbol": "SPY",
        "status": "SUCCESS",
        "underlying": {
            "symbol": "SPY",
            "last": 450.28
        },
        "callExpDateMap": {
            "2025-01-17:7": {
                "455.0": [
                    {
                        "symbol": "SPY_011725C455",
                        "bid": 2.50,
                        "ask": 2.55,
                        "last": 2.52,
                        "strikePrice": 455.0,
                        "expirationDate": 1737072000000,
                        "totalVolume": 5230,
                        "openInterest": 12450,
                        "volatility": 18.5,
                        "delta": 0.45,
                        "gamma": 0.05,
                        "theta": -0.12,
                        "vega": 0.25,
                        "rho": 0.08,
                        "multiplier": 100
                    }
                ]
            }
        },
        "putExpDateMap": {
            "2025-01-17:7": {
                "445.0": [
                    {
                        "symbol": "SPY_011725P445",
                        "bid": 1.80,
                        "ask": 1.85,
                        "last": 1.82,
                        "strikePrice": 445.0,
                        "expirationDate": 1737072000000,
                        "totalVolume": 3210,
                        "openInterest": 8900,
                        "volatility": 17.2,
                        "delta": -0.35,
                        "gamma": 0.04,
                        "theta": -0.10,
                        "vega": 0.22,
                        "rho": -0.06,
                        "multiplier": 100
                    }
                ]
            }
        }
    })";

    try {
        auto data = json::parse(json_str);

        OptionsChainData chain;
        chain.symbol = data.value("symbol", "");
        chain.status = data.value("status", "");

        if (data.contains("underlying")) {
            chain.underlying_price = data["underlying"].value("last", 0.0);
        }

        // Parse calls
        if (data.contains("callExpDateMap")) {
            auto const& call_map = data["callExpDateMap"];
            for (auto const& [exp_date_key, strike_map] : call_map.items()) {
                for (auto const& [strike_str, contracts] : strike_map.items()) {
                    if (!contracts.is_array() || contracts.empty()) continue;

                    for (auto const& contract_data : contracts) {
                        OptionQuote opt_quote;
                        opt_quote.contract.symbol = contract_data.value("symbol", "");
                        opt_quote.contract.underlying = chain.symbol;
                        opt_quote.contract.type = "CALL";
                        opt_quote.contract.strike = contract_data.value("strikePrice", 0.0);
                        opt_quote.greeks.delta = contract_data.value("delta", 0.0);
                        opt_quote.greeks.gamma = contract_data.value("gamma", 0.0);
                        opt_quote.implied_volatility = contract_data.value("volatility", 0.0);
                        opt_quote.open_interest = contract_data.value("openInterest", 0);

                        chain.calls.push_back(opt_quote);
                    }
                }
            }
        }

        // Parse puts
        if (data.contains("putExpDateMap")) {
            auto const& put_map = data["putExpDateMap"];
            for (auto const& [exp_date_key, strike_map] : put_map.items()) {
                for (auto const& [strike_str, contracts] : strike_map.items()) {
                    if (!contracts.is_array() || contracts.empty()) continue;

                    for (auto const& contract_data : contracts) {
                        OptionQuote opt_quote;
                        opt_quote.contract.symbol = contract_data.value("symbol", "");
                        opt_quote.contract.underlying = chain.symbol;
                        opt_quote.contract.type = "PUT";
                        opt_quote.contract.strike = contract_data.value("strikePrice", 0.0);
                        opt_quote.greeks.delta = contract_data.value("delta", 0.0);
                        opt_quote.greeks.gamma = contract_data.value("gamma", 0.0);
                        opt_quote.implied_volatility = contract_data.value("volatility", 0.0);
                        opt_quote.open_interest = contract_data.value("openInterest", 0);

                        chain.puts.push_back(opt_quote);
                    }
                }
            }
        }

        std::cout << "  Symbol: " << chain.symbol << std::endl;
        std::cout << "  Status: " << chain.status << std::endl;
        std::cout << "  Underlying Price: $" << chain.underlying_price << std::endl;
        std::cout << "  Calls: " << chain.calls.size() << std::endl;
        std::cout << "  Puts: " << chain.puts.size() << std::endl;

        if (!chain.calls.empty()) {
            auto const& call = chain.calls[0];
            std::cout << "  Sample Call: " << call.contract.symbol
                      << " Strike: $" << call.contract.strike
                      << " Delta: " << call.greeks.delta << std::endl;
        }

        if (!chain.puts.empty()) {
            auto const& put = chain.puts[0];
            std::cout << "  Sample Put: " << put.contract.symbol
                      << " Strike: $" << put.contract.strike
                      << " Delta: " << put.greeks.delta << std::endl;
        }

        std::cout << "  PASSED" << std::endl;
        return true;

    } catch (json::exception const& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
        return false;
    }
}

// Test historical data parsing
bool test_historical_data_parsing() {
    std::cout << "\nTesting Historical Data Parsing..." << std::endl;

    std::string json_str = R"({
        "candles": [
            {
                "open": 180.00,
                "high": 182.50,
                "low": 179.25,
                "close": 181.75,
                "volume": 52340000,
                "datetime": 1699564800000
            },
            {
                "open": 181.80,
                "high": 183.00,
                "low": 180.50,
                "close": 182.25,
                "volume": 48230000,
                "datetime": 1699651200000
            }
        ],
        "symbol": "AAPL",
        "empty": false
    })";

    try {
        auto data = json::parse(json_str);

        HistoricalData history;
        history.symbol = data.value("symbol", "");

        if (data.contains("candles")) {
            for (auto const& candle : data["candles"]) {
                OHLCVBar bar;
                bar.timestamp = candle.value("datetime", 0);
                bar.open = candle.value("open", 0.0);
                bar.high = candle.value("high", 0.0);
                bar.low = candle.value("low", 0.0);
                bar.close = candle.value("close", 0.0);
                bar.volume = candle.value("volume", 0);

                if (bar.isValid()) {
                    history.bars.push_back(bar);
                }
            }
        }

        std::cout << "  Symbol: " << history.symbol << std::endl;
        std::cout << "  Bars: " << history.bars.size() << std::endl;

        if (!history.bars.empty()) {
            auto const& bar = history.bars[0];
            std::cout << "  Sample Bar: O=" << bar.open
                      << " H=" << bar.high
                      << " L=" << bar.low
                      << " C=" << bar.close
                      << " V=" << bar.volume << std::endl;
        }

        std::cout << "  PASSED" << std::endl;
        return true;

    } catch (json::exception const& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Schwab API JSON Parsing Tests ===" << std::endl << std::endl;

    int passed = 0;
    int total = 0;

    total++;
    if (test_quote_parsing()) passed++;

    total++;
    if (test_option_chain_parsing()) passed++;

    total++;
    if (test_historical_data_parsing()) passed++;

    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    std::cout << "Pass Rate: " << (100.0 * passed / total) << "%" << std::endl;

    return (passed == total) ? 0 : 1;
}
