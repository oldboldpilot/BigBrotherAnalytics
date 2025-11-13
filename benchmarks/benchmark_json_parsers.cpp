/**
 * BigBrotherAnalytics - JSON Parser Performance Benchmarks
 *
 * Compares simdjson vs nlohmann/json performance across various workloads.
 * Validates the claimed 2.5x speedup for production use cases.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 */

#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>
#include <simdjson.h>
#include <string>
#include <vector>

// Import C++23 modules
import bigbrother.utils.simdjson_wrapper;

using json = nlohmann::json;

// ============================================================================
// Test Data - Real-world JSON from Schwab API and NewsAPI
// ============================================================================

// Schwab Quote Response (~500 bytes)
constexpr auto SCHWAB_QUOTE_JSON = R"({
    "AAPL": {
        "assetMainType": "EQUITY",
        "assetSubType": "COMMON_STOCK",
        "symbol": "AAPL",
        "bidPrice": 170.50,
        "askPrice": 170.52,
        "lastPrice": 170.51,
        "bidSize": 100,
        "askSize": 200,
        "totalVolume": 50000000,
        "quoteTime": 1699999999000,
        "tradeTime": 1699999998000,
        "mark": 170.51,
        "markChange": 2.35,
        "markPercentChange": 1.40,
        "highPrice": 172.00,
        "lowPrice": 169.00,
        "openPrice": 170.00,
        "closePrice": 168.16,
        "52WeekHigh": 198.23,
        "52WeekLow": 164.08
    }
})";

// NewsAPI Response (~2KB with 3 articles)
constexpr auto NEWS_API_JSON = R"({
    "status": "ok",
    "totalResults": 3,
    "articles": [
        {
            "source": {
                "id": "bloomberg",
                "name": "Bloomberg"
            },
            "author": "John Smith",
            "title": "Apple Stock Surges on Strong Q4 Earnings Beat",
            "description": "Apple Inc. reported quarterly earnings that exceeded Wall Street expectations, sending shares higher in after-hours trading.",
            "url": "https://example.com/article1",
            "urlToImage": "https://example.com/image1.jpg",
            "publishedAt": "2025-11-11T14:30:00Z",
            "content": "Apple Inc. (NASDAQ: AAPL) reported fourth-quarter earnings that beat analyst estimates..."
        },
        {
            "source": {
                "id": "reuters",
                "name": "Reuters"
            },
            "author": "Jane Doe",
            "title": "Tech Sector Leads Market Rally as Employment Data Beats Forecast",
            "description": "Technology stocks led broader market gains after better-than-expected jobs report.",
            "url": "https://example.com/article2",
            "urlToImage": "https://example.com/image2.jpg",
            "publishedAt": "2025-11-11T13:15:00Z",
            "content": "The technology sector rallied on Friday following strong employment data..."
        },
        {
            "source": {
                "id": "financial-times",
                "name": "Financial Times"
            },
            "author": "Robert Johnson",
            "title": "Federal Reserve Signals Pause in Rate Hikes",
            "description": "Fed officials indicated they may hold interest rates steady at upcoming meeting.",
            "url": "https://example.com/article3",
            "urlToImage": "https://example.com/image3.jpg",
            "publishedAt": "2025-11-11T12:00:00Z",
            "content": "Federal Reserve policymakers signaled a possible pause in interest rate increases..."
        }
    ]
})";

// Account Balance Response (~400 bytes)
constexpr auto ACCOUNT_BALANCE_JSON = R"({
    "securitiesAccount": {
        "accountNumber": "123456789",
        "hashValue": "HASH123ABC",
        "type": "MARGIN",
        "currentBalances": {
            "liquidationValue": 30000.00,
            "cashBalance": 28000.00,
            "cashAvailableForTrading": 28000.00,
            "buyingPower": 56000.00,
            "dayTradingBuyingPower": 112000.00,
            "marginBalance": 0.00,
            "marginEquity": 30000.00,
            "longMarketValue": 2000.00,
            "shortMarketValue": 0.00,
            "unsettledCash": 0.00,
            "maintenanceCall": 0.00,
            "regTCall": 0.00,
            "equityPercentage": 100.00
        }
    }
})";

// ============================================================================
// Benchmark: Schwab Quote Parsing (Hot Path - 120 req/min)
// ============================================================================

static void BM_NlohmannJson_QuoteParsing(benchmark::State& state) {
    for (auto _ : state) {
        try {
            auto j = json::parse(SCHWAB_QUOTE_JSON);

            auto const& aapl = j["AAPL"];
            double bid = aapl["bidPrice"];
            double ask = aapl["askPrice"];
            double last = aapl["lastPrice"];
            int64_t volume = aapl["totalVolume"];

            benchmark::DoNotOptimize(bid);
            benchmark::DoNotOptimize(ask);
            benchmark::DoNotOptimize(last);
            benchmark::DoNotOptimize(volume);
        } catch (...) {
            state.SkipWithError("Parse failed");
        }
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(SCHWAB_QUOTE_JSON).size());
}
BENCHMARK(BM_NlohmannJson_QuoteParsing);

static void BM_SimdJson_QuoteParsing(benchmark::State& state) {
    for (auto _ : state) {
        double bid = 0.0, ask = 0.0, last = 0.0;
        int64_t volume = 0;

        auto result = bigbrother::simdjson::parseAndProcess(SCHWAB_QUOTE_JSON, [&](auto& doc) {
            ::simdjson::ondemand::value root_value;
            if (doc.get_value().get(root_value) != ::simdjson::SUCCESS) return;

            ::simdjson::ondemand::value aapl;
            if (root_value["AAPL"].get(aapl) != ::simdjson::SUCCESS) return;

            aapl["bidPrice"].get_double().get(bid);
            aapl["askPrice"].get_double().get(ask);
            aapl["lastPrice"].get_double().get(last);
            aapl["totalVolume"].get_int64().get(volume);
        });

        benchmark::DoNotOptimize(bid);
        benchmark::DoNotOptimize(ask);
        benchmark::DoNotOptimize(last);
        benchmark::DoNotOptimize(volume);
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(SCHWAB_QUOTE_JSON).size());
}
BENCHMARK(BM_SimdJson_QuoteParsing);

// ============================================================================
// Benchmark: NewsAPI Parsing (96 req/day, 2KB response)
// ============================================================================

static void BM_NlohmannJson_NewsAPIParsing(benchmark::State& state) {
    for (auto _ : state) {
        try {
            auto j = json::parse(NEWS_API_JSON);

            int article_count = 0;
            if (j.contains("articles") && j["articles"].is_array()) {
                for (auto const& article : j["articles"]) {
                    std::string title = article.value("title", "");
                    std::string description = article.value("description", "");
                    std::string source_name = article["source"].value("name", "");

                    benchmark::DoNotOptimize(title);
                    benchmark::DoNotOptimize(description);
                    benchmark::DoNotOptimize(source_name);
                    article_count++;
                }
            }

            benchmark::DoNotOptimize(article_count);
        } catch (...) {
            state.SkipWithError("Parse failed");
        }
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(NEWS_API_JSON).size());
}
BENCHMARK(BM_NlohmannJson_NewsAPIParsing);

static void BM_SimdJson_NewsAPIParsing(benchmark::State& state) {
    for (auto _ : state) {
        int article_count = 0;

        auto result = bigbrother::simdjson::parseAndProcess(NEWS_API_JSON, [&](auto& doc) {
            ::simdjson::ondemand::value root_value;
            if (doc.get_value().get(root_value) != ::simdjson::SUCCESS) return;

            ::simdjson::ondemand::value articles_value;
            if (root_value["articles"].get(articles_value) != ::simdjson::SUCCESS) return;

            ::simdjson::ondemand::array articles_array;
            if (articles_value.get_array().get(articles_array) != ::simdjson::SUCCESS) return;

            for (auto article_result : articles_array) {
                ::simdjson::ondemand::value article;
                if (article_result.get(article) != ::simdjson::SUCCESS) continue;

                std::string_view title_sv, desc_sv, source_name_sv;
                article["title"].get_string().get(title_sv);
                article["description"].get_string().get(desc_sv);

                ::simdjson::ondemand::value source;
                if (article["source"].get(source) == ::simdjson::SUCCESS) {
                    source["name"].get_string().get(source_name_sv);
                }

                std::string title{title_sv};
                std::string description{desc_sv};
                std::string source_name{source_name_sv};

                benchmark::DoNotOptimize(title);
                benchmark::DoNotOptimize(description);
                benchmark::DoNotOptimize(source_name);
                article_count++;
            }
        });

        benchmark::DoNotOptimize(article_count);
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(NEWS_API_JSON).size());
}
BENCHMARK(BM_SimdJson_NewsAPIParsing);

// ============================================================================
// Benchmark: Account Balance Parsing (60 req/min)
// ============================================================================

static void BM_NlohmannJson_AccountBalanceParsing(benchmark::State& state) {
    for (auto _ : state) {
        try {
            auto j = json::parse(ACCOUNT_BALANCE_JSON);

            auto const& balances = j["securitiesAccount"]["currentBalances"];
            double total_equity = balances["liquidationValue"];
            double cash = balances["cashBalance"];
            double buying_power = balances["buyingPower"];
            double day_trading_bp = balances["dayTradingBuyingPower"];

            benchmark::DoNotOptimize(total_equity);
            benchmark::DoNotOptimize(cash);
            benchmark::DoNotOptimize(buying_power);
            benchmark::DoNotOptimize(day_trading_bp);
        } catch (...) {
            state.SkipWithError("Parse failed");
        }
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(ACCOUNT_BALANCE_JSON).size());
}
BENCHMARK(BM_NlohmannJson_AccountBalanceParsing);

static void BM_SimdJson_AccountBalanceParsing(benchmark::State& state) {
    for (auto _ : state) {
        double total_equity = 0.0, cash = 0.0, buying_power = 0.0, day_trading_bp = 0.0;

        auto result = bigbrother::simdjson::parseAndProcess(ACCOUNT_BALANCE_JSON, [&](auto& doc) {
            ::simdjson::ondemand::value root_value;
            if (doc.get_value().get(root_value) != ::simdjson::SUCCESS) return;

            ::simdjson::ondemand::value securities_account;
            if (root_value["securitiesAccount"].get(securities_account) != ::simdjson::SUCCESS) return;

            ::simdjson::ondemand::value balances;
            if (securities_account["currentBalances"].get(balances) != ::simdjson::SUCCESS) return;

            balances["liquidationValue"].get_double().get(total_equity);
            balances["cashBalance"].get_double().get(cash);
            balances["buyingPower"].get_double().get(buying_power);
            balances["dayTradingBuyingPower"].get_double().get(day_trading_bp);
        });

        benchmark::DoNotOptimize(total_equity);
        benchmark::DoNotOptimize(cash);
        benchmark::DoNotOptimize(buying_power);
        benchmark::DoNotOptimize(day_trading_bp);
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(ACCOUNT_BALANCE_JSON).size());
}
BENCHMARK(BM_SimdJson_AccountBalanceParsing);

// ============================================================================
// Benchmark: Simple Field Extraction (Micro-benchmark)
// ============================================================================

static void BM_NlohmannJson_SimpleFieldExtraction(benchmark::State& state) {
    constexpr auto SIMPLE_JSON = R"({"name": "test", "value": 42, "price": 3.14})";

    for (auto _ : state) {
        try {
            auto j = json::parse(SIMPLE_JSON);
            std::string name = j["name"];
            int64_t value = j["value"];
            double price = j["price"];

            benchmark::DoNotOptimize(name);
            benchmark::DoNotOptimize(value);
            benchmark::DoNotOptimize(price);
        } catch (...) {
            state.SkipWithError("Parse failed");
        }
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(SIMPLE_JSON).size());
}
BENCHMARK(BM_NlohmannJson_SimpleFieldExtraction);

static void BM_SimdJson_SimpleFieldExtraction(benchmark::State& state) {
    constexpr auto SIMPLE_JSON = R"({"name": "test", "value": 42, "price": 3.14})";

    for (auto _ : state) {
        auto name_result = bigbrother::simdjson::parseAndGet<std::string>(SIMPLE_JSON, "name");
        auto value_result = bigbrother::simdjson::parseAndGet<int64_t>(SIMPLE_JSON, "value");
        auto price_result = bigbrother::simdjson::parseAndGet<double>(SIMPLE_JSON, "price");

        if (name_result) benchmark::DoNotOptimize(*name_result);
        if (value_result) benchmark::DoNotOptimize(*value_result);
        if (price_result) benchmark::DoNotOptimize(*price_result);
    }

    state.SetBytesProcessed(state.iterations() * std::string_view(SIMPLE_JSON).size());
}
BENCHMARK(BM_SimdJson_SimpleFieldExtraction);

// ============================================================================
// Main
// ============================================================================

BENCHMARK_MAIN();
