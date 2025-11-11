/**
 * BigBrotherAnalytics - AlphaVantage News Module (C++23)
 *
 * AlphaVantage NEWS_SENTIMENT API integration.
 * Fetches financial news with AI-powered sentiment analysis.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase 5+: Multi-Source News Ingestion
 *
 * API Documentation: https://www.alphavantage.co/documentation/#news-sentiment
 *
 * Following C++ Core Guidelines:
 * - I.2: Avoid non-const global variables
 * - F.51: Prefer default arguments over overloading
 * - Trailing return type syntax
 */

// Global module fragment
module;

#include <chrono>
#include <curl/curl.h>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <iomanip>
#include <ctime>

// Module declaration
export module bigbrother.market_intelligence.alphavantage;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.circuit_breaker;
import bigbrother.market_intelligence.sentiment;

export namespace bigbrother::market_intelligence {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using namespace bigbrother::circuit_breaker;
using json = nlohmann::json;

// ============================================================================
// AlphaVantage Data Types
// ============================================================================

/**
 * AlphaVantage sentiment labels
 */
enum class AlphaSentimentLabel {
    Bearish,          // score < -0.35
    SomewhatBearish,  // score -0.35 to -0.15
    Neutral,          // score -0.15 to 0.15
    SomewhatBullish,  // score 0.15 to 0.35
    Bullish           // score > 0.35
};

/**
 * Ticker-specific sentiment from AlphaVantage
 */
struct TickerSentiment {
    std::string ticker;
    double sentiment_score{0.0};      // -1.0 (bearish) to 1.0 (bullish)
    std::string sentiment_label;      // "Bearish", "Neutral", "Bullish", etc.
    double relevance_score{0.0};      // 0.0 to 1.0
};

/**
 * AlphaVantage news article with AI-powered sentiment
 */
struct AlphaVantageArticle {
    std::string title;
    std::string url;
    std::string time_published;       // "20231115T093000"
    std::vector<std::string> authors;
    std::string summary;
    std::string banner_image;
    std::string source;
    std::string category_within_source;
    std::string source_domain;

    // Sentiment data
    std::vector<TickerSentiment> ticker_sentiment;
    double overall_sentiment_score{0.0};
    std::string overall_sentiment_label;

    // Topics (e.g., "Technology", "Finance", "Earnings")
    std::vector<std::string> topics;
};

/**
 * AlphaVantage API configuration
 */
struct AlphaVantageConfig {
    std::string api_key;
    std::string base_url{"https://www.alphavantage.co/query"};
    int timeout_seconds{30};
    int lookback_days{7};
    int max_results_per_symbol{50};  // AlphaVantage allows up to 1000

    // Circuit breaker configuration
    int circuit_breaker_failure_threshold{5};
    int circuit_breaker_timeout_seconds{60};
    int circuit_breaker_half_open_timeout_seconds{30};
    int circuit_breaker_half_open_max_calls{3};
};

// ============================================================================
// AlphaVantage News Collector
// ============================================================================

/**
 * AlphaVantage NEWS_SENTIMENT API client
 *
 * Fetches financial news with AI-powered sentiment analysis.
 * Provides higher quality sentiment scores than keyword-based methods.
 */
class AlphaVantageCollector {
  public:
    /**
     * Constructor
     *
     * @param config AlphaVantage configuration
     */
    explicit AlphaVantageCollector(AlphaVantageConfig config)
        : config_(std::move(config)),
          circuit_breaker_(CircuitConfig{
              .failure_threshold = config_.circuit_breaker_failure_threshold,
              .timeout = std::chrono::seconds(config_.circuit_breaker_timeout_seconds),
              .half_open_timeout = std::chrono::seconds(config_.circuit_breaker_half_open_timeout_seconds),
              .half_open_max_calls = config_.circuit_breaker_half_open_max_calls,
              .enable_logging = true,
              .name = "AlphaVantage"}) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        Logger::getInstance().info("AlphaVantage collector initialized");
        Logger::getInstance().info("  Base URL: {}", config_.base_url);
        Logger::getInstance().info("  Max results per symbol: {}", config_.max_results_per_symbol);
        Logger::getInstance().info("  Circuit breaker enabled: threshold={}, timeout={}s",
                                   config_.circuit_breaker_failure_threshold,
                                   config_.circuit_breaker_timeout_seconds);
    }

    /**
     * Destructor
     */
    ~AlphaVantageCollector() { curl_global_cleanup(); }

    // C.21: Rule of Five
    AlphaVantageCollector(AlphaVantageCollector const&) = delete;
    auto operator=(AlphaVantageCollector const&) -> AlphaVantageCollector& = delete;
    AlphaVantageCollector(AlphaVantageCollector&&) = delete;
    auto operator=(AlphaVantageCollector&&) -> AlphaVantageCollector& = delete;

    /**
     * Fetch news for a symbol
     *
     * @param symbol Stock symbol (e.g., "AAPL")
     * @param from_date Start date (ISO 8601 format: "20231115T0930")
     * @param to_date End date (ISO 8601 format)
     * @return Result containing vector of articles
     */
    [[nodiscard]] auto fetchNews(std::string const& symbol, std::string const& from_date = "",
                                 std::string const& to_date = "")
        -> Result<std::vector<AlphaVantageArticle>> {

        Logger::getInstance().info("AlphaVantage: Fetching news for symbol: {}", symbol);

        // Build API URL
        std::string url = buildAPIUrl(symbol, from_date, to_date);

        // Call API with circuit breaker protection
        try {
            auto response = circuit_breaker_.call<json>([this, &url]() -> std::expected<json, std::string> {
                auto result = callAlphaVantageAPI(url);

                if (!result) {
                    return std::unexpected(result.error().message);
                }

                return result.value();
            });

            if (!response) {
                return std::unexpected(
                    Error::make(ErrorCode::NetworkError,
                                "Failed to fetch news from AlphaVantage: " + response.error()));
            }

            // Parse response
            auto articles = parseArticles(response.value(), symbol);

            Logger::getInstance().info("  AlphaVantage: Fetched {} articles for {}",
                                       articles.size(), symbol);

            return articles;

        } catch (std::exception const& e) {
            Logger::getInstance().error("AlphaVantage error: {}", e.what());
            return std::unexpected(
                Error::make(ErrorCode::NetworkError, std::string("AlphaVantage error: ") + e.what()));
        } catch (...) {
            Logger::getInstance().error("Unknown AlphaVantage error");
            return std::unexpected(Error::make(ErrorCode::UnknownError, "Unknown error occurred"));
        }
    }

    /**
     * Fetch news for multiple symbols with rate limiting
     *
     * @param symbols Vector of stock symbols
     * @param from_date Start date
     * @param to_date End date
     * @return Result containing map of symbol -> articles
     */
    [[nodiscard]] auto fetchNewsBatch(std::vector<std::string> const& symbols,
                                      std::string const& from_date = "",
                                      std::string const& to_date = "")
        -> Result<std::map<std::string, std::vector<AlphaVantageArticle>>> {

        std::map<std::string, std::vector<AlphaVantageArticle>> results;

        Logger::getInstance().info("AlphaVantage: Batch fetching news for {} symbols", symbols.size());

        size_t success_count = 0;
        size_t failure_count = 0;

        for (auto const& symbol : symbols) {
            auto news_result = fetchNews(symbol, from_date, to_date);

            if (news_result) {
                results[symbol] = news_result.value();
                success_count++;
            } else {
                Logger::getInstance().warn("Failed to fetch news for {}: {}",
                                           symbol, news_result.error().message);
                failure_count++;
                results[symbol] = {};  // Empty vector for failed symbols
            }

            // Rate limiting: 1 request per second to be conservative
            // (AlphaVantage allows more, but we don't want to hammer the API)
            if (&symbol != &symbols.back()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        Logger::getInstance().info("AlphaVantage: Batch complete - {} successes, {} failures",
                                   success_count, failure_count);

        if (success_count == 0) {
            return std::unexpected(
                Error::make(ErrorCode::NetworkError, "All AlphaVantage requests failed"));
        }

        return results;
    }

    /**
     * Get circuit breaker state
     */
    [[nodiscard]] auto getCircuitBreakerState() const -> CircuitState {
        return circuit_breaker_.getState();
    }

    /**
     * Get circuit breaker statistics
     */
    [[nodiscard]] auto getCircuitBreakerStats() const -> CircuitStats {
        return circuit_breaker_.getStats();
    }

    /**
     * Check if circuit breaker is open
     */
    [[nodiscard]] auto isCircuitBreakerOpen() const -> bool {
        return circuit_breaker_.isOpen();
    }

    /**
     * Reset circuit breaker to CLOSED state
     */
    auto resetCircuitBreaker() -> void {
        circuit_breaker_.reset();
    }

  private:
    AlphaVantageConfig config_;
    CircuitBreaker circuit_breaker_;
    mutable std::mutex api_mutex_;

    /**
     * Build AlphaVantage API URL
     *
     * Example: https://www.alphavantage.co/query?function=NEWS_SENTIMENT
     *          &tickers=AAPL&time_from=20231101T0000&limit=50&apikey=XXX
     */
    [[nodiscard]] auto buildAPIUrl(std::string const& symbol, std::string const& from_date,
                                    std::string const& to_date) const -> std::string {
        std::ostringstream url;
        url << config_.base_url << "?function=NEWS_SENTIMENT";
        url << "&tickers=" << symbol;

        if (!from_date.empty()) {
            url << "&time_from=" << from_date;
        }

        if (!to_date.empty()) {
            url << "&time_to=" << to_date;
        }

        url << "&limit=" << config_.max_results_per_symbol;
        url << "&apikey=" << config_.api_key;

        return url.str();
    }

    /**
     * CURL write callback
     */
    static auto writeCallback(void* contents, size_t size, size_t nmemb, void* userp) -> size_t {
        static_cast<std::string*>(userp)->append(static_cast<char*>(contents), size * nmemb);
        return size * nmemb;
    }

    /**
     * Call AlphaVantage API
     *
     * Thread-safe HTTP request with timeout and error handling.
     */
    [[nodiscard]] auto callAlphaVantageAPI(std::string const& url) -> Result<json> {
        std::lock_guard<std::mutex> lock(api_mutex_);

        std::string response_data;
        CURL* curl = curl_easy_init();

        if (!curl) {
            return std::unexpected(Error::make(ErrorCode::NetworkError, "Failed to initialize CURL"));
        }

        // RAII wrapper for CURL cleanup
        struct CurlCleanup {
            CURL* curl;
            explicit CurlCleanup(CURL* c) : curl(c) {}
            ~CurlCleanup() {
                if (curl) {
                    curl_easy_cleanup(curl);
                }
            }
            // Rule of Five: Prevent copying and moving
            CurlCleanup(CurlCleanup const&) = delete;
            auto operator=(CurlCleanup const&) -> CurlCleanup& = delete;
            CurlCleanup(CurlCleanup&&) = delete;
            auto operator=(CurlCleanup&&) -> CurlCleanup& = delete;
        } cleanup{curl};

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.timeout_seconds);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "BigBrotherAnalytics/1.0");
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                                                std::string("CURL error: ") +
                                                    curl_easy_strerror(res)));
        }

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code != 200) {
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                                                "HTTP error: " + std::to_string(http_code)));
        }

        // Parse JSON response
        try {
            json response_json = json::parse(response_data);

            // Check for API error messages
            if (response_json.contains("Error Message")) {
                return std::unexpected(Error::make(ErrorCode::NetworkError,
                                                    response_json["Error Message"].get<std::string>()));
            }

            if (response_json.contains("Note")) {
                // API rate limit message
                Logger::getInstance().warn("AlphaVantage Note: {}",
                                           response_json["Note"].get<std::string>());
            }

            return response_json;

        } catch (json::exception const& e) {
            return std::unexpected(
                Error::make(ErrorCode::UnknownError, std::string("JSON parse error: ") + e.what()));
        }
    }

    /**
     * Parse AlphaVantage JSON response into articles
     */
    [[nodiscard]] auto parseArticles(json const& response_json, std::string const& symbol) const
        -> std::vector<AlphaVantageArticle> {

        std::vector<AlphaVantageArticle> articles;

        if (!response_json.contains("feed")) {
            Logger::getInstance().warn("AlphaVantage: No 'feed' field in response");
            return articles;
        }

        auto const& feed = response_json["feed"];
        if (!feed.is_array()) {
            Logger::getInstance().warn("AlphaVantage: 'feed' is not an array");
            return articles;
        }

        for (auto const& item : feed) {
            AlphaVantageArticle article;

            // Basic fields
            article.title = item.value("title", "");
            article.url = item.value("url", "");
            article.time_published = item.value("time_published", "");
            article.summary = item.value("summary", "");
            article.banner_image = item.value("banner_image", "");
            article.source = item.value("source", "");
            article.category_within_source = item.value("category_within_source", "");
            article.source_domain = item.value("source_domain", "");

            // Authors array
            if (item.contains("authors") && item["authors"].is_array()) {
                for (auto const& author : item["authors"]) {
                    article.authors.push_back(author.get<std::string>());
                }
            }

            // Topics array
            if (item.contains("topics") && item["topics"].is_array()) {
                for (auto const& topic : item["topics"]) {
                    if (topic.contains("topic")) {
                        article.topics.push_back(topic["topic"].get<std::string>());
                    }
                }
            }

            // Overall sentiment
            article.overall_sentiment_score = std::stod(item.value("overall_sentiment_score", "0.0"));
            article.overall_sentiment_label = item.value("overall_sentiment_label", "Neutral");

            // Ticker-specific sentiment
            if (item.contains("ticker_sentiment") && item["ticker_sentiment"].is_array()) {
                for (auto const& ts : item["ticker_sentiment"]) {
                    TickerSentiment ticker_sent;
                    ticker_sent.ticker = ts.value("ticker", "");
                    ticker_sent.sentiment_score = std::stod(ts.value("ticker_sentiment_score", "0.0"));
                    ticker_sent.sentiment_label = ts.value("ticker_sentiment_label", "Neutral");
                    ticker_sent.relevance_score = std::stod(ts.value("relevance_score", "0.0"));

                    article.ticker_sentiment.push_back(ticker_sent);
                }
            }

            articles.push_back(article);
        }

        Logger::getInstance().info("  AlphaVantage: Parsed {} articles", articles.size());

        return articles;
    }

    /**
     * Convert AlphaVantage timestamp to Unix timestamp
     *
     * Input: "20231115T093000"
     * Output: Unix timestamp (seconds since epoch)
     */
    [[nodiscard]] static auto parseTimestamp(std::string const& time_str) -> Timestamp {
        // Parse "20231115T093000" format
        std::tm tm = {};
        std::istringstream ss(time_str);
        ss >> std::get_time(&tm, "%Y%m%dT%H%M%S");

        if (ss.fail()) {
            Logger::getInstance().warn("Failed to parse timestamp: {}", time_str);
            return 0;
        }

        return static_cast<Timestamp>(std::mktime(&tm));
    }
};

} // namespace bigbrother::market_intelligence
