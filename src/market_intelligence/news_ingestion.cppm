/**
 * BigBrotherAnalytics - News Ingestion Module (C++23)
 *
 * NewsAPI integration with DuckDB storage.
 * Fetches financial news and performs sentiment analysis.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase 5+: News Ingestion System
 *
 * Following C++ Core Guidelines:
 * - I.2: Avoid non-const global variables
 * - F.51: Prefer default arguments over overloading
 * - Trailing return type syntax
 */

// Global module fragment
module;

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <chrono>
#include <thread>
#include <expected>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

// Module declaration
export module bigbrother.market_intelligence.news;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.database;
import bigbrother.market_intelligence.sentiment;

export namespace bigbrother::market_intelligence {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using json = nlohmann::json;

// ============================================================================
// News Article Data Types
// ============================================================================

/**
 * News article structure
 * C.1: Use struct for passive data
 */
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
    Timestamp published_at{0};
    Timestamp fetched_at{0};
    double sentiment_score{0.0};
    std::string sentiment_label;
    std::vector<std::string> positive_keywords;
    std::vector<std::string> negative_keywords;
};

/**
 * News API configuration
 */
struct NewsAPIConfig {
    std::string api_key;
    std::string base_url{"https://newsapi.org/v2"};
    int requests_per_day{100};
    int lookback_days{7};
    int timeout_seconds{30};
};

// ============================================================================
// News API Collector
// ============================================================================

/**
 * NewsAPI client for fetching financial news
 *
 * Handles rate limiting, error recovery, and data storage.
 * C.2: Use class when invariants exist
 */
class NewsAPICollector {
public:
    /**
     * Constructor
     *
     * @param config NewsAPI configuration
     */
    explicit NewsAPICollector(NewsAPIConfig config)
        : config_(std::move(config))
        , sentiment_analyzer_()
    {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        Logger::getInstance().info("NewsAPI collector initialized");
        Logger::getInstance().info("  Base URL: {}", config_.base_url);
        Logger::getInstance().info("  Daily limit: {}", config_.requests_per_day);
    }

    /**
     * Destructor
     */
    ~NewsAPICollector() {
        curl_global_cleanup();
    }

    // C.21: Rule of Five (prevent copying, delete moving due to CURL global state)
    NewsAPICollector(NewsAPICollector const&) = delete;
    auto operator=(NewsAPICollector const&) -> NewsAPICollector& = delete;
    NewsAPICollector(NewsAPICollector&&) = delete;
    auto operator=(NewsAPICollector&&) -> NewsAPICollector& = delete;

    /**
     * Fetch news for a symbol
     *
     * @param symbol Stock symbol (e.g., "AAPL")
     * @param from_date Start date (ISO 8601 format)
     * @param to_date End date (ISO 8601 format)
     * @return Result containing vector of articles
     *
     * F.20: Return by value
     */
    [[nodiscard]] auto fetchNews(
        std::string const& symbol,
        std::string const& from_date = "",
        std::string const& to_date = ""
    ) -> Result<std::vector<NewsArticle>> {

        Logger::getInstance().info("Fetching news for symbol: {}", symbol);

        // Build API URL
        std::string url = buildAPIUrl(symbol, from_date, to_date);

        // Call API directly
        try {
            auto response = callNewsAPI(url);

            if (!response) {
                return std::unexpected(Error::make(ErrorCode::NetworkError,
                    "Failed to fetch news from API: " + response.error().message));
            }

            // Parse response
            auto articles = parseArticles(response.value(), symbol);

            // Apply sentiment analysis
            for (auto& article : articles) {
                auto sentiment = sentiment_analyzer_.analyze(
                    article.title + " " + article.description
                );

                article.sentiment_score = sentiment.score;
                article.sentiment_label = sentiment.label;
                article.positive_keywords = sentiment.positive_keywords;
                article.negative_keywords = sentiment.negative_keywords;
            }

            Logger::getInstance().info("  Fetched {} articles for {}", articles.size(), symbol);

            return articles;

        } catch (std::exception const& e) {
            Logger::getInstance().error("Error fetching news: {}", e.what());
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                std::string("NewsAPI error: ") + e.what()));
        } catch (...) {
            Logger::getInstance().error("Unknown error fetching news");
            return std::unexpected(Error::make(ErrorCode::UnknownError,
                "Unknown error occurred"));
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
    [[nodiscard]] auto fetchNewsBatch(
        std::vector<std::string> const& symbols,
        std::string const& from_date = "",
        std::string const& to_date = ""
    ) -> Result<std::map<std::string, std::vector<NewsArticle>>> {

        std::map<std::string, std::vector<NewsArticle>> results;

        Logger::getInstance().info("Batch fetching news for {} symbols", symbols.size());

        size_t success_count = 0;
        size_t failure_count = 0;

        for (auto const& symbol : symbols) {
            // Rate limiting: sleep between requests (100 requests/day = ~900ms between calls)
            if (success_count > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }

            auto result = fetchNews(symbol, from_date, to_date);

            if (result) {
                results[symbol] = result.value();
                success_count++;
            } else {
                Logger::getInstance().warn("  Failed to fetch news for {}: {}",
                    symbol, result.error().message);
                failure_count++;
            }
        }

        Logger::getInstance().info("Batch fetch complete: {} succeeded, {} failed",
            success_count, failure_count);

        return results;
    }

    /**
     * Store articles to DuckDB
     *
     * @param articles Vector of articles to store
     * @param db_path Path to DuckDB database
     * @return Result indicating success/failure
     */
    [[nodiscard]] auto storeArticles(
        std::vector<NewsArticle> const& articles,
        std::string const& db_path
    ) -> Result<void> {

        if (articles.empty()) {
            return {};  // Success, nothing to store
        }

        // NOTE: Database storage is handled by Python layer
        // This C++ method provides the interface for future direct storage
        Logger::getInstance().info("Prepared {} articles for storage (handled by Python layer)",
            articles.size());
        Logger::getInstance().debug("  Database path: {}", db_path);

        return {};  // Success
    }

private:
    /**
     * Build NewsAPI URL
     */
    [[nodiscard]] auto buildAPIUrl(
        std::string const& symbol,
        std::string const& from_date,
        std::string const& to_date
    ) const -> std::string {

        std::string url = config_.base_url + "/everything?";
        url += "q=" + urlEncode(symbol);
        url += "&apiKey=" + config_.api_key;
        url += "&language=en";
        url += "&sortBy=publishedAt";
        url += "&pageSize=20";  // Max results per request

        if (!from_date.empty()) {
            url += "&from=" + from_date;
        }

        if (!to_date.empty()) {
            url += "&to=" + to_date;
        }

        return url;
    }

    /**
     * URL encode string
     */
    [[nodiscard]] static auto urlEncode(std::string const& value) -> std::string {
        CURL* curl = curl_easy_init();
        if (curl == nullptr) {
            return value;
        }

        char* encoded = curl_easy_escape(curl, value.c_str(), static_cast<int>(value.length()));
        std::string result(encoded);
        curl_free(encoded);
        curl_easy_cleanup(curl);

        return result;
    }

    /**
     * Call NewsAPI (HTTP GET)
     */
    [[nodiscard]] auto callNewsAPI(std::string const& url) const -> Result<json> {
        CURL* curl = curl_easy_init();
        if (curl == nullptr) {
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                "Failed to initialize CURL"));
        }

        std::string response_data;

        // Callback to capture response
        auto write_callback = [](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
            auto* str = static_cast<std::string*>(userdata);
            str->append(ptr, size * nmemb);
            return size * nmemb;
        };

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.timeout_seconds);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "BigBrotherAnalytics/1.0");

        CURLcode res = curl_easy_perform(curl);

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                std::string("CURL error: ") + curl_easy_strerror(res)));
        }

        if (http_code != 200) {
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                "HTTP error: " + std::to_string(http_code)));
        }

        try {
            return json::parse(response_data);
        } catch (json::exception const& e) {
            return std::unexpected(Error::make(ErrorCode::NetworkError,
                std::string("JSON parse error: ") + e.what()));
        } catch (...) {
            return std::unexpected(Error::make(ErrorCode::UnknownError,
                "Unknown error parsing JSON"));
        }
    }

    /**
     * Parse articles from NewsAPI JSON response
     */
    [[nodiscard]] auto parseArticles(json const& response, std::string const& symbol) const
        -> std::vector<NewsArticle> {

        std::vector<NewsArticle> articles;

        if (!response.contains("articles") || !response["articles"].is_array()) {
            return articles;
        }

        auto now = std::chrono::system_clock::now().time_since_epoch().count();

        for (auto const& item : response["articles"]) {
            NewsArticle article;

            // Generate article ID from URL
            article.article_id = std::to_string(std::hash<std::string>{}(
                item.value("url", "")
            ));

            article.symbol = symbol;
            article.title = item.value("title", "");
            article.description = item.value("description", "");
            article.content = item.value("content", "");
            article.url = item.value("url", "");
            article.author = item.value("author", "");

            // Parse source
            if (item.contains("source") && item["source"].is_object()) {
                article.source_name = item["source"].value("name", "");
                article.source_id = item["source"].value("id", "");
            }

            // Parse published date (ISO 8601 timestamp)
            std::string published_str = item.value("publishedAt", "");
            // TODO: Parse ISO 8601 timestamp properly
            article.published_at = now;  // Simplified for now

            article.fetched_at = now;

            articles.push_back(article);
        }

        return articles;
    }

    NewsAPIConfig config_;
    SentimentAnalyzer sentiment_analyzer_;
};

} // export namespace bigbrother::market_intelligence
