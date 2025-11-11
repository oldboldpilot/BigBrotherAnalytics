/**
 * BigBrotherAnalytics - News Ingestion Python Bindings
 *
 * pybind11 bindings for C++23 news ingestion and sentiment analysis.
 * Provides Python access to NewsAPI collector and sentiment analyzer.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase 5+: News Ingestion System
 *
 * Tagged: PYTHON_BINDINGS
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

// Import C++23 modules
import bigbrother.market_intelligence.sentiment;
import bigbrother.market_intelligence.news;
import bigbrother.market_intelligence.alphavantage;
import bigbrother.circuit_breaker;

namespace py = pybind11;

using namespace bigbrother::market_intelligence;
using namespace bigbrother::circuit_breaker;

// Tagged: PYTHON_BINDINGS
PYBIND11_MODULE(news_ingestion_py, m) {
    m.doc() = "BigBrotherAnalytics News Ingestion - Python Bindings";

    // ========================================================================
    // Circuit Breaker
    // ========================================================================

    py::enum_<CircuitState>(m, "CircuitState")
        .value("CLOSED", CircuitState::CLOSED, "Normal operation, requests pass through")
        .value("OPEN", CircuitState::OPEN, "Circuit is open, requests fail fast")
        .value("HALF_OPEN", CircuitState::HALF_OPEN, "Testing recovery, allow limited requests")
        .export_values();

    py::class_<CircuitStats>(m, "CircuitStats")
        .def(py::init<>())
        .def_readwrite("state", &CircuitStats::state)
        .def_readwrite("total_calls", &CircuitStats::total_calls)
        .def_readwrite("success_count", &CircuitStats::success_count)
        .def_readwrite("failure_count", &CircuitStats::failure_count)
        .def_readwrite("consecutive_failures", &CircuitStats::consecutive_failures)
        .def_readwrite("last_error", &CircuitStats::last_error)
        .def("get_success_rate", &CircuitStats::getSuccessRate)
        .def("__repr__", [](CircuitStats const& s) {
            return "<CircuitStats state=" + std::to_string(static_cast<int>(s.state)) +
                   " total=" + std::to_string(s.total_calls) +
                   " success=" + std::to_string(s.success_count) +
                   " failures=" + std::to_string(s.failure_count) + ">";
        });

    // ========================================================================
    // Sentiment Analysis
    // ========================================================================

    py::class_<SentimentResult>(m, "SentimentResult")
        .def(py::init<>())
        .def_readwrite("score", &SentimentResult::score)
        .def_readwrite("label", &SentimentResult::label)
        .def_readwrite("positive_keywords", &SentimentResult::positive_keywords)
        .def_readwrite("negative_keywords", &SentimentResult::negative_keywords)
        .def_readwrite("positive_score", &SentimentResult::positive_score)
        .def_readwrite("negative_score", &SentimentResult::negative_score)
        .def_readwrite("total_words", &SentimentResult::total_words)
        .def_readwrite("keyword_density", &SentimentResult::keyword_density)
        .def("__repr__", [](SentimentResult const& r) {
            return "<SentimentResult score=" + std::to_string(r.score) +
                   " label='" + r.label + "'>";
        });

    py::class_<SentimentAnalyzer>(m, "SentimentAnalyzer")
        .def(py::init<>())
        .def("analyze", &SentimentAnalyzer::analyze,
            "Analyze sentiment of text",
            py::arg("text"))
        .def("analyze_batch", &SentimentAnalyzer::analyzeBatch,
            "Analyze sentiment of multiple texts",
            py::arg("texts"));

    // ========================================================================
    // News Ingestion
    // ========================================================================

    py::enum_<SourceQuality>(m, "SourceQuality")
        .value("All", SourceQuality::All, "No filtering - accept all sources")
        .value("Premium", SourceQuality::Premium, "WSJ, Bloomberg, Reuters, FT, etc.")
        .value("Verified", SourceQuality::Verified, "Major news outlets with editorial standards")
        .value("Exclude", SourceQuality::Exclude, "Explicitly excluded sources")
        .export_values();

    py::class_<NewsArticle>(m, "NewsArticle")
        .def(py::init<>())
        .def_readwrite("article_id", &NewsArticle::article_id)
        .def_readwrite("symbol", &NewsArticle::symbol)
        .def_readwrite("title", &NewsArticle::title)
        .def_readwrite("description", &NewsArticle::description)
        .def_readwrite("content", &NewsArticle::content)
        .def_readwrite("url", &NewsArticle::url)
        .def_readwrite("source_name", &NewsArticle::source_name)
        .def_readwrite("source_id", &NewsArticle::source_id)
        .def_readwrite("author", &NewsArticle::author)
        .def_readwrite("published_at", &NewsArticle::published_at)
        .def_readwrite("fetched_at", &NewsArticle::fetched_at)
        .def_readwrite("sentiment_score", &NewsArticle::sentiment_score)
        .def_readwrite("sentiment_label", &NewsArticle::sentiment_label)
        .def_readwrite("positive_keywords", &NewsArticle::positive_keywords)
        .def_readwrite("negative_keywords", &NewsArticle::negative_keywords)
        .def("__repr__", [](NewsArticle const& a) {
            return "<NewsArticle symbol='" + a.symbol +
                   "' title='" + a.title.substr(0, 50) + "...'>";
        });

    py::class_<NewsAPIConfig>(m, "NewsAPIConfig")
        .def(py::init<>())
        .def_readwrite("api_key", &NewsAPIConfig::api_key)
        .def_readwrite("base_url", &NewsAPIConfig::base_url)
        .def_readwrite("requests_per_day", &NewsAPIConfig::requests_per_day)
        .def_readwrite("lookback_days", &NewsAPIConfig::lookback_days)
        .def_readwrite("timeout_seconds", &NewsAPIConfig::timeout_seconds)
        .def_readwrite("quality_filter", &NewsAPIConfig::quality_filter)
        .def_readwrite("preferred_sources", &NewsAPIConfig::preferred_sources)
        .def_readwrite("excluded_sources", &NewsAPIConfig::excluded_sources)
        .def_readwrite("circuit_breaker_failure_threshold", &NewsAPIConfig::circuit_breaker_failure_threshold)
        .def_readwrite("circuit_breaker_timeout_seconds", &NewsAPIConfig::circuit_breaker_timeout_seconds)
        .def_readwrite("circuit_breaker_half_open_timeout_seconds", &NewsAPIConfig::circuit_breaker_half_open_timeout_seconds)
        .def_readwrite("circuit_breaker_half_open_max_calls", &NewsAPIConfig::circuit_breaker_half_open_max_calls);

    py::class_<NewsAPICollector>(m, "NewsAPICollector")
        .def(py::init<NewsAPIConfig>())
        .def("fetch_news", &NewsAPICollector::fetchNews,
            "Fetch news for a symbol",
            py::arg("symbol"),
            py::arg("from_date") = "",
            py::arg("to_date") = "")
        .def("fetch_news_batch", &NewsAPICollector::fetchNewsBatch,
            "Fetch news for multiple symbols with rate limiting",
            py::arg("symbols"),
            py::arg("from_date") = "",
            py::arg("to_date") = "")
        .def("store_articles", &NewsAPICollector::storeArticles,
            "Store articles to DuckDB",
            py::arg("articles"),
            py::arg("db_path"))
        .def("get_circuit_breaker_state", &NewsAPICollector::getCircuitBreakerState,
            "Get current circuit breaker state")
        .def("get_circuit_breaker_stats", &NewsAPICollector::getCircuitBreakerStats,
            "Get circuit breaker statistics")
        .def("is_circuit_breaker_open", &NewsAPICollector::isCircuitBreakerOpen,
            "Check if circuit breaker is open")
        .def("reset_circuit_breaker", &NewsAPICollector::resetCircuitBreaker,
            "Manually reset circuit breaker to CLOSED state");

    // ========================================================================
    // Utility Functions
    // ========================================================================

    m.def("analyze_sentiment", [](std::string const& text) -> SentimentResult {
        SentimentAnalyzer analyzer;
        return analyzer.analyze(text);
    }, "Quick sentiment analysis function",
       py::arg("text"));

    // ========================================================================
    // AlphaVantage News
    // ========================================================================

    py::class_<TickerSentiment>(m, "TickerSentiment")
        .def(py::init<>())
        .def_readwrite("ticker", &TickerSentiment::ticker)
        .def_readwrite("sentiment_score", &TickerSentiment::sentiment_score)
        .def_readwrite("sentiment_label", &TickerSentiment::sentiment_label)
        .def_readwrite("relevance_score", &TickerSentiment::relevance_score)
        .def("__repr__", [](TickerSentiment const& ts) {
            return "<TickerSentiment ticker='" + ts.ticker +
                   "' score=" + std::to_string(ts.sentiment_score) +
                   " label='" + ts.sentiment_label + "'>";
        });

    py::class_<AlphaVantageArticle>(m, "AlphaVantageArticle")
        .def(py::init<>())
        .def_readwrite("title", &AlphaVantageArticle::title)
        .def_readwrite("url", &AlphaVantageArticle::url)
        .def_readwrite("time_published", &AlphaVantageArticle::time_published)
        .def_readwrite("authors", &AlphaVantageArticle::authors)
        .def_readwrite("summary", &AlphaVantageArticle::summary)
        .def_readwrite("banner_image", &AlphaVantageArticle::banner_image)
        .def_readwrite("source", &AlphaVantageArticle::source)
        .def_readwrite("category_within_source", &AlphaVantageArticle::category_within_source)
        .def_readwrite("source_domain", &AlphaVantageArticle::source_domain)
        .def_readwrite("ticker_sentiment", &AlphaVantageArticle::ticker_sentiment)
        .def_readwrite("overall_sentiment_score", &AlphaVantageArticle::overall_sentiment_score)
        .def_readwrite("overall_sentiment_label", &AlphaVantageArticle::overall_sentiment_label)
        .def_readwrite("topics", &AlphaVantageArticle::topics)
        .def("__repr__", [](AlphaVantageArticle const& a) {
            return "<AlphaVantageArticle title='" + a.title.substr(0, 50) +
                   "...' sentiment=" + std::to_string(a.overall_sentiment_score) + ">";
        });

    py::class_<AlphaVantageConfig>(m, "AlphaVantageConfig")
        .def(py::init<>())
        .def_readwrite("api_key", &AlphaVantageConfig::api_key)
        .def_readwrite("base_url", &AlphaVantageConfig::base_url)
        .def_readwrite("timeout_seconds", &AlphaVantageConfig::timeout_seconds)
        .def_readwrite("lookback_days", &AlphaVantageConfig::lookback_days)
        .def_readwrite("max_results_per_symbol", &AlphaVantageConfig::max_results_per_symbol)
        .def_readwrite("circuit_breaker_failure_threshold", &AlphaVantageConfig::circuit_breaker_failure_threshold)
        .def_readwrite("circuit_breaker_timeout_seconds", &AlphaVantageConfig::circuit_breaker_timeout_seconds)
        .def_readwrite("circuit_breaker_half_open_timeout_seconds", &AlphaVantageConfig::circuit_breaker_half_open_timeout_seconds)
        .def_readwrite("circuit_breaker_half_open_max_calls", &AlphaVantageConfig::circuit_breaker_half_open_max_calls);

    py::class_<AlphaVantageCollector>(m, "AlphaVantageCollector")
        .def(py::init<AlphaVantageConfig>())
        .def("fetch_news", &AlphaVantageCollector::fetchNews,
            "Fetch news for a symbol from AlphaVantage",
            py::arg("symbol"),
            py::arg("from_date") = "",
            py::arg("to_date") = "")
        .def("fetch_news_batch", &AlphaVantageCollector::fetchNewsBatch,
            "Fetch news for multiple symbols from AlphaVantage",
            py::arg("symbols"),
            py::arg("from_date") = "",
            py::arg("to_date") = "")
        .def("get_circuit_breaker_state", &AlphaVantageCollector::getCircuitBreakerState,
            "Get current circuit breaker state")
        .def("get_circuit_breaker_stats", &AlphaVantageCollector::getCircuitBreakerStats,
            "Get circuit breaker statistics")
        .def("is_circuit_breaker_open", &AlphaVantageCollector::isCircuitBreakerOpen,
            "Check if circuit breaker is open")
        .def("reset_circuit_breaker", &AlphaVantageCollector::resetCircuitBreaker,
            "Manually reset circuit breaker to CLOSED state");
}
