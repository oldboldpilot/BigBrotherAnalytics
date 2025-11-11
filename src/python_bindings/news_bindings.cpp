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

namespace py = pybind11;

using namespace bigbrother::market_intelligence;

// Tagged: PYTHON_BINDINGS
PYBIND11_MODULE(news_ingestion_py, m) {
    m.doc() = "BigBrotherAnalytics News Ingestion - Python Bindings";

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
        .def_readwrite("excluded_sources", &NewsAPIConfig::excluded_sources);

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
            py::arg("db_path"));
        // Note: Circuit breaker removed from C++ implementation

    // ========================================================================
    // Utility Functions
    // ========================================================================

    m.def("analyze_sentiment", [](std::string const& text) -> SentimentResult {
        SentimentAnalyzer analyzer;
        return analyzer.analyze(text);
    }, "Quick sentiment analysis function",
       py::arg("text"));
}
