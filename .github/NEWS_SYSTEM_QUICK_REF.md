# News Ingestion System - Quick Reference Card

**For**: AI Assistants (GitHub Copilot, Claude Code)
**Updated**: 2025-11-10

---

## Module Imports

```cpp
// News ingestion modules
import bigbrother.market_intelligence.sentiment;
import bigbrother.market_intelligence.news;

// Dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.circuit_breaker;
```

---

## Build Commands

```bash
# Build order (sequential)
cd build
cmake -G Ninja ..
ninja utils                    # 1. Core utilities
ninja market_intelligence      # 2. Market intelligence (includes news/sentiment)
ninja news_ingestion_py       # 3. Python bindings

# Verify Python bindings
PYTHONPATH=build:$PYTHONPATH python3 -c "from build import news_ingestion_py; print('OK')"
```

---

## Error Handling

```cpp
// ✅ CORRECT - Always use std::unexpected
return std::unexpected(Error::make(ErrorCode::NetworkError, "Connection timeout"));

// ❌ WRONG - Never use raw Error{}
return Error{"message"};  // Compile error!
```

---

## Trailing Return Types

```cpp
// ✅ CORRECT (clang-tidy passes)
auto analyze(std::string const& text) -> SentimentResult;
[[nodiscard]] auto getScore() const -> double;

// ❌ WRONG (clang-tidy ERROR)
SentimentResult analyze(std::string const& text);
double getScore() const;
```

---

## Python Usage

```bash
# ✅ CORRECT - Always use uv
uv run python scripts/data_collection/news_ingestion.py
uv run streamlit run dashboard/app.py
uv add newsapi-python

# ❌ WRONG
python script.py       # Don't use bare python
pip install package    # Don't use pip
```

---

## Sentiment Analyzer API

```cpp
// C++ API
SentimentAnalyzer analyzer;
auto result = analyzer.analyze("Stock surges on earnings");
// result.score: -1.0 to 1.0
// result.label: "positive" | "negative" | "neutral"
// result.positive_keywords: ["surges"]
```

```python
# Python API
from build import news_ingestion_py

analyzer = news_ingestion_py.SentimentAnalyzer()
result = analyzer.analyze("Stock surges on earnings")
print(f"Score: {result.score}, Label: {result.label}")
```

---

## News Collector API

```cpp
// C++ API
NewsAPIConfig config;
config.api_key = "your_key";
config.requests_per_day = 100;

NewsAPICollector collector(config);
auto articles = collector.fetchNews("AAPL", "2025-11-03", "2025-11-10");
collector.storeArticles(articles, "data/bigbrother.duckdb");
```

```python
# Python API
config = news_ingestion_py.NewsAPIConfig()
config.api_key = "your_key"

collector = news_ingestion_py.NewsAPICollector(config)
articles = collector.fetch_news("AAPL")
collector.store_articles(articles, "data/bigbrother.duckdb")
```

---

## Database Schema

```sql
CREATE TABLE news_articles (
    article_id VARCHAR PRIMARY KEY,    -- MD5(url)
    symbol VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    sentiment_score DOUBLE,            -- -1.0 to 1.0
    sentiment_label VARCHAR,           -- 'positive', 'negative', 'neutral'
    positive_keywords TEXT[],
    negative_keywords TEXT[],
    published_at TIMESTAMP NOT NULL
);

-- Indexes
CREATE INDEX idx_news_symbol ON news_articles(symbol);
CREATE INDEX idx_news_published ON news_articles(published_at DESC);
CREATE INDEX idx_news_sentiment ON news_articles(sentiment_label, sentiment_score);
```

---

## Key Files

```
src/market_intelligence/
├── sentiment_analyzer.cppm    # Sentiment analysis (60+ keywords)
└── news_ingestion.cppm         # NewsAPI client (circuit breaker)

src/python_bindings/
└── news_bindings.cpp            # pybind11 interface

scripts/
├── monitoring/setup_news_database.py
└── data_collection/news_ingestion.py

dashboard/
└── app.py                       # show_news_feed() function
```

---

## Common Operations

```bash
# Setup database (one-time)
uv run python scripts/monitoring/setup_news_database.py

# Fetch news for 10 symbols
uv run python scripts/data_collection/news_ingestion.py

# View in dashboard
uv run streamlit run dashboard/app.py
# Navigate to: "News Feed"

# Query database
duckdb data/bigbrother.duckdb "
    SELECT symbol, COUNT(*) as count
    FROM news_articles
    GROUP BY symbol
    ORDER BY count DESC;
"
```

---

## Troubleshooting

**Module not found**:
```bash
ninja utils market_intelligence  # Build dependencies first
```

**Python import error**:
```bash
export PYTHONPATH=/home/muyiwa/Development/BigBrotherAnalytics/build:$PYTHONPATH
export LD_LIBRARY_PATH=/home/muyiwa/Development/BigBrotherAnalytics/build/lib:$LD_LIBRARY_PATH
```

**clang-tidy errors**:
```cpp
// Fix trailing return types
auto func() -> ReturnType { }  // Not: ReturnType func() { }

// Add [[nodiscard]] on getters
[[nodiscard]] auto getValue() const -> int;
```

---

## Full Documentation

- **AI_CONTEXT.md** - Comprehensive AI assistant context (1050 lines)
- **NEWS_INGESTION_SYSTEM.md** - Complete architecture (526 lines)
- **NEWS_INGESTION_QUICKSTART.md** - 3-step quick start (404 lines)
- **CODING_STANDARDS.md** - C++23 coding standards

---

**Quick Links**:
- Build: `ninja utils market_intelligence news_ingestion_py`
- Run: `uv run python scripts/data_collection/news_ingestion.py`
- View: `uv run streamlit run dashboard/app.py` → "News Feed"
