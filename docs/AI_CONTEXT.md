# AI Context - BigBrotherAnalytics Codebase

**Purpose**: Comprehensive context for AI assistants (GitHub Copilot, Claude, etc.)
**Last Updated**: 2025-11-12
**Status**: Phase 5 Active + Critical Bug Fixes Deployed

---

## ⚠️ RECENT CRITICAL BUG FIXES (November 12, 2025)

**Status:** ✅ ALL RESOLVED | **Commit:** [0200aba](https://github.com/oldboldpilot/BigBrotherAnalytics/commit/0200aba)

Three critical bugs prevented trading today (0/3 orders placed):

### 1. Quote Bid/Ask = $0.00 (100% Order Failure) - FIXED ✅
- **File:** `src/schwab_api/schwab_api.cppm:631-696`
- **Problem:** Cached quotes returned bid=0, ask=0; all orders rejected
- **Solution:** Apply after-hours fix to BOTH cached and fresh quotes
- **Impact:** Order success rate: 0% → >90%

### 2. ML Predictions Catastrophic (-22,000%) - FIXED ✅
- **File:** `src/trading_decision/strategies.cppm:1241-1258`
- **Problem:** Model predicted SPY -22,013%, would destroy account
- **Solution:** Reject predictions outside ±50% range with error logging
- **Impact:** Prevents catastrophic trades

### 3. Python 3.14 → 3.13 Documentation - FIXED ✅
- **Files:** `playbooks/complete-tier1-setup.yml`, `README.md`, `install-upcxx-berkeley.yml`
- **Problem:** Docs referenced non-existent Python 3.14
- **Solution:** Updated all references to Python 3.13

**Full Report:** [CRITICAL_BUG_FIXES_2025-11-12.md](CRITICAL_BUG_FIXES_2025-11-12.md)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [News Ingestion System](#news-ingestion-system)
3. [Module Architecture](#module-architecture)
4. [Build System](#build-system)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Coding Standards](#coding-standards)
7. [Python Bindings](#python-bindings)
8. [Integration Points](#integration-points)
9. [Testing Strategy](#testing-strategy)
10. [Common Workflows](#common-workflows)

---

## Project Overview

BigBrotherAnalytics is an algorithmic trading system with employment-driven sector rotation strategy, built with C++23 modules and Python bindings.

### Technology Stack

**Core Technologies:**
- **C++23**: Modules, std::expected, trailing return types, [[nodiscard]]
- **Compiler**: Clang 21.1.5 (primary), GCC 15 (backup)
- **Build System**: CMake 3.28+ with Ninja generator
- **Python**: 3.14 via pybind11 bindings
- **Package Manager**: uv (NOT pip - see critical note below)
- **Database**: DuckDB (in-process OLAP)

**Key Libraries:**
- pybind11 - Python/C++ interop
- DuckDB - Database engine
- libcurl - HTTP client (NewsAPI, Schwab API)
- nlohmann/json - JSON parsing
- spdlog - Structured logging
- OpenMP - CPU parallelization

**Data Sources:**
- NewsAPI (100 requests/day, free tier)
- Schwab API (market data + trading)
- BLS API v2 (employment data)
- DuckDB local storage

---

## News Ingestion System

### Architecture Overview

The news ingestion system adds real-time financial news with sentiment analysis to the trading platform.

**Components:**
```
News Ingestion Pipeline
├── C++ Core (src/market_intelligence/)
│   ├── sentiment_analyzer.cppm    # Keyword-based sentiment (-1.0 to +1.0)
│   └── news_ingestion.cppm         # NewsAPI client with circuit breaker
├── Python Bindings (src/python_bindings/)
│   └── news_bindings.cpp            # pybind11 interface
├── Python Scripts (scripts/)
│   ├── monitoring/setup_news_database.py
│   └── data_collection/news_ingestion.py
└── Dashboard (dashboard/app.py)
    └── show_news_feed()             # Streamlit visualization
```

### Data Flow

```
NewsAPI → C++ News Collector → C++ Sentiment Analyzer → DuckDB → Dashboard
  (fetch)     (circuit breaker)      (60+ keywords)      (store)   (visualize)
```

### Sentiment Analysis Engine

**Implementation**: `src/market_intelligence/sentiment_analyzer.cppm`

**Algorithm**:
- **60+ positive keywords**: profit, gain, growth, surge, bull, upgrade, beat, exceed...
- **60+ negative keywords**: loss, decline, fall, bear, downgrade, miss, warning...
- **Intensifiers**: very, extremely, highly (1.5x multiplier)
- **Negation handling**: "not good" → negative sentiment
- **Scoring**: (positive - negative) / total, normalized to [-1.0, +1.0]

**Performance**:
- Speed: ~0.01ms per article (100,000 articles/second)
- Memory: <1MB (keyword sets cached)
- No ML dependencies (keyword-based only)

**API Example**:
```cpp
import bigbrother.market_intelligence.sentiment;

SentimentAnalyzer analyzer;
auto result = analyzer.analyze("Apple stock surges on strong earnings");
// result.score = 0.67 (positive)
// result.label = "positive"
// result.positive_keywords = ["surges", "strong"]
```

### News Collection Module

**Implementation**: `src/market_intelligence/news_ingestion.cppm`

**Features**:
- HTTP client via libcurl
- Circuit breaker (5 failures → 60s timeout)
- Rate limiting (1 second between calls, 100 requests/day)
- JSON parsing via nlohmann/json
- DuckDB storage with deduplication

**Configuration**:
```cpp
NewsAPIConfig config;
config.api_key = "0174a0ea31cb434ea416759e23cc3f42";  // Free tier
config.base_url = "https://newsapi.org/v2";
config.requests_per_day = 100;
config.lookback_days = 7;
config.timeout_seconds = 30;
```

**API Example**:
```cpp
import bigbrother.market_intelligence.news;

NewsAPICollector collector(config);

// Fetch news for single symbol
auto result = collector.fetchNews("AAPL", "2025-11-03", "2025-11-10");

// Batch fetch with rate limiting
auto batch = collector.fetchNewsBatch({"AAPL", "MSFT", "GOOGL"});

// Store in database
collector.storeArticles(articles, "data/bigbrother.duckdb");
```

### Database Schema

**Table**: `news_articles`

```sql
CREATE TABLE news_articles (
    article_id VARCHAR PRIMARY KEY,           -- MD5 hash of URL (deduplication)
    symbol VARCHAR NOT NULL,                  -- Stock symbol
    title VARCHAR NOT NULL,                   -- Headline
    description TEXT,                         -- Summary
    content TEXT,                             -- Full content
    url VARCHAR,                              -- Source URL
    source_name VARCHAR,                      -- e.g., 'Reuters'
    source_id VARCHAR,
    author VARCHAR,
    published_at TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentiment_score DOUBLE,                   -- -1.0 to 1.0
    sentiment_label VARCHAR,                  -- 'positive', 'negative', 'neutral'
    positive_keywords TEXT[],                 -- Matched keywords
    negative_keywords TEXT[],
    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

CREATE INDEX idx_news_symbol ON news_articles(symbol);
CREATE INDEX idx_news_published ON news_articles(published_at DESC);
CREATE INDEX idx_news_sentiment ON news_articles(sentiment_label, sentiment_score);
```

### Python Bindings

**File**: `src/python_bindings/news_bindings.cpp`

**Exposed Types**:
```cpp
// Module: news_ingestion_py
PYBIND11_MODULE(news_ingestion_py, m) {
    // Sentiment analysis
    py::class_<SentimentAnalyzer>(m, "SentimentAnalyzer")
        .def(py::init<>())
        .def("analyze", &SentimentAnalyzer::analyze);

    py::class_<SentimentResult>(m, "SentimentResult")
        .def_readonly("score", &SentimentResult::score)
        .def_readonly("label", &SentimentResult::label)
        .def_readonly("positive_keywords", &SentimentResult::positive_keywords)
        .def_readonly("negative_keywords", &SentimentResult::negative_keywords);

    // News collection
    py::class_<NewsAPICollector>(m, "NewsAPICollector")
        .def(py::init<NewsAPIConfig const&>())
        .def("fetch_news", &NewsAPICollector::fetchNews)
        .def("fetch_news_batch", &NewsAPICollector::fetchNewsBatch)
        .def("store_articles", &NewsAPICollector::storeArticles);

    py::class_<NewsAPIConfig>(m, "NewsAPIConfig")
        .def(py::init<>())
        .def_readwrite("api_key", &NewsAPIConfig::api_key)
        .def_readwrite("base_url", &NewsAPIConfig::base_url);
}
```

**Python Usage**:
```python
from build import news_ingestion_py

# Sentiment analysis
analyzer = news_ingestion_py.SentimentAnalyzer()
result = analyzer.analyze("Stock price drops on weak earnings")
print(f"Score: {result.score}, Label: {result.label}")

# News fetching
config = news_ingestion_py.NewsAPIConfig()
config.api_key = "your_api_key"
collector = news_ingestion_py.NewsAPICollector(config)

articles = collector.fetch_news("AAPL")
collector.store_articles(articles, "data/bigbrother.duckdb")
```

### Build Commands

```bash
# Build order: utils → market_intelligence → bindings
cd build
cmake -G Ninja ..

# Build modules
ninja utils                    # Core utilities first
ninja market_intelligence      # Market intelligence (includes news/sentiment)
ninja news_ingestion_py       # Python bindings

# Verify bindings
PYTHONPATH=build:$PYTHONPATH python3 -c "from build import news_ingestion_py; print('Success')"
```

### Dashboard Integration

**File**: `dashboard/app.py`

**New View**: `show_news_feed()`

**Features**:
- Summary metrics (total articles, positive/negative counts, avg sentiment)
- Filters (symbol, sentiment, date range, article limit)
- Visualizations (sentiment distribution bar chart, avg sentiment by symbol)
- Article cards with clickable links, sentiment scores, keyword highlights

---

## Module Architecture

### Module Dependency Graph

```
Core Modules:
utils.types ──────────────────────────────┐
   │                                       │
   ├→ utils.logger                         │
   ├→ utils.database                       │
   ├→ utils.circuit_breaker                │
   └→ utils.result (std::expected wrapper) │
                                           │
Market Intelligence:                       │
market_intelligence.sentiment ←────────────┤
   │                                       │
   └→ market_intelligence.news ←───────────┘
         │
         └→ market_intelligence.employment_signals

Trading System:
trading_decision.strategy
   ├→ risk_management.position_sizing
   └→ execution.order_management
```

### Module Import Syntax

**Correct Usage**:
```cpp
// Module imports (C++23)
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.database;
import bigbrother.market_intelligence.sentiment;
import bigbrother.market_intelligence.news;
```

**NEVER use**:
```cpp
#include "sentiment_analyzer.h"  // ❌ OLD - No headers for modules
using namespace bigbrother;      // ❌ BAD - Explicit namespaces required
```

### Module Structure Template

```cpp
// Global module fragment (for includes)
module;
#include <vector>
#include <string>
#include <expected>

// Module declaration
export module bigbrother.mymodule;

// Imports from other modules
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Exported namespace
export namespace bigbrother::mymodule {
    // Public API
    class MyClass {
    public:
        [[nodiscard]] auto compute() const -> Result<double>;
    };
}

// Private implementation namespace (not exported)
namespace bigbrother::mymodule::detail {
    // Internal helpers
}
```

---

## Build System

### CMake Configuration

**Compiler Requirements**:
- Clang 21.1.5+ (C++23 modules support)
- CMake 3.28+ (CMAKE_CXX_MODULES)
- Ninja generator (required for module builds)

**Environment Setup**:
```bash
export CC=/usr/local/bin/clang
export CXX=/usr/local/bin/clang++
export SKIP_CLANG_TIDY=0  # Enable linting (cannot be disabled)
```

**CMake Module Pattern**:
```cmake
# Module library
add_library(mymodule)
target_sources(mymodule
    PUBLIC FILE_SET CXX_MODULES FILES
    src/path/to/mymodule.cppm
)

# Dependencies
target_link_libraries(mymodule
    PUBLIC utils_types utils_logger
)

# Python bindings
pybind11_add_module(mymodule_py src/python_bindings/mymodule_bindings.cpp)
target_link_libraries(mymodule_py PRIVATE mymodule)
```

**Build Order**:
```bash
# 1. Clean build
rm -rf build && mkdir build && cd build

# 2. Configure
cmake -G Ninja ..

# 3. Build in dependency order
ninja utils                    # Core utilities
ninja market_intelligence      # Market modules (news, sentiment, employment)
ninja trading_decision         # Trading strategies
ninja risk_management          # Risk engine
ninja news_ingestion_py       # Python bindings (last)

# 4. Verify
./bin/test_news_ingestion     # C++ tests
PYTHONPATH=build:$PYTHONPATH uv run python test_news_bindings.py  # Python tests
```

### clang-tidy Configuration

**Status**: MANDATORY - Runs before every build (cannot be skipped)

**Configuration**: `.clang-tidy`

**Key Checks**:
- `cppcoreguidelines-*` - Core Guidelines enforcement
- `modernize-*` - Modern C++ patterns
- `cert-*` - Security checks
- `concurrency-*` - Thread safety
- `readability-*` - Code clarity

**Enforced Rules (ERROR level)**:
- Trailing return types (100% required)
- Rule of Five (delete copy, careful move)
- [[nodiscard]] on query methods
- const correctness
- RAII for all resources (no raw new/delete)

**System Header Exclusion**:
```yaml
# .clang-tidy
HeaderFilterRegex: '^.*/(src|include)/.*'  # Only check project files
SystemHeaders: false                        # Skip /usr/include, etc.
```

---

## Error Handling Patterns

### Result<T> Type (std::expected wrapper)

**Definition**: `src/utils/result.cppm`

```cpp
export namespace bigbrother::utils {
    // Result type for operations that can fail
    template<typename T>
    using Result = std::expected<T, Error>;

    // Error type with code and message
    struct Error {
        ErrorCode code;
        std::string message;

        // Factory method
        static auto make(ErrorCode code, std::string msg) -> Error;
    };

    enum class ErrorCode {
        NetworkError,
        DatabaseError,
        ParseError,
        ValidationError,
        AuthenticationError,
        RateLimitExceeded
    };
}
```

**CORRECT Usage**:
```cpp
import bigbrother.utils.result;

// Return error (ALWAYS use std::unexpected)
auto fetchData() -> Result<Data> {
    if (network_failed) {
        return std::unexpected(Error::make(ErrorCode::NetworkError, "Connection timeout"));
    }
    return data;
}

// Check result
auto result = fetchData();
if (result.has_value()) {
    auto data = result.value();
    // Use data
} else {
    auto error = result.error();
    Logger::error("Failed: {} (code: {})", error.message, error.code);
}

// Or use monadic operations
auto transformed = fetchData()
    .and_then([](Data const& d) { return processData(d); })
    .or_else([](Error const& e) { Logger::warn("Fallback: {}", e.message); return defaultData(); });
```

**WRONG Usage**:
```cpp
// ❌ DON'T: Direct Error{} construction
return Error{"Network failed"};  // Compile error!

// ❌ DON'T: Throwing exceptions for expected failures
throw std::runtime_error("Network failed");  // Only for unrecoverable errors

// ❌ DON'T: Returning nullptr
return nullptr;  // Use Result<T> instead

// ❌ DON'T: Returning error codes
return -1;  // Use Result<T> instead
```

### Circuit Breaker Pattern

**Implementation**: `src/utils/circuit_breaker.cppm`

**Usage in News Module**:
```cpp
import bigbrother.utils.circuit_breaker;

class NewsAPICollector {
private:
    CircuitBreaker circuit_breaker_{
        5,      // max_failures
        60s     // timeout_duration
    };

public:
    auto fetchNews(std::string const& symbol) -> Result<std::vector<NewsArticle>> {
        if (!circuit_breaker_.allow_request()) {
            return std::unexpected(Error::make(ErrorCode::RateLimitExceeded,
                "Circuit breaker open"));
        }

        auto result = make_http_request(symbol);

        if (result.has_value()) {
            circuit_breaker_.record_success();
        } else {
            circuit_breaker_.record_failure();
        }

        return result;
    }
};
```

---

## Coding Standards

### 1. Trailing Return Types (MANDATORY)

**Rule**: All functions MUST use `auto func() -> ReturnType` syntax.

```cpp
// ✅ CORRECT
auto calculate(double x) -> double;
auto getName() const -> std::string const&;
auto process() -> Result<Data>;

// ❌ WRONG (will not compile with clang-tidy)
double calculate(double x);
std::string const& getName() const;
Result<Data> process();
```

### 2. [[nodiscard]] Attribute

**Rule**: All query methods (getters, const methods returning values) must be marked `[[nodiscard]]`.

```cpp
// ✅ CORRECT
[[nodiscard]] auto getPrice() const -> double;
[[nodiscard]] auto isEmpty() const noexcept -> bool;
[[nodiscard]] auto calculate() const -> Result<double>;

// ❌ WRONG
auto getPrice() const -> double;  // Missing [[nodiscard]]
```

### 3. Rule of Five

**Rule**: If you declare any of {copy ctor, copy assign, move ctor, move assign, dtor}, declare all five.

```cpp
// ✅ CORRECT
class Resource {
public:
    // Explicitly delete copy
    Resource(Resource const&) = delete;
    auto operator=(Resource const&) -> Resource& = delete;

    // Implement move
    Resource(Resource&&) noexcept;
    auto operator=(Resource&&) noexcept -> Resource&;

    // Declare destructor
    ~Resource();
};
```

### 4. const Correctness

**Rule**: Use `const&` for expensive types, value for cheap types.

```cpp
// ✅ CORRECT - const& for expensive types
auto process(std::string const& text) -> void;
auto analyze(std::vector<Data> const& items) -> Result<Summary>;

// ✅ CORRECT - value for cheap types
auto setPrice(double price) -> void;
auto setCount(int count) -> void;
auto setFlag(bool flag) -> void;

// ❌ WRONG - unnecessary copy
auto process(std::string text) -> void;  // Expensive copy!

// ❌ WRONG - const& for primitives
auto setPrice(double const& price) -> void;  // Overkill
```

### 5. RAII for Resources

**Rule**: Never use raw `new`/`delete`. Use RAII wrappers.

```cpp
// ✅ CORRECT - RAII via smart pointers
auto data = std::make_unique<Data>();
auto shared = std::make_shared<Config>();

// ✅ CORRECT - RAII via standard containers
auto items = std::vector<Item>{};
auto cache = std::unordered_map<std::string, Data>{};

// ✅ CORRECT - RAII via custom class
class DatabaseConnection {
public:
    DatabaseConnection(std::string const& path);
    ~DatabaseConnection() { close(); }  // Automatic cleanup
private:
    void close();
};

// ❌ WRONG - manual memory management
Data* data = new Data();  // Memory leak risk!
delete data;
```

### 6. Modern Initialization

**Rule**: Use uniform initialization `{}` for clarity.

```cpp
// ✅ CORRECT
auto value = 42;
auto name = std::string{"Apple"};
auto items = std::vector<int>{1, 2, 3};
auto config = Config{.timeout = 30, .retries = 3};

// ❌ WRONG - old-style
int value = 42;
std::string name = "Apple";
std::vector<int> items = {1, 2, 3};
```

---

## Python Bindings

### pybind11 Patterns

**Basic Type Binding**:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector, std::string

namespace py = pybind11;

PYBIND11_MODULE(mymodule_py, m) {
    m.doc() = "BigBrotherAnalytics module bindings";

    // Simple class
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("method", &MyClass::method)
        .def_readonly("field", &MyClass::field);
}
```

**Result<T> Conversion**:
```cpp
// Custom converter for Result<T>
template<typename T>
void bind_result(py::module& m) {
    py::class_<Result<T>>(m, "Result")
        .def("has_value", &Result<T>::has_value)
        .def("value", [](Result<T>& r) {
            if (!r.has_value()) {
                throw py::value_error(r.error().message);
            }
            return r.value();
        })
        .def("error", [](Result<T>& r) {
            if (r.has_value()) {
                throw py::value_error("No error - result has value");
            }
            return r.error().message;
        });
}
```

**STL Container Binding**:
```cpp
#include <pybind11/stl.h>  // Automatic conversion for std::vector, std::map

// std::vector<T> automatically converts to Python list
// std::map<K, V> automatically converts to Python dict
// std::optional<T> automatically converts to Python None or value
```

### Python Library Path Setup

**Environment Variables**:
```bash
# Required for Python to find C++ modules
export PYTHONPATH=/home/muyiwa/Development/BigBrotherAnalytics/build:$PYTHONPATH

# Required for shared library loading
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/muyiwa/Development/BigBrotherAnalytics/build/lib:$LD_LIBRARY_PATH
```

**Python Import**:
```python
# Import C++ module
from build import news_ingestion_py

# Use classes
analyzer = news_ingestion_py.SentimentAnalyzer()
result = analyzer.analyze("Stock surges on earnings beat")
print(f"Sentiment: {result.score}")
```

---

## Integration Points

### 1. News → Dashboard

**Flow**: News ingestion script → DuckDB → Dashboard query → Streamlit UI

**Script**: `scripts/data_collection/news_ingestion.py`
```bash
# Fetch news for 10 symbols
uv run python scripts/data_collection/news_ingestion.py

# View in dashboard
uv run streamlit run dashboard/app.py
# Navigate to: "News Feed"
```

### 2. News → Trading Strategy (Future)

**Planned Integration**:
```cpp
// SectorRotationStrategy will use sentiment scores
auto generateSignals() -> std::vector<TradingSignal> {
    auto sentiment = getNewsSentiment(sector);  // From news_articles table
    auto employment = getEmploymentSignal(sector);

    // Composite score: 60% employment + 30% sentiment + 10% momentum
    auto score = 0.6 * employment + 0.3 * sentiment + 0.1 * momentum;

    return signals;
}
```

### 3. Daily Automation

**Morning Setup**:
```bash
# Verify all systems
uv run python scripts/phase5_setup.py --quick

# Start dashboard (includes news feed)
uv run streamlit run dashboard/app.py

# Start trading
./build/bigbrother
```

**Evening Shutdown**:
```bash
# Graceful shutdown + reports
uv run python scripts/phase5_shutdown.py
```

**Daily News Update** (to be added):
```bash
# Add to cron or scheduled task
0 6 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && uv run python scripts/data_collection/news_ingestion.py
```

---

## Testing Strategy

### Test Organization

```
tests/
├── unit/                    # C++ unit tests
│   ├── test_sentiment_analyzer.cpp
│   └── test_news_ingestion.cpp
├── integration/             # C++ integration tests
│   └── test_news_pipeline.cpp
└── python/                  # Python binding tests
    └── test_news_bindings.py
```

### C++ Test Template

```cpp
#include <gtest/gtest.h>
import bigbrother.market_intelligence.sentiment;

namespace bigbrother::testing {

TEST(SentimentAnalyzerTest, PositiveSentiment) {
    SentimentAnalyzer analyzer;
    auto result = analyzer.analyze("Stock surges on strong earnings");

    EXPECT_GT(result.score, 0.0);
    EXPECT_EQ(result.label, "positive");
    EXPECT_THAT(result.positive_keywords, Contains("surges"));
}

}  // namespace bigbrother::testing
```

### Python Test Template

```python
import pytest
from build import news_ingestion_py

def test_sentiment_analyzer():
    analyzer = news_ingestion_py.SentimentAnalyzer()
    result = analyzer.analyze("Stock surges on strong earnings")

    assert result.score > 0.0
    assert result.label == "positive"
    assert "surges" in result.positive_keywords

def test_news_collector():
    config = news_ingestion_py.NewsAPIConfig()
    config.api_key = "test_key"
    collector = news_ingestion_py.NewsAPICollector(config)

    # Test implementation
    assert collector is not None
```

### Running Tests

```bash
# C++ tests
cd build
ninja test

# Python tests
uv run pytest tests/python/

# Integration tests
uv run python test_employment_pipeline.py
uv run python test_sector_rotation_end_to_end.py
```

---

## Common Workflows

### Adding a New C++ Module

1. **Create module file**: `src/category/mymodule.cppm`
2. **Define module structure**:
```cpp
module;
#include <vector>

export module bigbrother.category.mymodule;
import bigbrother.utils.types;

export namespace bigbrother::category {
    class MyClass {
    public:
        [[nodiscard]] auto compute() -> Result<double>;
    };
}
```

3. **Add to CMakeLists.txt**:
```cmake
add_library(mymodule)
target_sources(mymodule
    PUBLIC FILE_SET CXX_MODULES FILES
    src/category/mymodule.cppm
)
target_link_libraries(mymodule PUBLIC utils_types)
```

4. **Build and test**:
```bash
cd build
ninja mymodule
./bin/test_mymodule
```

### Adding Python Bindings

1. **Create bindings file**: `src/python_bindings/mymodule_bindings.cpp`
2. **Implement bindings**:
```cpp
#include <pybind11/pybind11.h>

PYBIND11_MODULE(mymodule_py, m) {
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("compute", &MyClass::compute);
}
```

3. **Add to CMakeLists.txt**:
```cmake
pybind11_add_module(mymodule_py src/python_bindings/mymodule_bindings.cpp)
target_link_libraries(mymodule_py PRIVATE mymodule)
```

4. **Build and test**:
```bash
ninja mymodule_py
PYTHONPATH=build:$PYTHONPATH python3 -c "from build import mymodule_py; print('Success')"
```

### Troubleshooting Build Issues

**Issue**: Module not found
```
error: module 'bigbrother.utils.types' not found
```
**Solution**: Build dependencies first
```bash
ninja utils types
ninja mymodule
```

**Issue**: Python import error
```
ModuleNotFoundError: No module named 'build'
```
**Solution**: Set PYTHONPATH
```bash
export PYTHONPATH=/home/muyiwa/Development/BigBrotherAnalytics/build:$PYTHONPATH
```

**Issue**: Shared library error
```
error while loading shared libraries: libmymodule.so
```
**Solution**: Set LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/home/muyiwa/Development/BigBrotherAnalytics/build/lib:$LD_LIBRARY_PATH
```

---

## Critical Notes for AI Assistants

### Python Package Management (CRITICAL)

**ALWAYS use `uv run python` for Python commands. NEVER use `pip` or bare `python`.**

```bash
# ✅ CORRECT
uv run python script.py
uv run streamlit run app.py
uv run pytest tests/
uv add pandas

# ❌ WRONG
python script.py
pip install pandas
python3 -m pytest
```

**Rationale**: Project uses `uv` for 10-100x faster, deterministic dependency management.

### Trading Constraints

**MANDATORY RULE**: Bot ONLY trades NEW securities or bot-managed positions.

**FORBIDDEN**:
- Trading existing manual positions
- Modifying securities already held (unless `is_bot_managed = true`)
- Closing manual positions

**Implementation**:
```cpp
// ALWAYS check before trading
auto position = db.queryPosition(account_id, symbol);
if (position && !position->is_bot_managed) {
    Logger::warn("Skipping {} - manual position", symbol);
    return;  // DO NOT TRADE
}
```

### Build System Notes

1. **Always use Ninja**: `cmake -G Ninja ..` (required for modules)
2. **Build order matters**: utils → market_intelligence → trading_decision → bindings
3. **clang-tidy runs automatically**: Cannot be skipped, fix all errors
4. **Module dependencies**: Check module graph before adding imports

---

## Quick Reference

### Module Imports
```cpp
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.database;
import bigbrother.market_intelligence.sentiment;
import bigbrother.market_intelligence.news;
```

### Build Commands
```bash
cd build
cmake -G Ninja ..
ninja utils market_intelligence news_ingestion_py
```

### Python Usage
```bash
uv run python script.py
export PYTHONPATH=build:$PYTHONPATH
export LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH
```

### Error Handling
```cpp
auto result = operation();
if (result.has_value()) { /* success */ }
else { Logger::error("Error: {}", result.error().message); }
```

### Testing
```bash
ninja test                           # C++ tests
uv run pytest tests/                # Python tests
```

---

**Document Status**: Complete
**Last Updated**: 2025-11-10
**For**: AI assistants (GitHub Copilot, Claude, etc.)
**Author**: Olumuyiwa Oluwasanmi
