# BigBrotherAnalytics Codebase Structure & Patterns

## Executive Summary

BigBrotherAnalytics is a sophisticated trading and economic data analysis system built with:
- **Database**: DuckDB (embedded SQL database at `data/bigbrother.duckdb`)
- **Frontend**: Streamlit dashboard (`dashboard/app.py`)
- **Data Collection**: Multiple API integrations (BLS, FRED, Schwab)
- **Trading**: Phase 5 implementation with Schwab API OAuth integration
- **Configuration**: YAML-based config system with environment variable support

---

## 1. DUCKDB SETUP & DATABASE PATTERNS

### Database Location
- **Main Database**: `data/bigbrother.duckdb` (16MB - production)
- **Paper Trading DB**: `data/paper_trading.duckdb` (for testing)
- **Test Orders DB**: `data/test_orders.duckdb` (for validation)

### Connection Pattern

```python
import duckdb
from pathlib import Path

# Database path
DB_PATH = Path("data/bigbrother.duckdb")

# Connection (read-only for dashboards, read-write for data collection)
conn = duckdb.connect(str(DB_PATH), read_only=True)  # Dashboard pattern
conn = duckdb.connect(str(DB_PATH))  # Data collection pattern

# Query execution
result = conn.execute("SELECT * FROM table_name").df()  # Returns pandas DataFrame
result = conn.execute("SELECT * FROM table_name").fetchall()  # Returns list of tuples
```

### Key Database Tables

#### SECTOR TABLES
- **sectors**: 11 GICS sectors (Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate)
- **company_sectors**: Maps 24 stocks to sectors
- **sector_performance**: Daily ETF performance tracking

#### EMPLOYMENT TABLES
- **sector_employment**: BLS employment data by sector
- **employment_events**: Tracks layoffs, hiring, freezes
- **jobless_claims**: Weekly jobless claims from FRED
- **monthly_jobs_report**: Monthly employment situation report
- **sector_news_sentiment**: News sentiment by sector (EMPTY - ready for news system)

#### TRADING TABLES
- **orders**: Order lifecycle tracking
- **order_updates**: Audit trail for order modifications
- **order_fills**: Execution details
- **order_rejections**: Rejection tracking
- **order_performance**: P&L analysis
- **positions**: Active positions
- **positions_history**: Historical positions

#### OAUTH TOKENS
- **oauth_tokens**: Schwab API OAuth token storage

### Schema Files
```
scripts/database_schema_*.sql:
├── sectors.sql        # Sector master data
├── employment.sql     # Employment data
├── orders.sql         # Trading orders
├── oauth.sql          # OAuth tokens
├── tax.sql            # Tax tracking
└── alerts.sql         # System alerts
```

---

## 2. DATA COLLECTION SCRIPTS

### Directory Structure
```
scripts/data_collection/
├── download_historical.py         # Yahoo Finance historical data
├── bls_employment.py              # BLS employment API integration
├── bls_jobless_claims.py          # FRED jobless claims API
├── protected_external_apis.py     # Circuit breaker wrapper
├── circuit_breaker_wrapper.py     # Resilience pattern
└── __pycache__/
```

### Key Classes

#### BLSEmploymentCollector
**File**: `scripts/data_collection/bls_employment.py`

Fetches employment data from Bureau of Labor Statistics API.

**Key Methods**:
- `__init__(api_key, db_path)` - Initialize with API key from config or environment
- `fetch_series(series_ids, start_year, end_year)` - Fetch from BLS API (supports batch of 50)
- `collect_sector_employment(years)` - Main method for employment data
- `collect_jobless_claims(weeks)` - Collect weekly claims

**Configuration Mapping**:
```python
SECTOR_SERIES = {
    'mining_logging': 'CES1000000001',
    'construction': 'CES2000000001',
    'manufacturing': 'CES3000000001',
    # ... 20 total series
}
```

#### FREDJoblessClaimsCollector
**File**: `scripts/data_collection/bls_jobless_claims.py`

Fetches jobless claims from Federal Reserve Economic Data API.

**Series**:
- ICSA: Initial jobless claims (weekly, seasonally adjusted)
- CCSA: Continued claims (weekly, seasonally adjusted)

#### NewsCollector (RECOMMENDED FOR YOUR IMPLEMENTATION)
**File**: `scripts/data_collection/news_collector.py` (to be created)

Fetches and analyzes financial news articles.

**Key Methods**:
- `fetch_sector_news(sector_name, days_back)` - Fetch articles for a sector
- `calculate_sentiment(article)` - Analyze article sentiment (-1.0 to 1.0)
- `store_news(articles, sector_code)` - Store in database

---

## 3. DASHBOARD STRUCTURE (STREAMLIT)

### Main File
**Path**: `dashboard/app.py` (1200+ lines)

**Framework**: Streamlit

**Pattern**: Single-file app with modular view functions

### Key Functions

```python
@st.cache_resource
def get_db_connection():
    """Cached DuckDB connection"""

def load_positions():        # Active positions
def load_sector_employment():  # Sector data with growth rates
def calculate_signal(growth_rate):  # Convert to trading signal

def show_overview():         # Overview dashboard
def show_positions():        # Real-time positions
def show_pnl_analysis():     # P&L analysis
def show_employment_signals(): # Sector signals
def show_trade_history():    # Historical trades
def show_alerts():           # System alerts
def show_system_health():    # System status
```

### Supporting Files
- `dashboard/tax_implications_view.py` - Tax view
- `dashboard/circuit_breaker_monitor.py` - API health
- `dashboard/generate_sample_data.py` - Test data

---

## 4. TRADING SYSTEM (PHASE 5)

### Phase 5 Setup
**File**: `scripts/phase5_setup.py`

Handles:
- OAuth token refresh with automatic mechanism
- Database initialization
- System health checks
- Schwab API connectivity testing
- Optional auto-start of services

**Key Features**:
- Reads `configs/schwab_app_config.yaml` for app credentials
- Updates `configs/schwab_tokens.json` with new tokens
- Refreshes tokens when expired (30-min access token, 7-day refresh)
- Stores tokens in `oauth_tokens` DuckDB table

### Configuration Files

#### `configs/config.yaml` (Production)
```yaml
trading:
  paper_trading: true         # SET TO FALSE FOR LIVE TRADING
  cycle_interval_ms: 60000    # 1 minute
  trading_hours_only: true
  market_open_time: "09:30"   # ET
  market_close_time: "16:00"  # ET

risk:
  account_value: 30000.0
  max_daily_loss: 900.0       # 3% of capital
  max_position_size: 1500.0   # 5% of capital
  max_concurrent_positions: 10

strategies:
  delta_neutral_straddle:
    enabled: true
    profit_target_percent: 75.0
    stop_loss_percent: 50.0
```

#### `configs/paper_trading.yaml` (Testing)
```yaml
database:
  path: "data/paper_trading.duckdb"

risk:
  account_value: 5000.0       # Small test account
  max_daily_loss: 100.0       # 2% of paper account
  max_position_size: 100.0    # $50-100 per trade
```

### Symbols Traded (24 stocks)
- **Tech (8)**: AAPL, MSFT, GOOGL, META, NVDA, AMZN, TSLA, AMD
- **Healthcare (3)**: JNJ, PFE, UNH
- **Financials (4)**: JPM, BAC, WFC, GS
- **Consumer (3)**: WMT, HD, MCD
- **Industrials (3)**: BA, CAT, UPS
- **Energy (2)**: XOM, CVX
- **Utilities (1)**: NEE
- **Materials (1)**: LIN
- **Sector ETFs (11)**: XLE, XLB, XLI, XLY, XLP, XLV, XLF, XLK, XLC, XLU, XLRE

---

## 5. CONFIGURATION SYSTEM

### API Keys Configuration
**File**: `configs/api_keys.yaml` (NOT COMMITTED - in .gitignore)

```yaml
fred_api_key: your_key_here
bls_api_key: your_key_here
news_api_key: your_key_here
alpha_vantage_api_key: your_key_here

schwab:
  app_key: your_app_key_here
  app_secret: your_app_secret_here
```

**Loading Pattern**:
1. Try `configs/api_keys.yaml`
2. Fall back to environment variables (e.g., `FRED_API_KEY`)

### Environment Variables
- `FRED_API_KEY` - Federal Reserve Economic Data
- `BLS_API_KEY` - Bureau of Labor Statistics
- `NEWS_API_KEY` - News API
- `SCHWAB_CLIENT_ID` - Schwab API client ID
- `SCHWAB_CLIENT_SECRET` - Schwab API secret
- `SCHWAB_ACCOUNT_ID` - Schwab account ID

---

## 6. DATA FLOW PIPELINE

### Economic Data Collection
```
1. BLS Employment (1st Friday after month-end)
   └─ scripts/data_collection/bls_employment.py
   └─ Stores in: sector_employment table

2. FRED Jobless Claims (Thursdays)
   └─ scripts/data_collection/bls_jobless_claims.py
   └─ Stores in: jobless_claims table

3. Historical Market Data (on-demand)
   └─ scripts/data_collection/download_historical.py
   └─ Stores in: data/raw/*.parquet

4. News & Sentiment (DAILY - your implementation)
   └─ scripts/data_collection/news_collector.py
   └─ Stores in: sector_news_sentiment table
```

### Automated Updates
**Directory**: `scripts/automated_updates/`

```
daily_update.py              # Main daily update orchestrator
daily_employment_update.py   # Employment-specific updates
recalculate_signals.py       # Recalculate trading signals
notify.py                    # Alert notifications
```

---

## 7. CODE PATTERNS & IMPORTS

### Standard Imports
```python
# Core
import os, sys, json, logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Data
import pandas as pd, numpy as np, requests

# Database & Config
import duckdb, yaml

# APIs
import yfinance as yf
from fredapi import Fred

# Frontend
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Custom
from circuit_breaker_wrapper import CircuitBreakerWrapper
from protected_external_apis import ProtectedFREDAPI
```

### Logging Pattern
```python
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# For file logging:
LOG_DIR = Path('logs/automated_updates')
LOG_DIR.mkdir(parents=True, exist_ok=True)
```

### Path Resolution
```python
from pathlib import Path

# From any script location:
BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = BASE_DIR / 'data' / 'bigbrother.duckdb'
CONFIG_PATH = BASE_DIR / 'configs' / 'config.yaml'
```

### Configuration Loading
```python
import yaml

def _load_api_key_from_config(self) -> Optional[str]:
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'api_keys.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('fred_api_key')
    return None

api_key = self._load_api_key_from_config() or os.getenv('FRED_API_KEY')
```

### DuckDB Queries
```python
conn = duckdb.connect("data/bigbrother.duckdb")

# DataFrame result
df = conn.execute("SELECT * FROM table_name").df()

# Insert with parameter binding
conn.execute(
    "INSERT INTO table (col1, col2) VALUES (?, ?)",
    [val1, val2]
)

# Complex query with CTEs
query = """
    WITH latest AS (
        SELECT * FROM table
        WHERE date = (SELECT MAX(date) FROM table)
    )
    SELECT * FROM latest
"""
df = conn.execute(query).df()
```

### API with Circuit Breaker
```python
from circuit_breaker_wrapper import CircuitBreakerWrapper

class APIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.circuit_breaker = CircuitBreakerWrapper(
            name="API_NAME",
            failure_threshold=5,
            timeout_seconds=60
        )
    
    def fetch(self, url: str, params: Dict) -> Dict:
        def _fetch():
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        return self.circuit_breaker.call(_fetch)
```

---

## 8. RECOMMENDED STRUCTURE FOR NEWS INGESTION

### 1. Create News Collector

**File**: `scripts/data_collection/news_collector.py`

```python
from circuit_breaker_wrapper import CircuitBreakerWrapper
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import duckdb
import requests
import yaml
import logging

logger = logging.getLogger(__name__)

class NewsCollector:
    NEWS_SOURCES = {
        'newsapi': 'https://newsapi.org/v2/everything',
        'alphavantage': 'https://www.alphavantage.co/query',
    }
    
    def __init__(self, api_key: Optional[str] = None, db_path: str = "data/bigbrother.duckdb"):
        self.api_key = api_key or self._load_api_key_from_config()
        self.db_path = Path(db_path)
        self.circuit_breaker = CircuitBreakerWrapper(
            name="NEWS_API",
            failure_threshold=5,
            timeout_seconds=60
        )
    
    def _load_api_key_from_config(self) -> Optional[str]:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'api_keys.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('news_api_key')
        return None
    
    def fetch_sector_news(self, sector_name: str, days_back: int = 7) -> List[Dict]:
        """Fetch news articles for a sector"""
        def _fetch():
            params = {
                'q': sector_name,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.api_key,
                'from': (datetime.now() - timedelta(days=days_back)).isoformat()
            }
            response = requests.get(self.NEWS_SOURCES['newsapi'], params=params, timeout=30)
            response.raise_for_status()
            return response.json().get('articles', [])
        
        return self.circuit_breaker.call(_fetch)
    
    def calculate_sentiment(self, article: Dict) -> float:
        """Calculate sentiment score from article"""
        # TODO: Implement with textblob, vader, or transformers
        # For now, return placeholder
        return 0.0
    
    def store_news(self, articles: List[Dict], sector_code: int) -> None:
        """Store articles in database"""
        conn = duckdb.connect(str(self.db_path))
        
        for article in articles:
            sentiment = self.calculate_sentiment(article)
            conn.execute("""
                INSERT INTO sector_news_sentiment 
                (sector_code, news_date, sentiment_score, news_count, impact_magnitude)
                VALUES (?, ?, ?, ?, ?)
            """, [
                sector_code,
                datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                sentiment,
                1,
                'High' if abs(sentiment) > 0.5 else 'Medium'
            ])
        
        conn.close()
```

### 2. Integrate into Daily Updates

**File**: `scripts/automated_updates/daily_update.py`

```python
from scripts.data_collection.news_collector import NewsCollector

class DailyDataUpdater:
    def __init__(self, ...):
        self.news_collector = NewsCollector(api_key=self.news_api_key)
    
    def update_sector_news(self) -> bool:
        """Fetch and store sector news"""
        sectors_map = {
            'Energy': 10, 'Materials': 15, 'Industrials': 20,
            'Consumer Discretionary': 25, 'Consumer Staples': 30,
            'Health Care': 35, 'Financials': 40, 'Technology': 45,
            'Communication Services': 50, 'Utilities': 55, 'Real Estate': 60
        }
        
        for sector_name, sector_code in sectors_map.items():
            articles = self.news_collector.fetch_sector_news(sector_name)
            if articles:
                self.news_collector.store_news(articles, sector_code)
        
        return True
```

### 3. Add Dashboard View

**File**: `dashboard/app.py`

```python
def show_news_sentiment():
    """Display sector news and sentiment analysis"""
    st.header("📰 Sector News & Sentiment")
    
    conn = get_db_connection()
    
    query = """
        SELECT s.sector_name, sns.sentiment_score, sns.news_count, sns.news_date
        FROM sector_news_sentiment sns
        JOIN sectors s ON sns.sector_code = s.sector_code
        WHERE sns.news_date >= CURRENT_DATE - INTERVAL 30 DAYS
        ORDER BY sns.news_date DESC
    """
    
    df = conn.execute(query).df()
    
    if not df.empty:
        fig = px.line(df, x='news_date', y='sentiment_score', 
                      color='sector_name', title='Sector Sentiment Trends')
        st.plotly_chart(fig, use_container_width=True)
```

### 4. Update API Keys Example

**File**: `configs/api_keys.yaml.example`

```yaml
# Add this section:
news_api_key: your_newsapi_key_here

# Supported sources:
# - NewsAPI (100 requests/day free): https://newsapi.org/
# - Alpha Vantage (500 calls/day free): https://www.alphavantage.co/
# - Finnhub (free tier): https://finnhub.io/
```

---

## 9. PHASE 5+ NEWS INGESTION SYSTEM

### Architecture Overview

The news ingestion system is implemented as a hybrid C++23/Python architecture for optimal performance:

**C++ Modules (Performance Layer)**:
- `bigbrother.market_intelligence.sentiment` - Keyword-based sentiment analysis
- `bigbrother.market_intelligence.news` - NewsAPI client with CURL

**Python Layer (Integration)**:
- `scripts/data_collection/news_ingestion.py` - Python orchestration script
- `src/python_bindings/news_bindings.cpp` - pybind11 bindings (119 lines)

**Database Schema**:
- `news_articles` table with 15 columns for comprehensive article storage

**Dashboard Integration**:
- News Feed tab in Streamlit dashboard (`show_news_feed()` function)

### C++ Module Details

#### 1. Sentiment Analyzer Module (281 lines)
**File**: `src/market_intelligence/sentiment_analyzer.cppm`

**Module Name**: `bigbrother.market_intelligence.sentiment`

**Key Components**:
```cpp
// Data type
struct SentimentResult {
    double score;                           // -1.0 to 1.0
    std::string label;                      // "positive", "negative", "neutral"
    std::vector<std::string> positive_keywords;
    std::vector<std::string> negative_keywords;
    double positive_score;
    double negative_score;
    size_t total_words;
    double keyword_density;
};

// Analyzer class
class SentimentAnalyzer {
public:
    SentimentAnalyzer();
    [[nodiscard]] auto analyze(std::string const& text) const -> SentimentResult;
};
```

**Features**:
- 60+ positive keywords (e.g., "profit", "growth", "bullish", "beat")
- 60+ negative keywords (e.g., "loss", "decline", "bearish", "miss")
- Keyword density calculation
- No ML dependencies (fast, lightweight)
- Returns structured sentiment result

#### 2. News Ingestion Module (402 lines)
**File**: `src/market_intelligence/news_ingestion.cppm`

**Module Name**: `bigbrother.market_intelligence.news`

**Key Components**:
```cpp
// Article data type
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
    Timestamp published_at;
    Timestamp fetched_at;
    double sentiment_score;
    std::string sentiment_label;
    std::vector<std::string> positive_keywords;
    std::vector<std::string> negative_keywords;
};

// Configuration
struct NewsAPIConfig {
    std::string api_key;
    std::string base_url = "https://newsapi.org/v2";
    int requests_per_day = 100;
    int lookback_days = 7;
    int timeout_seconds = 30;
};

// API client
class NewsAPICollector {
public:
    explicit NewsAPICollector(NewsAPIConfig config);
    ~NewsAPICollector();

    [[nodiscard]] auto fetch_news(std::string const& symbol)
        -> std::expected<std::vector<NewsArticle>, Error>;

    [[nodiscard]] auto store_articles(std::vector<NewsArticle> const& articles)
        -> std::expected<void, Error>;
};
```

**Features**:
- HTTP client using libcurl for NewsAPI requests
- Automatic sentiment analysis integration
- DuckDB storage via database module
- Error handling with `std::expected<T, Error>`
- Rate limiting support (100 requests/day)
- RAII compliance (Rule of Five with deleted move for CURL state)

### Database Schema

**Table**: `news_articles` (15 columns)

```sql
CREATE TABLE IF NOT EXISTS news_articles (
    article_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    description TEXT,
    content TEXT,
    url VARCHAR,
    source_name VARCHAR,
    source_id VARCHAR,
    author VARCHAR,
    published_at TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentiment_score DOUBLE,
    sentiment_label VARCHAR,
    positive_keywords TEXT[],
    negative_keywords TEXT[]
);

-- Performance indexes
CREATE INDEX idx_news_symbol ON news_articles(symbol);
CREATE INDEX idx_news_published ON news_articles(published_at);
```

**Setup Script**: `scripts/monitoring/setup_news_database.py`

### Python Integration

#### Data Collection Script
**File**: `scripts/data_collection/news_ingestion.py`

Main script that uses Python bindings to call C++ news collector:
```python
import news_ingestion_py  # C++ module via pybind11

# Configure and run
collector = news_ingestion_py.NewsAPICollector(config)
articles = collector.fetch_news("AAPL")
collector.store_articles(articles)
```

#### Python Bindings
**File**: `src/python_bindings/news_bindings.cpp` (119 lines)

pybind11 bindings exposing C++ classes to Python:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Import C++ modules
import bigbrother.market_intelligence.sentiment;
import bigbrother.market_intelligence.news;

PYBIND11_MODULE(news_ingestion_py, m) {
    // Expose SentimentResult
    py::class_<SentimentResult>(m, "SentimentResult")
        .def_readonly("score", &SentimentResult::score)
        .def_readonly("label", &SentimentResult::label);

    // Expose NewsAPICollector
    py::class_<NewsAPICollector>(m, "NewsAPICollector")
        .def(py::init<NewsAPIConfig>())
        .def("fetch_news", &NewsAPICollector::fetch_news)
        .def("store_articles", &NewsAPICollector::store_articles);
}
```

### Dashboard Integration

**File**: `dashboard/app.py` (lines 736-800+)

**Function**: `show_news_feed()`

Displays news feed with sentiment analysis in Streamlit dashboard:
```python
def show_news_feed():
    """Display news feed with sentiment analysis"""
    st.header("News Feed")

    # Check if table exists
    conn = get_db_connection()

    # Query recent articles
    df = conn.execute("""
        SELECT
            symbol, title, description, source_name,
            published_at, sentiment_score, sentiment_label,
            positive_keywords, negative_keywords
        FROM news_articles
        ORDER BY published_at DESC
        LIMIT 100
    """).df()

    # Display with sentiment indicators
    for _, article in df.iterrows():
        sentiment_color = "green" if article['sentiment_score'] > 0 else "red"
        st.markdown(f"**{article['title']}** ({article['symbol']})")
        st.caption(f"Sentiment: {article['sentiment_label']}")
```

### Build System Integration

**CMake Configuration**:
```cmake
# Market intelligence library
add_library(market_intelligence)
target_sources(market_intelligence
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/market_intelligence/sentiment_analyzer.cppm
            src/market_intelligence/news_ingestion.cppm
)
target_link_libraries(market_intelligence
    PUBLIC utils
    PRIVATE CURL::libcurl
)

# Python bindings
pybind11_add_module(news_ingestion_py src/python_bindings/news_bindings.cpp)
target_link_libraries(news_ingestion_py PRIVATE market_intelligence)
```

**Build Requirements**:
- CMake 3.28+ with Ninja generator
- Clang 21+ with C++23 modules support
- libcurl (for HTTP requests)
- nlohmann/json (for JSON parsing)
- pybind11 (for Python bindings)

### Coding Standards Used

All news ingestion code follows these standards:

1. **Trailing Return Type**: `auto func() -> ReturnType` (100% coverage)
2. **Error Handling**: `std::expected<T, Error>` with `std::unexpected(Error::make())`
3. **Attributes**: `[[nodiscard]]` on all query methods
4. **RAII**: Rule of Five compliance (deleted move for CURL state)
5. **Core Guidelines**:
   - C.1: Use struct for passive data (NewsArticle, SentimentResult)
   - C.2: Use class when invariants exist (NewsAPICollector, SentimentAnalyzer)
   - C.21: Define or delete all copy/move operations
   - F.16: Use const& for read-only parameters
   - F.20: Return by value for output
   - R.1: RAII for resource management (CURL handles)

### Integration with Phase 5

**Startup** (`scripts/phase5_setup.py`):
- Initializes news database schema
- Verifies news_articles table exists
- Sets up indexes for performance

**Shutdown** (`scripts/phase5_shutdown.py`):
- Terminates running news ingestion processes
- Flushes any pending articles to database

### Module Dependency Graph

```
bigbrother.market_intelligence.news
  ├── depends on: bigbrother.utils.types
  ├── depends on: bigbrother.utils.logger
  ├── depends on: bigbrother.utils.database
  └── depends on: bigbrother.market_intelligence.sentiment
      ├── depends on: bigbrother.utils.types
      └── depends on: bigbrother.utils.logger
```

**Build Order**: types → logger → database → sentiment → news → bindings

---

## QUICK REFERENCE FOR AGENTS

### Key Absolute Paths
```
Database:
  /home/muyiwa/Development/BigBrotherAnalytics/data/bigbrother.duckdb

Scripts:
  /home/muyiwa/Development/BigBrotherAnalytics/scripts/
  /home/muyiwa/Development/BigBrotherAnalytics/scripts/data_collection/
  /home/muyiwa/Development/BigBrotherAnalytics/scripts/automated_updates/
  /home/muyiwa/Development/BigBrotherAnalytics/scripts/monitoring/

C++ Modules:
  /home/muyiwa/Development/BigBrotherAnalytics/src/market_intelligence/
  /home/muyiwa/Development/BigBrotherAnalytics/src/python_bindings/

Dashboard:
  /home/muyiwa/Development/BigBrotherAnalytics/dashboard/app.py

Config:
  /home/muyiwa/Development/BigBrotherAnalytics/configs/api_keys.yaml
  /home/muyiwa/Development/BigBrotherAnalytics/configs/config.yaml
```

### Common Patterns
1. **Path Resolution**: `BASE_DIR = Path(__file__).parent.parent.parent`
2. **API Key Loading**: Try config file first, then environment variable
3. **Circuit Breaker**: Wrap external API calls for resilience
4. **Database Access**: Use `.df()` to get pandas DataFrames from DuckDB
5. **Logging**: Always set up module-level logger at top of script
6. **C++ Error Handling**: Use `std::expected<T, Error>` with `std::unexpected`
7. **Trailing Return Types**: Always use `auto func() -> ReturnType` in C++

### Files to Match Style
- Data collectors: `/home/muyiwa/Development/BigBrotherAnalytics/scripts/data_collection/bls_employment.py`
- Dashboard views: `/home/muyiwa/Development/BigBrotherAnalytics/dashboard/app.py`
- Automated updates: `/home/muyiwa/Development/BigBrotherAnalytics/scripts/automated_updates/daily_update.py`
- C++ modules: `/home/muyiwa/Development/BigBrotherAnalytics/src/market_intelligence/sentiment_analyzer.cppm`
- Python bindings: `/home/muyiwa/Development/BigBrotherAnalytics/src/python_bindings/news_bindings.cpp`

