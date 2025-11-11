# News Source Filtering - Quick Reference

## Quick Start

### Python
```python
import news_ingestion_py as news

# Premium sources only
config = news.NewsAPIConfig()
config.api_key = "your_key"
config.quality_filter = news.SourceQuality.Premium

collector = news.NewsAPICollector(config)
result = collector.fetch_news("AAPL")
```

### C++
```cpp
import bigbrother.market_intelligence.news;

NewsAPIConfig config;
config.api_key = "your_key";
config.quality_filter = SourceQuality::Premium;

NewsAPICollector collector(config);
auto result = collector.fetchNews("AAPL");
```

## Quality Levels

| Level | Count | Description |
|-------|-------|-------------|
| `SourceQuality.All` | ~all | No filtering, accept all non-excluded sources |
| `SourceQuality.Premium` | 7 | WSJ, Bloomberg, Reuters, FT, Barron's, IBD, Economist |
| `SourceQuality.Verified` | 30 | Premium + major news outlets (CNBC, CNN, etc.) |
| `SourceQuality.Exclude` | 0 | Filter all sources (testing only) |

**Default:** `SourceQuality.Verified`

## Source Lists

### Premium (7)
- The Wall Street Journal
- Bloomberg
- Reuters
- Financial Times
- Barron's
- Investor's Business Daily
- The Economist

### Verified (30)
All Premium + CNBC, CNN Business, MarketWatch, Yahoo Finance, Seeking Alpha, Business Insider, Forbes, Fortune, TechCrunch, NYT, Washington Post, AP, BBC, CBS, NBC, ABC, USA Today, TheStreet, Benzinga, Motley Fool

### Excluded (9)
Blogger, WordPress, Medium, Tumblr, Reddit, Unknown Source, Google News, News Aggregator, RSS Feed

## Custom Sources

### Add Preferred Sources
```python
config.preferred_sources = ["The Guardian", "Financial Post"]
```

### Add Excluded Sources
```python
config.excluded_sources = ["Unreliable Blog", "Spam Network"]
```

## Common Patterns

### High-Quality Trading Signals
```python
config.quality_filter = news.SourceQuality.Premium
```

### Balanced Coverage
```python
config.quality_filter = news.SourceQuality.Verified  # Default
```

### Maximum Coverage
```python
config.quality_filter = news.SourceQuality.All
```

### Custom Filter
```python
config.quality_filter = news.SourceQuality.Verified
config.preferred_sources = ["Your Trusted Source"]
config.excluded_sources = ["Source to Avoid"]
```

## Filtering Logic

1. Check if source is in **excluded list** → filter out
2. Apply **quality filter**:
   - `All`: Accept if not excluded
   - `Premium`: Accept if in premium list
   - `Verified`: Accept if in verified list
   - `Exclude`: Reject all

## Logging

```
INFO:   Quality filter: 2
DEBUG:  Filtered out article from source: WordPress
INFO:   Filtered 3 articles based on source quality
INFO:   Fetched 17 articles for AAPL
```

## Files Modified

1. `/home/muyiwa/Development/BigBrotherAnalytics/src/market_intelligence/news_ingestion.cppm`
   - Added `SourceQuality` enum
   - Updated `NewsAPIConfig` struct
   - Enhanced `parseArticles()` with filtering
   - Added source lists and filtering logic

2. `/home/muyiwa/Development/BigBrotherAnalytics/src/python_bindings/news_bindings.cpp`
   - Exposed `SourceQuality` enum to Python
   - Added config fields to Python bindings

## Documentation

- **Examples:** `NEWS_SOURCE_FILTERING_EXAMPLE.md`
- **Implementation:** `NEWS_SOURCE_FILTERING_IMPLEMENTATION.md`
- **Quick Reference:** This file

## Build

```bash
cmake --build build --target news_ingestion_py
```

Status: ✓ Build succeeds with no errors
