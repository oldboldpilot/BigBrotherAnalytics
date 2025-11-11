# News Source Filtering - Usage Examples

## Overview

The news ingestion module now supports filtering articles based on source quality and reliability. This helps ensure you receive news from trusted, high-quality sources.

## Source Quality Levels

### SourceQuality.All
- No filtering applied
- Accepts all sources except explicitly excluded ones
- Use when you want maximum coverage

### SourceQuality.Premium
- Top-tier financial news outlets only
- Includes: WSJ, Bloomberg, Reuters, Financial Times, Barron's, Investor's Business Daily, The Economist
- Use for highest quality financial news

### SourceQuality.Verified
- Major news outlets with editorial standards
- Includes all Premium sources plus: CNBC, CNN Business, MarketWatch, Yahoo Finance, Seeking Alpha, Business Insider, Forbes, Fortune, TechCrunch, NYT, Washington Post, AP, BBC, CBS, NBC, ABC, USA Today, TheStreet, Benzinga, Motley Fool
- Default setting - balanced quality and coverage

### SourceQuality.Exclude
- Rejects all sources (used for testing)

## Built-in Source Lists

### Premium Sources (7 sources)
- The Wall Street Journal
- Bloomberg
- Reuters
- Financial Times
- Barron's
- Investor's Business Daily
- The Economist

### Verified Sources (30 sources)
All premium sources plus:
- CNBC
- CNN Business
- MarketWatch
- Yahoo Finance
- Seeking Alpha
- Business Insider
- Forbes
- Fortune
- TechCrunch
- The New York Times
- Washington Post
- Associated Press
- BBC News
- CBS News
- NBC News
- ABC News
- USA Today
- TheStreet
- Benzinga
- Motley Fool

### Excluded Sources (9 types)
Automatically filtered out:
- Blogger
- WordPress
- Medium
- Tumblr
- Reddit
- Unknown Source
- Google News
- News Aggregator
- RSS Feed

## Python Usage Examples

### Example 1: Basic Setup with Default Verified Sources

```python
import news_ingestion_py as news

# Create config with default Verified quality filter
config = news.NewsAPIConfig()
config.api_key = "your_api_key_here"
config.quality_filter = news.SourceQuality.Verified  # This is the default

# Create collector
collector = news.NewsAPICollector(config)

# Fetch news - only verified sources will be included
result = collector.fetch_news("AAPL")
if result:
    articles = result.value()
    print(f"Received {len(articles)} articles from verified sources")
    for article in articles:
        print(f"  - {article.source_name}: {article.title}")
```

### Example 2: Premium Sources Only

```python
import news_ingestion_py as news

# Create config for premium sources only
config = news.NewsAPIConfig()
config.api_key = "your_api_key_here"
config.quality_filter = news.SourceQuality.Premium

collector = news.NewsAPICollector(config)

# Only WSJ, Bloomberg, Reuters, FT, Barron's, IBD, Economist
result = collector.fetch_news("TSLA")
if result:
    articles = result.value()
    print(f"Received {len(articles)} premium articles")
```

### Example 3: Custom Preferred Sources

```python
import news_ingestion_py as news

# Add custom sources to the verified list
config = news.NewsAPIConfig()
config.api_key = "your_api_key_here"
config.quality_filter = news.SourceQuality.Verified

# Add your preferred sources
config.preferred_sources = [
    "The Guardian",
    "Financial Post",
    "Your Custom Source"
]

collector = news.NewsAPICollector(config)
result = collector.fetch_news("NVDA")
```

### Example 4: Custom Excluded Sources

```python
import news_ingestion_py as news

# Exclude additional sources
config = news.NewsAPIConfig()
config.api_key = "your_api_key_here"
config.quality_filter = news.SourceQuality.All

# Add sources to exclude (in addition to built-in excluded list)
config.excluded_sources = [
    "Questionable Blog",
    "Spam Source",
    "Ad Network News"
]

collector = news.NewsAPICollector(config)
result = collector.fetch_news("MSFT")
```

### Example 5: No Filtering (Accept All Sources)

```python
import news_ingestion_py as news

# Accept all sources except built-in excluded list
config = news.NewsAPIConfig()
config.api_key = "your_api_key_here"
config.quality_filter = news.SourceQuality.All

collector = news.NewsAPICollector(config)
result = collector.fetch_news("GOOGL")
if result:
    articles = result.value()
    print(f"Received {len(articles)} articles from all sources")
```

### Example 6: Batch Processing with Quality Filter

```python
import news_ingestion_py as news

config = news.NewsAPIConfig()
config.api_key = "your_api_key_here"
config.quality_filter = news.SourceQuality.Premium

collector = news.NewsAPICollector(config)

# Fetch news for multiple symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
result = collector.fetch_news_batch(symbols)

if result:
    news_map = result.value()
    for symbol, articles in news_map.items():
        print(f"\n{symbol}: {len(articles)} premium articles")
        for article in articles:
            print(f"  - {article.source_name}: {article.title[:60]}...")
```

## C++ Usage Examples

### Example 1: Basic C++ Usage

```cpp
#include <iostream>
import bigbrother.market_intelligence.news;

using namespace bigbrother::market_intelligence;

int main() {
    // Create config with premium sources
    NewsAPIConfig config;
    config.api_key = "your_api_key_here";
    config.quality_filter = SourceQuality::Premium;

    // Create collector
    NewsAPICollector collector(config);

    // Fetch news
    auto result = collector.fetchNews("AAPL");

    if (result) {
        auto articles = result.value();
        std::cout << "Received " << articles.size() << " premium articles\n";

        for (const auto& article : articles) {
            std::cout << "  - " << article.source_name
                      << ": " << article.title << "\n";
        }
    } else {
        std::cerr << "Error: " << result.error().message << "\n";
    }

    return 0;
}
```

### Example 2: Custom Source Lists in C++

```cpp
NewsAPIConfig config;
config.api_key = "your_api_key_here";
config.quality_filter = SourceQuality::Verified;

// Add preferred sources
config.preferred_sources = {
    "The Guardian",
    "Financial Post"
};

// Add excluded sources
config.excluded_sources = {
    "Unreliable Blog",
    "Spam Network"
};

NewsAPICollector collector(config);
```

## Logging

The system logs filtering activity:

```
INFO: NewsAPI collector initialized
INFO:   Base URL: https://newsapi.org/v2
INFO:   Daily limit: 100
INFO:   Quality filter: 2
INFO: Fetching news for symbol: AAPL
DEBUG:  Filtered out article from source: WordPress Blog
DEBUG:  Filtered out article from source: Reddit
INFO:   Filtered 3 articles based on source quality
INFO:   Fetched 17 articles for AAPL
```

## Best Practices

1. **Use Verified as Default**: Balances quality and coverage
2. **Use Premium for Trading Signals**: When making trading decisions
3. **Add Custom Sources Carefully**: Only add sources you trust
4. **Monitor Filtering Logs**: Check DEBUG logs to see what's being filtered
5. **Customize for Your Use Case**: Adjust based on your needs

## Notes

- Source matching is case-sensitive and uses substring matching
- Sources in the excluded list take precedence over preferred sources
- The Premium list is a subset of the Verified list
- Custom preferred_sources are added to the Verified list
- Custom excluded_sources are added to the built-in excluded list
