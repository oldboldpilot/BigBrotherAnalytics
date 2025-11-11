# Schwab API News Investigation Guide

**Date**: 2025-11-10
**Purpose**: Determine if Schwab API provides financial news endpoints

---

## Investigation Steps

### 1. Check Official Schwab API Documentation

**Action**: Log into your Schwab Developer Portal account

```bash
# Navigate to:
https://developer.schwab.com/

# Look for:
- API Reference / Endpoint Documentation
- Market Data endpoints
- Any endpoint containing "news", "articles", "headlines"
```

**What to Look For:**
- Endpoint path (e.g., `/marketdata/v1/news`)
- Parameters (symbol, date range, limit)
- Response format (JSON structure)
- Rate limits
- Authentication requirements

### 2. Test with Existing Code

If you find a news endpoint, test it with our existing Schwab client:

```cpp
// Add to schwab_api.cppm
/**
 * GET /marketdata/v1/news (if it exists)
 */
[[nodiscard]] auto getNews(std::string const& symbol,
                           std::string const& from_date = "",
                           std::string const& to_date = "")
    -> Result<std::vector<NewsArticle>> {

    std::string url = std::string(SCHWAB_API_BASE_URL) +
                     "/marketdata/v1/news?symbol=" + symbol;

    if (!from_date.empty()) {
        url += "&from=" + from_date;
    }
    if (!to_date.empty()) {
        url += "&to=" + to_date;
    }

    // Make authenticated request
    auto response = makeAuthenticatedRequest(url, "GET");

    if (!response) {
        return std::unexpected(response.error());
    }

    // Parse response and return articles
    return parseNewsResponse(response.value());
}
```

### 3. Check Alternative Schwab Data Sources

**StreetSmart Edge Platform:**
- Schwab's trading platform shows news
- Investigate if this uses an internal API we could access
- Check browser network tab while viewing news in StreetSmart

**Schwab.com Website:**
- Symbol pages show news (e.g., schwab.com/research/stocks/AAPL)
- Inspect network requests for API calls
- Look for endpoints like:
  - `api.schwab.com/news`
  - `content.schwab.com/articles`
  - `research.schwab.com/api`

---

## Expected API Response Format

If Schwab provides news, it will likely look like:

```json
{
  "news": [
    {
      "headline": "Apple Reports Strong Q4 Earnings",
      "summary": "Apple Inc. reported quarterly earnings...",
      "provider": "Dow Jones",
      "datetime": 1699564800000,
      "url": "https://www.wsj.com/articles/...",
      "hasPaywall": false,
      "relatedSymbols": ["AAPL"]
    }
  ]
}
```

---

## Integration Plan (If News Endpoint Exists)

### Phase 1: Add Schwab News to Existing Infrastructure

**1. Extend news_ingestion.cppm**

```cpp
// Add Schwab news source to NewsAPIConfig
enum class NewsSource {
    NewsAPI,      // Current implementation
    Schwab,       // New: Schwab API
    Combined      // Both sources merged
};

struct NewsAPIConfig {
    // ... existing fields ...

    // New fields for Schwab
    NewsSource source{NewsSource::Combined};
    bool prefer_schwab{true};  // Use Schwab first, fallback to NewsAPI
};
```

**2. Create SchwabNewsCollector**

```cpp
class SchwabNewsCollector {
public:
    explicit SchwabNewsCollector(SchwabAPI& schwab_client);

    [[nodiscard]] auto fetchNews(std::string const& symbol,
                                 std::string const& from_date,
                                 std::string const& to_date)
        -> Result<std::vector<NewsArticle>>;

    [[nodiscard]] auto fetchNewsBatch(
        std::vector<std::string> const& symbols)
        -> Result<std::map<std::string, std::vector<NewsArticle>>>;

private:
    SchwabAPI& schwab_client_;
    SentimentAnalyzer sentiment_analyzer_;
};
```

**3. Unified News Collection**

```cpp
// Combined news from both sources
class UnifiedNewsCollector {
public:
    UnifiedNewsCollector(NewsAPICollector& newsapi,
                        SchwabNewsCollector& schwab);

    // Fetches from both sources, merges, deduplicates
    [[nodiscard]] auto fetchNews(std::string const& symbol)
        -> Result<std::vector<NewsArticle>>;

private:
    auto mergeAndDeduplicate(
        std::vector<NewsArticle> const& newsapi_articles,
        std::vector<NewsArticle> const& schwab_articles)
        -> std::vector<NewsArticle>;
};
```

### Phase 2: Update Dashboard

```python
# Add source filter to dashboard
source_filter = st.selectbox(
    "News Source",
    ["All", "NewsAPI", "Schwab", "Both"]
)

# Show source badge on article cards
if article['source'] == 'schwab':
    st.badge("Schwab", type="info")
else:
    st.badge("NewsAPI", type="secondary")
```

---

## Benefits of Adding Schwab News

### 1. **No Rate Limits** (Likely)
- NewsAPI: 100 requests/day (free tier)
- Schwab: Likely higher limits for authenticated users
- **Result**: Can fetch news more frequently

### 2. **Better Coverage**
- Schwab may have different news sources
- Combine with NewsAPI for comprehensive coverage
- **Result**: More articles per symbol

### 3. **Real-time Updates**
- Schwab news may be more current
- Integrated with trading platform
- **Result**: Faster sentiment signals

### 4. **Authorized Access**
- Already authenticated with Schwab
- No separate API key needed
- **Result**: Simpler configuration

### 5. **Consistent Data**
- Same source as trading data
- Timestamps align with market events
- **Result**: Better correlation analysis

---

## Fallback Strategy

If Schwab doesn't provide news API:

### Alternative 1: Schwab Research Integration
- Use Schwab's research portal API (if available)
- May require different authentication

### Alternative 2: Multi-Source Aggregation
Keep current NewsAPI + add:
- **AlphaVantage**: 5 requests/min free tier
- **Finnhub**: 60 requests/min free tier
- **Yahoo Finance**: Unofficial API, no key needed

### Alternative 3: Web Scraping (Last Resort)
- Scrape Schwab.com symbol pages for news
- Use BeautifulSoup/Playwright
- **Caution**: May violate ToS, fragile

---

## Implementation Priority

**If Schwab News API Exists:**
1. Add Schwab news fetching (2-3 days)
2. Merge with NewsAPI data (1 day)
3. Update dashboard with source filter (1 day)
4. Test and validate (1 day)
5. **Total**: ~1 week

**If Schwab News API Doesn't Exist:**
1. Continue with current NewsAPI implementation
2. Consider adding AlphaVantage or Finnhub as backup
3. Monitor Schwab developer portal for updates

---

## Next Steps

1. **Check Schwab Developer Portal**: Log in and search for news endpoints
2. **Test in Browser**: Inspect network calls on schwab.com symbol pages
3. **Contact Schwab Support**: Ask if news API is available
4. **Report Findings**: Document what you discover
5. **Implement if Available**: Follow integration plan above

---

**Status**: Investigation Required
**Priority**: Medium (Current NewsAPI implementation works well)
**Timeline**: 1-2 hours to investigate, 1 week to implement if available
