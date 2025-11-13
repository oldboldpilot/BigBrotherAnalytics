# Schwab Market Data API Reference

**Date:** November 9, 2025
**API Version:** Schwab Trader API v1
**Status:** Implementation in progress
**Source:** https://developer.schwab.com/

---

## Overview

**Schwab provides comprehensive market data through their Trader API.** We use Schwab's market data exclusively - no third-party data providers needed.

### Key Benefits
- ✅ **Free market data** with your Schwab account
- ✅ **Real-time quotes** for equities and options
- ✅ **Historical data** (OHLCV bars)
- ✅ **Options chains** with greeks and implied volatility
- ✅ **Market hours** information
- ✅ **Market movers** (top gainers/losers)
- ✅ **Integrated with trading** (same API, same authentication)

---

## Schwab Market Data Endpoints

### Base URL
```
https://api.schwabapi.com/marketdata/v1
```

### Authentication
All market data requests require OAuth 2.0 access token in the header:
```
Authorization: Bearer {access_token}
```

---

## 1. Quotes (Real-Time & Delayed)

### Get Single Quote
```http
GET /marketdata/v1/quotes/{symbol}
```

**Example:**
```bash
GET /marketdata/v1/quotes/AAPL
```

**Response:**
```json
{
  "AAPL": {
    "symbol": "AAPL",
    "description": "Apple Inc",
    "bidPrice": 180.50,
    "askPrice": 180.55,
    "lastPrice": 180.52,
    "bidSize": 100,
    "askSize": 100,
    "lastSize": 50,
    "highPrice": 182.00,
    "lowPrice": 179.50,
    "openPrice": 180.00,
    "closePrice": 179.75,
    "netChange": 0.77,
    "totalVolume": 45230000,
    "quoteTime": 1699564800000,
    "tradeTime": 1699564800000,
    "mark": 180.525,
    "52WeekHigh": 198.23,
    "52WeekLow": 124.17
  }
}
```

### Get Multiple Quotes
```http
GET /marketdata/v1/quotes?symbols={symbol1,symbol2,symbol3}
```

**Example:**
```bash
GET /marketdata/v1/quotes?symbols=AAPL,MSFT,GOOGL
```

**Use Cases:**
- Get current price for trading decisions
- Monitor sector ETFs (XLE, XLV, XLK, etc.)
- Real-time position valuation
- Calculate P&L

---

## 2. Options Chains

### Get Option Chain
```http
GET /marketdata/v1/chains?symbol={symbol}
```

**Parameters:**
- `symbol` (required): Underlying symbol (e.g., "SPY")
- `contractType`: CALL, PUT, or ALL (default: ALL)
- `strikeCount`: Number of strikes above/below ATM (default: all)
- `includeUnderlyingQuote`: true/false (default: false)
- `strategy`: SINGLE, ANALYTICAL, COVERED, etc. (default: SINGLE)
- `interval`: Strike price interval
- `strike`: Specific strike price
- `range`: ITM, NTM, OTM, SAK, SBK, SNK, ALL
- `fromDate`: From date (yyyy-MM-dd)
- `toDate`: To date (yyyy-MM-dd)
- `volatility`: Volatility to use in calculations
- `underlyingPrice`: Underlying price for calculations
- `interestRate`: Interest rate to use
- `daysToExpiration`: Days to expiration
- `expMonth`: Expiration month (ALL, JAN, FEB, etc.)
- `optionType`: S (Standard), NS (Non-Standard)

**Example:**
```bash
GET /marketdata/v1/chains?symbol=SPY&contractType=CALL&strikeCount=10&range=OTM
```

**Response:**
```json
{
  "symbol": "SPY",
  "status": "SUCCESS",
  "underlying": {
    "symbol": "SPY",
    "description": "SPDR S&P 500 ETF Trust",
    "bid": 450.25,
    "ask": 450.30,
    "last": 450.28,
    "mark": 450.275
  },
  "strategy": "SINGLE",
  "callExpDateMap": {
    "2025-01-17:7": {
      "455.0": [
        {
          "putCall": "CALL",
          "symbol": "SPY_011725C455",
          "description": "SPY Jan 17 2025 455 Call",
          "bid": 2.50,
          "ask": 2.55,
          "last": 2.52,
          "mark": 2.525,
          "bidSize": 50,
          "askSize": 50,
          "lastSize": 10,
          "highPrice": 2.80,
          "lowPrice": 2.40,
          "openPrice": 2.45,
          "closePrice": 2.48,
          "totalVolume": 5230,
          "openInterest": 12450,
          "volatility": 18.5,
          "delta": 0.45,
          "gamma": 0.05,
          "theta": -0.12,
          "vega": 0.25,
          "rho": 0.08,
          "timeValue": 2.52,
          "theoreticalOptionValue": 2.525,
          "theoreticalVolatility": 18.5,
          "strikePrice": 455.0,
          "expirationDate": "2025-01-17",
          "daysToExpiration": 7,
          "expirationType": "R",
          "multiplier": 100,
          "settlementType": "P",
          "deliverableNote": "",
          "percentChange": 1.6,
          "markChange": 0.045,
          "markPercentChange": 1.8,
          "inTheMoney": false
        }
      ]
    }
  }
}
```

**Use Cases:**
- Options strategy selection (iron condors, straddles, etc.)
- Volatility analysis
- Greeks calculation for risk management
- Options arbitrage detection

---

## 3. Historical Data (OHLCV Bars)

### Get Price History
```http
GET /marketdata/v1/pricehistory?symbol={symbol}
```

**Parameters:**
- `symbol` (required): Symbol
- `periodType`: day, month, year, ytd (default: day)
- `period`: Number of periods (depends on periodType)
  - day: 1, 2, 3, 4, 5, 10
  - month: 1, 2, 3, 6
  - year: 1, 2, 3, 5, 10, 15, 20
  - ytd: 1
- `frequencyType`: minute, daily, weekly, monthly
  - Valid for day: minute
  - Valid for month: daily, weekly
  - Valid for year: daily, weekly, monthly
  - Valid for ytd: daily, weekly
- `frequency`: Interval (depends on frequencyType)
  - minute: 1, 5, 10, 15, 30
  - daily: 1
  - weekly: 1
  - monthly: 1
- `startDate`: Start date (epoch milliseconds)
- `endDate`: End date (epoch milliseconds)
- `needExtendedHoursData`: true/false (default: true)
- `needPreviousClose`: true/false (default: false)

**Example:**
```bash
# Last 10 days, daily bars
GET /marketdata/v1/pricehistory?symbol=AAPL&periodType=day&period=10&frequencyType=daily&frequency=1

# Intraday 5-minute bars
GET /marketdata/v1/pricehistory?symbol=SPY&periodType=day&period=1&frequencyType=minute&frequency=5
```

**Response:**
```json
{
  "candles": [
    {
      "open": 180.00,
      "high": 182.50,
      "low": 179.25,
      "close": 181.75,
      "volume": 52340000,
      "datetime": 1699564800000
    },
    {
      "open": 181.80,
      "high": 183.00,
      "low": 180.50,
      "close": 182.25,
      "volume": 48230000,
      "datetime": 1699651200000
    }
  ],
  "symbol": "AAPL",
  "empty": false
}
```

**Use Cases:**
- Backtesting strategies
- Technical analysis (RSI, MACD, moving averages)
- Trend detection
- Historical volatility calculation
- Chart data for dashboards

---

## 4. Market Movers

### Get Market Movers
```http
GET /marketdata/v1/movers/{index}
```

**Parameters:**
- `index` (required): $DJI, $COMPX, $SPX
- `sort`: VOLUME, TRADES, PERCENT_CHANGE_UP, PERCENT_CHANGE_DOWN
- `frequency`: 0 (all day), 1 (1 min), 5 (5 min), 10 (10 min), 30 (30 min), 60 (60 min)

**Example:**
```bash
GET /marketdata/v1/movers/$SPX?sort=PERCENT_CHANGE_UP&frequency=0
```

**Response:**
```json
{
  "screeners": [
    {
      "symbol": "NVDA",
      "description": "NVIDIA Corporation",
      "lastPrice": 485.50,
      "netChange": 12.75,
      "netPercentChange": 2.70,
      "totalVolume": 35000000,
      "trades": 125000
    },
    {
      "symbol": "TSLA",
      "description": "Tesla Inc",
      "lastPrice": 242.80,
      "netChange": 6.30,
      "netPercentChange": 2.66,
      "totalVolume": 42000000,
      "trades": 98000
    }
  ]
}
```

**Use Cases:**
- Identify trending stocks
- Sector momentum analysis
- Volatility detection
- News-driven trading opportunities

---

## 5. Market Hours

### Get Market Hours
```http
GET /marketdata/v1/markets?markets={market}&date={date}
```

**Parameters:**
- `markets`: equity, option, bond, future, forex
- `date`: yyyy-MM-dd (default: today)

**Example:**
```bash
GET /marketdata/v1/markets?markets=equity,option&date=2025-11-09
```

**Response:**
```json
{
  "equity": {
    "EQ": {
      "date": "2025-11-09",
      "marketType": "EQUITY",
      "exchange": "NULL",
      "category": "NULL",
      "product": "EQ",
      "productName": "equity",
      "isOpen": true,
      "sessionHours": {
        "preMarket": [
          {
            "start": "2025-11-09T07:00:00-05:00",
            "end": "2025-11-09T09:30:00-05:00"
          }
        ],
        "regularMarket": [
          {
            "start": "2025-11-09T09:30:00-05:00",
            "end": "2025-11-09T16:00:00-05:00"
          }
        ],
        "postMarket": [
          {
            "start": "2025-11-09T16:00:00-05:00",
            "end": "2025-11-09T20:00:00-05:00"
          }
        ]
      }
    }
  }
}
```

**Use Cases:**
- Validate market open before trading
- Schedule trading hours
- Handle market holidays
- Extended hours trading decisions

---

## Rate Limits

**Schwab API Rate Limits:**
- **120 calls per minute** per account
- Burst allowance for short peaks
- Rate limit headers included in response:
  - `X-RateLimit-Limit`: Total allowed per minute
  - `X-RateLimit-Remaining`: Remaining calls
  - `X-RateLimit-Reset`: Time when limit resets

**Our Implementation:**
- Track call count per minute
- Implement exponential backoff on rate limit errors
- Queue requests when approaching limit
- Log rate limit violations

---

## Caching Strategy

**DuckDB Caching Schema:**

```sql
CREATE TABLE market_data_cache (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL,  -- 'quote', 'options', 'historical'
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    UNIQUE(symbol, data_type)
);

CREATE INDEX idx_cache_expires ON market_data_cache(expires_at);
```

**Cache TTLs:**
- Quotes: 1 second (real-time)
- Options chains: 5 seconds
- Historical data: 1 hour
- Market hours: 24 hours
- Movers: 1 minute

**Benefits:**
- Reduce API calls (stay under 120/min limit)
- Faster response times
- Lower latency for frequent queries
- Automatic cache expiration

---

## Implementation Status

### ✅ Completed
- Data structures defined (Quote, OptionContract, OHLCVBar, etc.)
- Test framework created (test_market_data.py, 24K)
- DuckDB caching schema designed

### 🟡 In Progress
- getQuote() / getQuotes() implementation
- getOptionChain() implementation
- getHistoricalData() implementation
- Rate limiting logic
- Cache integration

### ⏸️ Pending
- getMovers() implementation
- getMarketHours() implementation
- WebSocket streaming (optional for V1)

---

## Usage Examples

### Example 1: Get Current Quote for Trading
```cpp
auto schwab = SchwabClient(oauth_config);

// Get quote for sector ETF
auto quote_result = schwab.getQuote("XLE");
if (quote_result) {
    auto const& quote = *quote_result;
    Logger::info("XLE: ${} (bid: ${}, ask: ${})",
                quote.last, quote.bid, quote.ask);

    // Use in trading decision
    if (quote.last > 80.00) {
        // Place buy order...
    }
}
```

### Example 2: Get Options Chain for Strategy
```cpp
// Get SPY options chain
auto chain_request = OptionsChainRequest::forSymbol("SPY");
chain_request.contract_type = "ALL";
chain_request.strike_count = 10;
chain_request.range = "OTM";

auto chain_result = schwab.getOptionChain(chain_request);
if (chain_result) {
    auto const& chain = *chain_result;
    Logger::info("SPY options: {} calls, {} puts",
                chain.calls.size(), chain.puts.size());

    // Find iron condor strikes
    auto ic_strategy = findIronCondorStrikes(chain);
}
```

### Example 3: Get Historical Data for Backtesting
```cpp
// Get 1 year of daily data
auto hist_result = schwab.getHistoricalData(
    "SPY",
    "year",     // periodType
    1,          // period
    "daily",    // frequencyType
    1           // frequency
);

if (hist_result) {
    auto const& bars = hist_result->bars;
    Logger::info("Retrieved {} bars for SPY", bars.size());

    // Calculate indicators
    auto sma_200 = calculateSMA(bars, 200);
    auto rsi = calculateRSI(bars, 14);
}
```

---

## Integration with Trading Strategies

**SectorRotationStrategy Integration:**

```cpp
// 1. Get current quotes for all sector ETFs
std::vector<std::string> sector_etfs = {
    "XLE", "XLB", "XLI", "XLY", "XLP",
    "XLV", "XLF", "XLK", "XLC", "XLU", "XLRE"
};

auto quotes_result = schwab.getQuotes(sector_etfs);

// 2. Update StrategyContext with current prices
if (quotes_result) {
    for (auto const& quote : *quotes_result) {
        // IMPORTANT: Validate symbols before using them
        if (utils::isValidStockSymbol(quote.symbol)) {
            context.current_quotes[quote.symbol] = quote;
        }
    }
}

// 3. Generate trading signals
auto signals = sector_rotation_strategy.generateSignals(context);

// 4. Execute trades based on signals (with manual position checks!)
for (auto const& signal : signals) {
    if (shouldExecute(signal)) {
        schwab.placeOrder(createOrder(signal));
    }
}
```

---

## Error Handling

**Common Errors:**

| HTTP Code | Error | Handling |
|-----------|-------|----------|
| 401 | Unauthorized | Refresh OAuth token |
| 429 | Rate Limit | Exponential backoff, queue request |
| 400 | Invalid Symbol | Validate symbol, log error |
| 404 | Not Found | Symbol doesn't exist |
| 500 | Server Error | Retry with exponential backoff |
| 503 | Service Unavailable | Wait and retry |

**Implementation:**
```cpp
auto getQuote(std::string const& symbol) -> Result<Quote> {
    // Check cache first
    if (auto cached = cache.get(symbol, "quote")) {
        return *cached;
    }

    // Make API call with retry logic
    for (int retry = 0; retry < 3; ++retry) {
        auto response = makeRequest(
            fmt::format("/marketdata/v1/quotes/{}", symbol)
        );

        if (response.status == 429) {
            // Rate limit - wait and retry
            std::this_thread::sleep_for(std::chrono::seconds(1 << retry));
            continue;
        }

        if (response.status == 401) {
            // Token expired - refresh and retry
            token_manager.refreshAccessToken();
            continue;
        }

        if (response.status == 200) {
            auto quote = parseQuote(response.body);
            cache.set(symbol, "quote", quote, 1s);
            return quote;
        }

        return makeError<Quote>(
            ErrorCode::ApiError,
            fmt::format("Failed to get quote: {}", response.error)
        );
    }

    return makeError<Quote>(ErrorCode::MaxRetriesExceeded, "Max retries exceeded");
}
```

---

## Summary

**Schwab provides ALL the market data we need:**
- ✅ Real-time quotes (FREE with account)
- ✅ Options chains with greeks
- ✅ Historical OHLCV data
- ✅ Market movers and hours
- ✅ Integrated with trading API
- ✅ No third-party data subscriptions needed

**Next Steps:**
1. Complete getQuote() / getQuotes() implementation
2. Complete getOptionChain() implementation
3. Complete getHistoricalData() implementation
4. Test with real Schwab API credentials
5. Validate rate limiting under load

---

**Last Updated:** November 9, 2025
**API Documentation:** https://developer.schwab.com/products/trader-api--individual/details/documentation
**Status:** Ready for implementation
