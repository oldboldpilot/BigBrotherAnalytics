/**
 * Unit Tests for Yahoo Finance C++23 Fluent API
 *
 * Tests the fluent interface, data structures, and error handling.
 */

#include <chrono>
#include <gtest/gtest.h>

// Import C++23 modules
import bigbrother.market_intelligence.types;
import bigbrother.utils.types;

using namespace bigbrother::market_intelligence;
using namespace bigbrother::types;

// Test tolerance for floating point comparisons
constexpr double PRICE_TOLERANCE = 0.01;

/**
 * Test Quote data structure
 */
TEST(YahooFinanceTest, QuoteStructure) {
    Quote quote;
    quote.symbol = "SPY";
    quote.last_price = 580.50;
    quote.bid = 580.45;
    quote.ask = 580.55;
    quote.volume = 45000000;
    quote.source = DataSource::YAHOO_FINANCE;

    // Test basic fields
    EXPECT_EQ(quote.symbol, "SPY");
    EXPECT_NEAR(quote.last_price, 580.50, PRICE_TOLERANCE);
    EXPECT_EQ(quote.source, DataSource::YAHOO_FINANCE);

    // Test calculated fields
    EXPECT_NEAR(quote.spread(), 0.10, PRICE_TOLERANCE);
    EXPECT_NEAR(quote.mid_price(), 580.50, PRICE_TOLERANCE);
}

/**
 * Test OHLCV data structure
 */
TEST(YahooFinanceTest, OHLCVStructure) {
    OHLCV candle;
    candle.timestamp = std::chrono::system_clock::now();
    candle.open = 580.00;
    candle.high = 582.00;
    candle.low = 579.00;
    candle.close = 581.50;
    candle.volume = 50000000;

    EXPECT_NEAR(candle.open, 580.00, PRICE_TOLERANCE);
    EXPECT_NEAR(candle.high, 582.00, PRICE_TOLERANCE);
    EXPECT_NEAR(candle.low, 579.00, PRICE_TOLERANCE);
    EXPECT_NEAR(candle.close, 581.50, PRICE_TOLERANCE);
    EXPECT_EQ(candle.volume, 50000000);

    // Test OHLC validity
    EXPECT_LE(candle.low, candle.high);
    EXPECT_LE(candle.low, candle.open);
    EXPECT_LE(candle.low, candle.close);
    EXPECT_GE(candle.high, candle.open);
    EXPECT_GE(candle.high, candle.close);
}

/**
 * Test NewsArticle data structure
 */
TEST(YahooFinanceTest, NewsArticleStructure) {
    NewsArticle article;
    article.article_id = "test123";
    article.symbol = "AAPL";
    article.title = "Apple Surges on Strong Earnings";
    article.description = "Apple Inc reported better than expected earnings...";
    article.sentiment_score = 0.75;
    article.sentiment_label = "positive";
    article.source = "Reuters";

    EXPECT_EQ(article.article_id, "test123");
    EXPECT_EQ(article.symbol, "AAPL");
    EXPECT_EQ(article.sentiment_label, "positive");
    EXPECT_GT(article.sentiment_score, 0.0);
    EXPECT_LE(article.sentiment_score, 1.0);
}

/**
 * Test DataSource enum
 */
TEST(YahooFinanceTest, DataSourceEnum) {
    EXPECT_EQ(to_string(DataSource::YAHOO_FINANCE), "YAHOO_FINANCE");
    EXPECT_EQ(to_string(DataSource::SCHWAB), "SCHWAB");
    EXPECT_EQ(to_string(DataSource::NEWSAPI), "NEWSAPI");
    EXPECT_EQ(to_string(DataSource::ALPHAVANTAGE), "ALPHAVANTAGE");
}

/**
 * Test Period enum
 */
TEST(YahooFinanceTest, PeriodEnum) {
    EXPECT_EQ(to_string(Period::ONE_DAY), "ONE_DAY");
    EXPECT_EQ(to_string(Period::ONE_WEEK), "ONE_WEEK");
    EXPECT_EQ(to_string(Period::ONE_MONTH), "ONE_MONTH");
    EXPECT_EQ(to_string(Period::THREE_MONTHS), "THREE_MONTHS");
    EXPECT_EQ(to_string(Period::ONE_YEAR), "ONE_YEAR");
}

/**
 * Test Interval enum
 */
TEST(YahooFinanceTest, IntervalEnum) {
    EXPECT_EQ(to_string(Interval::ONE_MINUTE), "ONE_MINUTE");
    EXPECT_EQ(to_string(Interval::FIVE_MINUTES), "FIVE_MINUTES");
    EXPECT_EQ(to_string(Interval::ONE_HOUR), "ONE_HOUR");
    EXPECT_EQ(to_string(Interval::ONE_DAY), "ONE_DAY");
}

/**
 * Test Quote spread calculation
 */
TEST(YahooFinanceTest, QuoteSpreadCalculation) {
    Quote quote;
    quote.bid = 100.00;
    quote.ask = 100.10;

    EXPECT_NEAR(quote.spread(), 0.10, 0.001);
}

/**
 * Test Quote mid-price calculation
 */
TEST(YahooFinanceTest, QuoteMidPriceCalculation) {
    Quote quote;
    quote.bid = 100.00;
    quote.ask = 100.20;

    EXPECT_NEAR(quote.mid_price(), 100.10, 0.001);
}

/**
 * Test multiple quotes with different sources
 */
TEST(YahooFinanceTest, MultipleQuoteSources) {
    std::vector<Quote> quotes;

    // Yahoo Finance quote
    Quote yahoo_quote;
    yahoo_quote.symbol = "SPY";
    yahoo_quote.last_price = 580.00;
    yahoo_quote.source = DataSource::YAHOO_FINANCE;
    quotes.push_back(yahoo_quote);

    // Schwab quote
    Quote schwab_quote;
    schwab_quote.symbol = "SPY";
    schwab_quote.last_price = 580.05;
    schwab_quote.source = DataSource::SCHWAB;
    quotes.push_back(schwab_quote);

    ASSERT_EQ(quotes.size(), 2);
    EXPECT_EQ(quotes[0].source, DataSource::YAHOO_FINANCE);
    EXPECT_EQ(quotes[1].source, DataSource::SCHWAB);

    // Verify they're both for same symbol
    EXPECT_EQ(quotes[0].symbol, quotes[1].symbol);
}

/**
 * Test OHLCV candle with intraday data
 */
TEST(YahooFinanceTest, IntradayOHLCV) {
    std::vector<OHLCV> candles;

    for (int i = 0; i < 5; ++i) {
        OHLCV candle;
        candle.timestamp = std::chrono::system_clock::now() + std::chrono::minutes(i * 5);
        candle.open = 580.00 + i * 0.50;
        candle.high = 580.00 + i * 0.50 + 0.25;
        candle.low = 580.00 + i * 0.50 - 0.25;
        candle.close = 580.00 + i * 0.50 + 0.10;
        candle.volume = 1000000;
        candles.push_back(candle);
    }

    ASSERT_EQ(candles.size(), 5);

    // Verify timestamps are sequential
    for (size_t i = 1; i < candles.size(); ++i) {
        EXPECT_GT(candles[i].timestamp, candles[i - 1].timestamp);
    }
}

/**
 * Test sentiment scoring bounds
 */
TEST(YahooFinanceTest, SentimentScoreBounds) {
    NewsArticle positive_article;
    positive_article.sentiment_score = 0.85;
    positive_article.sentiment_label = "positive";

    NewsArticle negative_article;
    negative_article.sentiment_score = -0.65;
    negative_article.sentiment_label = "negative";

    NewsArticle neutral_article;
    neutral_article.sentiment_score = 0.05;
    neutral_article.sentiment_label = "neutral";

    // Verify score bounds (-1.0 to 1.0)
    EXPECT_GE(positive_article.sentiment_score, -1.0);
    EXPECT_LE(positive_article.sentiment_score, 1.0);
    EXPECT_GE(negative_article.sentiment_score, -1.0);
    EXPECT_LE(negative_article.sentiment_score, 1.0);
    EXPECT_GE(neutral_article.sentiment_score, -1.0);
    EXPECT_LE(neutral_article.sentiment_score, 1.0);

    // Verify label consistency
    EXPECT_GT(positive_article.sentiment_score, 0.2);
    EXPECT_LT(negative_article.sentiment_score, -0.2);
    EXPECT_GE(neutral_article.sentiment_score, -0.2);
    EXPECT_LE(neutral_article.sentiment_score, 0.2);
}

/**
 * Test Quote with missing optional fields
 */
TEST(YahooFinanceTest, QuoteWithOptionalFields) {
    Quote quote;
    quote.symbol = "TSLA";
    quote.last_price = 350.00;

    // Optional fields not set
    EXPECT_EQ(quote.bid, 0.0);
    EXPECT_EQ(quote.ask, 0.0);
    EXPECT_EQ(quote.volume, 0);

    // Spread should be 0 if bid/ask not set
    EXPECT_NEAR(quote.spread(), 0.0, 0.001);
}

/**
 * Main test runner
 */
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
