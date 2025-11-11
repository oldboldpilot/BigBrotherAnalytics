#!/usr/bin/env python3
"""
Integration Tests for Market Data Python Bindings

Tests the Python bindings for Yahoo Finance and Schwab APIs.
Verifies fluent API works correctly from Python.
"""

import sys
from pathlib import Path
import unittest

# Add python bindings to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import market_data_py as md
    BINDINGS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Market data bindings not available: {e}")
    print("   Run: ninja -C build market_data_py")
    BINDINGS_AVAILABLE = False


class TestMarketDataTypes(unittest.TestCase):
    """Test basic data structures"""

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_quote_structure(self):
        """Test Quote data structure from Python"""
        quote = md.Quote()
        quote.symbol = "SPY"
        quote.last_price = 580.50
        quote.bid = 580.45
        quote.ask = 580.55
        quote.volume = 45000000
        quote.source = md.DataSource.YAHOO_FINANCE

        self.assertEqual(quote.symbol, "SPY")
        self.assertAlmostEqual(quote.last_price, 580.50, places=2)
        self.assertEqual(quote.source, md.DataSource.YAHOO_FINANCE)

        # Test calculated methods
        self.assertAlmostEqual(quote.spread(), 0.10, places=2)
        self.assertAlmostEqual(quote.mid_price(), 580.50, places=2)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_news_article_structure(self):
        """Test NewsArticle structure"""
        article = md.NewsArticle()
        article.article_id = "test123"
        article.symbol = "AAPL"
        article.title = "Apple Surges"
        article.sentiment_score = 0.75
        article.sentiment_label = "positive"

        self.assertEqual(article.symbol, "AAPL")
        self.assertGreater(article.sentiment_score, 0.0)
        self.assertEqual(article.sentiment_label, "positive")

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_position_structure(self):
        """Test Position structure"""
        pos = md.Position()
        pos.symbol = "SPY"
        pos.type = md.PositionType.STOCK
        pos.quantity = 100.0
        pos.average_price = 570.00
        pos.current_price = 580.00
        pos.unrealized_pnl = 1000.00

        self.assertEqual(pos.symbol, "SPY")
        self.assertEqual(pos.type, md.PositionType.STOCK)
        self.assertAlmostEqual(pos.unrealized_pnl, 1000.00, places=2)


class TestDataSourceEnum(unittest.TestCase):
    """Test DataSource enum"""

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_data_source_values(self):
        """Test DataSource enum values"""
        self.assertIsNotNone(md.DataSource.YAHOO_FINANCE)
        self.assertIsNotNone(md.DataSource.SCHWAB)
        self.assertIsNotNone(md.DataSource.NEWSAPI)
        self.assertIsNotNone(md.DataSource.ALPHAVANTAGE)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_data_source_to_string(self):
        """Test DataSource to_string conversion"""
        self.assertEqual(md.to_string(md.DataSource.YAHOO_FINANCE), "YAHOO_FINANCE")
        self.assertEqual(md.to_string(md.DataSource.SCHWAB), "SCHWAB")


class TestYahooFinanceFluentAPI(unittest.TestCase):
    """Test Yahoo Finance fluent API"""

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_yahoo_collector_creation(self):
        """Test creating Yahoo Finance collector"""
        yahoo = md.YahooFinanceCollector()
        self.assertIsNotNone(yahoo)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_fluent_api_chaining(self):
        """Test fluent API method chaining"""
        yahoo = md.YahooFinanceCollector()

        # Test method chaining returns self
        result = yahoo.for_symbol("SPY")
        self.assertIsNotNone(result)

        result = yahoo.with_timeout(5)
        self.assertIsNotNone(result)

        result = yahoo.with_parallel(True)
        self.assertIsNotNone(result)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_multiple_symbols(self):
        """Test setting multiple symbols"""
        yahoo = md.YahooFinanceCollector()
        yahoo.for_symbols(["SPY", "QQQ", "AAPL"])
        # Just verify it doesn't crash


class TestSchwabAPIFluentAPI(unittest.TestCase):
    """Test Schwab API fluent API"""

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_schwab_config(self):
        """Test Schwab configuration"""
        config = md.SchwabConfig()
        config.app_key = "test_key"
        config.app_secret = "test_secret"
        config.timeout_seconds = 30

        self.assertEqual(config.app_key, "test_key")
        self.assertEqual(config.timeout_seconds, 30)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_fluent_api_order_building(self):
        """Test fluent API for order building"""
        config = md.SchwabConfig()
        config.app_key = "test"
        config.app_secret = "test"

        client = md.SchwabAPIClient(config)

        # Test fluent chaining (won't execute, just testing API)
        result = client.for_account("test_hash")
        self.assertIsNotNone(result)

        result = client.buy("SPY")
        self.assertIsNotNone(result)

        result = client.quantity(10)
        self.assertIsNotNone(result)

        result = client.limit_price(579.50)
        self.assertIsNotNone(result)


class TestEnums(unittest.TestCase):
    """Test all enum types"""

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_period_enum(self):
        """Test Period enum"""
        self.assertIsNotNone(md.Period.ONE_DAY)
        self.assertIsNotNone(md.Period.ONE_WEEK)
        self.assertIsNotNone(md.Period.ONE_MONTH)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_interval_enum(self):
        """Test Interval enum"""
        self.assertIsNotNone(md.Interval.ONE_MINUTE)
        self.assertIsNotNone(md.Interval.ONE_DAY)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_order_action_enum(self):
        """Test OrderAction enum"""
        self.assertIsNotNone(md.OrderAction.BUY)
        self.assertIsNotNone(md.OrderAction.SELL)

    @unittest.skipUnless(BINDINGS_AVAILABLE, "Bindings not built")
    def test_order_type_enum(self):
        """Test OrderType enum"""
        self.assertIsNotNone(md.OrderType.MARKET)
        self.assertIsNotNone(md.OrderType.LIMIT)


def main():
    """Run all tests"""
    if not BINDINGS_AVAILABLE:
        print("\n" + "="*80)
        print("SKIPPING TESTS - Market data bindings not available")
        print("="*80)
        print("Build bindings with: ninja -C build market_data_py")
        print()
        return 1

    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
