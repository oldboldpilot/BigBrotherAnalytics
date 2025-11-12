#!/usr/bin/env python3
"""
BigBrotherAnalytics - Schwab Account API Tests

Comprehensive test suite for account data endpoints:
- Account information retrieval
- Position tracking and monitoring
- Transaction history
- Balance queries
- Portfolio analytics
- DuckDB persistence

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "build"))

import unittest
import time
from datetime import datetime, timedelta
import duckdb

# Import C++ bindings (will be available after compilation)
try:
    import bigbrother_schwab as schwab
    import bigbrother_duckdb as db
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False
    print("Warning: C++ bindings not available. Compile first.")


class TestAccountEndpoints(unittest.TestCase):
    """Test Schwab account API endpoints"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not BINDINGS_AVAILABLE:
            cls.skipTest(cls, "C++ bindings not available")

        # Create test configuration
        cls.config = {
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "redirect_uri": "https://localhost:8080/callback"
        }

        # Initialize Schwab client
        cls.client = schwab.SchwabClient(cls.config)

        # Test account ID
        cls.test_account_id = "XXXX1234"

        print("\n" + "="*70)
        print("ACCOUNT API TESTS - Testing $30K Account Access")
        print("="*70)

    def test_01_get_accounts(self):
        """Test GET /trader/v1/accounts"""
        print("\n[TEST] Fetching all accounts...")

        result = self.client.account().getAccounts()

        self.assertTrue(result.is_ok(), "Failed to get accounts")

        accounts = result.value()
        self.assertIsInstance(accounts, list)
        self.assertGreater(len(accounts), 0, "No accounts returned")

        # Verify account structure
        account = accounts[0]
        self.assertIn("account_id", account)
        self.assertIn("account_hash", account)
        self.assertIn("account_type", account)

        print(f"  ✓ Retrieved {len(accounts)} account(s)")
        print(f"  ✓ Account Type: {account['account_type']}")
        print(f"  ✓ Day Trader: {account.get('is_day_trader', False)}")

    def test_02_get_account_details(self):
        """Test GET /trader/v1/accounts/{accountHash}"""
        print(f"\n[TEST] Fetching account details for {self.test_account_id}...")

        result = self.client.account().getAccount(self.test_account_id)

        self.assertTrue(result.is_ok(), "Failed to get account details")

        account = result.value()
        self.assertEqual(account["account_id"], self.test_account_id)
        self.assertIsNotNone(account.get("account_hash"))

        print(f"  ✓ Account ID: {account['account_id']}")
        print(f"  ✓ Account Hash: {account['account_hash'][:20]}...")
        print(f"  ✓ Account Type: {account['account_type']}")

    def test_03_get_positions(self):
        """Test GET /trader/v1/accounts/{accountHash}/positions"""
        print(f"\n[TEST] Fetching positions for {self.test_account_id}...")

        result = self.client.account().getPositions(self.test_account_id)

        self.assertTrue(result.is_ok(), "Failed to get positions")

        positions = result.value()
        self.assertIsInstance(positions, list)

        print(f"  ✓ Retrieved {len(positions)} position(s)")

        if len(positions) > 0:
            pos = positions[0]
            print(f"\n  Example Position:")
            print(f"    Symbol: {pos['symbol']}")
            print(f"    Quantity: {pos['quantity']}")
            print(f"    Average Cost: ${pos['average_cost']:.2f}")
            print(f"    Current Price: ${pos['current_price']:.2f}")
            print(f"    Market Value: ${pos['market_value']:.2f}")
            print(f"    Unrealized P/L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.2f}%)")

    def test_04_get_specific_position(self):
        """Test getPosition for specific symbol"""
        print(f"\n[TEST] Fetching specific position...")

        # First get all positions
        result = self.client.account().getPositions(self.test_account_id)
        self.assertTrue(result.is_ok())

        positions = result.value()
        if len(positions) > 0:
            symbol = positions[0]["symbol"]

            # Get specific position
            pos_result = self.client.account().getPosition(self.test_account_id, symbol)
            self.assertTrue(pos_result.is_ok())

            position = pos_result.value()
            self.assertEqual(position["symbol"], symbol)

            print(f"  ✓ Found position for {symbol}")
            print(f"    P/L: ${position['unrealized_pnl']:.2f}")
        else:
            print("  ⊗ No positions to test with")

    def test_05_get_balances(self):
        """Test getBalances endpoint"""
        print(f"\n[TEST] Fetching account balances...")

        result = self.client.account().getBalances(self.test_account_id)

        self.assertTrue(result.is_ok(), "Failed to get balances")

        balance = result.value()

        # Verify $30K account
        self.assertAlmostEqual(balance["total_equity"], 30000.0, delta=100.0)
        self.assertGreater(balance["buying_power"], 0)

        print(f"\n  Account Balance Summary:")
        print(f"    Total Equity: ${balance['total_equity']:,.2f}")
        print(f"    Cash: ${balance['cash']:,.2f}")
        print(f"    Cash Available: ${balance['cash_available']:,.2f}")
        print(f"    Buying Power: ${balance['buying_power']:,.2f}")
        print(f"    Day Trading BP: ${balance['day_trading_buying_power']:,.2f}")
        print(f"    Margin Balance: ${balance['margin_balance']:,.2f}")
        print(f"    Long Market Value: ${balance['long_market_value']:,.2f}")
        print(f"    Margin Usage: {balance.get('margin_usage_percent', 0):.2f}%")

        # Safety checks
        self.assertFalse(balance.get("has_margin_call", False),
                        "⚠ MARGIN CALL DETECTED!")

    def test_06_get_buying_power(self):
        """Test getBuyingPower endpoint"""
        print(f"\n[TEST] Fetching buying power...")

        result = self.client.account().getBuyingPower(self.test_account_id)

        self.assertTrue(result.is_ok())

        buying_power = result.value()
        self.assertGreater(buying_power, 0, "Buying power should be positive")

        print(f"  ✓ Buying Power: ${buying_power:,.2f}")

    def test_07_get_transactions(self):
        """Test GET /trader/v1/accounts/{accountHash}/transactions"""
        print(f"\n[TEST] Fetching transaction history...")

        # Last 30 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        result = self.client.account().getTransactions(
            self.test_account_id,
            start_date,
            end_date
        )

        self.assertTrue(result.is_ok(), "Failed to get transactions")

        transactions = result.value()
        self.assertIsInstance(transactions, list)

        print(f"  ✓ Retrieved {len(transactions)} transaction(s)")
        print(f"    Date Range: {start_date} to {end_date}")

        if len(transactions) > 0:
            txn = transactions[0]
            print(f"\n  Example Transaction:")
            print(f"    ID: {txn['transaction_id']}")
            print(f"    Symbol: {txn.get('symbol', 'N/A')}")
            print(f"    Type: {txn['type']}")
            print(f"    Amount: ${txn['net_amount']:.2f}")
            print(f"    Date: {txn['transaction_date']}")


class TestPortfolioAnalytics(unittest.TestCase):
    """Test portfolio analytics and risk metrics"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not BINDINGS_AVAILABLE:
            cls.skipTest(cls, "C++ bindings not available")

        cls.client = schwab.SchwabClient({
            "client_id": "test_client_id",
            "client_secret": "test_secret"
        })
        cls.test_account_id = "XXXX1234"

        print("\n" + "="*70)
        print("PORTFOLIO ANALYTICS TESTS")
        print("="*70)

    def test_01_portfolio_summary(self):
        """Test getPortfolioSummary"""
        print("\n[TEST] Calculating portfolio summary...")

        result = self.client.account().getPortfolioSummary(self.test_account_id)

        self.assertTrue(result.is_ok())

        summary = result.value()

        print(f"\n  Portfolio Summary:")
        print(f"    Total Equity: ${summary['total_equity']:,.2f}")
        print(f"    Total Market Value: ${summary['total_market_value']:,.2f}")
        print(f"    Total Cost Basis: ${summary['total_cost_basis']:,.2f}")
        print(f"    Unrealized P/L: ${summary['total_unrealized_pnl']:,.2f} ({summary['total_unrealized_pnl_percent']:.2f}%)")
        print(f"    Day P/L: ${summary['total_day_pnl']:,.2f} ({summary['total_day_pnl_percent']:.2f}%)")
        print(f"    Position Count: {summary['position_count']}")
        print(f"    Long Positions: {summary['long_position_count']}")
        print(f"    Short Positions: {summary['short_position_count']}")
        print(f"    Largest Position: {summary['largest_position_percent']:.2f}%")
        print(f"    Concentration Index: {summary['portfolio_concentration']:.4f}")

    def test_02_sector_exposure(self):
        """Test sector exposure calculation"""
        print("\n[TEST] Calculating sector exposure...")

        # Would need actual sector mapping
        # This is a placeholder test
        print("  ⊗ Sector mapping integration pending")

    def test_03_risk_metrics(self):
        """Test risk metric calculation"""
        print("\n[TEST] Calculating risk metrics...")

        # Would integrate with PortfolioAnalyzer
        # This is a placeholder test
        print("  ⊗ Risk metrics integration pending")


class TestDuckDBPersistence(unittest.TestCase):
    """Test DuckDB position tracking"""

    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        if not BINDINGS_AVAILABLE:
            cls.skipTest(cls, "C++ bindings not available")

        cls.db_path = "test_account.duckdb"
        cls.conn = db.connect(cls.db_path)

        # Load schema
        schema_path = project_root / "scripts" / "account_schema.sql"
        if schema_path.exists():
            with open(schema_path) as f:
                schema_sql = f.read()
                # Execute schema in chunks (split by semicolon)
                for stmt in schema_sql.split(";"):
                    if stmt.strip():
                        try:
                            cls.conn.execute_void(stmt)
                        except:
                            pass  # Some statements may fail if already exist

        print("\n" + "="*70)
        print("DUCKDB PERSISTENCE TESTS")
        print("="*70)

    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        if hasattr(cls, "db_path") and os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def test_01_create_tables(self):
        """Test table creation"""
        print("\n[TEST] Verifying DuckDB schema...")

        tables = self.conn.list_tables()

        expected_tables = [
            "accounts", "account_balances", "positions",
            "position_history", "position_changes",
            "transactions", "portfolio_snapshots"
        ]

        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not found")

        print(f"  ✓ All {len(expected_tables)} tables created successfully")

    def test_02_insert_account(self):
        """Test account insertion"""
        print("\n[TEST] Inserting test account...")

        query = """
        INSERT INTO accounts (account_id, account_hash, account_type, is_day_trader)
        VALUES ('TEST1234', 'HASH_TEST1234', 'MARGIN', TRUE)
        ON CONFLICT DO NOTHING
        """

        self.conn.execute_void(query)

        # Verify
        result = self.conn.execute("SELECT * FROM accounts WHERE account_id = 'TEST1234'")
        data = result.to_dict()

        self.assertEqual(len(data["account_id"]), 1)
        print("  ✓ Account inserted successfully")

    def test_03_insert_position(self):
        """Test position insertion"""
        print("\n[TEST] Inserting test position...")

        query = """
        INSERT INTO positions (
            account_id, symbol, asset_type, quantity,
            average_cost, current_price, market_value, cost_basis,
            unrealized_pnl, unrealized_pnl_percent
        ) VALUES (
            'TEST1234', 'SPY', 'EQUITY', 10,
            580.0, 590.0, 5900.0, 5800.0,
            100.0, 1.72
        )
        ON CONFLICT (account_id, symbol) DO UPDATE SET
            quantity = EXCLUDED.quantity,
            current_price = EXCLUDED.current_price
        """

        self.conn.execute_void(query)

        # Verify
        result = self.conn.execute("SELECT * FROM positions WHERE symbol = 'SPY'")
        data = result.to_dict()

        self.assertEqual(len(data["symbol"]), 1)
        self.assertEqual(data["quantity"][0], 10)

        print("  ✓ Position inserted successfully")
        print(f"    Symbol: SPY, Qty: 10, P/L: $100.00")

    def test_04_query_views(self):
        """Test analytical views"""
        print("\n[TEST] Querying analytical views...")

        # Test current positions view
        result = self.conn.execute("SELECT * FROM v_current_positions")
        print(f"  ✓ v_current_positions: {len(result)} rows")

        # Test latest balance view
        # Note: May be empty if no balances inserted
        result = self.conn.execute("SELECT * FROM v_latest_balance")
        print(f"  ✓ v_latest_balance: {len(result)} rows")


class TestPositionTracker(unittest.TestCase):
    """Test automatic position tracking"""

    @classmethod
    def setUpClass(cls):
        """Set up position tracker"""
        if not BINDINGS_AVAILABLE:
            cls.skipTest(cls, "C++ bindings not available")

        print("\n" + "="*70)
        print("POSITION TRACKER TESTS")
        print("="*70)

    def test_01_position_tracker_creation(self):
        """Test PositionTracker instantiation"""
        print("\n[TEST] Creating PositionTracker...")

        # Would create PositionTracker instance
        # This is a placeholder test
        print("  ⊗ PositionTracker integration pending (C++ compilation required)")

    def test_02_auto_refresh(self):
        """Test automatic 30-second refresh"""
        print("\n[TEST] Testing automatic position refresh...")

        # Would start tracker and verify refresh
        # This is a placeholder test
        print("  ⊗ Auto-refresh testing pending")


def run_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("BigBrotherAnalytics - Schwab Account API Test Suite")
    print("Testing $30K Account Data Endpoints")
    print("="*70)

    if not BINDINGS_AVAILABLE:
        print("\n⚠ ERROR: C++ bindings not available")
        print("Please compile the project first:")
        print("  mkdir -p build && cd build")
        print("  cmake .. && make")
        return 1

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAccountEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioAnalytics))
    suite.addTests(loader.loadTestsFromTestCase(TestDuckDBPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionTracker))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
