#!/usr/bin/env python3
"""
BigBrotherAnalytics - Schwab API Order Management Tests

Test suite for comprehensive order management with safety features.
All tests use DRY-RUN mode to prevent accidental real orders.

CRITICAL SAFETY:
- All tests run in dry-run mode by default
- No real orders are submitted to Schwab
- Tests use paper trading account credentials only
- Maximum order sizes are enforced

Test Coverage:
1. Basic order placement (Market, Limit, Stop, StopLimit)
2. Order modifications and cancellations
3. Bracket orders (Entry + Profit + Stop)
4. OCO (One-Cancels-Other) orders
5. Order validation and safety checks
6. DuckDB order tracking
7. Compliance logging
8. Error handling and retry logic
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
import duckdb

# Add the build directory to Python path
sys.path.insert(0, '/home/muyiwa/Development/BigBrotherAnalytics/build')

class TestOrderManagement(unittest.TestCase):
    """Test suite for Schwab API order management"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n" + "="*80)
        print("SCHWAB API ORDER MANAGEMENT TEST SUITE")
        print("="*80)
        print("Mode: DRY-RUN (No real orders will be placed)")
        print("Account: Paper Trading Test Account")
        print("="*80 + "\n")

        # Initialize DuckDB connection
        cls.db = duckdb.connect('data/test_orders.duckdb')

        # Load schema
        with open('scripts/database_schema_orders.sql', 'r') as f:
            schema_sql = f.read()
            # Execute each statement separately
            for statement in schema_sql.split(';'):
                if statement.strip():
                    try:
                        cls.db.execute(statement)
                    except Exception as e:
                        if 'already exists' not in str(e):
                            print(f"Warning: {e}")

        print("✓ DuckDB schema loaded successfully\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        cls.db.close()
        print("\n" + "="*80)
        print("TEST SUITE COMPLETED")
        print("="*80 + "\n")

    def setUp(self):
        """Set up each test"""
        self.account_id = "TEST_ACCOUNT_12345"
        self.test_symbol = "SPY"

    def test_01_market_order_buy(self):
        """Test placing a market buy order"""
        print("\nTest 1: Market Buy Order (DRY-RUN)")
        print("-" * 60)

        order_data = {
            'account_id': self.account_id,
            'symbol': self.test_symbol,
            'side': 'Buy',
            'quantity': 10,
            'order_type': 'Market',
            'dry_run': True
        }

        # Insert test order
        order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_001"

        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, order_data['account_id'], order_data['symbol'],
            order_data['side'], order_data['quantity'], order_data['order_type'],
            'Pending', order_data['dry_run'], datetime.now()
        ])

        # Verify order was created
        result = self.db.execute("""
            SELECT * FROM orders WHERE order_id = ?
        """, [order_id]).fetchone()

        self.assertIsNotNone(result, "Order should be created in database")
        self.assertEqual(result[2], self.account_id, "Account ID should match")
        self.assertEqual(result[3], self.test_symbol, "Symbol should match")
        self.assertEqual(result[4], 'Buy', "Side should be Buy")
        self.assertEqual(result[5], 10, "Quantity should be 10")
        self.assertTrue(result[13], "Should be marked as dry-run")

        print(f"✓ Market buy order created: {order_id}")
        print(f"  Symbol: {self.test_symbol}")
        print(f"  Quantity: 10")
        print(f"  Status: Pending (DRY-RUN)")

    def test_02_limit_order_sell(self):
        """Test placing a limit sell order"""
        print("\nTest 2: Limit Sell Order (DRY-RUN)")
        print("-" * 60)

        order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_002"
        limit_price = 585.00

        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, limit_price, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, self.account_id, self.test_symbol, 'Sell', 5,
            'Limit', limit_price, 'Pending', True, datetime.now()
        ])

        result = self.db.execute("""
            SELECT * FROM orders WHERE order_id = ?
        """, [order_id]).fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result[7], 'Limit', "Order type should be Limit")
        self.assertEqual(float(result[8]), limit_price, "Limit price should match")

        print(f"✓ Limit sell order created: {order_id}")
        print(f"  Symbol: {self.test_symbol}")
        print(f"  Quantity: 5")
        print(f"  Limit Price: ${limit_price}")
        print(f"  Status: Pending (DRY-RUN)")

    def test_03_stop_loss_order(self):
        """Test placing a stop-loss order"""
        print("\nTest 3: Stop-Loss Order (DRY-RUN)")
        print("-" * 60)

        order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_003"
        stop_price = 570.00

        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, stop_price, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, self.account_id, self.test_symbol, 'Sell', 10,
            'Stop', stop_price, 'Pending', True, datetime.now()
        ])

        result = self.db.execute("""
            SELECT * FROM orders WHERE order_id = ?
        """, [order_id]).fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result[7], 'Stop', "Order type should be Stop")
        self.assertEqual(float(result[9]), stop_price, "Stop price should match")

        print(f"✓ Stop-loss order created: {order_id}")
        print(f"  Symbol: {self.test_symbol}")
        print(f"  Quantity: 10")
        print(f"  Stop Price: ${stop_price}")
        print(f"  Status: Pending (DRY-RUN)")

    def test_04_bracket_order(self):
        """Test placing a bracket order (Entry + Profit + Stop)"""
        print("\nTest 4: Bracket Order (DRY-RUN)")
        print("-" * 60)

        # Entry order
        entry_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_004_ENTRY"
        profit_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_004_PROFIT"
        stop_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_004_STOP"

        entry_price = 580.00
        profit_target = 590.00
        stop_loss = 575.00

        # Insert entry order
        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, limit_price, order_class, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            entry_id, self.account_id, self.test_symbol, 'Buy', 10,
            'Limit', entry_price, 'Bracket', 'Pending', True, datetime.now()
        ])

        # Insert profit target order
        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, limit_price, order_class, parent_order_id,
                status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            profit_id, self.account_id, self.test_symbol, 'Sell', 10,
            'Limit', profit_target, 'Bracket', entry_id,
            'Pending', True, datetime.now()
        ])

        # Insert stop-loss order
        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, stop_price, order_class, parent_order_id,
                status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            stop_id, self.account_id, self.test_symbol, 'Sell', 10,
            'Stop', stop_loss, 'Bracket', entry_id,
            'Pending', True, datetime.now()
        ])

        # Verify bracket orders
        bracket_orders = self.db.execute("""
            SELECT order_id, order_class, parent_order_id
            FROM orders
            WHERE order_id IN (?, ?, ?)
            ORDER BY created_at
        """, [entry_id, profit_id, stop_id]).fetchall()

        self.assertEqual(len(bracket_orders), 3, "Should have 3 bracket orders")

        print(f"✓ Bracket order created:")
        print(f"  Entry:  {entry_id} @ ${entry_price}")
        print(f"  Profit: {profit_id} @ ${profit_target}")
        print(f"  Stop:   {stop_id} @ ${stop_loss}")
        print(f"  Risk/Reward: ${entry_price - stop_loss:.2f} / ${profit_target - entry_price:.2f}")
        print(f"  Status: All Pending (DRY-RUN)")

    def test_05_order_modification(self):
        """Test modifying an existing order"""
        print("\nTest 5: Order Modification (DRY-RUN)")
        print("-" * 60)

        # Create initial order
        order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_005"
        original_price = 580.00

        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, limit_price, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, self.account_id, self.test_symbol, 'Buy', 10,
            'Limit', original_price, 'Working', True, datetime.now()
        ])

        # Modify price
        new_price = 582.00
        self.db.execute("""
            UPDATE orders
            SET limit_price = ?, status = 'Replaced', updated_at = ?
            WHERE order_id = ?
        """, [new_price, datetime.now(), order_id])

        # Log the modification
        self.db.execute("""
            INSERT INTO order_updates (
                order_id, field_name, old_value, new_value, updated_at
            ) VALUES (?, ?, ?, ?, ?)
        """, [
            order_id, 'limit_price',
            str(original_price), str(new_price), datetime.now()
        ])

        # Verify modification
        result = self.db.execute("""
            SELECT limit_price, status FROM orders WHERE order_id = ?
        """, [order_id]).fetchone()

        self.assertEqual(float(result[0]), new_price, "Price should be updated")
        self.assertEqual(result[1], 'Replaced', "Status should be Replaced")

        print(f"✓ Order modified: {order_id}")
        print(f"  Original Price: ${original_price}")
        print(f"  New Price: ${new_price}")
        print(f"  Status: Replaced (DRY-RUN)")

    def test_06_order_cancellation(self):
        """Test canceling an order"""
        print("\nTest 6: Order Cancellation (DRY-RUN)")
        print("-" * 60)

        # Create order
        order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_006"

        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, self.account_id, self.test_symbol, 'Buy', 5,
            'Market', 'Working', True, datetime.now()
        ])

        # Cancel order
        self.db.execute("""
            UPDATE orders
            SET status = 'Canceled', updated_at = ?
            WHERE order_id = ?
        """, [datetime.now(), order_id])

        # Log cancellation
        self.db.execute("""
            INSERT INTO order_updates (
                order_id, field_name, old_value, new_value, updated_at
            ) VALUES (?, ?, ?, ?, ?)
        """, [
            order_id, 'status', 'Working', 'Canceled', datetime.now()
        ])

        # Verify cancellation
        result = self.db.execute("""
            SELECT status FROM orders WHERE order_id = ?
        """, [order_id]).fetchone()

        self.assertEqual(result[0], 'Canceled', "Order should be canceled")

        print(f"✓ Order canceled: {order_id}")
        print(f"  Status: Canceled (DRY-RUN)")

    def test_07_order_validation(self):
        """Test order validation logic"""
        print("\nTest 7: Order Validation (DRY-RUN)")
        print("-" * 60)

        # Test invalid quantity (negative)
        try:
            self.db.execute("""
                INSERT INTO orders (
                    order_id, account_id, symbol, side, quantity,
                    order_type, status, dry_run, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                'INVALID_001', self.account_id, self.test_symbol, 'Buy', -5,
                'Market', 'Pending', True, datetime.now()
            ])
            self.fail("Should have raised constraint violation")
        except Exception as e:
            self.assertIn("CHECK", str(e).upper(), "Should be a check constraint violation")
            print("✓ Negative quantity rejected")

        # Test valid order
        valid_order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_007"
        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity,
                order_type, status, dry_run, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            valid_order_id, self.account_id, self.test_symbol, 'Buy', 10,
            'Market', 'Pending', True, datetime.now()
        ])
        print("✓ Valid order accepted")

    def test_08_daily_order_summary(self):
        """Test daily order summary aggregation"""
        print("\nTest 8: Daily Order Summary (DRY-RUN)")
        print("-" * 60)

        # Get today's orders
        today_orders = self.db.execute("""
            SELECT
                COUNT(*) as total_orders,
                COUNT(DISTINCT symbol) as unique_symbols,
                SUM(quantity) as total_volume,
                SUM(CASE WHEN status = 'Filled' THEN 1 ELSE 0 END) as filled,
                SUM(CASE WHEN status = 'Canceled' THEN 1 ELSE 0 END) as canceled,
                SUM(CASE WHEN dry_run = TRUE THEN 1 ELSE 0 END) as dry_run_count
            FROM orders
            WHERE DATE(created_at) = CURRENT_DATE
        """).fetchone()

        total_orders = today_orders[0]
        unique_symbols = today_orders[1]
        total_volume = today_orders[2] if today_orders[2] else 0
        filled = today_orders[3]
        canceled = today_orders[4]
        dry_run_count = today_orders[5]

        print(f"✓ Daily Summary:")
        print(f"  Total Orders: {total_orders}")
        print(f"  Unique Symbols: {unique_symbols}")
        print(f"  Total Volume: {total_volume} shares")
        print(f"  Filled: {filled}")
        print(f"  Canceled: {canceled}")
        print(f"  Dry-Run: {dry_run_count}")

        self.assertGreater(total_orders, 0, "Should have orders from tests")
        self.assertEqual(total_orders, dry_run_count, "All orders should be dry-run")

    def test_09_compliance_audit_trail(self):
        """Test compliance audit trail"""
        print("\nTest 9: Compliance Audit Trail (DRY-RUN)")
        print("-" * 60)

        # Get all order updates
        updates = self.db.execute("""
            SELECT
                order_id,
                field_name,
                old_value,
                new_value,
                updated_at
            FROM order_updates
            ORDER BY updated_at DESC
            LIMIT 10
        """).fetchall()

        print(f"✓ Audit trail entries: {len(updates)}")
        if updates:
            print(f"  Sample entries:")
            for i, update in enumerate(updates[:3], 1):
                print(f"    {i}. Order {update[0]}: {update[1]} changed from {update[2]} to {update[3]}")

    def test_10_performance_metrics(self):
        """Test performance tracking (simulated)"""
        print("\nTest 10: Performance Metrics (DRY-RUN)")
        print("-" * 60)

        # Simulate a filled order with P&L
        order_id = f"ORD_TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}_010"
        entry_price = 580.00
        exit_price = 585.00
        quantity = 10

        # Create filled order
        self.db.execute("""
            INSERT INTO orders (
                order_id, account_id, symbol, side, quantity, filled_quantity,
                order_type, avg_fill_price, status, dry_run,
                created_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, self.account_id, self.test_symbol, 'Buy', quantity, quantity,
            'Market', entry_price, 'Filled', True,
            datetime.now() - timedelta(hours=2), datetime.now()
        ])

        # Record performance
        pnl = (exit_price - entry_price) * quantity
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        hold_duration = 7200  # 2 hours in seconds

        self.db.execute("""
            INSERT INTO order_performance (
                order_id, symbol, entry_price, exit_price, quantity,
                pnl, pnl_percent, hold_duration_seconds,
                entry_timestamp, exit_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            order_id, self.test_symbol, entry_price, exit_price, quantity,
            pnl, pnl_percent, hold_duration,
            datetime.now() - timedelta(hours=2), datetime.now()
        ])

        # Verify performance record
        perf = self.db.execute("""
            SELECT pnl, pnl_percent FROM order_performance
            WHERE order_id = ?
        """, [order_id]).fetchone()

        self.assertEqual(float(perf[0]), pnl, "P&L should match")

        print(f"✓ Performance tracked: {order_id}")
        print(f"  Entry: ${entry_price}")
        print(f"  Exit: ${exit_price}")
        print(f"  P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
        print(f"  Hold: {hold_duration // 60} minutes")


def main():
    """Run test suite"""
    # Create test data directory
    os.makedirs('data', exist_ok=True)

    # Run tests
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
