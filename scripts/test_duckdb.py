#!/usr/bin/env python3
"""
DuckDB Test Script

Tests basic DuckDB operations to ensure the database is working correctly
before building the data pipeline.

Tests:
1. Create in-memory database and table
2. Insert data
3. Query data
4. Create persistent database file
5. Read/write Parquet files
6. Performance test (1M rows)
7. ACID transaction test
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_test(test_name):
    """Print test name"""
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}Test: {test_name}{NC}")
    print(f"{BLUE}{'='*60}{NC}")


def print_success(message):
    """Print success message"""
    print(f"{GREEN}✓ {message}{NC}")


def print_error(message):
    """Print error message"""
    print(f"{RED}✗ {message}{NC}")


def print_info(message):
    """Print info message"""
    print(f"{YELLOW}  {message}{NC}")


def test_inmemory_database():
    """Test 1: In-memory database operations"""
    print_test("In-Memory Database")

    try:
        # Create in-memory database
        conn = duckdb.connect(database=':memory:')
        print_info("Created in-memory database")

        # Create table
        conn.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name VARCHAR,
                price DOUBLE,
                timestamp TIMESTAMP
            )
        """)
        print_info("Created table 'test_table'")

        # Insert data
        conn.execute("""
            INSERT INTO test_table VALUES
            (1, 'AAPL', 150.25, '2024-01-01 09:30:00'),
            (2, 'GOOGL', 2800.50, '2024-01-01 09:31:00'),
            (3, 'MSFT', 380.75, '2024-01-01 09:32:00')
        """)
        print_info("Inserted 3 rows")

        # Query data
        result = conn.execute("SELECT * FROM test_table ORDER BY id").fetchall()
        print_info(f"Queried data: {len(result)} rows returned")

        # Verify results
        assert len(result) == 3
        assert result[0][1] == 'AAPL'

        conn.close()
        print_success("In-memory database test passed")
        return True

    except Exception as e:
        print_error(f"In-memory database test failed: {e}")
        return False


def test_persistent_database():
    """Test 2: Persistent database file"""
    print_test("Persistent Database File")

    db_path = Path('data/test.duckdb')

    try:
        # Remove existing test database
        if db_path.exists():
            db_path.unlink()
            print_info("Removed existing test database")

        # Create persistent database
        conn = duckdb.connect(database=str(db_path))
        print_info(f"Created persistent database at {db_path}")

        # Create table and insert data
        conn.execute("""
            CREATE TABLE market_data (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
        """)

        conn.execute("""
            INSERT INTO market_data VALUES
            ('AAPL', '2024-01-01', 150.0, 152.0, 149.0, 151.5, 1000000),
            ('AAPL', '2024-01-02', 151.5, 153.0, 150.5, 152.8, 1200000)
        """)
        print_info("Created table and inserted data")

        # Close and reopen to test persistence
        conn.close()
        conn = duckdb.connect(database=str(db_path))

        # Query data
        result = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()
        assert result[0] == 2
        print_info(f"Verified persistence: {result[0]} rows found after reopening")

        conn.close()
        db_path.unlink()  # Clean up

        print_success("Persistent database test passed")
        return True

    except Exception as e:
        print_error(f"Persistent database test failed: {e}")
        if db_path.exists():
            db_path.unlink()
        return False


def test_parquet_io():
    """Test 3: Parquet read/write"""
    print_test("Parquet Read/Write")

    parquet_path = Path('data/test.parquet')

    try:
        # Create sample DataFrame
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'] * 100,
            'price': np.random.uniform(100, 3000, 300),
            'volume': np.random.randint(1000000, 10000000, 300),
            'timestamp': pd.date_range('2024-01-01', periods=300, freq='1min')
        })
        print_info(f"Created test DataFrame with {len(df)} rows")

        # Write to Parquet
        df.to_parquet(parquet_path, engine='pyarrow')
        file_size = parquet_path.stat().st_size / 1024  # KB
        print_info(f"Wrote Parquet file: {file_size:.2f} KB")

        # Read with DuckDB
        conn = duckdb.connect(database=':memory:')
        result = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()
        assert result[0] == len(df)
        print_info(f"DuckDB read Parquet: {result[0]} rows")

        # Query Parquet directly (without loading into memory)
        avg_price = conn.execute(f"""
            SELECT symbol, AVG(price) as avg_price
            FROM '{parquet_path}'
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()
        print_info(f"Computed aggregates on Parquet: {len(avg_price)} groups")

        conn.close()
        parquet_path.unlink()  # Clean up

        print_success("Parquet read/write test passed")
        return True

    except Exception as e:
        print_error(f"Parquet test failed: {e}")
        if parquet_path.exists():
            parquet_path.unlink()
        return False


def test_performance():
    """Test 4: Performance with 1M rows"""
    print_test("Performance Test (1M rows)")

    try:
        conn = duckdb.connect(database=':memory:')

        # Create table
        start_time = time.time()
        conn.execute("""
            CREATE TABLE large_table AS
            SELECT
                i as id,
                'SYMBOL_' || (i % 100) as symbol,
                100 + (random() * 100) as price,
                1000000 + (random() * 1000000)::BIGINT as volume
            FROM range(1000000) t(i)
        """)
        create_time = time.time() - start_time
        print_info(f"Created table with 1M rows in {create_time:.3f} seconds")

        # Count rows
        start_time = time.time()
        count = conn.execute("SELECT COUNT(*) FROM large_table").fetchone()[0]
        count_time = time.time() - start_time
        print_info(f"Counted {count:,} rows in {count_time:.3f} seconds")

        # Aggregate query
        start_time = time.time()
        result = conn.execute("""
            SELECT symbol, AVG(price) as avg_price, SUM(volume) as total_volume
            FROM large_table
            GROUP BY symbol
            ORDER BY total_volume DESC
            LIMIT 10
        """).fetchall()
        agg_time = time.time() - start_time
        print_info(f"Aggregated by 100 symbols in {agg_time:.3f} seconds")
        print_info(f"Top symbol: {result[0][0]} (avg price: ${result[0][1]:.2f})")

        # Filter and sort
        start_time = time.time()
        result = conn.execute("""
            SELECT * FROM large_table
            WHERE price > 150
            ORDER BY price DESC
            LIMIT 100
        """).fetchall()
        filter_time = time.time() - start_time
        print_info(f"Filtered and sorted {len(result)} rows in {filter_time:.3f} seconds")

        conn.close()

        # Performance checks
        assert create_time < 5.0, f"Table creation too slow: {create_time:.3f}s"
        assert agg_time < 1.0, f"Aggregation too slow: {agg_time:.3f}s"

        print_success("Performance test passed")
        print_info(f"Total time: {create_time + count_time + agg_time + filter_time:.3f} seconds")
        return True

    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False


def test_transactions():
    """Test 5: ACID transaction test"""
    print_test("ACID Transaction Test")

    try:
        conn = duckdb.connect(database=':memory:')

        # Create table
        conn.execute("""
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY,
                name VARCHAR,
                balance DOUBLE
            )
        """)

        conn.execute("""
            INSERT INTO accounts VALUES
            (1, 'Account A', 1000.0),
            (2, 'Account B', 500.0)
        """)
        print_info("Created accounts table")

        # Test successful transaction
        conn.begin()
        conn.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
        conn.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
        conn.commit()
        print_info("Committed successful transaction")

        balances = conn.execute("SELECT balance FROM accounts ORDER BY id").fetchall()
        assert balances[0][0] == 900.0
        assert balances[1][0] == 600.0
        print_info(f"Account A: ${balances[0][0]}, Account B: ${balances[1][0]}")

        # Test rollback
        conn.begin()
        conn.execute("UPDATE accounts SET balance = balance - 200 WHERE id = 1")
        conn.execute("UPDATE accounts SET balance = balance + 200 WHERE id = 2")
        conn.rollback()
        print_info("Rolled back transaction")

        balances = conn.execute("SELECT balance FROM accounts ORDER BY id").fetchall()
        assert balances[0][0] == 900.0  # Should be unchanged
        assert balances[1][0] == 600.0  # Should be unchanged
        print_info(f"After rollback - Account A: ${balances[0][0]}, Account B: ${balances[1][0]}")

        conn.close()

        print_success("ACID transaction test passed")
        return True

    except Exception as e:
        print_error(f"Transaction test failed: {e}")
        return False


def main():
    """Run all tests"""
    print(f"\n{GREEN}{'='*60}{NC}")
    print(f"{GREEN}DuckDB Test Suite{NC}")
    print(f"{GREEN}{'='*60}{NC}")

    tests = [
        test_inmemory_database,
        test_persistent_database,
        test_parquet_io,
        test_performance,
        test_transactions
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_error(f"Unexpected error in {test_func.__name__}: {e}")
            results.append(False)

    # Summary
    print(f"\n{GREEN}{'='*60}{NC}")
    print(f"{GREEN}Test Summary{NC}")
    print(f"{GREEN}{'='*60}{NC}")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"{GREEN}All tests passed! ({passed}/{total}){NC}")
        return 0
    else:
        print(f"{RED}Some tests failed: {passed}/{total} passed{NC}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
