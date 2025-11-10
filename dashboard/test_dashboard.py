#!/usr/bin/env python3
"""
Test script to validate dashboard components
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import duckdb
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "bigbrother.duckdb"

def test_database_connection():
    """Test database connection"""
    print("Testing database connection...")
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        print("✓ Database connection successful")
        return conn
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None

def test_positions_query(conn):
    """Test positions query"""
    print("\nTesting positions query...")
    try:
        query = """
            SELECT
                id, symbol, quantity, avg_cost, current_price,
                market_value, unrealized_pnl, is_bot_managed,
                managed_by, bot_strategy
            FROM positions
            LIMIT 5
        """
        df = conn.execute(query).df()
        print(f"✓ Positions query successful - {len(df)} rows returned")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        if not df.empty:
            print(f"  Sample symbol: {df.iloc[0]['symbol']}")
        return True
    except Exception as e:
        print(f"✗ Positions query failed: {e}")
        return False

def test_positions_history_query(conn):
    """Test positions history query"""
    print("\nTesting positions history query...")
    try:
        query = """
            SELECT
                timestamp, symbol, quantity, average_price,
                current_price, unrealized_pnl
            FROM positions_history
            ORDER BY timestamp DESC
            LIMIT 5
        """
        df = conn.execute(query).df()
        print(f"✓ Positions history query successful - {len(df)} rows returned")
        return True
    except Exception as e:
        print(f"✗ Positions history query failed: {e}")
        return False

def test_sectors_query(conn):
    """Test sectors query"""
    print("\nTesting sectors query...")
    try:
        query = """
            SELECT
                sector_code, sector_name, sector_etf, category
            FROM sectors
        """
        df = conn.execute(query).df()
        print(f"✓ Sectors query successful - {len(df)} rows returned")
        print(f"  Sectors: {', '.join(df['sector_name'].tolist()[:5])}...")
        return True
    except Exception as e:
        print(f"✗ Sectors query failed: {e}")
        return False

def test_employment_query(conn):
    """Test employment query with growth calculations"""
    print("\nTesting employment query...")
    try:
        query = """
            WITH latest_data AS (
                SELECT
                    se.sector_code,
                    s.sector_name,
                    s.sector_etf,
                    se.report_date,
                    se.employment_count,
                    ROW_NUMBER() OVER (PARTITION BY se.sector_code ORDER BY se.report_date DESC) as rn
                FROM sector_employment se
                JOIN sectors s ON se.sector_code = s.sector_code
                WHERE se.employment_count IS NOT NULL
            ),
            current_data AS (
                SELECT * FROM latest_data WHERE rn = 1
            ),
            previous_data AS (
                SELECT * FROM latest_data WHERE rn = 4
            )
            SELECT
                c.sector_code,
                c.sector_name,
                c.sector_etf,
                c.employment_count as current_employment,
                p.employment_count as previous_employment,
                CASE
                    WHEN p.employment_count IS NOT NULL AND p.employment_count > 0
                    THEN ((c.employment_count - p.employment_count) * 100.0 / p.employment_count)
                    ELSE 0
                END as growth_rate_3m
            FROM current_data c
            LEFT JOIN previous_data p ON c.sector_code = p.sector_code
            ORDER BY growth_rate_3m DESC
        """
        df = conn.execute(query).df()
        print(f"✓ Employment query successful - {len(df)} rows returned")
        if not df.empty:
            print(f"  Top growth sector: {df.iloc[0]['sector_name']} ({df.iloc[0]['growth_rate_3m']:.2f}%)")
        return True
    except Exception as e:
        print(f"✗ Employment query failed: {e}")
        return False

def test_streamlit_imports():
    """Test that all required packages can be imported"""
    print("\nTesting package imports...")
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False

    try:
        import plotly
        print(f"✓ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False

    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False

    try:
        import duckdb
        print(f"✓ DuckDB {duckdb.__version__}")
    except ImportError as e:
        print(f"✗ DuckDB import failed: {e}")
        return False

    return True

def main():
    print("BigBrother Dashboard - Component Test")
    print("=" * 60)

    # Test imports
    if not test_streamlit_imports():
        print("\n✗ Package import tests failed")
        return False

    # Test database
    conn = test_database_connection()
    if not conn:
        print("\n✗ Database tests failed")
        return False

    # Test queries
    tests_passed = True
    tests_passed &= test_positions_query(conn)
    tests_passed &= test_positions_history_query(conn)
    tests_passed &= test_sectors_query(conn)
    tests_passed &= test_employment_query(conn)

    conn.close()

    print("\n" + "=" * 60)
    if tests_passed:
        print("✓ All tests passed!")
        print("\nDashboard is ready to run:")
        print("  cd dashboard")
        print("  uv run streamlit run app.py")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
