#!/usr/bin/env python3
"""
BigBrotherAnalytics: Setup Alerts Database Schema
Creates the alerts table and related structures in DuckDB

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 4, Week 3: Custom Alerts System
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import duckdb


def setup_alerts_schema(db_path: str) -> bool:
    """
    Set up alerts database schema.

    Args:
        db_path: Path to DuckDB database

    Returns:
        True if successful
    """
    try:
        # Connect to database
        conn = duckdb.connect(db_path)

        # Read schema SQL
        schema_file = BASE_DIR / 'scripts' / 'database_schema_alerts.sql'
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute schema (split by semicolon)
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        for stmt in statements:
            try:
                conn.execute(stmt)
            except Exception as e:
                print(f"Error executing statement: {e}")
                print(f"Statement: {stmt[:100]}...")

        conn.commit()

        # Verify tables created
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name IN ('alerts', 'alert_throttle', 'alert_delivery_log')
        """).fetchall()

        print(f"\nTables created: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")

        # Check initial alert
        initial_alert = conn.execute("""
            SELECT alert_type, severity, message
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchone()

        if initial_alert:
            print(f"\nInitial alert: {initial_alert}")

        conn.close()
        print("\nAlerts database schema created successfully!")
        return True

    except Exception as e:
        print(f"Error setting up alerts schema: {e}")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Setup alerts database schema')
    parser.add_argument(
        '--db',
        default=str(BASE_DIR / 'data' / 'bigbrother.duckdb'),
        help='Path to DuckDB database'
    )

    args = parser.parse_args()

    success = setup_alerts_schema(args.db)
    sys.exit(0 if success else 1)
