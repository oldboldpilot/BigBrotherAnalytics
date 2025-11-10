#!/usr/bin/env python3
"""
BigBrotherAnalytics: Setup Metrics Table
Create database table for tracking performance metrics

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import duckdb
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = BASE_DIR / "data" / "bigbrother.duckdb"


def setup_metrics_table():
    """Create metrics table in database."""
    print(f"Setting up metrics table in: {DB_PATH}")

    # Connect to database
    db = duckdb.connect(str(DB_PATH))

    # Create metrics table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY,
        metric_name VARCHAR NOT NULL,
        value DOUBLE NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        context VARCHAR,
        tags VARCHAR,
        INDEX idx_metric_name (metric_name),
        INDEX idx_timestamp (timestamp)
    );
    """

    try:
        db.execute(create_table_sql)
        print("✓ Metrics table created successfully")

        # Verify table exists
        tables = db.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]

        if 'metrics' in table_names:
            print("✓ Metrics table verified")

            # Show table schema
            schema = db.execute("DESCRIBE metrics").fetchall()
            print("\nMetrics table schema:")
            for row in schema:
                print(f"  - {row[0]}: {row[1]}")

            # Check if there are any existing metrics
            count = db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
            print(f"\nExisting metrics: {count}")

        else:
            print("✗ Metrics table not found after creation")

    except Exception as e:
        print(f"✗ Error creating metrics table: {e}")
        raise

    finally:
        db.close()

    print("\nMetrics table setup complete!")


if __name__ == '__main__':
    setup_metrics_table()
