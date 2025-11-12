#!/usr/bin/env python3
"""
Setup Trading Signals Tracking Table

Creates the trading_signals table and associated views for comprehensive
signal tracking and analysis.
"""

import duckdb
from pathlib import Path
import sys

def setup_table():
    """Setup trading signals table"""

    print("=" * 80)
    print("Setting up Trading Signals Tracking Table")
    print("=" * 80)

    # Connect to database
    db_path = Path('data/bigbrother.duckdb')
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run setup scripts first to create the database")
        return 1

    conn = duckdb.connect(str(db_path))

    # Read schema file
    schema_file = Path('scripts/database_schema_trading_signals.sql')
    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        return 1

    print(f"\n📄 Reading schema from {schema_file}")

    with open(schema_file) as f:
        sql = f.read()

    # Execute schema
    print("🔨 Creating table and views...")

    try:
        conn.execute(sql)
        conn.commit()
        print("✅ Schema executed successfully")
    except Exception as e:
        print(f"❌ Error executing schema: {e}")
        return 1

    # Verify table exists
    print("\n🔍 Verifying table structure...")

    table_info = conn.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'trading_signals'
        ORDER BY ordinal_position
    """).fetchall()

    if not table_info:
        print("❌ Table not created properly")
        return 1

    print(f"✅ Table created with {len(table_info)} columns:")
    for col_name, col_type in table_info[:5]:  # Show first 5
        print(f"   - {col_name}: {col_type}")
    print(f"   ... and {len(table_info) - 5} more columns")

    # Verify views
    print("\n🔍 Verifying views...")

    views = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        AND table_type = 'VIEW'
        AND table_name LIKE 'v_%signal%'
    """).fetchall()

    print(f"✅ Created {len(views)} views:")
    for (view_name,) in views:
        print(f"   - {view_name}")

    # Check existing data
    count = conn.execute("SELECT COUNT(*) FROM trading_signals").fetchone()[0]
    print(f"\n📊 Current signals in database: {count}")

    conn.close()

    print("\n" + "=" * 80)
    print("✅ Trading Signals Table Setup Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Update C++ code to log signals to this table")
    print("2. Test signal generation with: ./build/bin/bigbrother")
    print("3. View signals in dashboard: streamlit run dashboard/app.py")

    return 0

if __name__ == "__main__":
    sys.exit(setup_table())
