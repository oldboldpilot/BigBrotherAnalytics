#!/usr/bin/env python3
"""
Tax Database Setup Script

Initializes tax tracking tables in DuckDB database.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import duckdb
from pathlib import Path

def setup_tax_database():
    """Create tax tracking tables in the database"""

    # Database path
    db_path = Path(__file__).parent.parent.parent / "data" / "bigbrother.duckdb"

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return False

    print(f"📊 Setting up tax tracking in database: {db_path}")

    try:
        # Connect to database
        conn = duckdb.connect(str(db_path))

        # Read and execute schema
        schema_path = Path(__file__).parent.parent / "database_schema_tax.sql"

        if not schema_path.exists():
            print(f"❌ Schema file not found at {schema_path}")
            return False

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Execute each statement
        statements = schema_sql.split(';')
        for i, stmt in enumerate(statements, 1):
            stmt = stmt.strip()
            if stmt and not stmt.startswith('--') and not stmt.startswith('COMMENT'):
                try:
                    conn.execute(stmt)
                    print(f"  ✅ Statement {i}/{len(statements)} executed")
                except Exception as e:
                    error_msg = str(e).lower()
                    # Ignore expected errors
                    if "already exists" in error_msg or "does not exist" in error_msg:
                        pass  # Table indexes being created before table exists - will be recreated
                    else:
                        print(f"  ⚠️  Statement {i}: {str(e)}")

        # Verify tables created
        tables = conn.execute("SHOW TABLES").fetchall()
        tax_tables = [t[0] for t in tables if 'tax' in t[0].lower()]

        print(f"\n✅ Tax tracking setup complete")
        print(f"   Created tables: {', '.join(tax_tables)}")

        # Show current configuration
        config = conn.execute("SELECT * FROM tax_config WHERE id = 1").fetchone()
        if config:
            print(f"\n📋 Current Tax Configuration:")
            print(f"   Short-term rate: {config[2]*100:.1f}%")
            print(f"   Long-term rate: {config[3]*100:.1f}%")
            print(f"   State tax rate: {config[4]*100:.1f}%")
            print(f"   Medicare surtax: {config[5]*100:.2f}%")
            print(f"   Trading fee: {config[6]*100:.1f}%")
            print(f"   Pattern day trader: {config[7]}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error setting up tax database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_tax_database()
