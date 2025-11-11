#!/usr/bin/env python3
"""
Setup Tax Lots Table

Creates the tax_lots table for tracking individual purchase lots
for bot-managed positions. This enables:
- LIFO/FIFO tax lot selection
- Holding period tracking (short-term vs long-term)
- Accurate cost basis per lot
- Tax optimization

Usage:
    uv run python scripts/setup_tax_lots_table.py
"""

import duckdb
from pathlib import Path
from datetime import datetime

def setup_tax_lots_table():
    """Create tax_lots table for bot position tracking"""

    db_path = Path('data/bigbrother.duckdb')
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run Phase 5 setup first: uv run python scripts/phase5_setup.py")
        return 1

    print("=" * 80)
    print("Setting up Tax Lots Table")
    print("=" * 80)

    conn = duckdb.connect(str(db_path))

    # Create tax_lots table
    print("\n📊 Creating tax_lots table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tax_lots (
            id INTEGER PRIMARY KEY,
            account_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            asset_type VARCHAR NOT NULL,           -- EQUITY, OPTION, FUTURE, etc.
            quantity DOUBLE NOT NULL,
            entry_price DOUBLE NOT NULL,
            entry_date TIMESTAMP NOT NULL,
            strategy VARCHAR,

            -- Options-specific fields
            option_type VARCHAR,                    -- CALL, PUT
            strike_price DOUBLE,
            expiration_date DATE,
            underlying_symbol VARCHAR,

            -- Closing information
            is_closed BOOLEAN DEFAULT false,
            close_type VARCHAR,                     -- SOLD, EXPIRED, EXERCISED, ASSIGNED
            close_price DOUBLE,
            close_date TIMESTAMP,
            realized_pnl DOUBLE,

            -- Tax treatment
            holding_period_days INTEGER,
            is_long_term BOOLEAN,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ tax_lots table created")

    # Create indexes for performance
    print("\n🔍 Creating indexes...")
    try:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tax_lots_symbol
            ON tax_lots(symbol)
        """)
        print("   ✅ Index on symbol")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tax_lots_entry_date
            ON tax_lots(entry_date)
        """)
        print("   ✅ Index on entry_date")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tax_lots_is_closed
            ON tax_lots(is_closed)
        """)
        print("   ✅ Index on is_closed")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tax_lots_strategy
            ON tax_lots(strategy)
        """)
        print("   ✅ Index on strategy")
    except Exception as e:
        print(f"   ⚠️  Index creation: {e}")

    # Create view for open tax lots
    print("\n📈 Creating views...")
    conn.execute("""
        CREATE OR REPLACE VIEW v_open_tax_lots AS
        SELECT
            id,
            account_id,
            symbol,
            asset_type,
            quantity,
            entry_price,
            entry_date,
            strategy,
            option_type,
            strike_price,
            expiration_date,
            underlying_symbol,
            DATEDIFF('day', entry_date, CURRENT_TIMESTAMP) as holding_period_days,
            CASE
                WHEN expiration_date IS NOT NULL
                THEN DATEDIFF('day', CURRENT_TIMESTAMP, expiration_date)
                ELSE NULL
            END as days_to_expiration,
            CASE
                WHEN DATEDIFF('day', entry_date, CURRENT_TIMESTAMP) > 365
                THEN true
                ELSE false
            END as is_long_term,
            entry_price * quantity * 100 as cost_basis,  -- Options are per-contract (100 shares)
            CASE
                WHEN asset_type = 'OPTION'
                THEN symbol || ' ' || option_type || ' $' || CAST(strike_price AS VARCHAR) || ' ' || CAST(expiration_date AS VARCHAR)
                ELSE symbol
            END as display_name
        FROM tax_lots
        WHERE is_closed = false
        ORDER BY entry_date DESC
    """)
    print("   ✅ v_open_tax_lots view")

    # Create view for closed tax lots (realized gains/losses)
    conn.execute("""
        CREATE OR REPLACE VIEW v_closed_tax_lots AS
        SELECT
            id,
            account_id,
            symbol,
            asset_type,
            quantity,
            entry_price,
            entry_date,
            close_price,
            close_date,
            close_type,
            strategy,
            option_type,
            strike_price,
            expiration_date,
            underlying_symbol,
            realized_pnl,
            holding_period_days,
            is_long_term,
            CASE
                WHEN is_long_term THEN 'Long-term'
                ELSE 'Short-term'
            END as tax_treatment,
            CASE
                WHEN asset_type = 'OPTION'
                THEN symbol || ' ' || option_type || ' $' || CAST(strike_price AS VARCHAR) || ' ' || CAST(expiration_date AS VARCHAR)
                ELSE symbol
            END as display_name
        FROM tax_lots
        WHERE is_closed = true
        ORDER BY close_date DESC
    """)
    print("   ✅ v_closed_tax_lots view")

    # Create summary view
    conn.execute("""
        CREATE OR REPLACE VIEW v_tax_lots_summary AS
        SELECT
            symbol,
            asset_type,
            strategy,
            COUNT(*) as total_lots,
            SUM(quantity) as total_quantity,
            SUM(CASE
                WHEN asset_type = 'OPTION' THEN entry_price * quantity * 100
                ELSE entry_price * quantity
            END) as total_cost_basis,
            AVG(entry_price) as avg_entry_price,
            MIN(entry_date) as earliest_entry,
            MAX(entry_date) as latest_entry,
            SUM(CASE WHEN expiration_date IS NOT NULL
                     AND expiration_date < CURRENT_DATE
                THEN 1 ELSE 0
            END) as expired_options
        FROM tax_lots
        WHERE is_closed = false
        GROUP BY symbol, asset_type, strategy
        ORDER BY total_cost_basis DESC
    """)
    print("   ✅ v_tax_lots_summary view")

    conn.commit()

    # Show table info
    print("\n📋 Table Schema:")
    result = conn.execute("DESCRIBE tax_lots").fetchall()
    for row in result:
        print(f"   {row[0]:25s} {row[1]:15s}")

    # Check if there's existing data
    count = conn.execute("SELECT COUNT(*) FROM tax_lots").fetchone()[0]
    print(f"\n📊 Current tax lots: {count}")

    conn.close()

    print("\n" + "=" * 80)
    print("✅ Tax Lots Table Setup Complete!")
    print("=" * 80)
    print("\nUsage:")
    print("  View open lots:   SELECT * FROM v_open_tax_lots")
    print("  View closed lots: SELECT * FROM v_closed_tax_lots")
    print("  Summary by symbol: SELECT * FROM v_tax_lots_summary")
    print()

    return 0

if __name__ == "__main__":
    exit(setup_tax_lots_table())
