#!/usr/bin/env python3
"""
Add Greeks Columns to Tax Lots Table

Adds option Greeks columns to the existing tax_lots table
for tracking Delta, Gamma, Theta, Vega, Rho at entry time.

Usage:
    uv run python scripts/add_greeks_columns.py
"""

import duckdb
from pathlib import Path

def add_greeks_columns():
    """Add Greeks columns to tax_lots table"""

    db_path = Path('data/bigbrother.duckdb')
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    print("=" * 80)
    print("Adding Greeks Columns to Tax Lots Table")
    print("=" * 80)

    conn = duckdb.connect(str(db_path))

    # Check if tax_lots table exists
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]

    if 'tax_lots' not in table_names:
        print("❌ tax_lots table not found")
        print("   Run: uv run python scripts/setup_tax_lots_table.py")
        return 1

    print("\n📊 Checking existing columns...")
    columns = conn.execute("DESCRIBE tax_lots").fetchall()
    column_names = [c[0] for c in columns]

    greeks_columns = {
        'entry_delta': 'DOUBLE',
        'entry_gamma': 'DOUBLE',
        'entry_theta': 'DOUBLE',
        'entry_vega': 'DOUBLE',
        'entry_rho': 'DOUBLE',
        'entry_iv': 'DOUBLE'
    }

    # Add missing columns
    print("\n📈 Adding Greeks columns...")
    columns_added = 0
    for col_name, col_type in greeks_columns.items():
        if col_name not in column_names:
            try:
                conn.execute(f"ALTER TABLE tax_lots ADD COLUMN {col_name} {col_type}")
                print(f"   ✅ Added {col_name}")
                columns_added += 1
            except Exception as e:
                print(f"   ⚠️  Failed to add {col_name}: {e}")
        else:
            print(f"   ℹ️  {col_name} already exists")

    if columns_added > 0:
        conn.commit()

    # Recreate views with Greeks columns
    print("\n📈 Updating views...")

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
            entry_delta,
            entry_gamma,
            entry_theta,
            entry_vega,
            entry_rho,
            entry_iv,
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
            CASE
                WHEN asset_type = 'OPTION' THEN entry_price * quantity * 100
                ELSE entry_price * quantity
            END as cost_basis,
            CASE
                WHEN asset_type = 'OPTION'
                THEN symbol || ' ' || option_type || ' $' || CAST(strike_price AS VARCHAR) || ' ' || CAST(expiration_date AS VARCHAR)
                ELSE symbol
            END as display_name
        FROM tax_lots
        WHERE is_closed = false
        ORDER BY entry_date DESC
    """)
    print("   ✅ v_open_tax_lots view updated")

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
            entry_delta,
            entry_gamma,
            entry_theta,
            entry_vega,
            entry_rho,
            entry_iv,
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
    print("   ✅ v_closed_tax_lots view updated")

    conn.commit()

    # Show updated schema
    print("\n📋 Updated Schema:")
    result = conn.execute("DESCRIBE tax_lots").fetchall()
    for row in result:
        if 'entry_' in row[0] and ('delta' in row[0] or 'gamma' in row[0] or
                                     'theta' in row[0] or 'vega' in row[0] or
                                     'rho' in row[0] or 'iv' in row[0]):
            print(f"   {row[0]:25s} {row[1]:15s}")

    conn.close()

    print("\n" + "=" * 80)
    print("✅ Greeks Columns Added Successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Implement trinomial tree pricer in C++23")
    print("  2. Create Python bindings for Greeks calculation")
    print("  3. Update dashboard to display Greeks")
    print()

    return 0

if __name__ == "__main__":
    exit(add_greeks_columns())
