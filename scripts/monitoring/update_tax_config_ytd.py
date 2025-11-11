#!/usr/bin/env python3
"""
Update Tax Configuration with Base Income Context

Tracks 2025 YTD tax liability on trading profits, assuming $300K base income
already earned from other sources.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Usage:
    uv run python scripts/monitoring/update_tax_config_ytd.py
"""

import duckdb
from pathlib import Path
from datetime import datetime

def update_tax_config():
    """Update tax_config with base income context"""

    db_path = Path(__file__).parent.parent.parent / "data" / "bigbrother.duckdb"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    print("📊 Updating tax configuration for 2025 YTD tracking...")

    try:
        conn = duckdb.connect(str(db_path))

        # Check if base_annual_income column exists
        columns = conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tax_config'
        """).fetchall()

        column_names = [c[0] for c in columns]

        if 'base_annual_income' not in column_names:
            print("   Adding base_annual_income column...")
            conn.execute("""
                ALTER TABLE tax_config
                ADD COLUMN base_annual_income DOUBLE DEFAULT 0.0
            """)

        if 'tax_year' not in column_names:
            print("   Adding tax_year column...")
            conn.execute("""
                ALTER TABLE tax_config
                ADD COLUMN tax_year INTEGER DEFAULT 2025
            """)

        # Update configuration with context
        conn.execute("""
            UPDATE tax_config SET
                base_annual_income = 300000.00,  -- $300K already earned
                tax_year = 2025,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """)

        print(f"\n✅ Tax configuration updated")

        # Show current configuration
        config = conn.execute("SELECT * FROM tax_config WHERE id = 1").fetchone()

        print(f"\n📋 2025 Tax Configuration:")
        print(f"   Tax Year: 2025")
        print(f"   Base Income (already earned): ${config[-2]:,.0f}")
        print(f"   ─────────────────────────────────")
        print(f"   Federal short-term rate: {config[2]*100:.0f}%")
        print(f"   Federal long-term rate: {config[3]*100:.0f}%")
        print(f"   State tax: {config[4]*100:.1f}%")
        print(f"   Medicare surtax: {config[5]*100:.1f}%")
        print(f"   Trading fees: {config[6]*100:.1f}%")
        print(f"   ─────────────────────────────────")
        print(f"   Effective short-term: {(config[2] + config[4] + config[5])*100:.1f}%")
        print(f"   Effective long-term: {(config[3] + config[4] + config[5])*100:.1f}%")

        # Show YTD trading activity
        ytd = conn.execute("""
            SELECT
                COALESCE(SUM(gross_pnl), 0) as total_gross_pnl,
                COALESCE(SUM(trading_fees), 0) as total_fees,
                COALESCE(SUM(pnl_after_fees), 0) as total_after_fees,
                COALESCE(SUM(tax_owed), 0) as total_tax,
                COALESCE(SUM(net_pnl_after_tax), 0) as total_net,
                COUNT(*) as trade_count
            FROM tax_records
            WHERE EXTRACT(YEAR FROM exit_time) = 2025
        """).fetchone()

        print(f"\n📈 2025 YTD Trading Activity:")
        print(f"   Base income: ${config[-2]:,.0f} (from other sources)")
        print(f"   Trading gross P&L: ${ytd[0]:,.2f}")
        print(f"   Trading fees (3%): ${ytd[1]:,.2f}")
        print(f"   P&L after fees: ${ytd[2]:,.2f}")
        print(f"   ─────────────────────────────────")
        print(f"   Tax on trading profits: ${ytd[3]:,.2f}")
        print(f"   Net trading profit: ${ytd[4]:,.2f}")
        print(f"   ─────────────────────────────────")
        print(f"   Total 2025 income: ${config[-2] + ytd[2]:,.2f}")
        print(f"   Total trades closed: {ytd[5]}")

        if ytd[5] == 0:
            print(f"\n   ℹ️  No trades closed yet in 2025")
            print(f"   ℹ️  Taxes will accumulate as trades close")

        print(f"\n💡 Tax Tracking:")
        print(f"   • Each closed trade automatically calculates tax")
        print(f"   • YTD totals accumulate throughout 2025")
        print(f"   • Dashboard shows real-time tax liability")
        print(f"   • Tax records reset at year-end (Jan 1, 2026)")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error updating tax config: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    success = update_tax_config()

    if success:
        print("\n✅ Tax configuration updated for 2025 YTD tracking")
    else:
        print("\n❌ Failed to update tax configuration")

if __name__ == "__main__":
    main()
