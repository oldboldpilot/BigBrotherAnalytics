#!/usr/bin/env python3
"""
Update Tax Rates for California (Married Filing Jointly, $300K Income)

California-specific tax configuration for married filing jointly status:
- $300K income = 24% federal bracket + 9.3% California state
- Short-term capital gains: 24% federal + 9.3% CA
- Long-term capital gains: 15% federal + 9.3% CA
- Combined with 3.8% Medicare surtax (NIIT)

2025 Federal Tax Brackets (Married Filing Jointly):
- $191,950 - $364,200: 24% ← $300K falls here

2025 Federal Long-term Capital Gains (Married Filing Jointly):
- $94,050 - $583,750: 15% ← $300K falls here

California State Tax Brackets (Married Filing Jointly):
- $136,700 - $701,900: 9.3% ← $300K falls here

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Usage:
    uv run python scripts/monitoring/update_tax_rates_california.py
"""

import duckdb
from pathlib import Path

def update_tax_rates():
    """Update tax_config for California married filing jointly at $300K income"""

    db_path = Path(__file__).parent.parent.parent / "data" / "bigbrother.duckdb"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    print("📊 Updating tax rates for California (Married Filing Jointly, $300K income)...")

    try:
        conn = duckdb.connect(str(db_path))

        # Tax rates for California married filing jointly at $300K income
        short_term_rate = 0.24   # 24% federal (in 24% bracket)
        long_term_rate = 0.15    # 15% federal (under $583,750 threshold)
        state_rate = 0.093       # 9.3% California state (in 9.3% bracket)
        medicare_surtax = 0.038  # 3.8% NIIT

        # Calculate effective rates
        effective_short_term = short_term_rate + state_rate + medicare_surtax
        effective_long_term = long_term_rate + state_rate + medicare_surtax

        print(f"\n📋 Tax Rates for California (Married Filing Jointly):")
        print(f"   Income: $300,000")
        print(f"   Filing Status: Married Filing Jointly")
        print(f"   State: California")
        print(f"   ─────────────────────────────────")
        print(f"   Federal short-term: {short_term_rate*100:.0f}% (24% bracket)")
        print(f"   Federal long-term: {long_term_rate*100:.0f}% (15% bracket)")
        print(f"   California state tax: {state_rate*100:.1f}% (9.3% bracket)")
        print(f"   Medicare surtax: {medicare_surtax*100:.1f}% (NIIT)")
        print(f"   ─────────────────────────────────")
        print(f"   Effective short-term: {effective_short_term*100:.1f}%")
        print(f"   Effective long-term: {effective_long_term*100:.1f}%")

        # Check if filing_status column exists
        columns = conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tax_config'
        """).fetchall()

        column_names = [c[0] for c in columns]

        if 'filing_status' not in column_names:
            print("\n   Adding filing_status column...")
            conn.execute("""
                ALTER TABLE tax_config
                ADD COLUMN filing_status VARCHAR DEFAULT 'single'
            """)

        # Update tax_config table
        conn.execute("""
            UPDATE tax_config
            SET
                short_term_rate = ?,
                long_term_rate = ?,
                state_tax_rate = ?,
                medicare_surtax = ?,
                filing_status = 'married_joint',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, [short_term_rate, long_term_rate, state_rate, medicare_surtax])

        print(f"\n✅ Tax configuration updated for California")

        # Recalculate existing tax records
        record_count = conn.execute("SELECT COUNT(*) FROM tax_records").fetchone()[0]

        if record_count > 0:
            print(f"\n📊 Recalculating {record_count} existing tax records...")

            # Get all tax records
            records = conn.execute("""
                SELECT
                    id, gross_pnl, trading_fees, pnl_after_fees,
                    is_long_term, short_term_gain, long_term_gain
                FROM tax_records
            """).fetchall()

            # Recalculate each record
            for record in records:
                record_id = record[0]
                pnl_after_fees = record[3]
                is_long_term = record[4]
                short_term_gain = record[5]
                long_term_gain = record[6]

                # Calculate new tax owed
                if is_long_term:
                    tax_rate = effective_long_term
                    tax_owed = max(0, long_term_gain * tax_rate)
                else:
                    tax_rate = effective_short_term
                    tax_owed = max(0, short_term_gain * tax_rate)

                net_after_tax = pnl_after_fees - tax_owed

                # Update record
                conn.execute("""
                    UPDATE tax_records
                    SET
                        federal_tax_rate = ?,
                        state_tax_rate = ?,
                        medicare_surtax = ?,
                        effective_tax_rate = ?,
                        tax_owed = ?,
                        net_pnl_after_tax = ?
                    WHERE id = ?
                """, [
                    long_term_rate if is_long_term else short_term_rate,
                    state_rate,
                    medicare_surtax,
                    tax_rate,
                    tax_owed,
                    net_after_tax,
                    record_id
                ])

            print(f"   ✅ Recalculated {record_count} tax records")

        # Show updated summary
        config = conn.execute("SELECT base_annual_income, tax_year FROM tax_config WHERE id = 1").fetchone()
        base_income = config[0] if config and config[0] else 0.0
        tax_year = config[1] if config and config[1] else 2025

        summary = conn.execute("SELECT * FROM v_ytd_tax_summary").fetchone()
        if summary and summary[0] is not None:
            print(f"\n📈 {tax_year} YTD Tax Summary (Updated for California):")
            print(f"   ─────────────────────────────────")
            if base_income > 0:
                print(f"   Base income: ${base_income:,.0f} (from other sources)")
            print(f"   Trading gross P&L: ${summary[0]:,.2f}")
            print(f"   Trading fees (3%): ${summary[1]:,.2f}")
            print(f"   P&L after fees: ${summary[2]:,.2f}")
            print(f"   ─────────────────────────────────")
            print(f"   Tax on trading profits: ${summary[3]:,.2f}")
            print(f"   Net trading profit: ${summary[4]:,.2f}")
            print(f"   ─────────────────────────────────")
            if base_income > 0:
                print(f"   Total {tax_year} income: ${base_income + summary[2]:,.2f}")
            print(f"   Effective tax rate: {summary[5]*100:.1f}%")
            print(f"   Total trades: {summary[6]}")
        else:
            print(f"\n   ℹ️  No trading activity yet (no tax records)")

        # Show California tax details
        print(f"\n💡 California Tax Rates (Married Filing Jointly at $300K):")
        print(f"   Short-term capital gains:")
        print(f"   • Federal: 24.0%")
        print(f"   • California: 9.3%")
        print(f"   • Medicare: 3.8%")
        print(f"   • Total: 37.1%")
        print(f"   ")
        print(f"   Long-term capital gains:")
        print(f"   • Federal: 15.0%")
        print(f"   • California: 9.3%")
        print(f"   • Medicare: 3.8%")
        print(f"   • Total: 28.1%")
        print(f"   ")
        print(f"   Tax Impact on $10,000 Gain:")
        print(f"   • Short-term: $3,710 tax")
        print(f"   • Long-term: $2,810 tax")
        print(f"   • Savings by holding >365 days: $900 (9.0%)")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error updating tax rates: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    success = update_tax_rates()

    if success:
        print("\n✅ Tax rates updated for California (Married Filing Jointly)")
        print("   ")
        print("   Next steps:")
        print("   1. Verify setup: uv run python scripts/phase5_setup.py --quick")
        print("   2. View tax config: SELECT * FROM tax_config;")
    else:
        print("\n❌ Failed to update tax rates")

if __name__ == "__main__":
    main()
