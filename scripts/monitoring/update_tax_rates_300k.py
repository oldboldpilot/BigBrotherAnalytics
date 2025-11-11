#!/usr/bin/env python3
"""
Update Tax Rates for $300K Income Bracket

Adjusts tax configuration to reflect $300K annual income:
- Short-term capital gains: 35% federal (not 24%)
- Long-term capital gains: 20% federal (not 15%)
- Combined with 5% state + 3.8% Medicare surtax

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Usage:
    uv run python scripts/monitoring/update_tax_rates_300k.py
"""

import duckdb
from pathlib import Path
from datetime import datetime

def update_tax_rates():
    """Update tax_config for $300K income bracket"""

    db_path = Path(__file__).parent.parent.parent / "data" / "bigbrother.duckdb"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    print("📊 Updating tax rates for $300K income bracket...")

    try:
        conn = duckdb.connect(str(db_path))

        # Federal tax brackets for 2025 (Single filer):
        # $100,000 - $191,950: 24%
        # $191,950 - $243,725: 32%
        # $243,725 - $609,350: 35% ← $300K falls here

        # Long-term capital gains for $300K income:
        # Over $518,900: 20%
        # $44,625 - $492,300: 15%
        # Since $300K income + gains likely exceeds $492,300, use 20%

        # Update configuration
        short_term_rate = 0.35  # 35% federal for $300K income
        long_term_rate = 0.20   # 20% federal for high income
        state_rate = 0.05       # 5% state
        medicare_surtax = 0.038 # 3.8% NIIT (Net Investment Income Tax)

        # Calculate effective rates
        effective_short_term = short_term_rate + state_rate + medicare_surtax
        effective_long_term = long_term_rate + state_rate + medicare_surtax

        print(f"\n📋 New Tax Rates for $300K Income:")
        print(f"   Federal short-term: {short_term_rate*100:.0f}%")
        print(f"   Federal long-term: {long_term_rate*100:.0f}%")
        print(f"   State tax: {state_rate*100:.1f}%")
        print(f"   Medicare surtax: {medicare_surtax*100:.1f}%")
        print(f"   ─────────────────────────────────")
        print(f"   Effective short-term: {effective_short_term*100:.1f}%")
        print(f"   Effective long-term: {effective_long_term*100:.1f}%")

        # Update tax_config table
        conn.execute("""
            UPDATE tax_config
            SET
                short_term_rate = ?,
                long_term_rate = ?,
                state_tax_rate = ?,
                medicare_surtax = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, [short_term_rate, long_term_rate, state_rate, medicare_surtax])

        print(f"\n✅ Tax configuration updated")

        # Check if there are any existing tax records to recalculate
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
        summary = conn.execute("SELECT * FROM v_ytd_tax_summary").fetchone()
        if summary and summary[0] is not None:  # Check if there's actual data
            print(f"\n📈 Updated YTD Tax Summary:")
            print(f"   Gross P&L: ${summary[0]:,.2f}")
            print(f"   Trading Fees (3%): ${summary[1]:,.2f}")
            print(f"   P&L After Fees: ${summary[2]:,.2f}")
            print(f"   Tax Owed: ${summary[3]:,.2f}")
            print(f"   Net After Tax: ${summary[4]:,.2f}")
            print(f"   Effective Tax Rate: {summary[5]*100:.1f}%")
            print(f"   Total Trades: {summary[6]}")
        else:
            print(f"\n   ℹ️  No trading activity yet (no tax records)")

        # Show tax planning insights
        print(f"\n💡 Tax Planning for $300K Income Bracket:")
        print(f"   • Short-term trades taxed at {effective_short_term*100:.1f}% (very high!)")
        print(f"   • Long-term trades taxed at {effective_long_term*100:.1f}%")
        print(f"   • Tax savings for holding >365 days: {(effective_short_term - effective_long_term)*100:.1f}%")
        print(f"   • On $10K gain: ${10000*effective_short_term:,.0f} (short) vs ${10000*effective_long_term:,.0f} (long)")
        print(f"   • Annual tax-loss harvesting limit: $3,000")
        print(f"   • Consider tax-advantaged accounts for high-frequency trading")

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
        print("\n✅ Tax rates updated for $300K income bracket")
    else:
        print("\n❌ Failed to update tax rates")

if __name__ == "__main__":
    main()
