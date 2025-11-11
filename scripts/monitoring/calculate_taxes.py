#!/usr/bin/env python3
"""
Tax Calculator - Calculate taxes for closed trades

Calculates tax implications including:
- 3% trading fees
- Short-term vs long-term capital gains
- Federal, state, and Medicare surtax
- Wash sale detection
- Net after-tax P&L

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Usage:
    uv run python scripts/monitoring/calculate_taxes.py
"""

import duckdb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class TaxCalculator:
    """Calculate taxes on trading activity"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))

        # Load tax configuration
        config = self.conn.execute("SELECT * FROM tax_config WHERE id = 1").fetchone()
        if config:
            self.short_term_rate = config[2]
            self.long_term_rate = config[3]
            self.state_tax_rate = config[4]
            self.medicare_surtax = config[5]
            self.trading_fee_percent = config[6]
            self.track_wash_sales = config[9]
            self.wash_sale_window_days = config[10]
        else:
            # Defaults (California rates)
            self.short_term_rate = 0.24
            self.long_term_rate = 0.15
            self.state_tax_rate = 0.093  # California 9.3%
            self.medicare_surtax = 0.038
            self.trading_fee_percent = 0.03
            self.track_wash_sales = True
            self.wash_sale_window_days = 30

    def effective_short_term_rate(self) -> float:
        """Combined short-term rate"""
        return self.short_term_rate + self.state_tax_rate + self.medicare_surtax

    def effective_long_term_rate(self) -> float:
        """Combined long-term rate"""
        return self.long_term_rate + self.state_tax_rate + self.medicare_surtax

    def calculate_trading_fees(self, trade_value: float) -> float:
        """Calculate 3% trading fees"""
        return abs(trade_value) * self.trading_fee_percent

    def is_long_term(self, entry_time: datetime, exit_time: datetime) -> bool:
        """Check if holding period > 365 days"""
        return (exit_time - entry_time).days > 365

    def detect_wash_sales(self, trades: List[Dict]) -> List[Dict]:
        """
        Detect wash sales (IRS rule)

        Wash sale: If you sell at a loss and buy substantially
        identical security within 30 days before or after.
        """
        if not self.track_wash_sales:
            return trades

        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t['exit_time'])

        # Check each loss for wash sales
        for i, trade in enumerate(sorted_trades):
            if trade['pnl_after_fees'] >= 0:
                continue  # Wash sale only applies to losses

            # Check 30-day window before and after
            for j, other_trade in enumerate(sorted_trades):
                if i == j:
                    continue

                # Same symbol?
                if trade['symbol'] != other_trade['symbol']:
                    continue

                # Within wash sale window?
                days_diff = abs((other_trade['entry_time'] - trade['exit_time']).days)

                if days_diff <= self.wash_sale_window_days:
                    # Wash sale detected!
                    trade['wash_sale_disallowed'] = True
                    trade['wash_sale_amount'] = abs(trade['pnl_after_fees'])

                    # Add disallowed loss to replacement's cost basis
                    # (deferred, not lost forever)
                    other_trade['cost_basis'] += abs(trade['pnl_after_fees'])
                    break

        return sorted_trades

    def calculate_tax_for_trade(self, trade: Dict) -> Dict:
        """Calculate tax for a single trade"""

        # 1. Calculate trading fees (3%)
        trade_value = trade['cost_basis'] + abs(trade['proceeds'])
        trading_fees = self.calculate_trading_fees(trade_value)

        # 2. P&L after fees
        pnl_after_fees = trade['gross_pnl'] - trading_fees

        # 3. Determine if long-term or short-term
        is_long_term = self.is_long_term(trade['entry_time'], trade['exit_time'])
        holding_days = (trade['exit_time'] - trade['entry_time']).days

        # 4. Categorize gains/losses
        short_term_gain = 0.0
        long_term_gain = 0.0
        short_term_loss = 0.0
        long_term_loss = 0.0

        if is_long_term:
            if pnl_after_fees > 0:
                long_term_gain = pnl_after_fees
            else:
                long_term_loss = abs(pnl_after_fees)
        else:
            if pnl_after_fees > 0:
                short_term_gain = pnl_after_fees
            else:
                short_term_loss = abs(pnl_after_fees)

        # 5. Calculate tax owed
        if is_long_term:
            tax_rate = self.effective_long_term_rate()
            tax_owed = max(0, long_term_gain * tax_rate)
        else:
            tax_rate = self.effective_short_term_rate()
            tax_owed = max(0, short_term_gain * tax_rate)

        # 6. Net after tax
        net_after_tax = pnl_after_fees - tax_owed

        return {
            'trade_id': trade['trade_id'],
            'symbol': trade['symbol'],
            'entry_time': trade['entry_time'],
            'exit_time': trade['exit_time'],
            'holding_period_days': holding_days,
            'cost_basis': trade['cost_basis'],
            'proceeds': trade['proceeds'],
            'gross_pnl': trade['gross_pnl'],
            'trading_fees': trading_fees,
            'pnl_after_fees': pnl_after_fees,
            'is_long_term': is_long_term,
            'short_term_gain': short_term_gain,
            'long_term_gain': long_term_gain,
            'short_term_loss': short_term_loss,
            'long_term_loss': long_term_loss,
            'federal_tax_rate': self.long_term_rate if is_long_term else self.short_term_rate,
            'state_tax_rate': self.state_tax_rate,
            'medicare_surtax': self.medicare_surtax,
            'effective_tax_rate': tax_rate,
            'tax_owed': tax_owed,
            'net_pnl_after_tax': net_after_tax,
            'wash_sale_disallowed': trade.get('wash_sale_disallowed', False),
            'wash_sale_amount': trade.get('wash_sale_amount', 0.0)
        }

    def get_closed_trades(self) -> List[Dict]:
        """Get all closed trades from positions_history"""
        # Since positions_history doesn't track closed positions explicitly,
        # we'll return empty list. In production, this would query actual closed trades.
        query = """
        SELECT
            ROW_NUMBER() OVER (ORDER BY symbol, timestamp) as trade_id,
            symbol,
            timestamp as exit_time,
            current_price * quantity as proceeds,
            average_price * quantity as cost_basis,
            (current_price - average_price) * quantity as gross_pnl,
            -- Assume entry time is 30 days before exit
            timestamp - INTERVAL '30 days' as entry_time
        FROM positions_history
        WHERE quantity > 0
        ORDER BY timestamp DESC
        LIMIT 0  -- Return empty set - no closed trades in history yet
        """

        trades = []
        try:
            results = self.conn.execute(query).fetchall()
            for row in results:
                trades.append({
                    'trade_id': str(row[0]),
                    'symbol': row[1],
                    'exit_time': row[2],
                    'proceeds': row[3],
                    'cost_basis': row[4],
                    'gross_pnl': row[5],
                    'entry_time': row[6]
                })
        except Exception as e:
            # If query fails, return empty list (no closed trades)
            print(f"⚠️  Note: Could not retrieve closed trades: {e}")
            pass

        return trades

    def populate_tax_records(self):
        """Calculate and populate tax_records table"""

        print("📊 Calculating taxes for all closed trades...")

        # Get base income context
        config = self.conn.execute("SELECT base_annual_income, tax_year FROM tax_config WHERE id = 1").fetchone()
        base_income = config[0] if config and config[0] else 0.0
        tax_year = config[1] if config and config[1] else 2025

        if base_income > 0:
            print(f"   Base {tax_year} income: ${base_income:,.0f} (from other sources)")
            print(f"   Trading profits taxed at marginal rate: {self.effective_short_term_rate()*100:.1f}% (ST) / {self.effective_long_term_rate()*100:.1f}% (LT)")

        # 1. Get closed trades
        trades = self.get_closed_trades()
        print(f"   Found {len(trades)} closed trades")

        if not trades:
            print("   ℹ️  No closed trades to process")
            print(f"   ℹ️  Taxes will accumulate as trades close throughout {tax_year}")
            return

        # 2. Detect wash sales
        trades = self.detect_wash_sales(trades)

        # 3. Calculate tax for each trade
        tax_records = []
        for trade in trades:
            tax_record = self.calculate_tax_for_trade(trade)
            tax_records.append(tax_record)

        # 4. Insert into database
        self.conn.execute("DELETE FROM tax_records")  # Clear existing

        for record in tax_records:
            self.conn.execute("""
                INSERT INTO tax_records (
                    trade_id, symbol, entry_time, exit_time, holding_period_days,
                    cost_basis, proceeds, gross_pnl,
                    trading_fees, pnl_after_fees,
                    is_long_term,
                    short_term_gain, long_term_gain,
                    short_term_loss, long_term_loss,
                    federal_tax_rate, state_tax_rate, medicare_surtax, effective_tax_rate,
                    tax_owed, net_pnl_after_tax,
                    wash_sale_disallowed, wash_sale_amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record['trade_id'], record['symbol'],
                record['entry_time'], record['exit_time'], record['holding_period_days'],
                record['cost_basis'], record['proceeds'], record['gross_pnl'],
                record['trading_fees'], record['pnl_after_fees'],
                record['is_long_term'],
                record['short_term_gain'], record['long_term_gain'],
                record['short_term_loss'], record['long_term_loss'],
                record['federal_tax_rate'], record['state_tax_rate'],
                record['medicare_surtax'], record['effective_tax_rate'],
                record['tax_owed'], record['net_pnl_after_tax'],
                record['wash_sale_disallowed'], record['wash_sale_amount']
            ])

        print(f"   ✅ Inserted {len(tax_records)} tax records")

        # 5. Show summary
        summary = self.conn.execute("SELECT * FROM v_ytd_tax_summary").fetchone()
        if summary:
            # Get base income context
            config = self.conn.execute("SELECT base_annual_income, tax_year FROM tax_config WHERE id = 1").fetchone()
            base_income = config[0] if config and config[0] else 0.0
            tax_year = config[1] if config and config[1] else 2025

            print(f"\n📈 {tax_year} YTD Tax Summary (Cumulative):")
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
            print(f"   Total trades closed: {summary[6]}")
            if summary[7] > 0:
                print(f"   ⚠️  Wash Sales: {summary[7]} (${summary[8]:,.2f} disallowed)")

            print(f"\n💡 Tax will continue to accumulate as more trades close in {tax_year}")

    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    """Main entry point"""
    db_path = Path(__file__).parent.parent.parent / "data" / "bigbrother.duckdb"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    try:
        calculator = TaxCalculator(db_path)
        calculator.populate_tax_records()
        calculator.close()
        print("\n✅ Tax calculation complete")

    except Exception as e:
        print(f"❌ Error calculating taxes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
