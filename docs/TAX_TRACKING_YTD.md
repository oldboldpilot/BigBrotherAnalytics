# Year-To-Date Tax Tracking System

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-10
**Tax Year:** 2025
**Status:** Active

---

## Overview

The tax tracking system calculates and accumulates taxes **incrementally** throughout 2025 as trades close, with the assumption that you've already earned **$300,000** from other sources this year.

---

## How It Works

### 1. Base Income Context
```
2025 Base Income: $300,000 (from other sources)
└── Puts you in 35% federal bracket for short-term gains
└── 20% federal bracket for long-term gains
```

### 2. Incremental Tax Calculation

**Every time a trade closes:**
```
1. Calculate trading fees (3% of trade value)
2. Determine if short-term (<365 days) or long-term (>365 days)
3. Apply marginal tax rate:
   • Short-term: 43.8% (35% federal + 5% state + 3.8% Medicare)
   • Long-term: 28.8% (20% federal + 5% state + 3.8% Medicare)
4. Add to YTD cumulative total
5. Update dashboard real-time
```

### 3. Year-To-Date Accumulation

**YTD Summary shows cumulative totals:**
```
Base income:              $300,000  (from other sources)
Trading gross P&L:        $X,XXX    (accumulates each trade)
Trading fees (3%):        $XXX      (accumulates each trade)
P&L after fees:           $X,XXX    (accumulates each trade)
─────────────────────────────────────
Tax on trading profits:   $XXX      (accumulates each trade)
Net trading profit:       $X,XXX    (after tax, accumulates)
─────────────────────────────────────
Total 2025 income:        $30X,XXX  (base + trading)
```

---

## Example: Progressive Tax Accumulation

### Scenario: 5 Trades Throughout 2025

#### Trade 1 (January): +$1,000 Short-Term
```
Gross profit: $1,000
Fees (3%): -$60
After fees: $940
Tax (43.8%): -$412
Net: $528

YTD Cumulative:
  Total income: $300,940 (base + trading)
  Total tax: $412
  Net profit: $528
```

#### Trade 2 (March): +$2,500 Short-Term
```
Gross profit: $2,500
Fees (3%): -$150
After fees: $2,350
Tax (43.8%): -$1,029
Net: $1,321

YTD Cumulative:
  Total income: $303,290 (base + $3,290 trading)
  Total tax: $1,441 ($412 + $1,029)
  Net profit: $1,849 ($528 + $1,321)
```

#### Trade 3 (June): -$500 (Loss)
```
Gross profit: -$500
Fees (3%): -$30
After fees: -$530
Tax: $0 (loss = no tax)
Tax benefit: $232 (can offset future gains)
Net: -$530

YTD Cumulative:
  Total income: $302,760 (base + $2,760 trading)
  Total tax: $1,441 (no new tax on loss)
  Net profit: $1,319 ($1,849 - $530)
```

#### Trade 4 (September): +$5,000 Long-Term (held >365 days)
```
Gross profit: $5,000
Fees (3%): -$300
After fees: $4,700
Tax (28.8%): -$1,354 ← Lower rate!
Net: $3,346

YTD Cumulative:
  Total income: $307,460 (base + $7,460 trading)
  Total tax: $2,795 ($1,441 + $1,354)
  Net profit: $4,665 ($1,319 + $3,346)
```

#### Trade 5 (December): +$3,000 Short-Term
```
Gross profit: $3,000
Fees (3%): -$180
After fees: $2,820
Tax (43.8%): -$1,235
Net: $1,585

YTD 2025 Final:
  Total income: $310,280 (base + $10,280 trading)
  Total tax: $4,030
  Net profit: $6,250
  Effective rate: 39.2%

  Quarterly estimate Q4: ~$1,000 due Jan 15, 2026
```

---

## Dashboard Integration

### Real-Time Tax View

Navigate to **"Tax Implications"** in dashboard:

**YTD Summary (Live):**
- Base income: $300,000
- Trading activity: Updates with each closed trade
- Cumulative tax liability: Real-time
- Quarterly estimate projections

**Monthly Breakdown:**
```
January:   2 trades, $1,500 profit, $657 tax
February:  0 trades, $0 profit, $0 tax
March:     1 trade, $2,500 profit, $1,095 tax
...
December:  3 trades, $4,200 profit, $1,840 tax
```

**Symbol Tax Efficiency:**
```
AAPL:  65% efficiency (net/gross after tax)
TSLA:  58% efficiency
MSFT:  72% efficiency ← Held long-term
```

---

## Quarterly Tax Payments

### 2025 Payment Schedule

**Q1 (April 15):**
- Based on Jan-Mar trading profits
- Estimate: YTD tax × 0.25

**Q2 (June 15):**
- Based on Apr-May trading profits
- Estimate: YTD tax × 0.5 - Q1 paid

**Q3 (September 15):**
- Based on Jun-Aug trading profits
- Estimate: YTD tax × 0.75 - (Q1 + Q2) paid

**Q4 (January 15, 2026):**
- Based on Sep-Dec trading profits
- Estimate: YTD tax - (Q1 + Q2 + Q3) paid

### Safe Harbor Rule
With $300K+ income, pay **110% of 2024 tax** to avoid penalties.

---

## Tax-Loss Harvesting Strategy

### Throughout 2025

**Offset gains with losses:**
```
Example: By November
  YTD gains: $15,000
  Unrealized losses: -$3,000

Action: Sell losing positions before Dec 31
  Taxable gains: $15,000 - $3,000 = $12,000
  Tax before harvest: $6,570
  Tax after harvest: $5,256
  Saved: $1,314
```

**Rules:**
- Losses offset unlimited capital gains
- $3,000 excess loss deduction against ordinary income
- Remaining losses carry forward to 2026
- Watch out for wash sales (30-day rule)

---

## Year-End Reset

**December 31, 2025:**
```
Final YTD Summary Generated:
  2025 total income: $300,000 + trading profits
  2025 total tax: Cumulative from all trades
  2025 net profit: After-tax trading results

→ Form 1040 Schedule D (Capital Gains/Losses)
→ Form 8949 (Sales and Dispositions)
→ Tax return due: April 15, 2026
```

**January 1, 2026:**
```
System resets for 2026:
  New base income: (update if changed)
  YTD totals: Reset to $0
  Tax rates: Update for 2026 brackets
  New tax records for 2026 trades
```

---

## Command Reference

### Check Current YTD Status
```bash
uv run python scripts/monitoring/update_tax_config_ytd.py
```
Shows:
- Base income: $300,000
- YTD trading P&L
- Cumulative tax liability
- Total 2025 income

### Recalculate All Taxes
```bash
uv run python scripts/monitoring/calculate_taxes.py
```
Recalculates taxes for all closed trades and shows updated YTD summary.

### Update Tax Rates
```bash
uv run python scripts/monitoring/update_tax_rates_300k.py
```
Updates tax_config if rates change mid-year.

### View Dashboard
```bash
uv run streamlit run dashboard/app.py
# Navigate to: Tax Implications
```
Real-time cumulative YTD tax tracking.

---

## Database Structure

### Tax Records Table
```sql
-- Each closed trade gets one record
CREATE TABLE tax_records (
    id INTEGER PRIMARY KEY,
    trade_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    gross_pnl DOUBLE NOT NULL,
    trading_fees DOUBLE NOT NULL,
    pnl_after_fees DOUBLE NOT NULL,
    is_long_term BOOLEAN NOT NULL,
    effective_tax_rate DOUBLE NOT NULL,
    tax_owed DOUBLE NOT NULL,
    net_pnl_after_tax DOUBLE NOT NULL,
    ...
);
```

### YTD View (Automatic Aggregation)
```sql
-- Sums all 2025 trades
CREATE VIEW v_ytd_tax_summary AS
SELECT
    SUM(gross_pnl) as total_gross_pnl,
    SUM(trading_fees) as total_trading_fees,
    SUM(pnl_after_fees) as total_pnl_after_fees,
    SUM(tax_owed) as total_tax_owed,
    SUM(net_pnl_after_tax) as total_net_after_tax,
    COUNT(*) as total_trades
FROM tax_records
WHERE EXTRACT(YEAR FROM exit_time) = 2025;
```

### Tax Config
```sql
-- Stores base income assumption
UPDATE tax_config SET
    base_annual_income = 300000.00,
    tax_year = 2025,
    short_term_rate = 0.35,
    long_term_rate = 0.20
WHERE id = 1;
```

---

## Key Takeaways

1. **Base Income: $300,000** (already earned from other sources in 2025)

2. **Trading Profits: Incremental**
   - Each trade adds to YTD totals
   - Taxed at marginal rate (43.8% ST / 28.8% LT)
   - Dashboard shows cumulative tax liability

3. **Automatic Accumulation**
   - Every closed trade → tax calculated
   - YTD view → sums all 2025 trades
   - Real-time dashboard updates

4. **Year-Specific**
   - Tracks 2025 only (filters by exit_time year)
   - Resets January 1, 2026
   - Historical data preserved

5. **Tax Planning**
   - Quarterly estimated payments
   - Tax-loss harvesting throughout year
   - Year-end optimization

6. **Dashboard Monitoring**
   - Real-time YTD tax liability
   - Cumulative totals
   - Tax efficiency by symbol
   - Monthly breakdown

---

## Summary

**The system correctly:**
- ✅ Assumes $300K base income (from other sources)
- ✅ Taxes trading profits at marginal rate (43.8% / 28.8%)
- ✅ Accumulates incrementally as trades close
- ✅ Shows cumulative YTD totals
- ✅ Tracks 2025 only (year-specific)
- ✅ Updates dashboard real-time
- ✅ Supports quarterly tax planning
- ✅ Enables tax-loss harvesting

**Trading profits are added to $300K base, taxed at higher marginal rates, and tracked cumulatively throughout 2025.**

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** November 10, 2025
**System Status:** Configured for 2025 YTD incremental tax tracking
