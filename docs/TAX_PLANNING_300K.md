# Tax Planning for $300K Income Bracket

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-10
**Status:** Active Configuration

## Executive Summary

With $300K annual income, trading profits face **significantly higher tax rates**:
- **Short-term capital gains: 43.8%** (nearly half of profits!)
- **Long-term capital gains: 28.8%**
- **Tax savings for holding >365 days: 15.0%**

This document outlines tax-efficient trading strategies to minimize tax burden while maximizing after-tax returns.

---

## Current Tax Configuration

### Federal Tax Brackets (2025, Single Filer)
- **$300K income falls in 35% bracket** ($243,725 - $609,350)
- Short-term capital gains taxed as ordinary income: **35%**
- Long-term capital gains (>365 days): **20%** (over $492,300 threshold)

### Complete Tax Breakdown

| Component | Short-Term | Long-Term |
|-----------|------------|-----------|
| Federal tax | 35.0% | 20.0% |
| State tax | 5.0% | 5.0% |
| Medicare surtax (NIIT) | 3.8% | 3.8% |
| **Total effective rate** | **43.8%** | **28.8%** |
| **Trading fees** | **3.0%** | **3.0%** |
| **Combined tax + fees** | **46.8%** | **31.8%** |

### Tax Impact Examples

**$10,000 Gain:**
- Short-term: Pay $4,380 in taxes → Keep $5,620
- Long-term: Pay $2,880 in taxes → Keep $7,120
- **Holding >365 days saves $1,500 (15% more profit!)**

**$50,000 Gain:**
- Short-term: Pay $21,900 in taxes → Keep $28,100
- Long-term: Pay $14,400 in taxes → Keep $35,600
- **Holding >365 days saves $7,500**

**$100,000 Gain:**
- Short-term: Pay $43,800 in taxes → Keep $56,200
- Long-term: Pay $28,800 in taxes → Keep $71,200
- **Holding >365 days saves $15,000**

---

## Tax-Efficient Trading Strategies

### 1. Holding Period Optimization

**Short-Term Trading (< 365 days):**
- ❌ 43.8% tax rate - extremely expensive
- ❌ Best avoided unless conviction is very high
- ✅ Only for high-probability setups (>65% win rate)
- ✅ Use strict stop-losses to minimize taxable gains

**Long-Term Trading (> 365 days):**
- ✅ 28.8% tax rate - 15% savings
- ✅ Preferred for large positions
- ✅ Consider for core holdings in strong sectors
- ✅ Reduces trading frequency and fees

### 2. Tax-Loss Harvesting

**Strategy:** Offset gains with losses to reduce taxable income.

**Rules:**
- Losses can offset unlimited capital gains
- Net losses can offset $3,000 of ordinary income per year
- Excess losses carry forward indefinitely

**Watch Out for Wash Sales:**
- Cannot claim loss if you buy same/similar security within 30 days
- System automatically tracks wash sales
- Disallowed losses defer to replacement security's cost basis

**Example:**
```
Gains:  $20,000 (would owe $8,760 at 43.8%)
Losses: -$5,000 (harvested before year-end)
─────────────────
Net:    $15,000 (owe $6,570)
Tax savings: $2,190
```

### 3. Position Sizing for Tax Efficiency

Given 43.8% short-term tax rate, adjust position sizing:

**Before-tax thinking:**
- Win $1,000 → Keep $1,000
- Lose $1,000 → Lose $1,000
- Risk/reward: 1:1

**After-tax reality:**
- Win $1,000 → Keep $562 (after 43.8% tax)
- Lose $1,000 → Deduct $438 (tax benefit)
- **Effective risk/reward: 1:0.56**

**Implication:** Need **78% win rate** just to break even on 1:1 setups!

**Better approach:**
- Target 2:1 or 3:1 reward/risk ratios
- Only take trades with >60% probability
- Favor long-term positions for large gains

### 4. Trading Fee Management (3%)

**3% fee applies to trade value (entry + exit):**

**Example:**
- Buy $10,000 of stock
- Sell for $12,000 (20% gain)
- Total trade value: $22,000
- Trading fees: $660 (3% of $22,000)
- Gross profit: $2,000
- Net after fees: $1,340

**Minimize fee impact:**
- ✅ Reduce trading frequency
- ✅ Avoid round-trips on small gains (<5%)
- ✅ Hold positions longer (fewer trades)
- ✅ Focus on high-conviction trades
- ❌ Avoid scalping (fees eat profits)

---

## Tax-Advantaged Account Strategies

### Consider These for High-Frequency Trading:

1. **IRA/Roth IRA**
   - Traditional IRA: Tax-deferred growth, deduct contributions
   - Roth IRA: Tax-free growth and withdrawals (if qualified)
   - No capital gains tax inside account
   - Contribution limit: $7,000/year (2025)

2. **Solo 401(k)** (if self-employed)
   - Contribution limit: $69,000/year (2025)
   - Tax-deferred growth
   - Consider for majority of trading activity

3. **Health Savings Account (HSA)**
   - Triple tax advantage
   - Contribution limit: $4,150/year (2025, individual)
   - Can invest in stocks tax-free

### Taxable Account Best Practices:
- Use for long-term positions (>365 days)
- Tax-loss harvest systematically
- Defer gains to future years when possible
- Track cost basis meticulously

---

## Quarterly Tax Planning

### Estimated Tax Payments

With $300K income + trading profits, you likely need quarterly estimated payments:

**Deadlines:**
- Q1: April 15
- Q2: June 15
- Q3: September 15
- Q4: January 15 (following year)

**Safe harbor:** Pay 110% of prior year's tax (since AGI > $150K)

**Trading profit estimation:**
```
Quarterly gross profit: $10,000
Trading fees (3%):      -$600
Net profit:             $9,400
Tax (43.8%):            -$4,117
─────────────────────────────
Quarterly estimated payment: ~$4,200
```

### Year-End Tax Planning

**Before December 31:**
1. ✅ Review YTD gains/losses
2. ✅ Harvest losses to offset gains
3. ✅ Defer gains to next year if possible
4. ✅ Consider charitable donations of appreciated securities
5. ✅ Max out retirement account contributions

**Dashboard View:**
- YTD P&L tracker shows real-time tax estimate
- Tax Implications view shows detailed breakdown
- Monthly tax summary for planning

---

## System Configuration

### Current Settings (data/bigbrother.duckdb)

```sql
-- Tax configuration for $300K income
UPDATE tax_config SET
    short_term_rate = 0.35,      -- 35% federal
    long_term_rate = 0.20,       -- 20% federal
    state_tax_rate = 0.05,       -- 5% state
    medicare_surtax = 0.038,     -- 3.8% NIIT
    trading_fee_percent = 0.03,  -- 3% fees
    track_wash_sales = true,
    wash_sale_window_days = 30
WHERE id = 1;
```

### Update Tax Rates Anytime:

```bash
# Update to $300K income bracket
uv run python scripts/monitoring/update_tax_rates_300k.py

# Calculate taxes for all trades
uv run python scripts/monitoring/calculate_taxes.py

# View results in dashboard
uv run streamlit run dashboard/app.py
# → Navigate to "Tax Implications" view
```

---

## Trading Strategy Adjustments

### Given 43.8% Short-Term Tax Rate:

**1. Increase Win Rate Requirements**
- Before: 55% win rate profitable
- After-tax: Need 60-65% win rate
- Solution: Trade only A+ setups

**2. Favor Asymmetric Payoffs**
- Target 3:1 or better reward/risk
- One $3K win covers three $1K losses (after tax)
- Avoid low probability / low reward trades

**3. Extend Holding Periods**
- Consider 366+ day holds for large positions
- 15% tax savings = significant edge
- Plan entries for potential long-term holds

**4. Tax-Loss Harvest Strategically**
- Harvest losses in profitable years
- Carry forward excess losses
- Offset short-term gains first (highest rate)

**5. Sector Rotation Timing**
- If sector rotation signal is strong, enter large position
- Hold >365 days if fundamentals remain strong
- Save 15% on gains vs frequent rotation

---

## Dashboard Integration

### Tax Implications View

Navigate to "Tax Implications" in dashboard to see:

1. **YTD Tax Summary:**
   - Gross P&L
   - Trading fees (3%)
   - P&L after fees
   - Tax owed (at 43.8% / 28.8%)
   - Net after-tax profit
   - Effective tax rate
   - Wash sale tracking

2. **Monthly Breakdown:**
   - Tax liability by month
   - Short-term vs long-term gains
   - Tax efficiency trends

3. **Symbol Analysis:**
   - Tax efficiency by symbol
   - Which positions most tax-efficient
   - Optimize future allocations

### Real-Time Tax Tracking

Every trade automatically:
- ✅ Calculates 3% trading fee
- ✅ Determines short-term vs long-term
- ✅ Applies correct tax rate (43.8% or 28.8%)
- ✅ Detects wash sales
- ✅ Updates YTD tax liability
- ✅ Projects quarterly estimated payment

---

## Paper Trading Considerations

### Phase 5: Paper Trading with Real Tax Impact

**Conservative limits:**
- Max position: $100
- Max daily loss: $50
- Max concurrent positions: 3

**Tax impact simulation:**
```
$100 position × 20% gain = $20 profit
Trading fees (3%): -$6
Net before tax: $14
Short-term tax (43.8%): -$6.13
Net after-tax: $7.87

ROI before tax: 14%
ROI after tax: 7.87%
─────────────────────
Tax reduces ROI by 43.8%!
```

**Key insight:** Every trade must clear **50% higher hurdle** to be profitable after tax.

---

## Long-Term Planning

### Goal: Minimize Total Tax Burden

**Year 1 (Current):**
- Income: $300K
- Tax rate: 43.8% short-term
- Strategy: Focus on long-term holds, minimize short-term trades

**Year 2-3:**
- Build long-term portfolio (>365 days)
- Tax-loss harvest systematically
- Transition to 28.8% rate on majority of gains

**Year 5+:**
- Consider moving trading to tax-advantaged accounts
- Evaluate relocation to lower-tax state (0% state tax saves 5%)
- Optimize entity structure (LLC, S-corp, etc.)

### Tax Savings Potential

**Aggressive short-term trading:**
- $100K profit
- Taxes: $43,800 (43.8%)
- Keep: $56,200

**Tax-optimized long-term approach:**
- $100K profit (same trades, held longer)
- Taxes: $28,800 (28.8%)
- Keep: $71,200
- **Extra profit: $15,000 (26.7% more!)**

---

## Resources

### IRS Publications
- [Publication 550](https://www.irs.gov/pub/irs-pdf/p550.pdf) - Investment Income and Expenses
- [Publication 564](https://www.irs.gov/pub/irs-pdf/p564.pdf) - Mutual Fund Distributions
- [Topic 409](https://www.irs.gov/taxtopics/tc409) - Capital Gains and Losses

### Tax Planning Tools
- Dashboard Tax Implications view (real-time)
- YTD tax summary (updated with each trade)
- Wash sale detection (automatic)

### Professional Advice
Consider consulting:
- CPA specializing in active traders
- Tax attorney for entity structuring
- Financial advisor for tax-advantaged accounts

---

## Summary: Key Takeaways

1. **Tax rates are VERY HIGH at $300K income:**
   - Short-term: 43.8% (nearly half!)
   - Long-term: 28.8% (15% savings)

2. **Trading fees add 3% to every trade** → Total burden up to 46.8%

3. **Holding >365 days saves 15%** → Significant edge

4. **Tax-loss harvesting is critical** → Offset gains, carry forward losses

5. **Position sizing must account for after-tax returns** → Need higher win rate

6. **Dashboard tracks everything** → Real-time tax liability, wash sales, efficiency

7. **Consider tax-advantaged accounts** → IRA, 401(k), HSA for high-frequency trading

8. **Plan quarterly estimated payments** → Avoid penalties

9. **Year-end planning is essential** → Harvest losses, defer gains, max contributions

10. **Long-term strategy wins** → Transition from 43.8% to 28.8% rate over time

---

**Bottom Line:**
At $300K income, taxes are the single largest trading expense. Every strategy must be evaluated on **after-tax** returns. The system now accurately reflects this reality.

**Next Steps:**
1. ✅ Monitor dashboard Tax Implications view
2. ✅ Focus on high-probability trades (>65% win rate)
3. ✅ Consider long-term holds for large positions
4. ✅ Tax-loss harvest systematically
5. ✅ Plan quarterly estimated payments

**Trading is now configured for $300K income tax bracket.**
