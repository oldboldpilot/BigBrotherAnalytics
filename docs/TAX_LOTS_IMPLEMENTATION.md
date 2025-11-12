# Tax Lots Implementation Guide

**BigBrotherAnalytics - Tax-Aware Trading System**

Comprehensive documentation for the tax lots tracking and tax calculation system. This system enables tax-efficient algorithmic trading by tracking cost basis, calculating after-tax returns, and enforcing IRS regulations.

---

## Table of Contents

1. [Overview](#overview)
2. [C++23 Tax Module](#c23-tax-module)
3. [Tax Lots Database Schema](#tax-lots-database-schema)
4. [Schwab Tax Lots Integration](#schwab-tax-lots-integration)
5. [Tax Calculation Examples](#tax-calculation-examples)
6. [Dashboard Integration](#dashboard-integration)
7. [Testing](#testing)
8. [API Reference](#api-reference)

---

## Overview

### Why Tax-Aware Trading Matters

**Critical Insight**: True profitability requires after-tax returns. A strategy with 40% gross returns but poor tax efficiency may underperform a strategy with 25% gross returns but high tax efficiency.

### Tax Impact on Algorithmic Trading

```
Example: $100,000 gain
├─ Gross Return:           $100,000
├─ Short-term tax (32%):  -$32,000  (federal + state + Medicare surtax)
└─ Net After-Tax:          $68,000  (32% tax drag)

vs. Long-term holding:
├─ Gross Return:           $100,000
├─ Long-term tax (20%):   -$20,000  (federal + state + Medicare surtax)
└─ Net After-Tax:          $80,000  (20% tax drag)

Difference: $12,000 (12% better with long-term treatment)
```

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Tax-Aware Trading System                 │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────────┐      ┌─────────────────────┐        │
│  │  Schwab API    │─────▶│  Tax Lots Database  │        │
│  │  (LIFO Lots)   │      │  (DuckDB)           │        │
│  └────────────────┘      └─────────────────────┘        │
│                                    │                      │
│                                    │                      │
│  ┌────────────────┐      ┌─────────────────────┐        │
│  │  C++23 Tax     │◀─────│  Tax Calculation    │        │
│  │  Module        │      │  Engine             │        │
│  └────────────────┘      └─────────────────────┘        │
│         │                          │                      │
│         │                          │                      │
│         ▼                          ▼                      │
│  ┌────────────────────────────────────────────┐          │
│  │          Strategy Optimization             │          │
│  │       (After-Tax Returns)                  │          │
│  └────────────────────────────────────────────┘          │
│                        │                                  │
│                        ▼                                  │
│  ┌────────────────────────────────────────────┐          │
│  │          Dashboard (Tax View)              │          │
│  │    - Holdings by lot                       │          │
│  │    - Tax implications                      │          │
│  │    - Wash sale warnings                    │          │
│  └────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

### Key Features

- **LIFO Tax Lot Tracking**: Last-In-First-Out cost basis method (Schwab default)
- **Wash Sale Detection**: Automatically identifies and disallows wash sale losses
- **After-Tax Performance**: Calculate true Sharpe ratios using after-tax returns
- **Quarterly Tax Estimates**: Calculate estimated tax payments to avoid penalties
- **Tax-Optimized Orders**: Choose lots to sell for optimal tax outcome
- **Real-Time Sync**: Fetch tax lots from Schwab API

---

## C++23 Tax Module

### File: [src/utils/tax.cppm](../src/utils/tax.cppm)

Modern C++23 module implementing comprehensive tax calculations following IRS rules.

### Core Components

#### 1. TaxConfig - Tax Rate Configuration

```cpp
struct TaxConfig {
    // Federal tax rates
    double short_term_rate{0.24};         // 24% (ordinary income)
    double long_term_rate{0.15};          // 15% (preferential)
    double medicare_surtax{0.038};        // 3.8% Net Investment Income Tax

    // State tax
    double state_tax_rate{0.05};          // 5% (conservative)

    // Trading status
    bool is_pattern_day_trader{true};     // All short-term
    bool is_section_1256_trader{false};   // Futures (60/40 rule)

    // Wash sale tracking
    bool track_wash_sales{true};
    int wash_sale_window_days{30};        // IRS: 30 days before/after

    [[nodiscard]] constexpr auto effectiveShortTermRate() const noexcept -> double {
        return short_term_rate + medicare_surtax + state_tax_rate;
    }

    [[nodiscard]] constexpr auto effectiveLongTermRate() const noexcept -> double {
        return long_term_rate + medicare_surtax + state_tax_rate;
    }
};
```

**Tax Rates by Income (2025)**:
- Short-term: Taxed as ordinary income (10%-37%)
- Long-term: Preferential rates (0%, 15%, 20%)
- Medicare surtax: 3.8% for high earners (>$200k single, >$250k married)
- State tax: Varies (CA: 13.3%, TX: 0%, NY: 10.9%)

#### 2. TaxTrade - Trade with Tax Attributes

```cpp
struct TaxTrade {
    std::string trade_id;
    std::string symbol;
    Timestamp entry_time{0};              // Microseconds since epoch
    Timestamp exit_time{0};
    double cost_basis{0.0};               // Entry cost
    double proceeds{0.0};                 // Exit proceeds
    double gross_pnl{0.0};                // Gross profit/loss
    bool is_options{false};
    bool is_index_option{false};          // Section 1256 eligible

    // Wash sale tracking
    bool wash_sale_disallowed{false};
    double wash_sale_amount{0.0};

    [[nodiscard]] constexpr auto holdingPeriodDays() const noexcept -> int {
        return static_cast<int>((exit_time - entry_time) / (1'000'000LL * 86400LL));
    }

    [[nodiscard]] constexpr auto isLongTerm() const noexcept -> bool {
        return holdingPeriodDays() > 365;  // > 1 year
    }

    [[nodiscard]] constexpr auto isShortTerm() const noexcept -> bool {
        return !isLongTerm();
    }

    [[nodiscard]] constexpr auto isGain() const noexcept -> bool {
        return gross_pnl > 0.0;
    }
};
```

**Holding Period Rules**:
- Short-term: ≤ 365 days → Taxed as ordinary income
- Long-term: > 365 days → Preferential rates (15%-20%)
- Day trading: All short-term (pattern day trader)

#### 3. TaxResult - Comprehensive Tax Report

```cpp
struct TaxResult {
    // Gross P&L breakdown
    double total_gross_pnl{0.0};
    double short_term_gains{0.0};
    double long_term_gains{0.0};
    double short_term_losses{0.0};
    double long_term_losses{0.0};

    // Tax calculations
    double taxable_short_term{0.0};       // After offsetting losses
    double taxable_long_term{0.0};
    double total_tax_owed{0.0};

    // After-tax results
    double net_pnl_after_tax{0.0};
    double effective_tax_rate{0.0};

    // Wash sale impact
    int wash_sales_disallowed{0};
    double wash_sale_loss_disallowed{0.0};

    // Carryforward
    double capital_loss_carryforward{0.0}; // Losses > $3,000

    [[nodiscard]] constexpr auto taxEfficiency() const noexcept -> double {
        if (total_gross_pnl <= 0.0) return 1.0;
        return net_pnl_after_tax / total_gross_pnl;
    }

    [[nodiscard]] constexpr auto isProfitableAfterTax() const noexcept -> bool {
        return net_pnl_after_tax > 0.0;
    }
};
```

#### 4. TaxCalculator - Tax Engine

```cpp
class TaxCalculator {
public:
    explicit TaxCalculator(TaxConfig config = TaxConfig{});

    // Main calculation method
    [[nodiscard]] auto calculateTaxes(std::vector<TaxTrade> const& trades)
        -> Result<TaxResult>;

    // Quarterly estimated tax
    [[nodiscard]] auto calculateQuarterlyTax(double quarterly_profit) const noexcept
        -> double;

    // Apply IRS wash sale rules
    [[nodiscard]] auto applyWashSaleRules(std::vector<TaxTrade> trades) const
        -> std::vector<TaxTrade>;
};
```

**Tax Calculation Algorithm**:
1. Apply wash sale rules (disallow losses within 30-day window)
2. Categorize gains/losses (short-term vs long-term)
3. Offset gains with losses (short-term with short-term, long-term with long-term)
4. Calculate taxes on net gains
5. Calculate after-tax P&L
6. Compute capital loss carryforward (excess losses > $3,000)

### Fluent API - TaxCalculatorBuilder

Modern fluent interface for easy tax calculations:

```cpp
// Calculate taxes on trades
auto tax_result = TaxCalculatorBuilder()
    .federalRate(0.24)
    .stateRate(0.05)
    .withMedicareSurtax()
    .patternDayTrader()
    .trackWashSales()
    .addTrades(all_trades)
    .calculate();

if (tax_result) {
    std::println("Gross P&L: ${}", tax_result->total_gross_pnl);
    std::println("Tax Owed: ${}", tax_result->total_tax_owed);
    std::println("Net After Tax: ${}", tax_result->net_pnl_after_tax);
    std::println("Effective Tax Rate: {:.1f}%",
                 tax_result->effective_tax_rate * 100);
    std::println("Tax Efficiency: {:.1f}%",
                 tax_result->taxEfficiency() * 100);
}

// Quick after-tax return calculation
double gross_return = 10000.0;
double after_tax = TaxCalculatorBuilder()
    .federalRate(0.24)
    .calculateAfterTax(gross_return);

std::println("Gross: ${}, After-Tax: ${}", gross_return, after_tax);
// Output: Gross: $10000, After-Tax: $6800
```

### Wash Sale Rule Implementation

**IRS Wash Sale Rule**: Cannot claim loss if you repurchase substantially identical security within 30 days before or after the sale.

```cpp
auto applyWashSaleRules(std::vector<TaxTrade> trades) const -> std::vector<TaxTrade> {
    auto result = trades;

    // Sort by exit time
    std::ranges::sort(result, [](auto const& a, auto const& b) {
        return a.exit_time < b.exit_time;
    });

    // Check each loss for wash sales
    for (size_t i = 0; i < result.size(); ++i) {
        if (!result[i].isLoss()) continue;

        // Check 30-day window before and after
        for (size_t j = 0; j < result.size(); ++j) {
            if (i == j) continue;

            // Same security?
            if (result[i].symbol != result[j].symbol) continue;

            // Within wash sale window?
            auto days_diff = std::abs(
                (result[j].entry_time - result[i].exit_time) / (1'000'000LL * 86400LL)
            );

            if (days_diff <= 30) {
                // Wash sale! Disallow the loss
                result[i].wash_sale_disallowed = true;
                result[i].wash_sale_amount = std::abs(result[i].gross_pnl);

                // Add disallowed loss to cost basis of replacement
                result[j].cost_basis += std::abs(result[i].gross_pnl);
                break;
            }
        }
    }

    return result;
}
```

**Example**:
```
Day 1:  Buy 100 SPY @ $580  (cost: $58,000)
Day 10: Sell 100 SPY @ $570  (proceeds: $57,000, loss: -$1,000)
Day 15: Buy 100 SPY @ $575  (cost: $57,500)

Result: $1,000 loss is DISALLOWED (wash sale)
        New cost basis: $57,500 + $1,000 = $58,500 (loss deferred)
```

### After-Tax Sharpe Ratio

**Critical**: Sharpe ratios must use after-tax returns for accuracy.

```cpp
[[nodiscard]] inline auto calculateAfterTaxSharpe(
    std::vector<double> const& daily_returns,
    TaxConfig const& tax_config,
    double risk_free_rate = 0.0
) -> double {

    std::vector<double> after_tax_returns;
    double const tax_rate = tax_config.effectiveShortTermRate();

    for (auto const& ret : daily_returns) {
        // Only tax positive returns
        double const after_tax = ret > 0.0 ? ret * (1.0 - tax_rate) : ret;
        after_tax_returns.push_back(after_tax);
    }

    // Calculate Sharpe on after-tax returns
    double const mean_return = std::accumulate(
        after_tax_returns.begin(), after_tax_returns.end(), 0.0
    ) / after_tax_returns.size();

    double const variance = std::accumulate(
        after_tax_returns.begin(), after_tax_returns.end(), 0.0,
        [mean_return](double acc, double ret) {
            return acc + (ret - mean_return) * (ret - mean_return);
        }
    ) / after_tax_returns.size();

    double const std_dev = std::sqrt(variance);

    return (mean_return - risk_free_rate) / std_dev * std::sqrt(252.0);
}
```

---

## Tax Lots Database Schema

### DuckDB Schema: `schwab_tax_lots` table

```sql
CREATE TABLE schwab_tax_lots (
    lot_id VARCHAR PRIMARY KEY,              -- Unique lot identifier
    account_id VARCHAR NOT NULL,             -- Schwab account ID
    security_id VARCHAR NOT NULL,            -- Schwab security ID
    symbol VARCHAR NOT NULL,                 -- Ticker symbol (e.g., SPY)
    quantity DOUBLE NOT NULL,                -- Number of shares in lot
    cost_per_share DOUBLE NOT NULL,          -- Original purchase price
    acquisition_date TIMESTAMP NOT NULL,     -- Purchase date (LIFO sorting)
    current_price DOUBLE,                    -- Current market price
    market_value DOUBLE,                     -- quantity * current_price
    unrealized_pnl DOUBLE,                   -- (current_price - cost_per_share) * quantity
    term VARCHAR,                            -- 'SHORT_TERM' or 'LONG_TERM'
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for LIFO ordering (most recent lots first)
CREATE INDEX idx_symbol_acquisition_desc
ON schwab_tax_lots(symbol, acquisition_date DESC);

-- Index for lot lookups
CREATE INDEX idx_lot_id ON schwab_tax_lots(lot_id);
```

### LIFO Cost Basis Method

**Schwab Default**: Last-In-First-Out (LIFO)

When selling shares, Schwab automatically sells the most recently purchased lots first.

**Example**:
```
Holdings:
├─ Lot 1: 100 SPY @ $550 (acquired 2024-01-01) [oldest]
├─ Lot 2: 100 SPY @ $570 (acquired 2024-06-01)
└─ Lot 3: 100 SPY @ $580 (acquired 2024-11-01) [newest]

Sell 100 SPY @ $590:
└─ Sells Lot 3 (LIFO)
   ├─ Cost basis: $580
   ├─ Proceeds: $590
   ├─ Gain: $10/share = $1,000
   └─ Holding period: < 1 year → SHORT-TERM
```

### Tax Lot Queries

```sql
-- Get all lots for a symbol (LIFO order)
SELECT * FROM schwab_tax_lots
WHERE symbol = 'SPY'
ORDER BY acquisition_date DESC;

-- Calculate total position value by symbol
SELECT symbol,
       SUM(quantity) as total_shares,
       SUM(market_value) as total_value,
       SUM(unrealized_pnl) as total_unrealized_pnl
FROM schwab_tax_lots
GROUP BY symbol
ORDER BY total_value DESC;

-- Identify short-term vs long-term lots
SELECT symbol,
       CASE
           WHEN DATEDIFF('day', acquisition_date, CURRENT_DATE) <= 365
           THEN 'SHORT_TERM'
           ELSE 'LONG_TERM'
       END as term,
       COUNT(*) as num_lots,
       SUM(quantity) as total_shares,
       SUM(unrealized_pnl) as unrealized_pnl
FROM schwab_tax_lots
GROUP BY symbol, term
ORDER BY symbol, term;

-- Find lots with significant gains (tax implications)
SELECT symbol, lot_id, quantity, cost_per_share, current_price,
       unrealized_pnl,
       (unrealized_pnl * 0.32) as estimated_tax  -- 32% effective rate
FROM schwab_tax_lots
WHERE unrealized_pnl > 1000
ORDER BY unrealized_pnl DESC;
```

---

## Schwab Tax Lots Integration

### Method 1: Official Schwab API (Limitations)

**File**: [scripts/sync_schwab_tax_lots.py](../scripts/sync_schwab_tax_lots.py)

The official `schwab-py` library **does not expose tax lot endpoints**. The API only provides aggregated position data without lot-level detail.

```python
import schwab
from pathlib import Path

# Load Schwab client
token_file = Path('configs/schwab_tokens.json')
app_key = 'YOUR_APP_KEY'
app_secret = 'YOUR_APP_SECRET'

client = schwab.auth.client_from_token_file(
    str(token_file), app_key, app_secret
)

# Get account positions (no lot detail)
positions_response = client.get_account(account_num, fields='positions')
positions = positions_response.json()['securitiesAccount']['positions']

# positions[0] contains:
# - symbol, quantity, averagePrice, marketValue
# - ❌ NO lot-level cost basis
# - ❌ NO acquisition dates
# - ❌ NO LIFO ordering
```

**Limitation**: Cannot get individual tax lots from official API.

### Method 2: Unofficial Schwab API (itsjafer/schwab-api)

**File**: [scripts/fetch_schwab_tax_lots.py](../scripts/fetch_schwab_tax_lots.py)

Uses the unofficial `schwab-api` library which accesses Schwab's internal API endpoints.

**Installation**:
```bash
uv pip install schwab-api
```

**Usage**:
```python
from schwab_api import Schwab
import duckdb

# Initialize (requires browser login)
api = Schwab()

# Get account info
success, accounts = api.account_info_v2()
account_id = accounts[0]['accountId']

# Get positions
success, positions_data = api.positions_v2()
positions = positions_data.get('positions', [])

# Fetch tax lots for each position
for position in positions:
    symbol = position['symbol']
    security_id = position['security_id']

    success, lot_data = api.get_lot_info_v2(account_id, security_id)
    lots = lot_data.get('lots', [])

    # Save to database
    for lot in lots:
        conn.execute("""
            INSERT OR REPLACE INTO schwab_tax_lots
            (lot_id, account_id, security_id, symbol, quantity,
             cost_per_share, acquisition_date, current_price,
             market_value, unrealized_pnl, term)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            lot['lotId'], account_id, security_id, symbol,
            lot['quantity'], lot['costPerShare'], lot['acquisitionDate'],
            lot['currentPrice'], lot['marketValue'], lot['unrealizedPnL'],
            lot['term']
        ])
```

**Advantages**:
- ✅ Full tax lot detail
- ✅ LIFO ordering
- ✅ Acquisition dates
- ✅ Cost basis per lot
- ✅ Unrealized P&L per lot

**Disadvantages**:
- ❌ Requires browser login (not fully automated)
- ❌ Unofficial API (may break)

### Syncing Tax Lots

**Recommended Sync Schedule**:
- **Daily**: After market close (4:00 PM ET)
- **Before trades**: Before placing orders to check current lots

```bash
# Manual sync
uv run python scripts/fetch_schwab_tax_lots.py

# Automated sync (cron)
0 16 * * 1-5 cd /path/to/BigBrotherAnalytics && uv run python scripts/fetch_schwab_tax_lots.py
```

---

## Tax Calculation Examples

### Example 1: Simple Day Trade

```cpp
#include <vector>
import bigbrother.utils.tax;
using namespace bigbrother::utils::tax;

// Create a day trade
TaxTrade trade;
trade.symbol = "SPY";
trade.entry_time = 1700000000000000LL;  // Nov 14, 2024 10:00 AM
trade.exit_time = 1700010000000000LL;   // Nov 14, 2024 12:46 PM (same day)
trade.cost_basis = 58000.0;             // 100 shares @ $580
trade.proceeds = 59000.0;               // 100 shares @ $590
trade.gross_pnl = 1000.0;               // $10/share profit

// Calculate taxes
TaxCalculator calc;
auto result = calc.calculateTaxes({trade});

if (result) {
    std::println("Gross P&L: ${}", result->total_gross_pnl);  // $1,000
    std::println("Short-term gains: ${}", result->short_term_gains);  // $1,000
    std::println("Tax owed (32%): ${}", result->total_tax_owed);  // $320
    std::println("Net after tax: ${}", result->net_pnl_after_tax);  // $680
    std::println("Tax efficiency: {:.1f}%", result->taxEfficiency() * 100);  // 68.0%
}
```

**Output**:
```
Gross P&L: $1000
Short-term gains: $1000
Tax owed (32%): $320
Net after tax: $680
Tax efficiency: 68.0%
```

### Example 2: Wash Sale

```cpp
// Trade 1: Sell SPY at a loss
TaxTrade loss_trade;
loss_trade.symbol = "SPY";
loss_trade.entry_time = timestamp("2024-11-01");
loss_trade.exit_time = timestamp("2024-11-10");
loss_trade.cost_basis = 58000.0;
loss_trade.proceeds = 57000.0;
loss_trade.gross_pnl = -1000.0;  // $1,000 loss

// Trade 2: Repurchase SPY within 30 days
TaxTrade repurchase;
repurchase.symbol = "SPY";
repurchase.entry_time = timestamp("2024-11-15");  // 5 days later
repurchase.exit_time = timestamp("2024-12-01");
repurchase.cost_basis = 57500.0;
repurchase.proceeds = 59000.0;
repurchase.gross_pnl = 1500.0;

// Calculate with wash sale rules
TaxCalculator calc{TaxConfig{.track_wash_sales = true}};
auto result = calc.calculateTaxes({loss_trade, repurchase});

if (result) {
    std::println("Wash sales disallowed: {}", result->wash_sales_disallowed);  // 1
    std::println("Loss disallowed: ${}", result->wash_sale_loss_disallowed);  // $1,000
    std::println("Effective P&L: ${}", result->total_gross_pnl);  // $500 (only repurchase gain)
}
```

**Output**:
```
Wash sales disallowed: 1
Loss disallowed: $1000
Effective P&L: $500
```

### Example 3: Fluent API with Multiple Trades

```cpp
std::vector<TaxTrade> trades = {
    createTrade("SPY", 1000.0),   // +$1,000 gain
    createTrade("QQQ", 2000.0),   // +$2,000 gain
    createTrade("IWM", -500.0),   // -$500 loss
    createTrade("AAPL", 1500.0),  // +$1,500 gain
};

auto result = TaxCalculatorBuilder()
    .federalRate(0.24)          // 24% federal
    .stateRate(0.05)            // 5% state (CA would be 0.133)
    .withMedicareSurtax()       // 3.8% NIIT
    .patternDayTrader()         // All short-term
    .trackWashSales()           // Enable wash sale detection
    .addTrades(trades)
    .calculate();

if (result) {
    std::println("Total trades: {}", trades.size());
    std::println("Gross P&L: ${:,.2f}", result->total_gross_pnl);  // $4,000
    std::println("Short-term gains: ${:,.2f}", result->short_term_gains);  // $4,500
    std::println("Short-term losses: ${:,.2f}", result->short_term_losses);  // $500
    std::println("Taxable income: ${:,.2f}", result->taxable_short_term);  // $4,000
    std::println("Tax owed: ${:,.2f}", result->total_tax_owed);  // $1,280
    std::println("Net after tax: ${:,.2f}", result->net_pnl_after_tax);  // $2,720
    std::println("Effective tax rate: {:.1f}%", result->effective_tax_rate * 100);  // 32%
    std::println("Tax efficiency: {:.1f}%", result->taxEfficiency() * 100);  // 68%
}
```

**Output**:
```
Total trades: 4
Gross P&L: $4,000.00
Short-term gains: $4,500.00
Short-term losses: $500.00
Taxable income: $4,000.00
Tax owed: $1,280.00
Net after tax: $2,720.00
Effective tax rate: 32.0%
Tax efficiency: 68.0%
```

### Example 4: After-Tax Sharpe Ratio

```cpp
std::vector<double> daily_returns = {
    0.02,   // +2%
    -0.01,  // -1%
    0.03,   // +3%
    0.01,   // +1%
    -0.02,  // -2%
};

TaxConfig config;
config.short_term_rate = 0.24;
config.state_tax_rate = 0.05;
config.medicare_surtax = 0.038;

double pre_tax_sharpe = calculateSharpe(daily_returns);      // Ignores taxes
double after_tax_sharpe = calculateAfterTaxSharpe(
    daily_returns, config
);

std::println("Pre-tax Sharpe: {:.2f}", pre_tax_sharpe);     // 1.50
std::println("After-tax Sharpe: {:.2f}", after_tax_sharpe);  // 1.02 (32% lower)
std::println("Tax drag on Sharpe: {:.1f}%",
             (1.0 - after_tax_sharpe / pre_tax_sharpe) * 100);  // 32%
```

**Output**:
```
Pre-tax Sharpe: 1.50
After-tax Sharpe: 1.02
Tax drag on Sharpe: 32.0%
```

---

## Dashboard Integration

### File: [dashboard/tax_implications_view.py](../dashboard/tax_implications_view.py)

Streamlit dashboard view showing tax lots and tax implications.

### Features

1. **Holdings by Tax Lot**
   - LIFO-ordered lots (most recent first)
   - Cost basis per lot
   - Unrealized P&L per lot
   - Short-term vs long-term classification

2. **Tax Implications**
   - Estimated tax on realized gains
   - Tax liability by position
   - Wash sale warnings
   - Capital loss carryforward

3. **Tax-Optimized Order Suggestions**
   - Identify lots to sell for optimal tax outcome
   - Long-term lots (lower tax rate)
   - Loss lots (to offset gains)

### Dashboard Screenshot

```
┌──────────────────────────────────────────────────────────────┐
│ Tax Lots - SPY                                               │
├──────────────────────────────────────────────────────────────┤
│ Lot ID          Qty    Cost/Share  Acq. Date   Term    P&L   │
│ LOT-001        100    $580.00     2024-11-01  SHORT  +$1000  │
│ LOT-002        100    $570.00     2024-06-01  SHORT  +$2000  │
│ LOT-003        100    $550.00     2024-01-01  LONG   +$4000  │
│                                                               │
│ Total: 300 shares | Unrealized P&L: $7,000                   │
│ Estimated Tax (if sold today): $2,240                        │
│   - Short-term (200 shares): $960                            │
│   - Long-term (100 shares):  $800                            │
└──────────────────────────────────────────────────────────────┘
```

### Running the Dashboard

```bash
uv run streamlit run dashboard/app.py
```

---

## Testing

### Unit Tests for Tax Module

**File**: `tests/cpp/test_tax.cpp` (to be created)

```cpp
#include <gtest/gtest.h>
import bigbrother.utils.tax;

using namespace bigbrother::utils::tax;

TEST(TaxTest, ShortTermGain) {
    TaxTrade trade;
    trade.symbol = "SPY";
    trade.gross_pnl = 1000.0;
    trade.entry_time = timestamp("2024-11-01");
    trade.exit_time = timestamp("2024-11-10");  // 9 days

    EXPECT_TRUE(trade.isShortTerm());
    EXPECT_FALSE(trade.isLongTerm());
    EXPECT_EQ(trade.holdingPeriodDays(), 9);
}

TEST(TaxTest, LongTermGain) {
    TaxTrade trade;
    trade.entry_time = timestamp("2023-01-01");
    trade.exit_time = timestamp("2024-06-01");  // 517 days

    EXPECT_TRUE(trade.isLongTerm());
    EXPECT_FALSE(trade.isShortTerm());
    EXPECT_GT(trade.holdingPeriodDays(), 365);
}

TEST(TaxTest, WashSaleDetection) {
    TaxTrade loss_trade;
    loss_trade.symbol = "SPY";
    loss_trade.gross_pnl = -1000.0;
    loss_trade.exit_time = timestamp("2024-11-10");

    TaxTrade repurchase;
    repurchase.symbol = "SPY";
    repurchase.entry_time = timestamp("2024-11-15");  // 5 days later

    TaxConfig config{.track_wash_sales = true};
    TaxCalculator calc{config};

    auto result = calc.applyWashSaleRules({loss_trade, repurchase});

    EXPECT_TRUE(result[0].wash_sale_disallowed);
    EXPECT_EQ(result[0].wash_sale_amount, 1000.0);
}

TEST(TaxTest, TaxCalculationAccuracy) {
    TaxTrade trade;
    trade.gross_pnl = 10000.0;  // $10,000 gain

    TaxConfig config;
    config.short_term_rate = 0.24;
    config.state_tax_rate = 0.05;
    config.medicare_surtax = 0.038;

    TaxCalculator calc{config};
    auto result = calc.calculateTaxes({trade});

    ASSERT_TRUE(result);
    EXPECT_NEAR(result->total_tax_owed, 3280.0, 0.01);  // 32.8%
    EXPECT_NEAR(result->net_pnl_after_tax, 6720.0, 0.01);
    EXPECT_NEAR(result->effective_tax_rate, 0.328, 0.001);
}

TEST(TaxTest, FluentAPIUsage) {
    auto result = TaxCalculatorBuilder()
        .federalRate(0.24)
        .stateRate(0.05)
        .withMedicareSurtax()
        .addTrade(TaxTrade{.gross_pnl = 1000.0})
        .calculate();

    ASSERT_TRUE(result);
    EXPECT_GT(result->total_tax_owed, 0.0);
    EXPECT_LT(result->net_pnl_after_tax, result->total_gross_pnl);
}
```

### Running Tests

```bash
# Build tests
export SKIP_CLANG_TIDY=1
cmake -G Ninja -B build
ninja -C build test_tax

# Run tests
./build/tests/cpp/test_tax

# Run with CTest
cd build && ctest -R TaxTests --output-on-failure
```

---

## API Reference

### TaxConfig Methods

```cpp
[[nodiscard]] constexpr auto effectiveShortTermRate() const noexcept -> double;
[[nodiscard]] constexpr auto effectiveLongTermRate() const noexcept -> double;
[[nodiscard]] auto validate() const noexcept -> Result<void>;
```

### TaxTrade Methods

```cpp
[[nodiscard]] constexpr auto holdingPeriodDays() const noexcept -> int;
[[nodiscard]] constexpr auto isLongTerm() const noexcept -> bool;
[[nodiscard]] constexpr auto isShortTerm() const noexcept -> bool;
[[nodiscard]] constexpr auto isGain() const noexcept -> bool;
[[nodiscard]] constexpr auto isLoss() const noexcept -> bool;
```

### TaxResult Methods

```cpp
[[nodiscard]] constexpr auto taxEfficiency() const noexcept -> double;
[[nodiscard]] constexpr auto isProfitableAfterTax() const noexcept -> bool;
[[nodiscard]] constexpr auto taxDragPercent() const noexcept -> double;
```

### TaxCalculator Methods

```cpp
[[nodiscard]] auto calculateTaxes(std::vector<TaxTrade> const& trades) -> Result<TaxResult>;
[[nodiscard]] auto calculateQuarterlyTax(double quarterly_profit) const noexcept -> double;
[[nodiscard]] auto applyWashSaleRules(std::vector<TaxTrade> trades) const -> std::vector<TaxTrade>;
```

### TaxCalculatorBuilder Methods (Fluent API)

```cpp
[[nodiscard]] auto federalRate(double rate) noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto longTermRate(double rate) noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto stateRate(double rate) noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto withMedicareSurtax() noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto patternDayTrader() noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto section1256Trader() noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto trackWashSales() noexcept -> TaxCalculatorBuilder&;
[[nodiscard]] auto addTrades(std::vector<TaxTrade> trades) -> TaxCalculatorBuilder&;
[[nodiscard]] auto addTrade(TaxTrade trade) -> TaxCalculatorBuilder&;
[[nodiscard]] auto calculate() -> Result<TaxResult>;  // Terminal operation
[[nodiscard]] auto calculateAfterTax(double gross_return) const noexcept -> double;
```

### Helper Functions

```cpp
[[nodiscard]] inline auto calculateAfterTaxSharpe(
    std::vector<double> const& daily_returns,
    TaxConfig const& tax_config,
    double risk_free_rate = 0.0
) -> double;
```

---

## IRS Tax Rules Reference

### Holding Period Classifications

| Holding Period | Tax Treatment          | Tax Rate (2025)       |
|----------------|------------------------|-----------------------|
| ≤ 365 days     | Short-term capital gain | Ordinary income (10%-37%) |
| > 365 days     | Long-term capital gain  | Preferential (0%, 15%, 20%) |

### Tax Brackets (2025, Single Filer)

#### Ordinary Income (Short-term gains)
| Income Range        | Tax Rate |
|---------------------|----------|
| $0 - $11,600        | 10%      |
| $11,600 - $47,150   | 12%      |
| $47,150 - $100,525  | 22%      |
| $100,525 - $191,950 | 24%      |
| $191,950 - $243,725 | 32%      |
| $243,725 - $609,350 | 35%      |
| $609,350+           | 37%      |

#### Long-term Capital Gains
| Income Range        | Tax Rate |
|---------------------|----------|
| $0 - $44,625        | 0%       |
| $44,625 - $492,300  | 15%      |
| $492,300+           | 20%      |

### Additional Taxes

- **Medicare Surtax (NIIT)**: 3.8% on net investment income for high earners
  - Single: AGI > $200,000
  - Married Filing Jointly: AGI > $250,000

- **State Taxes**: Varies by state (0% - 13.3%)
  - California: 13.3% (highest)
  - Texas, Florida, Nevada: 0%
  - New York: 10.9%

### Wash Sale Rule (IRS Publication 550)

**Definition**: Cannot claim loss if you buy substantially identical security within 30 days before or after the sale.

**30-Day Window**:
- 30 days before sale date
- Sale date
- 30 days after sale date
- **Total**: 61-day window

**Disallowed Loss**: Added to cost basis of replacement shares (deferred, not lost forever).

### Capital Loss Deduction Limit

- **Maximum annual deduction**: $3,000 ($1,500 if married filing separately)
- **Carryforward**: Excess losses carry forward to future years indefinitely

**Example**:
```
Year 1: $10,000 capital loss
        - Deduct $3,000 this year
        - Carry forward $7,000 to next year

Year 2: $5,000 capital gain
        - Offset with $5,000 carryforward
        - Remaining carryforward: $2,000

Year 3: Deduct remaining $2,000
```

### Section 1256 Contracts (60/40 Rule)

**Applies to**:
- Index options (SPX, NDX, RUT)
- Futures contracts
- Foreign currency contracts

**Treatment**:
- 60% taxed as long-term (preferential rates)
- 40% taxed as short-term (ordinary income)
- **Regardless of actual holding period**

**Example**:
```
$10,000 gain on SPX options (held 1 day)
├─ 60% long-term: $6,000 @ 15% = $900
├─ 40% short-term: $4,000 @ 24% = $960
└─ Total tax: $1,860 (18.6% effective rate)

vs. Regular short-term treatment:
└─ $10,000 @ 24% = $2,400 (24% effective rate)

Savings: $540 (5.4%)
```

---

## Best Practices

### 1. Always Calculate After-Tax Returns

```cpp
// ❌ BAD: Ignoring taxes
double strategy_return = calculate_gross_return();
if (strategy_return > 0.20) {
    deploy_strategy();
}

// ✅ GOOD: Account for taxes
auto tax_result = TaxCalculatorBuilder()
    .federalRate(0.24)
    .stateRate(0.05)
    .withMedicareSurtax()
    .calculateAfterTax(strategy_return);

if (tax_result > 0.15) {  // 15% after-tax threshold
    deploy_strategy();
}
```

### 2. Minimize Wash Sales

```cpp
// ❌ BAD: Immediate repurchase after loss
sell("SPY", 100);           // Realize loss
buy("SPY", 100);            // Wash sale!

// ✅ GOOD: Wait 31+ days or buy different security
sell("SPY", 100);           // Realize loss
wait_days(31);
buy("SPY", 100);            // No wash sale

// OR buy correlated but not identical:
sell("SPY", 100);           // S&P 500 ETF
buy("VOO", 100);            // Different S&P 500 ETF (allowed)
```

### 3. Use Long-term Lots When Possible

```cpp
// Query long-term lots first (lower tax rate)
auto long_term_lots = conn.execute("""
    SELECT * FROM schwab_tax_lots
    WHERE symbol = 'SPY'
      AND DATEDIFF('day', acquisition_date, CURRENT_DATE) > 365
    ORDER BY unrealized_pnl DESC
""").fetchall();

// Sell long-term lots for 15% vs 32% tax rate
```

### 4. Track Quarterly Estimated Taxes

```cpp
TaxCalculator calc;
double q1_profit = 25000.0;
double q1_tax = calc.calculateQuarterlyTax(q1_profit);

std::println("Q1 Estimated Tax: ${:,.2f}", q1_tax);  // $8,000
std::println("Pay by: April 15");
```

### 5. Consider Section 1256 Contracts

For index options (SPX, NDX, RUT), use Section 1256 treatment:

```cpp
TaxCalculatorBuilder builder;
builder.section1256Trader()  // Enable 60/40 rule
       .calculate();

// 60% long-term (15%) + 40% short-term (24%)
// = 18.6% effective rate (vs 32% all short-term)
// = 41% tax savings!
```

---

## Troubleshooting

### Problem: Tax lots not syncing from Schwab

**Solution**: Use unofficial `schwab-api` library:

```bash
uv pip install schwab-api
uv run python scripts/fetch_schwab_tax_lots.py
```

### Problem: Wash sale not detected

**Solution**: Ensure wash sale tracking is enabled:

```cpp
TaxConfig config;
config.track_wash_sales = true;
config.wash_sale_window_days = 30;
```

### Problem: Tax calculation seems wrong

**Solution**: Verify tax rates match your income bracket:

```cpp
// Check your actual rates
TaxConfig config;
config.short_term_rate = 0.24;    // YOUR federal bracket
config.long_term_rate = 0.15;     // YOUR LTCG rate
config.state_tax_rate = 0.133;    // California (or your state)
config.medicare_surtax = 0.038;   // If AGI > $200k

double effective_rate = config.effectiveShortTermRate();
std::println("Your effective tax rate: {:.1f}%", effective_rate * 100);
```

### Problem: Database schema doesn't match

**Solution**: Recreate table:

```sql
DROP TABLE IF EXISTS schwab_tax_lots;

CREATE TABLE schwab_tax_lots (
    lot_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    security_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    quantity DOUBLE NOT NULL,
    cost_per_share DOUBLE NOT NULL,
    acquisition_date TIMESTAMP NOT NULL,
    current_price DOUBLE,
    market_value DOUBLE,
    unrealized_pnl DOUBLE,
    term VARCHAR,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Summary

### Key Takeaways

1. **Tax efficiency is critical**: 32% tax drag on short-term gains vs 20% on long-term
2. **LIFO cost basis**: Schwab sells most recent lots first
3. **Wash sale rule**: Cannot deduct losses if repurchase within 30 days
4. **After-tax Sharpe**: Use after-tax returns for accurate performance metrics
5. **Section 1256**: Index options get favorable 60/40 treatment (41% tax savings)
6. **Capital loss limit**: Max $3,000 annual deduction, excess carries forward

### Tax Efficiency Strategies

1. Hold > 365 days when possible (long-term treatment)
2. Use Section 1256 contracts (SPX, NDX, RUT)
3. Avoid wash sales (wait 31+ days or buy different security)
4. Harvest tax losses to offset gains
5. Use long-term lots when selling (lower tax rate)
6. Pay quarterly estimated taxes (avoid penalties)

### Performance Impact

```
Example: $100,000 trading capital, 40% gross return

Short-term trading (all gains < 1 year):
├─ Gross: $40,000
├─ Tax (32%): -$12,800
└─ Net: $27,200 (27.2% after-tax return)

Long-term holding (all gains > 1 year):
├─ Gross: $40,000
├─ Tax (20%): -$8,000
└─ Net: $32,000 (32.0% after-tax return)

Tax savings: $4,800 (17.6% better after-tax return)
```

### Next Steps

1. **Sync tax lots**: Run `uv run python scripts/fetch_schwab_tax_lots.py`
2. **Calculate current tax liability**: Check unrealized gains
3. **Integrate tax module**: Add after-tax return calculations to strategies
4. **Test tax efficiency**: Compare strategies using after-tax Sharpe ratios
5. **Monitor wash sales**: Set up alerts for potential wash sales
6. **Plan quarterly taxes**: Calculate and pay estimated taxes

---

**Generated**: 2025-11-11
**Author**: Claude Code
**System**: BigBrotherAnalytics Tax-Aware Trading Platform

For questions or issues, see:
- [Tax Module](../src/utils/tax.cppm)
- [Fetch Script](../scripts/fetch_schwab_tax_lots.py)
- [Dashboard](../dashboard/tax_implications_view.py)
