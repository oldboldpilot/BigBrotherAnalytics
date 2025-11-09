# Fluent Risk Manager API

## Overview

The RiskManager now implements a comprehensive fluent API pattern following the Schwab API design philosophy. This enables clean, chainable method calls for risk configuration, assessment, and portfolio analysis.

```cpp
// Modern fluent style
auto risk_mgr = RiskManager()
    .setMaxDailyLoss(900.0)
    .setMaxPositionSize(1500.0)
    .setMaxPortfolioHeat(0.15);
```

## Table of Contents

1. [Fluent Configuration](#fluent-configuration)
2. [Trade Risk Assessment](#trade-risk-assessment)
3. [Portfolio Analysis](#portfolio-analysis)
4. [Specialized Calculators](#specialized-calculators)
5. [Daily P&L Management](#daily-pnl-management)
6. [Complete Examples](#complete-examples)

---

## Fluent Configuration

### Quick Setup

Configure RiskManager with method chaining:

```cpp
RiskManager risk_mgr;

risk_mgr.setMaxDailyLoss(1000.0)
    .setMaxPositionSize(2000.0)
    .setMaxPortfolioHeat(0.20)
    .setMaxConcurrentPositions(15)
    .setAccountValue(50000.0)
    .requireStopLoss(true);
```

### Configuration Methods

All configuration methods return `RiskManager&` for chaining:

| Method | Parameter | Description |
|--------|-----------|-------------|
| `setMaxDailyLoss(double)` | Amount in $ | Maximum daily loss allowed |
| `setMaxPositionSize(double)` | Amount in $ | Maximum per position |
| `setMaxPortfolioHeat(double)` | Percentage [0-1] | Maximum portfolio risk exposure |
| `setMaxConcurrentPositions(int)` | Count | Maximum open positions |
| `setAccountValue(double)` | Amount in $ | Total account value |
| `requireStopLoss(bool)` | true/false | Enforce stop losses |
| `withLimits(RiskLimits)` | RiskLimits struct | Set all limits at once |

### Example: Progressive Configuration

```cpp
RiskManager risk_mgr(RiskLimits::forThirtyKAccount());

// Start conservative
risk_mgr.setMaxDailyLoss(500.0)
    .setMaxPositionSize(800.0);

// Later, adjust based on performance
if (portfolio_performance > 10000.0) {
    risk_mgr.setMaxDailyLoss(1500.0)
        .setMaxPositionSize(2500.0);
}
```

---

## Trade Risk Assessment

### TradeRiskBuilder Pattern

Assess individual trades using fluent builder pattern:

```cpp
auto risk = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .withQuantity(10)
    .atPrice(450.00)
    .withStop(440.00)
    .withTarget(465.00)
    .withProbability(0.65)
    .assess();

if (risk.has_value() && risk->approved) {
    execute_trade(risk->symbol, risk->position_size);
}
```

### Builder Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `forSymbol(string)` | Symbol | Ticker symbol |
| `withQuantity(int)` | Number of shares | Position quantity |
| `atPrice(double)` | Price in $ | Entry price |
| `withStop(double)` | Price in $ | Stop loss price |
| `withTarget(double)` | Price in $ | Profit target |
| `withProbability(double)` | [0-1] | Win probability estimate |
| `.assess()` | Terminal | Execute assessment, returns Result<TradeRisk> |

### Trade Risk Result

```cpp
struct TradeRisk {
    double position_size;           // Approved position size
    double max_loss;                // Maximum loss in dollars
    double expected_return;         // Expected return in dollars
    double expected_value;          // EV = (P*W) - ((1-P)*L)
    double win_probability;         // Win rate
    double kelly_fraction;          // Kelly criterion fraction
    double risk_reward_ratio;       // Expected return / max loss
    bool approved;                  // Trade approved?
    std::string rejection_reason;   // Why rejected
};
```

### Example: Multi-Trade Assessment

```cpp
std::vector<std::string> symbols = {"SPY", "QQQ", "XLE"};
std::vector<double> prices = {450.0, 350.0, 80.0};

for (size_t i = 0; i < symbols.size(); ++i) {
    auto risk = risk_mgr.assessTrade()
        .forSymbol(symbols[i])
        .withQuantity(10)
        .atPrice(prices[i])
        .withStop(prices[i] * 0.98)      // 2% stop
        .withTarget(prices[i] * 1.03)    // 3% target
        .withProbability(0.60)
        .assess();

    if (risk && risk->approved) {
        // Execute trade
    }
}
```

### Example: Conservative Risk Management

```cpp
auto result = risk_mgr.assessTrade()
    .forSymbol("TSLA")
    .withQuantity(5)
    .atPrice(250.00)
    .withStop(240.00)           // Tight stop: 4%
    .withTarget(275.00)         // Target: 10%
    .withProbability(0.55)      // Conservative estimate
    .assess();

// Check risk metrics
if (result) {
    std::cout << "Risk: $" << result->max_loss << "\n";
    std::cout << "Reward: $" << result->expected_return << "\n";
    std::cout << "R:R Ratio: " << result->risk_reward_ratio << "\n";
    std::cout << "Kelly Fraction: " << result->kelly_fraction << "\n";
}
```

---

## Portfolio Analysis

### PortfolioRiskBuilder Pattern

Analyze complete portfolio exposure:

```cpp
auto portfolio = risk_mgr.portfolio()
    .addPosition("SPY", 10, 450.00, 0.05)
    .addPosition("XLE", 20, 80.00, 0.08)
    .addPosition("TLT", 50, 100.00, 0.03)
    .calculateHeat()
    .calculateVaR(0.95)
    .analyze();

if (portfolio) {
    std::cout << "Total Value: $" << portfolio->total_value << "\n";
    std::cout << "Portfolio Heat: " << portfolio->portfolio_heat << "\n";
    std::cout << "Active Positions: " << portfolio->active_positions << "\n";
}
```

### Builder Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `addPosition(symbol, qty, price, vol?)` | String, double, double, optional | Add position to analysis |
| `calculateHeat()` | None | Calculate portfolio heat (exposure/account) |
| `calculateVaR(confidence)` | double [0-1] | Calculate Value at Risk |
| `analyze()` | Terminal | Execute analysis, returns Result<PortfolioRisk> |

### Portfolio Risk Result

```cpp
struct PortfolioRisk {
    double total_value;             // Sum of all positions
    double total_exposure;          // Market exposure
    double daily_pnl;               // Unrealized P&L
    double daily_loss_remaining;    // Available loss budget
    int active_positions;           // Number of open positions
    double portfolio_heat;          // Exposure / Account value
    double max_drawdown;            // Historical max drawdown
    std::string getRiskLevel();     // "HIGH", "MEDIUM", "LOW"
};
```

### Example: Sector Analysis

```cpp
// Analyze technology sector exposure
auto tech_portfolio = risk_mgr.portfolio()
    .addPosition("AAPL", 20, 175.00, 0.25)
    .addPosition("MSFT", 15, 380.00, 0.22)
    .addPosition("NVDA", 10, 800.00, 0.35)
    .calculateHeat()
    .analyze();

if (tech_portfolio) {
    if (tech_portfolio->portfolio_heat > 0.30) {
        // Reduce tech exposure
    }
}
```

### Example: Multi-Asset Portfolio

```cpp
auto full_portfolio = risk_mgr.portfolio()
    // Equities
    .addPosition("SPY", 20, 450.00, 0.08)
    .addPosition("QQQ", 15, 350.00, 0.15)

    // Commodities
    .addPosition("XLE", 50, 80.00, 0.10)
    .addPosition("GLD", 30, 200.00, 0.04)

    // Fixed Income
    .addPosition("TLT", 100, 100.00, 0.03)

    .calculateHeat()
    .calculateVaR(0.95)
    .analyze();

if (full_portfolio) {
    std::cout << "Risk Level: " << full_portfolio->getRiskLevel() << "\n";
}
```

---

## Specialized Calculators

### Kelly Criterion Calculator

Optimal position sizing using Kelly formula:

```cpp
auto kelly_fraction = risk_mgr.kelly()
    .withWinRate(0.55)              // 55% win rate
    .withWinLossRatio(1.8)          // Win $1.80 per $1 lost
    .withDrawdownLimit(0.25)        // Optional: max drawdown
    .calculate();

if (kelly_fraction) {
    std::cout << "Kelly Fraction: " << *kelly_fraction << "\n";
    // Use this to size position
}
```

#### Kelly Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `withWinRate(double)` | [0-1] | Historical win rate |
| `withWinLossRatio(double)` | > 0 | Avg win / Avg loss |
| `withDrawdownLimit(double)` | [0-1] | Risk constraint |
| `calculate()` | Terminal | Compute Kelly fraction |

#### Kelly Formula

```
f* = (p * b - q) / b

where:
  p = win probability
  q = 1 - p (loss probability)
  b = win/loss ratio
  f* = fractional Kelly
```

#### Example: Conservative Sizing

```cpp
// 60% win rate, 2:1 reward:risk
auto kelly = risk_mgr.kelly()
    .withWinRate(0.60)
    .withWinLossRatio(2.0)
    .calculate();

if (kelly) {
    // Use half-Kelly for safety
    double position_fraction = *kelly * 0.5;
    double position_size = account_value * position_fraction;
}
```

### Position Sizer Builder

Calculate optimal position sizes using various methods:

```cpp
auto size = risk_mgr.positionSizer()
    .withMethod(SizingMethod::KellyCriterion)
    .withAccountValue(30000.0)
    .withWinProbability(0.60)
    .withWinAmount(100.0)
    .withLossAmount(80.0)
    .calculate();

if (size) {
    std::cout << "Position Size: $" << *size << "\n";
}
```

#### Sizing Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `FixedDollar` | Fixed $1000 per trade | Baseline trades |
| `FixedPercent` | 5% of account | Conservative approach |
| `KellyCriterion` | Full Kelly formula | Optimal (aggressive) |
| `KellyHalf` | 50% of Kelly | Conservative Kelly |
| `VolatilityAdjusted` | Inverse volatility weighting | Vol-sensitive |
| `RiskParity` | Equal risk per position | Balanced portfolio |

#### Example: Dynamic Sizing

```cpp
// Size position based on win rate
auto size = risk_mgr.positionSizer()
    .withMethod(win_rate > 0.60 ? SizingMethod::KellyHalf
                                : SizingMethod::FixedPercent)
    .withAccountValue(account_value)
    .withWinProbability(win_rate)
    .withWinAmount(expected_win)
    .withLossAmount(max_loss)
    .calculate();
```

### Monte Carlo Simulator Builder

Run probabilistic simulations on option positions:

```cpp
PricingParams params{
    .spot_price = 450.0,
    .strike_price = 450.0,
    .time_to_expiration = 0.25,
    .risk_free_rate = 0.05,
    .volatility = 0.20,
    .option_type = OptionType::Call
};

auto simulation = risk_mgr.monteCarlo()
    .forOption(params)
    .withSimulations(10000)
    .withSteps(100)
    .withPositionSize(10.0)
    .run();

if (simulation) {
    std::cout << "Expected Value: $" << simulation->expected_value << "\n";
    std::cout << "Probability of Profit: "
              << simulation->probability_of_profit * 100 << "%\n";
    std::cout << "Max Loss (95% VaR): $" << simulation->var_95 << "\n";
}
```

#### Simulation Result

```cpp
struct SimulationResult {
    double expected_value;          // Mean P&L
    double std_deviation;           // Standard deviation
    double probability_of_profit;   // Prob(PnL > 0)
    double var_95;                  // 95% Value at Risk
    double var_99;                  // 99% Value at Risk
    double max_profit;              // Maximum possible profit
    double max_loss;                // Maximum possible loss
    std::vector<double> pnl_distribution;  // All outcomes
};
```

#### Example: Options Strategy Analysis

```cpp
auto call_risk = risk_mgr.monteCarlo()
    .forOption(call_params)
    .withSimulations(50000)
    .withPositionSize(100.0)
    .run();

auto put_risk = risk_mgr.monteCarlo()
    .forOption(put_params)
    .withSimulations(50000)
    .withPositionSize(100.0)
    .run();

if (call_risk && put_risk) {
    // Compare straddle risk vs individual legs
}
```

---

## Daily P&L Management

### P&L Tracking

Track and manage daily profit/loss:

```cpp
// Update P&L throughout the day
risk_mgr.updateDailyPnL(500.0);    // Win
risk_mgr.updateDailyPnL(-200.0);   // Loss
risk_mgr.updateDailyPnL(300.0);    // Win

double daily_pnl = risk_mgr.getDailyPnL();           // +600
double loss_remaining = risk_mgr.getDailyLossRemaining();
bool limit_reached = risk_mgr.isDailyLossLimitReached();
```

### P&L Methods

| Method | Return | Description |
|--------|--------|-------------|
| `updateDailyPnL(double)` | RiskManager& | Add P&L to daily total |
| `getDailyPnL()` | double | Current daily P&L |
| `getDailyLossRemaining()` | double | Remaining loss budget |
| `isDailyLossLimitReached()` | bool | Hit daily limit? |
| `resetDaily()` | RiskManager& | Reset for next trading day |

### Example: End-of-Day Reset

```cpp
// At market close
double final_pnl = risk_mgr.getDailyPnL();
if (final_pnl < 0) {
    std::cout << "Daily loss: $" << final_pnl << "\n";
}

// Reset for tomorrow
risk_mgr.resetDaily();
```

---

## Complete Examples

### Example 1: Simple Trade Workflow

```cpp
#include <iostream>
import bigbrother.risk_management;

using namespace bigbrother::risk;

int main() {
    // Create risk manager
    RiskManager risk_mgr;

    // Configure for the day
    risk_mgr.setMaxDailyLoss(900.0)
        .setMaxPositionSize(1500.0)
        .setMaxPortfolioHeat(0.15);

    // Evaluate SPY trade
    auto risk = risk_mgr.assessTrade()
        .forSymbol("SPY")
        .withQuantity(10)
        .atPrice(450.00)
        .withStop(440.00)
        .withTarget(465.00)
        .withProbability(0.65)
        .assess();

    if (!risk) {
        std::cerr << "Assessment failed: " << risk.error().message << "\n";
        return 1;
    }

    if (risk->approved) {
        std::cout << "Trade approved!\n";
        std::cout << "Position size: $" << risk->position_size << "\n";
        std::cout << "Max loss: $" << risk->max_loss << "\n";
        std::cout << "Expected value: $" << risk->expected_value << "\n";
        // Execute trade...
    } else {
        std::cout << "Trade rejected: " << risk->rejection_reason << "\n";
    }

    return 0;
}
```

### Example 2: Portfolio Heat Management

```cpp
// Monitor portfolio heat throughout the day
RiskManager risk_mgr;

// Open positions
risk_mgr.portfolio()
    .addPosition("SPY", 10, 450.00)
    .calculateHeat()
    .analyze()
    .and_then([](auto portfolio) {
        if (portfolio.portfolio_heat > 0.20) {
            std::cout << "Portfolio heat too high, consider closing positions\n";
        }
        return portfolio;
    });
```

### Example 3: Kelly Criterion Position Sizing

```cpp
// Strategy: 60% win rate, 2:1 reward:risk
auto kelly = risk_mgr.kelly()
    .withWinRate(0.60)
    .withWinLossRatio(2.0)
    .calculate();

auto size = risk_mgr.positionSizer()
    .withMethod(SizingMethod::KellyHalf)
    .withAccountValue(30000.0)
    .withWinProbability(0.60)
    .withWinAmount(200.0)
    .withLossAmount(100.0)
    .calculate();

if (kelly && size) {
    std::cout << "Kelly fraction: " << *kelly << " (half-Kelly used)\n";
    std::cout << "Position size: $" << *size << "\n";
}
```

### Example 4: Options Risk Simulation

```cpp
// Analyze covered call strategy
PricingParams call_params{
    .spot_price = 450.0,
    .strike_price = 460.0,      // Out of money
    .time_to_expiration = 0.25,
    .risk_free_rate = 0.05,
    .volatility = 0.20,
    .option_type = OptionType::Call
};

auto sim = risk_mgr.monteCarlo()
    .forOption(call_params)
    .withSimulations(10000)
    .withPositionSize(100.0)    // 100 shares
    .run();

if (sim) {
    std::cout << "Probability of profit: "
              << sim->probability_of_profit * 100 << "%\n";
    std::cout << "Expected P&L: $" << sim->expected_value << "\n";

    // Assess if strategy meets criteria
    if (sim->probability_of_profit > 0.70) {
        // Execute covered call
    }
}
```

### Example 5: Multiple Trade Scenario

```cpp
// Backtest scenario: assess multiple trades
std::vector<struct TradeSetup {
    std::string symbol;
    int quantity;
    double price, stop, target;
    double win_prob;
}> trades = {
    {"SPY", 10, 450.0, 440.0, 465.0, 0.65},
    {"QQQ", 15, 350.0, 340.0, 370.0, 0.60},
    {"XLE", 20, 80.0, 75.0, 90.0, 0.55}
};

for (auto const& trade : trades) {
    auto result = risk_mgr.assessTrade()
        .forSymbol(trade.symbol)
        .withQuantity(trade.quantity)
        .atPrice(trade.price)
        .withStop(trade.stop)
        .withTarget(trade.target)
        .withProbability(trade.win_prob)
        .assess();

    if (result && result->approved) {
        std::cout << trade.symbol << ": APPROVED\n";
    } else {
        std::cout << trade.symbol << ": REJECTED\n";
    }
}

// Analyze combined portfolio
auto portfolio = risk_mgr.portfolio()
    .addPosition("SPY", 10, 450.0)
    .addPosition("QQQ", 15, 350.0)
    .addPosition("XLE", 20, 80.0)
    .calculateHeat()
    .analyze();

if (portfolio) {
    std::cout << "\nPortfolio heat: " << portfolio->portfolio_heat << "\n";
    std::cout << "Risk level: " << portfolio->getRiskLevel() << "\n";
}
```

---

## Design Principles

### 1. Fluent Interface (Method Chaining)

```cpp
// Read like natural language
risk_mgr.setMaxDailyLoss(1000)
    .setMaxPositionSize(2000)
    .setMaxConcurrentPositions(10);
```

### 2. Terminal Operations

```cpp
// Terminal operations return Result<T>
auto result = builder.method1().method2().assess();  // Terminal
if (result) { /* use value */ }
```

### 3. Thread Safety

All operations are thread-safe with mutex protection for shared state.

### 4. C++23 Modern Features

- Trailing return syntax: `auto method() -> ReturnType&`
- `[[nodiscard]]` attributes for important operations
- `std::expected` for error handling

### 5. Backward Compatibility

Existing methods continue to work:

```cpp
// Old style still works
auto risk = risk_mgr.assessTrade("SPY", 1000, 450, 440, 465, 0.65);

// New fluent style
auto risk2 = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .withQuantity(10)
    .atPrice(450)
    // ...
    .assess();
```

---

## Performance Considerations

- **Configuration**: O(1) - instant
- **Trade Assessment**: O(1) - constant time
- **Portfolio Analysis**: O(n) - linear in number of positions
- **Kelly Calculation**: O(1) - simple arithmetic
- **Position Sizer**: O(1) - lookup and calculation
- **Monte Carlo**: O(simulations × steps) - parallelized with OpenMP

---

## Error Handling

```cpp
auto result = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .assess();

if (!result) {
    std::cerr << "Error: " << result.error().message << "\n";
    std::cerr << "Code: " << static_cast<int>(result.error().code) << "\n";
}
```

---

## Migration Guide

### From Old API

```cpp
// Old
auto risk = risk_mgr.assessTrade("SPY", 1000.0, 450.0, 440.0, 465.0, 0.65);

// New (recommended)
auto risk = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .withQuantity(10)
    .atPrice(450.0)
    .withStop(440.0)
    .withTarget(465.0)
    .withProbability(0.65)
    .assess();
```

Both work - migrate at your own pace!

