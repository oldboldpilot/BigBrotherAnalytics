# Options Trading System Documentation

## Overview

BigBrotherAnalytics provides a comprehensive, professional-grade options trading system with ML-driven strategy selection, real-time Greeks monitoring, and advanced risk management.

**Version:** 4.0
**Last Updated:** 2025-11-14
**Author:** Olumuyiwa Oluwasanmi

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Supported Strategies](#supported-strategies)
4. [Greeks Calculation](#greeks-calculation)
5. [ML Integration](#ml-integration)
6. [Risk Management](#risk-management)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)
9. [Performance Optimization](#performance-optimization)

---

## Features

### Core Capabilities
- **Trinomial Tree Pricing**: American and European options with high accuracy
- **Real-Time Greeks**: Delta, gamma, theta, vega, rho calculated using finite differences
- **SIMD Optimization**: AVX2/NEON vectorization for performance
- **OpenMP Parallelization**: Multi-threaded Greeks calculation

### Trading Strategies
- ✅ **Covered Calls**: Income generation from stock holdings
- ✅ **Cash-Secured Puts**: Stock acquisition at discount
- ✅ **Bull/Bear Call Spreads**: Directional strategies with defined risk
- ✅ **Bull/Bear Put Spreads**: Alternative directional plays
- ✅ **Iron Condors**: Range-bound profit with limited risk
- ✅ **Protective Puts**: Portfolio hedging

### ML Integration
- **Price Prediction**: Uses PricePredictor v4.0 (INT32 SIMD, 85 features, 98.18% accuracy)
- **Directional Bias**: Adjusts strike selection based on predictions
- **Confidence Scoring**: Strategy confidence tied to prediction confidence
- **Implied Volatility Analysis**: IV rank and IV percentile calculations

### Risk Management
- **Position Sizing**: Maximum % of portfolio per position
- **Greeks Limits**: Delta, gamma, theta, vega exposure caps
- **Collateral Management**: Automated collateral calculations
- **Stop Loss**: Automated stop-loss triggers (50% of max loss default)

---

## Architecture

### Module Structure

```
src/options/
├── trinomial_pricer.cppm       # Trinomial tree option pricing
├── strategies.cppm             # Options strategy implementations
├── greeks_calculator.cppm      # Real-time Greeks calculation
└── options.cppm                # Main export module

tests/
└── test_options_strategies.cpp # Comprehensive test suite
```

### Class Hierarchy

```
options::TrinomialPricer
    └─> price() -> PricingResult (price + Greeks)

options::OptionsStrategyEngine
    ├─> generateCoveredCall()
    ├─> generateCashSecuredPut()
    ├─> generateBullCallSpread()
    ├─> generateBearCallSpread()
    ├─> generateBullPutSpread()
    ├─> generateBearPutSpread()
    ├─> generateIronCondor()
    └─> generateProtectivePut()

options::GreeksCalculator
    ├─> calculatePositionGreeks()
    ├─> calculatePortfolioGreeks()
    ├─> calculateDeltaHedge()
    └─> evaluateGammaScalping()

options::GreeksMonitor
    ├─> addPosition()
    ├─> updatePosition()
    └─> getPortfolioGreeks()
```

---

## Supported Strategies

### 1. Covered Call

**Description**: Sell call options against owned stock to generate income.

**When to Use**:
- Neutral to slightly bullish outlook
- Want to generate income from holdings
- Willing to cap upside at strike price

**Risk Profile**:
- Max Profit: (Strike - Current Price) × Shares + Premium
- Max Loss: Stock drops to $0 - Premium collected
- Breakeven: Entry Price - Premium per share

**Requirements**:
- Own at least 100 shares (1 contract)
- No cash required (covered by shares)

**Example**:
```cpp
OptionsStrategyEngine engine;
auto position = engine.generateCoveredCall(
    "AAPL",           // Symbol
    150.0,            // Current price
    200,              // Shares owned
    100000.0          // Account value
);

// Sells 2 call contracts (200 shares / 100)
// Strike: ~$157.50 (5% OTM)
// Premium collected: ~$300-500
```

---

### 2. Cash-Secured Put

**Description**: Sell put options with cash reserve to buy stock at lower price.

**When to Use**:
- Bullish or neutral on stock
- Willing to buy stock at lower price
- Want to generate income while waiting

**Risk Profile**:
- Max Profit: Premium collected
- Max Loss: (Strike × 100) - Premium (if stock goes to $0)
- Breakeven: Strike - Premium per share

**Requirements**:
- Cash equal to (Strike × 100) per contract
- Account value sufficient for margin requirements

**Example**:
```cpp
auto position = engine.generateCashSecuredPut(
    "TSLA",           // Symbol
    200.0,            // Current price
    50000.0,          // Available cash
    100000.0          // Account value
);

// Strike: $194 (3% below current)
// Collateral per contract: $19,400
// Max contracts: 2 (based on cash available)
```

---

### 3. Bull Call Spread

**Description**: Buy lower strike call, sell higher strike call. Bullish strategy with defined risk.

**When to Use**:
- Moderately bullish outlook
- Want limited risk exposure
- Expect move above lower strike but below upper strike

**Risk Profile**:
- Max Profit: (Upper Strike - Lower Strike) × 100 - Net Debit
- Max Loss: Net Debit (premium paid)
- Breakeven: Lower Strike + Net Debit per share

**Example**:
```cpp
auto position = engine.generateBullCallSpread(
    "SPY",            // Symbol
    450.0,            // Current price
    10000.0,          // Available cash
    100000.0          // Account value
);

// Long call: $450 strike
// Short call: $472.50 strike (5% spread)
// Net debit: ~$800 per contract
// Max profit: ~$1,450 per contract
```

---

### 4. Iron Condor

**Description**: Sell OTM put spread + sell OTM call spread. Profits in range-bound market.

**When to Use**:
- Neutral outlook (low volatility expected)
- Stock expected to stay within range
- Want to collect premium from time decay

**Risk Profile**:
- Max Profit: Net premium collected
- Max Loss: (Spread width × 100) - Net premium
- Breakeven: Two breakevens (one on put side, one on call side)

**Example**:
```cpp
auto position = engine.generateIronCondor(
    "QQQ",            // Symbol
    380.0,            // Current price
    15000.0,          // Available cash
    100000.0          // Account value
);

// Put spread: $353/$361 (7% / 5% OTM)
// Call spread: $399/$407 (5% / 7% OTM)
// Net credit: ~$200 per condor
// Max risk: ~$600 per condor
// Risk/Reward: 1:3 ratio
```

---

### 5. Protective Put

**Description**: Buy put options to hedge existing stock position against downside.

**When to Use**:
- Own stock with unrealized gains
- Concerned about short-term downside
- Want insurance without selling stock

**Risk Profile**:
- Max Profit: Unlimited (stock can rise indefinitely)
- Max Loss: (Current Price - Strike) × Shares + Premium
- Breakeven: Current Price + Premium per share

**Example**:
```cpp
auto position = engine.generateProtectivePut(
    "NVDA",           // Symbol
    500.0,            // Current price
    100,              // Shares owned
    5000.0            // Available cash
);

// Strike: $475 (5% below current)
// Protection cost: ~$800-1,200
// Limits max loss to ~$3,300 (vs $50,000 without)
```

---

## Greeks Calculation

### What are Greeks?

Greeks measure how option prices change with respect to various factors:

| Greek | Measures | Range | Interpretation |
|-------|----------|-------|----------------|
| **Delta (Δ)** | Price sensitivity | -1 to +1 | 0.50 = 50% probability of expiring ITM |
| **Gamma (Γ)** | Delta sensitivity | 0 to ∞ | High gamma = delta changes rapidly |
| **Theta (Θ)** | Time decay | Negative | -0.10 = lose $0.10 per day |
| **Vega (ν)** | Volatility sensitivity | 0 to ∞ | 0.20 = +$0.20 for 1% IV increase |
| **Rho (ρ)** | Interest rate sensitivity | -∞ to +∞ | 0.05 = +$0.05 for 1% rate increase |

### Calculation Method

Greeks are calculated using finite differences on the trinomial tree:

```cpp
// Delta: ∂V/∂S
delta = (V(S + dS) - V(S - dS)) / (2 * dS)

// Gamma: ∂²V/∂S²
gamma = (V(S + dS) - 2*V(S) + V(S - dS)) / (dS²)

// Theta: ∂V/∂t
theta = (V(t - dt) - V(t)) / dt

// Vega: ∂V/∂σ
vega = (V(σ + dσ) - V(σ)) / dσ

// Rho: ∂V/∂r
rho = (V(r + dr) - V(r)) / dr
```

### Portfolio Greeks

Aggregate Greeks across all positions:

```cpp
GreeksCalculator calc;
std::vector<PositionGreeks> positions = getActivePositions();

auto portfolio = calc.calculatePortfolioGreeks(positions, portfolio_value);

std::cout << "Total Delta: " << portfolio.total_delta << std::endl;
std::cout << "Total Theta: $" << portfolio.dollar_theta << "/day" << std::endl;
std::cout << "Delta Neutrality: " << portfolio.delta_neutrality_score * 100 << "%" << std::endl;
```

### Delta Hedging

Achieve delta neutrality by trading underlying stock:

```cpp
auto hedge = calc.calculateDeltaHedge(portfolio, current_stock_price);

// If delta = +0.45, need to SELL 45 shares to neutralize
std::cout << hedge.action << " " << hedge.shares_to_trade << " shares" << std::endl;
std::cout << "Cost: $" << hedge.cost_estimate << std::endl;
```

### Gamma Scalping

Profit from gamma when market moves:

```cpp
auto scalp = calc.evaluateGammaScalping(portfolio, current_price, price_change);

if (scalp) {
    std::cout << "Rehedge by trading " << scalp->shares_to_adjust << " shares" << std::endl;
    std::cout << "Expected P/L: $" << scalp->expected_profit << std::endl;
}
```

---

## ML Integration

### Price Prediction Integration

Options strategy selection is driven by ML price predictions from `PricePredictor`:

```cpp
// PricePredictor automatically used inside strategy generation
auto position = engine.generateCoveredCall(symbol, current_price, shares, account_value);

// Internally:
// 1. Get prediction: predictor.predictPrice(symbol)
// 2. Adjust strike based on directional bias:
//    - Bullish (>2% predicted gain): Sell 5% OTM call
//    - Neutral (<2% predicted gain): Sell 2% OTM call (ATM)
// 3. Set confidence from prediction confidence
```

### Strike Selection Logic

| Strategy | ML Prediction | Strike Selection |
|----------|---------------|------------------|
| Covered Call | Bullish (>2%) | 105% of current (5% OTM) |
| Covered Call | Neutral | 102% of current (2% OTM) |
| Cash-Secured Put | Bullish/Neutral | 97% of current (3% OTM) |
| Cash-Secured Put | Bearish (<-3%) | Skip (don't sell puts) |
| Bull Call Spread | Bullish (>2%) | Long: ATM, Short: +5% |
| Iron Condor | Neutral (<5% move) | ±5-7% OTM |
| Iron Condor | High volatility | Skip (expect large move) |

### Confidence Scoring

Strategy confidence inherits from ML prediction confidence:

```cpp
auto position = engine.generateBullCallSpread(...);

// position.confidence = prediction.confidence_20d
// Only actionable if confidence > 0.60 (60%)
```

---

## Risk Management

### Position Sizing

Configurable limits prevent over-concentration:

```cpp
struct StrategyConfig {
    double max_position_size_pct = 0.05;        // 5% max per position
    double max_portfolio_allocation_pct = 0.20; // 20% max total options
    ...
};
```

### Greeks Limits

Enforce risk caps on portfolio exposure:

```cpp
struct StrategyConfig {
    double max_delta_per_position = 0.50;       // Max delta per position
    double max_portfolio_delta = 2.00;          // Max total delta
    double max_theta_per_position = -50.0;      // Max theta decay
    double max_vega_per_position = 100.0;       // Max vega
    ...
};
```

### Position Validation

Every position validated before execution:

```cpp
bool is_valid = engine.validatePosition(
    position,
    current_portfolio_delta,
    current_portfolio_theta,
    account_value
);

if (!is_valid) {
    // Position rejected due to risk limits
}
```

### Stop Loss

Automated stop-loss at 50% of max loss (configurable):

```cpp
struct StrategyConfig {
    double stop_loss_pct = 0.50;  // Stop at 50% of max loss
};

// If max loss = $1,000, stop triggered at -$500 unrealized P/L
```

---

## Usage Examples

### Example 1: Generate Covered Call

```cpp
#include <iostream>
import bigbrother.options;

int main() {
    options::OptionsStrategyEngine engine;

    auto position = engine.generateCoveredCall(
        "AAPL",      // Symbol
        150.0,       // Current price
        300,         // Shares owned (3 contracts)
        100000.0     // Account value
    );

    if (position) {
        std::cout << "Strategy: Covered Call on " << position->symbol << std::endl;
        std::cout << "Strike: $" << position->legs[0].strike << std::endl;
        std::cout << "Premium: $" << position->legs[0].premium * 3 << std::endl;
        std::cout << "Max Profit: $" << position->max_profit << std::endl;
        std::cout << "Rationale: " << position->rationale << std::endl;
    }
}
```

### Example 2: Monitor Portfolio Greeks

```cpp
#include <iostream>
import bigbrother.options;

int main() {
    options::GreeksMonitor monitor;
    options::GreeksCalculator calc;

    // Add positions
    auto greeks1 = calc.calculatePositionGreeks(
        "SPY", 450.0, 450.0, 30.0/365.0, 0.20, 0.05,
        options::OptionType::CALL, 2  // Long 2 calls
    );
    greeks1.position_id = "POS_001";
    monitor.addPosition(greeks1);

    auto greeks2 = calc.calculatePositionGreeks(
        "SPY", 450.0, 460.0, 30.0/365.0, 0.20, 0.05,
        options::OptionType::CALL, -2  // Short 2 calls
    );
    greeks2.position_id = "POS_002";
    monitor.addPosition(greeks2);

    // Get portfolio Greeks
    auto portfolio = monitor.getPortfolioGreeks(100000.0);

    std::cout << "Portfolio Delta: " << portfolio.total_delta << std::endl;
    std::cout << "Portfolio Theta: $" << portfolio.dollar_theta << "/day" << std::endl;
    std::cout << "Delta Neutrality: " << portfolio.delta_neutrality_score * 100 << "%" << std::endl;

    // Calculate hedge
    auto hedge = calc.calculateDeltaHedge(portfolio, 450.0);
    std::cout << "Hedge: " << hedge.action << " " << hedge.shares_to_trade << " shares" << std::endl;
}
```

### Example 3: Iron Condor with ML

```cpp
#include <iostream>
import bigbrother.options;
import bigbrother.market_intelligence.price_predictor;

int main() {
    options::OptionsStrategyEngine engine;

    // ML predictor already integrated - no manual prediction needed
    auto position = engine.generateIronCondor(
        "QQQ",       // Symbol
        380.0,       // Current price
        15000.0,     // Available cash
        100000.0     // Account value
    );

    if (position) {
        std::cout << "Iron Condor Generated!" << std::endl;
        std::cout << "Legs: " << position->legs.size() << std::endl;
        std::cout << "Max Profit: $" << position->max_profit << std::endl;
        std::cout << "Max Risk: $" << position->max_loss << std::endl;
        std::cout << "Confidence: " << position->confidence * 100 << "%" << std::endl;
        std::cout << "Rationale: " << position->rationale << std::endl;

        // Validate against risk limits
        bool is_valid = engine.validatePosition(
            *position,
            0.0,        // Current portfolio delta
            -100.0,     // Current portfolio theta
            100000.0    // Account value
        );

        if (is_valid) {
            std::cout << "Position passed risk validation" << std::endl;
            // Execute position...
        }
    } else {
        std::cout << "No iron condor opportunity (likely high predicted movement)" << std::endl;
    }
}
```

---

## API Reference

### OptionsStrategyEngine

```cpp
class OptionsStrategyEngine {
public:
    explicit OptionsStrategyEngine(StrategyConfig config = StrategyConfig{});

    // Strategy generation
    auto generateCoveredCall(string const& symbol, double current_price,
                             int shares_owned, double account_value)
        -> optional<OptionsPosition>;

    auto generateCashSecuredPut(string const& symbol, double current_price,
                                double available_cash, double account_value)
        -> optional<OptionsPosition>;

    auto generateBullCallSpread(string const& symbol, double current_price,
                                double available_cash, double account_value)
        -> optional<OptionsPosition>;

    auto generateBearCallSpread(...) -> optional<OptionsPosition>;
    auto generateBullPutSpread(...) -> optional<OptionsPosition>;
    auto generateBearPutSpread(...) -> optional<OptionsPosition>;

    auto generateIronCondor(string const& symbol, double current_price,
                            double available_cash, double account_value)
        -> optional<OptionsPosition>;

    auto generateProtectivePut(string const& symbol, double current_price,
                               int shares_owned, double available_cash)
        -> optional<OptionsPosition>;

    // Risk management
    auto validatePosition(OptionsPosition const& position,
                         double current_portfolio_delta,
                         double current_portfolio_theta,
                         double account_value) const -> bool;

    // Strategy recommendation
    auto recommendStrategy(MarketOutlook outlook, bool owns_stock,
                          double cash_available, double stock_price) const
        -> vector<StrategyType>;
};
```

### GreeksCalculator

```cpp
class GreeksCalculator {
public:
    auto calculatePositionGreeks(string const& symbol, double spot,
                                 double strike, double time_to_expiry,
                                 double volatility, double risk_free_rate,
                                 OptionType type, int quantity)
        -> PositionGreeks;

    auto calculatePortfolioGreeks(vector<PositionGreeks> const& positions,
                                  double portfolio_value)
        -> PortfolioGreeks;

    auto calculateDeltaHedge(PortfolioGreeks const& portfolio,
                            double current_stock_price)
        -> HedgeRecommendation;

    auto evaluateGammaScalping(PortfolioGreeks const& portfolio,
                               double current_price, double price_change)
        -> optional<ScalpingOpportunity>;

    auto estimateThetaDecay(PortfolioGreeks const& portfolio, int days)
        -> double;

    auto estimateVolatilityPnL(PortfolioGreeks const& portfolio,
                               double vol_change_pct)
        -> double;
};
```

---

## Performance Optimization

### SIMD Vectorization

Trinomial tree calculations use AVX2/NEON SIMD:

```cpp
// Processes 4 tree nodes simultaneously
__m256d vec_up = _mm256_loadu_pd(&option_values[i + 1][j + 2]);
__m256d vec_mid = _mm256_loadu_pd(&option_values[i + 1][j + 1]);
__m256d vec_down = _mm256_loadu_pd(&option_values[i + 1][j]);
__m256d vec_result = _mm256_mul_pd(vec_pu, vec_up);
vec_result = _mm256_fmadd_pd(vec_pm, vec_mid, vec_result);
```

**Performance**: 3-4x speedup vs scalar code

### OpenMP Parallelization

Greeks calculation parallelized across multiple threads:

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    { V_up = calculatePrice(...); }

    #pragma omp section
    { V_down = calculatePrice(...); }

    #pragma omp section
    { V_t_minus = calculatePrice(...); }
}
```

**Performance**: 5-7x speedup on 8-core systems

### Benchmarks

| Operation | Time (single-threaded) | Time (optimized) | Speedup |
|-----------|----------------------|-----------------|---------|
| Price 1 option | 0.8ms | 0.2ms | 4.0x |
| Calculate all Greeks | 6.5ms | 1.1ms | 5.9x |
| Price iron condor (4 legs) | 3.2ms | 0.6ms | 5.3x |
| Portfolio Greeks (10 positions) | 65ms | 9ms | 7.2x |

---

## Best Practices

### 1. Strategy Selection
- **Use ML predictions**: Let PricePredictor guide strategy choice
- **Check IV rank**: Sell premium when IV > 50th percentile
- **Diversify**: Don't concentrate in one strategy type

### 2. Position Management
- **Size appropriately**: Never exceed 5% per position
- **Monitor Greeks**: Check portfolio Greeks daily
- **Adjust proactively**: Rehedge when delta > ±0.10 from neutral

### 3. Risk Control
- **Set stops**: Always use stop-loss orders
- **Limit exposure**: Cap total options at 20% of portfolio
- **Track theta**: Know your daily time decay

### 4. Execution
- **Check liquidity**: Only trade options with >100 daily volume
- **Use limit orders**: Never use market orders for options
- **Time entries**: Enter positions 30-45 DTE for optimal theta

---

## Troubleshooting

### Issue: No strategies generated

**Cause**: ML prediction indicates high volatility or strong directional move

**Solution**: Check prediction output. Iron condors require low predicted movement (<5%)

### Issue: Position validation fails

**Cause**: Position exceeds risk limits (delta, theta, vega, or position size)

**Solution**: Reduce position size or adjust existing positions to free up Greek capacity

### Issue: Greeks calculation seems incorrect

**Cause**: Stale implied volatility or price data

**Solution**: Ensure IV and spot price are up-to-date before calculating Greeks

---

## References

- [Options Pricing: Trinomial Trees](https://en.wikipedia.org/wiki/Trinomial_tree)
- [The Greeks: Essential Risk Metrics](https://www.investopedia.com/terms/g/greeks.asp)
- [Iron Condor Strategy Guide](https://www.optionsplaybook.com/option-strategies/iron-condor/)
- BigBrotherAnalytics ML Documentation: `docs/ML_QUANTIZATION.md`

---

## Contributing

To add new options strategies:

1. Add strategy to `StrategyType` enum in `src/options/strategies.cppm`
2. Implement `generate<StrategyName>()` method in `OptionsStrategyEngine`
3. Add validation logic in `validatePosition()`
4. Add test cases in `tests/test_options_strategies.cpp`
5. Update this documentation

---

## License

© 2025 Olumuyiwa Oluwasanmi. All rights reserved.
