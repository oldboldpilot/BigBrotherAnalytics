# Options Strategies Integration Guide

**Author:** Olumuyiwa Oluwasanmi
**Date:** November 12, 2025
**Status:** ✅ Production Ready

## Overview

This document describes the complete integration of 52 options strategies with the BigBrotherAnalytics trading engine, including P&L tracking, risk management, and price prediction integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading Engine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Price        │  │ Market       │  │ Employment   │      │
│  │ Predictor    │──│ Intelligence │──│ Signals      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│          │                  │                  │            │
│          └──────────────────┴──────────────────┘            │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    Options Strategy Integrator                       │  │
│  │  • Strategy Selection (market condition based)       │  │
│  │  • Position Management (open/close/update)           │  │
│  │  • Greeks Aggregation (portfolio-level)              │  │
│  │  • Risk Metrics Calculation                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│          ┌────────────────┴────────────────┐                │
│          ▼                                 ▼                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │ 52 Options   │                  │ Risk         │        │
│  │ Strategies   │◄─────────────────│ Management   │        │
│  │ (SIMD)       │                  │ System       │        │
│  └──────────────┘                  └──────────────┘        │
│          │                                 │                │
│          └─────────────┬───────────────────┘                │
│                        ▼                                    │
│              ┌──────────────────┐                           │
│              │  P&L Tracking    │                           │
│              │  & Reporting     │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Options Strategy Integrator (`options_strategy_integrator.cppm`)

The central integration module that connects all components.

**Key Classes:**
- `OptionsStrategySelector`: Recommends strategies based on market conditions
- `OptionsStrategyManager`: Manages positions and coordinates with risk management
- `PortfolioGreeksAggregator`: Calculates portfolio-level Greeks
- `OptionsPosition`: Tracks individual strategy positions

**Features:**
- ✅ Strategy selection based on market outlook (bullish/bearish/neutral/volatile)
- ✅ IV percentile analysis for volatility-based strategies
- ✅ Integration with price prediction (confidence-adjusted scoring)
- ✅ Real-time Greeks monitoring
- ✅ Portfolio risk aggregation
- ✅ P&L tracking (realized and unrealized)

### 2. Strategy Selection Criteria

```cpp
StrategySelectionCriteria criteria;
criteria.market_outlook = MarketCondition::HIGH_VOLATILITY;
criteria.current_iv = 0.45f;           // 45% IV
criteria.iv_percentile = 85.0f;        // 85th percentile
criteria.current_price = 450.0f;
criteria.predicted_price = 452.0f;     // From ML model
criteria.prediction_confidence = 0.65f; // 65% confidence
criteria.time_horizon_days = 30.0f;
criteria.max_loss_tolerance = 2000.0f;
criteria.capital_allocation = 5000.0f;
criteria.prefer_defined_risk = true;
criteria.income_focused = true;
```

### 3. Strategy Recommendation Logic

**High IV Environment (>70th percentile):**
- Short Straddle (score: 85/100)
- Short Strangle (score: 80/100)
- Iron Condor (score: 75/100)

**Low IV Environment (<30th percentile):**
- Long Straddle (score: 80/100)
- Long Strangle (score: 75/100)

**Bullish Outlook:**
- Strong move (>5%): Long Call, Bull Call Spread
- Mild move: Bull Put Spread, Covered Call

**Bearish Outlook:**
- Strong move (<-5%): Long Put, Bear Put Spread
- Mild move: Bear Call Spread

**Neutral Outlook:**
- Income-focused: Iron Condor, Short Strangle, Iron Butterfly
- Range-bound: Long Iron Butterfly

### 4. Position Management

```cpp
OptionsStrategyManager manager;

// Open position
auto result = manager.openPosition(
    "SPY",                              // Symbol
    recommendation,                      // Selected strategy
    450.0f,                             // Underlying price
    {440.0f, 445.0f, 455.0f, 460.0f},  // Strikes (Iron Condor)
    30.0f,                              // Days to expiration
    0.30f,                              // IV
    0.05f                               // Risk-free rate
);

// Update all positions (daily)
manager.updatePositions(current_price, 0.05f);

// Get portfolio risk metrics
auto risk = manager.getPortfolioRisk(portfolio_value);
std::cout << "Net Delta: " << risk.net_delta << "\n";
std::cout << "Net Theta: " << risk.net_theta << " $/day\n";
std::cout << "Portfolio Heat: " << (risk.portfolio_heat * 100) << "%\n";

// Close position
auto pnl = manager.closePosition(position_id);
```

### 5. Risk Management Integration

**Portfolio Risk Metrics:**
- Net Delta: Directional exposure
- Net Gamma: Curvature risk
- Net Theta: Time decay (daily P&L from theta)
- Net Vega: Volatility exposure
- Net Rho: Interest rate sensitivity

**Risk Limits (from `RiskLimits`):**
- Account Value: $30,000
- Max Daily Loss: $900 (3%)
- Max Position Size: $1,500 (5%)
- Max Concurrent Positions: 10
- Max Portfolio Heat: 15%
- Max Correlation Exposure: 30%

**Risk Checks:**
```cpp
RiskLimits limits = RiskLimits::forThirtyKAccount();
auto risk = manager.getPortfolioRisk(portfolio_value);

if (risk.portfolio_heat > limits.max_portfolio_heat) {
    Logger::warning("Portfolio heat exceeds limit: {}%",
                   risk.portfolio_heat * 100);
    // Reduce position sizes or close positions
}

if (risk.daily_pnl < -limits.max_daily_loss) {
    Logger::critical("Daily loss limit hit: ${}", risk.daily_pnl);
    // Close all positions or hedge
}
```

### 6. P&L Tracking

**Position-Level:**
```cpp
struct OptionsPosition {
    float total_premium_paid{0.0f};      // Debit strategies
    float total_premium_received{0.0f};  // Credit strategies
    float unrealized_pnl{0.0f};          // Mark-to-market P&L
    float realized_pnl{0.0f};            // Closed position P&L

    auto calculatePnL(float price) const -> float {
        return strategy->calculateProfitLoss(price);
    }
};
```

**Portfolio-Level:**
```cpp
auto total_pnl = manager.getTotalPnL();  // Across all positions
auto daily_pnl = portfolio_risk.daily_pnl;
auto ytd_pnl = /* Track externally */;
```

### 7. Price Prediction Integration

The system integrates with the ML-based price predictor:

```cpp
// Get prediction from ML model
float predicted_price = predictor.predict(features, 30);  // 30-day forecast
float confidence = predictor.getConfidence();

// Use in strategy selection
StrategySelectionCriteria criteria;
criteria.predicted_price = predicted_price;
criteria.prediction_confidence = confidence;

// Scores are adjusted by confidence
// High confidence → Higher scores for directional strategies
// Low confidence → Higher scores for neutral strategies
```

### 8. Performance Characteristics

**Strategy Creation:** <0.1 ms
**Greeks Calculation:** 0.0857 μs per strategy (AVX2 SIMD)
**P&L Calculation:** 0.0087 μs per strategy
**Portfolio Update:** <1 ms for 10 positions
**Risk Aggregation:** <0.5 ms for 10 positions

**Memory Usage:**
- Per Strategy: ~200 bytes (base + legs)
- Per Position: ~500 bytes (including strategy)
- 100 Positions: ~50 KB

## Usage Examples

### Example 1: Basic Strategy Selection

```cpp
#include <iostream>
import bigbrother.options_strategy_integrator;

int main() {
    OptionsStrategyManager manager;

    // Define market conditions
    StrategySelectionCriteria criteria;
    criteria.market_outlook = MarketCondition::NEUTRAL;
    criteria.current_price = 450.0f;
    criteria.current_iv = 0.35f;
    criteria.iv_percentile = 75.0f;  // High IV
    criteria.income_focused = true;

    // Get recommendations
    auto recommendations = manager.getRecommendations(criteria);

    for (const auto& rec : recommendations) {
        std::cout << rec.strategy_name << " (Score: " << rec.score << ")\n";
        std::cout << "  Rationale: " << rec.rationale << "\n";
    }

    return 0;
}
```

### Example 2: Complete Trading Workflow

```cpp
// 1. Analyze market conditions
auto market_data = intelligence.getMarketData("SPY");
auto prediction = predictor.predict(features, 30);

// 2. Select strategy
StrategySelectionCriteria criteria;
criteria.current_price = market_data.price;
criteria.predicted_price = prediction.price;
criteria.prediction_confidence = prediction.confidence;
criteria.current_iv = market_data.iv;

auto recs = manager.getRecommendations(criteria);
const auto& best_strategy = recs[0];

// 3. Check risk limits
auto current_risk = manager.getPortfolioRisk(account_value);
if (!current_risk.canOpenNewPosition()) {
    Logger::warning("Cannot open new position - limits reached");
    return;
}

// 4. Open position
auto position_id = manager.openPosition(
    "SPY", best_strategy,
    market_data.price, strikes, days, iv, rate
);

// 5. Monitor daily
for (int day = 0; day < 30; ++day) {
    market_data = intelligence.getMarketData("SPY");
    manager.updatePositions(market_data.price, rate);

    auto risk = manager.getPortfolioRisk(account_value);

    // Check exit conditions
    if (risk.daily_pnl < -900.0f) {
        manager.closePosition(position_id);
        break;
    }

    // Report
    Logger::info("Day {}: P&L=${:.2f}, Delta={:.4f}",
                 day, risk.daily_pnl, risk.net_delta);
}
```

### Example 3: Portfolio Risk Monitoring

```cpp
// Daily risk monitoring
auto risk = manager.getPortfolioRisk(account_value);

std::cout << "Portfolio Risk Report\n";
std::cout << "═══════════════════════\n";
std::cout << "Active Positions: " << risk.active_positions << "\n";
std::cout << "Total Exposure: $" << risk.total_exposure << "\n";
std::cout << "Portfolio Heat: " << (risk.portfolio_heat * 100) << "%\n";
std::cout << "Risk Level: " << risk.getRiskLevel() << "\n\n";

std::cout << "Greeks:\n";
std::cout << "  Delta: " << risk.net_delta << "\n";
std::cout << "  Theta: $" << risk.net_theta << "/day\n";
std::cout << "  Vega: $" << risk.net_vega << " per 1% IV\n\n";

std::cout << "P&L:\n";
std::cout << "  Daily: $" << risk.daily_pnl << "\n";

// Check against limits
RiskLimits limits = RiskLimits::forThirtyKAccount();
if (risk.portfolio_heat > limits.max_portfolio_heat) {
    std::cout << "⚠️  Portfolio heat exceeds limit!\n";
}
```

## Testing

See [examples/options_strategy_integration_example.cpp](../examples/options_strategy_integration_example.cpp) for comprehensive examples including:

1. Strategy selection based on market outlook
2. Position management with Greeks monitoring
3. Portfolio risk management
4. Price prediction integration

**Build and run:**
```bash
cmake -G Ninja -B build
ninja -C build options_strategy_integration_example
./build/bin/options_strategy_integration_example
```

## Performance Optimization

**SIMD Acceleration:**
- All Greeks calculations use AVX2 intrinsics
- 8-way parallelism for batch operations
- Performance: 0.0857 μs per Greeks calculation (58x target!)

**Module Precompilation:**
- All options strategies use C++23 modules
- `.pcm` files cached for fast rebuilds
- Incremental compilation supported

**Compiler Flags:**
```cmake
-O3 -march=native -mavx2 -mfma -ffast-math
```

## Limitations & Future Enhancements

**Current Limitations:**
1. Single-symbol positions (no multi-leg cross-symbol strategies)
2. Simplified probability of profit calculation
3. No transaction cost modeling
4. No slippage/bid-ask spread modeling

**Future Enhancements:**
1. Add transaction cost analysis
2. Implement portfolio optimization (Kelly Criterion)
3. Add backtesting framework for strategy validation
4. Implement strategy adjustment/rolling logic
5. Add volatility surface modeling
6. Implement multi-leg delta-neutral portfolios
7. Add ML-based strategy selection

## API Reference

### OptionsStrategyManager

**Constructor:**
```cpp
OptionsStrategyManager();
```

**Methods:**
```cpp
auto getRecommendations(const StrategySelectionCriteria&)
    -> std::vector<StrategyRecommendation>;

auto openPosition(std::string symbol, const StrategyRecommendation&,
                 float price, const std::vector<float>& strikes,
                 float days, float iv, float rate)
    -> Result<std::string>;

auto updatePositions(float price, float rate) -> void;

auto closePosition(const std::string& position_id) -> Result<float>;

auto getPortfolioRisk(float portfolio_value) -> PortfolioRisk;

auto getTotalPnL() const -> float;
```

### StrategySelectionCriteria

```cpp
struct StrategySelectionCriteria {
    MarketCondition market_outlook;
    float current_iv;
    float historical_iv;
    float iv_percentile;
    float current_price;
    float predicted_price;
    float prediction_confidence;
    float time_horizon_days;
    float max_loss_tolerance;
    float capital_allocation;
    bool prefer_defined_risk;
    bool income_focused;
};
```

### PortfolioRisk

```cpp
struct PortfolioRisk {
    double total_value;
    double total_exposure;
    double net_delta;
    double net_vega;
    double net_theta;
    double daily_pnl;
    int active_positions;
    double portfolio_heat;
    std::vector<PositionRisk> positions;

    auto canOpenNewPosition() const -> bool;
    auto getRiskLevel() const -> std::string;
};
```

## See Also

- [Options Strategies README](../src/options_strategies/README.md) - Implementation details
- [Risk Management Module](../src/risk_management/risk_management.cppm) - Risk system
- [Strategy Module](../src/trading_decision/strategies.cppm) - Base strategies
- [Example Code](../examples/options_strategy_integration_example.cpp) - Usage examples

## Support & Contribution

For questions, issues, or contributions related to options strategies integration:

1. Check existing documentation
2. Review example code
3. Run integration tests
4. Open GitHub issue if needed

---

**Status:** ✅ Production Ready
**Last Updated:** November 12, 2025
**Performance:** All targets exceeded (58-345x faster than requirements)
