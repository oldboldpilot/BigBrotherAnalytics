# Fluent Risk Manager API - Quick Reference

## Quick Start

```cpp
import bigbrother.risk_management;

RiskManager risk_mgr;

// Configure
risk_mgr.setMaxDailyLoss(900.0)
    .setMaxPositionSize(1500.0);

// Assess trade
auto risk = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .withQuantity(10)
    .atPrice(450.00)
    .withStop(440.00)
    .withTarget(465.00)
    .withProbability(0.65)
    .assess();

// Check result
if (risk && risk->approved) {
    // Execute trade
}
```

## Configuration Methods

```cpp
risk_mgr.setMaxDailyLoss(900.0)           // → RiskManager&
        .setMaxPositionSize(1500.0)       // → RiskManager&
        .setMaxPortfolioHeat(0.15)        // → RiskManager&
        .setMaxConcurrentPositions(10)    // → RiskManager&
        .setAccountValue(30000.0)         // → RiskManager&
        .requireStopLoss(true);           // → RiskManager&
```

## Trade Assessment (TradeRiskBuilder)

```cpp
risk_mgr.assessTrade()              // → TradeRiskBuilder
    .forSymbol("SPY")               // → TradeRiskBuilder&
    .withQuantity(10)               // → TradeRiskBuilder&
    .atPrice(450.00)                // → TradeRiskBuilder&
    .withStop(440.00)               // → TradeRiskBuilder&
    .withTarget(465.00)             // → TradeRiskBuilder&
    .withProbability(0.65)          // → TradeRiskBuilder&
    .assess()                       // → Result<TradeRisk> (terminal)
```

## Portfolio Analysis (PortfolioRiskBuilder)

```cpp
risk_mgr.portfolio()                              // → PortfolioRiskBuilder
    .addPosition("SPY", 10, 450.00, 0.05)        // → PortfolioRiskBuilder&
    .addPosition("XLE", 20, 80.00, 0.08)         // → PortfolioRiskBuilder&
    .calculateHeat()                              // → PortfolioRiskBuilder&
    .calculateVaR(0.95)                           // → PortfolioRiskBuilder&
    .analyze()                                    // → Result<PortfolioRisk> (terminal)
```

## Kelly Criterion (KellyCalculator)

```cpp
risk_mgr.kelly()                        // → KellyCalculator
    .withWinRate(0.55)                  // → KellyCalculator&
    .withWinLossRatio(1.8)              // → KellyCalculator&
    .withDrawdownLimit(0.25)            // → KellyCalculator&
    .calculate()                        // → Result<double> (terminal)
```

## Position Sizing (PositionSizerBuilder)

```cpp
risk_mgr.positionSizer()                    // → PositionSizerBuilder
    .withMethod(SizingMethod::KellyHalf)    // → PositionSizerBuilder&
    .withAccountValue(30000.0)              // → PositionSizerBuilder&
    .withWinProbability(0.60)               // → PositionSizerBuilder&
    .withWinAmount(100.0)                   // → PositionSizerBuilder&
    .withLossAmount(80.0)                   // → PositionSizerBuilder&
    .withVolatility(0.25)                   // → PositionSizerBuilder&
    .calculate()                            // → Result<double> (terminal)
```

## Monte Carlo Simulation (MonteCarloSimulatorBuilder)

```cpp
risk_mgr.monteCarlo()                   // → MonteCarloSimulatorBuilder
    .forOption(params)                  // → MonteCarloSimulatorBuilder&
    .withSimulations(10000)             // → MonteCarloSimulatorBuilder&
    .withSteps(100)                     // → MonteCarloSimulatorBuilder&
    .withPositionSize(100.0)            // → MonteCarloSimulatorBuilder&
    .run()                              // → Result<SimulationResult> (terminal)
```

## Daily P&L Management

```cpp
risk_mgr.updateDailyPnL(500.0)              // → RiskManager&
        .updateDailyPnL(-200.0);             // → RiskManager&

double pnl = risk_mgr.getDailyPnL();                    // double
double remaining = risk_mgr.getDailyLossRemaining();    // double
bool breached = risk_mgr.isDailyLossLimitReached();     // bool

risk_mgr.resetDaily();                      // → RiskManager&
```

## Error Handling Pattern

```cpp
auto result = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .assess();

if (!result) {
    std::cerr << result.error().message << "\n";
    return;
}

// Use result
if (result->approved) {
    // Execute
}
```

## Sizing Methods

```cpp
enum class SizingMethod {
    FixedDollar,          // Fixed $1000
    FixedPercent,         // 5% of account
    KellyCriterion,       // Full Kelly
    KellyHalf,            // Conservative
    VolatilityAdjusted,   // Vol-sensitive
    RiskParity            // Equal risk
};
```

## Complete Workflow Example

```cpp
// 1. Create and configure
RiskManager risk_mgr;
risk_mgr.setMaxDailyLoss(900.0)
    .setMaxPositionSize(1500.0);

// 2. Assess trade
auto risk = risk_mgr.assessTrade()
    .forSymbol("SPY")
    .withQuantity(10)
    .atPrice(450.0)
    .withStop(440.0)
    .withTarget(465.0)
    .withProbability(0.65)
    .assess();

if (!risk || !risk->approved) {
    return;  // Trade rejected
}

// 3. Calculate Kelly-based sizing
auto kelly = risk_mgr.kelly()
    .withWinRate(0.65)
    .withWinLossRatio(2.0)
    .calculate();

// 4. Calculate position size
auto size = risk_mgr.positionSizer()
    .withMethod(SizingMethod::KellyHalf)
    .withWinProbability(0.65)
    .withWinAmount(150.0)
    .withLossAmount(100.0)
    .calculate();

// 5. Execute trade with calculated size

// 6. Update portfolio
auto portfolio = risk_mgr.portfolio()
    .addPosition("SPY", 10, 450.0)
    .calculateHeat()
    .analyze();

// 7. Track P&L
risk_mgr.updateDailyPnL(pnl);
if (risk_mgr.isDailyLossLimitReached()) {
    // Stop trading
}

// 8. Reset at end of day
risk_mgr.resetDaily();
```

## Key Points

- **Chaining**: All configuration methods return `RiskManager&`
- **Builders**: Use builder pattern for complex operations
- **Terminals**: Methods like `.assess()`, `.analyze()`, `.calculate()` are terminal
- **Results**: Terminal operations return `Result<T>` with error handling
- **Thread-Safe**: All operations are mutex-protected
- **C++23**: Uses trailing return syntax and modern features

## Files

- **Implementation**: `src/risk_management/risk_management.cppm`
- **Tests**: `tests/cpp/test_risk_fluent_api.cpp`
- **Documentation**: `docs/FLUENT_RISK_API.md`
- **Examples**: `examples/fluent_risk_examples.cpp`
- **Summary**: `FLUENT_API_IMPLEMENTATION.md`

