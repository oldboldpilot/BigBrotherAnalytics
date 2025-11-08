#include "strategy_volatility_arb.hpp"

namespace bigbrother::strategy {

// VolatilityArbitrageStrategy default constructors
VolatilityArbitrageStrategy::VolatilityArbitrageStrategy()
    : BaseStrategy("VolatilityArbitrage", "Vol arb strategy") {}

VolatilityArbitrageStrategy::VolatilityArbitrageStrategy(Parameters params)
    : BaseStrategy("VolatilityArbitrage", "Vol arb strategy"),
      params_{std::move(params)} {}

[[nodiscard]] auto VolatilityArbitrageStrategy::generateSignals(
    StrategyContext const& context
) -> std::vector<TradingSignal> {
    // Stub
    return {};
}

[[nodiscard]] auto VolatilityArbitrageStrategy::getParameters() const
    -> std::unordered_map<std::string, std::string> {
    return {};  // Stub
}

// ============================================================================
// MeanReversionStrategy Implementation
// ============================================================================

MeanReversionStrategy::MeanReversionStrategy()
    : BaseStrategy("MeanReversion", "Mean reversion strategy") {}

MeanReversionStrategy::MeanReversionStrategy(Parameters params)
    : BaseStrategy("MeanReversion", "Mean reversion strategy"),
      params_{std::move(params)} {}

[[nodiscard]] auto MeanReversionStrategy::generateSignals(
    StrategyContext const& context
) -> std::vector<TradingSignal> {
    // Stub
    return {};
}

[[nodiscard]] auto MeanReversionStrategy::getParameters() const
    -> std::unordered_map<std::string, std::string> {
    return {};  // Stub
}

}
