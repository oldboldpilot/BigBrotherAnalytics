#include "strategy_volatility_arb.hpp"

namespace bigbrother::strategy {

// VolatilityArbitrageStrategy default constructors
VolatilityArbitrageStrategy::VolatilityArbitrageStrategy()
    : BaseStrategy("VolatilityArbitrage", "Vol arb strategy") {}

VolatilityArbitrageStrategy::VolatilityArbitrageStrategy(Parameters params)
    : BaseStrategy("VolatilityArbitrage", "Vol arb strategy"),
      params_{std::move(params)} {}

// MeanReversionStrategy default constructors
MeanReversionStrategy::MeanReversionStrategy()
    : BaseStrategy("MeanReversion", "Mean reversion strategy") {}

MeanReversionStrategy::MeanReversionStrategy(Parameters params)
    : BaseStrategy("MeanReversion", "Mean reversion strategy"),
      params_{std::move(params)} {}

}
