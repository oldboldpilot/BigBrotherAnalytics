#pragma once

/**
 * Compatibility header for options pricing
 * The full C++23 module is available in options_pricing.cppm
 */

#include "../utils/types.hpp"

// Minimal compatibility
namespace bigbrother::options {
    using namespace bigbrother::types;

    // Pricing parameters for compatibility
    struct PricingParams {
        Price spot_price{0.0};
        Price strike_price{0.0};
        double time_to_expiration{0.0};
        double risk_free_rate{0.0};
        double volatility{0.0};
        double dividend_yield{0.0};
        OptionType option_type{OptionType::Call};

        [[nodiscard]] auto validate() const -> Result<void> {
            if (spot_price <= 0.0) {
                return makeError<void>(ErrorCode::InvalidParameter, "Invalid spot price");
            }
            return {};
        }
    };

    // Pricing result
    struct PricingResult {
        double option_price{0.0};
        Greeks greeks{};
    };

    // Placeholder pricer
    class OptionsPricer {
    public:
        enum class Model { Auto, BlackScholes, Trinomial };

        static auto price(PricingParams const&, Model = Model::Auto) -> Result<PricingResult> {
            return PricingResult{};
        }

        static auto greeks(PricingParams const&, Model = Model::Auto) -> Result<Greeks> {
            return Greeks{};
        }
    };
}
