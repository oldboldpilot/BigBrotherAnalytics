#pragma once

/**
 * Compatibility header for correlation module
 * Full C++23 implementation in correlation.cppm
 */

#include "../utils/types.hpp"
#include <vector>
#include <span>
#include <string>
#include <unordered_map>

namespace bigbrother::correlation {

using namespace bigbrother::types;

// Forward declarations for compatibility
enum class CorrelationType {
    Pearson,
    Spearman,
    Kendall,
    Distance
};

struct CorrelationResult {
    std::string symbol1;
    std::string symbol2;
    double correlation{0.0};
    double p_value{0.0};
    int sample_size{0};
    int lag{0};
    CorrelationType type{CorrelationType::Pearson};

    [[nodiscard]] constexpr auto isSignificant(double alpha = 0.05) const noexcept -> bool {
        return p_value < alpha;
    }
};

struct TimeSeries {
    std::string symbol;
    std::vector<double> values;
    std::vector<Timestamp> timestamps;

    [[nodiscard]] auto size() const noexcept -> size_t {
        return values.size();
    }
};

class CorrelationMatrix {
public:
    CorrelationMatrix() = default;
    [[nodiscard]] auto get(std::string const&, std::string const&) const -> double { return 0.0; }
};

class CorrelationCalculator {
public:
    [[nodiscard]] static auto pearson(std::span<double const>, std::span<double const>) noexcept -> Result<double> {
        return 0.0;
    }
};

// Signal generation from correlations
class CorrelationSignalGenerator {
public:
    struct CorrelationSignal {
        std::string leading_symbol;
        std::string lagging_symbol;
        int optimal_lag{0};
        double correlation{0.0};
        double confidence{0.0};
    };
};

} // namespace bigbrother::correlation
