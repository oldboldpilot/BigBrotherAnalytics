#pragma once

/**
 * Compatibility header for math (forwards to existing C++ standard library)
 * The full C++23 module is available in math.cppm
 */

#include <vector>
#include <span>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <numbers>
#include <concepts>

// Basic compatibility - full module available in math.cppm
namespace bigbrother::utils::math {

    // Mean
    template<typename R>
    inline auto mean(R&& range) -> double {
        if (std::ranges::empty(range)) return 0.0;
        auto sum = std::accumulate(std::ranges::begin(range), std::ranges::end(range), 0.0);
        return sum / std::ranges::size(range);
    }

    // Standard deviation
    template<typename R>
    inline auto stddev(R&& range) -> double {
        if (std::ranges::size(range) < 2) return 0.0;
        auto m = mean(range);
        auto variance = std::accumulate(
            std::ranges::begin(range), std::ranges::end(range), 0.0,
            [m](double acc, double val) {
                auto diff = val - m;
                return acc + diff * diff;
            }
        ) / (std::ranges::size(range) - 1);
        return std::sqrt(variance);
    }

    // Percentile
    template<typename R>
    inline auto percentile(R&& range, double p) -> double {
        if (std::ranges::empty(range)) return 0.0;
        std::vector<double> sorted(std::ranges::begin(range), std::ranges::end(range));
        std::ranges::sort(sorted);
        auto index = p * (sorted.size() - 1);
        auto lower = static_cast<size_t>(std::floor(index));
        auto upper = static_cast<size_t>(std::ceil(index));
        if (lower == upper) return sorted[lower];
        auto weight = index - lower;
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
    }
}
