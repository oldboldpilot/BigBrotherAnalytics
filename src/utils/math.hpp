#pragma once

#include <vector>
#include <span>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <numbers>
#include <concepts>

namespace bigbrother::utils::math {

// Concepts for generic numeric operations
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::floating_point<T>;

// Statistical functions using C++23 ranges

/**
 * Calculate mean of a range
 * @param range Input range (uses move semantics for efficiency)
 * @return Mean value
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto mean(R&& range) noexcept -> double {
    if (std::ranges::empty(range)) {
        return 0.0;
    }

    auto const sum = std::ranges::fold_left(
        std::forward<R>(range), 0.0,
        [](auto acc, auto val) { return acc + static_cast<double>(val); }
    );

    return sum / static_cast<double>(std::ranges::size(range));
}

/**
 * Calculate variance
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto variance(R&& range) noexcept -> double {
    if (std::ranges::size(range) < 2) {
        return 0.0;
    }

    auto const mu = mean(range);

    auto const sum_sq_diff = std::ranges::fold_left(
        std::forward<R>(range) | std::views::transform([mu](auto val) {
            auto const diff = static_cast<double>(val) - mu;
            return diff * diff;
        }),
        0.0,
        std::plus<>{}
    );

    return sum_sq_diff / static_cast<double>(std::ranges::size(range) - 1);
}

/**
 * Calculate standard deviation
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto stddev(R&& range) noexcept -> double {
    return std::sqrt(variance(std::forward<R>(range)));
}

/**
 * Calculate covariance between two ranges
 */
template<std::ranges::range R1, std::ranges::range R2>
    requires Numeric<std::ranges::range_value_t<R1>> &&
             Numeric<std::ranges::range_value_t<R2>>
[[nodiscard]] constexpr auto covariance(R1&& range1, R2&& range2) noexcept -> double {
    auto const n = std::min(std::ranges::size(range1), std::ranges::size(range2));

    if (n < 2) {
        return 0.0;
    }

    auto const mean1 = mean(range1);
    auto const mean2 = mean(range2);

    auto const sum = std::ranges::fold_left(
        std::views::zip(std::forward<R1>(range1), std::forward<R2>(range2)) |
        std::views::take(n) |
        std::views::transform([mean1, mean2](auto const& pair) {
            auto const [v1, v2] = pair;
            return (static_cast<double>(v1) - mean1) *
                   (static_cast<double>(v2) - mean2);
        }),
        0.0,
        std::plus<>{}
    );

    return sum / static_cast<double>(n - 1);
}

/**
 * Calculate Pearson correlation coefficient
 */
template<std::ranges::range R1, std::ranges::range R2>
    requires Numeric<std::ranges::range_value_t<R1>> &&
             Numeric<std::ranges::range_value_t<R2>>
[[nodiscard]] constexpr auto correlation(R1&& range1, R2&& range2) noexcept -> double {
    auto const cov = covariance(range1, range2);
    auto const std1 = stddev(std::forward<R1>(range1));
    auto const std2 = stddev(std::forward<R2>(range2));

    if (std1 == 0.0 || std2 == 0.0) {
        return 0.0;
    }

    return cov / (std1 * std2);
}

/**
 * Calculate percentile (uses COW for sorted data)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto percentile(R&& range, double p) -> double {
    if (std::ranges::empty(range)) {
        return 0.0;
    }

    // Copy-on-write: only copy if we need to sort
    std::vector<std::ranges::range_value_t<R>> sorted;
    sorted.reserve(std::ranges::size(range));
    std::ranges::copy(std::forward<R>(range), std::back_inserter(sorted));
    std::ranges::sort(sorted);

    auto const index = p * static_cast<double>(sorted.size() - 1);
    auto const lower_idx = static_cast<size_t>(std::floor(index));
    auto const upper_idx = static_cast<size_t>(std::ceil(index));

    if (lower_idx == upper_idx) {
        return static_cast<double>(sorted[lower_idx]);
    }

    // Linear interpolation
    auto const weight = index - static_cast<double>(lower_idx);
    return static_cast<double>(sorted[lower_idx]) * (1.0 - weight) +
           static_cast<double>(sorted[upper_idx]) * weight;
}

/**
 * Calculate rolling window statistics (uses views for efficiency)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto rolling_mean(R&& range, size_t window_size)
    -> std::vector<double> {

    std::vector<double> result;
    result.reserve(std::ranges::size(range) - window_size + 1);

    for (auto window : std::forward<R>(range) | std::views::slide(window_size)) {
        result.push_back(mean(window));
    }

    return result;  // NRVO (Named Return Value Optimization)
}

/**
 * Calculate exponential moving average
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto exponential_moving_average(R&& range, double alpha)
    -> std::vector<double> {

    std::vector<double> result;
    result.reserve(std::ranges::size(range));

    double ema = 0.0;
    bool first = true;

    for (auto val : std::forward<R>(range)) {
        if (first) {
            ema = static_cast<double>(val);
            first = false;
        } else {
            ema = alpha * static_cast<double>(val) + (1.0 - alpha) * ema;
        }
        result.push_back(ema);
    }

    return result;  // Move semantics applied automatically
}

// Financial math functions

/**
 * Calculate log returns
 */
template<std::ranges::range R>
    requires FloatingPoint<std::ranges::range_value_t<R>>
[[nodiscard]] auto log_returns(R&& range) -> std::vector<double> {
    if (std::ranges::size(range) < 2) {
        return {};
    }

    auto adjacent_pairs = std::forward<R>(range) | std::views::adjacent<2>;

    std::vector<double> returns;
    returns.reserve(std::ranges::size(range) - 1);

    for (auto [prev, curr] : adjacent_pairs) {
        if (prev > 0.0) {
            returns.push_back(std::log(curr / prev));
        }
    }

    return returns;
}

/**
 * Calculate simple returns
 */
template<std::ranges::range R>
    requires FloatingPoint<std::ranges::range_value_t<R>>
[[nodiscard]] auto simple_returns(R&& range) -> std::vector<double> {
    if (std::ranges::size(range) < 2) {
        return {};
    }

    auto adjacent_pairs = std::forward<R>(range) | std::views::adjacent<2>;

    std::vector<double> returns;
    returns.reserve(std::ranges::size(range) - 1);

    for (auto [prev, curr] : adjacent_pairs) {
        if (prev != 0.0) {
            returns.push_back((curr - prev) / prev);
        }
    }

    return returns;
}

/**
 * Calculate Sharpe ratio
 */
template<std::ranges::range R>
    requires FloatingPoint<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto sharpe_ratio(R&& returns, double risk_free_rate = 0.0)
    noexcept -> double {

    if (std::ranges::size(returns) < 2) {
        return 0.0;
    }

    auto const mu = mean(returns);
    auto const sigma = stddev(std::forward<R>(returns));

    if (sigma == 0.0) {
        return 0.0;
    }

    return (mu - risk_free_rate) / sigma * std::sqrt(252.0);  // Annualized
}

/**
 * Calculate maximum drawdown
 */
template<std::ranges::range R>
    requires FloatingPoint<std::ranges::range_value_t<R>>
[[nodiscard]] auto max_drawdown(R&& prices) -> double {
    if (std::ranges::empty(prices)) {
        return 0.0;
    }

    double peak = -std::numeric_limits<double>::infinity();
    double max_dd = 0.0;

    for (auto price : std::forward<R>(prices)) {
        peak = std::max(peak, price);
        auto const drawdown = (peak - price) / peak;
        max_dd = std::max(max_dd, drawdown);
    }

    return max_dd;
}

/**
 * Normalize data to [0, 1] range (uses move semantics)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto normalize(R&& range) -> std::vector<double> {
    if (std::ranges::empty(range)) {
        return {};
    }

    auto const [min_it, max_it] = std::ranges::minmax_element(range);
    auto const min_val = static_cast<double>(*min_it);
    auto const max_val = static_cast<double>(*max_it);
    auto const range_val = max_val - min_val;

    if (range_val == 0.0) {
        return std::vector<double>(std::ranges::size(range), 0.5);
    }

    return std::forward<R>(range) |
           std::views::transform([min_val, range_val](auto val) {
               return (static_cast<double>(val) - min_val) / range_val;
           }) |
           std::ranges::to<std::vector>();
}

/**
 * Z-score standardization (uses move semantics)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto standardize(R&& range) -> std::vector<double> {
    if (std::ranges::size(range) < 2) {
        return {};
    }

    auto const mu = mean(range);
    auto const sigma = stddev(range);

    if (sigma == 0.0) {
        return std::vector<double>(std::ranges::size(range), 0.0);
    }

    return std::forward<R>(range) |
           std::views::transform([mu, sigma](auto val) {
               return (static_cast<double>(val) - mu) / sigma;
           }) |
           std::ranges::to<std::vector>();
}

// Cumulative functions using ranges

/**
 * Cumulative sum (uses move semantics)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto cumsum(R&& range) -> std::vector<double> {
    std::vector<double> result;
    result.reserve(std::ranges::size(range));

    double sum = 0.0;
    for (auto val : std::forward<R>(range)) {
        sum += static_cast<double>(val);
        result.push_back(sum);
    }

    return result;
}

/**
 * Cumulative product (uses move semantics)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] auto cumprod(R&& range) -> std::vector<double> {
    std::vector<double> result;
    result.reserve(std::ranges::size(range));

    double prod = 1.0;
    for (auto val : std::forward<R>(range)) {
        prod *= static_cast<double>(val);
        result.push_back(prod);
    }

    return result;
}

// Linear algebra utilities

/**
 * Dot product using ranges
 */
template<std::ranges::range R1, std::ranges::range R2>
    requires Numeric<std::ranges::range_value_t<R1>> &&
             Numeric<std::ranges::range_value_t<R2>>
[[nodiscard]] constexpr auto dot_product(R1&& range1, R2&& range2) noexcept -> double {
    return std::ranges::fold_left(
        std::views::zip_transform(
            [](auto a, auto b) {
                return static_cast<double>(a) * static_cast<double>(b);
            },
            std::forward<R1>(range1),
            std::forward<R2>(range2)
        ),
        0.0,
        std::plus<>{}
    );
}

/**
 * Vector magnitude (L2 norm)
 */
template<std::ranges::range R>
    requires Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto magnitude(R&& range) noexcept -> double {
    return std::sqrt(dot_product(range, range));
}

/**
 * Cosine similarity
 */
template<std::ranges::range R1, std::ranges::range R2>
    requires Numeric<std::ranges::range_value_t<R1>> &&
             Numeric<std::ranges::range_value_t<R2>>
[[nodiscard]] constexpr auto cosine_similarity(R1&& range1, R2&& range2)
    noexcept -> double {

    auto const dot = dot_product(range1, range2);
    auto const mag1 = magnitude(range1);
    auto const mag2 = magnitude(std::forward<R2>(range2));

    if (mag1 == 0.0 || mag2 == 0.0) {
        return 0.0;
    }

    return dot / (mag1 * mag2);
}

} // namespace bigbrother::utils::math
