/**
 * BigBrotherAnalytics - Stop Loss Manager Module (C++23)
 *
 * Fluent API for intelligent stop loss management with multiple strategies.
 * Protects capital through hard, trailing, time-based, and volatility stops.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-13
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax (ES.20)
 * - Fluent API for method chaining
 * - [[nodiscard]] for pure functions
 * - Thread-safe operations with mutex
 * - noexcept for performance-critical paths
 */

// Global module fragment
module;

#include <algorithm>
#include <chrono>
#include <format>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.risk.stop_loss;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.timer;

export namespace bigbrother::risk {

using namespace bigbrother::types;
using bigbrother::utils::Logger;


// ============================================================================
// Stop Loss Types
// ============================================================================

enum class StopType {
    Hard,              // Fixed price stop
    Trailing,          // Stop that trails price movement
    TimeStop,          // Exit after time limit
    VolatilityStop,    // Stop based on volatility expansion
    Greeks             // Stop based on option Greeks deterioration
};

// ============================================================================
// Stop Loss Configuration
// ============================================================================

struct Stop {
    std::string position_id;
    StopType type{StopType::Hard};
    Price trigger_price{0.0};
    Price initial_price{0.0};
    double trail_amount{0.0};      // For trailing stops or volatility threshold
    int64_t expiration{0};         // For time stops (timestamp)
    bool triggered{false};

    [[nodiscard]] auto isTriggered(Price current_price) const noexcept -> bool;
};

// ============================================================================
// Stop Loss Manager - Fluent API
// ============================================================================

class StopLossManager {
public:
    // Factory method
    [[nodiscard]] static auto create() noexcept -> StopLossManager {
        return StopLossManager{};
    }

    // Fluent API - Add various stop types
    [[nodiscard]] auto addHardStop(
        std::string position_id,
        Price trigger_price,
        Price initial_price
    ) noexcept -> StopLossManager& {
        addStopInternal(
            std::move(position_id),
            StopType::Hard,
            trigger_price,
            initial_price,
            0.0
        );
        return *this;
    }

    [[nodiscard]] auto addTrailingStop(
        std::string position_id,
        Price trigger_price,
        Price initial_price,
        double trail_amount
    ) noexcept -> StopLossManager& {
        addStopInternal(
            std::move(position_id),
            StopType::Trailing,
            trigger_price,
            initial_price,
            trail_amount
        );
        return *this;
    }

    [[nodiscard]] auto addTimeStop(
        std::string position_id,
        Price trigger_price,
        Price initial_price,
        int64_t expiration_time
    ) noexcept -> StopLossManager& {
        std::lock_guard lock{mutex_};

        // Remove existing stop for this position
        removeStopInternal(position_id);

        // Add new time stop
        Stop stop{
            .position_id = std::move(position_id),
            .type = StopType::TimeStop,
            .trigger_price = trigger_price,
            .initial_price = initial_price,
            .trail_amount = 0.0,
            .expiration = expiration_time,
            .triggered = false
        };

        stops_.push_back(std::move(stop));

        Logger::getInstance().info(
            "Added time stop for {}: expiration={}",
            stop.position_id,
            expiration_time
        );

        return *this;
    }

    [[nodiscard]] auto addVolatilityStop(
        std::string position_id,
        Price initial_price,
        double volatility_threshold
    ) noexcept -> StopLossManager& {
        addStopInternal(
            std::move(position_id),
            StopType::VolatilityStop,
            0.0,  // Not used for volatility stops
            initial_price,
            volatility_threshold
        );
        return *this;
    }

    [[nodiscard]] auto addGreeksStop(
        std::string position_id,
        Price trigger_price,
        Price initial_price
    ) noexcept -> StopLossManager& {
        addStopInternal(
            std::move(position_id),
            StopType::Greeks,
            trigger_price,
            initial_price,
            0.0
        );
        return *this;
    }

    // Generic add stop (for advanced use)
    [[nodiscard]] auto addStop(
        std::string position_id,
        StopType type,
        Price trigger_price,
        Price initial_price,
        double trail_amount = 0.0
    ) noexcept -> StopLossManager& {
        addStopInternal(
            std::move(position_id),
            type,
            trigger_price,
            initial_price,
            trail_amount
        );
        return *this;
    }

    // Remove stop (chainable)
    [[nodiscard]] auto removeStop(std::string const& position_id) noexcept -> StopLossManager& {
        std::lock_guard lock{mutex_};
        removeStopInternal(position_id);
        return *this;
    }

    // Update all stops and return triggered positions
    [[nodiscard]] auto update(
        std::unordered_map<std::string, Price> const& current_prices
    ) noexcept -> std::vector<std::string> {

        std::lock_guard lock{mutex_};

        std::vector<std::string> triggered_positions;

        for (auto& stop : stops_) {
            if (stop.triggered) {
                continue;  // Already triggered
            }

            // Find current price for this position
            auto it = current_prices.find(stop.position_id);
            if (it == current_prices.end()) {
                continue;  // No price update for this position
            }

            Price const current_price = it->second;

            // Check if stop is triggered
            if (stop.isTriggered(current_price)) {
                stop.triggered = true;
                triggered_positions.push_back(stop.position_id);

                Logger::getInstance().warn(
                    "STOP LOSS TRIGGERED: {} at ${:.2f} (stop: ${:.2f})",
                    stop.position_id,
                    current_price,
                    stop.trigger_price
                );
            }

            // Update trailing stops
            if (stop.type == StopType::Trailing && !stop.triggered) {
                updateTrailingStop(stop, current_price);
            }
        }

        return triggered_positions;
    }

    // Query methods
    [[nodiscard]] auto getActiveStops() const noexcept -> std::vector<Stop> {
        std::lock_guard lock{mutex_};
        return stops_;
    }

    [[nodiscard]] auto getStopCount() const noexcept -> size_t {
        std::lock_guard lock{mutex_};
        return stops_.size();
    }

    [[nodiscard]] auto getTriggeredCount() const noexcept -> size_t {
        std::lock_guard lock{mutex_};
        return std::count_if(stops_.begin(), stops_.end(),
            [](Stop const& s) { return s.triggered; }
        );
    }

    [[nodiscard]] auto hasStop(std::string const& position_id) const noexcept -> bool {
        std::lock_guard lock{mutex_};
        return std::any_of(stops_.begin(), stops_.end(),
            [&](Stop const& s) { return s.position_id == position_id; }
        );
    }

    // Clear all stops (chainable)
    [[nodiscard]] auto clearAll() noexcept -> StopLossManager& {
        std::lock_guard lock{mutex_};
        stops_.clear();
        Logger::getInstance().info("Cleared all stops");
        return *this;
    }

    // Clear only triggered stops (chainable)
    [[nodiscard]] auto clearTriggered() noexcept -> StopLossManager& {
        std::lock_guard lock{mutex_};
        stops_.erase(
            std::remove_if(stops_.begin(), stops_.end(),
                [](Stop const& s) { return s.triggered; }),
            stops_.end()
        );
        Logger::getInstance().debug("Cleared triggered stops");
        return *this;
    }

private:
    StopLossManager() = default;

    mutable std::mutex mutex_;
    std::vector<Stop> stops_;

    // Internal helper methods
    auto addStopInternal(
        std::string position_id,
        StopType type,
        Price trigger_price,
        Price initial_price,
        double trail_amount
    ) noexcept -> void {
        std::lock_guard lock{mutex_};

        // Remove existing stop for this position
        removeStopInternal(position_id);

        // Add new stop
        Stop stop{
            .position_id = position_id,
            .type = type,
            .trigger_price = trigger_price,
            .initial_price = initial_price,
            .trail_amount = trail_amount,
            .expiration = 0,
            .triggered = false
        };

        stops_.push_back(std::move(stop));

        Logger::getInstance().info(
            "Added {} stop for {}: trigger=${:.2f}",
            stopTypeToString(type),
            position_id,
            trigger_price
        );
    }

    auto removeStopInternal(std::string const& position_id) noexcept -> void {
        // Must be called with mutex locked
        stops_.erase(
            std::remove_if(stops_.begin(), stops_.end(),
                [&](Stop const& s) { return s.position_id == position_id; }),
            stops_.end()
        );
    }

    auto updateTrailingStop(Stop& stop, Price current_price) noexcept -> void {
        // Must be called with mutex locked
        // Adjust trigger price if beneficial
        if (stop.initial_price > stop.trigger_price) {
            // Long position - raise stop if price increases
            double const new_stop = current_price - stop.trail_amount;
            if (new_stop > stop.trigger_price) {
                Logger::getInstance().debug(
                    "Trailing stop updated for {}: ${:.2f} -> ${:.2f}",
                    stop.position_id,
                    stop.trigger_price,
                    new_stop
                );
                stop.trigger_price = new_stop;
            }
        } else {
            // Short position - lower stop if price decreases
            double const new_stop = current_price + stop.trail_amount;
            if (new_stop < stop.trigger_price) {
                Logger::getInstance().debug(
                    "Trailing stop updated for {}: ${:.2f} -> ${:.2f}",
                    stop.position_id,
                    stop.trigger_price,
                    new_stop
                );
                stop.trigger_price = new_stop;
            }
        }
    }

    [[nodiscard]] static auto stopTypeToString(StopType type) noexcept -> char const* {
        switch (type) {
            case StopType::Hard: return "hard";
            case StopType::Trailing: return "trailing";
            case StopType::TimeStop: return "time";
            case StopType::VolatilityStop: return "volatility";
            case StopType::Greeks: return "greeks";
            default: return "unknown";
        }
    }
};

// ============================================================================
// Stop::isTriggered Implementation
// ============================================================================

[[nodiscard]] auto Stop::isTriggered(Price current_price) const noexcept -> bool {
    if (triggered) {
        return true;  // Already triggered
    }

    switch (type) {
        case StopType::Hard: {
            // Simple price-based stop
            // Trigger if price crosses threshold
            if (initial_price > trigger_price) {
                // Long position - stop if price falls below
                return current_price <= trigger_price;
            } else {
                // Short position - stop if price rises above
                return current_price >= trigger_price;
            }
        }

        case StopType::Trailing: {
            // Trailing stop follows price
            // For long: stop trails below price by trail_amount
            // For short: stop trails above price by trail_amount

            if (initial_price > trigger_price) {
                // Long position
                double const trailing_stop = current_price - trail_amount;
                return current_price <= trailing_stop;
            } else {
                // Short position
                double const trailing_stop = current_price + trail_amount;
                return current_price >= trailing_stop;
            }
        }

        case StopType::TimeStop: {
            // Exit after time limit
            auto const now = utils::Timer::now();
            return now >= expiration;
        }

        case StopType::VolatilityStop: {
            // Stop based on volatility expansion
            // trail_amount represents volatility threshold
            double const price_change_pct = std::abs(
                (current_price - initial_price) / initial_price
            );
            return price_change_pct > trail_amount;
        }

        case StopType::Greeks: {
            // Stop based on Greeks deterioration
            // (Implementation would need Greeks data)
            // For now, treat like hard stop
            return current_price <= trigger_price;
        }

        default:
            return false;
    }
}

} // namespace bigbrother::risk
