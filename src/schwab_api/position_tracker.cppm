/**
 * BigBrotherAnalytics - Position Tracker Module (C++23)
 *
 * Automatic position tracking with real-time monitoring:
 * - 30-second refresh interval
 * - Position change detection (new, closed, quantity changes)
 * - DuckDB persistence with historical tracking
 * - P/L calculations and portfolio analytics
 * - Thread-safe background updates
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 */

module;

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// DuckDB bridge - isolates DuckDB incomplete types from C++23 modules
#include "schwab_api/duckdb_bridge.hpp"

export module position_tracker;

import account_manager;
import account_types;

namespace bigbrother::schwab {

/**
 * Position Tracker - Automatic position monitoring
 *
 * Features:
 * - Polls positions every 30 seconds (configurable)
 * - Persists to DuckDB for historical tracking
 * - Detects position changes (entries/exits/quantity changes)
 * - Calculates real-time P/L and portfolio metrics
 * - Thread-safe background updates
 */
export class PositionTracker {
  public:
    /**
     * Constructor
     *
     * @param account_mgr AccountManager for fetching positions
     * @param db_path Path to DuckDB database
     * @param refresh_interval_seconds Refresh interval (default 30s)
     */
    explicit PositionTracker(std::shared_ptr<AccountManager> account_mgr, std::string db_path,
                             int refresh_interval_seconds = 30)
        : account_mgr_{std::move(account_mgr)}, db_path_{std::move(db_path)},
          refresh_interval_{refresh_interval_seconds}, running_{false}, paused_{false},
          update_count_{0} {

        // Open DuckDB connection using bridge library
        db_ = duckdb_bridge::openDatabase(db_path_);
        conn_ = duckdb_bridge::createConnection(*db_);

        // Logger::getInstance().info("PositionTracker initialized (refresh interval: {}s)", refresh_interval_);
    }

    // Rule of Five - non-copyable, non-movable due to thread
    PositionTracker(PositionTracker const&) = delete;
    auto operator=(PositionTracker const&) -> PositionTracker& = delete;
    PositionTracker(PositionTracker&&) noexcept = delete;
    auto operator=(PositionTracker&&) noexcept -> PositionTracker& = delete;

    ~PositionTracker() {
        stop();
        // Bridge handles clean up automatically (RAII)
    }

    // ========================================================================
    // Control Methods
    // ========================================================================

    /**
     * Start automatic position tracking
     *
     * @param account_id Account to track
     */
    auto start(std::string account_id) -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        if (running_) {
            // Logger::getInstance().warn("PositionTracker already running");
            return;
        }

        account_id_ = std::move(account_id);
        running_ = true;
        paused_ = false;
        update_count_ = 0;

        // Logger::getInstance().info("Starting PositionTracker for account: {}", account_id_);

        // Start background thread
        tracking_thread_ = std::thread([this]() -> void { trackingLoop(); });
    }

    /**
     * Stop automatic position tracking
     */
    auto stop() -> void {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_)
                return;

            // Logger::getInstance().info("Stopping PositionTracker (total updates: {})", update_count_.load());
            running_ = false;
        }

        cv_.notify_all();

        if (tracking_thread_.joinable()) {
            tracking_thread_.join();
        }
    }

    /**
     * Pause tracking (without stopping thread)
     */
    auto pause() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = true;
        // Logger::getInstance().info("PositionTracker paused");
    }

    /**
     * Resume tracking
     */
    auto resume() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = false;
        cv_.notify_one();
        // Logger::getInstance().info("PositionTracker resumed");
    }

    /**
     * Force immediate position refresh
     */
    auto refreshNow() -> void {
        // Logger::getInstance().info("Forcing immediate position refresh");
        cv_.notify_one();
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    /**
     * Get current positions from cache
     */
    [[nodiscard]] auto getCurrentPositions() const -> std::vector<Position> {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_positions_;
    }

    /**
     * Get position for specific symbol
     */
    [[nodiscard]] auto getPosition(std::string const& symbol) const -> std::optional<Position> {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = std::find_if(cached_positions_.begin(), cached_positions_.end(),
                               [&symbol](auto const& pos) -> bool { return pos.symbol == symbol; });

        if (it != cached_positions_.end()) {
            return *it;
        }

        return std::nullopt;
    }

    /**
     * Get position count
     */
    [[nodiscard]] auto getPositionCount() const noexcept -> size_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_positions_.size();
    }

    /**
     * Check if tracking is running
     */
    [[nodiscard]] auto isRunning() const noexcept -> bool { return running_; }

    /**
     * Check if tracking is paused
     */
    [[nodiscard]] auto isPaused() const noexcept -> bool { return paused_; }

    /**
     * Get last update timestamp
     */
    [[nodiscard]] auto getLastUpdateTime() const noexcept -> Timestamp {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_update_timestamp_;
    }

    /**
     * Get total update count
     */
    [[nodiscard]] auto getUpdateCount() const noexcept -> int { return update_count_; }

    // ========================================================================
    // Portfolio Analytics
    // ========================================================================

    /**
     * Calculate total unrealized P/L across all positions
     */
    [[nodiscard]] auto calculateTotalPnL() const -> double {
        std::lock_guard<std::mutex> lock(mutex_);

        double total_pnl = 0.0;
        for (auto const& pos : cached_positions_) {
            total_pnl += pos.unrealized_pnl;
        }

        return total_pnl;
    }

    /**
     * Calculate total day P/L across all positions
     */
    [[nodiscard]] auto calculateDayPnL() const -> double {
        std::lock_guard<std::mutex> lock(mutex_);

        double day_pnl = 0.0;
        for (auto const& pos : cached_positions_) {
            day_pnl += pos.day_pnl;
        }

        return day_pnl;
    }

    /**
     * Calculate portfolio heat (total position value / total equity)
     *
     * @param total_equity Total account equity
     * @return Portfolio heat percentage
     */
    [[nodiscard]] auto calculatePortfolioHeat(double total_equity) const -> double {
        if (total_equity <= 0.0)
            return 0.0;

        std::lock_guard<std::mutex> lock(mutex_);

        double total_position_value = 0.0;
        for (auto const& pos : cached_positions_) {
            total_position_value += std::abs(pos.market_value);
        }

        return (total_position_value / total_equity) * 100.0;
    }

  private:
    // ========================================================================
    // Tracking Loop
    // ========================================================================

    auto trackingLoop() -> void {
        // Logger::getInstance().info("PositionTracker thread started ({}s refresh)", refresh_interval_);

        // Perform initial update immediately
        updatePositions();

        while (running_) {
            // Wait for refresh interval or stop signal
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::seconds(refresh_interval_),
                         [this]() -> bool { return !running_ || !paused_; });

            lock.unlock();

            if (!running_)
                break;

            if (!paused_) {
                updatePositions();
            }
        }

        // Logger::getInstance().info("PositionTracker thread stopped");
    }

    // ========================================================================
    // Position Update Logic
    // ========================================================================

    auto updatePositions() -> void {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Logger::getInstance().debug("Fetching positions from Schwab API (update #{})",
        //                            update_count_.load(std::memory_order_relaxed) + 1);

        // Fetch current positions from Schwab API
        auto positions_result = account_mgr_->getPositions(account_id_);

        if (!positions_result) {
            // Logger::getInstance().error("Failed to fetch positions: {}", positions_result.error());
            return;
        }

        auto const& new_positions = *positions_result;

        // Detect changes (compare with cached positions)
        detectPositionChanges(new_positions);

        // Update cache
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cached_positions_ = new_positions;
            last_update_timestamp_ = getCurrentTimestamp();
        }

        // Persist to DuckDB
        persistPositions(new_positions);
        persistPositionHistory(new_positions);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        auto const current_update = update_count_.fetch_add(1, std::memory_order_relaxed) + 1;

        // Logger::getInstance().info("Position update #{} complete: {} positions (took {}ms)",
        //                           current_update, new_positions.size(), duration.count());

        // Log P/L summary
        logPnLSummary(new_positions);
    }

    // ========================================================================
    // Change Detection
    // ========================================================================

    auto detectPositionChanges(std::vector<Position> const& new_positions) -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        // Build maps for comparison
        std::unordered_map<std::string, Position> old_map;
        for (auto const& pos : cached_positions_) {
            old_map[pos.symbol] = pos;
        }

        std::unordered_map<std::string, Position> new_map;
        for (auto const& pos : new_positions) {
            new_map[pos.symbol] = pos;
        }

        // Detect NEW positions (OPENED)
        for (auto const& [symbol, pos] : new_map) {
            if (old_map.find(symbol) == old_map.end()) {
                // Logger::getInstance().info("NEW POSITION OPENED: {} ({} shares @ ${:.2f})",
                //                           symbol, pos.quantity, pos.average_cost);
                recordPositionChange("OPENED", pos, std::nullopt);
            }
        }

        // Detect CLOSED positions
        for (auto const& [symbol, pos] : old_map) {
            if (new_map.find(symbol) == new_map.end()) {
                // Logger::getInstance().warn("POSITION CLOSED: {} (was {} shares)", symbol, pos.quantity);
                recordPositionChange("CLOSED", pos, std::nullopt);
            }
        }

        // Detect quantity CHANGES
        for (auto const& [symbol, new_pos] : new_map) {
            if (old_map.find(symbol) != old_map.end()) {
                auto const& old_pos = old_map[symbol];

                if (std::abs(new_pos.quantity - old_pos.quantity) > quantity_epsilon) {
                    std::string change_type =
                        (new_pos.quantity > old_pos.quantity) ? "INCREASED" : "DECREASED";

                    // Logger::getInstance().info("POSITION {}: {} ({} -> {} shares, change: {})",
                    //                           change_type, symbol, old_pos.quantity,
                    //                           new_pos.quantity, new_pos.quantity - old_pos.quantity);

                    recordPositionChange(change_type, new_pos, old_pos);
                }
            }
        }
    }

    auto recordPositionChange(std::string const& change_type, Position const& new_pos,
                              std::optional<Position> const& old_pos) -> void {
        try {
            // Note: Using executeQuery for now. TODO: Enhance bridge for proper prepared statements
            std::string query =
                "INSERT INTO position_changes ("
                "account_id, symbol, change_type, "
                "quantity_before, quantity_after, quantity_change, "
                "average_cost_before, average_cost_after, "
                "price_at_change, timestamp) VALUES ('" +
                new_pos.account_id + "', '" + new_pos.symbol + "', '" + change_type + "', " +
                std::to_string(old_pos ? old_pos->quantity : 0.0) + ", " +
                std::to_string(new_pos.quantity) + ", " +
                std::to_string(new_pos.quantity - (old_pos ? old_pos->quantity : 0.0)) + ", " +
                std::to_string(old_pos ? old_pos->average_cost : 0.0) + ", " +
                std::to_string(new_pos.average_cost) + ", " +
                std::to_string(new_pos.current_price) + ", CURRENT_TIMESTAMP)";

            duckdb_bridge::executeQuery(*conn_, query);

        } catch (std::exception const& e) {
            // Logger::getInstance().error("Failed to record position change: {}", e.what());
        }
    }

    // ========================================================================
    // DuckDB Persistence
    // ========================================================================

    auto persistPositions(std::vector<Position> const& positions) -> void {
        try {
            // Begin transaction for atomic update
            duckdb_bridge::executeQuery(*conn_, "BEGIN TRANSACTION");

            // Delete existing positions for this account
            std::string delete_query = "DELETE FROM positions WHERE account_id = '" + account_id_ + "'";
            duckdb_bridge::executeQuery(*conn_, delete_query);

            // Insert current positions
            for (auto const& pos : positions) {
                insertPosition(pos);
            }

            // Commit transaction
            duckdb_bridge::executeQuery(*conn_, "COMMIT");

            // Logger::getInstance().debug("Persisted {} positions to database", positions.size());

        } catch (std::exception const& e) {
            duckdb_bridge::executeQuery(*conn_, "ROLLBACK");
            // Logger::getInstance().error("Database persistence error: {}", e.what());
        }
    }

    auto insertPosition(Position const& pos) -> void {
        // Note: Using executeQuery for now. TODO: Enhance bridge for proper prepared statements
        std::string query =
            "INSERT INTO positions ("
            "account_id, symbol, asset_type, cusip, "
            "quantity, long_quantity, short_quantity, "
            "average_cost, current_price, market_value, cost_basis, "
            "unrealized_pnl, unrealized_pnl_percent, "
            "day_pnl, day_pnl_percent, previous_close, "
            "is_bot_managed, managed_by, opened_by, bot_strategy, opened_at, "
            "updated_at) VALUES ('" +
            pos.account_id + "', '" + pos.symbol + "', '" + pos.asset_type + "', '" + pos.cusip + "', " +
            std::to_string(pos.quantity) + ", " +
            std::to_string(pos.long_quantity) + ", " +
            std::to_string(pos.short_quantity) + ", " +
            std::to_string(pos.average_cost) + ", " +
            std::to_string(pos.current_price) + ", " +
            std::to_string(pos.market_value) + ", " +
            std::to_string(pos.cost_basis) + ", " +
            std::to_string(pos.unrealized_pnl) + ", " +
            std::to_string(pos.unrealized_pnl_percent) + ", " +
            std::to_string(pos.day_pnl) + ", " +
            std::to_string(pos.day_pnl_percent) + ", " +
            std::to_string(pos.previous_close) + ", " +
            std::to_string(pos.is_bot_managed) + ", '" +
            pos.managed_by + "', '" + pos.opened_by + "', '" + pos.bot_strategy + "', " +
            std::to_string(pos.opened_at) + ", CURRENT_TIMESTAMP)";

        duckdb_bridge::executeQuery(*conn_, query);
    }

    auto persistPositionHistory(std::vector<Position> const& positions) -> void {
        try {
            for (auto const& pos : positions) {
                // Note: Using executeQuery for now. TODO: Enhance bridge for proper prepared statements
                std::string query =
                    "INSERT INTO position_history ("
                    "account_id, symbol, asset_type, quantity, "
                    "average_cost, current_price, market_value, cost_basis, "
                    "unrealized_pnl, unrealized_pnl_percent, day_pnl, "
                    "timestamp) VALUES ('" +
                    pos.account_id + "', '" + pos.symbol + "', '" + pos.asset_type + "', " +
                    std::to_string(pos.quantity) + ", " +
                    std::to_string(pos.average_cost) + ", " +
                    std::to_string(pos.current_price) + ", " +
                    std::to_string(pos.market_value) + ", " +
                    std::to_string(pos.cost_basis) + ", " +
                    std::to_string(pos.unrealized_pnl) + ", " +
                    std::to_string(pos.unrealized_pnl_percent) + ", " +
                    std::to_string(pos.day_pnl) + ", CURRENT_TIMESTAMP)";

                duckdb_bridge::executeQuery(*conn_, query);
            }
        } catch (std::exception const& e) {
            // Logger::getInstance().error("Failed to persist position history: {}", e.what());
        }
    }

    // ========================================================================
    // Logging
    // ========================================================================

    auto logPnLSummary([[maybe_unused]] std::vector<Position> const& positions) const -> void {
        double total_market_value = 0.0;
        double total_cost_basis = 0.0;
        double total_unrealized_pnl = 0.0;
        double total_day_pnl = 0.0;

        int manual_count = 0;
        int bot_count = 0;

        for (auto const& pos : positions) {
            total_market_value += pos.market_value;
            total_cost_basis += pos.cost_basis;
            total_unrealized_pnl += pos.unrealized_pnl;
            total_day_pnl += pos.day_pnl;

            if (pos.isBotManaged()) {
                bot_count++;
            } else {
                manual_count++;
            }
        }

        double total_pnl_percent = 0.0;
        if (total_cost_basis > 0.0) {
            total_pnl_percent = (total_unrealized_pnl / total_cost_basis) * 100.0;
        }

        // Logger::getInstance().info("=== PORTFOLIO SUMMARY ===");
        // Logger::getInstance().info("  Positions: {} (Manual: {}, Bot: {})", positions.size(),
        //                           manual_count, bot_count);
        // Logger::getInstance().info("  Market Value: ${:.2f}", total_market_value);
        // Logger::getInstance().info("  Cost Basis: ${:.2f}", total_cost_basis);
        // Logger::getInstance().info("  Unrealized P/L: ${:.2f} ({:.2f}%)", total_unrealized_pnl,
        //                           total_pnl_percent);
        // Logger::getInstance().info("  Day P/L: ${:.2f}", total_day_pnl);
        // Logger::getInstance().info("========================");

        // Suppress unused variable warnings
        (void)total_market_value;
        (void)total_cost_basis;
        (void)total_unrealized_pnl;
        (void)total_day_pnl;
        (void)manual_count;
        (void)bot_count;
        (void)total_pnl_percent;
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    [[nodiscard]] auto getCurrentTimestamp() const noexcept -> Timestamp {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }

    // ========================================================================
    // Member Variables
    // ========================================================================

    std::shared_ptr<AccountManager> account_mgr_;
    std::string db_path_;
    std::string account_id_;
    int refresh_interval_;

    // DuckDB bridge handles - clean interface, no incomplete types
    std::unique_ptr<duckdb_bridge::DatabaseHandle> db_;
    std::unique_ptr<duckdb_bridge::ConnectionHandle> conn_;

    // Threading
    std::thread tracking_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    std::atomic<int> update_count_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Cache
    std::vector<Position> cached_positions_;
    Timestamp last_update_timestamp_{0};

    // Constants
    static constexpr double quantity_epsilon = 0.0001;
};

} // namespace bigbrother::schwab
