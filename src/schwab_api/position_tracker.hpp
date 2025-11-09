/**
 * BigBrotherAnalytics - Position Tracker (C++23)
 *
 * Automatic position tracking with:
 * - Real-time position updates (30-second refresh)
 * - DuckDB persistence
 * - Position change detection
 * - P/L calculation
 * - Thread-safe operations
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#pragma once

#include "account_types.hpp"
#include "account_manager.hpp"
#include <duckdb.hpp>
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <unordered_map>

namespace bigbrother::schwab {

/**
 * Position Tracker - Automatic position monitoring
 *
 * Features:
 * - Polls positions every 30 seconds
 * - Persists to DuckDB for historical tracking
 * - Detects position changes (entries/exits)
 * - Calculates real-time P/L
 * - Thread-safe background updates
 */
class PositionTracker {
public:
    /**
     * Constructor
     *
     * @param account_mgr AccountManager for fetching positions
     * @param db_path Path to DuckDB database
     * @param refresh_interval_seconds Refresh interval (default 30s)
     */
    explicit PositionTracker(
        std::shared_ptr<AccountManager> account_mgr,
        std::string db_path,
        int refresh_interval_seconds = 30
    ) : account_mgr_{std::move(account_mgr)},
        db_path_{std::move(db_path)},
        refresh_interval_{refresh_interval_seconds},
        running_{false},
        paused_{false} {

        // Open DuckDB connection
        db_ = std::make_unique<duckdb::DuckDB>(db_path_);
        conn_ = std::make_unique<duckdb::Connection>(*db_);
    }

    // Rule of Five - non-copyable, non-movable due to thread
    PositionTracker(PositionTracker const&) = delete;
    auto operator=(PositionTracker const&) -> PositionTracker& = delete;
    PositionTracker(PositionTracker&&) noexcept = delete;
    auto operator=(PositionTracker&&) noexcept -> PositionTracker& = delete;

    ~PositionTracker() {
        stop();
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
            return;  // Already running
        }

        account_id_ = std::move(account_id);
        running_ = true;
        paused_ = false;

        // Start background thread
        tracking_thread_ = std::thread([this]() { trackingLoop(); });
    }

    /**
     * Stop automatic position tracking
     */
    auto stop() -> void {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) return;
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
    }

    /**
     * Resume tracking
     */
    auto resume() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = false;
        cv_.notify_one();
    }

    /**
     * Force immediate position refresh
     */
    auto refreshNow() -> void {
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
    [[nodiscard]] auto getPosition(std::string const& symbol) const
        -> std::optional<Position> {

        std::lock_guard<std::mutex> lock(mutex_);

        for (auto const& pos : cached_positions_) {
            if (pos.symbol == symbol) {
                return pos;
            }
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
    [[nodiscard]] auto isRunning() const noexcept -> bool {
        return running_;
    }

    /**
     * Check if tracking is paused
     */
    [[nodiscard]] auto isPaused() const noexcept -> bool {
        return paused_;
    }

    /**
     * Get last update timestamp
     */
    [[nodiscard]] auto getLastUpdateTime() const noexcept -> Timestamp {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_update_timestamp_;
    }

private:
    // ========================================================================
    // Tracking Loop
    // ========================================================================

    auto trackingLoop() -> void {
        while (running_) {
            if (!paused_) {
                updatePositions();
            }

            // Wait for refresh interval or stop signal
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::seconds(refresh_interval_),
                        [this]() { return !running_ || !paused_; });
        }
    }

    // ========================================================================
    // Position Update Logic
    // ========================================================================

    auto updatePositions() -> void {
        // Fetch current positions from Schwab API
        auto positions_result = account_mgr_->getPositions(account_id_);

        if (!positions_result) {
            // Log error but continue tracking
            return;
        }

        auto const& new_positions = *positions_result;

        // Detect changes
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
    }

    // ========================================================================
    // Change Detection
    // ========================================================================

    auto detectPositionChanges(std::vector<Position> const& new_positions) -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        // Build map of old positions
        std::unordered_map<std::string, Position> old_map;
        for (auto const& pos : cached_positions_) {
            old_map[pos.symbol] = pos;
        }

        // Build map of new positions
        std::unordered_map<std::string, Position> new_map;
        for (auto const& pos : new_positions) {
            new_map[pos.symbol] = pos;
        }

        // Detect new positions (OPENED)
        for (auto const& [symbol, pos] : new_map) {
            if (old_map.find(symbol) == old_map.end()) {
                recordPositionChange("OPENED", pos, std::nullopt);
            }
        }

        // Detect closed positions
        for (auto const& [symbol, pos] : old_map) {
            if (new_map.find(symbol) == new_map.end()) {
                recordPositionChange("CLOSED", pos, std::nullopt);
            }
        }

        // Detect quantity changes
        for (auto const& [symbol, new_pos] : new_map) {
            if (old_map.find(symbol) != old_map.end()) {
                auto const& old_pos = old_map[symbol];

                if (new_pos.quantity != old_pos.quantity) {
                    std::string change_type = (new_pos.quantity > old_pos.quantity)
                        ? "INCREASED" : "DECREASED";
                    recordPositionChange(change_type, new_pos, old_pos);
                }
            }
        }
    }

    auto recordPositionChange(
        std::string const& change_type,
        Position const& new_pos,
        std::optional<Position> const& old_pos
    ) -> void {
        // Insert into position_changes table
        try {
            std::string query = R"(
                INSERT INTO position_changes (
                    account_id, symbol, change_type,
                    quantity_before, quantity_after, quantity_change,
                    average_cost_before, average_cost_after,
                    price_at_change, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            )";

            auto stmt = conn_->Prepare(query);
            stmt->Execute(
                new_pos.account_id,
                new_pos.symbol,
                change_type,
                old_pos ? old_pos->quantity : 0,
                new_pos.quantity,
                new_pos.quantity - (old_pos ? old_pos->quantity : 0),
                old_pos ? old_pos->average_cost : 0.0,
                new_pos.average_cost,
                new_pos.current_price
            );
        } catch (...) {
            // Log error but continue
        }
    }

    // ========================================================================
    // DuckDB Persistence
    // ========================================================================

    auto persistPositions(std::vector<Position> const& positions) -> void {
        try {
            // Begin transaction
            conn_->Query("BEGIN TRANSACTION");

            // Delete existing positions for this account
            auto delete_stmt = conn_->Prepare(
                "DELETE FROM positions WHERE account_id = ?"
            );
            delete_stmt->Execute(account_id_);

            // Insert current positions
            for (auto const& pos : positions) {
                std::string query = R"(
                    INSERT INTO positions (
                        account_id, symbol, asset_type, cusip,
                        quantity, long_quantity, short_quantity,
                        average_cost, current_price, market_value, cost_basis,
                        unrealized_pnl, unrealized_pnl_percent,
                        day_pnl, day_pnl_percent, previous_close,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                )";

                auto stmt = conn_->Prepare(query);
                stmt->Execute(
                    pos.account_id, pos.symbol, pos.asset_type, pos.cusip,
                    pos.quantity, pos.long_quantity, pos.short_quantity,
                    pos.average_cost, pos.current_price, pos.market_value, pos.cost_basis,
                    pos.unrealized_pnl, pos.unrealized_pnl_percent,
                    pos.day_pnl, pos.day_pnl_percent, pos.previous_close
                );
            }

            // Commit transaction
            conn_->Query("COMMIT");

        } catch (...) {
            conn_->Query("ROLLBACK");
            // Log error but continue
        }
    }

    auto persistPositionHistory(std::vector<Position> const& positions) -> void {
        try {
            for (auto const& pos : positions) {
                std::string query = R"(
                    INSERT INTO position_history (
                        account_id, symbol, asset_type, quantity,
                        average_cost, current_price, market_value, cost_basis,
                        unrealized_pnl, unrealized_pnl_percent, day_pnl,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                )";

                auto stmt = conn_->Prepare(query);
                stmt->Execute(
                    pos.account_id, pos.symbol, pos.asset_type, pos.quantity,
                    pos.average_cost, pos.current_price, pos.market_value, pos.cost_basis,
                    pos.unrealized_pnl, pos.unrealized_pnl_percent, pos.day_pnl
                );
            }
        } catch (...) {
            // Log error but continue
        }
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

    // DuckDB
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;

    // Threading
    std::thread tracking_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Cache
    std::vector<Position> cached_positions_;
    Timestamp last_update_timestamp_{0};
};

} // namespace bigbrother::schwab
