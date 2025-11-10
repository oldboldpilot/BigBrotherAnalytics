/**
 * BigBrotherAnalytics - Position Tracker Implementation (C++23)
 *
 * Automatic position tracking with 30-second refresh:
 * - Real-time position monitoring
 * - Change detection (new positions, closures, quantity changes)
 * - DuckDB persistence with history
 * - P/L calculations
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include "account_manager.hpp"
#include "account_types.hpp"
#include "position_tracker.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <thread>
#include <unordered_map>
#include <vector>

namespace bigbrother::schwab {

// ============================================================================
// PositionTracker Extended Implementation
// ============================================================================

class PositionTrackerImpl {
  public:
    explicit PositionTrackerImpl(std::shared_ptr<AccountManager> account_mgr, std::string db_path,
                                 int refresh_interval_seconds = 30)
        : account_mgr_{std::move(account_mgr)}, db_path_{std::move(db_path)},
          refresh_interval_{refresh_interval_seconds}, running_{false}, paused_{false},
          update_count_{0} {

        // Open DuckDB connection
        db_ = std::make_unique<duckdb::DuckDB>(db_path_);
        conn_ = std::make_unique<duckdb::Connection>(*db_);

        spdlog::info("PositionTracker initialized (refresh interval: {}s)", refresh_interval_);
    }

    ~PositionTrackerImpl() { stop(); }

    // Rule of Five: Explicitly delete copy operations, default move operations
    PositionTrackerImpl(PositionTrackerImpl const&) = delete;
    auto operator=(PositionTrackerImpl const&) -> PositionTrackerImpl& = delete;
    PositionTrackerImpl(PositionTrackerImpl&&) noexcept = default;
    auto operator=(PositionTrackerImpl&&) noexcept -> PositionTrackerImpl& = default;

    // ========================================================================
    // Control Methods
    // ========================================================================

    auto start(std::string account_id) -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        if (running_) {
            spdlog::warn("PositionTracker already running");
            return;
        }

        account_id_ = std::move(account_id);
        running_ = true;
        paused_ = false;
        update_count_ = 0;

        spdlog::info("Starting PositionTracker for account: {}", account_id_);

        // Start background thread
        tracking_thread_ = std::thread([this]() { trackingLoop(); });
    }

    auto stop() -> void {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_)
                return;

            spdlog::info("Stopping PositionTracker (total updates: {})", update_count_.load());
            running_ = false;
        }

        cv_.notify_all();

        if (tracking_thread_.joinable()) {
            tracking_thread_.join();
        }
    }

    auto pause() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = true;
        spdlog::info("PositionTracker paused");
    }

    auto resume() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = false;
        cv_.notify_one();
        spdlog::info("PositionTracker resumed");
    }

    auto refreshNow() -> void {
        spdlog::info("Forcing immediate position refresh");
        cv_.notify_one();
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    [[nodiscard]] auto getCurrentPositions() const -> std::vector<Position> {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_positions_;
    }

    [[nodiscard]] auto getPosition(std::string const& symbol) const -> std::optional<Position> {

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = std::find_if(cached_positions_.begin(), cached_positions_.end(),
                               [&symbol](auto const& pos) -> bool { return pos.symbol == symbol; });

        if (it != cached_positions_.end()) {
            return *it;
        }

        return std::nullopt;
    }

    [[nodiscard]] auto getPositionCount() const noexcept -> size_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_positions_.size();
    }

    [[nodiscard]] auto isRunning() const noexcept -> bool { return running_; }

    [[nodiscard]] auto isPaused() const noexcept -> bool { return paused_; }

    [[nodiscard]] auto getLastUpdateTime() const noexcept -> Timestamp {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_update_timestamp_;
    }

    [[nodiscard]] auto getUpdateCount() const noexcept -> int { return update_count_; }

    // ========================================================================
    // Portfolio Analytics
    // ========================================================================

    [[nodiscard]] auto calculateTotalPnL() const -> double {
        std::lock_guard<std::mutex> lock(mutex_);

        double total_pnl = 0.0;
        for (auto const& pos : cached_positions_) {
            total_pnl += pos.unrealized_pnl;
        }

        return total_pnl;
    }

    [[nodiscard]] auto calculateDayPnL() const -> double {
        std::lock_guard<std::mutex> lock(mutex_);

        double day_pnl = 0.0;
        for (auto const& pos : cached_positions_) {
            day_pnl += pos.day_pnl;
        }

        return day_pnl;
    }

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
    // Tracking Loop (Runs in background thread)
    // ========================================================================

    auto trackingLoop() -> void {
        spdlog::info("PositionTracker thread started ({}s refresh)", refresh_interval_);

        // Initial update immediately
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

        spdlog::info("PositionTracker thread stopped");
    }

    // ========================================================================
    // Position Update Logic
    // ========================================================================

    auto updatePositions() -> void {
        auto start_time = std::chrono::high_resolution_clock::now();

        spdlog::debug("Fetching positions from Schwab API (update #{})",
                      update_count_.load(std::memory_order_relaxed) + 1);

        // Fetch current positions from Schwab API
        auto positions_result = account_mgr_->getPositions(account_id_);

        if (!positions_result) {
            spdlog::error("Failed to fetch positions: {}", positions_result.error());
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

        spdlog::info("Position update #{} complete: {} positions (took {}ms)", current_update,
                     new_positions.size(), duration.count());

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
                spdlog::info("NEW POSITION OPENED: {} ({} shares @ ${:.2f})", symbol, pos.quantity,
                             pos.average_cost);
                recordPositionChange("OPENED", pos, std::nullopt);
            }
        }

        // Detect CLOSED positions
        for (auto const& [symbol, pos] : old_map) {
            if (new_map.find(symbol) == new_map.end()) {
                spdlog::warn("POSITION CLOSED: {} (was {} shares)", symbol, pos.quantity);
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

                    spdlog::info("POSITION {}: {} ({} -> {} shares, change: {})", change_type,
                                 symbol, old_pos.quantity, new_pos.quantity,
                                 new_pos.quantity - old_pos.quantity);

                    recordPositionChange(change_type, new_pos, old_pos);
                }
            }
        }
    }

    auto recordPositionChange(std::string const& change_type, Position const& new_pos,
                              std::optional<Position> const& old_pos) -> void {
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
                new_pos.account_id, new_pos.symbol, change_type, old_pos ? old_pos->quantity : 0.0,
                new_pos.quantity, new_pos.quantity - (old_pos ? old_pos->quantity : 0.0),
                old_pos ? old_pos->average_cost : 0.0, new_pos.average_cost, new_pos.current_price);

        } catch (std::exception const& e) {
            spdlog::error("Failed to record position change: {}", e.what());
        }
    }

    // ========================================================================
    // DuckDB Persistence
    // ========================================================================

    auto persistPositions(std::vector<Position> const& positions) -> void {
        try {
            // Begin transaction for atomic update
            conn_->Query("BEGIN TRANSACTION");

            // Delete existing positions for this account
            auto delete_stmt = conn_->Prepare("DELETE FROM positions WHERE account_id = ?");
            delete_stmt->Execute(account_id_);

            // Insert current positions
            for (auto const& pos : positions) {
                insertPosition(pos);
            }

            // Commit transaction
            conn_->Query("COMMIT");

            spdlog::debug("Persisted {} positions to database", positions.size());

        } catch (std::exception const& e) {
            conn_->Query("ROLLBACK");
            spdlog::error("Database persistence error: {}", e.what());
        }
    }

    auto insertPosition(Position const& pos) -> void {
        std::string query = R"(
            INSERT INTO positions (
                account_id, symbol, asset_type, cusip,
                quantity, long_quantity, short_quantity,
                average_cost, current_price, market_value, cost_basis,
                unrealized_pnl, unrealized_pnl_percent,
                day_pnl, day_pnl_percent, previous_close,
                is_bot_managed, managed_by, opened_by, bot_strategy, opened_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        )";

        auto stmt = conn_->Prepare(query);
        stmt->Execute(pos.account_id, pos.symbol, pos.asset_type, pos.cusip, pos.quantity,
                      pos.long_quantity, pos.short_quantity, pos.average_cost, pos.current_price,
                      pos.market_value, pos.cost_basis, pos.unrealized_pnl,
                      pos.unrealized_pnl_percent, pos.day_pnl, pos.day_pnl_percent,
                      pos.previous_close, pos.is_bot_managed, pos.managed_by, pos.opened_by,
                      pos.bot_strategy, pos.opened_at);
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
                stmt->Execute(pos.account_id, pos.symbol, pos.asset_type, pos.quantity,
                              pos.average_cost, pos.current_price, pos.market_value, pos.cost_basis,
                              pos.unrealized_pnl, pos.unrealized_pnl_percent, pos.day_pnl);
            }
        } catch (std::exception const& e) {
            spdlog::error("Failed to persist position history: {}", e.what());
        }
    }

    // ========================================================================
    // Logging
    // ========================================================================

    auto logPnLSummary(std::vector<Position> const& positions) const -> void {
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

        spdlog::info("=== PORTFOLIO SUMMARY ===");
        spdlog::info("  Positions: {} (Manual: {}, Bot: {})", positions.size(), manual_count,
                     bot_count);
        spdlog::info("  Market Value: ${:.2f}", total_market_value);
        spdlog::info("  Cost Basis: ${:.2f}", total_cost_basis);
        spdlog::info("  Unrealized P/L: ${:.2f} ({:.2f}%)", total_unrealized_pnl,
                     total_pnl_percent);
        spdlog::info("  Day P/L: ${:.2f}", total_day_pnl);
        spdlog::info("========================");
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
    std::atomic<int> update_count_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Cache
    std::vector<Position> cached_positions_;
    Timestamp last_update_timestamp_{0};
};

} // namespace bigbrother::schwab
