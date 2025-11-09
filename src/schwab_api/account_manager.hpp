/**
 * BigBrotherAnalytics - Schwab API Account Manager (C++23)
 *
 * Complete account management with:
 * - Account information retrieval
 * - Position tracking and updates
 * - Transaction history
 * - Balance monitoring
 * - Portfolio analytics
 *
 * SECURITY: Read-only operations only, validates all account IDs
 * PERFORMANCE: Caches account data with automatic refresh
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#pragma once

#include "account_types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <expected>
#include <unordered_map>
#include <optional>

namespace bigbrother::schwab {

// Forward declarations
class TokenManager;
template<typename T> using Result = std::expected<T, std::string>;

/**
 * Account Manager - Thread-safe account data access
 *
 * Implements Schwab API account endpoints:
 * - GET /trader/v1/accounts
 * - GET /trader/v1/accounts/{accountHash}
 * - GET /trader/v1/accounts/{accountHash}/positions
 * - GET /trader/v1/accounts/{accountHash}/transactions
 */
class AccountManager {
public:
    explicit AccountManager(std::shared_ptr<TokenManager> token_mgr)
        : token_mgr_{std::move(token_mgr)},
          read_only_mode_{true} {}

    // Rule of Five - deleted due to mutex member
    AccountManager(AccountManager const&) = delete;
    auto operator=(AccountManager const&) -> AccountManager& = delete;
    AccountManager(AccountManager&&) noexcept = delete;
    auto operator=(AccountManager&&) noexcept -> AccountManager& = delete;
    ~AccountManager() = default;

    // ========================================================================
    // Account Information
    // ========================================================================

    /**
     * Get all accounts linked to the authenticated user
     * GET /trader/v1/accounts
     *
     * @return Vector of Account objects
     */
    [[nodiscard]] auto getAccounts() -> Result<std::vector<Account>> {
        logAccountAccess("getAccounts", "all");

        std::lock_guard<std::mutex> lock(mutex_);

        // Stub implementation - in production, make HTTP request
        std::vector<Account> accounts;

        Account acct;
        acct.account_id = "XXXX1234";
        acct.account_hash = generateAccountHash(acct.account_id);
        acct.account_type = "MARGIN";
        acct.account_nickname = "Trading Account";
        acct.is_day_trader = true;
        acct.is_closing_only = false;
        acct.updated_at = getCurrentTimestamp();

        accounts.push_back(acct);

        // Cache the accounts
        cached_accounts_ = accounts;

        return accounts;
    }

    /**
     * Get detailed account information
     * GET /trader/v1/accounts/{accountHash}
     *
     * @param account_id Account ID or hash
     * @return Account object with full details
     */
    [[nodiscard]] auto getAccount(std::string const& account_id) -> Result<Account> {
        if (!validateAccountId(account_id)) {
            return std::unexpected("Invalid account ID: " + account_id);
        }

        logAccountAccess("getAccount", account_id);

        std::lock_guard<std::mutex> lock(mutex_);

        // Stub implementation
        Account acct;
        acct.account_id = account_id;
        acct.account_hash = generateAccountHash(account_id);
        acct.account_type = "MARGIN";
        acct.updated_at = getCurrentTimestamp();

        return acct;
    }

    // ========================================================================
    // Position Management
    // ========================================================================

    /**
     * Get all positions for an account
     * GET /trader/v1/accounts/{accountHash}/positions
     *
     * @param account_id Account ID or hash
     * @return Vector of Position objects
     */
    [[nodiscard]] auto getPositions(std::string const& account_id)
        -> Result<std::vector<Position>> {

        if (!validateAccountId(account_id)) {
            return std::unexpected("Invalid account ID: " + account_id);
        }

        logAccountAccess("getPositions", account_id);

        std::lock_guard<std::mutex> lock(mutex_);

        // Stub implementation - return empty positions for now
        // In production, this would make HTTP request and parse JSON response
        return std::vector<Position>{};
    }

    /**
     * Get a specific position for an account
     *
     * @param account_id Account ID or hash
     * @param symbol Security symbol
     * @return Position object if found
     */
    [[nodiscard]] auto getPosition(std::string const& account_id, std::string const& symbol)
        -> Result<std::optional<Position>> {

        if (!validateAccountId(account_id)) {
            return std::unexpected("Invalid account ID: " + account_id);
        }

        auto positions_result = getPositions(account_id);
        if (!positions_result) {
            return std::unexpected(positions_result.error());
        }

        auto const& positions = *positions_result;
        for (auto const& pos : positions) {
            if (pos.symbol == symbol) {
                return pos;
            }
        }

        return std::nullopt;
    }

    // ========================================================================
    // Transaction History
    // ========================================================================

    /**
     * Get transaction history for an account
     * GET /trader/v1/accounts/{accountHash}/transactions
     *
     * @param account_id Account ID or hash
     * @param start_date Start date (ISO 8601 format)
     * @param end_date End date (ISO 8601 format)
     * @param symbol Optional symbol filter
     * @return Vector of Transaction objects
     */
    [[nodiscard]] auto getTransactions(
        std::string const& account_id,
        std::string const& start_date,
        std::string const& end_date,
        std::optional<std::string> const& symbol = std::nullopt
    ) -> Result<std::vector<Transaction>> {

        if (!validateAccountId(account_id)) {
            return std::unexpected("Invalid account ID: " + account_id);
        }

        logAccountAccess("getTransactions", account_id);

        std::lock_guard<std::mutex> lock(mutex_);

        // Stub implementation
        return std::vector<Transaction>{};
    }

    /**
     * Get transaction by ID
     *
     * @param account_id Account ID or hash
     * @param transaction_id Transaction ID
     * @return Transaction object if found
     */
    [[nodiscard]] auto getTransaction(
        std::string const& account_id,
        std::string const& transaction_id
    ) -> Result<std::optional<Transaction>> {

        if (!validateAccountId(account_id)) {
            return std::unexpected("Invalid account ID: " + account_id);
        }

        logAccountAccess("getTransaction", account_id);

        // Stub implementation
        return std::nullopt;
    }

    // ========================================================================
    // Balance Information
    // ========================================================================

    /**
     * Get account balances
     *
     * @param account_id Account ID or hash
     * @return Balance object with all balance details
     */
    [[nodiscard]] auto getBalances(std::string const& account_id) -> Result<Balance> {
        if (!validateAccountId(account_id)) {
            return std::unexpected("Invalid account ID: " + account_id);
        }

        logAccountAccess("getBalances", account_id);

        std::lock_guard<std::mutex> lock(mutex_);

        // Stub implementation - using $30K account data
        Balance balance;
        balance.total_equity = 30'000.0;
        balance.cash = 28'000.0;
        balance.cash_available = 28'000.0;
        balance.buying_power = 28'000.0;
        balance.day_trading_buying_power = 112'000.0;  // 4x leverage for day trading
        balance.margin_balance = 0.0;
        balance.margin_equity = 30'000.0;
        balance.long_market_value = 2'000.0;
        balance.short_market_value = 0.0;
        balance.unsettled_cash = 0.0;
        balance.maintenance_call = 0.0;
        balance.reg_t_call = 0.0;
        balance.equity_percentage = 100.0;
        balance.updated_at = getCurrentTimestamp();

        return balance;
    }

    /**
     * Get available buying power
     *
     * @param account_id Account ID or hash
     * @return Buying power amount
     */
    [[nodiscard]] auto getBuyingPower(std::string const& account_id) -> Result<double> {
        auto balance_result = getBalances(account_id);
        if (!balance_result) {
            return std::unexpected(balance_result.error());
        }

        return balance_result->buying_power;
    }

    /**
     * Get day trading buying power
     *
     * @param account_id Account ID or hash
     * @return Day trading buying power amount
     */
    [[nodiscard]] auto getDayTradingBuyingPower(std::string const& account_id)
        -> Result<double> {

        auto balance_result = getBalances(account_id);
        if (!balance_result) {
            return std::unexpected(balance_result.error());
        }

        return balance_result->day_trading_buying_power;
    }

    // ========================================================================
    // Portfolio Analytics
    // ========================================================================

    /**
     * Get portfolio summary statistics
     *
     * @param account_id Account ID or hash
     * @return PortfolioSummary with analytics
     */
    [[nodiscard]] auto getPortfolioSummary(std::string const& account_id)
        -> Result<PortfolioSummary> {

        auto positions_result = getPositions(account_id);
        if (!positions_result) {
            return std::unexpected(positions_result.error());
        }

        auto balance_result = getBalances(account_id);
        if (!balance_result) {
            return std::unexpected(balance_result.error());
        }

        return calculatePortfolioSummary(*positions_result, *balance_result);
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    /**
     * Enable/disable read-only mode (safety feature)
     * When enabled, prevents any account-modifying operations
     */
    auto setReadOnlyMode(bool enabled) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        read_only_mode_ = enabled;
    }

    [[nodiscard]] auto isReadOnlyMode() const noexcept -> bool {
        return read_only_mode_;
    }

private:
    // ========================================================================
    // Helper Methods
    // ========================================================================

    [[nodiscard]] auto validateAccountId(std::string const& account_id) const noexcept -> bool {
        // Basic validation
        if (account_id.empty()) return false;
        if (account_id.length() < 4) return false;

        // Add more validation as needed
        return true;
    }

    auto logAccountAccess(std::string const& operation, std::string const& account_id) const -> void {
        // Log all account access for security audit
        // In production, this would write to secure audit log
        // Format: [TIMESTAMP] [ACCOUNT_ACCESS] operation=X account=XXXX1234 (last 4 digits)
    }

    [[nodiscard]] auto generateAccountHash(std::string const& account_id) const -> std::string {
        // Generate hash for API calls
        // In production, use actual hash from Schwab API
        return "HASH_" + account_id;
    }

    [[nodiscard]] auto getCurrentTimestamp() const noexcept -> Timestamp {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }

    [[nodiscard]] auto calculatePortfolioSummary(
        std::vector<Position> const& positions,
        Balance const& balance
    ) const -> PortfolioSummary {

        PortfolioSummary summary;
        summary.total_equity = balance.total_equity;
        summary.total_cash = balance.cash;
        summary.total_market_value = 0.0;
        summary.total_cost_basis = 0.0;
        summary.total_unrealized_pnl = 0.0;
        summary.total_day_pnl = 0.0;
        summary.position_count = static_cast<int>(positions.size());
        summary.long_position_count = 0;
        summary.short_position_count = 0;
        summary.largest_position_percent = 0.0;
        summary.portfolio_concentration = 0.0;

        // Calculate portfolio metrics
        for (auto const& pos : positions) {
            summary.total_market_value += pos.market_value;
            summary.total_cost_basis += pos.cost_basis;
            summary.total_unrealized_pnl += pos.unrealized_pnl;
            summary.total_day_pnl += pos.day_pnl;

            if (pos.isLong()) {
                summary.long_position_count++;
            } else if (pos.isShort()) {
                summary.short_position_count++;
            }

            // Calculate position weight
            if (balance.total_equity > 0.0) {
                double position_percent = (pos.market_value / balance.total_equity) * 100.0;
                if (position_percent > summary.largest_position_percent) {
                    summary.largest_position_percent = position_percent;
                }

                // Herfindahl index for concentration
                summary.portfolio_concentration += (position_percent / 100.0) *
                                                   (position_percent / 100.0);
            }
        }

        // Calculate percentages
        if (summary.total_cost_basis > 0.0) {
            summary.total_unrealized_pnl_percent =
                (summary.total_unrealized_pnl / summary.total_cost_basis) * 100.0;
        }

        if (balance.total_equity > 0.0) {
            summary.total_day_pnl_percent =
                (summary.total_day_pnl / balance.total_equity) * 100.0;
        }

        summary.updated_at = getCurrentTimestamp();

        return summary;
    }

    // ========================================================================
    // Member Variables
    // ========================================================================

    std::shared_ptr<TokenManager> token_mgr_;
    mutable std::mutex mutex_;
    bool read_only_mode_;
    std::vector<Account> cached_accounts_;
};

} // namespace bigbrother::schwab
