/**
 * BigBrotherAnalytics - Schwab API Account Types (C++23)
 *
 * Comprehensive account data structures for Schwab API integration
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace bigbrother::schwab {

using Timestamp = int64_t;
using Price = double;
using Quantity = int64_t;

// ============================================================================
// Account Information
// ============================================================================

/**
 * Account details from Schwab API
 */
struct Account {
    std::string account_id;           // Account number
    std::string account_hash;         // Hashed account ID for API calls
    std::string account_type;         // CASH, MARGIN, IRA, etc.
    std::string account_nickname;     // User-defined nickname
    bool is_day_trader{false};        // Pattern day trader flag
    bool is_closing_only{false};      // Closing only restriction
    Timestamp updated_at{0};

    [[nodiscard]] auto isValid() const noexcept -> bool {
        return !account_id.empty() && !account_hash.empty();
    }

    [[nodiscard]] auto isMarginAccount() const noexcept -> bool {
        return account_type == "MARGIN";
    }
};

// ============================================================================
// Balance Information
// ============================================================================

/**
 * Comprehensive account balance details
 */
struct Balance {
    double total_equity{0.0};              // Total account value
    double cash{0.0};                      // Cash balance
    double cash_available{0.0};            // Available to withdraw
    double buying_power{0.0};              // Standard buying power
    double day_trading_buying_power{0.0};  // Day trading buying power
    double margin_balance{0.0};            // Margin used
    double margin_equity{0.0};             // Equity minus margin
    double long_market_value{0.0};         // Value of long positions
    double short_market_value{0.0};        // Value of short positions
    double unsettled_cash{0.0};            // Unsettled funds
    double maintenance_call{0.0};          // Maintenance margin call
    double reg_t_call{0.0};                // Regulation T call
    double equity_percentage{0.0};         // Equity as % of account value
    Timestamp updated_at{0};

    [[nodiscard]] auto hasSufficientFunds(double required) const noexcept -> bool {
        return buying_power >= required;
    }

    [[nodiscard]] auto getMarginUsagePercent() const noexcept -> double {
        if (total_equity <= 0.0) return 0.0;
        return (margin_balance / total_equity) * 100.0;
    }

    [[nodiscard]] auto hasMarginCall() const noexcept -> bool {
        return maintenance_call > 0.0 || reg_t_call > 0.0;
    }

    [[nodiscard]] auto getTotalCallAmount() const noexcept -> double {
        return maintenance_call + reg_t_call;
    }
};

// ============================================================================
// Position Information
// ============================================================================

/**
 * Position details for a security
 */
struct Position {
    std::string account_id;
    std::string symbol;
    std::string asset_type;              // EQUITY, OPTION, BOND, etc.
    std::string cusip;                   // CUSIP identifier
    Quantity quantity{0};                // Total quantity (positive = long, negative = short)
    Quantity long_quantity{0};           // Long position quantity
    Quantity short_quantity{0};          // Short position quantity
    Price average_cost{0.0};             // Cost basis per share
    Price current_price{0.0};            // Current market price
    double market_value{0.0};            // Current market value
    double cost_basis{0.0};              // Total cost basis
    double unrealized_pnl{0.0};          // Unrealized profit/loss
    double unrealized_pnl_percent{0.0};  // Unrealized P/L %
    double day_pnl{0.0};                 // Intraday P/L
    double day_pnl_percent{0.0};         // Intraday P/L %
    Price previous_close{0.0};           // Previous day close
    Timestamp updated_at{0};

    [[nodiscard]] auto getCurrentValue() const noexcept -> double {
        return static_cast<double>(quantity) * current_price;
    }

    [[nodiscard]] auto calculatePnL() const noexcept -> double {
        return market_value - cost_basis;
    }

    [[nodiscard]] auto calculatePnLPercent() const noexcept -> double {
        if (cost_basis == 0.0) return 0.0;
        return (unrealized_pnl / cost_basis) * 100.0;
    }

    [[nodiscard]] auto isLong() const noexcept -> bool {
        return quantity > 0;
    }

    [[nodiscard]] auto isShort() const noexcept -> bool {
        return quantity < 0;
    }

    [[nodiscard]] auto isOption() const noexcept -> bool {
        return asset_type == "OPTION";
    }
};

// ============================================================================
// Transaction Information
// ============================================================================

/**
 * Transaction types
 */
enum class TransactionType {
    Trade,
    ReceiveAndDeliver,
    DividendOrInterest,
    ACHReceipt,
    ACHDisbursement,
    CashReceipt,
    CashDisbursement,
    ElectronicFund,
    WireOut,
    WireIn,
    Journal,
    Memorandum,
    MarginCall,
    MoneyMarket,
    SMA
};

/**
 * Transaction instruction
 */
enum class TransactionInstruction {
    Buy,
    Sell,
    BuyToCover,
    SellShort,
    None
};

/**
 * Transaction record
 */
struct Transaction {
    std::string transaction_id;
    std::string account_id;
    std::string symbol;
    TransactionType type{TransactionType::Trade};
    TransactionInstruction instruction{TransactionInstruction::None};
    std::string description;
    Timestamp transaction_date{0};
    Timestamp settlement_date{0};
    double net_amount{0.0};              // Net amount (after fees/commissions)
    double gross_amount{0.0};            // Gross amount (before fees/commissions)
    Quantity quantity{0};
    Price price{0.0};
    double commission{0.0};
    double fees{0.0};
    double reg_fee{0.0};                 // Regulatory fees
    double sec_fee{0.0};                 // SEC fees
    std::string position_effect;         // OPENING, CLOSING
    std::string asset_type;              // EQUITY, OPTION, etc.

    [[nodiscard]] auto isTradeTransaction() const noexcept -> bool {
        return type == TransactionType::Trade;
    }

    [[nodiscard]] auto isBuy() const noexcept -> bool {
        return instruction == TransactionInstruction::Buy ||
               instruction == TransactionInstruction::BuyToCover;
    }

    [[nodiscard]] auto isSell() const noexcept -> bool {
        return instruction == TransactionInstruction::Sell ||
               instruction == TransactionInstruction::SellShort;
    }

    [[nodiscard]] auto getTotalCost() const noexcept -> double {
        return commission + fees + reg_fee + sec_fee;
    }
};

// ============================================================================
// Portfolio Analytics
// ============================================================================

/**
 * Portfolio summary statistics
 */
struct PortfolioSummary {
    double total_equity{0.0};
    double total_cash{0.0};
    double total_market_value{0.0};
    double total_cost_basis{0.0};
    double total_unrealized_pnl{0.0};
    double total_unrealized_pnl_percent{0.0};
    double total_day_pnl{0.0};
    double total_day_pnl_percent{0.0};
    int position_count{0};
    int long_position_count{0};
    int short_position_count{0};
    double largest_position_percent{0.0};
    double portfolio_concentration{0.0};  // Herfindahl index
    Timestamp updated_at{0};

    [[nodiscard]] auto getDiversification() const noexcept -> double {
        if (position_count <= 1) return 0.0;
        return 1.0 - portfolio_concentration;
    }
};

/**
 * Position risk metrics
 */
struct PositionRisk {
    std::string symbol;
    double position_size_percent{0.0};   // % of portfolio
    double var_95{0.0};                  // Value at Risk (95%)
    double expected_shortfall{0.0};      // Conditional VaR
    double beta{0.0};                    // Market beta
    double volatility{0.0};              // Historical volatility
    double sharpe_ratio{0.0};            // Risk-adjusted return

    [[nodiscard]] auto isHighRisk() const noexcept -> bool {
        return volatility > 0.30 || position_size_percent > 20.0;
    }
};

} // namespace bigbrother::schwab
