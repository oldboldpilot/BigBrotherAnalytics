/**
 * BigBrotherAnalytics - Schwab Account Python Bindings (C++23)
 *
 * Python bindings for Schwab account data endpoints:
 * - Account information
 * - Position tracking with classification
 * - Transaction history
 * - Balance queries
 * - Portfolio analytics
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../schwab_api/account_types.hpp"
#include "../schwab_api/account_manager.hpp"
#include "../schwab_api/position_tracker.hpp"
#include "../schwab_api/portfolio_analyzer.hpp"

namespace py = pybind11;
using namespace bigbrother::schwab;

PYBIND11_MODULE(bigbrother_schwab_account, m) {
    m.doc() = "Schwab Account API - Python Bindings for Account Data and Position Tracking";

    // ========================================================================
    // Account Types
    // ========================================================================

    py::class_<Account>(m, "Account")
        .def(py::init<>())
        .def_readwrite("account_id", &Account::account_id)
        .def_readwrite("account_hash", &Account::account_hash)
        .def_readwrite("account_type", &Account::account_type)
        .def_readwrite("account_nickname", &Account::account_nickname)
        .def_readwrite("is_day_trader", &Account::is_day_trader)
        .def_readwrite("is_closing_only", &Account::is_closing_only)
        .def_readwrite("updated_at", &Account::updated_at)
        .def("is_valid", &Account::isValid)
        .def("is_margin_account", &Account::isMarginAccount)
        .def("__repr__", [](Account const& a) {
            return "<Account " + a.account_id + " (" + a.account_type + ")>";
        });

    py::class_<Balance>(m, "Balance")
        .def(py::init<>())
        .def_readwrite("total_equity", &Balance::total_equity)
        .def_readwrite("cash", &Balance::cash)
        .def_readwrite("cash_available", &Balance::cash_available)
        .def_readwrite("buying_power", &Balance::buying_power)
        .def_readwrite("day_trading_buying_power", &Balance::day_trading_buying_power)
        .def_readwrite("margin_balance", &Balance::margin_balance)
        .def_readwrite("margin_equity", &Balance::margin_equity)
        .def_readwrite("long_market_value", &Balance::long_market_value)
        .def_readwrite("short_market_value", &Balance::short_market_value)
        .def_readwrite("unsettled_cash", &Balance::unsettled_cash)
        .def_readwrite("maintenance_call", &Balance::maintenance_call)
        .def_readwrite("reg_t_call", &Balance::reg_t_call)
        .def_readwrite("equity_percentage", &Balance::equity_percentage)
        .def_readwrite("updated_at", &Balance::updated_at)
        .def("has_sufficient_funds", &Balance::hasSufficientFunds)
        .def("get_margin_usage_percent", &Balance::getMarginUsagePercent)
        .def("has_margin_call", &Balance::hasMarginCall)
        .def("get_total_call_amount", &Balance::getTotalCallAmount)
        .def("__repr__", [](Balance const& b) {
            return "<Balance equity=$" + std::to_string(b.total_equity) +
                   " buying_power=$" + std::to_string(b.buying_power) + ">";
        });

    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("account_id", &Position::account_id)
        .def_readwrite("symbol", &Position::symbol)
        .def_readwrite("asset_type", &Position::asset_type)
        .def_readwrite("cusip", &Position::cusip)
        .def_readwrite("quantity", &Position::quantity)
        .def_readwrite("long_quantity", &Position::long_quantity)
        .def_readwrite("short_quantity", &Position::short_quantity)
        .def_readwrite("average_cost", &Position::average_cost)
        .def_readwrite("current_price", &Position::current_price)
        .def_readwrite("market_value", &Position::market_value)
        .def_readwrite("cost_basis", &Position::cost_basis)
        .def_readwrite("unrealized_pnl", &Position::unrealized_pnl)
        .def_readwrite("unrealized_pnl_percent", &Position::unrealized_pnl_percent)
        .def_readwrite("day_pnl", &Position::day_pnl)
        .def_readwrite("day_pnl_percent", &Position::day_pnl_percent)
        .def_readwrite("previous_close", &Position::previous_close)
        .def_readwrite("updated_at", &Position::updated_at)
        // CRITICAL: Position classification fields
        .def_readwrite("is_bot_managed", &Position::is_bot_managed)
        .def_readwrite("managed_by", &Position::managed_by)
        .def_readwrite("opened_by", &Position::opened_by)
        .def_readwrite("bot_strategy", &Position::bot_strategy)
        .def_readwrite("opened_at", &Position::opened_at)
        // Methods
        .def("get_current_value", &Position::getCurrentValue)
        .def("calculate_pnl", &Position::calculatePnL)
        .def("calculate_pnl_percent", &Position::calculatePnLPercent)
        .def("is_long", &Position::isLong)
        .def("is_short", &Position::isShort)
        .def("is_option", &Position::isOption)
        .def("is_bot_managed", &Position::isBotManaged)
        .def("is_manual_position", &Position::isManualPosition)
        .def("mark_as_bot_managed", &Position::markAsBotManaged)
        .def("mark_as_manual", &Position::markAsManual)
        .def("__repr__", [](Position const& p) {
            std::string managed = p.is_bot_managed ? "BOT" : "MANUAL";
            return "<Position " + p.symbol +
                   " qty=" + std::to_string(p.quantity) +
                   " pnl=$" + std::to_string(p.unrealized_pnl) +
                   " managed=" + managed + ">";
        });

    py::enum_<TransactionType>(m, "TransactionType")
        .value("Trade", TransactionType::Trade)
        .value("ReceiveAndDeliver", TransactionType::ReceiveAndDeliver)
        .value("DividendOrInterest", TransactionType::DividendOrInterest)
        .value("ACHReceipt", TransactionType::ACHReceipt)
        .value("ACHDisbursement", TransactionType::ACHDisbursement)
        .value("CashReceipt", TransactionType::CashReceipt)
        .value("CashDisbursement", TransactionType::CashDisbursement)
        .value("ElectronicFund", TransactionType::ElectronicFund)
        .value("WireOut", TransactionType::WireOut)
        .value("WireIn", TransactionType::WireIn)
        .value("Journal", TransactionType::Journal)
        .value("Memorandum", TransactionType::Memorandum)
        .value("MarginCall", TransactionType::MarginCall)
        .value("MoneyMarket", TransactionType::MoneyMarket)
        .value("SMA", TransactionType::SMA)
        .export_values();

    py::enum_<TransactionInstruction>(m, "TransactionInstruction")
        .value("Buy", TransactionInstruction::Buy)
        .value("Sell", TransactionInstruction::Sell)
        .value("BuyToCover", TransactionInstruction::BuyToCover)
        .value("SellShort", TransactionInstruction::SellShort)
        .value("None", TransactionInstruction::None)
        .export_values();

    py::class_<Transaction>(m, "Transaction")
        .def(py::init<>())
        .def_readwrite("transaction_id", &Transaction::transaction_id)
        .def_readwrite("account_id", &Transaction::account_id)
        .def_readwrite("symbol", &Transaction::symbol)
        .def_readwrite("type", &Transaction::type)
        .def_readwrite("instruction", &Transaction::instruction)
        .def_readwrite("description", &Transaction::description)
        .def_readwrite("transaction_date", &Transaction::transaction_date)
        .def_readwrite("settlement_date", &Transaction::settlement_date)
        .def_readwrite("net_amount", &Transaction::net_amount)
        .def_readwrite("gross_amount", &Transaction::gross_amount)
        .def_readwrite("quantity", &Transaction::quantity)
        .def_readwrite("price", &Transaction::price)
        .def_readwrite("commission", &Transaction::commission)
        .def_readwrite("fees", &Transaction::fees)
        .def_readwrite("reg_fee", &Transaction::reg_fee)
        .def_readwrite("sec_fee", &Transaction::sec_fee)
        .def_readwrite("position_effect", &Transaction::position_effect)
        .def_readwrite("asset_type", &Transaction::asset_type)
        .def("is_trade_transaction", &Transaction::isTradeTransaction)
        .def("is_buy", &Transaction::isBuy)
        .def("is_sell", &Transaction::isSell)
        .def("get_total_cost", &Transaction::getTotalCost)
        .def("__repr__", [](Transaction const& t) {
            return "<Transaction " + t.transaction_id +
                   " " + t.symbol + " $" + std::to_string(t.net_amount) + ">";
        });

    py::class_<PortfolioSummary>(m, "PortfolioSummary")
        .def(py::init<>())
        .def_readwrite("total_equity", &PortfolioSummary::total_equity)
        .def_readwrite("total_cash", &PortfolioSummary::total_cash)
        .def_readwrite("total_market_value", &PortfolioSummary::total_market_value)
        .def_readwrite("total_cost_basis", &PortfolioSummary::total_cost_basis)
        .def_readwrite("total_unrealized_pnl", &PortfolioSummary::total_unrealized_pnl)
        .def_readwrite("total_unrealized_pnl_percent", &PortfolioSummary::total_unrealized_pnl_percent)
        .def_readwrite("total_day_pnl", &PortfolioSummary::total_day_pnl)
        .def_readwrite("total_day_pnl_percent", &PortfolioSummary::total_day_pnl_percent)
        .def_readwrite("position_count", &PortfolioSummary::position_count)
        .def_readwrite("long_position_count", &PortfolioSummary::long_position_count)
        .def_readwrite("short_position_count", &PortfolioSummary::short_position_count)
        .def_readwrite("largest_position_percent", &PortfolioSummary::largest_position_percent)
        .def_readwrite("portfolio_concentration", &PortfolioSummary::portfolio_concentration)
        .def_readwrite("updated_at", &PortfolioSummary::updated_at)
        .def("get_diversification", &PortfolioSummary::getDiversification);

    // ========================================================================
    // Portfolio Analytics Types
    // ========================================================================

    py::class_<SectorExposure>(m, "SectorExposure")
        .def(py::init<>())
        .def_readwrite("sector_name", &SectorExposure::sector_name)
        .def_readwrite("market_value", &SectorExposure::market_value)
        .def_readwrite("percent_of_portfolio", &SectorExposure::percent_of_portfolio)
        .def_readwrite("position_count", &SectorExposure::position_count)
        .def_readwrite("total_pnl", &SectorExposure::total_pnl)
        .def_readwrite("avg_pnl_percent", &SectorExposure::avg_pnl_percent);

    py::class_<RiskMetrics>(m, "RiskMetrics")
        .def(py::init<>())
        .def_readwrite("portfolio_heat", &RiskMetrics::portfolio_heat)
        .def_readwrite("value_at_risk_95", &RiskMetrics::value_at_risk_95)
        .def_readwrite("expected_shortfall", &RiskMetrics::expected_shortfall)
        .def_readwrite("portfolio_beta", &RiskMetrics::portfolio_beta)
        .def_readwrite("portfolio_volatility", &RiskMetrics::portfolio_volatility)
        .def_readwrite("sharpe_ratio", &RiskMetrics::sharpe_ratio)
        .def_readwrite("sortino_ratio", &RiskMetrics::sortino_ratio)
        .def_readwrite("max_drawdown", &RiskMetrics::max_drawdown)
        .def_readwrite("concentration_risk", &RiskMetrics::concentration_risk)
        .def_readwrite("positions_at_risk", &RiskMetrics::positions_at_risk);

    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("total_return", &PerformanceMetrics::total_return)
        .def_readwrite("total_return_percent", &PerformanceMetrics::total_return_percent)
        .def_readwrite("day_pnl", &PerformanceMetrics::day_pnl)
        .def_readwrite("day_pnl_percent", &PerformanceMetrics::day_pnl_percent)
        .def_readwrite("week_pnl", &PerformanceMetrics::week_pnl)
        .def_readwrite("month_pnl", &PerformanceMetrics::month_pnl)
        .def_readwrite("ytd_pnl", &PerformanceMetrics::ytd_pnl)
        .def_readwrite("annualized_return", &PerformanceMetrics::annualized_return)
        .def_readwrite("win_rate", &PerformanceMetrics::win_rate)
        .def_readwrite("profit_factor", &PerformanceMetrics::profit_factor)
        .def_readwrite("avg_win", &PerformanceMetrics::avg_win)
        .def_readwrite("avg_loss", &PerformanceMetrics::avg_loss)
        .def_readwrite("largest_win", &PerformanceMetrics::largest_win)
        .def_readwrite("largest_loss", &PerformanceMetrics::largest_loss);

    // ========================================================================
    // Result Type for Error Handling
    // ========================================================================

    // Helper function to convert Result to Python dict
    m.def("unwrap_result", [](py::object result) -> py::object {
        // In production, implement proper Result type handling
        return result;
    });

    // ========================================================================
    // AccountManager Class
    // ========================================================================

    py::class_<AccountManager, std::shared_ptr<AccountManager>>(m, "AccountManager")
        .def(py::init<std::shared_ptr<TokenManager>>())
        .def("get_accounts", [](AccountManager& mgr) -> py::object {
            auto result = mgr.getAccounts();
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, "Get all accounts linked to the authenticated user")
        .def("get_account", [](AccountManager& mgr, std::string const& account_id) -> py::object {
            auto result = mgr.getAccount(account_id);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, py::arg("account_id"), "Get detailed account information")
        .def("get_positions", [](AccountManager& mgr, std::string const& account_id) -> py::object {
            auto result = mgr.getPositions(account_id);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, py::arg("account_id"), "Get all positions for an account")
        .def("get_position", [](AccountManager& mgr, std::string const& account_id, std::string const& symbol) -> py::object {
            auto result = mgr.getPosition(account_id, symbol);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, py::arg("account_id"), py::arg("symbol"), "Get a specific position")
        .def("get_balances", [](AccountManager& mgr, std::string const& account_id) -> py::object {
            auto result = mgr.getBalances(account_id);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, py::arg("account_id"), "Get account balances")
        .def("get_buying_power", [](AccountManager& mgr, std::string const& account_id) -> double {
            auto result = mgr.getBuyingPower(account_id);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return *result;
        }, py::arg("account_id"), "Get available buying power")
        .def("get_transactions", [](AccountManager& mgr,
                                    std::string const& account_id,
                                    std::string const& start_date,
                                    std::string const& end_date) -> py::object {
            auto result = mgr.getTransactions(account_id, start_date, end_date);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, py::arg("account_id"), py::arg("start_date"), py::arg("end_date"),
           "Get transaction history")
        .def("get_portfolio_summary", [](AccountManager& mgr, std::string const& account_id) -> py::object {
            auto result = mgr.getPortfolioSummary(account_id);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return py::cast(*result);
        }, py::arg("account_id"), "Get portfolio summary statistics")
        .def("set_read_only_mode", &AccountManager::setReadOnlyMode,
             py::arg("enabled"), "Enable/disable read-only mode (safety feature)")
        .def("is_read_only_mode", &AccountManager::isReadOnlyMode,
             "Check if read-only mode is enabled");

    // ========================================================================
    // PortfolioAnalyzer Class
    // ========================================================================

    py::class_<PortfolioAnalyzer>(m, "PortfolioAnalyzer")
        .def(py::init<>())
        .def("analyze_portfolio", &PortfolioAnalyzer::analyzePortfolio,
             py::arg("positions"), py::arg("balance"),
             "Calculate comprehensive portfolio summary")
        .def("calculate_sector_exposure", &PortfolioAnalyzer::calculateSectorExposure,
             py::arg("positions"), py::arg("sector_map"), py::arg("total_equity"),
             "Calculate sector exposure breakdown")
        .def("calculate_risk_metrics", &PortfolioAnalyzer::calculateRiskMetrics,
             py::arg("positions"), py::arg("balance"),
             "Calculate risk metrics")
        .def("calculate_performance_metrics", &PortfolioAnalyzer::calculatePerformanceMetrics,
             py::arg("positions"), py::arg("transactions"), py::arg("balance"),
             "Calculate performance metrics")
        .def("get_largest_positions", &PortfolioAnalyzer::getLargestPositions,
             py::arg("positions"), py::arg("limit") = 10,
             "Find largest positions")
        .def("get_top_performers", &PortfolioAnalyzer::getTopPerformers,
             py::arg("positions"), py::arg("limit") = 10,
             "Find top performing positions")
        .def("get_worst_performers", &PortfolioAnalyzer::getWorstPerformers,
             py::arg("positions"), py::arg("limit") = 10,
             "Find worst performing positions")
        .def("has_concentration_risk", &PortfolioAnalyzer::hasConcentrationRisk,
             py::arg("positions"), py::arg("total_equity"), py::arg("threshold_percent") = 20.0,
             "Check for position concentration risk");

    // ========================================================================
    // PositionTracker Class
    // ========================================================================

    py::class_<PositionTracker, std::shared_ptr<PositionTracker>>(m, "PositionTracker")
        .def(py::init<std::shared_ptr<AccountManager>, std::string, int>(),
             py::arg("account_mgr"),
             py::arg("db_path"),
             py::arg("refresh_interval_seconds") = 30)
        .def("start", &PositionTracker::start,
             py::arg("account_id"),
             "Start automatic position tracking")
        .def("stop", &PositionTracker::stop,
             "Stop automatic position tracking")
        .def("pause", &PositionTracker::pause,
             "Pause tracking (without stopping thread)")
        .def("resume", &PositionTracker::resume,
             "Resume tracking")
        .def("refresh_now", &PositionTracker::refreshNow,
             "Force immediate position refresh")
        .def("get_current_positions", &PositionTracker::getCurrentPositions,
             "Get current positions from cache")
        .def("get_position", &PositionTracker::getPosition,
             py::arg("symbol"),
             "Get position for specific symbol")
        .def("get_position_count", &PositionTracker::getPositionCount,
             "Get position count")
        .def("is_running", &PositionTracker::isRunning,
             "Check if tracking is running")
        .def("is_paused", &PositionTracker::isPaused,
             "Check if tracking is paused")
        .def("get_last_update_time", &PositionTracker::getLastUpdateTime,
             "Get last update timestamp");

    // ========================================================================
    // Utility Functions
    // ========================================================================

    m.def("classify_existing_positions",
        [](AccountManager& mgr, std::string const& account_id) {
            // Call classification logic
            // This is a placeholder - actual implementation would be in AccountManager
            py::print("Classifying existing positions for account:", account_id);
        },
        py::arg("account_mgr"), py::arg("account_id"),
        "Classify existing positions as MANUAL or BOT-managed (CRITICAL for trading constraints)"
    );
}
