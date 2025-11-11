/**
 * BigBrotherAnalytics - Schwab Order Executor (C++23)
 *
 * ARCHITECTURE: Platform-Specific Implementation
 *
 * This class implements TradingPlatformInterface for Schwab API.
 * It acts as an adapter between:
 * - Platform-agnostic OrdersManager (uses TradingPlatformInterface)
 * - Schwab-specific OrderManager (uses Schwab API calls)
 *
 * LOOSE COUPLING:
 * - OrdersManager (core/trading) depends ONLY on TradingPlatformInterface
 * - SchwabOrderExecutor (schwab_api) implements that interface
 * - OrdersManager is NEVER aware of SchwabOrderExecutor
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase: 4 (Week 2 - Architecture Refactoring)
 */

// Global module fragment
module;

#include <chrono>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.schwab_api.order_executor;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.trading.order_types;
import bigbrother.trading.platform_interface;
import bigbrother.schwab_api; // For OrderManager

export namespace bigbrother::schwab {

using namespace bigbrother::types;
using namespace bigbrother::utils;
using namespace bigbrother::trading;

// ============================================================================
// Schwab Order Executor (Adapter Pattern)
// ============================================================================

/**
 * Schwab-specific implementation of TradingPlatformInterface
 *
 * ADAPTER PATTERN:
 * - Implements TradingPlatformInterface (target interface)
 * - Uses OrderManager (adaptee)
 * - Translates generic trading operations to Schwab-specific calls
 *
 * DEPENDENCY INJECTION:
 * - Injected into platform-agnostic OrdersManager
 * - OrdersManager calls methods on TradingPlatformInterface
 * - This executor translates to Schwab API calls
 */
class SchwabOrderExecutor : public TradingPlatformInterface {
  public:
    // Type aliases to disambiguate between schwab::Order and trading::Order
    using Order = trading::Order;
    using OrderSide = trading::OrderSide;
    using OrderType = trading::OrderType;
    using OrderDuration = trading::OrderDuration;
    using OrderStatus = trading::OrderStatus;

    /**
     * Constructor
     *
     * @param schwab_order_manager Schwab API order manager
     * @param account_id Default Schwab account ID
     */
    explicit SchwabOrderExecutor(std::shared_ptr<OrderManager> schwab_order_manager,
                                 std::string account_id)
        : schwab_order_manager_{std::move(schwab_order_manager)},
          account_id_{std::move(account_id)} {

        Logger::getInstance().info("SchwabOrderExecutor initialized for account: {}", account_id_);
    }

    // Rule of Five - use defaults (no owned resources except shared_ptr)
    ~SchwabOrderExecutor() override = default;

    SchwabOrderExecutor(SchwabOrderExecutor const&) = delete;
    auto operator=(SchwabOrderExecutor const&) -> SchwabOrderExecutor& = delete;
    SchwabOrderExecutor(SchwabOrderExecutor&&) noexcept = delete;
    auto operator=(SchwabOrderExecutor&&) noexcept -> SchwabOrderExecutor& = delete;

    // ========================================================================
    // TradingPlatformInterface Implementation
    // ========================================================================

    /**
     * Submit order to Schwab API
     */
    [[nodiscard]] auto submitOrder(Order const& order) -> Result<std::string> override {

        Logger::getInstance().info("[Schwab] Submitting order: {} {} shares", order.symbol,
                                   order.quantity);

        // Convert generic Order to Schwab Order format
        auto schwab_order = convertToSchwabOrder(order);

        // Submit to Schwab API via OrderManager
        auto result = schwab_order_manager_->placeOrder(schwab_order);

        if (!result) {
            Logger::getInstance().error("[Schwab] Order submission failed: {}",
                                        result.error().message);
            return std::unexpected(result.error());
        }

        Logger::getInstance().info("[Schwab] Order submitted successfully: {}", *result);
        return result;
    }

    /**
     * Cancel order on Schwab API
     */
    [[nodiscard]] auto cancelOrder(std::string const& account_id, std::string const& order_id)
        -> Result<void> override {

        Logger::getInstance().info("[Schwab] Canceling order: {}", order_id);

        // Delegate to Schwab OrderManager
        auto result = schwab_order_manager_->cancelOrder(order_id);

        if (!result) {
            Logger::getInstance().error("[Schwab] Order cancellation failed: {}",
                                        result.error().message);
            return std::unexpected(result.error());
        }

        Logger::getInstance().info("[Schwab] Order canceled successfully: {}", order_id);
        return result;
    }

    /**
     * Modify order on Schwab API
     */
    [[nodiscard]] auto modifyOrder(std::string const& order_id, Order const& modifications)
        -> Result<Order> override {

        Logger::getInstance().info("[Schwab] Modifying order: {}", order_id);

        // Schwab API: Cancel old order and place new one
        // (Schwab doesn't support direct order modification)
        auto cancel_result = schwab_order_manager_->cancelOrder(order_id);
        if (!cancel_result) {
            return makeError<Order>(cancel_result.error().code,
                                    "Failed to cancel old order: " + cancel_result.error().message);
        }

        // Place new order with modifications
        auto schwab_order = convertToSchwabOrder(modifications);
        auto place_result = schwab_order_manager_->placeOrder(schwab_order);
        if (!place_result) {
            return makeError<Order>(place_result.error().code, "Failed to place modified order: " +
                                                                   place_result.error().message);
        }

        // Return modified order with new ID
        Order modified_order = modifications;
        modified_order.order_id = *place_result;

        Logger::getInstance().info("[Schwab] Order modified successfully: {} -> {}", order_id,
                                   *place_result);
        return modified_order;
    }

    /**
     * Get order details from Schwab API
     */
    [[nodiscard]] auto getOrder(std::string const& account_id, std::string const& order_id)
        -> Result<Order> override {

        Logger::getInstance().info("[Schwab] Getting order: {}", order_id);

        // Query Schwab OrderManager for order status
        auto status_result = schwab_order_manager_->getOrderStatus(order_id);
        if (!status_result) {
            return makeError<Order>(status_result.error().code,
                                    "Order not found: " + status_result.error().message);
        }

        // Get order from internal storage
        // NOTE: In production, this should query Schwab API directly
        auto active_orders = schwab_order_manager_->getActiveOrders();
        for (auto const& schwab_order : active_orders) {
            if (schwab_order.order_id == order_id) {
                return convertFromSchwabOrder(schwab_order);
            }
        }

        return makeError<Order>(ErrorCode::RuntimeError, "Order not found in active orders");
    }

    /**
     * Get all orders from Schwab API
     */
    [[nodiscard]] auto getOrders(std::string const& account_id,
                                 std::optional<std::string> symbol = std::nullopt,
                                 std::optional<OrderStatus> status = std::nullopt)
        -> Result<std::vector<Order>> override {

        Logger::getInstance().info("[Schwab] Getting orders for account: {}", account_id);

        // Get all active orders from Schwab OrderManager
        auto schwab_orders = schwab_order_manager_->getActiveOrders();

        // Convert to generic Order format
        std::vector<Order> orders;
        for (auto const& schwab_order : schwab_orders) {
            // Apply filters if specified
            if (symbol && schwab_order.symbol != *symbol) {
                continue;
            }
            // Note: status filter would need conversion from trading::OrderStatus to
            // schwab::OrderStatus
            if (status) {
                auto schwab_status = static_cast<::bigbrother::schwab::OrderStatus>(*status);
                if (schwab_order.status != schwab_status) {
                    continue;
                }
            }

            orders.push_back(convertFromSchwabOrder(schwab_order));
        }

        Logger::getInstance().info("[Schwab] Retrieved {} orders", orders.size());
        return orders;
    }

    /**
     * Get positions from Schwab API
     *
     * NOTE: This should delegate to AccountClient for position data
     */
    [[nodiscard]] auto getPositions(std::string const& account_id)
        -> Result<std::vector<trading::Position>> override {

        Logger::getInstance().info("[Schwab] Getting positions for account: {}", account_id);

        // TODO: Integrate with AccountClient to get actual positions from Schwab API
        // For now, return empty vector
        Logger::getInstance().warn("[Schwab] Position retrieval not yet implemented");
        return std::vector<trading::Position>{};
    }

    /**
     * Get single position from Schwab API
     */
    [[nodiscard]] auto getPosition(std::string const& account_id, std::string const& symbol)
        -> Result<std::optional<trading::Position>> override {

        Logger::getInstance().info("[Schwab] Getting position: {} for account: {}", symbol,
                                   account_id);

        // TODO: Integrate with AccountClient
        Logger::getInstance().warn("[Schwab] Position retrieval not yet implemented");
        return std::optional<trading::Position>{};
    }

    // ========================================================================
    // Platform Metadata
    // ========================================================================

    [[nodiscard]] auto getPlatformName() const noexcept -> std::string override { return "Schwab"; }

    [[nodiscard]] auto isConnected() const noexcept -> bool override {
        // Check if OrderManager is available
        return schwab_order_manager_ != nullptr;
    }

  private:
    // ========================================================================
    // Type Conversion Helpers
    // ========================================================================

    /**
     * Convert generic Order to Schwab Order format
     *
     * NOTE: Schwab Order type (schwab_api.cppm) has fewer fields than generic Order
     * Schwab Order fields: order_id, symbol, type, duration, quantity,
     *                      limit_price, stop_price, status, created_at, updated_at
     *
     * Missing in Schwab: account_id, side, filled_quantity, trail_amount,
     *                    avg_fill_price, strategy_name, dry_run, rejection_reason, filled_at
     */
    [[nodiscard]] auto convertToSchwabOrder(trading::Order const& generic_order) const
        -> ::bigbrother::schwab::Order {
        ::bigbrother::schwab::Order schwab_order;

        // Copy common fields
        schwab_order.order_id = generic_order.order_id;
        schwab_order.symbol = generic_order.symbol;
        schwab_order.quantity = generic_order.quantity;
        schwab_order.limit_price = generic_order.limit_price;
        schwab_order.stop_price = generic_order.stop_price;

        // Convert timestamps: chrono::time_point -> Timestamp (int64_t milliseconds)
        schwab_order.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      generic_order.created_at.time_since_epoch())
                                      .count();
        schwab_order.updated_at = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      generic_order.updated_at.time_since_epoch())
                                      .count();

        // Convert enums (same underlying values, different namespaces)
        schwab_order.type = static_cast<::bigbrother::schwab::OrderType>(generic_order.type);
        schwab_order.duration =
            static_cast<::bigbrother::schwab::OrderDuration>(generic_order.duration);
        schwab_order.status = static_cast<::bigbrother::schwab::OrderStatus>(generic_order.status);

        return schwab_order;
    }

    /**
     * Convert Schwab Order to generic Order format
     *
     * Sets default values for fields that don't exist in Schwab Order
     */
    [[nodiscard]] auto convertFromSchwabOrder(::bigbrother::schwab::Order const& schwab_order) const
        -> trading::Order {
        trading::Order generic_order;

        // Copy common fields
        generic_order.order_id = schwab_order.order_id;
        generic_order.symbol = schwab_order.symbol;
        generic_order.quantity = schwab_order.quantity;
        generic_order.limit_price = schwab_order.limit_price;
        generic_order.stop_price = schwab_order.stop_price;

        // Convert timestamps: Timestamp (int64_t milliseconds) -> chrono::time_point
        generic_order.created_at = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(schwab_order.created_at));
        generic_order.updated_at = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(schwab_order.updated_at));

        // Convert enums
        generic_order.type = static_cast<trading::OrderType>(schwab_order.type);
        generic_order.duration = static_cast<trading::OrderDuration>(schwab_order.duration);
        generic_order.status = static_cast<trading::OrderStatus>(schwab_order.status);

        // Set defaults for fields not in Schwab Order
        generic_order.account_id = account_id_;       // Use executor's account ID
        generic_order.side = trading::OrderSide::Buy; // Default (should be set by caller)
        generic_order.filled_quantity = 0.0;
        generic_order.trail_amount = 0.0;
        generic_order.avg_fill_price = 0.0;
        generic_order.strategy_name = "";
        generic_order.dry_run = false;
        generic_order.rejection_reason = "";
        generic_order.filled_at = std::chrono::system_clock::time_point{};

        return generic_order;
    }

    std::shared_ptr<OrderManager> schwab_order_manager_;
    std::string account_id_;
};

} // namespace bigbrother::schwab
