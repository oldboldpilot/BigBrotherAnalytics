/**
 * BigBrotherAnalytics - Trading Platform Interface (C++23)
 *
 * ARCHITECTURE: Platform-Agnostic Order Execution Interface
 *
 * This interface provides the abstraction layer between:
 * - OrdersManager (platform-agnostic business logic)
 * - Platform executors (Schwab, IBKR, TD Ameritrade, etc.)
 *
 * LOOSE COUPLING: OrdersManager depends ONLY on this interface,
 * NOT on any specific platform implementation.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 * Phase: 4 (Week 2 - Architecture Refactoring)
 */

// Global module fragment
module;

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.trading.platform_interface;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.trading.order_types;

export namespace bigbrother::trading {

using namespace bigbrother::types;

// ============================================================================
// Trading Platform Interface (Abstract)
// ============================================================================

/**
 * Abstract interface for trading platform integration
 *
 * DESIGN PRINCIPLE: Dependency Inversion
 * - High-level OrdersManager depends on this abstraction
 * - Low-level platform implementations (Schwab, IBKR, etc.) implement this
 * - OrdersManager is NEVER aware of concrete platform types
 *
 * EXTENSIBILITY: Adding new platforms requires:
 * 1. Create new executor class implementing TradingPlatformInterface
 * 2. Inject into OrdersManager constructor
 * 3. Zero changes to OrdersManager code
 */
class TradingPlatformInterface {
  public:
    virtual ~TradingPlatformInterface() = default;

    // ========================================================================
    // Order Execution (Platform-Specific)
    // ========================================================================

    /**
     * Submit order to trading platform
     *
     * @param order Order to place
     * @return Platform-specific order ID or error
     */
    [[nodiscard]] virtual auto submitOrder(Order const& order) -> Result<std::string> = 0;

    /**
     * Cancel order on trading platform
     *
     * @param account_id Account identifier
     * @param order_id Platform-specific order ID
     * @return Success or error
     */
    [[nodiscard]] virtual auto cancelOrder(std::string const& account_id,
                                           std::string const& order_id) -> Result<void> = 0;

    /**
     * Modify existing order on trading platform
     *
     * @param order_id Platform-specific order ID
     * @param modifications Order modifications
     * @return Updated order or error
     */
    [[nodiscard]] virtual auto modifyOrder(std::string const& order_id, Order const& modifications)
        -> Result<Order> = 0;

    /**
     * Get order status from trading platform
     *
     * @param account_id Account identifier
     * @param order_id Platform-specific order ID
     * @return Order details or error
     */
    [[nodiscard]] virtual auto getOrder(std::string const& account_id, std::string const& order_id)
        -> Result<Order> = 0;

    /**
     * Get all orders for account (with optional filtering)
     *
     * @param account_id Account identifier
     * @param symbol Optional symbol filter
     * @param status Optional status filter
     * @return Vector of orders or error
     */
    [[nodiscard]] virtual auto getOrders(std::string const& account_id,
                                         std::optional<std::string> symbol = std::nullopt,
                                         std::optional<OrderStatus> status = std::nullopt)
        -> Result<std::vector<Order>> = 0;

    // ========================================================================
    // Position Queries (Platform-Specific)
    // ========================================================================

    /**
     * Get all positions from trading platform
     *
     * @param account_id Account identifier
     * @return Vector of positions or error
     */
    [[nodiscard]] virtual auto getPositions(std::string const& account_id)
        -> Result<std::vector<Position>> = 0;

    /**
     * Get single position from trading platform
     *
     * @param account_id Account identifier
     * @param symbol Security symbol
     * @return Position details or nullopt if not found
     */
    [[nodiscard]] virtual auto getPosition(std::string const& account_id, std::string const& symbol)
        -> Result<std::optional<Position>> = 0;

    // ========================================================================
    // Platform Metadata
    // ========================================================================

    /**
     * Get platform name (e.g., "Schwab", "IBKR", "TD Ameritrade")
     */
    [[nodiscard]] virtual auto getPlatformName() const noexcept -> std::string = 0;

    /**
     * Check if platform is connected and operational
     */
    [[nodiscard]] virtual auto isConnected() const noexcept -> bool = 0;

  protected:
    // Protected constructor - force inheritance
    TradingPlatformInterface() = default;

    // Non-copyable, non-movable (polymorphic base class)
    TradingPlatformInterface(TradingPlatformInterface const&) = delete;
    auto operator=(TradingPlatformInterface const&) -> TradingPlatformInterface& = delete;
    TradingPlatformInterface(TradingPlatformInterface&&) noexcept = delete;
    auto operator=(TradingPlatformInterface&&) noexcept -> TradingPlatformInterface& = delete;
};

// ============================================================================
// Factory Pattern (Optional - for dynamic platform selection)
// ============================================================================

/**
 * Platform type identifier
 */
enum class PlatformType {
    Schwab,
    InteractiveBrokers,
    TDAmeritrade,
    Alpaca,
    TradeStation,
    Mock // For testing
};

/**
 * Factory for creating platform executors
 *
 * NOTE: Concrete factories should be implemented in platform-specific modules
 */
class PlatformFactory {
  public:
    virtual ~PlatformFactory() = default;

    /**
     * Create platform executor
     *
     * @param type Platform type
     * @param config Platform-specific configuration
     * @return Platform executor or error
     */
    [[nodiscard]] virtual auto createPlatform(PlatformType type, std::string const& config)
        -> Result<std::unique_ptr<TradingPlatformInterface>> = 0;

  protected:
    PlatformFactory() = default;
    PlatformFactory(PlatformFactory const&) = delete;
    auto operator=(PlatformFactory const&) -> PlatformFactory& = delete;
    PlatformFactory(PlatformFactory&&) noexcept = delete;
    auto operator=(PlatformFactory&&) noexcept -> PlatformFactory& = delete;
};

} // namespace bigbrother::trading
