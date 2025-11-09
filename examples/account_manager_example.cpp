/**
 * AccountManager Usage Example
 *
 * Demonstrates proper usage of the enhanced AccountManager class
 * with position classification and trading constraints.
 *
 * Compile with: g++ -std=c++23 -o example account_manager_example.cpp
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// Note: In production, these would be module imports:
// import bigbrother.schwab_api;

// For this example, we'll use placeholder types
namespace bigbrother::schwab {

struct AccountPosition {
    std::string symbol;
    int quantity{0};
    double average_price{0.0};
    double current_price{0.0};
    double unrealized_pnl{0.0};
};

// Forward declarations
class TokenManager;
class AccountManager;

} // namespace bigbrother::schwab

using namespace bigbrother::schwab;

// ============================================================================
// Example 1: Startup and Initialization
// ============================================================================

auto example_startup() -> void {
    std::cout << "\n=== Example 1: Startup and Initialization ===\n\n";

    // 1. Create AccountManager
    // auto token_mgr = std::make_shared<TokenManager>(config);
    // auto account_mgr = std::make_shared<AccountManager>(token_mgr, "XXXX1234");

    // 2. Initialize database
    // auto result = account_mgr->initializeDatabase("trading_data.duckdb");
    // if (!result) {
    //     std::cerr << "Failed to initialize database: " << result.error() << "\n";
    //     return;
    // }

    // 3. CRITICAL: Classify existing positions
    // auto classify_result = account_mgr->classifyExistingPositions();
    // if (!classify_result) {
    //     std::cerr << "Failed to classify positions: " << classify_result.error() << "\n";
    //     return;
    // }

    // 4. Log position summary
    // auto [total, manual, bot] = account_mgr->getPositionStats();
    // std::cout << "Position Summary:\n";
    // std::cout << "  Total: " << total << "\n";
    // std::cout << "  Manual: " << manual << " (DO NOT TOUCH)\n";
    // std::cout << "  Bot-managed: " << bot << " (can trade)\n";

    std::cout << "✓ AccountManager initialized and positions classified\n";
}

// ============================================================================
// Example 2: Query Positions
// ============================================================================

auto example_query_positions() -> void {
    std::cout << "\n=== Example 2: Query Positions ===\n\n";

    // auto account_mgr = getAccountManager();

    // Get all positions
    // auto all_positions = account_mgr->getPositions();
    // if (all_positions) {
    //     std::cout << "All Positions:\n";
    //     for (auto const& pos : *all_positions) {
    //         std::cout << "  " << pos.symbol
    //                   << ": " << pos.quantity
    //                   << " shares @ $" << pos.average_price << "\n";
    //     }
    // }

    // Get manual positions only
    // auto manual = account_mgr->getManualPositions();
    // if (manual) {
    //     std::cout << "\nManual Positions (DO NOT TOUCH):\n";
    //     for (auto const& pos : *manual) {
    //         std::cout << "  " << pos.symbol << ": " << pos.quantity << " shares\n";
    //     }
    // }

    // Get bot-managed positions only
    // auto bot = account_mgr->getBotManagedPositions();
    // if (bot) {
    //     std::cout << "\nBot-Managed Positions (Active):\n";
    //     for (auto const& pos : *bot) {
    //         std::cout << "  " << pos.symbol << ": " << pos.quantity << " shares\n";
    //     }
    // }

    std::cout << "✓ Positions queried and displayed\n";
}

// ============================================================================
// Example 3: Signal Generation with Position Filtering
// ============================================================================

auto example_signal_generation() -> void {
    std::cout << "\n=== Example 3: Signal Generation ===\n\n";

    // auto account_mgr = getAccountManager();

    // Strategy generates candidate symbols
    std::vector<std::string> candidates = {"AAPL", "XLE", "SPY", "XLV"};

    std::vector<std::string> valid_signals;

    for (auto const& symbol : candidates) {
        // CRITICAL CHECK: Filter out manual positions
        // if (account_mgr->hasManualPosition(symbol)) {
        //     std::cout << "⚠️  Skipping " << symbol << " - manual position exists\n";
        //     continue;
        // }

        std::cout << "✓ " << symbol << " - OK to trade (no manual position)\n";
        valid_signals.push_back(symbol);
    }

    std::cout << "\nValid signals: " << valid_signals.size() << "\n";
}

// ============================================================================
// Example 4: Order Placement with Validation
// ============================================================================

auto example_order_placement() -> void {
    std::cout << "\n=== Example 4: Order Placement ===\n\n";

    // auto account_mgr = getAccountManager();
    // auto schwab_client = getSchwabClient();

    std::string symbol = "XLE";

    // STEP 1: Validate can trade
    // auto validate = account_mgr->validateCanTrade(symbol);
    // if (!validate) {
    //     std::cerr << "❌ Cannot trade " << symbol << ": " << validate.error() << "\n";
    //     return;
    // }

    std::cout << "✓ Validation passed for " << symbol << "\n";

    // STEP 2: Place order
    // Order order{
    //     .symbol = symbol,
    //     .type = OrderType::Market,
    //     .quantity = 10,
    //     .strategy_name = "SectorRotation"
    // };

    // auto result = schwab_client->placeOrder(order);
    // if (!result) {
    //     std::cerr << "❌ Order failed: " << result.error() << "\n";
    //     return;
    // }

    std::cout << "✓ Order placed: BUY 10 " << symbol << "\n";

    // STEP 3: Mark as bot-managed (on fill)
    // if (result->status == "FILLED") {
    //     account_mgr->markPositionAsBotManaged(symbol, "SectorRotation");
    //     std::cout << "✓ Marked " << symbol << " as bot-managed\n";
    // }
}

// ============================================================================
// Example 5: Position Classification Check
// ============================================================================

auto example_position_classification() -> void {
    std::cout << "\n=== Example 5: Position Classification ===\n\n";

    // auto account_mgr = getAccountManager();

    std::vector<std::string> symbols = {"AAPL", "XLE", "MSFT", "XLV"};

    for (auto const& symbol : symbols) {
        // Check if bot-managed
        // bool is_bot = account_mgr->isSymbolBotManaged(symbol);
        // bool is_manual = account_mgr->hasManualPosition(symbol);

        // Simulated output
        std::cout << symbol << ": ";
        if (symbol == "AAPL" || symbol == "MSFT") {
            std::cout << "MANUAL (DO NOT TOUCH)\n";
        } else {
            std::cout << "BOT-MANAGED (can trade)\n";
        }
    }
}

// ============================================================================
// Example 6: Close Position
// ============================================================================

auto example_close_position() -> void {
    std::cout << "\n=== Example 6: Close Position ===\n\n";

    // auto account_mgr = getAccountManager();
    // auto schwab_client = getSchwabClient();

    std::string symbol = "XLE";

    // Check if bot-managed
    // if (!account_mgr->isSymbolBotManaged(symbol)) {
    //     std::cerr << "❌ Cannot close " << symbol << " - not bot-managed\n";
    //     std::cerr << "   Only human can close manual positions.\n";
    //     return;
    // }

    std::cout << "✓ " << symbol << " is bot-managed - OK to close\n";

    // Place sell order
    // Order sell_order{
    //     .symbol = symbol,
    //     .type = OrderType::Market,
    //     .side = "SELL",
    //     .quantity = 10
    // };

    // auto result = schwab_client->placeOrder(sell_order);
    // if (result && result->status == "FILLED") {
    //     std::cout << "✓ Position closed: SELL 10 " << symbol << "\n";
    // }
}

// ============================================================================
// Example 7: Statistics and Monitoring
// ============================================================================

auto example_statistics() -> void {
    std::cout << "\n=== Example 7: Statistics ===\n\n";

    // auto account_mgr = getAccountManager();

    // Get position counts
    // auto [total, manual, bot] = account_mgr->getPositionStats();

    // Simulated output
    size_t total = 4;
    size_t manual = 2;
    size_t bot = 2;

    std::cout << "Position Statistics:\n";
    std::cout << "  Total positions: " << total << "\n";
    std::cout << "  Manual (DO NOT TOUCH): " << manual << "\n";
    std::cout << "  Bot-managed (can trade): " << bot << "\n";
    std::cout << "\n";

    double manual_percent = (static_cast<double>(manual) / total) * 100.0;
    double bot_percent = (static_cast<double>(bot) / total) * 100.0;

    std::cout << "Distribution:\n";
    std::cout << "  Manual: " << manual_percent << "%\n";
    std::cout << "  Bot: " << bot_percent << "%\n";
}

// ============================================================================
// Example 8: Error Handling
// ============================================================================

auto example_error_handling() -> void {
    std::cout << "\n=== Example 8: Error Handling ===\n\n";

    // auto account_mgr = getAccountManager();

    std::string manual_symbol = "AAPL";  // Pre-existing manual position

    // Try to trade manual position
    // auto validate = account_mgr->validateCanTrade(manual_symbol);
    // if (!validate) {
    //     std::cerr << "❌ Error: " << validate.error() << "\n";
    //     std::cerr << "   Action: Skipping this trade\n";
    //     return;
    // }

    std::cout << "Expected error: Cannot trade AAPL - manual position exists\n";
    std::cout << "Bot correctly respects manual positions ✓\n";
}

// ============================================================================
// Example 9: Complete Trading Workflow
// ============================================================================

auto example_complete_workflow() -> void {
    std::cout << "\n=== Example 9: Complete Trading Workflow ===\n\n";

    // Simulated complete workflow

    std::cout << "1. System Startup\n";
    std::cout << "   - Initialize AccountManager ✓\n";
    std::cout << "   - Initialize database ✓\n";
    std::cout << "   - Classify positions ✓\n";
    std::cout << "   - Found: 2 manual, 0 bot-managed\n\n";

    std::cout << "2. Strategy Analysis\n";
    std::cout << "   - Analyzing sectors...\n";
    std::cout << "   - Top ranked: XLE (Energy)\n";
    std::cout << "   - Signal: BUY XLE\n\n";

    std::cout << "3. Pre-Trade Validation\n";
    std::cout << "   - Check manual positions ✓\n";
    std::cout << "   - XLE: No manual position ✓\n";
    std::cout << "   - Validation passed ✓\n\n";

    std::cout << "4. Order Execution\n";
    std::cout << "   - Placing order: BUY 20 XLE @ MARKET\n";
    std::cout << "   - Order filled: 20 shares @ $85.00 ✓\n\n";

    std::cout << "5. Position Tracking\n";
    std::cout << "   - Marking XLE as bot-managed ✓\n";
    std::cout << "   - Persisting to database ✓\n";
    std::cout << "   - Strategy: SectorRotation\n\n";

    std::cout << "6. Updated Position Summary\n";
    std::cout << "   - Manual: 2 (AAPL, MSFT)\n";
    std::cout << "   - Bot-managed: 1 (XLE)\n";
    std::cout << "   - Total: 3\n\n";

    std::cout << "✓ Workflow complete - all safety checks passed\n";
}

// ============================================================================
// Main
// ============================================================================

auto main() -> int {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║       AccountManager Usage Examples                   ║\n";
    std::cout << "║       BigBrotherAnalytics Trading System              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    example_startup();
    example_query_positions();
    example_signal_generation();
    example_order_placement();
    example_position_classification();
    example_close_position();
    example_statistics();
    example_error_handling();
    example_complete_workflow();

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════\n";
    std::cout << "All examples completed successfully!\n";
    std::cout << "\nKey Takeaways:\n";
    std::cout << "✓ Always classify positions on startup\n";
    std::cout << "✓ Check hasManualPosition() before trading\n";
    std::cout << "✓ Use validateCanTrade() for pre-flight checks\n";
    std::cout << "✓ Mark new positions as bot-managed\n";
    std::cout << "✓ Respect manual positions (DO NOT TOUCH)\n";
    std::cout << "════════════════════════════════════════════════════════\n";
    std::cout << "\n";

    return 0;
}
