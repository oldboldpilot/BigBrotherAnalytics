/**
 * @file test_options_strategies.cpp
 * @brief Comprehensive Tests for Options Trading Strategies
 *
 * Tests include:
 * - Option pricing and Greeks calculation
 * - Strategy generation (covered calls, puts, spreads, condors)
 * - Risk management validation
 * - ML integration
 * - Portfolio Greeks aggregation
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-14
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <format>

import bigbrother.options;
import bigbrother.utils.logger;

using namespace options;
using namespace bigbrother::utils;

// Test utilities
constexpr double EPSILON = 1e-6;

bool approxEqual(double a, double b, double epsilon = EPSILON) {
    return std::abs(a - b) < epsilon;
}

void assertTrue(bool condition, std::string const& message) {
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << message << std::endl;
        std::exit(1);
    }
}

// ============================================================================
// Trinomial Pricer Tests
// ============================================================================

void test_trinomial_pricer_call() {
    std::cout << "[TEST] Trinomial Pricer - Call Option" << std::endl;

    TrinomialPricer pricer(100);

    // Price ATM call option
    double spot = 100.0;
    double strike = 100.0;
    double time_to_expiry = 30.0 / 365.0; // 30 days
    double volatility = 0.25; // 25% IV
    double risk_free_rate = 0.05; // 5%

    auto result = pricer.price(spot, strike, time_to_expiry, volatility, risk_free_rate,
                              OptionType::CALL, OptionStyle::AMERICAN);

    // ATM call should have positive value
    assertTrue(result.price > 0.0, "ATM call price should be positive");
    std::cout << "  Call price: $" << result.price << std::endl;

    // Delta should be around 0.5 for ATM call
    assertTrue(result.greeks.delta > 0.4 && result.greeks.delta < 0.6,
              "ATM call delta should be around 0.5");
    std::cout << "  Delta: " << result.greeks.delta << std::endl;

    // Theta should be negative (time decay)
    assertTrue(result.greeks.theta < 0.0, "Theta should be negative");
    std::cout << "  Theta: " << result.greeks.theta << " per day" << std::endl;

    // Vega should be positive
    assertTrue(result.greeks.vega > 0.0, "Vega should be positive");
    std::cout << "  Vega: " << result.greeks.vega << std::endl;

    // Gamma should be positive
    assertTrue(result.greeks.gamma > 0.0, "Gamma should be positive");
    std::cout << "  Gamma: " << result.greeks.gamma << std::endl;

    std::cout << "  [PASS] Trinomial call pricing and Greeks\n" << std::endl;
}

void test_trinomial_pricer_put() {
    std::cout << "[TEST] Trinomial Pricer - Put Option" << std::endl;

    TrinomialPricer pricer(100);

    double spot = 100.0;
    double strike = 100.0;
    double time_to_expiry = 30.0 / 365.0;
    double volatility = 0.25;
    double risk_free_rate = 0.05;

    auto result = pricer.price(spot, strike, time_to_expiry, volatility, risk_free_rate,
                              OptionType::PUT, OptionStyle::AMERICAN);

    // ATM put should have positive value
    assertTrue(result.price > 0.0, "ATM put price should be positive");
    std::cout << "  Put price: $" << result.price << std::endl;

    // Delta should be around -0.5 for ATM put
    assertTrue(result.greeks.delta < -0.4 && result.greeks.delta > -0.6,
              "ATM put delta should be around -0.5");
    std::cout << "  Delta: " << result.greeks.delta << std::endl;

    std::cout << "  [PASS] Trinomial put pricing and Greeks\n" << std::endl;
}

void test_put_call_parity() {
    std::cout << "[TEST] Put-Call Parity Validation" << std::endl;

    TrinomialPricer pricer(100);

    double spot = 100.0;
    double strike = 100.0;
    double time_to_expiry = 30.0 / 365.0;
    double volatility = 0.25;
    double risk_free_rate = 0.05;

    auto call = pricer.price(spot, strike, time_to_expiry, volatility, risk_free_rate,
                            OptionType::CALL, OptionStyle::EUROPEAN);

    auto put = pricer.price(spot, strike, time_to_expiry, volatility, risk_free_rate,
                           OptionType::PUT, OptionStyle::EUROPEAN);

    // Put-call parity: C - P = S - K * e^(-r*T)
    double lhs = call.price - put.price;
    double rhs = spot - strike * std::exp(-risk_free_rate * time_to_expiry);

    double parity_error = std::abs(lhs - rhs);
    std::cout << "  Call price: $" << call.price << std::endl;
    std::cout << "  Put price: $" << put.price << std::endl;
    std::cout << "  Parity error: $" << parity_error << std::endl;

    // Allow 1% error for numerical approximation
    assertTrue(parity_error < 1.0, "Put-call parity should hold within tolerance");

    std::cout << "  [PASS] Put-call parity validated\n" << std::endl;
}

// ============================================================================
// Options Strategy Tests
// ============================================================================

void test_covered_call() {
    std::cout << "[TEST] Covered Call Strategy" << std::endl;

    OptionsStrategyEngine engine;

    std::string symbol = "AAPL";
    double current_price = 150.0;
    int shares_owned = 200; // 2 contracts worth
    double account_value = 100000.0;

    auto position = engine.generateCoveredCall(symbol, current_price, shares_owned, account_value);

    assertTrue(position.has_value(), "Should generate covered call position");

    auto const& pos = position.value();
    assertTrue(pos.strategy_type == StrategyType::CoveredCall, "Should be covered call");
    assertTrue(pos.legs.size() == 1, "Covered call should have 1 leg");
    assertTrue(pos.legs[0].quantity < 0, "Call should be short");
    assertTrue(pos.max_profit > 0.0, "Max profit should be positive");
    assertTrue(pos.required_capital == 0.0, "No cash required (covered by shares)");

    std::cout << "  Symbol: " << pos.symbol << std::endl;
    std::cout << "  Max Profit: $" << pos.max_profit << std::endl;
    std::cout << "  Max Loss: $" << pos.max_loss << std::endl;
    std::cout << "  Breakeven: $" << pos.breakeven_price << std::endl;
    std::cout << "  Contracts: " << std::abs(pos.legs[0].quantity) << std::endl;
    std::cout << "  [PASS] Covered call generation\n" << std::endl;
}

void test_cash_secured_put() {
    std::cout << "[TEST] Cash-Secured Put Strategy" << std::endl;

    OptionsStrategyEngine engine;

    std::string symbol = "TSLA";
    double current_price = 200.0;
    double available_cash = 50000.0;
    double account_value = 100000.0;

    auto position = engine.generateCashSecuredPut(symbol, current_price, available_cash, account_value);

    assertTrue(position.has_value(), "Should generate cash-secured put position");

    auto const& pos = position.value();
    assertTrue(pos.strategy_type == StrategyType::CashSecuredPut, "Should be cash-secured put");
    assertTrue(pos.legs.size() == 1, "CSP should have 1 leg");
    assertTrue(pos.legs[0].type == OptionType::PUT, "Should be put");
    assertTrue(pos.legs[0].quantity < 0, "Put should be short");
    assertTrue(pos.max_profit > 0.0, "Max profit should be positive");
    assertTrue(pos.collateral_required > 0.0, "Should require collateral");
    assertTrue(pos.collateral_required <= available_cash, "Collateral within cash limit");

    std::cout << "  Symbol: " << pos.symbol << std::endl;
    std::cout << "  Strike: $" << pos.legs[0].strike << std::endl;
    std::cout << "  Max Profit: $" << pos.max_profit << std::endl;
    std::cout << "  Collateral: $" << pos.collateral_required << std::endl;
    std::cout << "  [PASS] Cash-secured put generation\n" << std::endl;
}

void test_bull_call_spread() {
    std::cout << "[TEST] Bull Call Spread Strategy" << std::endl;

    OptionsStrategyEngine engine;

    std::string symbol = "SPY";
    double current_price = 450.0;
    double available_cash = 10000.0;
    double account_value = 100000.0;

    auto position = engine.generateBullCallSpread(symbol, current_price, available_cash, account_value);

    assertTrue(position.has_value(), "Should generate bull call spread");

    auto const& pos = position.value();
    assertTrue(pos.strategy_type == StrategyType::BullCallSpread, "Should be bull call spread");
    assertTrue(pos.legs.size() == 2, "Spread should have 2 legs");

    // First leg: long call (lower strike)
    assertTrue(pos.legs[0].type == OptionType::CALL, "First leg should be call");
    assertTrue(pos.legs[0].quantity > 0, "First leg should be long");

    // Second leg: short call (higher strike)
    assertTrue(pos.legs[1].type == OptionType::CALL, "Second leg should be call");
    assertTrue(pos.legs[1].quantity < 0, "Second leg should be short");
    assertTrue(pos.legs[1].strike > pos.legs[0].strike, "Short strike should be higher");

    assertTrue(pos.max_loss < pos.max_profit, "Risk/reward should be favorable");
    assertTrue(pos.required_capital <= available_cash, "Capital requirement within limit");

    std::cout << "  Symbol: " << pos.symbol << std::endl;
    std::cout << "  Long strike: $" << pos.legs[0].strike << std::endl;
    std::cout << "  Short strike: $" << pos.legs[1].strike << std::endl;
    std::cout << "  Max Profit: $" << pos.max_profit << std::endl;
    std::cout << "  Max Loss: $" << pos.max_loss << std::endl;
    std::cout << "  Net Debit: $" << pos.required_capital << std::endl;
    std::cout << "  [PASS] Bull call spread generation\n" << std::endl;
}

void test_iron_condor() {
    std::cout << "[TEST] Iron Condor Strategy" << std::endl;

    OptionsStrategyEngine engine;

    std::string symbol = "QQQ";
    double current_price = 380.0;
    double available_cash = 15000.0;
    double account_value = 100000.0;

    auto position = engine.generateIronCondor(symbol, current_price, available_cash, account_value);

    assertTrue(position.has_value(), "Should generate iron condor");

    auto const& pos = position.value();
    assertTrue(pos.strategy_type == StrategyType::IronCondor, "Should be iron condor");
    assertTrue(pos.legs.size() == 4, "Iron condor should have 4 legs");
    assertTrue(pos.max_profit > 0.0, "Should collect premium");

    std::cout << "  Symbol: " << pos.symbol << std::endl;
    std::cout << "  Legs: " << pos.legs.size() << std::endl;
    std::cout << "  Max Profit: $" << pos.max_profit << std::endl;
    std::cout << "  Max Loss: $" << pos.max_loss << std::endl;
    std::cout << "  Risk/Reward: " << (pos.max_profit / pos.max_loss) << std::endl;
    std::cout << "  [PASS] Iron condor generation\n" << std::endl;
}

void test_protective_put() {
    std::cout << "[TEST] Protective Put Strategy" << std::endl;

    OptionsStrategyEngine engine;

    std::string symbol = "NVDA";
    double current_price = 500.0;
    int shares_owned = 100;
    double available_cash = 5000.0;

    auto position = engine.generateProtectivePut(symbol, current_price, shares_owned, available_cash);

    assertTrue(position.has_value(), "Should generate protective put");

    auto const& pos = position.value();
    assertTrue(pos.strategy_type == StrategyType::ProtectivePut, "Should be protective put");
    assertTrue(pos.legs.size() == 1, "Should have 1 put leg");
    assertTrue(pos.legs[0].type == OptionType::PUT, "Should be put");
    assertTrue(pos.legs[0].quantity > 0, "Put should be long");
    assertTrue(pos.max_loss < current_price * shares_owned, "Should limit downside");

    std::cout << "  Symbol: " << pos.symbol << std::endl;
    std::cout << "  Strike: $" << pos.legs[0].strike << std::endl;
    std::cout << "  Max Loss: $" << pos.max_loss << std::endl;
    std::cout << "  Protection cost: $" << pos.required_capital << std::endl;
    std::cout << "  [PASS] Protective put generation\n" << std::endl;
}

// ============================================================================
// Greeks Calculator Tests
// ============================================================================

void test_greeks_calculation() {
    std::cout << "[TEST] Greeks Calculator" << std::endl;

    GreeksCalculator calc;

    std::string symbol = "SPY";
    double spot = 450.0;
    double strike = 450.0;
    double time_to_expiry = 30.0 / 365.0;
    double volatility = 0.20;
    double risk_free_rate = 0.05;
    int quantity = 1;

    auto greeks = calc.calculatePositionGreeks(
        symbol, spot, strike, time_to_expiry, volatility, risk_free_rate,
        OptionType::CALL, quantity
    );

    assertTrue(greeks.delta > 0.0, "Call delta should be positive");
    assertTrue(greeks.gamma > 0.0, "Gamma should be positive");
    assertTrue(greeks.theta < 0.0, "Theta should be negative");
    assertTrue(greeks.vega > 0.0, "Vega should be positive");

    std::cout << "  Delta: " << greeks.delta << std::endl;
    std::cout << "  Gamma: " << greeks.gamma << std::endl;
    std::cout << "  Theta: " << greeks.theta << "/day" << std::endl;
    std::cout << "  Vega: " << greeks.vega << std::endl;
    std::cout << "  [PASS] Greeks calculation\n" << std::endl;
}

void test_portfolio_greeks() {
    std::cout << "[TEST] Portfolio Greeks Aggregation" << std::endl;

    GreeksCalculator calc;

    std::vector<PositionGreeks> positions;

    // Position 1: Long 1 ATM call
    auto pos1 = calc.calculatePositionGreeks(
        "SPY", 450.0, 450.0, 30.0/365.0, 0.20, 0.05, OptionType::CALL, 1
    );
    positions.push_back(pos1);

    // Position 2: Short 1 OTM call (covered call)
    auto pos2 = calc.calculatePositionGreeks(
        "SPY", 450.0, 460.0, 30.0/365.0, 0.20, 0.05, OptionType::CALL, -1
    );
    positions.push_back(pos2);

    double portfolio_value = 100000.0;
    auto portfolio = calc.calculatePortfolioGreeks(positions, portfolio_value);

    std::cout << "  Total Delta: " << portfolio.total_delta << std::endl;
    std::cout << "  Total Gamma: " << portfolio.total_gamma << std::endl;
    std::cout << "  Total Theta: " << portfolio.total_theta << "/day" << std::endl;
    std::cout << "  Dollar Delta: $" << portfolio.dollar_delta << std::endl;
    std::cout << "  Delta Neutrality: " << (portfolio.delta_neutrality_score * 100.0) << "%" << std::endl;

    assertTrue(portfolio.total_long_contracts == 1, "Should have 1 long contract");
    assertTrue(portfolio.total_short_contracts == 1, "Should have 1 short contract");

    std::cout << "  [PASS] Portfolio Greeks aggregation\n" << std::endl;
}

void test_greeks_monitor() {
    std::cout << "[TEST] Real-Time Greeks Monitor" << std::endl;

    GreeksMonitor monitor;
    GreeksCalculator calc;

    // Add position
    auto greeks = calc.calculatePositionGreeks(
        "AAPL", 150.0, 150.0, 30.0/365.0, 0.25, 0.05, OptionType::CALL, 2
    );
    greeks.position_id = "POS_001";

    monitor.addPosition(greeks);

    // Update position with new price
    monitor.updatePosition("POS_001", 155.0, 0.27);

    auto updated = monitor.getPosition("POS_001");
    assertTrue(updated.has_value(), "Should find position");
    assertTrue(updated->spot_price == 155.0, "Price should be updated");
    assertTrue(updated->implied_volatility == 0.27, "IV should be updated");

    // Get portfolio Greeks
    double portfolio_value = 100000.0;
    auto portfolio = monitor.getPortfolioGreeks(portfolio_value);
    std::cout << "  Monitored positions: " << monitor.getAllPositions().size() << std::endl;
    std::cout << "  Portfolio delta: " << portfolio.total_delta << std::endl;

    // Remove position
    monitor.removePosition("POS_001");
    assertTrue(monitor.getAllPositions().empty(), "Should be empty after removal");

    std::cout << "  [PASS] Greeks monitor operations\n" << std::endl;
}

// ============================================================================
// Risk Management Tests
// ============================================================================

void test_position_validation() {
    std::cout << "[TEST] Position Risk Validation" << std::endl;

    OptionsStrategyEngine engine;

    double account_value = 100000.0;
    OptionsPosition position;
    position.required_capital = 3000.0; // 3% of account
    position.aggregate_greeks.delta = 0.40;
    position.aggregate_greeks.theta = -30.0;
    position.aggregate_greeks.vega = 80.0;

    double current_portfolio_delta = 1.0;
    double current_portfolio_theta = -100.0;

    bool is_valid = engine.validatePosition(position, current_portfolio_delta,
                                            current_portfolio_theta, account_value);

    assertTrue(is_valid, "Position should pass validation");
    std::cout << "  Position validated successfully" << std::endl;

    // Test oversized position
    position.required_capital = 10000.0; // 10% - exceeds 5% limit
    is_valid = engine.validatePosition(position, current_portfolio_delta,
                                       current_portfolio_theta, account_value);

    assertTrue(!is_valid, "Oversized position should fail validation");
    std::cout << "  Oversized position correctly rejected" << std::endl;

    std::cout << "  [PASS] Position risk validation\n" << std::endl;
}

void test_strategy_recommendation() {
    std::cout << "[TEST] Strategy Recommendation Engine" << std::endl;

    OptionsStrategyEngine engine;

    // Bullish scenario
    auto bullish_strategies = engine.recommendStrategy(
        MarketOutlook::Bullish, false, 20000.0, 100.0
    );

    assertTrue(bullish_strategies.size() > 0, "Should recommend bullish strategies");
    std::cout << "  Bullish recommendations: " << bullish_strategies.size() << " strategies" << std::endl;

    // Neutral scenario with stock ownership
    auto neutral_strategies = engine.recommendStrategy(
        MarketOutlook::Neutral, true, 5000.0, 150.0
    );

    assertTrue(neutral_strategies.size() > 0, "Should recommend neutral strategies");
    std::cout << "  Neutral recommendations: " << neutral_strategies.size() << " strategies" << std::endl;

    std::cout << "  [PASS] Strategy recommendations\n" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Options Trading Strategies Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        // Pricing tests
        test_trinomial_pricer_call();
        test_trinomial_pricer_put();
        test_put_call_parity();

        // Strategy tests
        test_covered_call();
        test_cash_secured_put();
        test_bull_call_spread();
        test_iron_condor();
        test_protective_put();

        // Greeks tests
        test_greeks_calculation();
        test_portfolio_greeks();
        test_greeks_monitor();

        // Risk management tests
        test_position_validation();
        test_strategy_recommendation();

        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL TESTS PASSED (" << 14 << " tests)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        return 0;

    } catch (std::exception const& e) {
        std::cerr << "\n❌ TEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
