# E2E Test Quick Reference

## File Location
```
tests/cpp/test_schwab_e2e_workflow.cpp
```

## Quick Commands

### Build
```bash
cmake --build build --target test_schwab_e2e_workflow
```

### Run All Tests
```bash
./build/tests/cpp/test_schwab_e2e_workflow
```

### Run Specific Test
```bash
./build/tests/cpp/test_schwab_e2e_workflow --gtest_filter=SchwabE2EWorkflowTest.Scenario1_RejectManualPosition
```

## Test Scenarios

| # | Name | Symbol | Type | Expected Result |
|---|------|--------|------|-----------------|
| 1 | RejectManualPosition | AAPL | Manual | ❌ REJECT |
| 2 | AcceptNewSymbol | XLE | New | ✅ ACCEPT |
| 3 | CloseOwnPosition | XLE | Bot-managed | ✅ ACCEPT |
| 4 | TradeBotManagedPosition | SPY | Bot-managed | ✅ ACCEPT |
| 5 | DryRunMode | QQQ | New (dry-run) | ✅ LOG ONLY |
| 6 | CompleteWorkflow | Multiple | Mixed | ✅ FULL FLOW |
| 7 | RiskManagerIntegration | XLE/XLV/XLI | New | ✅ RISK CHECKS |

## Safety Rules Tested

### ❌ REJECT These
- Trading existing manual positions
- Closing existing manual positions
- Adding to manual positions

### ✅ ACCEPT These
- Trading NEW symbols (not in portfolio)
- Trading bot-managed positions
- Closing bot-managed positions
- Dry-run orders (log only)

## Key Mock Components

### MockAccountManager
```cpp
account_mgr->addManualPosition("AAPL", 10, 150.00);
account_mgr->addBotPosition("XLE", 20, 80.00, "SectorRotation");
auto pos = account_mgr->getPosition("AAPL");
```

### MockOrderManager
```cpp
auto result = order_mgr->placeOrder(order);
order_mgr->setDryRunMode(true);
auto log = order_mgr->getComplianceLog();
```

## Typical Order Flow

```cpp
// 1. Create order
Order order{};
order.symbol = "XLE";
order.side = OrderSide::Buy;
order.quantity = 20;
order.limit_price = 80.00;
order.strategy_name = "SectorRotation";

// 2. Place order (with safety checks)
auto result = order_mgr->placeOrder(order);

// 3. Check result
if (result.has_value()) {
    // Success
    Logger::info("Order filled: {}", result->order_id);
} else {
    // Rejected
    Logger::error("Order rejected: {}", result.error());
}
```

## Expected Output Pattern

```
=== SCENARIO X: Description ===
[INFO] Setup: ...
[INFO/ERROR] Action: ...
✅ SCENARIO X PASSED: Message
```

## Common Assertions

```cpp
// Order accepted
ASSERT_TRUE(result.has_value());
EXPECT_EQ(result->symbol, "XLE");

// Order rejected
ASSERT_FALSE(result.has_value());
EXPECT_TRUE(result.error().find("manual position") != std::string::npos);

// Position created
auto pos = account_mgr->getPosition("XLE");
ASSERT_TRUE(pos.has_value());
EXPECT_TRUE(pos->is_bot_managed);

// Compliance logged
auto log = order_mgr->getComplianceLog();
EXPECT_GT(log.size(), 0);
```

## Debugging Tips

### Enable Verbose Logging
```bash
./test_schwab_e2e_workflow --gtest_verbose
```

### Filter Specific Test
```bash
./test_schwab_e2e_workflow --gtest_filter="*Scenario1*"
```

### List All Tests
```bash
./test_schwab_e2e_workflow --gtest_list_tests
```

### Repeat Test N Times
```bash
./test_schwab_e2e_workflow --gtest_repeat=10
```

## Test Statistics

- **Total Tests:** 7
- **Total Lines:** 792
- **Mock Classes:** 2 (AccountManager, OrderManager)
- **Safety Checks:** 4 (manual position, close validation, signal filtering, position tagging)
- **Compliance Events:** Logged for every order

## Quick Checklist

Before committing changes to trading code:

- [ ] Run all E2E tests
- [ ] Verify all 7 scenarios pass
- [ ] Check compliance log for violations
- [ ] Verify manual positions protected
- [ ] Test dry-run mode works
- [ ] Validate risk manager integration

## One-Liner Test Run

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics && \
cmake --build build --target test_schwab_e2e_workflow && \
./build/tests/cpp/test_schwab_e2e_workflow
```

---

**Last Updated:** 2025-11-09
**Test Pass Rate:** 100% (7/7)
