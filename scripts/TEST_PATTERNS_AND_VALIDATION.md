# Trading Engine Test - Patterns and Validation Rules

## Overview

This document describes the patterns that the test script searches for in application logs to validate correct operation of the trading engine.

## Log Pattern Reference

### 1. Engine Initialization

**Pattern**: `BigBrotherAnalytics Trading Engine`

**Expected Output**:
```
║        BigBrotherAnalytics Trading Engine v1.0          ║
```

**Validates**: Core engine bootstrap is working

**Why It Matters**:
- Confirms executable runs without crashing
- Indicates logger is initialized
- Shows build succeeded

---

### 2. Configuration Loading

**Pattern**: `Loading configuration from`

**Expected Output**:
```
[INFO] Loading configuration from: configs/config.yaml
```

**Validates**: Configuration system initialization

**Why It Matters**:
- Confirms config file path is found
- Shows config loading mechanism works
- Indicates working directory is correct

---

### 3. Paper Trading Mode

**Pattern**: `PAPER TRADING MODE`

**Expected Output**:
```
═══════════════════════════════════════════════════════════
    PAPER TRADING MODE - NO REAL MONEY AT RISK
═══════════════════════════════════════════════════════════
```

**Validates**: Safety feature is enabled

**Why It Matters**:
- Confirms real money is not at risk
- Shows configuration check passed
- Critical for testing environments

**Requirement**: Must be enabled for integration tests

---

### 4. Database Initialization

**Pattern**: `Database initialized`

**Expected Output**:
```
[INFO] Database initialized: data/bigbrother.duckdb
```

**Validates**: Data persistence layer is ready

**Why It Matters**:
- Confirms DuckDB connection works
- Shows database file is accessible
- Indicates data storage is available

**Paths Checked**:
- `data/bigbrother.duckdb` (main database)
- `logs/` directory (log files)

---

### 5. Strategy Registration

**Pattern**: `Strategies registered` + Multiple strategy entries

**Expected Output**:
```
[INFO] Strategies registered:
[INFO]   - DeltaNeutralStraddle: Strategy
[INFO]   - DeltaNeutralStrangle: Strategy
[INFO]   - VolatilityArbitrage: Strategy
```

**Validates**: All trading strategies load correctly

**Why It Matters**:
- Confirms strategy factory works
- Shows all 3 strategies available:
  1. Delta Neutral Straddle
  2. Delta Neutral Strangle
  3. Volatility Arbitrage
- Indicates strategy manager initialization

**Minimum Requirement**: At least 1 strategy registered

---

### 6. Trading Cycle Execution

**Pattern**: `Trading Cycle Start`

**Expected Output**:
```
[DEBUG] ═══ Trading Cycle Start ═══
```

**Validates**: Main trading loop begins

**Why It Matters**:
- Confirms run() method executes
- Shows infinite loop mechanism works
- Indicates cycle frequency respected

**Frequency**: Should appear every `trading.cycle_interval_ms` milliseconds

---

### 7. Signal Generation

**Pattern**: `Generated.*trading signals`

**Expected Output**:
```
[INFO] Generated 2 trading signals
```

**Validates**: Strategy signal generation system

**Why It Matters**:
- Confirms strategies produce signals
- Shows signal filtering works
- Indicates market data integration

**Optional Test**: Not failing if 0 signals (market conditions may not trigger)

**Expected Signal Count**: 0-10 per cycle (depends on strategy parameters)

---

### 8. Position Tracking

**Pattern**: `Updating positions` OR `Retrieved.*positions`

**Expected Output**:
```
[DEBUG] Updating positions and P&L...
[DEBUG] Retrieved 3 current positions
```

**Validates**: Position tracking system

**Why It Matters**:
- Confirms Schwab API integration
- Shows position retrieval works
- Indicates P&L calculation

**Optional Test**: Not failing if 0 positions (first run typically has no positions)

**Related Method**: `updatePositions()` in main.cpp line 385

---

### 9. Stop Loss Checking

**Pattern**: `Checking stop losses`

**Expected Output**:
```
[DEBUG] Checking stop losses...
```

**Validates**: Risk management system

**Why It Matters**:
- Confirms stop loss routine runs
- Shows risk controls are active
- Indicates `checkStopLosses()` executes

**Optional Test**: Not failing if no output (only runs when positions exist)

**Related Method**: `checkStopLosses()` in main.cpp line 388

---

### 10. Account Information

**Pattern**: `Account.*buying power` OR `getAccountInfo`

**Expected Output**:
```
[DEBUG] Account: $30000.00 total, $20000.00 buying power
```

**Validates**: Schwab API account data retrieval

**Why It Matters**:
- Confirms Schwab API client initialization
- Shows account balance retrieval
- Indicates OAuth authentication works

**Related Method**: `buildContext()` in main.cpp line 315

**Note**: May be mocked in dry-run mode

---

### 11. Risk Management

**Pattern**: `risk` OR `Risk` (case-insensitive)

**Expected Output**:
```
[INFO] Risk limits initialized for $30,000 account
[DEBUG] Portfolio risk: Heat=5%, Daily Loss=$450 remaining
```

**Validates**: Portfolio risk monitoring

**Why It Matters**:
- Confirms risk manager is active
- Shows position risk calculation
- Indicates Kelly Criterion sizing

**Risk Parameters** (from config.yaml):
- Account Value: $30,000
- Max Daily Loss: $900 (3%)
- Max Position Size: $1,500 (5%)
- Max Portfolio Heat: 15%

---

### 12. Error-Free Execution

**Pattern Count**: `ERROR` entries = 0

**Good Output**:
```
[✓] No errors in execution
Error Count: 0
```

**Bad Output** (fails test):
```
[ERROR] Failed to connect to Schwab API
[ERROR] Position update failed
```

**Validates**: Clean execution with no runtime errors

**Why It Matters**:
- Confirms no exceptions occurred
- Shows all operations succeeded
- Indicates system is stable

**Note**: WARNING entries are acceptable, ERROR entries are not

---

### 13. Graceful Shutdown

**Pattern**: `Shutdown` OR `shutdown`

**Expected Output**:
```
[INFO] Shutdown signal received, closing positions and exiting...
[INFO] Shutdown complete
```

**Validates**: Clean shutdown process

**Why It Matters**:
- Confirms signal handling works
- Shows resource cleanup
- Indicates proper termination

**Trigger**: Timeout (SIGTERM) or Ctrl+C (SIGINT)

**Optional Test**: Not failing if not found (timeout kills process)

---

## Test Validation Rules

### Pass Criteria

A test PASSES if:
1. Pattern is found in log file within timeout
2. Pattern appears in expected format
3. Count/value is within acceptable range

### Fail Criteria

A test FAILS if:
1. Pattern is NOT found within timeout
2. Pattern appears malformed/unexpected
3. Count/value is outside acceptable range
4. Log file is empty or corrupt

### Optional vs Required

| Test | Type | Passes When |
|------|------|-------------|
| Engine Init | Required | Pattern found |
| Config Load | Required | Pattern found |
| Paper Trading | Required | Pattern found |
| Database | Required | Pattern found |
| Strategies | Required | ≥1 strategy |
| Trading Cycle | Required | Pattern found |
| Signals | Optional | Pattern found OR 0 signals OK |
| Positions | Optional | Pattern found OR 0 positions OK |
| Stop Loss | Optional | Pattern found OR no positions OK |
| Account Info | Optional | Pattern found OR mocked OK |
| Risk Mgmt | Required | Pattern found |
| Error Check | Required | Count = 0 |
| Shutdown | Optional | Pattern found OR timeout OK |

---

## Log Output Example

Here's a complete example of expected log output during test:

```
═══════════════════════════════════════════════════════════
    BigBrotherAnalytics Trading Engine v1.0
═══════════════════════════════════════════════════════════

[INFO] Loading configuration from: configs/config.yaml
[INFO] Logger initialized: logs/bigbrother.log (level: info)
[INFO] Database initialized: data/bigbrother.duckdb

═══════════════════════════════════════════════════════════
    PAPER TRADING MODE - NO REAL MONEY AT RISK
═══════════════════════════════════════════════════════════

[INFO] Initializing Schwab API client...
[INFO] Initializing trading strategies...
[INFO] Strategies registered:
[INFO]   - DeltaNeutralStraddle: Strategy
[INFO]   - DeltaNeutralStrangle: Strategy
[INFO]   - VolatilityArbitrage: Strategy

[INFO] Initialization complete!

[INFO] Starting trading engine...

[DEBUG] ═══ Trading Cycle Start ═══
[DEBUG] Account: $30000.00 total, $20000.00 buying power
[DEBUG] Retrieved 0 current positions
[INFO] Generated 2 trading signals
[INFO] Executed 1 trade
[INFO]   Order placed: ORD-2024-000123
[DEBUG] Updating positions and P&L...
[DEBUG] Checking stop losses...
[DEBUG] ═══ Trading Cycle End ═══

[DEBUG] ═══ Trading Cycle Start ═══
[DEBUG] Account: $30000.00 total, $19900.00 buying power
[DEBUG] Retrieved 1 current positions
[INFO] Generated 1 trading signals
[DEBUG] Updating positions and P&L...
[DEBUG] Total unrealized P&L: $50.00 (bot-managed positions: 1)
[DEBUG] Checking stop losses...
[DEBUG] ═══ Trading Cycle End ═══

[INFO] Shutdown signal received, closing positions and exiting...
[INFO] Shutdown complete
```

---

## Pattern Search Examples

### How the Script Searches

Using bash/grep patterns:

```bash
# Simple pattern search
if grep -q "PAPER TRADING MODE" "$log_file"; then
    echo "Paper trading confirmed"
fi

# Regex pattern search
if grep -q "Generated.*trading signals" "$log_file"; then
    echo "Signal generation working"
fi

# Count matches
count=$(grep -c "ERROR" "$log_file")
if [ "$count" -eq 0 ]; then
    echo "No errors found"
fi

# Extract value
value=$(grep "Account:" "$log_file" | grep -oE '[0-9]+\.[0-9]{2}' | head -1)
echo "Account value: $value"
```

### Common Pattern Types

**Exact Match**:
```bash
grep "Database initialized" log.txt
```

**Flexible Match** (regex):
```bash
grep "Generated.*trading signals" log.txt
```

**Case-Insensitive**:
```bash
grep -i "error" log.txt
```

**Count Lines**:
```bash
grep -c "Trading Cycle Start" log.txt
```

---

## Timeout Behavior

### Default Timeout: 30 seconds

The test runs the executable for 30 seconds, then:
1. Sends SIGTERM signal
2. Executable catches signal → executes shutdown
3. Process exits with code 0 (expected timeout)
4. Test validates patterns in log

### Adjusting Timeout

```bash
# 60 seconds
./scripts/test_trading_engine.sh --timeout 60

# 120 seconds
./scripts/test_trading_engine.sh --timeout 120
```

### Why Timeout?

- Prevents infinite loop
- Allows test to complete
- Tests graceful shutdown
- Simulates production behavior

---

## Expected Cycle Count

In default 30-second test with 60-second cycle interval:

- **1 full cycle** would normally execute
- **2-3 partial cycles** may appear in logs
- **Signal count**: 0-10 per cycle
- **Order count**: 0-5 per cycle

If you see fewer cycles, check:
1. `trading.cycle_interval_ms` in config
2. Build/initialization time
3. System performance

---

## Log File Locations

During test execution, three log locations exist:

### 1. Real-time Console Output
```
Displayed directly in terminal
```

### 2. Temporary Log (During Test)
```
/tmp/tmp.XXXXXX (deleted after test)
```

### 3. Final Test Log
```
logs/test_run_<timestamp>.log (persisted)
logs/bigbrother.log (main application log)
```

### Viewing Logs

```bash
# View latest test
tail -50 logs/test_run_*.log | sort | tail -50

# View main log
tail -100 logs/bigbrother.log

# Search in logs
grep "ERROR" logs/*.log

# Watch real-time
tail -f logs/bigbrother.log
```

---

## Troubleshooting Pattern Failures

### Pattern Not Found (but should be there)

1. **Check log file exists**:
   ```bash
   test -f logs/test_run_*.log && echo "Found" || echo "Missing"
   ```

2. **Search manually**:
   ```bash
   grep -i "pattern" logs/bigbrother.log
   ```

3. **Check for typos in pattern**:
   - Case sensitivity?
   - Regex special characters?
   - Whitespace differences?

4. **Increase timeout**:
   ```bash
   ./scripts/test_trading_engine.sh --timeout 60
   ```

### Too Many Error Lines

If error count > 0:

1. **View errors**:
   ```bash
   grep "ERROR" logs/bigbrother.log
   ```

2. **Common causes**:
   - Missing Schwab API credentials
   - Database connection failed
   - Strategy loading failed
   - Configuration syntax error

3. **Fix and retry**:
   ```bash
   ./scripts/test_trading_engine.sh --clean
   ```

### Missing Signals/Positions

Normal for first runs:
- Signals depend on market conditions
- Positions depend on orders executing
- Optional tests won't fail

---

## Customizing Test Patterns

To modify validation patterns, edit:

```bash
# Edit script
nano scripts/test_trading_engine.sh

# Find test functions
grep "test_.*() {" scripts/test_trading_engine.sh

# Modify pattern in function
# Example: Change "PAPER TRADING MODE" to "PAPER MODE"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-09 | Initial release |

---

## Related Documentation

- `TRADING_ENGINE_TEST_README.md` - Full documentation
- `TEST_QUICK_START.md` - Quick reference
- `scripts/test_trading_engine.sh` - Script source
- `src/main.cpp` - Trading engine implementation
- `configs/config.yaml` - Configuration reference

---

**Last Updated**: 2025-11-09
**Pattern Version**: 1.0
**Status**: Production Ready
