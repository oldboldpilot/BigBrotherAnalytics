# Trading Engine Integration Test Script

## Overview

The `test_trading_engine.sh` script provides a comprehensive end-to-end integration test for the BigBrotherAnalytics trading engine. It validates the complete trading pipeline in a single, reproducible execution.

## Features

- **Full Pipeline Testing**: Builds, configures, and runs the complete trading engine
- **Paper Trading Validation**: Ensures safe dry-run mode is enabled
- **Log-Based Monitoring**: Monitors execution logs for expected behavior patterns
- **13 Validation Tests**: Comprehensive test suite covering all key components
- **Clear Reporting**: Detailed pass/fail status with results summary
- **Error Handling**: Robust error checking and graceful failure modes
- **Repeatable Execution**: Can be run multiple times for regression testing
- **Timeout Protection**: Prevents hanging with configurable timeout

## Quick Start

### Basic Usage

Run the full test suite with default settings:

```bash
./scripts/test_trading_engine.sh
```

### With Options

```bash
# Clean rebuild before testing
./scripts/test_trading_engine.sh --clean

# Extended timeout for slower systems (60 seconds)
./scripts/test_trading_engine.sh --timeout 60

# Skip rebuild, run with existing binary
./scripts/test_trading_engine.sh --no-build

# Verbose output for debugging
./scripts/test_trading_engine.sh --verbose

# Combination of options
./scripts/test_trading_engine.sh --clean --timeout 45 --verbose
```

## Test Suite Details

The script runs 13 comprehensive validation tests:

### 1. **Engine Initialization** ✓
Verifies the trading engine starts and displays initialization banner
- **Pattern**: "BigBrotherAnalytics Trading Engine"
- **Validates**: Core engine bootstrap

### 2. **Configuration Loading** ✓
Confirms configuration file is loaded successfully
- **Pattern**: "Loading configuration from"
- **Validates**: Config system initialization

### 3. **Paper Trading Mode** ✓
Ensures paper trading (dry-run) is enabled
- **Pattern**: "PAPER TRADING MODE"
- **Validates**: Safety - no real funds at risk

### 4. **Database Initialization** ✓
Verifies DuckDB database is initialized
- **Pattern**: "Database initialized"
- **Validates**: Data persistence layer

### 5. **Strategy Registration** ✓
Confirms all trading strategies are registered
- **Pattern**: "Strategies registered"
- **Validates**: Strategy system setup
- **Strategies**: Straddle, Strangle, Volatility Arbitrage

### 6. **Trading Cycle Execution** ✓
Validates main trading loop begins
- **Pattern**: "Trading Cycle Start"
- **Validates**: Trading loop integration

### 7. **Signal Generation** ✓
Checks for trading signal generation
- **Pattern**: "Generated.*trading signals"
- **Validates**: Strategy signal generation
- **Note**: Optional - depends on market conditions

### 8. **Position Tracking** ✓
Verifies position tracking system
- **Pattern**: "Updating positions|Retrieved.*positions"
- **Validates**: Position management
- **Note**: Optional - depends on account state

### 9. **Stop Loss Checking** ✓
Confirms stop loss monitoring routine
- **Pattern**: "Checking stop losses"
- **Validates**: Risk management
- **Note**: Optional - only runs if positions exist

### 10. **Account Information** ✓
Validates Schwab API account data retrieval
- **Pattern**: "Account.*buying power|getAccountInfo"
- **Validates**: API integration
- **Note**: May be mocked in dry-run mode

### 11. **Risk Management** ✓
Ensures risk management system is active
- **Pattern**: "risk|Risk"
- **Validates**: Portfolio risk monitoring

### 12. **Error-Free Execution** ✓
Confirms no errors occurred during run
- **Pattern**: Count of "ERROR" entries
- **Validates**: Clean execution
- **Passes**: When error count = 0

### 13. **Graceful Shutdown** ✓
Verifies clean shutdown process
- **Pattern**: "Shutdown|shutdown"
- **Validates**: Proper resource cleanup

## Command-Line Options

```
--clean           Force clean rebuild (removes build directory)
--no-build        Skip build step, use existing binary
--verbose         Enable verbose output during execution
--timeout SECS    Set test timeout in seconds (default: 30)
--help            Display help message
```

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All tests passed | Check results - success! |
| 1 | Build failed | Review build errors, check CMake output |
| 2 | Test setup failed | Verify config file, check paths |
| 3 | Tests failed | Review failed assertions in output |
| 4 | Timeout | Increase timeout with `--timeout` option |

## Output Files

Test execution creates log files in the `logs/` directory:

```
logs/
├── bigbrother.log              # Main application log
└── test_run_<timestamp>.log    # Timestamped test run log
```

View logs during/after execution:

```bash
# Watch logs in real-time
tail -f logs/bigbrother.log

# View last test run
tail -100 logs/test_run_*.log | head -50

# Search for specific patterns
grep "Trading Cycle" logs/bigbrother.log
grep "WARN\|ERROR" logs/bigbrother.log
```

## Configuration File

The test uses the project configuration file: `configs/config.yaml`

**Critical settings for testing:**

```yaml
trading:
  paper_trading: true          # MUST be true for safe testing
  cycle_interval_ms: 60000     # Trading cycle frequency

risk:
  account_value: 30000.0       # Paper account size
  max_daily_loss: 900.0        # 3% daily loss limit

schwab:
  # OAuth credentials from environment variables
  client_id: "${SCHWAB_CLIENT_ID}"
  client_secret: "${SCHWAB_CLIENT_SECRET}"
```

**Verify configuration before running:**

```bash
# Check if paper trading is enabled
grep "paper_trading" configs/config.yaml

# Verify all required sections exist
grep "^database:\|^logging:\|^trading:\|^schwab:\|^risk:" configs/config.yaml
```

## Build Integration

The test script integrates with the existing CMake build system:

1. **Configuration**: Runs `cmake -DCMAKE_BUILD_TYPE=Release`
2. **Build**: Compiles with all available parallel jobs
3. **Compilation**: Uses Clang C++23 with libc++
4. **Output**: Binary placed at `build/bin/bigbrother`

### Clean Rebuild Example

For a fresh test without cached artifacts:

```bash
./scripts/test_trading_engine.sh --clean
```

This:
1. Removes entire `build/` directory
2. Re-runs CMake configuration
3. Recompiles all source files
4. Runs the full test suite

### Skip Build Example

For rapid testing iteration:

```bash
./scripts/test_trading_engine.sh --no-build
```

Use when:
- Code hasn't changed
- Testing different configurations
- Running the test multiple times
- Debugging log output

## Troubleshooting

### Build Fails

```bash
# Check CMake configuration
cd build
cmake --verbose

# Review build errors
grep -i error build_output.log

# Clean and retry
cd ..
./scripts/test_trading_engine.sh --clean
```

### Timeout Errors

Increase timeout for slower systems:

```bash
./scripts/test_trading_engine.sh --timeout 90
```

Common reasons:
- Slow disk I/O
- High system load
- Complex compilation
- First-time module precompilation

### Missing Configuration

```bash
# Verify config file exists
test -f configs/config.yaml && echo "Config found" || echo "Missing config"

# Check for required sections
grep "^schwab:" configs/config.yaml

# Verify OAuth environment variables
echo "Client ID: $SCHWAB_CLIENT_ID"
echo "Client Secret: $SCHWAB_CLIENT_SECRET"
```

### Log Files Not Created

```bash
# Create logs directory
mkdir -p logs

# Check directory permissions
ls -la logs/

# Verify database directory
mkdir -p data
ls -la data/
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Trading Engine Tests

on: [push, pull_request]

jobs:
  integration-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Trading Engine Tests
        run: |
          ./scripts/test_trading_engine.sh --timeout 60
```

### GitLab CI Example

```yaml
trading_engine_tests:
  stage: test
  script:
    - ./scripts/test_trading_engine.sh --timeout 60
  artifacts:
    paths:
      - logs/
    expire_in: 1 week
```

## Test Execution Example

```bash
$ ./scripts/test_trading_engine.sh

╔═════════════════════════════════════════════════════════════╗
║   Trading Engine End-to-End Integration Test               ║
║   BigBrotherAnalytics Automated Testing Suite              ║
╚═════════════════════════════════════════════════════════════╝

[INFO] Project Root:  /home/user/Development/BigBrotherAnalytics
[INFO] Build Dir:     /home/user/Development/BigBrotherAnalytics/build
[INFO] Timeout:       30s

════════════════════════════════════════════════════════════
>>> Setting Up Build Environment
════════════════════════════════════════════════════════════
[✓] Build environment ready

════════════════════════════════════════════════════════════
>>> Building Project
════════════════════════════════════════════════════════════
[INFO] Running CMake configuration...
[✓] CMake configuration completed
[INFO] Compiling (using 8 parallel jobs)...
[✓] Build completed successfully

════════════════════════════════════════════════════════════
>>> Setting Up Test Configuration
════════════════════════════════════════════════════════════
[FOUND] Config section: database
[FOUND] Config section: logging
[FOUND] Config section: trading
[FOUND] Config section: schwab
[FOUND] Config section: risk
[✓] Configuration verified

════════════════════════════════════════════════════════════
>>> Running Trading Engine Integration Tests
════════════════════════════════════════════════════════════
[INFO] Starting trading engine (30s timeout)...
[✓] Trading engine test completed

════════════════════════════════════════════════════════════
>>> Running Test Validations
════════════════════════════════════════════════════════════
[TEST] Engine Initialization
[FOUND] Engine initialization message
[TEST] Configuration Loading
[FOUND] Configuration loading
[TEST] Paper Trading Mode
[FOUND] Paper trading mode confirmation
[TEST] Database Initialization
[FOUND] Database initialization
[TEST] Strategy Registration
[FOUND] Strategy registration message
[TEST] Trading Cycle Execution
[FOUND] Trading cycle initiated
[TEST] Signal Generation
[FOUND] Signal generation
[TEST] Position Tracking
[✓] Position tracking - no active positions yet
[TEST] Stop Loss Checking
[✓] Stop loss checks operating
[TEST] Account Information
[FOUND] Account information retrieval
[TEST] Risk Management
[FOUND] Risk management processing
[TEST] Error-Free Execution
[FOUND] No errors in execution
[TEST] Graceful Shutdown
[FOUND] Graceful shutdown

════════════════════════════════════════════════════════════
Test Results Summary
════════════════════════════════════════════════════════════

Test Statistics:
  Total Tests Run:    13
  Tests Passed:       13
  Tests Failed:       0
  Success Rate:       100%

ALL TESTS PASSED!

Log Files:
  Test Log: logs/test_run_1731000000.log
```

## Advanced Usage

### Parallel Testing

Run multiple test configurations in sequence:

```bash
#!/bin/bash
echo "Test 1: Default configuration"
./scripts/test_trading_engine.sh

echo "Test 2: Extended timeout"
./scripts/test_trading_engine.sh --timeout 60

echo "Test 3: Clean rebuild"
./scripts/test_trading_engine.sh --clean --timeout 60
```

### Continuous Integration

```bash
#!/bin/bash
set -e

# Run test suite
./scripts/test_trading_engine.sh --clean --timeout 90

# Save results
TEST_LOG=$(ls -t logs/test_run_*.log | head -1)
cp "$TEST_LOG" test_results_$(date +%Y%m%d_%H%M%S).log

# Report status
if grep -q "ALL TESTS PASSED" <(tail -5 "$TEST_LOG"); then
    echo "✓ All tests passed"
    exit 0
else
    echo "✗ Tests failed"
    exit 1
fi
```

## Performance Monitoring

Monitor resource usage during tests:

```bash
# Watch CPU and memory
watch -n 1 'ps aux | grep bigbrother | grep -v grep'

# Monitor disk I/O
iotop -a

# Check system load
uptime
```

## Debugging

Enable verbose logging for troubleshooting:

```bash
# Run with maximum verbosity
./scripts/test_trading_engine.sh --verbose --timeout 60 2>&1 | tee debug_log.txt

# Extract specific sections
grep "\[ERROR\]" debug_log.txt
grep "\[WARN\]" debug_log.txt
grep "Trading Cycle" debug_log.txt
```

## Support

For issues or questions:

1. Check the test output for specific failure messages
2. Review log files in `logs/` directory
3. Verify configuration in `configs/config.yaml`
4. Check build output: `cat build_output.log`
5. Review CMake configuration: `cat build/CMakeCache.txt`

## Related Scripts

- `./scripts/build.sh` - C++ build script
- `./scripts/build_and_test_schwab.sh` - Schwab API tests
- `./scripts/run_schwab_tests.sh` - Integration test runner

## Version

- **Script Version**: 1.0
- **Created**: 2025-11-09
- **Maintained by**: BigBrotherAnalytics Team
- **Last Updated**: 2025-11-09
