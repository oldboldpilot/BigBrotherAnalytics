# Trading Engine Test - Quick Start Guide

## TL;DR - Run the Test in 30 Seconds

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
./scripts/test_trading_engine.sh
```

That's it! The script will:
1. Build the project
2. Configure for paper trading
3. Run the trading engine for 30 seconds
4. Validate 13 key components
5. Report pass/fail status

## What Gets Tested

| Component | Status | Details |
|-----------|--------|---------|
| Engine Initialization | ✓ | Core bootstrap |
| Configuration Loading | ✓ | Config system |
| Paper Trading Mode | ✓ | Safety check |
| Database | ✓ | DuckDB init |
| Strategies | ✓ | All 3 strategies register |
| Trading Cycle | ✓ | Main loop executes |
| Signal Generation | ✓ | Signals created |
| Position Tracking | ✓ | Positions monitored |
| Stop Losses | ✓ | Risk controls active |
| Account Info | ✓ | API integration |
| Risk Management | ✓ | Portfolio risk monitored |
| Error Checking | ✓ | No errors |
| Graceful Shutdown | ✓ | Clean exit |

## Common Commands

### Full test with clean rebuild
```bash
./scripts/test_trading_engine.sh --clean
```

### Faster test (skip rebuild)
```bash
./scripts/test_trading_engine.sh --no-build
```

### Extended timeout for slow systems
```bash
./scripts/test_trading_engine.sh --timeout 90
```

### All options combined
```bash
./scripts/test_trading_engine.sh --clean --timeout 60 --verbose
```

## Expected Output

When successful, you should see:

```
[✓] All tests passed
Success Rate: 100%

Log Files:
  Test Log: logs/test_run_1731000000.log
```

## If Something Fails

1. **Build failed**: Check CMake output
   ```bash
   cat build_output.log | tail -50
   ```

2. **Missing config**: Verify configuration
   ```bash
   test -f configs/config.yaml && echo "OK" || echo "Missing"
   ```

3. **Tests failed**: Check logs
   ```bash
   tail -50 logs/bigbrother.log
   ```

4. **Timeout**: Increase it
   ```bash
   ./scripts/test_trading_engine.sh --timeout 60
   ```

## View Test Logs

```bash
# Last 50 lines of main log
tail -50 logs/bigbrother.log

# Last 20 lines of last test run
tail -20 logs/test_run_*.log | head -20

# Search for errors
grep "ERROR" logs/bigbrother.log

# Watch logs in real-time
tail -f logs/bigbrother.log
```

## What the Script Does

### Phase 1: Build (2-5 minutes)
- Cleans build directory (if `--clean`)
- Runs CMake configuration
- Compiles with 8+ parallel jobs
- Produces `build/bin/bigbrother` executable

### Phase 2: Setup (10 seconds)
- Verifies configuration file
- Checks paper trading is enabled
- Validates database paths
- Confirms all required config sections

### Phase 3: Execution (30 seconds or custom timeout)
- Starts trading engine
- Runs trading cycles
- Monitors strategy signals
- Checks position tracking
- Validates risk management

### Phase 4: Validation (5 seconds)
- Parses logs for expected patterns
- Counts signals, positions, errors
- Validates each component
- Reports pass/fail for each test

### Phase 5: Reporting (2 seconds)
- Displays test results
- Shows success rate
- Lists log file locations
- Returns appropriate exit code

## File Locations

```
Project Structure:
├── build/
│   └── bin/
│       └── bigbrother              (executable)
├── configs/
│   └── config.yaml                 (test configuration)
├── logs/
│   ├── bigbrother.log             (application log)
│   └── test_run_*.log             (test run logs)
├── data/
│   └── bigbrother.duckdb          (database)
└── scripts/
    ├── test_trading_engine.sh     (this test)
    ├── TRADING_ENGINE_TEST_README.md (full docs)
    └── TEST_QUICK_START.md        (this file)
```

## Exit Codes

```
0 = Success (all tests passed)
1 = Build failed
2 = Setup failed
3 = Tests failed
4 = Timeout (increase with --timeout)
```

Example:
```bash
./scripts/test_trading_engine.sh
echo $?  # Prints 0 for success, non-zero for failure
```

## Environment Variables

Optional configuration:

```bash
# Set custom timeout
TEST_TIMEOUT=90 ./scripts/test_trading_engine.sh

# Use specific Schwab credentials
export SCHWAB_CLIENT_ID="your_id"
export SCHWAB_CLIENT_SECRET="your_secret"
./scripts/test_trading_engine.sh
```

## Automated Testing

Run as part of your workflow:

```bash
#!/bin/bash
set -e

echo "Starting test suite..."
./scripts/test_trading_engine.sh --clean --timeout 60

echo "Test completed successfully!"
echo "Results in: logs/test_run_*.log"
```

Or in a cronjob:

```cron
0 2 * * * cd /path/to/BigBrotherAnalytics && \
  ./scripts/test_trading_engine.sh --timeout 60 >> test_cron.log 2>&1
```

## Performance Tips

1. **First run**: Takes longer (module compilation)
   - Use `--no-build` for subsequent runs

2. **Parallel builds**: Automatically uses all CPU cores
   - Check with `nproc` command

3. **Memory usage**: Minimal for dry-run tests
   - ~100-200 MB during execution

4. **Disk space**: Ensure 500 MB free for build artifacts

## Next Steps

1. ✓ Run `./scripts/test_trading_engine.sh`
2. ✓ Check test passes (should see 13/13 ✓)
3. ✓ Review logs: `tail logs/bigbrother.log`
4. ✓ Read full docs: `TRADING_ENGINE_TEST_README.md`

## Support

**Script fails?**
- Run with `--timeout 60` for more time
- Try `--clean` for fresh build
- Check `logs/bigbrother.log` for details

**Need more info?**
- Read: `TRADING_ENGINE_TEST_README.md`
- Check: `scripts/test_trading_engine.sh` source
- View logs: `ls -lh logs/`

---

**Last Updated**: 2025-11-09
**Script Version**: 1.0
**Status**: Production Ready
