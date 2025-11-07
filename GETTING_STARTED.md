# Getting Started with BigBrotherAnalytics

This guide will help you build, configure, and run the trading system.

## Prerequisites

All dependencies from Tier 1 setup are already installed:
- ✅ GCC 15.2.0 with C++23
- ✅ Python 3.13
- ✅ OpenMP
- ✅ uv package manager
- ✅ 270+ Python packages

## Step 1: Install C++ Dependencies

Install additional C++ libraries needed for the trading system:

```bash
sudo ./scripts/install_cpp_deps.sh
```

This installs:
- DuckDB C++ library
- ONNX Runtime
- libcurl, nlohmann/json, yaml-cpp
- spdlog, websocketpp, Boost
- Google Test

**Estimated time:** 5-10 minutes

## Step 2: Configure API Keys

```bash
# Copy template
cp configs/api_keys.yaml.template configs/api_keys.yaml

# Edit with your actual keys
nano configs/api_keys.yaml
```

Required keys:
- **Schwab API** (get from https://developer.schwab.com/)
  - Client ID
  - Client Secret
  - Account ID

Optional keys:
- **FRED API** (get from https://fred.stlouisfed.org/docs/api/api_key.html)
- Alpha Vantage, News API (for enhanced data)

**Alternatively, use environment variables:**

```bash
export SCHWAB_CLIENT_ID="your_client_id_here"
export SCHWAB_CLIENT_SECRET="your_client_secret_here"
export SCHWAB_ACCOUNT_ID="your_account_number_here"
export FRED_API_KEY="your_fred_api_key_here"
```

## Step 3: Build the Project

```bash
# Build all C++ components
./scripts/build.sh

# Or build manually
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

**Build output:**
- `build/bin/bigbrother` - Main trading application
- `build/bin/backtest` - Backtesting engine
- `build/lib/*.so` - Shared libraries

**Estimated time:** 2-5 minutes

## Step 4: Run Tests

```bash
cd build
make test

# Or use ctest for detailed output
ctest --output-on-failure --verbose
```

**Expected results:**
- ✅ Options pricing tests (< 1μs latency)
- ✅ Correlation tests (< 10μs latency)
- ✅ All mathematical validations pass

## Step 5: Download Historical Data

```bash
# Set FRED API key (if not already set)
export FRED_API_KEY="your_key_here"

# Download 10 years of data
uv run python scripts/data_collection/download_historical.py
```

**Downloads:**
- 30+ stock symbols (SPY, QQQ, AAPL, etc.)
- Options chains for major symbols
- Economic indicators from FRED
- Stores in `data/raw/*.parquet`

**Estimated time:** 5-10 minutes
**Data size:** ~500 MB

## Step 6: Run Backtest

```bash
# Backtest straddle strategy
./build/bin/backtest \
    --strategy straddle \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --data data/raw/

# Backtest all strategies
./build/bin/backtest \
    --all-strategies \
    --start 2020-01-01 \
    --end 2024-01-01
```

**Output:**
- Performance metrics (Sharpe, win rate, drawdown)
- Success criteria validation
- Trade log exported to CSV
- Results saved in `data/backtest_results/`

**Success Criteria (per PRD):**
- Win rate > 60% ✓
- Sharpe ratio > 2.0 ✓
- Max drawdown < 15% ✓

## Step 7: Paper Trading (After Backtest Success)

```bash
# Verify paper trading mode is ON
grep "paper_trading" configs/config.yaml
# Should show: paper_trading: true

# Run trading engine
./build/bin/bigbrother --config configs/config.yaml
```

**Paper Trading:**
- Simulates orders without real money
- Full system validation
- Real-time data from Schwab
- Strategy execution testing
- Risk management validation

**Monitor:**
- Check `logs/bigbrother.log` for activity
- Watch for signal generation
- Verify risk limits enforced
- Track simulated P&L

## Step 8: Schwab API Authentication (First Time Only)

The first time you run the trading engine, you'll need to authenticate:

```bash
./build/bin/bigbrother --config configs/config.yaml
```

**OAuth Flow:**
1. Application prints authorization URL
2. Open URL in browser
3. Login to Schwab and approve
4. Copy authorization code from redirect URL
5. Paste code into terminal
6. Tokens saved to `configs/schwab_tokens.json`

**Token Refresh:**
- Automatic every 25 minutes
- Background thread handles refresh
- No manual intervention needed

## Common Commands

```bash
# Build in debug mode
./scripts/build.sh debug

# Clean and rebuild
./scripts/build.sh clean

# Run with verbose logging
./build/bin/bigbrother --config configs/config.yaml

# Run specific backtest
./build/bin/backtest --strategy vol_arb --start 2023-01-01

# Download fresh data
uv run python scripts/data_collection/download_historical.py

# Check build status
./scripts/build.sh test
```

## Directory Structure

```
BigBrotherAnalytics/
├── build/                 # Build output (executables, libraries)
│   ├── bin/
│   │   ├── bigbrother     # Main trading app
│   │   └── backtest       # Backtesting app
│   └── lib/               # Shared libraries
├── configs/               # Configuration files
│   ├── config.yaml        # Main config
│   └── api_keys.yaml      # API credentials (not in git)
├── data/
│   ├── raw/               # Downloaded data (Parquet files)
│   ├── processed/         # Processed features
│   └── backtest_results/  # Backtest outputs
├── logs/                  # Application logs
└── src/                   # C++ source code

```

## Next Steps

After successful paper trading (2 weeks minimum):

1. **Analyze Results**
   - Review trade logs
   - Calculate actual metrics
   - Compare to backtest results

2. **Tune Parameters**
   - Adjust strategy parameters
   - Optimize position sizing
   - Refine entry/exit criteria

3. **Decision Point**
   - If profitable: Switch to live trading
   - If not: Analyze failures, pivot strategy

4. **Live Trading (Manual Activation)**
   ```bash
   # Edit config
   nano configs/config.yaml
   # Change: paper_trading: false

   # Run with REAL MONEY
   ./build/bin/bigbrother --config configs/config.yaml
   ```

## Safety Checklist

Before live trading:
- [ ] Backtests show profitability (Win rate > 60%, Sharpe > 2.0)
- [ ] Paper trading successful for 2+ weeks
- [ ] Daily loss limit tested and working ($900 max)
- [ ] Stop losses trigger correctly
- [ ] Position limits enforced ($1,500 max)
- [ ] Schwab API authentication working
- [ ] All unit tests passing
- [ ] Monitoring and alerts configured

## Troubleshooting

### Build Fails

```bash
# Check GCC version
g++ --version  # Should be 15.2.0+

# Check dependencies
ldconfig -p | grep duckdb
ldconfig -p | grep onnxruntime

# Reinstall dependencies
sudo ./scripts/install_cpp_deps.sh
```

### Tests Fail

```bash
# Run tests with verbose output
cd build
ctest --output-on-failure --verbose

# Check specific test
./build/lib/test_options_pricing
```

### Data Download Fails

```bash
# Check internet connection
ping finance.yahoo.com

# Check Python packages
uv pip list | grep yfinance

# Reinstall if needed
uv pip install yfinance fredapi pandas pyarrow
```

### Schwab API Errors

```bash
# Check credentials
echo $SCHWAB_CLIENT_ID

# Verify redirect URI matches Schwab app settings
grep redirect_uri configs/config.yaml

# Check token file
ls -la configs/schwab_tokens.json

# Re-authenticate if needed
rm configs/schwab_tokens.json
./build/bin/bigbrother
```

## Performance Tuning

```bash
# Set OpenMP threads
export OMP_NUM_THREADS=32

# Enable MPI for correlation engine (multi-node)
mpirun -np 4 ./build/bin/bigbrother

# Profile performance
perf record ./build/bin/backtest --strategy straddle
perf report
```

## Support

- Documentation: `docs/`
- Architecture: `docs/architecture/`
- PRD: `docs/PRD.md`
- Issues: https://github.com/oldboldpilot/BigBrotherAnalytics/issues

## Success Metrics

**Target (per PRD):**
- Daily profit: > $150 (on $30k account)
- Win rate: > 60%
- Sharpe ratio: > 2.0
- Max drawdown: < 15%

**If achieved consistently over 3 months:**
- Proceed to Tier 2 (scale up capital)
- Add paid data feeds
- Deploy to production

**If not achieved:**
- Analyze failure modes
- Pivot strategies
- Stop with minimal losses (zero sunk infrastructure cost)

This is the beauty of the DuckDB-first, free-data approach:
**Zero risk if it doesn't work!**
