# Build Instructions - C++ Heavy Architecture

BigBrotherAnalytics is implemented primarily in C++23 for maximum performance and low-latency trading operations.

## Architecture Overview

**C++ Components (95% of codebase):**
- Core trading engine
- Options pricing (Black-Scholes, binomial/trinomial trees, Greeks)
- Correlation engine (Pearson, Spearman, time-lagged, MPI parallelized)
- Market data clients (Yahoo Finance, FRED, Schwab API)
- NLP sentiment analysis (ONNX Runtime for FinBERT inference)
- Risk management system
- Backtesting framework
- Order execution and monitoring

**Python Components (5% of codebase):**
- ML model training (export to ONNX for C++ inference)
- Monitoring dashboard (Plotly Dash)
- Optional data collection scripts

## Prerequisites

### Already Installed (from Tier 1 setup)
- GCC 15.2.0 with C++23 support ✓
- CMake 3.20+ ✓
- Python 3.13 ✓
- OpenMP ✓

### Need to Install

Run the automated installation script:

```bash
sudo ./scripts/install_cpp_deps.sh
```

This installs:
- **libcurl** - HTTP requests
- **nlohmann/json** - JSON parsing
- **yaml-cpp** - Configuration files
- **spdlog** - Fast logging
- **Google Test** - C++ unit testing
- **websocketpp** + **Boost** - WebSocket support
- **DuckDB C++ library** - Embedded database
- **ONNX Runtime C++ library** - ML inference

### Manual Installation (if script fails)

```bash
# Core libraries
sudo apt-get install -y \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    libyaml-cpp-dev \
    libspdlog-dev \
    libgtest-dev \
    libwebsocketpp-dev \
    libboost-all-dev

# DuckDB (download from GitHub releases)
wget https://github.com/duckdb/duckdb/releases/download/v1.1.3/libduckdb-linux-amd64.zip
unzip libduckdb-linux-amd64.zip
sudo cp duckdb.h duckdb.hpp /usr/local/include/
sudo cp libduckdb.so /usr/local/lib/
sudo ldconfig

# ONNX Runtime (download from GitHub releases)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
sudo cp -r onnxruntime-linux-x64-1.18.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.18.0/lib/* /usr/local/lib/
sudo ldconfig
```

## Build Process

### 1. Install Dependencies

```bash
# Automated installation
sudo ./scripts/install_cpp_deps.sh

# Verify installations
ldconfig -p | grep duckdb
ldconfig -p | grep onnxruntime
```

### 2. Configure Build

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

CMake will report which dependencies were found. Example output:

```
╔════════════════════════════════════════════════════════════╗
║     BigBrotherAnalytics C++ Configuration Summary          ║
╚════════════════════════════════════════════════════════════╝

Build Configuration:
  C++ Standard        : 23
  Build Type          : Release
  Compiler            : /usr/local/bin/g++

Core Dependencies:
  OpenMP              : 4.5 ✓
  Threads             : Found
  CURL                : 7.81.0

Math Libraries:
  BLAS/LAPACK         : System libraries

Optional Dependencies:
  DuckDB              : Found ✓
  ONNX Runtime        : Found ✓
  spdlog              : Found ✓
  nlohmann/json       : Found ✓
  yaml-cpp            : Found ✓

Executables:
  bigbrother          : Main trading application
  backtest            : Backtesting engine
```

### 3. Build

```bash
# Build all targets
make -j$(nproc)

# Or build specific targets
make bigbrother          # Main trading application
make backtest            # Backtesting engine
make utils               # Utilities library
make options_pricing     # Options pricing library
make correlation_engine  # Correlation engine
make schwab_api          # Schwab API client
```

### 4. Run Tests

```bash
# Run all C++ tests
make test

# Or use ctest for detailed output
ctest --output-on-failure --verbose
```

### 5. Install (Optional)

```bash
sudo make install
```

This installs:
- Executables to `/usr/local/bin/`
- Libraries to `/usr/local/lib/`

## Quick Build Script

For convenience, use the build script:

```bash
# Build in Release mode
./scripts/build.sh

# Build in Debug mode
./scripts/build.sh debug

# Clean and rebuild
./scripts/build.sh clean

# Build and run tests
./scripts/build.sh test
```

## Build Output

After successful build:

```
build/
├── bin/
│   ├── bigbrother       # Main trading application
│   └── backtest         # Backtesting engine
└── lib/
    ├── libutils.so
    ├── libmarket_intelligence.so
    ├── libcorrelation_engine.so
    ├── liboptions_pricing.so
    ├── libtrading_decision.so
    ├── librisk_management.so
    ├── libschwab_api.so
    └── libexplainability.so
```

## Running the Application

### Main Trading Application

```bash
# Run from build directory
./build/bin/bigbrother --config ../configs/config.yaml

# Run in paper trading mode
./build/bin/bigbrother --config ../configs/config.yaml --paper-trading

# Run with verbose logging
./build/bin/bigbrother --config ../configs/config.yaml --log-level debug
```

### Backtesting

```bash
# Run backtest on specific strategy
./build/bin/backtest \
    --strategy straddle \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --capital 30000

# Run multi-strategy backtest
./build/bin/backtest \
    --strategies straddle,strangle,vol_arb \
    --start-date 2020-01-01 \
    --end-date 2024-01-01
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Write C++ Code

Edit files in `src/*/` directories following C++23 best practices.

### 3. Write Tests

Add tests in `tests/cpp/` for new functionality.

### 4. Build and Test

```bash
cd build
make -j$(nproc)
make test
```

### 5. Fix Any Issues

```bash
# Check for memory leaks with valgrind
valgrind --leak-check=full ./bin/bigbrother --help

# Profile with perf
perf record ./bin/bigbrother --paper-trading
perf report
```

### 6. Commit and Push

```bash
git add .
git commit -m "Add new feature"
git push origin feature/my-new-feature
```

## Python ML Model Training

While the core is C++, ML models are trained in Python and exported to ONNX:

```bash
# Train FinBERT sentiment model (fine-tuned)
uv run python scripts/ml/train_sentiment_model.py

# Export to ONNX
uv run python scripts/ml/export_to_onnx.py \
    --model models/finbert_sentiment.pt \
    --output models/finbert_sentiment.onnx

# Test ONNX inference in C++
./build/bin/bigbrother --test-sentiment "Apple reports record earnings"
```

## Troubleshooting

### CMake can't find DuckDB

```bash
# Ensure library is in system path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig

# Or specify path manually
cmake -DDUCKDB_INCLUDE=/usr/local/include -DDUCKDB_LIB=/usr/local/lib/libduckdb.so ..
```

### ONNX Runtime not found

```bash
# Verify installation
ls -la /usr/local/lib/libonnxruntime*
ls -la /usr/local/include/onnxruntime*

# If missing, reinstall
sudo ./scripts/install_cpp_deps.sh
```

### Linking errors

```bash
# Update library cache
sudo ldconfig

# Check library dependencies
ldd build/bin/bigbrother
```

### Compilation errors with C++23

```bash
# Verify GCC version
g++ --version  # Should be 15.2.0 or higher

# Check C++23 support
g++ -std=c++23 -E -v -
```

## Performance Optimization

### Compiler Flags

The Release build uses aggressive optimizations:
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimizations
- `-ffast-math` - Fast floating-point math
- `-DNDEBUG` - Disable assertions

### Profiling

```bash
# CPU profiling with perf
perf record -g ./build/bin/bigbrother --backtest
perf report

# Memory profiling with valgrind
valgrind --tool=massif ./build/bin/bigbrother

# Cache analysis
perf stat -e cache-references,cache-misses ./build/bin/bigbrother
```

### OpenMP Tuning

```bash
# Set number of threads
export OMP_NUM_THREADS=32

# Tune scheduling
export OMP_SCHEDULE="dynamic,1"
```

### MPI Scaling (for correlation engine)

```bash
# Run on single node with 32 cores
mpirun -np 32 ./build/bin/bigbrother --correlation-only

# Run on cluster (when available)
mpirun -np 128 --hostfile hosts.txt ./build/bin/bigbrother
```

## CI/CD Integration

```yaml
# .github/workflows/build.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: sudo ./scripts/install_cpp_deps.sh

      - name: Build
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Release ..
          make -j$(nproc)

      - name: Test
        run: cd build && ctest --output-on-failure
```

## Next Steps

1. Install all dependencies: `sudo ./scripts/install_cpp_deps.sh`
2. Build the project: `./scripts/build.sh`
3. Run tests: `cd build && make test`
4. Start implementing core modules (see IMPLEMENTATION_PLAN.md)
5. Run backtests to validate strategies
6. Deploy to paper trading
7. Monitor performance and iterate

## Resources

- **C++23 Documentation**: https://en.cppreference.com/w/cpp/23
- **ONNX Runtime C++ API**: https://onnxruntime.ai/docs/api/c/
- **DuckDB C++ API**: https://duckdb.org/docs/api/cpp
- **OpenMP**: https://www.openmp.org/specifications/
- **MPI**: https://www.open-mpi.org/doc/
