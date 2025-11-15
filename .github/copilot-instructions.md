# BigBrotherAnalytics - Claude AI Guide

**Project:** High-performance AI-powered trading intelligence platform
**Phase:** Phase 5+ - Production 85-Feature INT32 SIMD ML Engine (Ready for Live Trading)
**Status:** 100% Production Ready - INT32 SIMD + MKL Fallback + Real-Time Risk Management
**Budget:** $2,000 position limit (paper trading validation)
**Goal:** ≥70% win rate (profitable after 37.1% tax + $0.65/contract fees)
**Last Tested:** November 14, 2025 - C++ Single Source of Truth standard established
**ML Model:** Production - 85 features (clean), 22,700 samples - **95.1% (1d)**, **97.1% (5d)**, **98.18% (20d)** ✅ TARGET MET
**Integration:** INT32 SIMD (AVX-512/AVX2/MKL/Scalar fallback), ~10μs inference, 98K predictions/sec

---

## C++ Single Source of Truth (MANDATORY)

**CRITICAL PRINCIPLE:** All data extraction, feature extraction, and quantization operations MUST be implemented in C++ with Python bindings for training. This ensures ZERO variation between training and inference - perfect parity guaranteed.

### The Standard

**ALL data loading, feature extraction, and quantization logic MUST:**
1. Be implemented in C++23 modules (`.cppm` files)
2. Expose Python bindings via pybind11 for training purposes ONLY
3. Have NO Python-only implementations for these operations
4. Follow the principle: modifications happen in ONE place (C++) and propagate everywhere
5. Eliminate feature drift by ensuring training and inference use identical code

### Why This Matters

**Problem Solved:**
- **Feature Drift:** Python training code diverges from C++ inference → model accuracy degrades
- **Debugging Nightmare:** Two implementations = two places to fix bugs
- **Performance Mismatch:** Python approximations ≠ C++ calculations → unpredictable results
- **Maintenance Hell:** Change in one place requires manual sync to another

**Benefits Achieved:**
- **Perfect Parity:** Training and inference use IDENTICAL code (byte-for-byte)
- **Zero Drift:** Impossible for features to diverge - single implementation
- **10-20x Faster Training:** C++ speed for feature generation
- **Type Safety:** C++23 strong typing catches errors at compile time
- **Single Point of Maintenance:** Fix once, propagate everywhere

### Implementation Examples from Our Codebase

**1. Data Loading - `src/ml/data_loader.cppm`**
```cpp
// C++23 module for data loading
export module bigbrother.ml.data_loader;

export namespace bigbrother::ml {
    class DataLoader {
    public:
        // Load historical market data
        auto loadHistoricalData(
            std::string const& symbol,
            std::chrono::system_clock::time_point start,
            std::chrono::system_clock::time_point end
        ) -> std::expected<std::vector<MarketBar>, Error>;

        // Load with validation
        auto loadAndValidate(/* params */) -> std::expected<Dataset, Error>;
    };
}
```

**Python Binding - `src/python_bindings/data_loader_bindings.cpp`**
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
import bigbrother.ml.data_loader;

PYBIND11_MODULE(data_loader_cpp, m) {
    py::class_<bigbrother::ml::DataLoader>(m, "DataLoader")
        .def(py::init<>())
        .def("load_historical_data", &DataLoader::loadHistoricalData);
}
```

**2. Feature Extraction - `src/market_intelligence/feature_extractor.cppm`**
```cpp
// C++23 module with 85-feature extraction
export module bigbrother.market_intelligence.feature_extractor;

export namespace bigbrother::market_intelligence {
    class FeatureExtractor {
    public:
        // Extract 85 features for ML model
        [[nodiscard]] auto toArray85(
            std::span<double const> price_history,
            std::span<double const> volume_history,
            std::chrono::system_clock::time_point timestamp
        ) const -> std::array<float, 85>;

        // Calculate Greeks using Black-Scholes
        [[nodiscard]] auto calculateGreeks(/* params */) const -> Greeks;
    };
}
```

**Python Binding - `src/python_bindings/feature_extractor_bindings.cpp`**
```cpp
PYBIND11_MODULE(feature_extractor_cpp, m) {
    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def("extract_features_85", &FeatureExtractor::toArray85)
        .def("calculate_greeks", &FeatureExtractor::calculateGreeks);
}
```

**3. INT32 Quantization - Integrated into `feature_extractor.cppm`**
```cpp
export namespace bigbrother::market_intelligence {
    class FeatureExtractor {
    public:
        // Quantize 85 features to INT32 (symmetric quantization)
        [[nodiscard]] auto quantizeFeatures85(
            std::array<float, 85> const& features
        ) const -> std::array<int32_t, 85>;

        // Dequantize INT32 back to float
        [[nodiscard]] auto dequantizeFeatures85(
            std::array<int32_t, 85> const& quantized
        ) const -> std::array<float, 85>;
    };
}
```

### Usage in Training Pipeline

**Training Script - `scripts/ml/prepare_features_cpp.py`**
```python
#!/usr/bin/env python3
"""
Training feature generation using C++ Single Source of Truth

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""
import sys
sys.path.insert(0, 'python')  # Add Python bindings directory

# Import C++ modules via pybind11
from data_loader_cpp import DataLoader
from feature_extractor_cpp import FeatureExtractor

def generate_training_data():
    # Use C++ data loader
    loader = DataLoader()
    data = loader.load_historical_data("SPY", start_date, end_date)

    # Use C++ feature extractor
    extractor = FeatureExtractor()
    features = extractor.extract_features_85(data['close'], data['volume'], timestamp)

    # Use C++ quantization
    quantized = extractor.quantize_features_85(features)

    # Save to training database
    save_to_parquet(features, "models/training_data/features_cpp_85.parquet")
```

### Usage in C++ Inference (Trading Bot)

**Trading Engine - `src/main.cpp`**
```cpp
import bigbrother.market_intelligence.feature_extractor;

auto main() -> int {
    using namespace bigbrother::market_intelligence;

    FeatureExtractor extractor;

    // Extract features from live market data
    auto features = extractor.toArray85(price_history, volume_history, timestamp);

    // Quantize for INT32 SIMD neural network
    auto quantized = extractor.quantizeFeatures85(features);

    // Run inference (uses same features as training!)
    auto prediction = predictor->predict(symbol, quantized);

    // Perfect parity guaranteed - same code path as training
}
```

### When to Use C++ vs Python

**C++ Implementation Required:**
- ✅ Data extraction from databases/APIs
- ✅ Feature calculation (technical indicators, Greeks, lags, autocorrelations)
- ✅ Quantization/dequantization
- ✅ Data preprocessing (normalization, scaling)
- ✅ Any operation used in BOTH training AND inference

**Python Implementation Allowed:**
- ✅ Model training (PyTorch, scikit-learn)
- ✅ Hyperparameter tuning
- ✅ Visualization and plotting
- ✅ Exploratory data analysis (EDA)
- ✅ Operations used ONLY in training (never in inference)

### How to Create Python Bindings

**Step 1: Implement in C++23 Module**
```cpp
// src/my_component/my_feature.cppm
export module bigbrother.my_component.my_feature;

export namespace bigbrother::my_component {
    class MyFeature {
    public:
        auto extractFeature(double input) -> double {
            return input * 2.0;  // Your logic here
        }
    };
}
```

**Step 2: Create Python Binding**
```cpp
// src/python_bindings/my_feature_bindings.cpp
#include <pybind11/pybind11.h>
import bigbrother.my_component.my_feature;

namespace py = pybind11;
using namespace bigbrother::my_component;

PYBIND11_MODULE(my_feature_cpp, m) {
    m.doc() = "C++ implementation of MyFeature";

    py::class_<MyFeature>(m, "MyFeature")
        .def(py::init<>())
        .def("extract_feature", &MyFeature::extractFeature,
             py::arg("input"),
             "Extract feature from input");
}
```

**Step 3: Add to CMakeLists.txt**
```cmake
# Build Python binding
pybind11_add_module(my_feature_py
    src/python_bindings/my_feature_bindings.cpp
)
target_link_libraries(my_feature_py PRIVATE my_feature_module)
set_target_properties(my_feature_py PROPERTIES
    OUTPUT_NAME "my_feature_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)
```

**Step 4: Build and Test**
```bash
# Build binding
ninja -C build my_feature_py

# Test in Python
PYTHONPATH=python:$PYTHONPATH uv run python -c "
from my_feature_cpp import MyFeature
f = MyFeature()
result = f.extract_feature(21.0)
assert result == 42.0
print('✅ Parity verified!')
"
```

### How to Test Parity

**Create Parity Test Script - `tests/test_feature_parity.py`**
```python
#!/usr/bin/env python3
"""
Test parity between C++ and training usage

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-14
"""
import sys
sys.path.insert(0, 'python')
from feature_extractor_cpp import FeatureExtractor
import numpy as np

def test_feature_extraction_parity():
    """Verify C++ features match expected values"""
    extractor = FeatureExtractor()

    # Test data
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    volumes = [1000, 1100, 1200, 1300, 1400]

    # Extract features
    features = extractor.extract_features_85(prices, volumes, timestamp)

    # Verify critical features
    assert len(features) == 85, "Must have exactly 85 features"
    assert features[0] == 104.0, "Feature[0] (close) must match"
    assert features[47] == 0, "Feature[47] (SPY encoding) must be 0"

    print("✅ Feature parity verified!")

def test_quantization_parity():
    """Verify quantization round-trip accuracy"""
    extractor = FeatureExtractor()

    features = np.random.randn(85).astype(np.float32)
    quantized = extractor.quantize_features_85(features)
    dequantized = extractor.dequantize_features_85(quantized)

    # Verify round-trip error < 1e-6
    max_error = np.max(np.abs(features - dequantized))
    assert max_error < 1e-6, f"Quantization error {max_error} too large"

    print("✅ Quantization parity verified!")

if __name__ == "__main__":
    test_feature_extraction_parity()
    test_quantization_parity()
```

**Run Parity Tests:**
```bash
PYTHONPATH=python:$PYTHONPATH uv run python tests/test_feature_parity.py
```

### Deprecated Python-Only Implementations

**DO NOT USE:**
- ❌ `scripts/ml/prepare_custom_features.py.deprecated` - Old Python feature extraction
- ❌ Any Python scripts that duplicate C++ functionality
- ❌ Manual feature calculations in Python notebooks

**Why Deprecated:**
- Different floating-point arithmetic than C++
- Hardcoded values (e.g., gamma=0.01) instead of calculated Greeks
- Price ratios instead of actual price lags
- Inconsistent autocorrelation window sizes
- Will cause model accuracy degradation over time

### Migration Checklist

When adding new features or modifying existing ones:

- [ ] Implement feature in C++23 module (`.cppm`)
- [ ] Add Python binding (`.cpp` in `src/python_bindings/`)
- [ ] Update CMakeLists.txt to build binding
- [ ] Build binding: `ninja -C build <binding_name>_py`
- [ ] Create parity test in `tests/`
- [ ] Verify parity: Run test script
- [ ] Update training script to use C++ binding
- [ ] Update C++ inference code to use same module
- [ ] Document in this guide
- [ ] Remove any deprecated Python-only implementations

### Reference Implementation

**Complete Example:** See existing implementation:
- **Data Loading:** `src/ml/data_loader.cppm` + `src/python_bindings/data_loader_bindings.cpp`
- **Feature Extraction:** `src/market_intelligence/feature_extractor.cppm` + `src/python_bindings/feature_extractor_bindings.cpp`
- **Training Pipeline:** `scripts/ml/prepare_features_cpp.py`
- **Parity Tests:** `tests/test_feature_parity.py`

### Enforcement

This standard is:
- **Documented:** In `.ai/claude.md`, `copilot-instructions.md`, `docs/CODING_STANDARDS.md`
- **Reviewed:** All PRs must demonstrate C++ implementation + Python bindings
- **Tested:** Parity tests required for all new features
- **Monitored:** Model accuracy tracked to detect feature drift

**Violations:** Any Python-only implementation of data/feature/quantization logic will be rejected in code review.

---

## Core Architecture

**Three Interconnected Systems:**
1. **Market Intelligence Engine** - Multi-source data ingestion, NLP, impact prediction, graph generation
   - **FRED Rate Provider:** Live risk-free rates from Federal Reserve (6 series, AVX2 SIMD, auto-refresh)
   - **ML Price Predictor (PRODUCTION):** INT32 SIMD neural network with 85 clean features (98.18% 20d accuracy)
     - **Module:** bigbrother.market_intelligence.price_predictor
     - **Architecture:** 85 → [256, 128, 64, 32] → 3 with 65,347 parameters
     - **Features:** 85 clean (58 base + 3 temporal + 20 first-order differences + 4 autocorrelation)
     - **Quantization:** INT32 symmetric quantization (30-bit precision)
     - **Inference:** INT32 SIMD with AVX-512/AVX2/MKL/Scalar fallback (~10μs, 98K predictions/sec)
     - **Data Quality:** 0 constant features (proper feature engineering)
   - **News Ingestion System:** NewsAPI integration with C++23 sentiment analysis (260 lines)
   - **Employment Signals:** BLS data integration with sector rotation (1,064+ records)
   - **Sentiment Analysis:** Keyword-based scoring (-1.0 to 1.0, 60+ keywords each direction)
2. **Correlation Analysis Tool** - Statistical relationships, time-lagged correlations, leading indicators
3. **Trading Decision Engine** - 52 options strategies integrated, explainable decisions, real-time risk management

**Technology Stack (Tier 1 POC):**
- **Languages:** C++23 (core), Python 3.13 (ML), Rust (optional), CUDA C++ (GPU kernels)
- **Database:** DuckDB ONLY (PostgreSQL deferred to Tier 2 after profitability)
- **Parallelization:** MPI, OpenMP, UPC++, GASNet-EX, OpenSHMEM (32+ cores)
- **ML/AI:** PyTorch, Transformers, XGBoost, SHAP
- **C++/Python Integration:** pybind11 for performance-critical code (bypasses GIL)
  - **Risk Analytics Bindings:** 8 modules exposed with RAII memory safety (Position Sizer, Monte Carlo, Stop Loss, Risk Manager, VaR Calculator, Stress Testing, Performance Metrics, Correlation Analyzer)
  - **Memory Management:** `std::shared_ptr` holder pattern - **NO raw new/delete**
  - **Integration Tests:** 210-line comprehensive test suite (8/8 modules passing)
- **GPU Acceleration:** JAX with CUDA 13.0 (NVIDIA RTX 4070, 12GB VRAM)
- **CUDA Development:** CUDA Toolkit 13.0 installed (nvcc compiler, cuBLAS, cuDNN)
- **Document Processing:** Maven + OpenJDK 25 + Apache Tika
- **Build System:** CMake 4.1.2+ with Ninja generator (required for C++23 modules)
  - **LLVM 21 Configuration:** Proper separation of compiler projects vs runtime libraries
  - **ENABLE_PROJECTS:** clang, clang-tools-extra, openmp, flang, mlir
  - **ENABLE_RUNTIMES:** libcxx, libcxxabi, **libunwind** (proper exception handling)
- **Code Quality:** clang-tidy (C++ Core Guidelines enforcement)
- **Package Manager:** uv (10-100x faster than pip, project-based, no venv needed)
- **Execution:** All Python code runs with `uv run python script.py`

**Performance Acceleration:**
- **JAX + GPU:** 3.8x faster dashboard (4.6s → 1.2s load time) - ACTIVE
- **JIT Compilation:** Pre-compiled during startup for instant runtime execution
- **Automatic Differentiation:** Exact Greeks calculation (not finite differences)
- **Batch Vectorization:** 10-50x speedup for large-scale operations
- **SIMD Risk Analytics (AVX-512/AVX2):** **PRODUCTION READY** - Comprehensive SIMD acceleration with Python bindings
  - **Monte Carlo Simulator:** **9.87M simulations/sec** (AVX2), 6-7x speedup over scalar
    - AVX-512: 8 doubles/iteration, AVX2: 4 doubles/iteration, scalar fallback
    - Statistics: vectorized_sum, vectorized_mean_variance with FMA instructions
    - Benchmark: 250K sims in 25.33ms (9.87M sims/sec peak)
    - **Python Bindings:** Full pybind11 integration with RAII memory safety
  - **Correlation Analyzer:** AVX-512/AVX2 Pearson correlation (6-8x speedup)
    - Replaces MKL with direct intrinsics for better control
    - Horizontal reduction for efficient cross-lane summation
  - **8 Python-Bound Modules:** Position Sizer, Monte Carlo, Stop Loss, Risk Manager, VaR Calculator, Stress Testing, Performance Metrics, Correlation Analyzer
  - **Memory Management:** `std::shared_ptr` holder pattern with automatic destruction - **NO raw new/delete**
  - **Move Semantics:** Added to 5 mutex-containing classes for pybind11 compatibility
  - **Integration Tests:** 210-line comprehensive test suite (100% passing, 8/8 modules)
  - **Documentation:** Comprehensive Doxygen-style comments + 35-line RAII memory safety guide
- **SIMD (AVX2):** C++ correlation engine (3-6x faster, 100K+ points/sec)
- **SIMD JSON Parsing (simdjson v4.2.1):** 3-32x faster JSON parsing (ACTIVE, migrated all hot paths)
  - Quote parsing: **32.2x** (3449ns → 107ns, 120 req/min)
  - NewsAPI: **23.0x** (8474ns → 369ns, 96 req/day)
  - Account data: **28.4x** (3383ns → 119ns, 60 req/min)
  - Annual savings: ~6.7B CPU cycles
- **OpenMP:** Multi-threaded options pricing and matrix operations
- **CUDA C++ Kernels:** Available for native GPU acceleration (100-1000x potential speedup)
- **Tensor Cores:** RTX 4070 supports FP16/BF16 mixed precision (2-4x additional boost)

**Memory Safety & Validation:**
- **Valgrind v3.24.0:** Memory leak detection and thread safety validation (ACTIVE)
  - Built from source with Clang 21 support
  - Zero memory leaks detected across all C++ modules
  - 23 unit tests + 8 benchmarks validated
  - Thread-safe thread_local pattern for simdjson
  - Report: `docs/VALGRIND_MEMORY_SAFETY_REPORT.md`
  - Test script: `benchmarks/run_valgrind_tests.sh`

## Critical Principles

1. **DuckDB-First:** Zero setup time. PostgreSQL ONLY after proving profitability.
2. **Options First:** Algorithmic options day trading before stock strategies.
3. **Explainability:** Every trade decision must be interpretable and auditable.
4. **Validation:** Free data (3-6 months) before paid subscriptions.
5. **Speed:** Microsecond-level latency for critical paths.

## Key Documents

- **PRD:** `docs/PRD.md` - Complete requirements and cost analysis
- **Database Strategy:** `docs/architecture/database-strategy-analysis.md` - DuckDB-first rationale
- **Playbook:** `playbooks/complete-tier1-setup.yml` - Environment setup (DuckDB, no PostgreSQL)
- **Architecture:** `docs/architecture/*` - Detailed system designs
- **ML Model Evolution:** `MODEL_EVOLUTION_SUMMARY.md` - 60-feature model development (17→42→52→60 features)
- **Custom Model Summary:** `CUSTOM_MODEL_SUMMARY.md` - Deployment guide for v3.0 model
- **News Ingestion:** `docs/NEWS_INGESTION_SYSTEM.md` - Complete architecture and implementation (620 lines)
- **News Quick Start:** `docs/NEWS_INGESTION_QUICKSTART.md` - Setup guide with actual build output (450 lines)
- **News Delivery:** `docs/NEWS_INGESTION_DELIVERY_SUMMARY.md` - Implementation summary and status
- **FRED Integration:** `docs/PRICE_PREDICTOR_SYSTEM.md` - FRED rates + ML price predictor (800 lines)
- **Implementation Summary:** `docs/IMPLEMENTATION_SUMMARY_2025-11-11.md` - FRED + Predictor delivery report
- **JAX Acceleration:** `docs/JAX_DASHBOARD_ACCELERATION.md` - GPU-accelerated dashboard (5-100x speedup)
- **GPU Performance:** `docs/GPU_ACCELERATION_RESULTS.md` - Benchmark results and optimization guide
- **Performance Optimizations:** `docs/PERFORMANCE_OPTIMIZATIONS.md` - OpenMP + SIMD + simdjson implementation details
- **Coding Standards:** `docs/CODING_STANDARDS.md` - simdjson usage patterns and migration guide
- **Testing:** `python_tests/README.md` - Centralized test suite documentation (12 test files)

## Socket-Based OAuth Token Refresh System

**Status:** ✅ PRODUCTION READY - Real-time token updates via Unix domain sockets

### Architecture

The OAuth token refresh system eliminates manual token management and bot restarts:

**C++ Trading Bot (token_manager.cpp):**
- **Socket Server:** Unix domain socket on `/tmp/bigbrother_token.sock`
- **Threading:** Separate non-blocking thread with `select()` timeout (1 second)
- **Thread Safety:** `std::mutex` protection for all OAuth2Config updates
- **Protocol:** JSON messages (`access_token`, `refresh_token`, `expires_at`)
- **Resource Management:** RAII-compliant (socket FD cleanup, thread join, file unlink)
- **Implementation:** 135 lines of production-ready C++ code

**Python Token Refresh Service (token_refresh_service.py):**
- **Refresh Cycle:** Every 25 minutes (configurable)
- **Token Source:** Schwab OAuth API (refresh_token grant)
- **Socket Client:** Connects to C++ bot, sends updated tokens
- **Error Handling:** Automatic retry on failures, comprehensive logging
- **Signal Handlers:** Graceful shutdown (SIGTERM, SIGINT)
- **Implementation:** 182 lines of production-ready Python code

**Token Parsing Fix (main.cpp):**
- Fixed `expires_at` parsing from ISO 8601 to Unix timestamps (`std::stoll()`)
- Proper `time_t` conversion with `std::chrono::system_clock::from_time_t()`
- Time remaining calculation and logging for monitoring

### Key Features

1. **Zero Downtime:** Tokens update while bot runs (no restarts)
2. **Thread-Safe:** All token updates protected by `std::mutex`
3. **Non-Blocking:** Socket server doesn't interfere with trading operations
4. **Fail-Safe:** Bot continues operating if refresh service unavailable
5. **Auditable:** Complete logging to `logs/token_refresh.log` and `logs/bigbrother.log`
6. **Robust:** Handles network failures, JSON parse errors, socket errors gracefully

### Integration Points

**Startup (scripts/phase5_setup.py):**
```bash
# Automatically starts token refresh service
uv run python scripts/phase5_setup.py --quick --start-all
```

**Shutdown (scripts/shutdown.sh):**
```bash
# Enhanced with pgrep pattern matching to kill ALL instances
./scripts/shutdown.sh
# Kills: bigbrother, streamlit, token_refresh_service.py
# Cleans: /tmp/bigbrother_token.sock, PID files
```

**Monitoring:**
```bash
# Token refresh logs
tail -f logs/token_refresh.log

# Bot token updates
tail -f logs/bigbrother.log | grep -i token

# Socket verification
ls -l /tmp/bigbrother_token.sock
```

### Benefits

- **Automatic:** No manual token management for 7 days
- **Seamless:** Trading continues uninterrupted during token updates
- **Reliable:** Tested with live OAuth flow and multiple concurrent instances
- **Maintainable:** Clean separation of concerns (Python handles OAuth, C++ receives updates)
- **Portable:** Works on any Unix system (Linux, macOS, WSL2)

### Testing & Validation

✅ Token refresh service tested with live Schwab OAuth flow
✅ Socket communication verified with running bot
✅ Shutdown script tested with multiple concurrent instances
✅ Token expiry parsing validated with Unix timestamps
✅ Time remaining calculations confirmed accurate
✅ No memory leaks (RAII-compliant resource management)

## Phase 5: Paper Trading Validation (ACTIVE)

**Timeline:** Days 0-21 | **Started:** November 10, 2025

### ⚠️ **CRITICAL BUG FIXES (November 12, 2025)** ⚠️

**Status:** ✅ ALL CRITICAL ISSUES RESOLVED | **Commit:** [0200aba](https://github.com/oldboldpilot/BigBrotherAnalytics/commit/0200aba)

**Why Trading Failed Today (0/3 orders placed):**

#### Bug #1: Quote Bid/Ask = $0.00 (CRITICAL - FIXED ✅)
- **Impact:** 100% order failure - all orders rejected "Limit price must be positive"
- **Root Cause:** Cached quotes returned bid/ask = 0.0, after-hours fix only ran on fresh API calls
- **Fix:** [schwab_api.cppm:631-696](../src/schwab_api/schwab_api.cppm#L631-L696) - Apply fix to BOTH cached and fresh quotes
- **Result:** No more $0.00 order failures

#### Bug #2: ML Predictions Catastrophic (-22,000%) (CRITICAL - FIXED ✅)
- **Impact:** Model predicted SPY -22,013%, QQQ -16,868%, IWM -11,252% (account-destroying)
- **Root Cause:** Only 12 days historical data vs 26 required, model using "approximate features"
- **Fix:** [strategies.cppm:1241-1258](../src/trading_decision/strategies.cppm#L1241-L1258) - Reject predictions outside ±50% range
- **Result:** Prevents catastrophic trades, safety net until model retrained
- **Next Step:** Collect 26+ days data: `uv run python scripts/data_collection/historical_data.py --days 30`

#### Bug #3: Python 3.14 → 3.13 (FIXED ✅)
- **Impact:** Documentation inconsistency
- **Fix:** Updated all playbooks and docs to Python 3.13

**Expected Improvements:**
- Order Success Rate: 0% → >90%
- Quote Validity: 0% → 100%
- ML Predictions Sane: 0% → 100%
- Risk of Catastrophic Loss: High → Low

**Full Report:** [docs/CRITICAL_BUG_FIXES_2025-11-12.md](../docs/CRITICAL_BUG_FIXES_2025-11-12.md)

---

### News Ingestion System (IMPLEMENTED)

**Status:** Production Ready | **Integration:** 8/8 Phase 5 checks passing (100%)

The system includes real-time financial news tracking with sentiment analysis:

**C++ Core Modules:**
- `src/market_intelligence/sentiment_analyzer.cppm` (260 lines) - Keyword-based sentiment analysis
- `src/market_intelligence/news_ingestion.cppm` (402 lines) - NewsAPI integration with rate limiting
- Python bindings: 236KB shared library (`news_ingestion_py.cpython-314-x86_64-linux-gnu.so`)

**Architecture:**
- **Direct error handling:** Uses `std::unexpected(Error::make(code, msg))` pattern (no circuit breaker)
- **Python-delegated storage:** Database writes handled by Python layer for simplicity
- **Result<T> pattern:** Comprehensive error propagation with detailed messages
- **Rate limiting:** 1 second between API calls (100 requests/day limit)

**Build System:**
```bash
# Build C++ modules (requires CMake + Ninja for C++23 module support)
cmake -G Ninja -B build
ninja -C build market_intelligence

# Build Python bindings
ninja -C build news_ingestion_py

# Set library path (required for shared library dependencies)
export LD_LIBRARY_PATH=/home/muyiwa/Development/BigBrotherAnalytics/build:$LD_LIBRARY_PATH
```

**Features:**
- **Sentiment Analysis:** Fast keyword-based scoring (-1.0 to 1.0, 60+ keywords each direction)
- **NewsAPI Integration:** Fetches news with automatic deduplication (article_id hash from URL)
- **Dashboard Integration:** News feed view with filtering and visualization
- **Database Schema:** `news_articles` table with sentiment metrics and indexes

**Validation:**
- clang-tidy: 0 errors, 36 acceptable warnings
- Build: SUCCESS (all modules compile)
- Integration: 8/8 Phase 5 checks passing

See `docs/NEWS_INGESTION_SYSTEM.md` for complete architecture and implementation details.

### FRED Risk-Free Rates Integration (IMPLEMENTED)

**Status:** Production Ready | **Integration:** Live Treasury rates with Python bindings

The system provides real-time risk-free rates from the Federal Reserve Economic Data (FRED) API:

**C++ Core Modules:**
- `src/market_intelligence/fred_rates.cppm` (455 lines) - FRED API client with AVX2 SIMD optimization
- `src/market_intelligence/fred_rate_provider.cppm` (280 lines) - Thread-safe singleton with auto-refresh
- `src/market_intelligence/fred_rates_simd.hpp` (350 lines) - AVX2 intrinsics for 4x speedup
- Python bindings: 364KB shared library (`fred_rates_py.cpython-313-x86_64-linux-gnu.so`)

**Live Data Sources:**
- 3-Month Treasury Bill (DGS3MO): 3.920%
- 2-Year Treasury Note (DGS2): 3.550%
- 5-Year Treasury Note (DGS5): 3.670%
- 10-Year Treasury Note (DGS10): 4.110%
- 30-Year Treasury Bond (DGS30): 4.700%
- Federal Funds Rate (DFF): 3.870%

**Features:**
- **SIMD Optimization:** AVX2 intrinsics for 4x faster JSON parsing
- **Auto-Refresh:** Background thread with configurable interval (default: 1 hour)
- **Thread-Safe:** Singleton pattern with std::mutex protection
- **Caching:** 1-hour TTL to minimize API calls
- **Python Integration:** Full pybind11 bindings for dashboard access

**Performance:**
- API response time: <300ms
- JSON parsing: 4x faster with AVX2 (0.8ms vs 3.2ms)
- Single rate fetch: 280ms (API latency dominates)
- Batch 6 series: 1.8s

**Usage:**
```cpp
// C++
auto& provider = FREDRateProvider::getInstance();
provider.initialize(api_key);
double rf_rate = provider.getRiskFreeRate();
provider.startAutoRefresh(3600);
```

```python
# Python
from fred_rates_py import FREDRatesFetcher, FREDConfig, RateSeries
config = FREDConfig()
config.api_key = "your_key"
fetcher = FREDRatesFetcher(config)
rate = fetcher.get_risk_free_rate(RateSeries.ThreeMonthTreasury)
```

**Dashboard Integration (November 12, 2025):**
- ✅ FRED rates widget displaying live Treasury yields
- ✅ Yield curve chart with proper gridlines (fixed plotly methods)
- ✅ 1-hour caching for API rate limit compliance
- ✅ Fallback to Python API (requests module) when C++ bindings unavailable
- ✅ 2Y-10Y spread analysis with recession indicators

**Known Issues Resolved:**
1. `ModuleNotFoundError: requests` - Installed via `uv pip install requests`
2. `AttributeError: update_yaxis` - Fixed to `update_yaxes` (plural)
3. Database path resolution - Fixed 3-level traversal in dashboard views
4. JAX groupby column naming - Added rename for sentiment aggregation

See `docs/PRICE_PREDICTOR_SYSTEM.md` for complete documentation.

### ML-Based Price Predictor v3.0 (C++ INTEGRATED & PROFITABLE)

**Status:** ✅ C++ SIMD-Optimized | **Integration:** ONNX Runtime + CUDA + AVX2 | **Accuracy:** 56.3% (5d), 56.6% (20d)

The system provides multi-horizon price forecasts using machine learning:

**C++ Core Modules:**
- `src/market_intelligence/feature_extractor.cppm` (620 lines) - 60-feature extraction with AVX2 SIMD
- `src/market_intelligence/price_predictor.cppm` (525 lines) - ONNX inference with StandardScaler
- `models/price_predictor.onnx` (15KB + 230KB weights) - Deployed v3.0 model

**Architecture v3.0:**
- **Input Layer:** 60 comprehensive features (identification + time + treasury + Greeks + sentiment + price + momentum + volatility + interactions + directionality)
- **Hidden Layers:** 256 → 128 → 64 → 32 neurons (ReLU + dropout 0.3→0.21)
- **Output Layer:** 3 forecasts (1-day, 5-day, 20-day price change %)
- **Loss Function:** DirectionalLoss (90% direction focus, 10% magnitude)
- **Normalization:** AVX2 SIMD StandardScaler (8x speedup)
- **Parameters:** 58,947 total (vs 27K in v1.0)

**Feature Categories (60 total):**
1. **Identification (3):** symbol_encoded, sector_encoded, is_option
2. **Time (8):** hour, minute, day_of_week, day_of_month, month, quarter, day_of_year, is_market_open
3. **Treasury Rates (7):** fed_funds, 3mo, 2yr, 5yr, 10yr, slope, inversion
4. **Options Greeks (6):** delta, gamma, theta, vega, rho, implied_volatility
5. **Sentiment (2):** avg_sentiment, news_count
6. **Price (5):** close, open, high, low, volume
7. **Momentum (7):** return_1d/5d/20d, RSI, MACD, signal, volume_ratio
8. **Volatility (4):** ATR, BB_upper/lower, BB_position
9. **Interactions (10):** sentiment×momentum, volume×RSI, yield×volatility, delta×IV, etc.
10. **Directionality (8):** trend_strength, price_above_MA5/20, momentum_3d, win_rate, etc.

**Performance:**
- Feature extraction: <0.5ms (60 features with SIMD)
- StandardScaler normalization: <0.1ms (AVX2 8x speedup)
- Single prediction: <1ms (ONNX Runtime + CUDA)
- Batch 1000: <15ms (GPU)
- Overall speedup: 4-8x vs scalar implementation

**Trading Signals:**
- **STRONG_BUY:** Expected gain > 5%
- **BUY:** Expected gain 2-5%
- **HOLD:** Expected change -2% to +2%
- **SELL:** Expected loss 2-5%
- **STRONG_SELL:** Expected loss > 5%

**Confidence Scoring:**
- Separate scores for 1-day, 5-day, 20-day forecasts
- Range: 0.0 (no confidence) to 1.0 (high confidence)
- Based on prediction magnitude and historical accuracy

**Example Output:**
```
Price Prediction for AAPL:
  1-Day:   +0.56%  (Confidence: 67.5%)  [HOLD]
  5-Day:   +1.49%  (Confidence: 57.5%)  [HOLD]
  20-Day:  +3.73%  (Confidence: 47.5%)  [BUY]
Overall Signal: 🟡 HOLD (Weighted Change: +0.80%)
```

**Training Results v3.0 (November 12, 2025):**
- **Training Data:** 24,300 samples from 20 symbols (SPY, QQQ, IWM, sectors, commodities, bonds)
- **Date Range:** 5 years (2020-12-10 to 2025-10-13)
- **Model Evolution:** v1.0 (17 features) → v2.0 (42 features, 47.8%) → v2.5 (52 features, 52.6%) → v3.0 (60 features, 56.3-56.6%)
- **Hardware:** RTX 4070 SUPER GPU (12GB VRAM, CUDA 12.8)
- **Training Time:** 20 epochs, early stopping, ~1.8 seconds
- **RMSE:** 2.38% (1d), 5.04% (5d), 8.80% (20d)
- **Directional Accuracy (Test Set):**
  - 1-day: 51.7% ⚠️ (not recommended for trading)
  - 5-day: **56.3%** ✅ **PROFITABLE** (above 55% target)
  - 20-day: **56.6%** ✅ **PROFITABLE** (above 55% target, BEST)
- **Model Files:**
  - PyTorch: `models/price_predictor_best.pth` (667KB)
  - ONNX: `models/price_predictor.onnx` + `.onnx.data` (15KB + 230KB)
  - Info: `models/price_predictor_info.json`
  - Metadata: `models/custom_features_metadata.json`
- **Database:** `data/custom_training_data.duckdb` (25MB, 24,300 samples, 60 features)

**C++ Integration (November 12, 2025):**
```cpp
// Extract 60 features with FeatureContext
FeatureContext context;
context.symbol_id = 0;  // SPY
context.timestamp = std::chrono::system_clock::now();
context.fed_funds_rate = 0.0387f;
context.treasury_10yr = 0.0411f;
context.avg_sentiment = 0.0f;
context.news_count = 0.0f;
// ... populate all context fields

auto features = FeatureExtractor::extractFeatures(
    close, open, high, low, volume,
    price_history, volume_history, context
);

// ONNX inference with CUDA
auto prediction = PricePredictor::getInstance().predict(symbol, features);
if (prediction) {
    // Use 5-day and 20-day signals (profitable)
    if (prediction->confidence_5d >= 0.55f && prediction->day_5_change > 2.0f) {
        // TRADE: 56.3% win rate
    }
}
```

**Next Steps:**
1. ✅ v3.0 model trained with 60 features (56.3% 5d, 56.6% 20d accuracy)
2. ✅ C++ integration with SIMD optimizations (AVX2 StandardScaler)
3. ✅ ONNX Runtime + CUDA deployed (<1ms inference)
4. ✅ Real-Time Risk Management (VaR + Sharpe with AVX2 SIMD)
5. 🔄 1-2 days paper trading with 5d/20d ML signals
6. 💰 GO LIVE (start with $500-$1000 positions)

See `MODEL_EVOLUTION_SUMMARY.md`, `CUSTOM_MODEL_SUMMARY.md`, and `docs/PRICE_PREDICTOR_SYSTEM.md` for complete documentation.

### ML Integration & Real-Time Risk Management (IMPLEMENTED)

**Status:** ✅ Production Ready | **Integration:** November 12, 2025 | **Build:** Successful

The trading engine now features fully automated ML predictions with real-time risk monitoring:

**ML Integration (ONNX Runtime + CUDA):**
- **MLPredictorStrategy** - New strategy class using trained model (180 lines)
- **ONNX Runtime C++ API** - Direct GPU-accelerated inference (<1ms)
- **Feature Extraction** - Real-time 17-feature extraction from market quotes
- **Automated Signals** - BUY/SELL/HOLD signals every 60 seconds
- **Integration Point:** `src/main.cpp:235` - Wired into StrategyManager
- **Model Files:** `models/price_predictor.onnx` (12KB + 50KB weights)

**Real-Time Risk Management:**
- **VaR (95% confidence)** - Historical simulation, calculated every cycle (~5μs)
- **Sharpe Ratio** - Risk-adjusted returns with AVX2 SIMD optimization (~8μs)
- **Automated Halts:**
  - VaR < -3% → Trading halted
  - Daily loss > $900 → Trading halted
- **Performance:** <15μs total risk calculation overhead per cycle

**SIMD Performance Optimizations:**
- **AVX2 Intrinsics** - 4-wide parallel processing for VaR/Sharpe
- **VaR Calculation:** ~5μs for 252 samples (4x speedup vs scalar)
- **Sharpe Ratio:** ~8μs for 252 samples (3.8x speedup vs scalar)
- **Compiler Flags:** Verified `-O3 -march=native -mavx2 -mfma`

**Trading Cycle (60 seconds):**
```
1. Fetch Market Data (Schwab API)
2. Update Risk Metrics (VaR, Sharpe)
3. Check Risk Thresholds
   ├── VaR < -3% → HALT
   └── Daily Loss > $900 → HALT
4. Generate Signals
   ├── MLPredictorStrategy (ONNX CUDA)
   ├── StraddleStrategy
   ├── StrangleStrategy
   └── VolatilityArbStrategy
5. Aggregate Signals
6. Execute Trades (if risk OK)
7. Repeat
```

**Files Modified/Created:**
1. `src/main.cpp` (Lines 235-238, 352-412, 651-670)
   - Added MLPredictorStrategy to strategy list
   - Integrated VaR/Sharpe into trading cycle
   - Added automated halt conditions

2. `src/risk_management/risk_management.cppm` (+200 lines)
   - Added `calculateVaR95()` with historical simulation
   - Added `calculateSharpeRatio()` with AVX2 SIMD
   - Added `updateReturnHistory()` buffer management

3. `src/risk_management/risk.cppm` (+2 lines)
   - Added `var_95` and `sharpe_ratio` fields to PortfolioRisk

4. `src/market_intelligence/price_predictor.cppm` (+150 lines)
   - Implemented ONNX Runtime C++ API integration
   - Added CUDA execution provider
   - Implemented `runInference()` method

5. `src/market_intelligence/feature_extractor.cppm` (Updated)
   - Updated PriceFeatures struct (17 required + 8 extended)
   - Reordered fields to match trained model exactly
   - Updated `toArray()` method

6. `src/trading_decision/strategies.cppm` (+180 lines)
   - Created MLPredictorStrategy class
   - Implemented `generateSignals()` with ML predictions
   - Implemented `extractFeatures()` from real-time quotes

7. `ML_INTEGRATION_DEPLOYMENT_GUIDE.md` (Created, 500+ lines)
   - Complete deployment documentation
   - Architecture overview with diagrams
   - Performance benchmarks
   - Troubleshooting guide

**Build Status:**
```bash
ninja -C build bigbrother
# [6/6] Linking CXX executable bin/bigbrother
# ✅ Build successful
```

**Recent Improvements (November 12, 2025):**
1. ✅ **Accurate Feature Extraction** - 30-day historical buffers implemented
   - Added price_history_, volume_history_, high_history_, low_history_
   - Accurate RSI(14), MACD, Bollinger Bands, ATR(14), Volume SMA(20)
   - Falls back to approximations only when history < 26 days
   - Expected 2-3% accuracy improvement (53.4% → 56%+ for 1-day)
   - File: `src/trading_decision/strategies.cppm`

**Known Limitations:**
1. **Model needs retraining pipeline** - Currently using static 5-year trained model
   - TODO: Implement weekly retraining with rolling 2-year window
   - Priority: Medium (current model is profitable)
   - Timeline: Week 2-4 after going live

**Expected Performance:**
- Monthly ROI: ~$275-400/month (conservative, $1K positions)
- Win Rate: 57.6% (5-day), 59.9% (20-day)
- Risk: Automated halts protect capital
- Timeline: 1-2 days paper trading → go live

See `ML_INTEGRATION_DEPLOYMENT_GUIDE.md` for complete deployment steps, monitoring, and troubleshooting.

---

## GPU & CUDA Infrastructure (AVAILABLE)

**Hardware Status:** ✅ Fully Configured | **Last Verified:** November 12, 2025

### GPU Hardware
- **Model:** NVIDIA GeForce RTX 4070
- **Architecture:** Ada Lovelace (4th Gen RTX)
- **VRAM:** 12,282 MB (12GB)
- **CUDA Cores:** 5,888
- **Tensor Cores:** 184 (4th gen - FP16/BF16/FP8/INT8)
- **RT Cores:** 46 (3rd gen ray tracing)
- **Base Clock:** 1.92 GHz
- **Boost Clock:** 2.48 GHz
- **Memory Bandwidth:** 504 GB/s
- **TDP:** 220W
- **Current Usage:** 2.2 GB VRAM, 10% utilization (idle)

### CUDA Software Stack
- **CUDA Driver:** 581.80 (supports CUDA 13.0)
- **CUDA Toolkit:** 13.0 (installed with nvcc compiler)
- **Compute Capability:** 8.9 (Ada Lovelace)
- **cuBLAS:** Installed (CUDA Basic Linear Algebra Subroutines)
- **cuDNN:** Available (Deep Neural Network library)
- **Status:** Ready for native CUDA C++ kernel development

### Current GPU Utilization

**Active (Python via JAX):**
- Dashboard acceleration: 3.8x speedup (4.6s → 1.2s load time)
- Greeks calculation: Automatic differentiation (exact, not finite difference)
- Batch operations: 10-50x speedup for vectorized computations
- JIT compilation: Pre-compiled at startup for instant runtime

**Available (C++ Native CUDA):**
- Feature extraction: Potential 100-200x speedup vs AVX2
- Neural network inference: 100-1000x speedup for batch predictions
- Matrix operations: cuBLAS optimized (2-5x faster than CPU BLAS)
- Tensor cores: Mixed precision (FP16/BF16) for 2-4x additional boost

### Performance Targets

**CPU Baseline (AVX2 + OpenMP):**
- Feature extraction: 0.6ms (25 features)
- Single prediction: 8.2ms
- Batch 1000: 950ms

**GPU Target (CUDA):**
- Feature extraction: <0.01ms (60x faster)
- Single prediction: 0.9ms (9x faster)
- Batch 1000: 8.5ms (111x faster)
- Batch 10000: 75ms (12,700x faster)

### Build Integration

**CMake Configuration (when ready):**
```cmake
# Enable CUDA support
find_package(CUDATool kit 13.0 REQUIRED)
enable_language(CUDA)

# CUDA targets
add_library(cuda_price_predictor
    src/market_intelligence/cuda_price_predictor.cu
)

set_target_properties(cuda_price_predictor PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89  # Ada Lovelace (RTX 4070)
)

target_compile_options(cuda_price_predictor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v
        -gencode arch=compute_89,code=sm_89  # RTX 4070
    >
)
```

**Usage Priority:**
1. ✅ **Current:** JAX GPU acceleration for dashboard (working great)
2. 🔵 **Optional:** Native CUDA C++ kernels (after model training complete)
3. 🔵 **Future:** Multi-GPU support for distributed backtesting

**Recommendation:** Focus on model training first. CUDA native kernels can wait until after we have a trained model generating actionable signals.

---

### Trading Reporting System (IMPLEMENTED)

**Status:** Production Ready | **Integration:** Comprehensive reporting infrastructure

The system includes automated daily and weekly report generation with signal analysis:

**Components:**
- `scripts/reporting/generate_daily_report.py` (750+ lines) - Daily trading analysis
- `scripts/reporting/generate_weekly_report.py` (680+ lines) - Weekly performance summaries
- `docs/TRADING_REPORTING_SYSTEM.md` (650+ lines) - Complete documentation

**Daily Reports Include:**
- Executive summary (account value, trades, execution rate)
- Trade execution details with Greeks at signal generation
- Signal analysis (generated, executed, rejected by reason)
- Risk compliance status (risk rejections, budget constraints)
- Market conditions (IV metrics, DTE analysis)
- Output formats: JSON (structured data) and HTML (browser-viewable)

**Weekly Reports Include:**
- Performance summary (execution rate, Sharpe ratio, risk/reward)
- Strategy comparison table (signals, returns, acceptance rates)
- Signal acceptance rates by strategy
- Risk analysis and budget impact modeling
- Automated recommendations based on metrics
- Output formats: JSON and HTML

**Database Integration:**
- Uses `trading_signals` table (auto-detected)
- 10+ analytical views for signal tracking
- Zero configuration required
- DuckDB read-only queries (safe, fast)

**Features:**
- Real-time metrics from live trading activity
- Cost distribution analysis for budget optimization
- Strategy performance comparison
- Rejection reason breakdown and trends
- Risk compliance monitoring
- Sharpe ratio calculation
- No external dependencies (DuckDB native)

See `docs/TRADING_REPORTING_SYSTEM.md` for complete architecture, API reference, and usage examples.

### Trading Platform Architecture (Loose Coupling via Dependency Inversion)

**Status:** IMPLEMENTED | **Location:** `src/core/trading/` | **Integration:** Production Ready

The system features a three-layer loosely coupled architecture for multi-platform trading support:

**Architecture Layers:**

1. **Platform-Agnostic Types** (`order_types.cppm` - 175 lines)
   - Common data structures: `Position`, `Order`, `OrderSide`, `OrderType`, `OrderStatus`
   - Safety flags: `is_bot_managed`, `managed_by`, `bot_strategy` (prevents manual position interference)
   - Shared across all trading platforms (Schwab, IBKR, TD Ameritrade, etc.)

2. **Abstract Interface** (`platform_interface.cppm` - 142 lines)
   - `TradingPlatformInterface` - Pure virtual base class defining platform contract
   - Core methods: `submitOrder()`, `cancelOrder()`, `modifyOrder()`, `getOrders()`, `getPositions()`
   - Enforces Dependency Inversion Principle (SOLID) - high-level logic depends on abstraction, not concrete implementations

3. **Platform-Agnostic Business Logic** (`orders_manager.cppm` - 600+ lines)
   - `OrdersManager` depends ONLY on `TradingPlatformInterface` abstraction
   - Zero coupling to platform-specific code (Schwab, IBKR, etc.)
   - Dependency injection via constructor: `OrdersManager(db_path, std::unique_ptr<TradingPlatformInterface> platform)`
   - Handles order lifecycle, state management, database persistence

4. **Platform-Specific Implementations** (`src/schwab_api/schwab_order_executor.cppm` - 382 lines)
   - Schwab `OrderExecutor` implements `TradingPlatformInterface`
   - Adapter pattern: converts between platform-specific types (`schwab::Order`) and common types (`trading::Order`)
   - Type conversions: `chrono::time_point` ↔ `int64_t` milliseconds
   - Injected at runtime into `OrdersManager` (runtime polymorphism)

**Key Benefits:**
- Multi-Platform Support: Add IBKR, TD Ameritrade, Alpaca without modifying `OrdersManager`
- Testability: Easy to mock platform implementations for unit testing
- Maintainability: Clean separation of concerns, platform-specific code isolated
- Scalability: New platforms added by implementing `TradingPlatformInterface`
- Type Safety: Compile-time interface verification with C++23 modules

**Build Configuration:**
- New library: `trading_core` (SHARED) with module chain: `order_types` → `platform_interface` → `orders_manager`
- `schwab_api` now links `trading_core` for loose coupling
- Module dependency chain enforced by C++23 module system

**Testing & Validation:**
- Regression test suite: `scripts/test_loose_coupling_architecture.sh` (379 lines)
- 12 comprehensive tests, 32 assertions
- All tests passing: 32/32 (100% success rate)
- Validates: dependency inversion, type conversion, no circular dependencies, CMake configuration

**Documentation:**
- Complete architecture guide: `docs/TRADING_PLATFORM_ARCHITECTURE.md` (590 lines)
- Step-by-step guide for adding new trading platforms (IBKR, TD Ameritrade, Alpaca, etc.)
- Design patterns: Dependency Inversion, Adapter, Dependency Injection

**Example Usage:**
```cpp
// Runtime dependency injection
auto schwab_platform = std::make_unique<schwab::OrderExecutor>(api_client);
auto orders_manager = OrdersManager("data/orders.db", std::move(schwab_platform));

// Business logic unchanged regardless of platform
orders_manager.submitOrder(order);
orders_manager.getPositions();
```

### Quick Start (New System Deployment)

**🚀 One-Command Bootstrap (Fresh Machine → Production Ready in 5-15 min):**
```bash
git clone <repo>
cd BigBrotherAnalytics
./scripts/bootstrap.sh

# This single script:
# 1. Checks prerequisites (ansible, uv, git)
# 2. Runs ansible playbook (Clang 21, libc++, OpenMP, MPI, DuckDB)
# 3. Compiles C++ project
# 4. Sets up Python environment
# 5. Initializes database and tax configuration
# 6. Verifies everything is working
```

**Result:** Complete portability - works on any Unix system without hardcoded paths.

### Daily Workflow (CRITICAL: Use `uv run python` for ALL Python commands)

**Morning (Pre-Market - Recommended):**
```bash
cd /home/muyiwa/Development/BigBrotherAnalytics

# Single command - verifies all systems + auto-refreshes token + starts services (10-15 sec)
uv run python scripts/phase5_setup.py --quick --start-all

# Automatic features:
# - Socket-based OAuth token refresh (real-time updates every 25 min, no restarts)
# - Kills duplicate processes (prevents port conflicts)
# - Starts dashboard (http://localhost:8501)
# - Starts trading engine (background)
# - Starts token refresh service (Python → C++ socket)
# - Comprehensive health checks
```

**Alternative (Manual Start):**
```bash
# Verify only (no auto-start)
uv run python scripts/phase5_setup.py --quick

# Manual start
uv run streamlit run dashboard/app.py
./build/bigbrother
```

**Evening (Market Close):**
```bash
# Graceful shutdown + reports + backup
uv run python scripts/phase5_shutdown.py
```

### Tax Configuration (2025)
- **Filing Status:** Married Filing Jointly
- **State:** California
- **Base Income:** $300,000 (from other sources)
- **Short-term:** 37.1% (24% federal + 9.3% CA + 3.8% Medicare)
- **Long-term:** 28.1% (15% federal + 9.3% CA + 3.8% Medicare)
- **YTD Tracking:** Incremental throughout 2025

### Phase 5 Complete (100% Production Ready)
- ✅ **Socket-based OAuth token refresh** (real-time updates, 25-min cycle, Python ↔ C++ IPC)
- ✅ **Unified setup script** (replaces 10+ commands, automatic OAuth refresh)
- ✅ **Auto-start services** (--start-all flag starts dashboard + trading engine)
- ✅ **Tax tracking** (married joint CA, YTD accumulation, 1.5% accurate fees)
- ✅ **End-of-day automation** (reports, tax calc, backup)
- ✅ **Paper trading config** ($2,000 position limit)
- ✅ **Manual position protection** (100% verified)
- ✅ **All tests passing** (87/87, 100% success rate)
- ✅ **Complete documentation**
- ✅ **Error handling & circuit breakers**
- ✅ **Performance optimization** (4.09x speedup)
- ✅ **Monitoring & alerts** (9 health checks, 27 types, token validation)
- ✅ **Health monitoring** (real-time token validation, system status)
- ✅ **News Ingestion System** (8/8 checks passing, 236KB Python bindings, sentiment analysis)
- ✅ **Trading Reporting System** (daily/weekly reports, signal analysis, HTML+JSON output)

### Success Criteria
- **Win Rate:** ≥55% (profitable after 37.1% tax + 1.5% fees)
- **Risk Limits:** $2,000 position, $2,000 daily loss, 2-3 concurrent
- **Tax Accuracy:** Real-time YTD cumulative tracking
- **Zero Manual Position Violations:** 100% protection
- **Token Management:** Socket-based real-time refresh (25-min cycle, zero downtime, thread-safe)

## AI Orchestration System

**For structured development, use the AI orchestration system:**

```
+------------------+
|   Orchestrator   | ← Coordinates all agents
+------------------+
        ↓
+------------------+
|    PRD Writer    | ← Requirements
+------------------+
        ↓
+------------------+
| System Architect | ← Architecture
+------------------+
        ↓
+------------------+
|  File Creator    | ← Implementation
+------------------+
        ↓
+---------------------------+
| Self-Correction (Hooks)   | ← Validation
| Playwright + Schema Guard |
+---------------------------+
```

**Available Agents:**
- `PROMPTS/orchestrator.md` - Coordinates multi-agent workflows
- `PROMPTS/prd_writer.md` - Updates PRD and requirements
- `PROMPTS/architecture_design.md` - Designs system architecture
- `PROMPTS/file_creator.md` - Generates implementation code
- `PROMPTS/self_correction.md` - Validates and auto-fixes code
- `PROMPTS/code_review.md` - Reviews code quality
- `PROMPTS/debugging.md` - Debugs issues systematically

**Workflows:**
- `WORKFLOWS/feature_implementation.md` - Implement new features
- `WORKFLOWS/bug_fix.md` - Fix bugs systematically

**See `ai/README.md` for complete orchestration guide.**

## AI Assistant Guidelines

**CRITICAL: Authorship Standard**
- **ALL files created/modified MUST include:** Author: Olumuyiwa Oluwasanmi
- Include author in file headers for: .cpp, .cppm, .hpp, .py, .sh, .yaml, .md
- See `docs/CODING_STANDARDS.md` Section 13 for complete authorship rules
- **NO co-authoring** - Only Olumuyiwa Oluwasanmi as author
- **NO AI attribution** - Do not add "Generated with", "Co-Authored-By", or any AI tool references
- **NO AI assistance mentions** - Do not include "with AI assistance" or similar phrases

**CRITICAL: Code Quality Enforcement**
- **ALWAYS run validation before committing:** `./scripts/validate_code.sh`
- **Automated checks include:**
  1. clang-tidy (COMPREHENSIVE - see below)
  2. Build verification with ninja
  3. Trailing return syntax
  4. Module structure
  5. [[nodiscard]] attributes
  6. Documentation completeness

**clang-tidy Comprehensive Checks:**
- cppcoreguidelines-* (C++ Core Guidelines)
- cert-* (CERT C++ Secure Coding Standard)
- concurrency-* (Thread safety, race conditions, deadlocks)
- performance-* (Optimization opportunities)
- portability-* (Cross-platform compatibility)
- openmp-* (OpenMP parallelization safety)
- mpi-* (MPI message passing correctness)
- modernize-* (Modern C++23 features)
- bugprone-* (Bug detection)
- readability-* (Code readability and naming)

**Note:** cppcheck removed - clang-tidy is more comprehensive

## Naming Conventions (CRITICAL FOR ALL AGENTS)

**IMPORTANT:** Follow these naming conventions exactly to avoid clang-tidy warnings:

| Entity | Convention | Example |
|--------|------------|---------|
| Namespaces | `lower_case` | `bigbrother::utils` |
| Classes/Structs | `CamelCase` | `RiskManager`, `TradingSignal` |
| Functions | `camelBack` | `calculatePrice()`, `getName()` |
| Variables/Parameters | `lower_case` | `spot_price`, `volatility` |
| Local constants | `lower_case` | `const auto sum = 0.0;` |
| Constexpr constants | `lower_case` | `constexpr auto pi = 3.14;` |
| Private members | `lower_case_` | `double price_;` (trailing _) |
| Enums | `CamelCase` | `enum class SignalType` |
| Enum values | `CamelCase` | `SignalType::Buy` |

**Key Rules:**
- **Local const variables:** Use `lower_case` (NOT UPPER_CASE) - Modern C++ convention
- **Function names:** Start lowercase, use camelCase (`calculatePrice`, not `CalculatePrice`)
- **Private members:** Always have trailing `_` (`price_`, not `price` or `m_price`)
- **Compile-time constants:** Prefer `lower_case` (can use `kCamelCase` if desired)

**Example:**
```cpp
auto calculateBlackScholes(
    double spot_price,           // parameter: lower_case
    double strike_price          // parameter: lower_case
) -> double {
    const auto time_value = 1.0;       // local const: lower_case
    const auto drift = 0.05;           // local const: lower_case
    auto result = spot_price * drift;  // variable: lower_case
    return result;
}

class OptionPricer {
private:
    double strike_;    // private member: lower_case with trailing _
    double spot_;      // private member: lower_case with trailing _
};
```

**Build and Test Workflow (MANDATORY):**
```bash
# 1. Make changes to code

# 2. Run validation (catches most issues)
./scripts/validate_code.sh

# 3. Build (clang-tidy runs AUTOMATICALLY before build)
#    NOTE: CMake defines `_LIBCPP_NO_ABI_TAG` to avoid libc++ abi_tag redeclaration errors.
#    Keep custom libc++ module path synchronized with this flag when precompiling `std`.
cd build && ninja
# CMake runs scripts/run_clang_tidy.sh automatically
# Build is BLOCKED if clang-tidy finds errors

# 4. Run tests
env LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    ./run_tests.sh

# 5. Commit (pre-commit hook runs clang-tidy AUTOMATICALLY)
git add -A && git commit -m "message

Author: Olumuyiwa Oluwasanmi"
# Pre-commit hook runs clang-tidy on staged files
# Commit is BLOCKED if clang-tidy finds errors
```

**CRITICAL: clang-tidy runs AUTOMATICALLY:**
- Before every build (CMake runs it)
- Before every commit (pre-commit hook)
- Bypassing is NOT ALLOWED without explicit justification

## C++23 Modules (MANDATORY)

**ALL new C++ code MUST use C++23 modules - NO traditional headers.**

### Module File Structure

**Every `.cppm` file follows this structure:**

```cpp
/**
 * BigBrotherAnalytics - Component Name
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: YYYY-MM-DD
 *
 * Following C++ Core Guidelines:
 * - Trailing return type syntax throughout
 * - std::expected for error handling
 */

// 1. Global module fragment (standard library ONLY)
module;
#include <vector>
#include <string>
#include <expected>

// 2. Module declaration
export module bigbrother.component.name;

// 3. Module imports (internal dependencies)
import bigbrother.utils.types;
import bigbrother.utils.logger;

// 4. Exported interface (public API)
export namespace bigbrother::component {
    [[nodiscard]] auto calculate() -> double;

    class PublicAPI {
    public:
        auto method() -> void;
    private:
        double value_;
    };
}

// 5. Private implementation (optional)
module :private;
namespace bigbrother::component {
    auto PublicAPI::method() -> void {
        const auto local_const = 42;  // lower_case
    }
}
```

### Module Naming Convention

```
bigbrother.<category>.<component>

Examples:
- bigbrother.utils.types
- bigbrother.utils.logger
- bigbrother.options.pricing
- bigbrother.risk_management
- bigbrother.schwab_api
- bigbrother.strategy
```

### Module Rules (Enforced by clang-tidy)

✅ **ALWAYS:**
- Use `.cppm` extension for module files
- Start with `module;` for standard library includes
- Use `export module bigbrother.category.component;`
- Use trailing return syntax: `auto func() -> ReturnType`
- Add `[[nodiscard]]` to all getters
- Use `module :private;` for implementation details
- Import with `import bigbrother.module.name;`

❌ **NEVER:**
- Use `#include` for project headers (only standard library)
- Mix modules and headers
- Create circular module dependencies
- Forget `export` keyword
- Use old-style function syntax
- Export implementation details

### CMake Integration

```cmake
add_library(bigbrother_modules)
target_sources(bigbrother_modules
    PUBLIC FILE_SET CXX_MODULES FILES
        src/utils/types.cppm
        src/options/pricing.cppm
        # ... other modules
)
```

### Compilation

```bash
# Build (modules compile to BMI files first)
cd build
env CC=/usr/local/bin/clang CXX=/usr/local/bin/clang++ cmake -G Ninja ..
ninja bigbrother
```

**Module Compilation Flow:**
```
module.cppm → BMI (.pcm) → object.o → linked executable
              ↑ cached
importing.cpp uses BMI (fast)
```

## DuckDB Bridge Pattern (C++23 Module Compatibility) ✅ COMPLETED

**Status:** ✅ MIGRATION COMPLETE | **Date:** November 12, 2025 | **Build:** 61/61 targets successful

### Overview

The DuckDB bridge pattern is a specialized design that isolates DuckDB's incomplete types from C++23 modules, enabling clean database integration without module compilation errors.

**Problem Solved:**
- C++23 modules cannot include `<duckdb.hpp>` directly due to incomplete forward declarations (`duckdb::QueryNode`, `duckdb::LogicalOperator`)
- These incomplete types cause instantiation errors when used in module headers
- Separate translation units can't avoid the error due to template instantiation requirements
- Building against DuckDB headers directly requires entire header parsing in module fragments

**Solution: Opaque Handle Pattern**
- Bridge library (`src/schwab_api/duckdb_bridge.{hpp,cpp}`) provides opaque handle types
- All DuckDB implementation hidden in .cpp file (not exposed to modules)
- Modules see only forward-declared handles, never actual DuckDB types
- Similar to established patterns in OpenMP/MPI/System APIs

### Architecture

**Bridge Components:**

```
┌─────────────────────────────────────────┐
│  C++23 Modules (resilient_database.cppm, │
│  token_manager.cpp, etc.)                │
│  - Use DatabaseHandle, ConnectionHandle  │
│  - See only opaque pointer wrappers      │
└──────────────┬──────────────────────────┘
               │ include "duckdb_bridge.hpp"
               ▼
┌─────────────────────────────────────────┐
│  duckdb_bridge.hpp (opaque declarations) │
│  - DatabaseHandle (Impl*)                │
│  - ConnectionHandle (Impl*)              │
│  - QueryResultHandle (void*)             │
│  - PreparedStatementHandle (void*)       │
└──────────────┬──────────────────────────┘
               │ link
               ▼
┌─────────────────────────────────────────┐
│  duckdb_bridge.cpp (implementation)      │
│  - Hidden: #include <duckdb.hpp>        │
│  - Actual DuckDB types instantiated     │
│  - All database operations implemented  │
└─────────────────────────────────────────┘
```

### Opaque Handle Types

All DuckDB operations use these lightweight handles:

```cpp
class DatabaseHandle {
  private:
    struct Impl;  // Forward declare, never instantiate in header
    std::unique_ptr<Impl> pImpl_;  // Only void* stored at runtime
};

class ConnectionHandle {
  private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

class QueryResultHandle {
  private:
    void* pImpl_{nullptr};  // Simple void* for query results
};

class PreparedStatementHandle {
  private:
    void* pImpl_{nullptr};  // Simple void* for prepared statements
};
```

**Why this works:**
- Handles are opaque - modules don't see actual DuckDB types
- Implementations only in .cpp file - DuckDB header never parsed by modules
- Pimpl pattern (pointer-to-implementation) isolates internals
- Runtime overhead: only pointer indirection (negligible)

### Bridge API Surface

All DuckDB operations exposed through simple C++ functions:

**Database Operations:**
```cpp
auto openDatabase(std::string const& path) -> std::unique_ptr<DatabaseHandle>;
```

**Connection Management:**
```cpp
auto createConnection(DatabaseHandle& db) -> std::unique_ptr<ConnectionHandle>;
```

**Synchronous Queries:**
```cpp
auto executeQuery(ConnectionHandle& conn, std::string const& query) -> bool;
```

**Queries with Results:**
```cpp
auto executeQueryWithResults(ConnectionHandle& conn, std::string const& query)
    -> std::unique_ptr<QueryResultHandle>;
```

**Result Set Access:**
```cpp
auto getRowCount(QueryResultHandle const& result) -> size_t;
auto getColumnCount(QueryResultHandle const& result) -> size_t;
auto getColumnName(QueryResultHandle const& result, size_t col_idx) -> std::string;
auto hasError(QueryResultHandle const& result) -> bool;
auto getErrorMessage(QueryResultHandle const& result) -> std::string;
```

**Value Extraction:**
```cpp
auto getValueAsString(QueryResultHandle const& result, size_t col_idx, size_t row_idx)
    -> std::string;
auto getValueAsInt(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> int32_t;
auto getValueAsDouble(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> double;
auto isValueNull(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> bool;
```

**Prepared Statements:**
```cpp
auto prepareStatement(ConnectionHandle& conn, std::string const& query)
    -> std::unique_ptr<PreparedStatementHandle>;

auto bindString(PreparedStatementHandle& stmt, int index, std::string const& value) -> bool;
auto bindInt(PreparedStatementHandle& stmt, int index, int value) -> bool;
auto bindDouble(PreparedStatementHandle& stmt, int index, double value) -> bool;
auto executeStatement(PreparedStatementHandle& stmt) -> bool;
```

### Usage Examples

**In C++23 Modules:**

```cpp
// In global module fragment
module;
#include "schwab_api/duckdb_bridge.hpp"  // ✅ Only bridge header needed
// #include <duckdb.hpp>                  // ❌ NEVER include DuckDB directly

export module my_module;

using namespace bigbrother::duckdb_bridge;

class MyDatabaseComponent {
  private:
    std::unique_ptr<DatabaseHandle> db_;
    std::unique_ptr<ConnectionHandle> conn_;

  public:
    auto initialize() -> void {
        // Open database using bridge
        db_ = openDatabase("data/bigbrother.duckdb");
        if (!db_) throw std::runtime_error("Failed to open database");

        // Create connection
        conn_ = createConnection(*db_);
        if (!conn_) throw std::runtime_error("Failed to create connection");

        // Execute DDL safely
        auto success = executeQuery(*conn_,
            "CREATE TABLE IF NOT EXISTS trading_signals ("
            "  id INTEGER PRIMARY KEY,"
            "  strategy TEXT,"
            "  signal TEXT"
            ")"
        );
        if (!success) throw std::runtime_error("Failed to create table");
    }

    auto queryData(std::string const& symbol) -> std::vector<std::string> {
        auto result = executeQueryWithResults(*conn_,
            "SELECT signal FROM trading_signals WHERE symbol = '" + symbol + "'"
        );

        std::vector<std::string> signals;
        for (size_t row = 0; row < getRowCount(*result); ++row) {
            signals.push_back(getValueAsString(*result, 0, row));
        }
        return signals;
    }
};
```

**With Prepared Statements:**

```cpp
auto insertTrade(std::string const& strategy, int profit) -> bool {
    auto stmt = prepareStatement(*conn_,
        "INSERT INTO trades (strategy, profit) VALUES (?, ?)"
    );

    bindString(*stmt, 1, strategy);
    bindInt(*stmt, 2, profit);
    return executeStatement(*stmt);
}
```

### Exception Handling Changes

**Before (Direct DuckDB):**
```cpp
try {
    auto result = conn_->query("SELECT * FROM trades");
} catch (duckdb::Exception& e) {
    logger_->error("Query failed: {}", e.what());
}
```

**After (Using Bridge):**
```cpp
auto result = executeQueryWithResults(*conn_, "SELECT * FROM trades");
if (!result || hasError(*result)) {
    logger_->error("Query failed: {}", getErrorMessage(*result));
}
```

**Key Change:**
- DuckDB's C++ exceptions are not exposed (C++ wrapper for C API)
- Bridge returns `bool` success/failure and provides error messages
- Calling code uses standard error handling, no DuckDB-specific exceptions
- Graceful degradation if database unavailable

### Migrated Files (Phase 2 Complete)

**Token Manager:**
- File: `src/schwab_api/token_manager.cpp`
- Changes: Replaced `#include <duckdb.hpp>` with `#include "duckdb_bridge.hpp"`
- Status: ✅ Building successfully, no segfaults, tokens updating in real-time

**Resilient Database Wrapper:**
- File: `src/utils/resilient_database.cppm`
- Changes: Uses bridge for all database operations
- Features: Automatic retry (lock contention), transaction management, connection pooling
- Status: ✅ C++23 module compiling, full functionality preserved

**Build Results:**
- Total targets: 61/61 ✅
- Compiler: Clang 21 with libc++
- Generator: Ninja
- C++ Standard: C++23 (`-std=c++2c`)
- Runtime: Fully functional, no segmentation faults

### Exception Handling Strategy

**Python Bindings (Keep Direct DuckDB):**
```cpp
// duckdb_bindings.cpp - Python integration
#include <duckdb.hpp>  // ✅ OK: Python bindings, not C++23 modules

try {
    auto result = conn_.query("SELECT * FROM positions");
} catch (duckdb::Exception& e) {
    throw py::error_already_set();
}
```

**Rationale:**
- Python bindings are not C++23 modules (traditional .cpp files)
- Direct DuckDB exceptions work fine in Python C++ code
- Fewer conversions = better performance for Python → C++ → DuckDB path
- Python layer already expects std::exception interface

**When to Switch to Bridge:**
- Any C++23 module needs database access → use bridge
- Any traditional .cpp module in a C++23 module chain → use bridge
- Only keep direct access in standalone .cpp files not imported by modules

### Design Principles

1. **Module Independence:** C++23 modules never see problematic headers
2. **Compile-Time Safety:** Build fails if module tries direct include
3. **Runtime Efficiency:** Opaque handles = single pointer indirection
4. **Error Handling:** Standard C++ patterns (return codes, optional messages)
5. **Extensibility:** Bridge API easily expandable for new operations
6. **Minimal Scope:** Bridge limited to DuckDB only (OpenMP, CUDA, etc. use different patterns)

### Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Direct `#include <duckdb.hpp>`** | None (fails with modules) | ❌ C++23 instantiation errors |
| **DuckDB Bridge (Current)** | ✅ Works with modules | Single pointer indirection |
| **PostgreSQL (Future)** | ✅ Async ready | Network overhead, setup complexity |
| **Header-only Library** | ❌ Modules still can't instantiate | Doesn't solve the problem |

### Performance Impact

**Negligible:**
- Single virtual pointer indirection per database operation
- DuckDB operations (milliseconds) dominate pointer overhead (<1 microsecond)
- ~0.1% theoretical overhead in micro-benchmarks
- Zero impact on trading decision latency

### Building with Bridge

```bash
# Standard build (bridge linked automatically)
cmake -G Ninja -B build
ninja -C build

# Verify bridge is being used
nm build/CMakeFiles/bigbrother.dir/src/schwab_api/token_manager.cpp.o | grep duckdb_bridge

# Test database operations
./build/bin/bigbrother --test-db
```

### Future Enhancements

- **Async DuckDB Operations:** Use bridge for non-blocking queries (Phase 6+)
- **Connection Pooling:** Bridge abstraction ready for pooling layer
- **Query Caching:** Bridge can track recent queries transparently
- **Multi-Database:** Bridge easily extended to support PostgreSQL alongside DuckDB

### See Also

- `AGENT_CODING_GUIDE.md` - Complete bridge API reference and code examples
- `docs/CPP23_MODULES_GUIDE.md` - Module compilation and structure
- `src/schwab_api/duckdb_bridge.hpp` - Bridge header (read for implementation details)
- `src/schwab_api/duckdb_bridge.cpp` - Bridge implementation (hidden DuckDB types)

### Complete Reference

**See:** `docs/CPP23_MODULES_GUIDE.md` - Comprehensive 1000+ line guide covering:
- Module structure patterns
- CMake integration
- Compilation process
- Best practices
- Common pitfalls
- Migration guide
- Real examples from BigBrotherAnalytics

**Project Status:**
- 30 C++23 modules implemented (market intelligence + Schwab API + account management)
- 100% trailing return syntax
- Zero traditional headers in new code
- Clang 21.1.5 required

### Schwab API C++23 Modules

**Architecture:**
- `bigbrother.schwab.account_types` (307 lines) - Account, Balance, Position, Transaction data structures
- `bigbrother.schwab_api` - OAuth token management + AccountClient (lightweight wrapper)
- `bigbrother.schwab.account_manager` (1080 lines) - Full account management with analytics

**Module Hierarchy:**
```
bigbrother.schwab.account_types (foundation)
  └── bigbrother.schwab_api (OAuth + API wrapper)
      └── bigbrother.schwab.account_manager (full implementation)
```

**Key Features:**
- OAuth integration via TokenManager
- Thread-safe operations with mutex protection
- Error handling with `std::expected<T, std::string>`
- Position tracking and transaction history
- Portfolio analytics (value calculation, P&L)
- Database integration (pending DuckDB API migration)

**Technical Highlights:**
- **spdlog Integration**: Uses `SPDLOG_USE_STD_FORMAT` for C++23 compatibility
- **Error Propagation**: Converts `Error` struct to `std::string` for `std::expected`
- **Rule of Five**: Explicit move deletion due to mutex member
- **AccountClient vs AccountManager**: Lightweight fluent API vs full-featured management

**Migration Benefits:**
- Faster compilation (module precompilation)
- Better encapsulation (clear exported API)
- Type safety (no ODR violations)
- Zero-warning build (stricter checks)

See [CODEBASE_STRUCTURE.md](../CODEBASE_STRUCTURE.md) Section 10 and [docs/ACCOUNT_MANAGER_CPP23_MIGRATION.md](../docs/ACCOUNT_MANAGER_CPP23_MIGRATION.md) for complete details.

### Building C++23 Modules with News Ingestion

**Prerequisites:**
- Clang 21.1.5+ (required for C++23 module support)
- Ninja build system (required for C++23 module compilation)
- clang-tidy (for validation)

**Build Commands:**
```bash
# Configure with Ninja generator (REQUIRED for C++23 modules)
cmake -G Ninja -B build

# Build market intelligence modules (includes news + sentiment)
ninja -C build market_intelligence

# Build Python bindings for news system
ninja -C build news_ingestion_py

# Verify build output (236KB library)
ls -lh build/news_ingestion_py.cpython-314-x86_64-linux-gnu.so
```

**clang-tidy Validation:**
```bash
# Validate all C++ files before building
./scripts/validate_code.sh

# Expected output:
# Files validated: 48
# Errors: 0
# Acceptable warnings: 36 (modernize-*, readability-*)
# Status: ✅ PASSED
```

**Handling Build Errors:**

1. **"module 'bigbrother.X' not found"**
   - Check CMakeLists.txt - ensure module is in FILE_SET CXX_MODULES
   - Verify module dependency order (utils → market_intelligence → bindings)

2. **"undefined symbol" errors when importing Python module**
   - Set LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH`
   - Verify shared libraries: `ldd build/news_ingestion_py.*.so`

3. **clang-tidy errors blocking build**
   - Fix all trailing return syntax issues: `auto func() -> ReturnType`
   - Add missing [[nodiscard]] attributes on getters
   - CMake runs clang-tidy AUTOMATICALLY before compilation

**News System Specific:**
- Module files: `src/market_intelligence/sentiment_analyzer.cppm` (260 lines)
- Module files: `src/market_intelligence/news_ingestion.cppm` (402 lines)
- Python bindings: `src/python_bindings/news_bindings.cpp` (110 lines)
- Output library: `news_ingestion_py.cpython-314-x86_64-linux-gnu.so` (236KB)

When helping with this project:
1. Always check database strategy first - use DuckDB for Tier 1, not PostgreSQL
2. **Read `docs/CPP23_MODULES_GUIDE.md` before writing C++ code**
3. Reference `ai/MANIFEST.md` for current goals and active agents
4. Check `ai/IMPLEMENTATION_PLAN.md` for task status and checkpoints
5. Use workflows in `ai/WORKFLOWS/` for repeatable processes
6. **For complex tasks, use the Orchestrator** (`PROMPTS/orchestrator.md`)
7. Focus on validation speed - POC has $30k at stake
