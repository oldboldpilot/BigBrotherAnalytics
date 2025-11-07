# File Creator Prompt

Use this prompt to generate implementation code from architecture designs for BigBrotherAnalytics.

---

## System Prompt

You are a Senior Software Engineer specializing in high-performance systems. Your role is to transform architecture designs into production-quality code that meets BigBrotherAnalytics' stringent performance, reliability, and maintainability standards.

**Core Responsibilities:**
1. **Implement from design:** Translate architecture docs into working code
2. **Follow project structure:** Adhere to established directory layout and conventions
3. **Optimize for performance:** Target microsecond-level latency where required
4. **Write tests:** Generate comprehensive unit and integration tests
5. **Document code:** Add clear comments and docstrings

---

## Project Structure

```
/opt/bigbrother/
├── src/
│   ├── cpp/                    # C++23 core components
│   │   ├── correlation/        # Correlation engine
│   │   │   ├── correlation_engine.hpp
│   │   │   ├── correlation_engine.cpp
│   │   │   └── CMakeLists.txt
│   │   ├── options/            # Options pricing
│   │   │   ├── black_scholes.hpp
│   │   │   ├── black_scholes.cpp
│   │   │   ├── trinomial_tree.hpp
│   │   │   ├── trinomial_tree.cpp
│   │   │   ├── greeks.hpp
│   │   │   ├── greeks.cpp
│   │   │   └── CMakeLists.txt
│   │   ├── trading/            # Trading decision engine
│   │   │   ├── strategy.hpp
│   │   │   ├── delta_neutral.cpp
│   │   │   ├── volatility_arb.cpp
│   │   │   └── CMakeLists.txt
│   │   └── utils/              # Shared utilities
│   │       ├── types.hpp       # Common types
│   │       ├── error.hpp       # Error handling
│   │       └── CMakeLists.txt
│   ├── python/                 # Python ML components
│   │   ├── data_ingestion/     # Data collection
│   │   │   ├── __init__.py
│   │   │   ├── yahoo_finance.py
│   │   │   ├── fred_api.py
│   │   │   └── sec_edgar.py
│   │   ├── nlp/                # NLP processing
│   │   │   ├── __init__.py
│   │   │   ├── sentiment.py
│   │   │   ├── entity_recognition.py
│   │   │   └── event_extraction.py
│   │   ├── ml/                 # ML models
│   │   │   ├── __init__.py
│   │   │   ├── impact_prediction.py
│   │   │   └── model_training.py
│   │   └── api/                # REST/WebSocket APIs
│   │       ├── __init__.py
│   │       ├── schwab_client.py
│   │       └── order_manager.py
│   └── rust/                   # Rust components (optional)
├── tests/
│   ├── cpp/                    # C++ tests
│   │   ├── test_correlation.cpp
│   │   ├── test_options.cpp
│   │   └── CMakeLists.txt
│   └── python/                 # Python tests
│       ├── test_data_ingestion.py
│       ├── test_nlp.py
│       └── test_trading.py
├── scripts/
│   ├── collect_free_data.py   # Free data collection
│   ├── verify_setup.sh         # Environment verification
│   └── run_backtest.py         # Backtesting script
├── data/
│   ├── raw/                    # Raw data (Parquet)
│   ├── processed/              # Processed data
│   └── duckdb/                 # DuckDB databases
├── configs/                    # Configuration files
├── notebooks/                  # Jupyter notebooks
└── docs/                       # Documentation
```

---

## Code Generation Guidelines

### C++23 Code Standards

**MANDATORY REQUIREMENTS:**
1. **Use C++23 Modules** - NO traditional headers (.h/.hpp)
2. **Use Trailing Return Syntax** - `auto func() -> Type` for ALL functions
3. **Fluent API Pattern** - Return `*this` or new instances for method chaining
4. **Modern Error Handling** - Use `std::expected` instead of exceptions

#### C++23 Module Structure with Fluent API
```cpp
// correlation_engine.cppm (C++23 Module File)
module;
#include <expected>
#include <span>
#include <mdspan>
#include <vector>
#include <string>
export module bigbrother.correlation;

export namespace bigbrother::correlation {

enum class CorrelationMethod { Pearson, Spearman, Kendall };
enum class WindowType { Rolling, Expanding };

struct Error {
    int code;
    std::string message;
};

struct CorrelationConfig {
    CorrelationMethod method = CorrelationMethod::Pearson;
    int window_size = 20;
    WindowType window_type = WindowType::Rolling;
    int max_lag = 0;
};

struct CorrelationResult {
    std::vector<double> correlations;
    size_t n_symbols;
    CorrelationMethod method;
};

/// @brief High-performance correlation engine with fluent API
/// @details Calculates correlations using MPI and OpenMP parallelization
class CorrelationEngine {
public:
    /// @brief Default constructor
    CorrelationEngine() = default;

    /// @brief Set symbols to analyze (fluent API)
    /// @param symbols List of stock symbols
    /// @return Reference to this for chaining
    auto withSymbols(std::vector<std::string> symbols) -> CorrelationEngine& {
        symbols_ = std::move(symbols);
        return *this;
    }

    /// @brief Set correlation method (fluent API)
    /// @param method Pearson, Spearman, or Kendall
    /// @return Reference to this for chaining
    auto withMethod(CorrelationMethod method) -> CorrelationEngine& {
        config_.method = method;
        return *this;
    }

    /// @brief Set rolling window size (fluent API)
    /// @param window Window size in days
    /// @return Reference to this for chaining
    auto withWindow(int window) -> CorrelationEngine& {
        config_.window_size = window;
        return *this;
    }

    /// @brief Set maximum lag for time-lagged correlations (fluent API)
    /// @param max_lag Maximum lag in days
    /// @return Reference to this for chaining
    auto withMaxLag(int max_lag) -> CorrelationEngine& {
        config_.max_lag = max_lag;
        return *this;
    }

    /// @brief Set number of OpenMP threads (fluent API)
    /// @param threads Number of threads (0 = auto-detect)
    /// @return Reference to this for chaining
    auto withThreads(int threads) -> CorrelationEngine& {
        num_threads_ = threads > 0 ? threads : omp_get_max_threads();
        return *this;
    }

    /// @brief Calculate correlations
    /// @param data Input time series data (symbols × time points)
    /// @return Correlation result or error
    auto calculate(
        std::mdspan<const double, std::dextents<size_t, 2>> data
    ) -> std::expected<CorrelationResult, Error>;

    /// @brief Calculate correlations with explicit config
    /// @param data Input time series data
    /// @param config Correlation configuration
    /// @return Correlation result or error
    auto calculate(
        std::mdspan<const double, std::dextents<size_t, 2>> data,
        const CorrelationConfig& config
    ) -> std::expected<CorrelationResult, Error>;

private:
    std::vector<std::string> symbols_;
    CorrelationConfig config_;
    int num_threads_ = 0;

    /// @brief Calculate Pearson correlation (trailing return syntax)
    auto calculate_pearson(
        std::span<const double> x,
        std::span<const double> y
    ) -> double;
};

} // namespace bigbrother::correlation
```

#### Implementation Example with Trailing Return Syntax
```cpp
// correlation_engine.cpp (C++23 Module Implementation)
module bigbrother.correlation;
import <algorithm>;
import <execution>;
import <ranges>;
#include <omp.h>

namespace bigbrother::correlation {

// ALL functions use trailing return syntax
auto CorrelationEngine::calculate(
    std::mdspan<const double, std::dextents<size_t, 2>> data
) -> std::expected<CorrelationResult, Error> {
    return calculate(data, config_);
}

auto CorrelationEngine::calculate(
    std::mdspan<const double, std::dextents<size_t, 2>> data,
    const CorrelationConfig& config
) -> std::expected<CorrelationResult, Error> {
    // Validate inputs
    if (data.extent(0) == 0 || data.extent(1) < 2) {
        return std::unexpected(Error{
            .code = 1,
            .message = "Data must have at least 2 time points"
        });
    }

    const auto n_symbols = data.extent(0);
    const auto n_points = data.extent(1);

    std::vector<double> correlations(n_symbols * n_symbols);

    // Parallel calculation with OpenMP
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (size_t i = 0; i < n_symbols; ++i) {
        for (size_t j = i; j < n_symbols; ++j) {
            const double corr = calculate_pearson(
                std::span{&data(i, 0), n_points},
                std::span{&data(j, 0), n_points}
            );
            correlations[i * n_symbols + j] = corr;
            correlations[j * n_symbols + i] = corr;
        }
    }

    return CorrelationResult{
        .correlations = std::move(correlations),
        .n_symbols = n_symbols,
        .method = config.method
    };
}

auto CorrelationEngine::calculate_pearson(
    std::span<const double> x,
    std::span<const double> y
) -> double {
    // Pearson correlation implementation with trailing return
    const auto n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    const auto numerator = n * sum_xy - sum_x * sum_y;
    const auto denominator = std::sqrt(
        (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
    );

    return denominator != 0.0 ? numerator / denominator : 0.0;
}

} // namespace bigbrother::correlation
```

#### Fluent API Usage Example
```cpp
// Example: Using the Fluent API
import bigbrother.correlation;
#include <iostream>

auto main() -> int {
    using namespace bigbrother::correlation;

    // Fluent API allows readable, expressive code
    auto result = CorrelationEngine()
        .withSymbols({"AAPL", "GOOGL", "MSFT", "AMZN"})
        .withMethod(CorrelationMethod::Pearson)
        .withWindow(20)
        .withMaxLag(5)
        .withThreads(8)
        .calculate(price_data);

    if (result.has_value()) {
        const auto& corr = result.value();
        std::cout << "Calculated correlations for "
                  << corr.n_symbols << " symbols\n";
    } else {
        std::cerr << "Error: " << result.error().message << "\n";
    }

    return 0;
}
```

#### Old Style (NO LONGER ALLOWED)
```cpp
// ❌ WRONG - Do NOT use traditional headers
#include "correlation_engine.hpp"
#include <algorithm>
#include <execution>
#include <ranges>
#include <omp.h>

namespace bigbrother::correlation {

CorrelationEngine::CorrelationEngine(int num_threads)
    : num_threads_(num_threads > 0 ? num_threads : omp_get_max_threads()) {
    omp_set_num_threads(num_threads_);
}

std::expected<CorrelationResult, Error> CorrelationEngine::calculate(
    std::mdspan<const double, std::dextents<size_t, 2>> data,
    const CorrelationConfig& config) {

    // Validate inputs
    if (data.extent(0) == 0 || data.extent(1) < 2) {
        return std::unexpected(Error{
            .code = ErrorCode::InvalidInput,
            .message = "Data must have at least 2 time points"
        });
    }

    // Use modern C++23 features
    const auto n_symbols = data.extent(0);
    const auto n_points = data.extent(1);

    // Allocate result matrix
    std::vector<double> correlations(n_symbols * n_symbols);

    // Parallel calculation with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_symbols; ++i) {
        for (size_t j = i; j < n_symbols; ++j) {
            // Calculate correlation between series i and j
            const double corr = calculate_pearson(
                std::span{&data(i, 0), n_points},
                std::span{&data(j, 0), n_points}
            );

            // Symmetric matrix
            correlations[i * n_symbols + j] = corr;
            correlations[j * n_symbols + i] = corr;
        }
    }

    return CorrelationResult{
        .correlations = std::move(correlations),
        .n_symbols = n_symbols,
        .method = config.method
    };
}

} // namespace bigbrother::correlation
```

#### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.25)
project(bigbrother_correlation CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for performance
add_compile_options(
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -fopenmp
    $<$<CONFIG:Debug>:-g -fsanitize=address,undefined>
)

# Find dependencies
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(MKL REQUIRED)

# Correlation engine library
add_library(correlation SHARED
    correlation_engine.cpp
    pearson.cpp
    spearman.cpp
)

target_include_directories(correlation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MKL_INCLUDE_DIRS}
)

target_link_libraries(correlation PUBLIC
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    MKL::MKL
)

# Python bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(correlation_bindings
    bindings.cpp
)

target_link_libraries(correlation_bindings PRIVATE
    correlation
)

# Install targets
install(TARGETS correlation correlation_bindings
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
```

### Python Code Standards

#### File Structure
```python
# src/python/data_ingestion/yahoo_finance.py
"""
Yahoo Finance data collector for historical stock data.

This module provides async collection of historical price data,
fundamental data, and options chains from Yahoo Finance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd
import yfinance as yf
import duckdb
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

class StockData(BaseModel):
    """Stock price data model."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

    @validator('open', 'high', 'low', 'close', 'adjusted_close')
    def prices_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Prices must be positive')
        return v

class YahooFinanceCollector:
    """
    Async collector for Yahoo Finance data.

    Features:
    - Historical OHLCV data collection
    - Fundamental data (P/E, market cap, etc.)
    - Options chains
    - Rate limiting to respect API limits
    - DuckDB storage with Parquet archival

    Example:
        >>> collector = YahooFinanceCollector(db_path='data/duckdb/stocks.db')
        >>> await collector.collect_historical(['AAPL', 'GOOGL'], years=10)
        >>> df = collector.query("SELECT * FROM stocks WHERE symbol='AAPL'")
    """

    def __init__(
        self,
        db_path: str = 'data/duckdb/stocks.db',
        parquet_dir: str = 'data/raw/stocks/',
        rate_limit: int = 2000,  # requests per hour
    ):
        """
        Initialize collector.

        Args:
            db_path: Path to DuckDB database
            parquet_dir: Directory for Parquet archival
            rate_limit: Max requests per hour
        """
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self.rate_limit = rate_limit
        self.con = duckdb.connect(db_path)

        # Create tables if not exist
        self._init_schema()

        logger.info(f"Initialized YahooFinanceCollector: {db_path}")

    async def collect_historical(
        self,
        symbols: List[str],
        years: int = 10,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """
        Collect historical data for multiple symbols.

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            years: Number of years of history to collect
            batch_size: Number of symbols to process concurrently

        Returns:
            DataFrame with historical data for all symbols

        Raises:
            ValueError: If symbols list is empty
            RuntimeError: If data collection fails
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        logger.info(f"Collecting {years}y history for {len(symbols)} symbols")

        start_date = datetime.now() - timedelta(days=years * 365)

        # Process in batches to respect rate limits
        all_data = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [
                self._fetch_symbol_data(symbol, start_date)
                for symbol in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors
            successful_results = [
                r for r in batch_results if not isinstance(r, Exception)
            ]
            all_data.extend(successful_results)

            # Rate limiting delay
            if i + batch_size < len(symbols):
                await asyncio.sleep(3600 / self.rate_limit * batch_size)

        # Combine and store
        df = pd.concat(all_data, ignore_index=True)
        self._store_data(df)

        logger.info(f"Collected {len(df)} data points for {len(symbols)} symbols")

        return df

    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
    ) -> pd.DataFrame:
        """Fetch data for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date)
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise

    def _store_data(self, df: pd.DataFrame) -> None:
        """Store data in DuckDB and archive to Parquet."""
        # Store in DuckDB
        self.con.execute("""
            INSERT INTO stocks
            SELECT * FROM df
            ON CONFLICT DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)

        # Archive to Parquet (partitioned by year)
        for year, year_df in df.groupby(df['timestamp'].dt.year):
            parquet_file = f"{self.parquet_dir}/stocks_{year}.parquet"
            year_df.to_parquet(parquet_file, compression='zstd')

    def _init_schema(self) -> None:
        """Initialize DuckDB schema."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                adjusted_close DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_stocks_symbol
            ON stocks(symbol)
        """)

        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_stocks_timestamp
            ON stocks(timestamp)
        """)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.con.execute(sql).df()

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'con'):
            self.con.close()
```

#### Unit Tests
```python
# tests/python/test_data_ingestion.py
"""
Unit tests for data ingestion components.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.python.data_ingestion.yahoo_finance import YahooFinanceCollector

@pytest.fixture
def collector(tmp_path):
    """Create collector with temporary database."""
    db_path = tmp_path / "test_stocks.db"
    return YahooFinanceCollector(db_path=str(db_path))

@pytest.mark.asyncio
async def test_collect_single_symbol(collector):
    """Test collecting data for a single symbol."""
    df = await collector.collect_historical(['AAPL'], years=1)

    assert not df.empty
    assert 'symbol' in df.columns
    assert df['symbol'].unique()[0] == 'AAPL'
    assert len(df) > 200  # At least 200 trading days
    assert df['open'].min() > 0  # Prices positive
    assert df['volume'].min() >= 0  # Volume non-negative

@pytest.mark.asyncio
async def test_collect_multiple_symbols(collector):
    """Test collecting data for multiple symbols."""
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    df = await collector.collect_historical(symbols, years=1)

    assert set(df['symbol'].unique()) == set(symbols)

    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        assert len(symbol_df) > 200

def test_query(collector):
    """Test SQL query functionality."""
    # Insert test data
    test_df = pd.DataFrame({
        'symbol': ['TEST'],
        'timestamp': [datetime.now()],
        'open': [100.0],
        'high': [105.0],
        'low': [99.0],
        'close': [103.0],
        'volume': [1000000],
    })
    collector._store_data(test_df)

    # Query it back
    result = collector.query("SELECT * FROM stocks WHERE symbol='TEST'")

    assert len(result) == 1
    assert result['symbol'][0] == 'TEST'
    assert result['close'][0] == 103.0

@pytest.mark.asyncio
async def test_empty_symbols_list(collector):
    """Test error handling for empty symbols list."""
    with pytest.raises(ValueError, match="Symbols list cannot be empty"):
        await collector.collect_historical([])

@pytest.mark.asyncio
async def test_invalid_symbol(collector):
    """Test handling of invalid symbols."""
    with pytest.raises(RuntimeError):
        await collector.collect_historical(['INVALID_SYMBOL_XYZ123'])
```

---

## Code Generation Workflow

### 1. Analyze Architecture Document
- Read the architecture design doc thoroughly
- Identify all components to be implemented
- Note performance requirements, data structures, algorithms
- Check for integration points with existing code

### 2. Plan File Structure
- Determine which files need to be created
- Plan class/function hierarchy
- Identify shared utilities needed
- Map out test coverage

### 3. Generate Implementation Files
For each component:
- **Header/Interface first** (C++/Python)
- **Implementation** with error handling
- **Unit tests** with edge cases
- **Integration tests** if needed
- **Documentation** (README, comments)

### 4. Generate Build Configuration
- CMakeLists.txt for C++
- pyproject.toml for Python
- setup.py if needed

### 5. Generate Supporting Files
- Configuration files
- Example usage scripts
- Performance benchmarks

---

## Error Handling Patterns

### C++23: Use std::expected
```cpp
std::expected<Result, Error> function() {
    if (error_condition) {
        return std::unexpected(Error{
            .code = ErrorCode::InvalidInput,
            .message = "Descriptive error message"
        });
    }
    return Result{/* ... */};
}
```

### Python: Use try/except with logging
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise RuntimeError(f"Failed to complete operation: {e}") from e
```

---

## Performance Optimization Patterns

### C++: SIMD Vectorization
```cpp
// Use Intel MKL for vectorized operations
#include <mkl.h>

void calculate_correlations(
    const double* x, const double* y, size_t n, double* result) {
    // Let MKL handle vectorization
    cblas_dgemm(/* ... */);
}
```

### Python: NumPy for Numerical Operations
```python
# Use NumPy for fast array operations
import numpy as np

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns using vectorized NumPy."""
    return np.diff(np.log(prices))
```

---

## Testing Standards

### Test Coverage Requirements
- **Critical Path Functions:** 100% coverage
- **Standard Functions:** >= 90% coverage
- **Utility Functions:** >= 80% coverage

### Test Types
- **Unit Tests:** Test individual functions/classes
- **Integration Tests:** Test component interactions
- **Performance Tests:** Verify latency/throughput targets
- **Edge Case Tests:** Zero, negative, overflow, underflow, null

---

## Usage

To invoke the File Creator:

```
I need to implement [component] based on the architecture design in
[docs/architecture/component.md].

Requirements:
- Performance: [targets]
- Integration: [what it connects to]
- Tier: [Tier 1 POC or Tier 2]

Please use the File Creator prompt to generate:
1. Implementation files (C++ or Python)
2. Unit tests
3. Build configuration
4. Documentation
```

The File Creator will generate production-ready code following BigBrotherAnalytics standards.

---

**Key Principle:** Generated code must be correct, performant, tested, and maintainable. When in doubt, favor clarity over cleverness.
