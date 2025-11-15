# BigBrotherAnalytics Coding Standards

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-10
**Version:** 1.1.0

This document defines the coding standards enforced by automated CI/CD checks.

## Overview

BigBrotherAnalytics follows modern C++23 and Python best practices with emphasis on:

**C++ Standards:**
- **Trailing return type syntax** for all functions
- **C++ Core Guidelines** compliance
- **Fluent API patterns** for usability
- **Module-based architecture** (C++23 modules)
- **Performance and safety** balance

**Python Standards:**
- **uv package manager** for all Python operations (REQUIRED)
- **Type hints** for all function signatures
- **Modern async/await** patterns for I/O operations
- **Pydantic** for data validation
- **Black + Ruff** for formatting and linting

---

## 0. Python Package Management (CRITICAL)

**Rule:** ALWAYS use `uv` for Python operations. Never use `pip`, `python`, or `python3` directly.

### ✅ Correct:
```bash
# Install packages
uv add requests pandas numpy

# Run scripts
uv run python scripts/fetch_data.py

# Run tests
uv run pytest tests/

# Sync dependencies
uv sync

# Create virtual environment (handled automatically by uv)
# No manual venv creation needed
```

### ❌ Incorrect:
```bash
pip install requests              # DON'T USE
python script.py                  # DON'T USE
python3 -m pytest                 # DON'T USE
pip3 install --upgrade package    # DON'T USE
```

**Rationale:**
- **10-100x faster** than pip
- **Deterministic** dependency resolution
- **Reproducible** builds across all machines
- **Zero configuration** - project already set up
- **Built-in virtual environment** management
- **Consistent** with project standards

**Configuration:**
- pyproject.toml already configured
- All dependencies managed via uv
- No manual pip usage required

---

## 1. C++ Single Source of Truth (CRITICAL FOR ML SYSTEMS)

**Rule:** ALL data extraction, feature extraction, and quantization operations MUST be implemented in C++ with Python bindings for training. NO Python-only implementations allowed.

### The Standard

This standard ensures ZERO variation between training and inference - perfect parity guaranteed.

**Requirements:**
1. Data loading, feature extraction, and quantization implemented in C++23 modules
2. Python bindings via pybind11 for training use ONLY
3. NO Python-only implementations for these operations
4. Modifications happen in ONE place (C++) and propagate everywhere
5. Parity tests required for all new features

### ✅ Correct Implementation:

**C++23 Module:**
```cpp
// src/market_intelligence/feature_extractor.cppm
export module bigbrother.market_intelligence.feature_extractor;

export namespace bigbrother::market_intelligence {
    class FeatureExtractor {
    public:
        [[nodiscard]] auto toArray85(
            std::span<double const> price_history,
            std::span<double const> volume_history,
            std::chrono::system_clock::time_point timestamp
        ) const -> std::array<float, 85>;

        [[nodiscard]] auto calculateGreeks(/* params */) const -> Greeks;

        [[nodiscard]] auto quantizeFeatures85(
            std::array<float, 85> const& features
        ) const -> std::array<int32_t, 85>;
    };
}
```

**Python Binding:**
```cpp
// src/python_bindings/feature_extractor_bindings.cpp
#include <pybind11/pybind11.h>
import bigbrother.market_intelligence.feature_extractor;

PYBIND11_MODULE(feature_extractor_cpp, m) {
    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def("extract_features_85", &FeatureExtractor::toArray85)
        .def("calculate_greeks", &FeatureExtractor::calculateGreeks)
        .def("quantize_features_85", &FeatureExtractor::quantizeFeatures85);
}
```

**Training Usage (Python):**
```python
#!/usr/bin/env python3
"""
Author: Olumuyiwa Oluwasanmi
"""
import sys
sys.path.insert(0, 'python')
from feature_extractor_cpp import FeatureExtractor

# Use C++ implementation for training
extractor = FeatureExtractor()
features = extractor.extract_features_85(prices, volumes, timestamp)
```

**Inference Usage (C++):**
```cpp
import bigbrother.market_intelligence.feature_extractor;

auto main() -> int {
    FeatureExtractor extractor;
    auto features = extractor.toArray85(prices, volumes, timestamp);
    // Perfect parity - same code as training!
}
```

### ❌ Incorrect (Python-Only Implementation):

```python
# ❌ WRONG - Python-only feature extraction
def extract_features(prices, volumes):
    # Hardcoded Greeks (will drift from C++ inference)
    gamma = 0.01
    theta = -0.05
    # Price ratios instead of actual prices (will drift)
    price_lag_1d = prices[-1] / prices[-2]
    return [gamma, theta, price_lag_1d, ...]
```

**Why This is Wrong:**
- Different floating-point arithmetic than C++
- Hardcoded values (gamma=0.01) vs calculated Greeks
- Price ratios vs actual prices
- Will cause model accuracy degradation over time
- Two implementations = two places for bugs

### When to Use C++ vs Python

**C++ Implementation Required (MANDATORY):**
- ✅ Data extraction from databases/APIs
- ✅ Feature calculation (technical indicators, Greeks, lags, autocorrelations)
- ✅ Quantization/dequantization
- ✅ Data preprocessing (normalization, scaling)
- ✅ ANY operation used in BOTH training AND inference

**Python Implementation Allowed:**
- ✅ Model training (PyTorch, scikit-learn)
- ✅ Hyperparameter tuning
- ✅ Visualization and plotting
- ✅ Exploratory data analysis (EDA)
- ✅ Operations used ONLY in training (never in inference)

### Parity Testing

**Required for Every Feature:**
```python
# tests/test_feature_parity.py
def test_greeks_parity():
    """Verify Greeks calculated, not hardcoded"""
    extractor = FeatureExtractor()

    greeks1 = extractor.calculate_greeks(spot=100, vol=0.20)
    greeks2 = extractor.calculate_greeks(spot=100, vol=0.40)

    # Greeks MUST differ (not hardcoded)
    assert abs(greeks1.gamma - greeks2.gamma) > 0.001
    print("✅ Greeks parity verified")

def test_quantization_parity():
    """Verify quantization round-trip accuracy"""
    extractor = FeatureExtractor()

    features = np.random.randn(85).astype(np.float32)
    quantized = extractor.quantize_features_85(features)
    dequantized = extractor.dequantize_features_85(quantized)

    max_error = np.max(np.abs(features - dequantized))
    assert max_error < 1e-6
    print(f"✅ Quantization error: {max_error:.2e} < 1e-6")
```

### Build Process

```bash
# 1. Build C++ module
ninja -C build market_intelligence

# 2. Build Python binding
ninja -C build feature_extractor_py

# 3. Test parity
PYTHONPATH=python:$PYTHONPATH uv run python tests/test_feature_parity.py

# 4. Use in training
PYTHONPATH=python:$PYTHONPATH uv run python scripts/ml/prepare_features_cpp.py
```

### Benefits Achieved

**Perfect Parity:**
- Training and inference use IDENTICAL code (byte-for-byte)
- Impossible for features to diverge
- Model accuracy stable over time

**10-20x Faster Training:**
- C++ feature extraction: ~0.5ms per sample
- Python feature extraction: ~10ms per sample

**Single Point of Maintenance:**
- Fix bug once in C++ → propagates to training AND inference
- No need to keep two implementations in sync

**Type Safety:**
- C++23 strong typing catches errors at compile time
- No runtime surprises from type mismatches

### Deprecated Code (DO NOT USE)

- ❌ `scripts/ml/prepare_custom_features.py.deprecated` - Old Python feature extraction
- ❌ Any Python scripts that duplicate C++ functionality
- ❌ Manual feature calculations in Python notebooks

### Enforcement

**Code Review:**
- Any Python-only implementation of data/feature/quantization logic will be REJECTED
- Parity tests required for all new features
- Deprecated Python code must be removed

**Automatic Checks:**
- Build system verifies Python bindings compile
- Parity tests run in CI/CD pipeline
- Model accuracy monitored for drift detection

**Rationale:**
- Eliminates feature drift and ensures perfect parity
- Reduces bugs and maintenance overhead
- Improves performance (10-20x faster training)
- Single source of truth for critical ML operations

---

## 2. Trailing Return Type Syntax

**Rule:** All functions MUST use trailing return syntax.

### ✅ Correct:
```cpp
auto calculatePrice(double spot, double strike) -> double {
    return spot - strike;
}

auto isValid() const noexcept -> bool {
    return value > 0.0;
}

[[nodiscard]] auto getSymbol() const noexcept -> std::string const& {
    return symbol_;
}
```

### ❌ Incorrect:
```cpp
double calculatePrice(double spot, double strike) {
    return spot - strike;
}

bool isValid() const noexcept {
    return value > 0.0;
}
```

**Rationale:**
- Consistent syntax across codebase
- Better readability for complex return types
- Aligns with modern C++ best practices
- Required for concepts and perfect forwarding

---

## 2. C++ Core Guidelines Compliance

We enforce key C++ Core Guidelines through clang-tidy:

### C.1: Use struct for passive data, class for invariants
```cpp
// ✅ Passive data - no invariants
struct Quote {
    std::string symbol;
    Price bid{0.0};
    Price ask{0.0};
};

// ✅ Active class - maintains invariants
class OptionPricer {
public:
    auto setStrike(Price strike) -> void {
        if (strike <= 0.0) throw std::invalid_argument("Strike must be positive");
        strike_ = strike;
    }
private:
    Price strike_{0.0};  // Invariant: always positive
};
```

### C.21: Define or delete all default operations (Rule of Five)
```cpp
class RiskManager {
public:
    RiskManager() = default;
    ~RiskManager() = default;

    // If you define one, define or delete all:
    RiskManager(RiskManager const&) = delete;
    auto operator=(RiskManager const&) -> RiskManager& = delete;
    RiskManager(RiskManager&&) noexcept = default;
    auto operator=(RiskManager&&) noexcept -> RiskManager& = default;
};
```

### F.16: Pass cheap types by value
```cpp
// ✅ Cheap types by value
auto setPrice(Price p) -> void { price_ = p; }

// ✅ Expensive types by const reference
auto setSymbol(std::string const& s) -> void { symbol_ = s; }

// ✅ Sink arguments by value + move
auto setSymbol(std::string s) -> void { symbol_ = std::move(s); }
```

### F.20: Return values, not output parameters
```cpp
// ✅ Return by value
auto calculateGreeks(Params const& p) -> Greeks {
    return Greeks{/* ... */};
}

// ❌ Output parameters
void calculateGreeks(Params const& p, Greeks& out) {
    out = Greeks{/* ... */};
}
```

### E: Use std::expected for error handling
```cpp
// ✅ Use std::expected for fallible operations
auto divideNumbers(double a, double b) -> std::expected<double, Error> {
    if (b == 0.0) {
        return std::unexpected(Error{ErrorCode::DivisionByZero, "Cannot divide by zero"});
    }
    return a / b;
}

// Usage:
auto result = divideNumbers(10.0, 2.0);
if (result) {
    std::cout << "Result: " << *result << '\n';
} else {
    std::cerr << "Error: " << result.error().message << '\n';
}
```

---

## 3. Naming Conventions (ENFORCED BY CLANG-TIDY)

**IMPORTANT:** These conventions are enforced by clang-tidy and must be followed by all AI agents and developers.

### Summary Table

| Entity | Convention | Example | Notes |
|--------|------------|---------|-------|
| **Namespaces** | `lower_case` | `bigbrother::utils` | Use nested namespaces |
| **Classes** | `CamelCase` | `RiskManager` | Start with capital |
| **Structs** | `CamelCase` | `TradingSignal` | Same as classes |
| **Functions/Methods** | `camelBack` | `calculatePrice()` | Start lowercase |
| **Local variables** | `lower_case` | `auto sum = 0.0;` | Snake case |
| **Parameters** | `lower_case` | `auto func(double spot_price)` | Snake case |
| **Member variables** | `lower_case` | `double price;` | Snake case |
| **Private members** | `lower_case_` | `double price_;` | Trailing underscore |
| **Local constants** | `lower_case` | `const auto elapsed = 100;` | **Modern C++ style** |
| **Constexpr variables** | `lower_case` | `constexpr auto pi = 3.14;` | **Can also use kName** |
| **Global constants** | `lower_case` or `kCamelCase` | `constexpr auto max_risk = 0.15;` | Prefer lower_case |
| **Enums** | `CamelCase` | `enum class SignalType` | Capital case |
| **Enum values** | `CamelCase` | `SignalType::Buy` | Capital case |

### Key Principle: Local Constants Use lower_case

**✅ Correct (Modern C++23):**
```cpp
auto calculateMean(std::vector<double> const& data) -> double {
    const auto sum = std::accumulate(data.begin(), data.end(), 0.0);
    const auto count = data.size();
    const auto mean = sum / count;
    return mean;
}
```

**❌ Incorrect (Old C style):**
```cpp
auto calculateMean(std::vector<double> const& data) -> double {
    const auto SUM = std::accumulate(data.begin(), data.end(), 0.0);  // Wrong!
    const auto COUNT = data.size();  // Wrong!
    const auto MEAN = SUM / COUNT;  // Wrong!
    return MEAN;
}
```

### Compile-Time Constants

For compile-time constants (constexpr, static constexpr), use either:

**Option 1: lower_case (Preferred)**
```cpp
constexpr auto pi = 3.14159265359;
constexpr auto speed_of_light = 299'792'458.0;
static constexpr auto max_iterations = 1000;
```

**Option 2: kCamelCase prefix (Also acceptable)**
```cpp
constexpr auto kPi = 3.14159265359;
constexpr auto kSpeedOfLight = 299'792'458.0;
static constexpr auto kMaxIterations = 1000;
```

### Complete Example

```cpp
export namespace bigbrother::pricing {  // namespace: lower_case

// Class: CamelCase
class OptionPricer {
public:
    // Function: camelBack
    [[nodiscard]] auto calculatePrice(
        double spot_price,        // parameter: lower_case
        double strike_price,      // parameter: lower_case
        double time_to_expiry     // parameter: lower_case
    ) const -> double {

        // Local constants: lower_case
        const auto risk_free = 0.041;
        const auto volatility = implied_vol_;

        // Local variables: lower_case
        auto d1 = calculateD1(spot_price, strike_price);
        auto nd1 = normalCdf(d1);

        return spot_price * nd1;
    }

private:
    double implied_vol_;      // private member: lower_case with trailing _
    double strike_;           // private member: lower_case with trailing _
};

// Struct: CamelCase
struct Greeks {
    double delta;    // member: lower_case
    double gamma;    // member: lower_case
    double theta;    // member: lower_case
};

// Enum: CamelCase, values: CamelCase
enum class OptionType {
    Call,
    Put
};

} // namespace bigbrother::pricing
```

## 4. Fluent API Patterns

All major components MUST provide fluent APIs for complex operations.

### Required Fluent APIs:
1. **OptionBuilder** - Options pricing configuration
2. **CorrelationAnalyzer** - Correlation analysis
3. **RiskAssessor** - Risk assessment
4. **TaxCalculatorBuilder** - Tax calculations
5. **BacktestRunner** - Backtesting configuration
6. **SchwabQuery** - API queries
7. **StrategyExecutor** - Strategy execution

### Fluent API Pattern:
```cpp
class OptionBuilder {
public:
    auto call() -> OptionBuilder& {
        type_ = OptionType::Call;
        return *this;
    }

    auto spot(Price s) -> OptionBuilder& {
        spot_ = s;
        return *this;
    }

    auto strike(Price k) -> OptionBuilder& {
        strike_ = k;
        return *this;
    }

    [[nodiscard]] auto price() -> Result<Price> {
        validate();
        return calculate_price();
    }

private:
    OptionType type_;
    Price spot_{0.0};
    Price strike_{0.0};
};

// Usage:
auto price = OptionBuilder()
    .call()
    .spot(150.0)
    .strike(155.0)
    .volatility(0.25)
    .price();
```

**Requirements:**
- Methods return `*this` or `Self&` for chaining
- Final method returns result (often with `[[nodiscard]]`)
- Validate state before execution
- Clear, declarative interface

---

## 4. C++23 Module System

### Module File Structure (.cppm):
```cpp
/**
 * Module documentation
 * - Purpose
 * - C++ Core Guidelines compliance
 * - Performance characteristics
 */

// Global module fragment (standard library only)
module;

#include <vector>
#include <string>
#include <expected>

// Module declaration
export module bigbrother.component.name;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

// Exported interface
export namespace bigbrother::component {
    // Public API
    auto function() -> ReturnType;

    class PublicClass {
        // ...
    };
}

// Private implementation (optional)
module :private;

namespace bigbrother::component {
    // Private helpers
}
```

### Module Rules:
1. Global module fragment (`module;`) comes first
2. Only standard library includes in global fragment
3. Use `import` for internal dependencies
4. Export namespace for public API
5. Use `module :private;` for private implementation

### Module Naming Convention:
```
bigbrother.component.subcomponent

Examples:
- bigbrother.utils.types
- bigbrother.utils.logger
- bigbrother.options.pricing
- bigbrother.risk_management
- bigbrother.strategy
```

---

## 5. Modern C++23 Features

### Required Features:
- **Concepts** for template constraints
- **Ranges** for efficient data pipelines
- **std::expected** for error handling
- **constexpr** for compile-time evaluation
- **noexcept** where appropriate
- **[[nodiscard]]** for queries and getters

### Examples:
```cpp
// Concepts
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

auto add(Numeric auto a, Numeric auto b) -> decltype(a + b) {
    return a + b;
}

// Ranges
auto process_prices(std::vector<Price> const& prices) -> double {
    return std::ranges::fold_left(
        prices | std::views::filter([](Price p) { return p > 0.0; })
               | std::views::transform([](Price p) { return p * 1.05; }),
        0.0,
        std::plus{}
    );
}

// [[nodiscard]]
[[nodiscard]] auto calculateDelta() const -> double;
[[nodiscard]] auto getSymbol() const noexcept -> std::string const&;
```

---

## 6. Performance Guidelines

### Container Selection (CRITICAL for Performance)

**Rule:** Prefer `std::unordered_map` over `std::map` for speed and flexibility unless ordering is required.

#### ✅ Preferred - std::unordered_map (O(1) average):
```cpp
// ✅ PREFERRED: Fast lookups, insertions (O(1) average)
std::unordered_map<std::string, Price> price_cache;
std::unordered_map<Symbol, QuoteData> market_data;
std::unordered_map<int, Trade> trade_history;

// For custom key types, provide hash function:
struct PairHash {
    auto operator()(std::pair<std::string, std::string> const& p) const noexcept -> size_t {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

std::unordered_map<std::pair<std::string, std::string>, double, PairHash> correlations;
```

#### ⚠️ Use std::map only when ordering required (O(log n)):
```cpp
// ⚠️ Only when you need sorted iteration
std::map<Timestamp, Trade> time_ordered_trades;  // Need chronological order
std::map<Price, int> price_ladder;  // Need price-ordered book
```

#### Additional Container Guidelines:
```cpp
// ✅ Use std::vector by default for sequences
std::vector<Trade> trades;

// ✅ Use std::flat_map (C++23) for small, cache-friendly maps
std::flat_map<std::string, Price> small_lookup;  // < 100 entries
```

**Rationale:**
- `unordered_map` is **faster** for most use cases (O(1) vs O(log n))
- More **flexible** - doesn't require operator< on keys
- Better for **real-time trading** where latency matters
- Only use `map` when you explicitly need sorted iteration

**Enforcement:**
- CodeQL checks prefer unordered_map usage
- Reviewers will question map usage without justification
- Pre-commit hooks suggest unordered_map alternatives

### Move Semantics:
```cpp
// ✅ Perfect forwarding in templates
template<typename T>
auto store(T&& value) -> void {
    data_.push_back(std::forward<T>(value));
}

// ✅ Move in sink parameters
auto setData(std::vector<Trade> data) -> void {
    data_ = std::move(data);
}
```

### Avoid Copies:
```cpp
// ✅ Return by value (RVO/NRVO)
auto getTradesForSymbol(Symbol const& symbol) -> std::vector<Trade> {
    std::vector<Trade> result;
    // ... fill result
    return result;  // No copy, move elision
}

// ✅ Use std::string_view for read-only strings
auto processSymbol(std::string_view symbol) -> void;
```

### JSON Parsing (SIMD-Accelerated)

**Rule:** Use simdjson wrapper for all JSON parsing. Achieves 3-32x speedups over nlohmann/json.

**Performance:** Benchmarked speedups on production data:
- Quote parsing: **32.2x faster** (3449ns → 107ns)
- NewsAPI parsing: **23.0x faster** (8474ns → 369ns)
- Account data: **28.4x faster** (3383ns → 119ns)
- Simple fields: **3.2x faster** (441ns → 136ns)

#### Three API Tiers:

**1. Simple API - Single Field Extraction:**
```cpp
import bigbrother.utils.simdjson_wrapper;

// ✅ Extract single fields from JSON
auto symbol_result = bigbrother::simdjson::parseAndGet<std::string>(json_response, "symbol");
auto price_result = bigbrother::simdjson::parseAndGet<double>(json_response, "price");
auto quantity_result = bigbrother::simdjson::parseAndGet<int64_t>(json_response, "quantity");

// Check result before using
if (symbol_result) {
    std::string symbol = *symbol_result;
}
```

**2. Callback API - Complex Parsing (RECOMMENDED):**
```cpp
// ✅ Parse multiple fields with full control
auto result = bigbrother::simdjson::parseAndProcess(json_response, [&](auto& doc) {
    ::simdjson::ondemand::value root_value;
    if (doc.get_value().get(root_value) != ::simdjson::SUCCESS) return;

    ::simdjson::ondemand::value aapl_value;
    if (root_value["AAPL"].get(aapl_value) != ::simdjson::SUCCESS) return;

    double bid = 0.0, ask = 0.0;
    aapl_value["bidPrice"].get_double().get(bid);
    aapl_value["askPrice"].get_double().get(ask);

    // Use extracted values...
});
```

**3. Fluent API - Builder Pattern:**
```cpp
// ✅ Chain multiple field extractions
std::string name;
double price = 0.0;
int64_t quantity = 0;

auto result = bigbrother::simdjson::from(json_response)
    .field<std::string>("name", name)
    .field<double>("price", price)
    .field<int64_t>("quantity", quantity)
    .parse();
```

#### Thread Safety:
```cpp
// ✅ Automatic thread safety via thread_local storage
// Multiple threads can parse concurrently without locks
std::vector<std::thread> threads;
for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&]() {
        auto result = bigbrother::simdjson::parseAndGet<std::string>(json, "field");
        // No race conditions - each thread has its own parser
    });
}
```

#### Migration from nlohmann/json:

**Before (nlohmann/json):**
```cpp
auto j = json::parse(response);
std::string symbol = j["symbol"];
double price = j["price"];
```

**After (simdjson wrapper):**
```cpp
auto result = bigbrother::simdjson::parseAndProcess(response, [&](auto& doc) {
    ::simdjson::ondemand::value root;
    if (doc.get_value().get(root) != ::simdjson::SUCCESS) return;

    std::string_view symbol_sv;
    root["symbol"].get_string().get(symbol_sv);
    std::string symbol{symbol_sv};

    double price;
    root["price"].get_double().get(price);
});
```

**Rationale:**
- **3-32x faster** parsing with SIMD instructions
- **Thread-safe** via thread_local storage
- **Zero-copy** on-demand parsing
- **Production-proven** in hot paths (120 req/min)
- **Automatic padding** for SIMD operations

**When NOT to use simdjson:**
- One-time startup configuration (use nlohmann for convenience)
- Small infrequent JSON (<1 req/min)
- When you need to modify and re-serialize JSON

---

## 7. Documentation Standards

### File-Level Documentation:
```cpp
/**
 * BigBrotherAnalytics - Component Name
 *
 * Brief description of component purpose.
 *
 * Following C++ Core Guidelines:
 * - C.1: struct for passive data, class for invariants
 * - F.16: Pass cheap types by value
 * - E: std::expected for error handling
 * - Trailing return type syntax throughout
 *
 * Performance: [describe performance characteristics]
 * Thread-Safety: [describe thread-safety guarantees]
 */
```

### Function Documentation:
```cpp
/**
 * Calculate option price using Black-Scholes model
 *
 * @param params Pricing parameters (spot, strike, volatility, etc.)
 * @return Option price, or error if parameters invalid
 *
 * Performance: < 1 microsecond for European options
 * Thread-Safety: Immutable, fully thread-safe
 */
[[nodiscard]] auto calculatePrice(PricingParams const& params)
    -> std::expected<Price, Error>;
```

---

## 8. Naming Conventions

### Enforced by clang-tidy:
```cpp
// Namespaces: lower_case
namespace bigbrother::options {}

// Classes/Structs: CamelCase
class OptionPricer {};
struct PricingParams {};

// Functions: camelBack
auto calculatePrice() -> Price;

// Variables: lower_case
auto strike_price = 100.0;

// Private members: lower_case with trailing underscore
class Foo {
private:
    Price strike_{0.0};
    std::string symbol_;
};

// Constants: UPPER_CASE
constexpr double PI = 3.14159265359;

// Enums: CamelCase
enum class OptionType { Call, Put };
```

---

## 9. CI/CD Enforcement

All standards are automatically enforced via GitHub Actions and local git hooks:

### Checks Run Locally (Pre-Commit Hook):
1. **Trailing Return Syntax** - Pattern matching
2. **[[nodiscard]] Attributes** - Getter verification
3. **C++23 Module Structure** - Syntax validation
4. **Documentation** - Header completeness
5. **clang-tidy** - Comprehensive analysis (see below)
6. **Code Formatting** - clang-format

### Checks Run on Every PR (GitHub Actions):
1. **CodeQL** - Security analysis (scheduled 2x daily)
2. **clang-tidy Comprehensive** - Full codebase analysis
3. **Fluent API Verification** - Pattern matching
4. **Module Standards** - Structure validation
5. **Container Selection** - Prefer unordered_map
6. **Documentation** - Completeness check

### clang-tidy Comprehensive Checks:

**Enabled Check Categories:**
- **cppcoreguidelines-*** - C++ Core Guidelines (all rules)
- **cert-*** - CERT C++ Secure Coding Standard
- **concurrency-*** - Thread safety, race conditions, deadlocks
- **performance-*** - Optimization opportunities, unnecessary copies
- **portability-*** - Cross-platform compatibility
- **openmp-*** - OpenMP parallelization safety, data races
- **mpi-*** - MPI message passing correctness
- **modernize-*** - Modern C++23 features
- **bugprone-*** - Common bug patterns
- **clang-analyzer-*** - Static analysis
- **readability-*** - Code readability

**Why No cppcheck:**
clang-tidy is more comprehensive and covers everything cppcheck does, plus:
- Better C++23 support
- C++ Core Guidelines built-in
- Parallelization checks (OpenMP, MPI)
- Security standards (CERT)
- Single tool for all checks

### Local Development:
```bash
# Run clang-tidy
clang-tidy src/your_file.cpp -- -std=c++23 -I./src

# Run cppcheck
cppcheck --enable=all --std=c++23 src/

# Format code
clang-format -i src/your_file.cpp
```

---

## 10. Examples

### Complete Module Example:
See `src/utils/types.cppm`, `src/utils/logger.cppm`, or `src/options_pricing.cppm` for reference implementations.

### Complete Fluent API Example:
See `src/options_pricing.cppm` for `OptionBuilder` implementation.

---

## References

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [C++23 Standard](https://en.cppreference.com/w/cpp/23)
- [BigBrotherAnalytics PRD](./PRD.md)
- [Architecture Documents](./architecture/)

---

## 11. File Headers and Authorship

**Rule:** All source files MUST include proper authorship information.

### Required File Header Template:

**For C++23 Modules (.cppm):**
```cpp
/**
 * BigBrotherAnalytics - [Component Name]
 *
 * [Brief description of component]
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: [Creation Date]
 *
 * Following C++ Core Guidelines:
 * - [List relevant guidelines]
 * - Trailing return type syntax throughout
 */

module;
// ... rest of module
```

**For Implementation Files (.cpp):**
```cpp
/**
 * [Component Name] Implementation
 * C++23 module implementation file
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: [Creation Date]
 */

import bigbrother.component;
// ... rest of implementation
```

**For Scripts (Python, Bash):**
```python
#!/usr/bin/env python3
"""
BigBrotherAnalytics: [Component Name]

[Brief description]

Author: Olumuyiwa Oluwasanmi
Date: [Creation Date]
"""
```

```bash
#!/bin/bash
# BigBrotherAnalytics - [Script Name]
# [Brief description]
#
# Author: Olumuyiwa Oluwasanmi
# Date: [Creation Date]
```

**For Configuration Files (YAML, TOML):**
```yaml
# BigBrotherAnalytics - [Config Name]
# Author: Olumuyiwa Oluwasanmi
# Date: [Creation Date]
```

**Authorship Standard:**
- **Primary Author:** Olumuyiwa Oluwasanmi (all files)
- **No Co-Authors:** All work attributed solely to Olumuyiwa Oluwasanmi
- **No AI Attribution:** Do not include AI tool attribution in files or commits
- **Version Control:** Git maintains complete contribution history

**This standard applies to:**
- All C++ source files (.cpp, .cppm, .hpp, .h)
- All Python scripts (.py)
- All shell scripts (.sh)
- All configuration files (.yaml, .yml, .toml, .json)
- All documentation files (.md)
- All CI/CD workflow files
- All git hooks

**Enforcement:**
- Pre-commit hook checks for author information
- CI/CD workflow validates authorship
- Code review process verifies compliance

---

## 12. Parallel Computing Standards

### OpenMP Directives

**Rule:** Use OpenMP for shared-memory parallelism on single machine.

**Correct Usage:**
```cpp
#include <omp.h>

// Parallel for loop
auto computeParallel(std::vector<double> const& data) -> std::vector<double> {
    std::vector<double> results(data.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < data.size(); ++i) {
        results[i] = expensiveComputation(data[i]);
    }
    
    return results;
}

// Parallel reduction
auto sumParallel(std::vector<double> const& data) -> double {
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    
    return sum;
}
```

**Guidelines:**
- Use `schedule(dynamic)` for unbalanced workloads
- Use `schedule(static)` for uniform workloads
- Always specify reduction operation for shared variables
- Test with `OMP_NUM_THREADS` environment variable
- Avoid false sharing (pad cache lines if needed)

### MPI Usage

**Rule:** Use MPI for distributed-memory parallelism across nodes.

**Correct Usage:**
```cpp
#ifdef USE_MPI
#include <mpi.h>

auto computeDistributed(std::vector<Data> const& data) -> Result {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Partition data by rank
    auto local_data = partitionByRank(data, rank, size);
    
    // Compute local results
    auto local_result = computeLocal(local_data);
    
    // Gather results at root
    if (rank == 0) {
        // Root collects all results
        MPI_Gather(/* ... */);
    } else {
        // Workers send results
        MPI_Send(/* ... */);
    }
    
    return global_result;
}
#endif
```

**Guidelines:**
- Always check for `USE_MPI` compile-time flag
- Initialize MPI early in main(): `MPI_Init(&argc, &argv)`
- Finalize before exit: `MPI_Finalize()`
- Use non-blocking communication when possible
- Handle rank 0 (root) specially for I/O
- Test with: `mpirun -np 4 ./application`

### Hybrid OpenMP + MPI

**Rule:** Combine for maximum performance on multi-node clusters.

**Correct Usage:**
```cpp
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <omp.h>

auto computeHybrid(std::vector<Data> const& data) -> Result {
    #ifdef USE_MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Each MPI rank gets a chunk
    auto local_data = partitionByRank(data, rank, size);
    #else
    auto local_data = data;
    #endif
    
    // Within each rank, use OpenMP threads
    std::vector<Result> local_results(local_data.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < local_data.size(); ++i) {
        local_results[i] = compute(local_data[i]);
    }
    
    #ifdef USE_MPI
    // Combine results across ranks
    Result global_result;
    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_result;
    #else
    return aggregateResults(local_results);
    #endif
}
```

**Guidelines:**
- MPI for inter-node, OpenMP for intra-node parallelism
- Each MPI rank spawns OpenMP threads
- Set thread count: `export OMP_NUM_THREADS=8`
- Launch: `mpirun -np 4 ./app` (4 ranks × 8 threads = 32 cores)
- Minimize MPI communication overhead
- Profile both MPI and OpenMP regions separately

### Thread Safety Requirements

**Rule:** All parallel code must be thread-safe.

**Thread-Safe Patterns:**
```cpp
// 1. Immutable data (preferred)
class ThreadSafeCalculator {
public:
    [[nodiscard]] auto calculate(double x) const -> double {
        // No mutable state - inherently thread-safe
        return compute(x);
    }
private:
    double const coefficient_;  // Immutable
};

// 2. Thread-local storage
auto processParallel(std::vector<double> const& data) -> std::vector<double> {
    std::vector<double> results(data.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        // Each thread has its own local_state
        thread_local LocalState state;
        results[i] = state.process(data[i]);
    }
    
    return results;
}

// 3. Mutex protection (use sparingly)
class ThreadSafeCache {
public:
    auto get(Key const& key) -> std::optional<Value> {
        std::shared_lock lock(mutex_);  // Read lock
        auto it = cache_.find(key);
        return it != cache_.end() ? std::optional{it->second} : std::nullopt;
    }
    
    auto insert(Key key, Value value) -> void {
        std::unique_lock lock(mutex_);  // Write lock
        cache_[std::move(key)] = std::move(value);
    }
    
private:
    std::unordered_map<Key, Value> cache_;
    mutable std::shared_mutex mutex_;
};
```

**Avoid:**
- Data races (multiple threads writing same memory)
- Deadlocks (lock ordering violations)
- False sharing (separate cache lines for thread-local data)
- Excessive locking (prefer lock-free algorithms)

### Performance Guidelines

**Parallelization Thresholds:**
```cpp
// Only parallelize if work is significant
constexpr size_t MIN_PARALLEL_SIZE = 10000;

auto processData(std::vector<double> const& data) -> std::vector<double> {
    if (data.size() < MIN_PARALLEL_SIZE) {
        // Sequential for small datasets
        return processSequential(data);
    } else {
        // Parallel for large datasets
        return processParallel(data);
    }
}
```

**Scaling Expectations:**
- OpenMP: 20-30x speedup on 32-core machine (parallel algorithms)
- MPI: 60-100x speedup on 4-node cluster (distributed algorithms)
- Hybrid: Best performance, but most complex

**Profiling:**
```bash
# OpenMP profiling
OMP_NUM_THREADS=1,2,4,8,16,32 ./benchmark

# MPI profiling
mpirun -np 1,2,4,8 ./benchmark

# Use Intel VTune, perf, or gprof for detailed analysis
```

---

## 13. Authorship and Attribution

**Rule:** All code and documentation authored by Olumuyiwa Oluwasanmi.

### Standard Attribution

**Author field:** Olumuyiwa Oluwasanmi

**No co-authors allowed:**
- No "Co-Authored-By" lines in commits
- No AI tool attribution in code or documentation
- No "Generated with" footers
- No "with AI assistance" mentions

### This applies to:
- All source code files (.cpp, .cppm, .hpp, .py)
- All documentation files (.md)
- Git commit messages
- Agent reports
- README files
- All project deliverables

### Examples

**Correct Commit Message:**
```bash
git commit -m "feat(options): Add Black-Scholes pricing engine

Implements European option pricing with Greeks calculation.

Performance:
- Pricing: < 1ms (p99)
- Greeks: < 0.5ms (p99)

Testing:
- 48 unit tests passed
- 96.2% coverage

Author: Olumuyiwa Oluwasanmi"
```

**Incorrect Commit Message:**
```bash
# ❌ WRONG - Do not include these lines
git commit -m "feat: Add feature

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Rationale

- **Clear ownership:** Single author simplifies intellectual property and accountability
- **Professional presentation:** Industry standard for individual projects
- **No ambiguity:** Clear who is responsible for the codebase
- **Version control:** Git history provides complete contribution tracking

---

## 14. Memory Safety Validation with Valgrind

**Rule:** All C++ code must pass Valgrind memory leak detection with zero leaks.

### Valgrind Requirements

**Mandatory for:**
- New C++ modules and libraries
- Performance-critical hot paths
- Multi-threaded code
- SIMD-accelerated code
- External library integrations

**Testing Frequency:**
- Weekly automated runs in CI/CD
- Before major releases
- After adding new dependencies
- When debugging memory issues

### Valgrind Configuration

**Tool Version:** Valgrind v3.24.0 (built from source with Clang 21)

**Installation:**
```bash
# Automated via Ansible playbook (Section 4.8)
ansible-playbook playbooks/complete-tier1-setup.yml

# Manual build (if needed)
cd /tmp
wget https://sourceware.org/pub/valgrind/valgrind-3.24.0.tar.bz2
tar -xjf valgrind-3.24.0.tar.bz2
cd valgrind-3.24.0
./autogen.sh
./configure --prefix=/usr/local --enable-only64bit \
    CC=/usr/local/bin/clang \
    CXX=/usr/local/bin/clang++
make -j$(nproc)
sudo make install

# Install debug symbols (CRITICAL for function redirection)
sudo apt-get install libc6-dbg          # Ubuntu
sudo dnf install glibc-debuginfo        # RHEL
```

### Running Memory Leak Tests

**Automated Test Script:**
```bash
# Run comprehensive memory safety validation
./benchmarks/run_valgrind_tests.sh
```

**Manual Tests:**
```bash
# Test 1: Unit tests memory leak detection
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --error-exitcode=1 \
         ./build/bin/your_unit_tests

# Test 2: Benchmark memory leak detection
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --error-exitcode=1 \
         ./build/bin/your_benchmark \
             --benchmark_min_time=0.01

# Test 3: Thread safety validation
valgrind --tool=helgrind \
         --error-exitcode=1 \
         ./build/bin/your_unit_tests \
             --gtest_filter="*Thread*"
```

### Acceptance Criteria

**Memory Leak Tests (memcheck):**
```
LEAK SUMMARY:
   definitely lost: 0 bytes in 0 blocks     ✅ REQUIRED
   indirectly lost: 0 bytes in 0 blocks     ✅ REQUIRED
     possibly lost: 0 bytes in 0 blocks     ✅ REQUIRED
   still reachable: < 1KB in < 10 blocks    ⚠️  Acceptable (library internals)
        suppressed: 0 bytes in 0 blocks     ✅ REQUIRED
```

**Thread Safety Tests (helgrind):**
- **0 real data races** (application code)
- Helgrind warnings from libc++ thread_local are **false positives** (acceptable)
- Pattern to ignore: `std::__1::__thread_local_data()`

### Known False Positives

**1. libc++ Thread-Local Storage**
```
Possible data race during read of size [1,4,8] at ... std::__1::__thread_local_data()
```
**Status:** Expected behavior, NOT a real race condition

**2. Google Benchmark "Still Reachable"**
```
still reachable: 272 bytes in 4 blocks
```
**Status:** Internal benchmark state, NOT a memory leak

### Integration with CI/CD

**GitHub Actions Workflow:**
```yaml
- name: Valgrind Memory Leak Tests
  run: |
    ./benchmarks/run_valgrind_tests.sh
  timeout-minutes: 15
  # Note: Valgrind is slow (5-10x overhead)
```

**Pre-Commit Hook (Optional):**
```bash
# .git/hooks/pre-commit
if [[ $(git diff --cached --name-only | grep -E '\.(cpp|cppm|hpp)$') ]]; then
    echo "Running Valgrind quick check..."
    valgrind --leak-check=full --error-exitcode=1 ./build/bin/quick_test
fi
```

### Documentation Requirements

**When adding new modules:**
1. Run Valgrind validation tests
2. Document results in module README
3. Add test to `run_valgrind_tests.sh` if needed
4. Update memory safety report

**Memory Safety Report Location:**
- `docs/VALGRIND_MEMORY_SAFETY_REPORT.md`

### Thread Safety Best Practices

**Use thread_local for non-thread-safe libraries:**
```cpp
// ✅ Correct: thread_local for simdjson parser
thread_local simdjson::ondemand::parser parser;

// ❌ Wrong: Shared parser across threads
static simdjson::ondemand::parser parser;  // Race condition!
```

**Avoid shared mutable state:**
```cpp
// ✅ Correct: Each thread has isolated state
auto processData(std::string_view json) -> Result<Data> {
    thread_local Parser parser;  // Thread-isolated
    return parser.parse(json);
}

// ❌ Wrong: Shared mutable state
static Parser shared_parser;  // Data race!
auto processData(std::string_view json) -> Result<Data> {
    std::lock_guard lock(mutex);  // Performance bottleneck
    return shared_parser.parse(json);
}
```

### Performance Impact

**Valgrind Overhead:**
- **10-20x slowdown** during memory checking
- **Not suitable for production** (development/testing only)
- **CI/CD recommendation:** Weekly runs, not every commit

**Native Linux vs WSL:**
- **Native Linux:** 5-10 minutes for full test suite
- **WSL2:** 30-60 minutes (use native Linux for CI/CD)

### Troubleshooting

**Error: "valgrind: Fatal error at startup: a function redirection..."**
```bash
# Solution: Install glibc debug symbols
sudo apt-get install libc6-dbg          # Ubuntu
sudo dnf install glibc-debuginfo        # RHEL
```

**Error: "Cannot find suitable architecture"**
```bash
# Solution: Rebuild Valgrind with matching compiler
./configure --enable-only64bit CC=/usr/local/bin/clang CXX=/usr/local/bin/clang++
```

**Too Many False Positives:**
```bash
# Create suppression file
valgrind --leak-check=full --gen-suppressions=all ./your_binary 2>&1 | tee valgrind.supp

# Use suppressions
valgrind --suppressions=valgrind.supp ./your_binary
```

### References

- **Valgrind Manual:** https://valgrind.org/docs/manual/manual.html
- **Helgrind Thread Safety:** https://valgrind.org/docs/manual/hg-manual.html
- **Memory Safety Report:** [VALGRIND_MEMORY_SAFETY_REPORT.md](VALGRIND_MEMORY_SAFETY_REPORT.md)
- **Test Script:** [benchmarks/run_valgrind_tests.sh](../benchmarks/run_valgrind_tests.sh)

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** 2025-11-14
**Version:** 2.0.0 - C++ Single Source of Truth Standard Established
