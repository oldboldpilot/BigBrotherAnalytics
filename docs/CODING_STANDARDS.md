# BigBrotherAnalytics C++ Coding Standards

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-08
**Version:** 1.0.0

This document defines the coding standards enforced by automated CI/CD checks.

## Overview

BigBrotherAnalytics follows modern C++23 best practices with emphasis on:
- **Trailing return type syntax** for all functions
- **C++ Core Guidelines** compliance
- **Fluent API patterns** for usability
- **Module-based architecture** (C++23 modules)
- **Performance and safety** balance

---

## 1. Trailing Return Type Syntax

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
- **AI Assistance:** Documented in git commits with "Co-Authored-By: Claude <noreply@anthropic.com>"
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

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** 2025-11-08
**Version:** 1.0.0
