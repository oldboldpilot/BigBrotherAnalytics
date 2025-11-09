# BigBrotherAnalytics AI Coding Assistant Instructions

## Project Overview

BigBrotherAnalytics is a high-performance AI-powered trading platform with **microsecond-level latency** requirements. **Tier 1 Foundation is COMPLETE** with 25 production-ready C++23 modules and comprehensive Python bindings integration.

**CRITICAL AUTHORSHIP STANDARD:**
- **ALL files MUST include:** Author: Olumuyiwa Oluwasanmi
- **Applies to:** Code, configs, docs, scripts, tests, CI/CD files
- **Format:** See Section 11 in docs/CODING_STANDARDS.md
- **NO co-authoring** - Only Olumuyiwa Oluwasanmi
- **Enforcement:** Pre-commit hooks + CI/CD checks

**CRITICAL BUILD STANDARDS:**
- **ALWAYS validate before commit:** `./scripts/validate_code.sh`
- **Local enforcement:** clang-tidy (11 check categories) in pre-commit hooks
- **Build verification:** ninja must succeed (clang-tidy runs AUTOMATICALLY)
- **Test verification:** All tests must pass
- **Note:** cppcheck removed - clang-tidy is comprehensive

**Agent Build Workflow:**
After any code modification, agents MUST:
1. Run `./scripts/validate_code.sh`
2. Build with `cd build && ninja` (clang-tidy runs AUTOMATICALLY before build)
3. Run tests with `./run_tests.sh`
4. Commit (clang-tidy runs AUTOMATICALLY in pre-commit hook)

**AUTOMATIC clang-tidy Enforcement:**
- Pre-Build: CMake runs clang-tidy before compiling (BLOCKS build if errors)
- Pre-Commit: Git hook runs clang-tidy on staged files (BLOCKS commit if errors)
- Cannot bypass without explicit SKIP_CLANG_TIDY=1 (NOT ALLOWED for code changes)

**Current Status (Nov 9, 2025):**
- ✅ 25 C++23 modules complete with 6 fluent APIs
- ✅ Python bindings fully integrated (DuckDB, Options, Correlation, Risk)
- ✅ Employment-driven sector rotation system operational
- ✅ Build system: 0 errors, 0 warnings (clang-tidy validated)
- ✅ Test pass rate: 92.8% (84/89 tests) - PRODUCTION READY
- ✅ DuckDB integration: 2,128 employment records, sub-10ms queries
- ✅ 11 GICS sectors with ETF mappings
- ✅ Real BLS data integration (4.7 years coverage)

**Three Core Systems:**
1. **Market Intelligence Engine** - Multi-source data processing and impact prediction (Python ML)
2. **Correlation Analysis Tool** - Time-series relationships and leading indicators (C++23/MPI, 60-100x speedup)
3. **Trading Decision Engine** - Options day trading with employment signals (C++23/Python hybrid)

## Critical Architecture Patterns

### Technology Stack (Tier 1 POC)
- **C++23 Modules** - All core components use `.cppm` files with `export module` syntax
- **DuckDB ONLY** - Zero-setup embedded database (PostgreSQL deferred until profitable)
- **Python 3.13 + uv** - All Python execution via `uv run python script.py` (no venv needed)
- **CMake + Ninja** - Build system with C++23 module support
- **Clang 21** - Required for C++23 modules compilation

### Key Project Documents (READ THESE FIRST)

**Requirements & Architecture:**
- `docs/PRD.md` (5000+ lines) - **AUTHORITATIVE** requirements document
  - Section 3.2.11: Department of Labor API integration
  - Section 3.2.12: 11 GICS Business Sectors (Energy, Tech, Healthcare, etc.)
  - Complete data sources, features, trading strategies
- `docs/architecture/` - 9+ detailed architecture documents
  - `database-strategy-analysis.md` - Why DuckDB-first
  - `market-intelligence-engine.md` - Data ingestion & NLP
  - `trading-correlation-analysis-tool.md` - Time-series analysis
  - `intelligent-trading-decision-engine.md` - Trading execution
  - `CPP23_MODULE_MIGRATION_PLAN.md` - Module patterns

**Coding Standards (MANDATORY):**
- `docs/CODING_STANDARDS.md` (623 lines) - **READ BEFORE CODING**
  - 11 comprehensive sections
  - Trailing return syntax (100% required)
  - C++ Core Guidelines enforcement
  - Prefer unordered_map over map
  - Fluent API requirements
  - File header templates with authorship
- `docs/BUILD_WORKFLOW.md` - Build process with automatic clang-tidy
- `ai/AGENT_BUILD_WORKFLOW.md` (300+ lines) - Complete workflow for AI agents

**Implementation Guides:**
- `TIER1_COMPLETE_FINAL.md` - Tier 1 status and timeline
- `TIER1_EXTENSION_CHECKLIST.md` (975 lines) - **250+ detailed tasks**
  - Section A-I: Employment data, sectors, quality, Python bindings
  - Section J: Module consolidation tasks
- `docs/EMPLOYMENT_DATA_INTEGRATION.md` (397 lines) - DoL API integration
- `BUILD_STATUS.md` - Current build state and known issues

### Module Structure Pattern
All 25 modules follow this structure:
```cpp
// Global module fragment (standard library includes ONLY)
module;
#include <vector>
#include <string>
#include <expected>

// Module declaration  
export module bigbrother.component.name;

export namespace bigbrother::component {
    // Exported interfaces with trailing return syntax
    auto calculate() -> Result<double>;
}

// Implementation section (optional)
module :private;
// Private implementation details
```

**6 Fluent APIs implemented**: OptionBuilder, CorrelationAnalyzer, RiskAssessor, SchwabQuery, TaxCalculatorBuilder, BacktestRunner

### Build System Workflow
```bash
# Configure (one time)
cd build && cmake .. -G Ninja

# Build
ninja -v

# Test  
ninja test
# OR
./run_tests.sh

# Python execution
uv run python script.py
```

## Database Strategy (Critical)

**Tier 1 (Current):** DuckDB ONLY - embedded, zero-setup, ACID compliant
- Use: All data storage, backtesting, real-time trading
- Location: `data/bigbrother.duckdb`
- Python: `import duckdb; conn = duckdb.connect('data/bigbrother.duckdb')`
- C++: DuckDB C++ API (headers in global module fragment)

**Never suggest PostgreSQL for Tier 1** - costs development time and adds complexity.

## Development Patterns

### AI Orchestration System
Specialized AI agents in `ai/` directory coordinate development workflows:
- **Agent Prompts**: `ai/PROMPTS/*.md` - Pre-configured agents for specific tasks
  - `orchestrator.md` - Multi-agent coordination
  - `file_creator.md` - Implementation generation
  - `self_correction.md` - Validation and fixes
  - `architecture_design.md` - System design
  - `prd_writer.md`, `code_review.md`, `debugging.md`
- **Workflows**: `ai/WORKFLOWS/*.md` - End-to-end process guides
  - `feature_implementation.md`, `bug_fix.md`
- **Context**: `ai/CLAUDE.md` - Always-loaded guide for AI assistants
- **Documentation Source**: Agents read from:
  - `docs/PRD.md` - Complete 5000+ line requirements document
  - `docs/architecture/*.md` - Detailed system designs (9+ architecture docs)

### File Organization
- **C++ modules**: `src/component_name/*.cppm` 
- **Python ML**: Root-level Python files with `uv` dependency management
- **Configs**: `configs/` directory with YAML templates
- **Documentation**: `docs/PRD.md` (5000+ lines, authoritative requirements)

### Trading Strategy Implementation
Example from `src/trading_decision/strategy_iron_condor.cppm`:
```cpp
export module bigbrother.strategy.iron_condor;

export namespace bigbrother::strategy {
    struct IronCondorParams {
        double min_iv_rank{50.0};        // IV rank threshold
        double profit_target_percent{50.0}; // Take profit at 50%
        // ... strategy parameters
    };
}
```

All strategies must include:
- Entry/exit rules in comments
- Risk management parameters
- Expected performance metrics
- Module-based organization

### Performance Requirements
- **Latency Target**: Microsecond-level for critical paths
- **Parallelization**: MPI, OpenMP, UPC++ for 32+ cores
- **Memory**: Cache-friendly containers (`std::flat_map`)
- **Error Handling**: `std::expected` for fallible operations

## Testing & Validation

### Build Verification
1. Check module compilation: `ninja -v` in `build/`
2. Verify executables: Check `build/bin/` and `build/lib/`
3. Run tests: `./run_tests.sh` or `ninja test`

### Python Package Management
- Use `uv` exclusively: `uv add package-name`
- Dependencies in `pyproject.toml` 
- No virtual environments needed

##clang-tidy Configuration (11 Check Categories)

**Comprehensive Checks Enabled:**
1. **cppcoreguidelines-*** - ALL C++ Core Guidelines rules
2. **cert-*** - CERT C++ Secure Coding Standard
3. **concurrency-*** - Thread safety, race conditions, deadlocks
4. **performance-*** - Optimization, unnecessary copies
5. **portability-*** - Cross-platform compatibility
6. **openmp-*** - OpenMP parallelization safety, data races
7. **mpi-*** - MPI message passing correctness
8. **modernize-*** - C++23 features, trailing return syntax
9. **bugprone-*** - Bug detection, logic errors
10. **clang-analyzer-*** - Static analysis
11. **readability-*** - Code clarity, naming

**Enforced as ERRORS (blocks build/commit):**
- modernize-use-trailing-return-type (ALL functions)
- cppcoreguidelines-special-member-functions (Rule of Five)
- modernize-use-nodiscard ([[nodiscard]] on getters)
- modernize-use-nullptr (no NULL)
- cppcoreguidelines-no-malloc (no malloc/free)

**File:** `.clang-tidy` (230+ lines with documentation)

## Naming Conventions (CRITICAL FOR ALL AI AGENTS)

**MANDATORY:** All code MUST follow these naming conventions to pass clang-tidy validation.

### Quick Reference Table

| Entity | Convention | Example | Why |
|--------|------------|---------|-----|
| **Namespaces** | `lower_case` | `bigbrother::utils` | C++ standard |
| **Classes/Structs** | `CamelCase` | `RiskManager`, `TradingSignal` | Clear type names |
| **Functions/Methods** | `camelBack` | `calculatePrice()`, `getName()` | Start lowercase |
| **Local variables** | `lower_case` | `auto spot_price = 150.0;` | Readable |
| **Parameters** | `lower_case` | `auto func(double strike_price)` | Consistent |
| **Local constants** | `lower_case` | `const auto sum = 0.0;` | **Modern C++** |
| **Constexpr constants** | `lower_case` | `constexpr auto pi = 3.14;` | **Modern C++** |
| **Member variables** | `lower_case` | `double price;` (public) | Standard |
| **Private members** | `lower_case_` | `double price_;` | Trailing _ |
| **Enums** | `CamelCase` | `enum class SignalType` | Clear types |
| **Enum values** | `CamelCase` | `SignalType::Buy`, `SignalType::Sell` | Clear values |

### Critical: Local Constants Use lower_case

**This is the MOST IMPORTANT rule** - it eliminates ~350 of the 416 clang-tidy warnings.

**✅ CORRECT (Modern C++23 - Use This!):**
```cpp
auto calculateMean(std::vector<double> const& values) -> double {
    const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    const auto count = static_cast<double>(values.size());
    const auto mean = sum / count;
    const auto squared_diff_sum = std::accumulate(
        values.begin(), values.end(), 0.0,
        [mean](double acc, double val) {
            const auto diff = val - mean;  // local const: lower_case
            return acc + diff * diff;
        });
    return mean;
}
```

**❌ WRONG (Old C Style - Causes Warnings!):**
```cpp
auto calculateMean(std::vector<double> const& values) -> double {
    const auto SUM = std::accumulate(values.begin(), values.end(), 0.0);  // WRONG!
    const auto COUNT = static_cast<double>(values.size());  // WRONG!
    const auto MEAN = SUM / COUNT;  // WRONG!
    return MEAN;
}
```

### Complete Real-World Example

```cpp
export namespace bigbrother::pricing {  // namespace: lower_case

// Enum: CamelCase, values: CamelCase
enum class OptionType {
    Call,
    Put
};

// Struct: CamelCase
struct Greeks {
    double delta;    // public member: lower_case
    double gamma;    // public member: lower_case
    double theta;    // public member: lower_case
};

// Class: CamelCase
class BlackScholesModel {
public:
    // Constructor
    explicit BlackScholesModel(double risk_free_rate)  // param: lower_case
        : risk_free_rate_{risk_free_rate} {}           // member init

    // Function: camelBack
    [[nodiscard]] auto calculatePrice(
        double spot_price,        // parameter: lower_case
        double strike_price,      // parameter: lower_case
        double volatility,        // parameter: lower_case
        double time_to_expiry,    // parameter: lower_case
        OptionType option_type    // parameter: lower_case (enum type)
    ) const -> double {

        // Local constants: lower_case (IMPORTANT!)
        const auto sqrt_t = std::sqrt(time_to_expiry);
        const auto d1 = calculateD1(spot_price, strike_price, volatility, sqrt_t);
        const auto d2 = d1 - volatility * sqrt_t;

        // Local variables: lower_case
        auto call_price = 0.0;
        auto put_price = 0.0;

        // Enum comparison
        if (option_type == OptionType::Call) {
            call_price = spot_price * normalCdf(d1);
        }

        return call_price;
    }

private:
    // Private member: lower_case with trailing _
    double risk_free_rate_;
    double cache_value_;

    // Private method: camelBack
    [[nodiscard]] auto calculateD1(
        double spot, double strike, double vol, double sqrt_t
    ) const -> double {
        const auto log_moneyness = std::log(spot / strike);  // local const: lower_case
        const auto drift = (risk_free_rate_ + 0.5 * vol * vol) * sqrt_t;
        return (log_moneyness + drift) / (vol * sqrt_t);
    }

    [[nodiscard]] auto normalCdf(double x) const -> double {
        // Implementation
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }
};

// Compile-time constants: lower_case or kCamelCase
constexpr auto pi = 3.14159265359;              // ✅ preferred
constexpr auto kGoldenRatio = 1.618033988749;   // ✅ also ok

} // namespace bigbrother::pricing
```

### Why These Conventions?

1. **lower_case for local consts** - Modern C++23 standard, matches STL patterns
2. **CamelCase for types** - Clear distinction between types and values
3. **camelBack for functions** - Standard in C++ (unlike Java/C# which use PascalCase)
4. **Trailing _ for private** - Clear visual distinction, prevents name conflicts

**This configuration is in `.clang-tidy` and enforced automatically.**

## Common Pitfalls

1. **Don't use PostgreSQL** - DuckDB only for Tier 1
2. **Module syntax matters** - Use trailing `export module` declarations
3. **OpenMP linkage** - Ensure `OpenMP::OpenMP_CXX` in CMake targets
4. **Library paths** - Set `LD_LIBRARY_PATH` for Clang 21 libraries
5. **Python execution** - Always use `uv run` prefix
6. **Trailing return syntax REQUIRED** - All functions: `auto func() -> ReturnType`
7. **Module imports** - Use `import bigbrother.module.name;` not `#include`
8. **Prefer unordered_map** - Use over map for O(1) performance (unless ordering required)
9. **Authorship** - ALL files must include: Author: Olumuyiwa Oluwasanmi
10. **clang-tidy will block you** - Fix errors before building or committing

## Employment Data & Sector Analysis (✅ COMPLETE - Nov 9, 2025)

**11 GICS Sectors Integrated:**
1. Energy (XLE) - Cyclical
2. Materials (XLB) - Cyclical
3. Industrials (XLI) - Sensitive
4. Consumer Discretionary (XLY) - Sensitive
5. Consumer Staples (XLP) - Defensive
6. Health Care (XLV) - Defensive
7. Financials (XLF) - Sensitive
8. Information Technology (XLK) - Sensitive
9. Communication Services (XLC) - Sensitive
10. Utilities (XLU) - Defensive
11. Real Estate (XLRE) - Sensitive

**Employment Signal System (OPERATIONAL):**
- **EmploymentSignalGenerator:** 479 lines, composite scoring algorithm
  - Trend analysis: 60% weight (3/6/12-month analysis)
  - Acceleration: 25% weight (rate of change)
  - Z-score: 15% weight (statistical significance)
  - Confidence scoring: 0.60-0.95 range
- **Data Coverage:** 4.7 years of BLS data (2,128 employment records)
- **Performance:** Sub-10ms queries with DuckDB
- **Integration:** Full C++ strategy integration via StrategyContext

**Sector Rotation Strategy (PRODUCTION READY):**
- **Implementation:** 632 lines of production C++ code
- **Scoring System:**
  - Employment signals: 60% weight
  - Sentiment analysis: 30% weight
  - Momentum indicators: 10% weight
- **Risk Management:** Integrated with RiskManager, sector limits enforced
- **Test Coverage:** 92.8% pass rate (84/89 tests)

**Data Sources:**
- Bureau of Labor Statistics (BLS) API - 19 employment series
- Weekly jobless claims (leading indicator)
- Monthly nonfarm payrolls (major market mover)
- Real-time DuckDB integration

**Documentation:**
- `docs/SECTOR_ROTATION_STRATEGY.md` (579 lines)
- `docs/employment_signals_architecture.md` (407 lines)
- `docs/EMPLOYMENT_DATA_INTEGRATION.md` (397 lines)
- PRD Section 3.2.11: U.S. Department of Labor
- PRD Section 3.2.12: Business Sector Classification

## Python Bindings with pybind11 (✅ COMPLETE - Nov 9, 2025)

**Framework Ready:**
- `src/python_bindings/bigbrother_bindings.cpp` - Main bindings file
- pybind11 in ansible playbook and build system
- Tagged with: PYTHON_BINDINGS for easy identification

**Completed Bindings (100% operational):**
1. **DuckDB C++ API Bindings** ✅
   - Direct database access from Python with C++ performance
   - 9 specialized functions (query, count, aggregate, join, filter, sort, analyze, optimize, vacuum)
   - Zero-copy NumPy transfers where possible
   - **Performance:** 1.41x ± 0.04x speedup, sub-10ms queries
   - **Test Coverage:** 100% (29/29 tests passing)

2. **Options Pricing Bindings** ✅
   - Black-Scholes, Trinomial Tree, Greeks calculation
   - GIL-free execution for parallel pricing
   - Proper error handling with std::expected
   - **Expected speedup:** 30-50x over pure Python

3. **Correlation Engine Bindings** ✅
   - 6 functions: pearson, spearman, cross_correlation, find_optimal_lag, rolling_correlation, correlation_matrix
   - OpenMP parallelization (bypasses GIL)
   - **Expected speedup:** 60-100x over pandas
   - **Documentation:** 371-line API reference, 187-line demo script

4. **Risk Management Bindings** ✅
   - Kelly Criterion, position sizing, Monte Carlo simulation
   - OpenMP parallelization for simulations
   - Comprehensive validation with std::expected
   - **Expected speedup:** 30-50x for Monte Carlo

**Module Size Growth:**
- DuckDB: 179KB → 188KB (+9KB, fully tested)
- Correlation: Significant enhancement (421 lines)
- Risk: Full integration with risk_management.cppm

**Documentation:**
- `docs/PYTHON_BINDINGS_GUIDE.md`
- `docs/CORRELATION_API_REFERENCE.md` (371 lines)
- `docs/implementation/CORRELATION_BINDINGS_WIRING.md` (316 lines)
- `docs/implementation/RISK_BINDINGS_WIRING.md` (286 lines)

## Implementation Task Lists

**Primary Checklist:**
- TIER1_EXTENSION_CHECKLIST (most tasks COMPLETE as of Nov 9)
  - Section A: BLS API Integration (25 tasks)
  - Section B: Private Sector Job Data (28 tasks)
  - Section C: 11 GICS Sectors Implementation (50 tasks)
  - Section D: Decision Engine Integration (25 tasks)
  - Section E: Database Schema (12 tasks)
  - Section F: Configuration & Testing (20 tasks)
  - Section G: Code Quality & Standards (30 tasks)
  - Section H: Python Bindings with pybind11 (40 tasks)
  - Section I: Documentation (10 tasks)
  - Section J: Module Consolidation (50 tasks)

**Status Tracking:**
- `BUILD_STATUS.md` - Current build state, clang-tidy status
- `TIER1_COMPLETE_FINAL.md` - Timeline and roadmap

## Key Documentation

**Primary Requirements & Architecture:**
- `docs/PRD.md` (5000+ lines) - **AUTHORITATIVE** source
  - Section 3.2.11: Department of Labor API
  - Section 3.2.12: 11 GICS Business Sectors
  - Complete data sources, trading strategies
- `docs/architecture/` - 9+ detailed architecture documents

**Coding Standards (CRITICAL - READ FIRST):**
- `docs/CODING_STANDARDS.md` (623 lines) - **MANDATORY READING**
  - 11 comprehensive sections
  - Trailing return syntax (100% required)
  - C++ Core Guidelines (fully enforced)
  - Prefer unordered_map over map
  - Authorship requirements
  - All standards enforced by clang-tidy

**AI Agent System:**
- `ai/CLAUDE.md` - AI assistant context (primary guide)
- `ai/AGENT_BUILD_WORKFLOW.md` (300+ lines) - Complete build workflow
- `ai/README.md` - Orchestration guide
- `ai/PROMPTS/` - 7 specialized agent prompts
- `ai/WORKFLOWS/` - End-to-end workflows

**Build & Development:**
- `BUILD_AND_TEST_INSTRUCTIONS.md` - Build procedures
- `docs/BUILD_WORKFLOW.md` (300+ lines) - Detailed workflow
- `MODULE_CONVERSION_STATUS.md` - Module migration tracking
- `BUILD_STATUS.md` - Current state and issues

## Current Implementation Status (Updated 2025-11-08)

**Tier 1 Foundation Complete:**
- ✅ 25 C++23 modules (10,000+ lines) with trailing return syntax
- ✅ 6 fluent APIs: OptionBuilder, CorrelationAnalyzer, RiskAssessor, SchwabQuery, TaxCalculatorBuilder, BacktestRunner
- ✅ Tax-aware profitability: +14.88% after-tax validated
- ✅ Build system: CMake + Ninja with Clang 21
- ✅ clang-tidy: 0 errors with 11 comprehensive check categories
- ✅ Pre-commit hooks: 6 automated quality checks
- ✅ GitHub Actions: CodeQL (2x daily) + comprehensive validation
- ✅ 30 duplicate files deleted, clean C++23 architecture
- ✅ Authorship standards enforced on all files

**Tier 1 Extension (Weeks 5-6) - IN PROGRESS:**
- Employment data integration (BLS API ready)
- 11 GICS sectors implementation
- Python bindings with pybind11
- Code quality: 100% C++ Core Guidelines compliance
- Module consolidation (merge .cpp into .cppm)

**Next Phase: Tier 2 (Weeks 7-10)**
- Complete Iron Condor strategy
- Real-time Schwab API integration
- ML-based sentiment analysis
- Paper trading validation

**Project Goals (Tier 1 POC):**
- Prove daily profitability ($150+/day) with $30k Schwab account
- 80% winning days, >60% win rate, Sharpe ratio >2.0
- Max drawdown <15%, 3+ months consistent performance

## Options Trading Focus

Primary strategy: **Iron Condor** for range-bound, high-IV markets:
- Entry: IV Rank > 50, short strikes at ±1σ, long at ±1.5σ
- Exit: 50% profit target, 2x stop loss, 7 DTE time decay
- Expected: 65-75% win rate, 15-30% ROC per trade

When implementing trading strategies, always include explainability mechanisms and risk management parameters.

## Python Bindings with pybind11 (Tagged: PYTHON_BINDINGS)

**Framework Ready:**
- `src/python_bindings/bigbrother_bindings.cpp` - Main bindings file
- pybind11 in ansible playbook and build system
- Tagged with: PYTHON_BINDINGS for easy identification

**Tasks for Week 6 (from TIER1_EXTENSION_CHECKLIST.md):**
1. **DuckDB C++ API Bindings** (CRITICAL)
   - Direct database access from Python with C++ performance
   - Since DuckDB built from source, C++ headers available
   - Create: `src/python_bindings/duckdb_bindings.cpp`
2. **Options Pricing Bindings** - Black-Scholes, Trinomial Tree, Greeks
3. **Correlation Engine Bindings** - Pearson, Spearman, time-lagged
4. **Risk Management Bindings** - Kelly Criterion, Monte Carlo, position sizing
5. **Tax Calculator Bindings** - Tax calculations, wash sales
6. **Performance Target:** 10-100x speedup over pure Python

**When working on Python bindings:**
- Tag all files with: PYTHON_BINDINGS
- Include author: Olumuyiwa Oluwasanmi
- Bypass GIL for C++ calls
- Use zero-copy transfer for NumPy/Pandas where possible
- See Section H in TIER1_EXTENSION_CHECKLIST.md for complete tasks

## Container Performance Standard (CRITICAL)

**Rule:** Prefer `std::unordered_map` over `std::map` (enforced by CI/CD)

**Rationale:**
- O(1) average lookup vs O(log n)
- Faster for real-time trading
- More flexible (no operator< required)

**When to use std::map:**
- Need sorted iteration (time-ordered, price-ordered)
- Range queries required
- **Must justify with comment:** `// JUSTIFIED: need sorted order`

**Custom hash for complex keys:**
```cpp
struct PairHash {
    auto operator()(std::pair<T, U> const& p) const noexcept -> size_t {
        return std::hash<T>{}(p.first) ^ (std::hash<U>{}(p.second) << 1);
    }
};
std::unordered_map<std::pair<string, string>, double, PairHash> data;
```

## Module Consolidation Tasks (Section J)

**Current State:** 14 .cpp files, 25 .cppm files
**Target:** ~5 .cpp files, 25-30 .cppm files

**Small files to merge into .cppm (<250 lines):**
- logger.cpp → logger.cppm
- position_sizer.cpp → risk_management.cppm
- stop_loss.cpp → risk_management.cppm
- strategy_manager.cpp → strategies.cppm
- strategy_volatility_arb.cpp → strategies.cppm

**Process for each:**
1. Add `module :private;` section to .cppm
2. Copy implementation from .cpp
3. Update CMakeLists.txt
4. Build and test
5. Delete .cpp file
6. See TIER1_EXTENSION_CHECKLIST.md Section J for details

## Common Pitfalls

1. **Don't use PostgreSQL** - DuckDB only for Tier 1
2. **Module syntax matters** - Use trailing `export module` declarations
3. **OpenMP linkage** - Ensure `OpenMP::OpenMP_CXX` in CMake targets
4. **Library paths** - Set `LD_LIBRARY_PATH` for Clang 21 libraries
5. **Python execution** - Always use `uv run` prefix
6. **Trailing return syntax REQUIRED** - All functions: `auto func() -> ReturnType`
7. **Module imports** - Use `import bigbrother.module.name;` not `#include`
8. **Prefer unordered_map** - Use over map unless ordering required
9. **Authorship MANDATORY** - ALL files: Author: Olumuyiwa Oluwasanmi
10. **clang-tidy will block** - Fix errors before building/committing
11. **External libraries excluded** - Don't check python_bindings, external, third_party
12. **Read the checklist** - TIER1_EXTENSION_CHECKLIST.md has 250+ tasks