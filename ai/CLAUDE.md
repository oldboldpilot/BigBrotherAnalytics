# BigBrotherAnalytics - Claude AI Guide

**Project:** High-performance AI-powered trading intelligence platform
**Phase:** Phase 5 - Paper Trading Validation (Days 0-21)
**Status:** 100% Production Ready
**Budget:** $100 position limit (paper trading validation)
**Goal:** ≥55% win rate (profitable after 37.1% tax + 3% fees)

## Core Architecture

**Three Interconnected Systems:**
1. **Market Intelligence Engine** - Multi-source data ingestion, NLP, impact prediction, graph generation
   - **News Ingestion System:** NewsAPI integration with C++23 sentiment analysis (260 lines)
   - **Employment Signals:** BLS data integration with sector rotation (1,064+ records)
   - **Sentiment Analysis:** Keyword-based scoring (-1.0 to 1.0, 60+ keywords each direction)
2. **Correlation Analysis Tool** - Statistical relationships, time-lagged correlations, leading indicators
3. **Trading Decision Engine** - Options day trading (initial focus), explainable decisions, risk management

**Technology Stack (Tier 1 POC):**
- **Languages:** C++23 (core), Python 3.13 (ML), Rust (optional)
- **Database:** DuckDB ONLY (PostgreSQL deferred to Tier 2 after profitability)
- **Parallelization:** MPI, OpenMP, UPC++, GASNet-EX, OpenSHMEM (32+ cores)
- **ML/AI:** PyTorch, Transformers, XGBoost, SHAP
- **C++/Python Integration:** pybind11 for performance-critical code (bypasses GIL)
- **Document Processing:** Maven + OpenJDK 25 + Apache Tika
- **Build System:** CMake 4.1.2+ with Ninja generator (required for C++23 modules)
- **Code Quality:** clang-tidy (C++ Core Guidelines enforcement)
- **Package Manager:** uv (10-100x faster than pip, project-based, no venv needed)
- **Execution:** All Python code runs with `uv run python script.py`

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
- **News Ingestion:** `docs/NEWS_INGESTION_SYSTEM.md` - Complete architecture and implementation (620 lines)
- **News Quick Start:** `docs/NEWS_INGESTION_QUICKSTART.md` - Setup guide with actual build output (450 lines)
- **News Delivery:** `docs/NEWS_INGESTION_DELIVERY_SUMMARY.md` - Implementation summary and status

## Phase 5: Paper Trading Validation (ACTIVE)

**Timeline:** Days 0-21 | **Started:** November 10, 2025

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
# - OAuth token refresh (no manual intervention for 7 days)
# - Kills duplicate processes (prevents port conflicts)
# - Starts dashboard (http://localhost:8501)
# - Starts trading engine (background)
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
- ✅ **Unified setup script** (replaces 10+ commands, automatic OAuth refresh)
- ✅ **Auto-start services** (--start-all flag starts dashboard + trading engine)
- ✅ **Tax tracking** (married joint CA, YTD accumulation, 1.5% accurate fees)
- ✅ **End-of-day automation** (reports, tax calc, backup)
- ✅ **Paper trading config** ($100 position limit)
- ✅ **Manual position protection** (100% verified)
- ✅ **All tests passing** (87/87, 100% success rate)
- ✅ **Complete documentation**
- ✅ **Error handling & circuit breakers**
- ✅ **Performance optimization** (4.09x speedup)
- ✅ **Monitoring & alerts** (9 health checks, 27 types, token validation)
- ✅ **Health monitoring** (real-time token validation, system status)
- ✅ **News Ingestion System** (8/8 checks passing, 236KB Python bindings, sentiment analysis)

### Success Criteria
- **Win Rate:** ≥55% (profitable after 37.1% tax + 1.5% fees)
- **Risk Limits:** $100 position, $100 daily loss, 2-3 concurrent
- **Tax Accuracy:** Real-time YTD cumulative tracking
- **Zero Manual Position Violations:** 100% protection
- **Token Management:** 100% automatic refresh (no manual intervention for 7 days)

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

### DuckDB Bridge Library (MANDATORY for C++23 Modules)

⚠️ **CRITICAL**: C++23 modules CANNOT include `<duckdb.hpp>` directly due to incomplete types (`duckdb::QueryNode`).

**Solution**: Use the DuckDB bridge library at `src/schwab_api/duckdb_bridge.{hpp,cpp}`

```cpp
// In global module fragment
module;
#include "schwab_api/duckdb_bridge.hpp"  // ✅ Use bridge
// #include <duckdb.hpp>                  // ❌ NEVER in modules

export module my_module;

using namespace bigbrother::duckdb_bridge;

class MyClass {
    std::unique_ptr<DatabaseHandle> db_;
    std::unique_ptr<ConnectionHandle> conn_;

    auto connect() -> void {
        db_ = openDatabase("data/bigbrother.duckdb");
        conn_ = createConnection(*db_);
        executeQuery(*conn_, "CREATE TABLE IF NOT EXISTS ...");
    }
};
```

**Bridge API**: `openDatabase()`, `createConnection()`, `executeQuery()`, `prepareStatement()`, `bindString/Int/Double()`, `executeStatement()`

**Why**: DuckDB C API avoids incomplete type instantiation errors. Similar pattern to OpenMP/MPI.

**See**: `AGENT_CODING_GUIDE.md` for complete bridge API reference and examples.

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
