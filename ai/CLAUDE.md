# BigBrotherAnalytics - Claude AI Guide

**Project:** High-performance AI-powered trading intelligence platform
**Phase:** Tier 1 POC (Weeks 1-12) - DuckDB-first validation
**Budget:** $30,000 Schwab margin account
**Goal:** Prove daily profitability ($150+/day) before scaling

## Core Architecture

**Three Interconnected Systems:**
1. **Market Intelligence Engine** - Multi-source data ingestion, NLP, impact prediction, graph generation
2. **Correlation Analysis Tool** - Statistical relationships, time-lagged correlations, leading indicators
3. **Trading Decision Engine** - Options day trading (initial focus), explainable decisions, risk management

**Technology Stack (Tier 1 POC):**
- **Languages:** C++23 (core), Python 3.13 (ML), Rust (optional)
- **Database:** DuckDB ONLY (PostgreSQL deferred to Tier 2 after profitability)
- **Parallelization:** MPI, OpenMP, UPC++, GASNet-EX, OpenSHMEM (32+ cores)
- **ML/AI:** PyTorch, Transformers, XGBoost, SHAP
- **C++/Python Integration:** pybind11 for performance-critical code (bypasses GIL)
- **Document Processing:** Maven + OpenJDK 25 + Apache Tika
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

## Current Status (as of 2025-11-06)

**Completed:**
- ✅ Planning phase complete (architecture, PRD, database strategy)
- ✅ DuckDB-first decision made and documented
- ✅ Ansible playbook updated for DuckDB-only Tier 1
- ✅ All documentation updated to reflect DuckDB-first approach

**Next Steps:**
- [ ] Run ansible playbook to set up Tier 1 environment
- [ ] Implement C++23 core components (options pricing, correlation engine)
- [ ] Build Python ML pipeline with DuckDB
- [ ] Validate with 10 years of free historical data

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
- See `docs/CODING_STANDARDS.md` Section 11 for templates
- **NO co-authoring** - Only Olumuyiwa Oluwasanmi as author

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
- 25 C++23 modules implemented
- 100% trailing return syntax
- Zero traditional headers in new code
- Clang 21.1.5 required

When helping with this project:
1. Always check database strategy first - use DuckDB for Tier 1, not PostgreSQL
2. **Read `docs/CPP23_MODULES_GUIDE.md` before writing C++ code**
3. Reference `ai/MANIFEST.md` for current goals and active agents
4. Check `ai/IMPLEMENTATION_PLAN.md` for task status and checkpoints
5. Use workflows in `ai/WORKFLOWS/` for repeatable processes
6. **For complex tasks, use the Orchestrator** (`PROMPTS/orchestrator.md`)
7. Focus on validation speed - POC has $30k at stake
