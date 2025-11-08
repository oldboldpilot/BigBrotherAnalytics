# BigBrotherAnalytics AI Coding Assistant Instructions

## Project Overview

BigBrotherAnalytics is a high-performance AI-powered trading platform with **microsecond-level latency** requirements. **Tier 1 Foundation is COMPLETE** with 17 production-ready C++23 modules (7,415 lines) and validated profitability.

**CRITICAL AUTHORSHIP STANDARD:**
- **ALL files MUST include:** Author: Olumuyiwa Oluwasanmi
- **Applies to:** Code, configs, docs, scripts, tests, CI/CD files
- **Format:** See Section 11 in docs/CODING_STANDARDS.md
- **Enforcement:** Pre-commit hooks + CI/CD checks

**Current Status (Nov 2025):**
- ✅ C++23 module migration complete - 17 modules with fluent APIs
- ✅ Tax-aware profitability validated: +14.88% after-tax on backtests
- ✅ Build system working with Clang 21 + Ninja
- ⚠️ Minor OpenMP configuration issues being resolved
- 🔄 Ready for production deployment phase

**Three Core Systems:**
1. **Market Intelligence Engine** - Multi-source data processing and impact prediction (Python ML)
2. **Correlation Analysis Tool** - Time-series relationships and leading indicators (C++23/MPI)
3. **Trading Decision Engine** - Options day trading with explainable decisions (C++23/Python)

## Critical Architecture Patterns

### Technology Stack (Tier 1 POC)
- **C++23 Modules** - All core components use `.cppm` files with `export module` syntax
- **DuckDB ONLY** - Zero-setup embedded database (PostgreSQL deferred until profitable)
- **Python 3.13 + uv** - All Python execution via `uv run python script.py` (no venv needed)
- **CMake + Ninja** - Build system with C++23 module support
- **Clang 21** - Required for C++23 modules compilation

### Module Structure Pattern
All 17 modules follow this structure:
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

## Common Pitfalls

1. **Don't use PostgreSQL** - DuckDB only for Tier 1
2. **Module syntax matters** - Use trailing `export module` declarations
3. **OpenMP linkage** - Ensure `OpenMP::OpenMP_CXX` in CMake targets
4. **Library paths** - Set `LD_LIBRARY_PATH` for Clang 21 libraries
5. **Python execution** - Always use `uv run` prefix
6. **Trailing return syntax required** - All functions: `auto func() -> ReturnType`
7. **Module imports** - Use `import bigbrother.module.name;` not `#include`

## Key Documentation

**Primary Requirements & Architecture:**
- `docs/PRD.md` - Complete requirements (5000+ lines) - authoritative source
- `docs/architecture/` - 9+ detailed architecture documents:
  - `database-strategy-analysis.md` - DuckDB rationale
  - `market-intelligence-engine.md` - Data ingestion & NLP system
  - `trading-correlation-analysis-tool.md` - Time-series analysis
  - `intelligent-trading-decision-engine.md` - Trading execution
  - `schwab-api-integration.md` - Broker integration
  - `systems-integration.md` - Component communication
  - `CPP23_MODULE_MIGRATION_PLAN.md` - C++23 module patterns

**AI Agent System:**
- `ai/CLAUDE.md` - AI assistant context and conventions
- `ai/README.md` - Complete orchestration guide (592 lines)
- `ai/PROMPTS/` - 7 specialized agent prompts
- `ai/WORKFLOWS/` - End-to-end development workflows

**Build & Development:**
- `BUILD_AND_TEST_INSTRUCTIONS.md` - Recent build fixes and validation
- `MODULE_CONVERSION_STATUS.md` - C++23 module migration tracking

## Current Implementation Status

**Tier 1 Foundation Complete (Nov 2025):**
- 17 C++23 modules (7,415 lines) with trailing return syntax
- 6 fluent APIs: OptionBuilder, CorrelationAnalyzer, RiskAssessor, SchwabQuery, TaxCalculatorBuilder, BacktestRunner
- Tax-aware profitability: +14.88% after-tax validated
- Build system: CMake + Ninja with Clang 21
- Outstanding: Minor OpenMP configuration mismatches

**Next Phase: Production Deployment**
- Paper trading integration with Schwab API
- Real-time market intelligence data pipeline
- ML model training for impact prediction
- Live correlation analysis

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