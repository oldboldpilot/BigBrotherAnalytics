# BigBrotherAnalytics AI Orchestration System

This directory contains the AI agent orchestration system for structured, high-quality development of BigBrotherAnalytics.

---

## Overview

The AI orchestration system coordinates multiple specialized AI agents to handle complex development tasks with consistency, quality, and automation.

> **Latest Update (2025-11-13):** **Phase 5 ACTIVE + SIMD Risk Analytics COMPLETE + ML Price Predictor v3.0 DEPLOYED**. System is 100% production ready with:
> - **✅ ML Price Predictor v3.0:** 60-feature neural network integrated into C++ engine
>   - **Architecture:** [256, 128, 64, 32] with DirectionalLoss (90% direction + 10% MSE)
>   - **Performance:** 56.3% (5-day), 56.6% (20-day) accuracy - **PROFITABLE** (>55% threshold)
>   - **Implementation:** ONNX Runtime with AVX2 SIMD normalization (8x speedup)
>   - **Code:** `price_predictor.cppm` (525 lines), `feature_extractor.cppm` (620 lines)
>   - **Training:** 24,300 samples, 20 symbols, 5 years data, DirectionalLoss function
>   - **Features:** 60 (identification, time, treasury, Greeks, sentiment, price, momentum, volatility, interactions, directionality)
> - **✅ SIMD Risk Analytics (AVX-512/AVX2):** Comprehensive SIMD acceleration for risk management
>   - **Monte Carlo Simulator:** 8M simulations/sec (AVX2), 6-7x speedup over scalar
>     - **Implementation:** 212 lines of SIMD code (vectorized_sum, vectorized_mean_variance, fast_exp_vector)
>     - **Benchmark:** 250K sims in 31.88ms (7.8M sims/sec), 100K sims in 12.53ms (8.0M sims/sec)
>     - **Architecture:** AVX-512 (8 doubles/iter), AVX2 (4 doubles/iter), scalar fallback
>     - **Documentation:** Comprehensive Doxygen-style comments with performance notes
>   - **Correlation Analyzer:** AVX-512/AVX2 Pearson correlation (6-8x speedup)
>     - **Migration:** Replaced MKL with direct intrinsics for better control
>     - **Features:** FMA instructions, horizontal reduction, unaligned loads
> - **✅ SIMD JSON Parsing (simdjson v4.2.1):** 3-32x faster JSON parsing, migrated all hot paths
>   - **Quote parsing:** 32.2x faster (3449ns → 107ns, 120 req/min)
>   - **NewsAPI:** 23.0x faster (8474ns → 369ns, 96 req/day)
>   - **Account data:** 28.4x faster (3383ns → 119ns, 60 req/min)
>   - **Annual savings:** ~6.7B CPU cycles
>   - **Testing:** 23 unit tests, 4 benchmark workloads, thread safety verified
> - **Automatic OAuth Token Refresh:** No manual intervention for 7 days (auto-refresh using Schwab API)
> - **Auto-Start Services:** Single command starts dashboard + trading engine (--start-all flag)
> - **Complete Portability:** One-command deployment via [bootstrap.sh](../scripts/bootstrap.sh) (fresh machine → production in 5-15 min)
> - **Auto-detection:** Compilers, libraries, and paths detected automatically (works on any Unix system)
> - **Tax Tracking:** California married filing jointly, $300K base income (37.1% ST / 28.1% LT, 1.5% trading fees)
> - **Unified Setup/Shutdown:** Morning setup (10-15 sec), evening shutdown (graceful + reports + backup)
> - **Process Management:** Auto-kills duplicate processes, prevents port conflicts
> - **Health Monitoring:** Real-time token validation, system status checks
> - **Security:** API keys removed from git, comprehensive .gitignore patterns
> - **Paper Trading:** $2,000 position limit, 2-3 concurrent, ≥55% win rate target
> - **News Ingestion System:** C++23 sentiment analysis (260 lines), NewsAPI integration (402 lines), 236KB Python bindings
> - **Trading Reporting System:** Daily/weekly reports (1,430+ lines), signal analysis, HTML+JSON output
> - All 20 autonomous agents across Phases 1-4 achieved 100% success rate.

### Orchestration Hierarchy

```
+------------------+
|   Orchestrator   | ← Coordinates all agents
+------------------+
        |
        v
+------------------+
|    PRD Writer    | ← Requirements & specifications
+------------------+
        |
        v
+------------------+
| System Architect | ← Architecture design
+------------------+
        |
        v
+------------------+
|  File Creator    | ← Implementation
+------------------+
        |
        v
+---------------------------+
| Self-Correction (Hooks)   | ← Validation & quality
| Playwright + Schema Guard |
+---------------------------+
```

---

## Trading Reporting System (Phase 5+)

**Status:** ✅ IMPLEMENTED - Production Ready | **Integration:** Comprehensive Signal Analysis

The Trading Reporting System provides automated daily and weekly report generation:

### System Components (4/4 Complete)

**Python Report Generators:**
1. `scripts/reporting/generate_daily_report.py` (750+ lines) - Daily trading analysis
2. `scripts/reporting/generate_weekly_report.py` (680+ lines) - Weekly performance summaries
3. `scripts/reporting/__init__.py` - Package initialization and unified API

**Documentation:**
4. `docs/TRADING_REPORTING_SYSTEM.md` (650+ lines) - Complete architecture and API reference

### Daily Report Features

- **Executive Summary:** Account value, signals (generated/executed/rejected), execution rates
- **Trade Execution Details:** Complete trade list with Greeks (Delta, Theta, Vega) at signal generation
- **Signal Analysis:** Breakdown by status, strategy, and rejection reasons
- **Risk Compliance:** Risk rejections, budget constraints, position sizing
- **Market Conditions:** IV percentile analysis, DTE metrics
- **Output Formats:** JSON (structured data) and HTML (browser-viewable)

### Weekly Report Features

- **Performance Summary:** Execution rates, Sharpe ratio, risk/reward ratios
- **Strategy Comparison:** Per-strategy signals, returns, and acceptance rates
- **Signal Acceptance Rates:** Daily breakdown with rejection reason trends
- **Risk Analysis:** Budget impact modeling, cost distribution analysis
- **Automated Recommendations:** Data-driven suggestions for optimization
- **Output Formats:** JSON and HTML

### Database Integration

- **Data Source:** `trading_signals` table (auto-detected)
- **Views:** 10+ analytical views for signal tracking
- **Zero Configuration:** Works automatically with existing database
- **Performance:** DuckDB queries complete in < 1 second

### Quick Reference

```bash
# Generate today's daily report
python scripts/reporting/generate_daily_report.py
# Output: reports/daily_report_YYYYMMDD.json & .html

# Generate this week's report
python scripts/reporting/generate_weekly_report.py
# Output: reports/weekly_report_YYYYMMDD_to_YYYYMMDD.json & .html

# Generate last week's report
python scripts/reporting/generate_weekly_report.py 1
```

**Documentation:** See `docs/TRADING_REPORTING_SYSTEM.md` for complete architecture, database schema, and API reference.

---

## News Ingestion System (Phase 5+)

**Status:** ✅ IMPLEMENTED - Production Ready | **Integration:** 8/8 checks passing (100%)

The News Ingestion System adds real-time financial news tracking with sentiment analysis:

### System Components (13/13 Complete)

**C++ Core Modules:**
1. `src/market_intelligence/sentiment_analyzer.cppm` (260 lines) - Keyword-based sentiment analysis
2. `src/market_intelligence/news_ingestion.cppm` (402 lines) - NewsAPI integration with rate limiting
3. `src/python_bindings/news_bindings.cpp` (110 lines) - pybind11 interface

**Python Scripts:**
4. `scripts/monitoring/setup_news_database.py` - Database initialization
5. `scripts/data_collection/news_ingestion.py` - Main ingestion script (320 lines)

**Documentation:**
6. `docs/NEWS_INGESTION_SYSTEM.md` - Complete architecture (620 lines)
7. `docs/NEWS_INGESTION_QUICKSTART.md` - Setup guide (450 lines)
8. `docs/NEWS_INGESTION_DELIVERY_SUMMARY.md` - Implementation summary
9. `docs/NEWS_CLANG_TIDY_REPORT.md` - Validation report

**Dashboard Integration:**
10. `dashboard/app.py` - News feed view (+200 lines)

**Build System:**
11. `CMakeLists.txt` - Updated (lines 293, 333-334) for C++23 modules
12. Build output: `news_ingestion_py.cpython-314-x86_64-linux-gnu.so` (236KB)

**Database Schema:**
13. `news_articles` table with sentiment metrics and indexes

### Technology Stack Updates

**Added Technologies:**
- **CMake + Ninja:** Required for C++23 module compilation
- **clang-tidy:** C++ Core Guidelines enforcement (0 errors, 36 acceptable warnings)
- **libcurl:** HTTP client for NewsAPI requests
- **nlohmann/json:** JSON parsing in C++
- **pybind11:** Python bindings for C++ modules

### Key Features

- **Sentiment Analysis:** 60+ positive/negative keywords, negation handling, intensifiers
- **NewsAPI Integration:** Rate limiting (100 requests/day), automatic deduplication
- **Error Handling:** Direct `std::unexpected(Error::make(code, msg))` pattern (no circuit breaker)
- **Python-Delegated Storage:** Database writes handled by Python layer for simplicity
- **Dashboard Visualization:** News feed with filtering, charts, sentiment color coding

### Quick Reference

```bash
# Build C++ modules
cmake -G Ninja -B build
ninja -C build market_intelligence

# Setup database
uv run python scripts/monitoring/setup_news_database.py

# Fetch news
uv run python scripts/data_collection/news_ingestion.py

# View dashboard (News Feed tab)
uv run streamlit run dashboard/app.py
```

**Documentation:** See `docs/NEWS_INGESTION_SYSTEM.md` for complete architecture and implementation details.

---

## Trading Platform Architecture (Phase 5+)

**Status:** IMPLEMENTED - Production Ready | **Location:** `src/core/trading/`

### Three-Layer Loosely Coupled Design

The trading system uses Dependency Inversion Principle (SOLID) for multi-platform support:

**1. Platform-Agnostic Types** (`order_types.cppm` - 175 lines)
- Common data structures shared across all platforms
- Types: `Position`, `Order`, `OrderSide`, `OrderType`, `OrderStatus`
- Safety flags: `is_bot_managed`, `managed_by`, `bot_strategy`

**2. Abstract Interface** (`platform_interface.cppm` - 142 lines)
- `TradingPlatformInterface` - Pure virtual base class
- Contract: `submitOrder()`, `cancelOrder()`, `modifyOrder()`, `getOrders()`, `getPositions()`
- High-level code depends on abstraction, not concrete implementations

**3. Platform-Agnostic Business Logic** (`orders_manager.cppm` - 600+ lines)
- `OrdersManager` depends ONLY on `TradingPlatformInterface`
- Zero coupling to platform-specific code
- Dependency injection: `OrdersManager(db_path, std::unique_ptr<TradingPlatformInterface> platform)`

**4. Platform-Specific Adapters** (e.g., `schwab_order_executor.cppm` - 382 lines)
- Implements `TradingPlatformInterface` for specific brokers
- Adapter pattern: converts platform types ↔ common types
- Injected at runtime into `OrdersManager`

### Key Benefits

- **Multi-Platform:** Add IBKR, TD Ameritrade, Alpaca without touching `OrdersManager`
- **Testability:** Easy to mock platform implementations
- **Maintainability:** Clean separation of concerns
- **Scalability:** Platform code isolated in adapters
- **Type Safety:** C++23 module compile-time verification

### Build Configuration

```cmake
# New library: trading_core
add_library(trading_core SHARED)
target_sources(trading_core
    PUBLIC FILE_SET CXX_MODULES FILES
        src/core/trading/order_types.cppm
        src/core/trading/platform_interface.cppm
        src/core/trading/orders_manager.cppm
)

# Platform implementations link trading_core
target_link_libraries(schwab_api PUBLIC trading_core)
```

### Testing

- Regression suite: `scripts/test_loose_coupling_architecture.sh` (379 lines)
- 12 tests, 32 assertions, 100% passing
- Validates: dependency inversion, type conversion, CMake config

### Documentation

- Architecture guide: `docs/TRADING_PLATFORM_ARCHITECTURE.md` (590 lines)
- Step-by-step guide for adding new trading platforms

---

## Directory Structure

```
ai/
├── README.md                    # This file
├── CLAUDE.md                    # Always-loaded guide for AI assistants
├── MANIFEST.md                  # Project goals and active agents
├── IMPLEMENTATION_PLAN.md       # Detailed task breakdown and checkpoints
├── PROMPTS/                     # Agent-specific prompts
│   ├── orchestrator.md          # Coordinates multi-agent workflows
│   ├── prd_writer.md            # Creates/updates requirements docs
│   ├── architecture_design.md   # Designs system architecture
│   ├── file_creator.md          # Generates implementation code
│   ├── self_correction.md       # Validates and auto-fixes code
│   ├── code_review.md           # Reviews code quality
│   └── debugging.md             # Debugs issues systematically
└── WORKFLOWS/                   # End-to-end workflow guides
    ├── feature_implementation.md  # Implement new features
    └── bug_fix.md                 # Fix bugs systematically
```

---

## Agent Descriptions

### 1. Orchestrator (`PROMPTS/orchestrator.md`)
**Purpose:** Coordinate complex multi-agent workflows

**When to Use:**
- Implementing complex features requiring multiple agents
- Large refactoring efforts
- System-wide changes

**Capabilities:**
- Task decomposition
- Agent coordination
- Progress tracking
- Error handling and recovery

**Example:**
```
I need to implement the Options Pricing Engine end-to-end.
Please orchestrate the implementation with all necessary agents.
```

---

### 2. PRD Writer (`PROMPTS/prd_writer.md`)
**Purpose:** Create and maintain Product Requirements Documents

**When to Use:**
- New feature requests
- Requirement changes
- Clarifying scope
- Documenting decisions

**Capabilities:**
- Captures complete requirements
- Defines success metrics
- Documents constraints
- Maintains changelog

**Example:**
```
Update the PRD to include real-time options pricing requirements
with performance targets.
```

---

### 3. System Architect (`PROMPTS/architecture_design.md`)
**Purpose:** Design system architecture and component interactions

**When to Use:**
- New component design
- System integration
- Performance optimization
- Technology decisions

**Capabilities:**
- Component architecture design
- API design (C++ and Python)
- Database schema design
- Performance analysis
- Risk assessment

**Example:**
```
Design the architecture for the Correlation Engine using MPI and OpenMP.
Target: 1000x1000 correlation matrix in < 10 seconds.
```

---

### 4. File Creator (`PROMPTS/file_creator.md`)
**Purpose:** Generate production-quality implementation code

**When to Use:**
- Implementing designs
- Creating new components
- Generating tests
- Boilerplate generation

**Capabilities:**
- C++23 code generation
- Python 3.14+ code generation
- Unit test generation
- CMake/build configuration
- Python bindings (pybind11)

**Example:**
```
Generate the Options Pricing Engine implementation based on
docs/architecture/options-pricing-engine.md
```

---

### 5. Self-Correction (`PROMPTS/self_correction.md`)
**Purpose:** Automated validation, testing, and quality assurance

**When to Use:**
- After code generation
- Before commits
- Pre-push validation
- CI/CD pipeline

**Capabilities:**
- Static analysis (clang-tidy, ruff, mypy)
- Schema validation (DuckDB)
- Unit/integration/performance testing
- Auto-fixing (formatting, linting)
- Security scanning

**Example:**
```
Run self-correction validation on the Options Pricing Engine
implementation before committing.
```

---

### 6. Code Reviewer (`PROMPTS/code_review.md`)
**Purpose:** Review code for quality, performance, and correctness

**When to Use:**
- Post-implementation review
- Pre-commit checks
- Performance audits
- Security reviews

**Capabilities:**
- C++23 best practices review
- Python code quality review
- Financial calculation verification
- Performance analysis
- Security checks

**Example:**
```
Review src/cpp/options/trinomial_tree.cpp for performance
and correctness.
```

---

### 7. Debugger (`PROMPTS/debugging.md`)
**Purpose:** Systematic debugging and issue resolution

**When to Use:**
- Bug reports
- Performance issues
- Crashes
- Test failures

**Capabilities:**
- Root cause analysis
- Memory leak detection
- Performance profiling
- Fix implementation
- Regression test generation

**Example:**
```
Debug the segfault in correlation engine when processing
1000+ symbols with MPI.
```

---

## Workflows

### Feature Implementation (`WORKFLOWS/feature_implementation.md`)

Complete workflow for implementing new features:
1. Orchestrator coordinates agents
2. PRD Writer updates requirements (if needed)
3. System Architect designs architecture
4. File Creator generates implementation
5. Self-Correction validates code
6. Commit to git and push

**Estimated Time:** 1-4 hours

---

### Bug Fix (`WORKFLOWS/bug_fix.md`)

Systematic workflow for fixing bugs:
1. Orchestrator analyzes bug report
2. Debugger identifies root cause
3. File Creator implements fix
4. Self-Correction validates fix
5. Commit with regression test

**Estimated Time:** 30 minutes to 4 hours

---

## Usage Patterns

### Simple Task (Single Agent)

For straightforward tasks, invoke a single agent directly:

```
Please review src/python/data_ingestion/yahoo_finance.py
using the code review prompt. Focus on error handling.
```

### Complex Task (Orchestrated)

For complex tasks, let Orchestrator coordinate:

```
I need to implement the complete Market Intelligence Engine
with NLP, sentiment analysis, and impact prediction.

Please orchestrate the full implementation using:
- PRD Writer (update requirements)
- System Architect (design architecture)
- File Creator (generate code)
- Self-Correction (validate)
```

---

## Integration with Development

### Local Development

```bash
# 1. Implement feature using AI agents
# (follow feature_implementation.md workflow)

# 2. Self-correction runs automatically via git hooks
# (installed by scripts/install_hooks.py)

# 3. Commit and push
git commit -m "feat: Add new feature"
git push origin master
```

### CI/CD Integration

The Self-Correction system integrates with GitHub Actions:

```yaml
# .github/workflows/validate.yml
# - Runs on every push
# - Blocks PR merge if validation fails
# - Reports results in PR comments
```

---

## Best Practices

### 1. Always Start with Orchestrator for Complex Tasks
Let the Orchestrator decide which agents to invoke and in what order.

### 2. Keep Agents Focused
Each agent has a specific responsibility. Don't mix concerns.

### 3. Validate Early and Often
Run Self-Correction after every code generation, not just before commit.

### 4. Document Decisions
PRD Writer and System Architect should document all architectural decisions.

### 5. Test Thoroughly
File Creator should generate comprehensive tests, not just happy paths.

---

## Configuration

### Agent Settings

Agents can be configured via environment variables or config files:

```bash
# Example: .env
AI_ORCHESTRATOR_MODE=sequential  # or parallel
AI_AUTO_FIX=true                 # Auto-fix simple issues
AI_VALIDATION_LEVEL=strict       # or relaxed
```

### Git Hooks

Install self-correction hooks:

```bash
python scripts/install_hooks.py

# Hooks will run on:
# - pre-commit: Format, lint, unit tests
# - pre-push: Full validation, integration tests
```

---

## Troubleshooting

### Agent Not Found

**Problem:** "Agent X not available"

**Solution:** Check that `PROMPTS/[agent_name].md` exists

---

### Validation Failing

**Problem:** Self-Correction reports failures

**Solution:**
1. Review validation report
2. Check which specific checks failed
3. Fix issues manually or route back to appropriate agent
4. Re-run validation

---

### Orchestration Hangs

**Problem:** Orchestrator stuck or not progressing

**Solution:**
1. Check for circular dependencies between agents
2. Verify all required inputs are available
3. Check agent-specific error logs

---

## Metrics and Monitoring

Track AI orchestration effectiveness:

```python
metrics = {
    'features_implemented': 0,
    'bugs_fixed': 0,
    'auto_fixes_applied': 0,
    'validation_success_rate': 0.0,
    'average_time_per_feature_hours': 0.0,
}
```

---

## Extending the System

### Adding New Agents

1. Create `PROMPTS/new_agent.md` with agent specification
2. Update `PROMPTS/orchestrator.md` with new agent details
3. Create workflow using the new agent (optional)
4. Update this README

### Adding New Workflows

1. Create `WORKFLOWS/new_workflow.md`
2. Document step-by-step process
3. Include example execution
4. Update this README

---

## References

- **PRD:** `docs/PRD.md`
- **Architecture:** `docs/architecture/*`
- **Implementation Plan:** `ai/IMPLEMENTATION_PLAN.md`
- **Manifest:** `ai/MANIFEST.md`

---

## Getting Started After Cloning Repository

When you clone this repository to a new machine, the AI orchestration system is immediately available:

### 1. Verify AI System Files
```bash
cd BigBrotherAnalytics
ls -la ai/

# You should see:
# - README.md (this file)
# - CLAUDE.md (AI assistant guide)
# - MANIFEST.md (project goals)
# - IMPLEMENTATION_PLAN.md (task breakdown)
# - PROMPTS/ (7 agent prompts)
# - WORKFLOWS/ (2 workflow guides)
```

### 2. Using AI Agents on New Machine

The AI agents are **self-contained markdown files** that work immediately after cloning:

**For Simple Tasks:**
```
When working with an AI assistant (like Claude Code), reference the prompts:

"Please implement the correlation engine using the File Creator prompt
from ai/PROMPTS/file_creator.md"
```

**For Complex Tasks:**
```
"Please orchestrate the implementation of the Options Pricing Engine
using the ai/PROMPTS/orchestrator.md workflow"
```

**Quick Reference:**
```
Always loaded: ai/CLAUDE.md (AI assistant will read this automatically)
Task planning: ai/IMPLEMENTATION_PLAN.md
Agent prompts: ai/PROMPTS/*.md
Workflows: ai/WORKFLOWS/*.md
```

### 3. Portability Checklist

✅ **Already Portable (No Setup Needed):**
- All AI agent prompts (markdown files)
- Architecture documentation
- PRD with complete requirements
- Workflow guides
- Implementation plans

⏸️ **Requires Setup on New Machine:**
- Development environment (run Ansible playbook)
- C++23 compiler (GCC 15)
- Python 3.14+
- DuckDB
- Build tools

**Setup Command:**
```bash
cd BigBrotherAnalytics
ansible-playbook playbooks/complete-tier1-setup.yml
```

### 4. Verifying AI System After Clone

```bash
# Verify all AI files present
test -d ai/PROMPTS && test -d ai/WORKFLOWS && echo "✅ AI system complete"

# Count documentation
find ai/ -name "*.md" | wc -l  # Should be 13 files

# Total lines
wc -l ai/**/*.md ai/*.md | tail -1  # Should be ~5,200 lines
```

### 5. Using on Different Machines

**Quick Deployment (Recommended):**
```bash
git clone https://github.com/oldboldpilot/BigBrotherAnalytics.git
cd BigBrotherAnalytics

# One-command deployment (5-15 minutes)
./scripts/bootstrap.sh

# Complete! System is production ready with:
# - Clang 21 toolchain
# - C++23 project compiled
# - Python environment configured
# - Database initialized
# - Tax rates configured (California, married filing jointly, $300K)
```

**Manual Deployment:**
```bash
git clone https://github.com/oldboldpilot/BigBrotherAnalytics.git
cd BigBrotherAnalytics

# 1. AI agents work immediately - no installation needed
# Just reference ai/PROMPTS/*.md when working with AI assistants

# 2. Install development environment:
ansible-playbook playbooks/complete-tier1-setup.yml

# 3. Build C++ project (auto-detects compilers and libraries):
mkdir build && cd build
cmake -G Ninja ..
ninja

# 4. Setup Python environment:
cd .. && uv sync

# 5. Initialize database:
uv run python scripts/monitoring/setup_tax_database.py
uv run python scripts/monitoring/update_tax_rates_california.py
```

**Team Collaboration:**
- All team members have access to same AI agent prompts
- Consistent code generation standards across team
- Shared workflows for features and bug fixes
- Version controlled AI instructions (in git)
- Portable build system (works on any Unix system after `git clone`)

---

## Transferability Guarantees

✅ **All AI agents are:**
- Pure markdown files (no external dependencies)
- Version controlled in git
- Self-contained with complete examples
- Platform independent
- Ready to use immediately after `git clone`

✅ **All requirements documented:**
- PRD has complete C++23 standards
- Architecture docs have fluent API patterns
- File Creator has comprehensive examples
- Standards are enforceable with static analysis tools

✅ **Works across:**
- Linux, macOS, Windows (WSL2)
- Different hardware (laptop to 64-core server)
- Different AI assistants (any can read the prompts)
- Team environments (consistent standards for all developers)

---

## Version History

**v1.0.0 (2025-11-06):**
- Initial AI orchestration system
- 7 specialized agents
- 2 complete workflows
- Self-correction with git hooks
- Complete C++23 modern standards
- Fluent API requirements
- C++ Core Guidelines compliance
- STL-first approach
- **Fully portable and transferrable**

---

**For questions or issues with the AI orchestration system, consult this README or the specific agent/workflow documentation.**

**The AI orchestration system is now ready to use on ANY machine after a simple `git clone`.**
