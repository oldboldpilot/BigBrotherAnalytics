# Workflow: Feature Implementation

This workflow guides implementing a new feature using the AI agent orchestration system.

---

## Overview

```
User Request → Orchestrator → PRD Writer → System Architect → File Creator → Self-Correction → Commit
```

---

## Step-by-Step Process

### Step 1: Initiate Orchestration

**User Action:**
```
I need to implement [Feature Name] for BigBrotherAnalytics.

Requirements:
- [Requirement 1]
- [Requirement 2]
- Performance target: [latency/throughput]
- Tier: [Tier 1 POC or Tier 2]

Please orchestrate the implementation.
```

**AI Response:**
```
Orchestrator activating for feature: [Feature Name]

Analysis:
- Complexity: [HIGH/MEDIUM/LOW]
- Required agents: PRD Writer, System Architect, File Creator, Self-Correction
- Execution mode: Sequential
- Estimated time: [duration]

Proceeding with agent coordination...
```

---

### Step 2: PRD Update (if needed)

**Orchestrator Action:**
```
Invoking PRD Writer to update docs/PRD.md with new feature specifications...
```

**PRD Writer Output:**
- Updates PRD Section X with feature requirements
- Adds functional requirements (FR-X.1, FR-X.2, ...)
- Adds non-functional requirements (performance, reliability)
- Defines success metrics
- Updates changelog

**Validation:**
- [ ] Feature requirements are clear and testable
- [ ] Success metrics defined
- [ ] Constraints documented
- [ ] Priority assigned

---

### Step 3: Architecture Design

**Orchestrator Action:**
```
Invoking System Architect to design feature architecture...
Input: PRD Section X, existing architecture docs
```

**System Architect Output:**
- Creates `docs/architecture/[feature-name].md` (or updates existing)
- Includes:
  - Component diagram
  - Data flow
  - API design (C++ and Python)
  - Database schema (DuckDB)
  - Performance considerations
  - Integration points
  - Testing strategy

**Validation:**
- [ ] Architecture aligns with existing systems
- [ ] Performance targets achievable
- [ ] DuckDB-first for Tier 1
- [ ] Clear integration points

---

### Step 4: Implementation

**Orchestrator Action:**
```
Invoking File Creator to generate implementation files...
Input: Architecture document, existing codebase
```

**File Creator Output:**

**C++ Components (if applicable):**
```
src/cpp/[component]/
├── [component].hpp          # Interface
├── [component].cpp          # Implementation
├── CMakeLists.txt           # Build config
└── bindings.cpp             # Python bindings (pybind11)
```

**Python Components:**
```
src/python/[module]/
├── __init__.py
├── [feature].py             # Main implementation
└── config.py                # Configuration
```

**Tests:**
```
tests/cpp/
└── test_[component].cpp     # C++ unit tests

tests/python/
└── test_[feature].py        # Python unit tests
```

**Supporting Files:**
- Scripts (if needed)
- Configuration files
- Documentation (README, usage examples)

**Validation:**
- [ ] Code follows project structure
- [ ] All functions have docstrings/comments
- [ ] Error handling implemented
- [ ] Tests generated

---

### Step 5: Self-Correction

**Orchestrator Action:**
```
Invoking Self-Correction to validate implementation...
```

**Self-Correction Checks:**

1. **Static Analysis**
   ```bash
   # Python
   ruff check src/python/
   mypy src/python/ --strict
   black --check src/python/

   # C++
   clang-tidy src/cpp/**/*.cpp
   cppcheck --enable=all src/cpp/
   ```

2. **Schema Validation**
   ```python
   # Verify DuckDB schema matches spec
   guard = SchemaGuard(db_path, schema_spec)
   errors = guard.validate()
   ```

3. **Unit Tests**
   ```bash
   pytest tests/python/ --cov=src/python --cov-fail-under=90
   ctest --output-on-failure  # C++ tests
   ```

4. **Integration Tests**
   ```bash
   pytest tests/ --run-integration
   ```

5. **Performance Tests**
   ```bash
   pytest tests/ --benchmark
   ```

**Auto-Fixes Applied:**
- Code formatting (black, clang-format)
- Import sorting (isort)
- Simple linting issues (ruff --fix)

**Issues Found:**
- [List of issues that couldn't be auto-fixed]

**Validation:**
- [ ] All tests pass
- [ ] Coverage >= 90%
- [ ] Performance targets met
- [ ] No security issues

---

### Step 6: Commit

**Orchestrator Action:**
```
All validation passed. Committing changes...
```

**Git Commit:**
```bash
git add docs/PRD.md
git add docs/architecture/[feature-name].md
git add src/cpp/[component]/
git add src/python/[module]/
git add tests/

git commit -m "feat([component]): Add [Feature Name]

Implements [brief description of feature].

Features:
- [Feature 1]
- [Feature 2]
- [Feature 3]

Performance:
- [Latency/throughput achieved]

Testing:
- Unit tests: [X passed]
- Integration tests: [Y passed]
- Coverage: [Z]%

References: docs/architecture/[feature-name].md

Author: Olumuyiwa Oluwasanmi"
```

**Push to GitHub:**
```bash
git push origin master
```

---

## Example: Implementing Options Pricing Engine

### Step 1: User Request
```
I need to implement the Options Pricing Engine.

Requirements:
- Black-Scholes-Merton for European options
- Trinomial Tree for American options
- Calculate all Greeks (Delta, Gamma, Theta, Vega, Rho)
- Performance: < 1ms per option (p99)
- Tier: Tier 1 POC

Please orchestrate the implementation.
```

### Step 2: PRD Update
PRD Writer updates `docs/PRD.md` Section 4.3 with detailed options pricing requirements.

### Step 3: Architecture Design
System Architect creates `docs/architecture/options-pricing-engine.md`:
```markdown
## Options Pricing Engine

### Architecture
[Market Data] → [Pricing Engine] → [Greeks Calculator] → [DuckDB]
                       ↓
                [Python Bindings] → [Trading Decision Engine]

### API Design
```cpp
namespace bigbrother::options {
    std::expected<OptionPrice, Error> price_option(
        const OptionContract& contract,
        const MarketData& market,
        PricingModel model = PricingModel::Trinomial
    );
}
```

### Performance Target
- Pricing: < 1ms (p99)
- Greeks: < 0.5ms (p99)
```

### Step 4: Implementation
File Creator generates:
```
src/cpp/options/
├── black_scholes.hpp
├── black_scholes.cpp
├── trinomial_tree.hpp
├── trinomial_tree.cpp
├── greeks.hpp
├── greeks.cpp
├── bindings.cpp
└── CMakeLists.txt

tests/cpp/
└── test_options.cpp
```

### Step 5: Self-Correction
```
✅ Static analysis: PASS
✅ Unit tests: 48/48 passed
✅ Coverage: 96.2%
✅ Performance: p99 = 0.87ms (target: 1.0ms)
✅ Security: No issues

Ready to commit!
```

### Step 6: Commit
```bash
feat(options): Add Options Pricing Engine with BSM and Trinomial models

Implements high-performance options pricing with microsecond latency.

Features:
- Black-Scholes-Merton model for European options
- Trinomial Tree model for American options
- Complete Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Python bindings via pybind11

Performance:
- Pricing latency: p50=0.42ms, p99=0.87ms (target: 1.0ms)
- Greeks calculation: p99=0.38ms

Testing:
- Unit tests: 48 passed
- Coverage: 96.2%
- Performance tests: All targets met

References: docs/architecture/options-pricing-engine.md
```

---

## Troubleshooting

### Issue: Validation Fails

**Scenario:** Self-Correction finds issues that can't be auto-fixed.

**Action:**
1. Review the validation report
2. Identify which agent's output needs fixing
3. Route back to that agent with corrections
4. Re-run validation

**Example:**
```
Self-Correction Report:
❌ Performance test failed: p99 latency = 1.2ms (target: 1.0ms)

Orchestrator: Routing back to System Architect to optimize algorithm...
System Architect: Recommending vectorization with Intel MKL...
File Creator: Updating implementation with SIMD optimizations...
Self-Correction: ✅ p99 latency = 0.85ms - PASS
```

### Issue: Architecture Conflicts with Existing System

**Scenario:** System Architect's design conflicts with existing architecture.

**Action:**
1. Orchestrator detects conflict
2. Consult existing architecture docs
3. Re-design to align with existing patterns
4. Validate consistency

---

## Success Criteria

Feature implementation is complete when:
- [ ] PRD updated with requirements
- [ ] Architecture document created/updated
- [ ] Implementation code generated
- [ ] Tests pass (unit, integration, performance)
- [ ] Coverage >= 90%
- [ ] Self-correction passed
- [ ] Committed to git
- [ ] Pushed to GitHub

---

**Estimated Time:** 1-4 hours depending on feature complexity

**Key Principle:** The orchestration system ensures consistency, quality, and completeness for every feature implementation.
