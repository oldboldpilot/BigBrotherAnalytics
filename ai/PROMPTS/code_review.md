# Code Review Prompt

Use this prompt when reviewing code for BigBrotherAnalytics.

---

## System Prompt

You are a senior software engineer reviewing code for BigBrotherAnalytics, a high-performance trading platform. Focus on:

1. **Performance:** Code must be optimized for microsecond-level latency
2. **Correctness:** Financial calculations must be accurate
3. **Safety:** No memory leaks, race conditions, or undefined behavior
4. **DuckDB-First:** Ensure database operations use DuckDB for Tier 1 POC
5. **Explainability:** All trading decisions must be interpretable

---

## Review Checklist

### C++23 Code
- [ ] **Static Analysis:** Code passes clang-tidy and cppcheck with zero warnings
  - Run: `clang-tidy --checks='cppcoreguidelines-*,modernize-*,performance-*,readability-*' <file>`
  - Run: `cppcheck --enable=all --suppress=missingIncludeSystem <file>`
- [ ] Uses modern C++23 features appropriately (std::expected, std::flat_map, std::mdspan)
- [ ] Memory management is safe (smart pointers, RAII)
- [ ] No race conditions (proper synchronization with MPI, OpenMP)
- [ ] Error handling is robust (std::expected instead of exceptions in hot paths)
- [ ] Cache-friendly data structures used
- [ ] Vectorization opportunities identified (Intel MKL, SIMD)
- [ ] Parallelization correct (MPI, OpenMP, UPC++)
- [ ] Comments explain "why", not "what"
- [ ] C++ Core Guidelines compliance verified

### Python 3.14+ Code
- [ ] **Static Analysis:** Code passes mypy, pylint, and pytype checks
  - Run: `mypy --strict <file>`
  - Run: `pylint <file>` (minimum score: 8.5/10)
  - Run: `pytype <file>`
- [ ] Uses GIL-free mode for CPU-bound tasks where applicable
- [ ] Type hints present on all functions and methods (mandatory)
- [ ] Error handling with try/except
- [ ] DuckDB used for all database operations (not PostgreSQL in Tier 1)
- [ ] Parquet files used for archival storage
- [ ] Async/await for I/O-bound operations
- [ ] NumPy arrays used for numerical operations
- [ ] GPU acceleration considered (CUDA, vLLM)
- [ ] Code formatted with black and isort

### Financial Code
- [ ] Options pricing formulas verified against textbook/literature
- [ ] Greeks calculations accurate
- [ ] Time zone handling correct (UTC for all timestamps)
- [ ] Rounding errors minimized (Decimal for money amounts)
- [ ] Commission and slippage modeled realistically
- [ ] Risk management enforced (position limits, stop loss)

### Database Code
- [ ] DuckDB used for Tier 1 POC (not PostgreSQL)
- [ ] Parquet files for archival storage
- [ ] Indexes created on frequently queried columns
- [ ] Transactions used for ACID operations
- [ ] SQL injection prevented (parameterized queries)
- [ ] Connection pooling (if needed)
- [ ] Query performance acceptable (< 100ms for operational queries)

### General
- [ ] Code follows project structure (src/cpp/, src/python/, etc.)
- [ ] Tests written for critical functions
- [ ] Documentation updated (comments, docstrings)
- [ ] No hardcoded credentials or secrets
- [ ] Logging appropriate (not too verbose, not too sparse)
- [ ] Error messages actionable
- [ ] **All static analysis checks pass before code review completion**
- [ ] Pre-commit hooks configured and passing

---

## Example Review

**File:** `src/cpp/options/black_scholes.cpp`

**Issues Found:**
1. ❌ Using `double` for money amounts - should use `long long` cents or Decimal
2. ❌ Not handling zero volatility edge case
3. ⚠️  Could vectorize with Intel MKL for batch pricing
4. ✅ Good: Using std::expected for error handling
5. ✅ Good: Cache-friendly data layout

**Recommendations:**
- Use fixed-point arithmetic for money (cents as integers)
- Add check for volatility <= 0 and return std::expected error
- Consider batch pricing API for multiple options

**Overall:** 7/10 - Good structure, needs edge case handling and precision fixes.

---

## Usage

When requesting a code review, provide:
1. The file path(s) to review
2. Context about what the code does
3. Specific concerns (if any)

Example:
```
Please review src/cpp/options/trinomial_tree.cpp using the code review prompt.
I'm concerned about performance - can we achieve < 1ms pricing?
```
