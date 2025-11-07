# Debugging Prompt

Use this prompt when debugging issues in BigBrotherAnalytics.

---

## System Prompt

You are an expert debugger helping troubleshoot issues in BigBrotherAnalytics, a high-performance trading platform. Apply systematic debugging:

1. **Reproduce:** Understand how to trigger the bug
2. **Isolate:** Narrow down the root cause
3. **Fix:** Propose and implement solution
4. **Verify:** Test the fix thoroughly
5. **Prevent:** Add tests to catch regressions

---

## Debugging Checklist

### Before Starting
- [ ] Can reproduce the bug consistently?
- [ ] Have minimal reproduction case?
- [ ] Know expected vs actual behavior?
- [ ] Checked recent changes (git log)?
- [ ] Reviewed error logs/stack traces?
- [ ] **Ran static analysis tools?** (Often catches bugs before runtime testing)
  - C++: `clang-tidy` and `cppcheck`
  - Python: `mypy`, `pylint`, and `pytype`

### Common Issues

#### Performance Issues
- [ ] Profile with perf/gprof (C++) or cProfile (Python)
- [ ] Check for unnecessary copies (C++)
- [ ] Look for N+1 query problems (database)
- [ ] Verify parallelization is working (MPI/OpenMP)
- [ ] Check cache efficiency (perf stat)
- [ ] Look for contention (lock profiling)

#### Memory Issues
- [ ] Check for leaks with valgrind/AddressSanitizer
- [ ] Verify smart pointer usage (C++)
- [ ] Look for circular references (Python)
- [ ] Check buffer overflows
- [ ] Profile memory usage over time

#### Database Issues
- [ ] Is DuckDB being used (not PostgreSQL in Tier 1)?
- [ ] Are indexes present on queried columns?
- [ ] Is the query optimized (EXPLAIN)?
- [ ] Are transactions being used correctly?
- [ ] Check for lock contention

#### Parallelization Issues
- [ ] Race conditions (use ThreadSanitizer)
- [ ] Deadlocks (check lock ordering)
- [ ] False sharing (check data layout)
- [ ] Load imbalance (profile per-thread)
- [ ] Synchronization overhead

#### Financial Calculation Issues
- [ ] Verify input data quality (no NaN, infinity)
- [ ] Check time zone handling (should be UTC)
- [ ] Verify date/time calculations (trading days vs calendar days)
- [ ] Look for floating-point precision issues
- [ ] Validate against known test cases

### Debugging Tools

**Static Analysis (Run First!):**
```bash
# C++ static analysis
clang-tidy --checks='cppcoreguidelines-*,modernize-*,performance-*,readability-*' <file>
cppcheck --enable=all --suppress=missingIncludeSystem <file>

# Python static analysis
mypy --strict <file>
pylint <file>
pytype <file>
```

**C++:**
```bash
# Memory errors
valgrind --leak-check=full ./program

# Address sanitizer
g++ -fsanitize=address -g program.cpp

# Thread sanitizer
g++ -fsanitize=thread -g program.cpp

# Profiling
perf record ./program
perf report

# GDB debugging
gdb ./program
(gdb) run
(gdb) backtrace
```

**Python:**
```bash
# Memory profiling
python -m memory_profiler script.py

# CPU profiling
python -m cProfile -s cumtime script.py

# Line-by-line profiling
kernprof -l -v script.py

# Debugging
python -m pdb script.py
```

**DuckDB:**
```sql
-- Query plan
EXPLAIN SELECT ...;

-- Query analysis
EXPLAIN ANALYZE SELECT ...;

-- Check indexes
PRAGMA show_tables;
PRAGMA table_info(table_name);
```

### Root Cause Analysis

1. **Gather data:**
   - Error messages, stack traces
   - Input that triggers the bug
   - System state (memory, CPU, disk)

2. **Form hypothesis:**
   - What could cause this behavior?
   - What changed recently?

3. **Test hypothesis:**
   - Add logging/assertions
   - Use debugger to inspect state
   - Run minimal reproduction

4. **Repeat until found**

### Fix Verification

- [ ] Bug no longer reproduces
- [ ] Added regression test
- [ ] No new issues introduced
- [ ] Performance not degraded
- [ ] Documentation updated

---

## Example Debugging Session

**Issue:** Correlation engine taking 60 seconds instead of < 10 seconds

**Investigation:**
1. Profile with perf: `perf record -g ./correlation_engine`
2. Find hotspot: 90% time in `std::map::insert`
3. Hypothesis: Using wrong data structure

**Root Cause:** Using `std::map` instead of cache-friendly `std::flat_map`

**Fix:**
```cpp
// Before
std::map<std::pair<std::string, std::string>, double> correlations;

// After
std::flat_map<std::pair<std::string, std::string>, double> correlations;
```

**Result:** 60s → 8s (7.5x speedup)

**Prevention:** Added performance regression test

---

## Usage

When reporting a bug for debugging, provide:
1. Error message / symptom
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment (OS, compiler, Python version)
6. Recent changes (git log)

Example:
```
The correlation engine is crashing with "Segmentation fault" when processing
1000+ symbols. Steps to reproduce: run `./correlation_engine --symbols 1000`.
Expected: correlation matrix calculated. Actual: segfault. Recent change:
switched to MPI parallelization. Please debug using the debugging prompt.
```
