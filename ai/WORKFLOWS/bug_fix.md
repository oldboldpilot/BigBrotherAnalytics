# Workflow: Bug Fix

This workflow guides fixing bugs using the AI agent orchestration system.

---

## Overview

```
Bug Report → Orchestrator → Debugger → File Creator → Self-Correction → Commit
```

---

## Step-by-Step Process

### Step 1: Bug Report

**User Action:**
```
Bug Report:
- Issue: [Description of the bug]
- Steps to reproduce: [Step-by-step reproduction]
- Expected behavior: [What should happen]
- Actual behavior: [What actually happens]
- Environment: [OS, compiler, Python version]
- Error logs: [Stack trace, error messages]
- Recent changes: [git log, if known]

Please debug and fix this issue.
```

**AI Response:**
```
Orchestrator activating for bug fix...

Analysis:
- Severity: [CRITICAL/HIGH/MEDIUM/LOW]
- Affected components: [List]
- Required agents: Debugger, File Creator, Self-Correction
- Execution mode: Sequential
- Estimated time: [duration]

Proceeding with debugging...
```

---

### Step 2: Root Cause Analysis

**Orchestrator Action:**
```
Invoking Debugger for root cause analysis...
```

**Debugger Tasks:**

1. **Reproduce the Bug**
   ```bash
   # Run reproduction steps
   [commands to trigger bug]

   # Capture error output
   [error logs, stack traces]
   ```

2. **Gather Context**
   - Read relevant source files
   - Check recent git commits
   - Review related test failures
   - Check system logs

3. **Form Hypothesis**
   ```
   Hypothesis: [What might be causing the bug]
   Evidence: [Supporting information]
   ```

4. **Test Hypothesis**
   ```bash
   # Add debug logging
   # Run with debugger (gdb, pdb)
   # Inspect state at failure point
   ```

5. **Identify Root Cause**
   ```
   Root Cause: [Exact cause of the bug]
   Location: [File:line number]
   Reason: [Why the bug occurs]
   ```

**Debugger Output:**
```markdown
## Root Cause Analysis

**Bug:** [Description]

**Root Cause:** [Detailed explanation]

**Location:**
- File: `src/cpp/options/trinomial_tree.cpp:142`
- Function: `calculate_option_value()`

**Issue:**
Off-by-one error in loop iteration causing array index out of bounds.

**Code:**
```cpp
// Bug: should be i < n_steps, not i <= n_steps
for (size_t i = 0; i <= n_steps; ++i) {  // ❌
    values[i] = calculate_value(i);
}
```

**Fix Strategy:**
Change loop condition to `i < n_steps` to prevent out-of-bounds access.

**Testing Plan:**
- Add unit test for edge case (n_steps = 100)
- Add regression test
- Verify with AddressSanitizer
```

**Validation:**
- [ ] Bug reproduced successfully
- [ ] Root cause identified
- [ ] Fix strategy defined
- [ ] Testing plan created

---

### Step 3: Implement Fix

**Orchestrator Action:**
```
Invoking File Creator to implement the fix...
Input: Root cause analysis, fix strategy
```

**File Creator Tasks:**

1. **Apply Fix**
   ```cpp
   // Fixed version
   for (size_t i = 0; i < n_steps; ++i) {  // ✅
       values[i] = calculate_value(i);
   }
   ```

2. **Add Regression Test**
   ```cpp
   // tests/cpp/test_options.cpp
   TEST(TrinomialTree, EdgeCaseHandling) {
       // Test case that would have triggered the bug
       OptionContract contract = create_test_contract();
       MarketData market = create_test_market();

       // This should not crash
       auto result = price_option(contract, market, PricingModel::Trinomial);

       ASSERT_TRUE(result.has_value());
       ASSERT_GT(result.value().price, 0.0);
   }
   ```

3. **Update Documentation (if needed)**
   - Add comment explaining the fix
   - Update known issues list (if maintained)

**File Creator Output:**
- Modified files list
- Added test files
- Updated documentation

**Validation:**
- [ ] Fix applied correctly
- [ ] Regression test added
- [ ] No new issues introduced

---

### Step 4: Self-Correction

**Orchestrator Action:**
```
Invoking Self-Correction to validate the fix...
```

**Self-Correction Checks:**

1. **Bug No Longer Reproduces**
   ```bash
   # Run original reproduction steps
   [commands that previously triggered bug]

   # Should now succeed
   ✅ Bug no longer reproduces
   ```

2. **Regression Test Passes**
   ```bash
   pytest tests/cpp/test_options.cpp::TrinomialTree::EdgeCaseHandling

   ✅ Test passes
   ```

3. **All Existing Tests Still Pass**
   ```bash
   pytest tests/ --tb=short

   ✅ 248 tests passed
   ```

4. **Memory Safety (if C++)**
   ```bash
   # Run with AddressSanitizer
   ASAN_OPTIONS=detect_leaks=1 ./build/test_runner

   ✅ No memory errors detected
   ```

5. **Performance Not Degraded**
   ```bash
   pytest tests/ --benchmark

   ✅ No performance regressions
   ```

**Self-Correction Output:**
```
Validation Report:
✅ Bug no longer reproduces
✅ Regression test passes
✅ All existing tests pass (248/248)
✅ Memory safety verified
✅ No performance degradation

Ready to commit!
```

**Validation:**
- [ ] Bug fixed
- [ ] Tests pass
- [ ] No regressions
- [ ] Performance maintained

---

### Step 5: Commit

**Orchestrator Action:**
```
All validation passed. Committing fix...
```

**Git Commit:**
```bash
git add src/cpp/options/trinomial_tree.cpp
git add tests/cpp/test_options.cpp

git commit -m "fix(options): Fix off-by-one error in trinomial tree calculation

Fixed array index out-of-bounds error in trinomial tree pricing loop.

Root Cause:
- Loop condition was 'i <= n_steps' instead of 'i < n_steps'
- Caused out-of-bounds access when n_steps = 100

Fix:
- Changed loop condition to 'i < n_steps'
- Added regression test for edge case

Testing:
- Added regression test
- Verified with AddressSanitizer (no memory errors)
- All 248 tests pass
- No performance degradation

Fixes: Issue #[number] (if applicable)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Push to GitHub:**
```bash
git push origin master
```

---

## Example: Fixing Correlation Engine Crash

### Step 1: Bug Report
```
Bug Report:
- Issue: Correlation engine crashes with "Segmentation fault"
- Steps to reproduce: ./correlation_engine --symbols 1000
- Expected: Calculate 1000x1000 correlation matrix
- Actual: Segfault after ~10 seconds
- Environment: Ubuntu 22.04, GCC 15, 32 cores
- Error logs:
  Segmentation fault (core dumped)
  Backtrace shows crash in MPI_Allreduce
- Recent changes: Switched to MPI parallelization (commit abc123)
```

### Step 2: Root Cause Analysis

**Debugger Investigation:**
```bash
# Run with gdb
gdb ./correlation_engine
(gdb) run --symbols 1000
# ... crash occurs
(gdb) backtrace

# Shows crash in MPI communication
```

**Root Cause Found:**
```markdown
## Root Cause

**Bug:** Segfault in MPI_Allreduce during correlation calculation

**Root Cause:**
Buffer size mismatch in MPI communication. Allocated buffer for n*(n-1)/2
correlations, but MPI_Allreduce expects n*n buffer size.

**Location:**
- File: `src/cpp/correlation/correlation_engine.cpp:87`
- Function: `parallel_correlate()`

**Fix Strategy:**
Allocate full n*n buffer for MPI communication, even though only
half is needed (symmetric matrix).
```

### Step 3: Implement Fix

**File Creator:**
```cpp
// Before (buggy)
std::vector<double> correlations(n * (n-1) / 2);  // ❌ Too small

// After (fixed)
std::vector<double> correlations(n * n);  // ✅ Full matrix
```

**Regression Test:**
```cpp
TEST(CorrelationEngine, LargeMatrixMPI) {
    // Test case that triggered the bug
    CorrelationEngine engine;
    auto data = generate_test_data(1000, 252);  // 1000 symbols

    // Should not crash
    auto result = engine.calculate(data, config);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().n_symbols, 1000);
}
```

### Step 4: Self-Correction
```
✅ Bug no longer reproduces (tested with 1000 symbols)
✅ Regression test passes
✅ All existing tests pass (248/248)
✅ Memory checked with AddressSanitizer (no issues)
✅ Performance: Slightly slower due to larger buffer (acceptable)

Ready to commit!
```

### Step 5: Commit
```bash
fix(correlation): Fix buffer size for MPI communication

Fixed segfault when calculating large correlation matrices with MPI.

Root Cause:
- MPI_Allreduce requires n*n buffer, not n*(n-1)/2
- Caused out-of-bounds memory access in MPI communication

Fix:
- Allocate full n*n buffer for symmetric matrix
- Memory overhead is acceptable for correctness

Testing:
- Added regression test for 1000 symbol matrix
- Verified with AddressSanitizer
- All 248 tests pass

Fixes: #42
```

---

## Troubleshooting

### Issue: Cannot Reproduce Bug

**Scenario:** Following reproduction steps doesn't trigger the bug.

**Actions:**
1. Request more details from user (environment, exact commands)
2. Check if bug is environment-specific (race condition, OS-specific)
3. Try with different configurations
4. Ask user for core dump or more detailed logs

### Issue: Fix Breaks Other Tests

**Scenario:** Fix resolves the bug but causes other tests to fail.

**Actions:**
1. Analyze why other tests are failing
2. Determine if tests have wrong assumptions
3. Update tests if they're testing incorrect behavior
4. Rethink fix strategy if it has unintended side effects

### Issue: Root Cause Unclear

**Scenario:** Cannot identify exact cause of bug.

**Actions:**
1. Add more debug logging
2. Use debugger to inspect state
3. Run with sanitizers (AddressSanitizer, ThreadSanitizer)
4. Bisect git history to find when bug was introduced
5. Escalate to human developer if still unclear

---

## Success Criteria

Bug fix is complete when:
- [ ] Bug reproduced successfully
- [ ] Root cause identified and documented
- [ ] Fix implemented correctly
- [ ] Regression test added
- [ ] All tests pass (no regressions)
- [ ] Memory safety verified (if C++)
- [ ] Performance not degraded
- [ ] Committed to git
- [ ] Pushed to GitHub

---

**Estimated Time:** 30 minutes to 4 hours depending on bug complexity

**Key Principle:** Systematic debugging with automated validation ensures bugs are fixed correctly without introducing regressions.
