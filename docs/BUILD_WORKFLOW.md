# BigBrotherAnalytics Build Workflow

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-08
**Version:** 1.0.0

---

## Overview

BigBrotherAnalytics enforces code quality through **mandatory clang-tidy checks** before every build and commit.

**Quality Gates:**
1. **Pre-Commit:** clang-tidy runs on staged files (blocks commit if errors)
2. **Pre-Build:** clang-tidy runs on all files (blocks build if errors)
3. **CI/CD:** Full validation on every PR + CodeQL 2x daily

---

## Build Workflow with Automatic clang-tidy

### Standard Build Process

```bash
# 1. Navigate to build directory
cd /home/muyiwa/Development/BigBrotherAnalytics/build

# 2. Configure (if first time or CMakeLists.txt changed)
env CC=/usr/local/bin/clang \
    CXX=/usr/local/bin/clang++ \
    cmake -G Ninja ..

# 3. Build (clang-tidy runs automatically BEFORE compilation)
ninja
```

**What happens during build:**

```
CMake Configure
    ↓
Run scripts/run_clang_tidy.sh (AUTOMATIC)
    ├─ Checks all .cpp and .cppm files
    ├─ Validates against .clang-tidy configuration
    ├─ Reports errors and warnings
    ├─ Blocks build if errors found
    ↓
If clang-tidy passes:
    ↓
Compile C++23 modules
    ↓
Link libraries and executables
    ↓
Build complete ✅
```

### Bypassing clang-tidy (NOT RECOMMENDED)

**Only for emergency situations:**

```bash
# Bypass pre-build check
export SKIP_CLANG_TIDY=1
ninja

# Bypass pre-commit check
git commit --no-verify
```

**⚠️ WARNING:** Bypassing clang-tidy will:
- Allow code quality issues into codebase
- May fail CI/CD checks
- Violate project standards
- Should only be used for documentation/config changes

---

## Commit Workflow with Automatic clang-tidy

### Standard Commit Process

```bash
# 1. Stage your changes
git add -A

# 2. Commit (pre-commit hook runs automatically)
git commit -m "your message

Author: Olumuyiwa Oluwasanmi"
```

**What happens during commit:**

```
git commit
    ↓
Pre-commit hook (.githooks/pre-commit) runs:
    ↓
1. Check trailing return syntax
2. Check [[nodiscard]] attributes
3. Check module structure
4. Check documentation
5. Run clang-tidy on staged files (MANDATORY)
    ├─ C++ Core Guidelines
    ├─ CERT Security
    ├─ Concurrency safety
    ├─ Performance checks
    ├─ OpenMP/MPI safety
    ├─ Blocks if errors found
6. Check code formatting
    ↓
If all checks pass:
    ↓
Commit succeeds ✅
    ↓
Post-commit hook shows tips
```

### Pre-Commit Hook Enforcement

**clang-tidy is MANDATORY:**
- Runs on all staged .cpp and .cppm files
- Checks against .clang-tidy configuration
- **Blocks commit if errors found**
- Requires clang-tidy to be installed

**If clang-tidy not installed:**
```bash
# Ubuntu/Debian
sudo apt-get install clang-tidy

# macOS
brew install llvm

# RHEL/Fedora
sudo dnf install clang-tools-extra
```

---

## clang-tidy Configuration

**File:** `.clang-tidy`

**Enabled Check Categories (11 total):**
1. **cppcoreguidelines-*** - C++ Core Guidelines (ALL rules)
2. **cert-*** - CERT C++ Secure Coding Standard
3. **concurrency-*** - Thread safety, race conditions, deadlocks
4. **performance-*** - Optimization opportunities
5. **portability-*** - Cross-platform compatibility
6. **openmp-*** - OpenMP parallelization safety
7. **mpi-*** - MPI message passing correctness
8. **modernize-*** - Modern C++23 features, trailing return syntax
9. **bugprone-*** - Common bug patterns
10. **clang-analyzer-*** - Static analysis
11. **readability-*** - Code clarity

**Errors (Will Block Build/Commit):**
- modernize-use-trailing-return-type
- cppcoreguidelines-special-member-functions
- cppcoreguidelines-no-malloc
- modernize-use-nullptr
- modernize-use-nodiscard

---

## Manual Validation

### Validate Before Making Changes

```bash
# Check current code quality
./scripts/validate_code.sh src/

# Or specific file
./scripts/validate_code.sh src/utils/logger.cpp
```

### Run clang-tidy Manually

```bash
# Single file
clang-tidy src/utils/logger.cpp -- -std=c++23 -I./src

# All files
./scripts/run_clang_tidy.sh

# See all enabled checks
clang-tidy --list-checks
```

### Fix clang-tidy Issues

```bash
# Auto-fix some issues (USE WITH CAUTION)
clang-tidy --fix src/your_file.cpp -- -std=c++23 -I./src

# Apply formatting
clang-format -i src/your_file.cpp
```

---

## Typical Development Workflow

### Making Code Changes

```bash
# 1. Make your changes
vim src/utils/new_feature.cpp

# 2. Validate changes
./scripts/validate_code.sh src/utils/new_feature.cpp

# 3. Fix any clang-tidy errors reported

# 4. Build (clang-tidy runs again automatically)
cd build && ninja

# 5. Run tests
env LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    ./run_tests.sh

# 6. Commit (pre-commit hook runs clang-tidy again)
git add src/utils/new_feature.cpp
git commit -m "feat: Add new feature

Author: Olumuyiwa Oluwasanmi"

# 7. Push to GitHub
git push origin master
```

### Workflow Diagram

```
Code Changes
    ↓
Manual Validation (optional but recommended)
    ├─ ./scripts/validate_code.sh
    ↓
Build
    ├─ CMake runs clang-tidy (AUTOMATIC)
    ├─ Build fails if errors
    ↓
Tests
    ├─ ./run_tests.sh
    ↓
Commit
    ├─ Pre-commit hook runs clang-tidy (AUTOMATIC)
    ├─ Commit fails if errors
    ↓
Push to GitHub
    ├─ CI/CD runs comprehensive checks
    ├─ PR fails if violations
```

---

## Troubleshooting

### "clang-tidy validation failed"

**During Build:**
```bash
# See detailed errors
./scripts/run_clang_tidy.sh

# Fix errors in reported files
# Then rebuild
```

**During Commit:**
```bash
# See which files have errors
# Errors shown in commit output

# Fix the files
# Then commit again
```

### "Too many warnings"

```bash
# Review warnings by category
clang-tidy src/file.cpp -- -std=c++23 -I./src | grep "warning:"

# Fix high-priority warnings first
# Performance warnings
# Concurrency warnings
# Then readability warnings
```

### "clang-tidy is slow"

**Pre-commit is fast** (only staged files)
**Pre-build checks all files** (2-5 minutes for full codebase)

**To speed up:**
- Fix errors incrementally
- Use ccache for compilation
- Run validation frequently to catch issues early

---

## Performance Expectations

**Pre-Commit Hook:**
- Time: 20-60 seconds (depends on files staged)
- Scope: Only staged C++ files

**Pre-Build Check:**
- Time: 2-5 minutes (full codebase)
- Scope: All C++ files in src/

**CI/CD:**
- Time: 5-10 minutes (includes build + tests)
- Scope: Full codebase + comprehensive analysis

---

## Environment Variables

### SKIP_CLANG_TIDY

**Purpose:** Bypass pre-build clang-tidy check

**Usage:**
```bash
export SKIP_CLANG_TIDY=1
ninja
```

**When to use:**
- Documentation-only changes
- Configuration file updates
- Emergency hotfixes (fix properly later)

**When NOT to use:**
- Any C++ code changes
- New features
- Bug fixes
- Refactoring

---

## CI/CD Integration

### GitHub Actions

**Triggers:**
- Every pull request
- Twice daily (8 AM & 8 PM UTC) for CodeQL

**Checks:**
1. clang-tidy comprehensive analysis
2. Build verification
3. Test suite execution
4. CodeQL security scan
5. Standards compliance

**PR Merge Requirements:**
- ✅ All clang-tidy checks pass
- ✅ Build succeeds
- ✅ All tests pass
- ✅ No security vulnerabilities

---

## Summary

**clang-tidy runs 3 times:**

1. **Before Build** (CMake) - All files, blocks build
2. **Before Commit** (Git hook) - Staged files, blocks commit
3. **Before Merge** (CI/CD) - All files, blocks PR merge

**This ensures:**
- 100% C++ Core Guidelines compliance
- Zero security vulnerabilities (CERT)
- Thread-safe concurrent code
- Optimized performance
- Safe OpenMP/MPI parallelization
- Professional code quality

**You can't build or commit without passing clang-tidy!** ✅

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** 2025-11-08
**Version:** 1.0.0
