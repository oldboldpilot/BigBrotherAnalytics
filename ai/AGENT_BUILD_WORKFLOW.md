# AI Agent Build & Test Workflow

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-08
**Applies To:** All AI agents (Claude, Copilot, custom agents)

---

## MANDATORY Workflow for Code Changes

When an AI agent makes ANY code modification, it MUST follow this exact workflow:

### Step 1: Validate Code Quality

**BEFORE committing, ALWAYS run:**
```bash
./scripts/validate_code.sh
```

**This automatically runs:**
1. clang-tidy (C++ Core Guidelines enforcement)
2. cppcheck (Static analysis)
3. Build verification
4. Standards checks

**Expected Output:**
```
✅ ALL CHECKS PASSED!
```

**If validation fails:**
- Fix all clang-tidy errors
- Fix all cppcheck errors
- Resolve build issues
- Do NOT proceed until validation passes

### Step 2: Build Project

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/build

# Configure (if needed)
env CC=/usr/local/bin/clang \
    CXX=/usr/local/bin/clang++ \
    cmake -G Ninja ..

# Build (clang-tidy runs AUTOMATICALLY before compilation)
ninja
```

**What Happens Automatically:**
1. CMake runs `scripts/run_clang_tidy.sh` before building
2. clang-tidy validates all C++ files against .clang-tidy config
3. **Build is BLOCKED if clang-tidy finds errors**
4. If validation passes, compilation proceeds

**Expected:**
- clang-tidy validation passes (0 errors)
- All modules compile
- Libraries and executables link successfully

**If clang-tidy fails:**
- Build stops immediately
- Error messages shown
- Fix errors in reported files
- Run `ninja` again

### Step 3: Run Tests

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics

# Set library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Run all tests
./run_tests.sh

# Or specific test
./build/bin/test_options_pricing
./build/bin/test_correlation
```

**Expected:** All tests pass

### Step 4: Commit Changes

```bash
git add -A
git commit -m "descriptive message

Author: Olumuyiwa Oluwasanmi"
```

**Pre-commit hook automatically runs:**
- Trailing return syntax check
- [[nodiscard]] verification
- Module structure validation
- Documentation check
- **clang-tidy** (C++ Core Guidelines)
- **cppcheck** (Static analysis)
- Code formatting

**If hook fails:**
- Review error messages
- Fix issues
- Retry commit

---

## Code Standards Checklist

Before considering code complete, verify:

### ✅ C++23 Module Structure
- [ ] Global module fragment (`module;`) for std library includes
- [ ] Module declaration (`export module bigbrother.component;`)
- [ ] Module imports (`import bigbrother.dependency;`)
- [ ] Export namespace (`export namespace bigbrother::component {}`)

### ✅ Function Syntax
- [ ] ALL functions use trailing return: `auto func() -> ReturnType`
- [ ] NO old-style: `ReturnType func()` ❌
- [ ] Constructors and destructors exempt (no return type)

### ✅ Error Handling
- [ ] Use `std::expected<T, Error>` for fallible operations
- [ ] Use `Result<T>` type alias from types.cppm
- [ ] Return errors with `makeError<T>()` or `std::unexpected()`

### ✅ Attributes
- [ ] `[[nodiscard]]` on all getters and query methods
- [ ] `constexpr` for compile-time evaluation where possible
- [ ] `noexcept` for functions that cannot throw

### ✅ C++ Core Guidelines
- [ ] C.1: Use struct for passive data, class for invariants
- [ ] C.21: Define or delete all 5 special members
- [ ] F.16: Pass cheap types by value
- [ ] F.20: Return values, not output parameters
- [ ] R.1: RAII for all resources

### ✅ Performance
- [ ] Prefer `std::unordered_map` over `std::map`
- [ ] Use custom hash for complex keys
- [ ] Move semantics for expensive types
- [ ] Avoid unnecessary copies

### ✅ Documentation
- [ ] File header with author: Olumuyiwa Oluwasanmi
- [ ] Brief description of component
- [ ] C++ Core Guidelines compliance noted
- [ ] Function documentation for public API

### ✅ Fluent APIs
- [ ] Builder pattern for complex construction
- [ ] Method chaining (return *this)
- [ ] [[nodiscard]] on terminal methods
- [ ] Clear, declarative interface

---

## Agent Responsibilities

### When Creating New Files:

**MUST include in file header:**
```cpp
/**
 * BigBrotherAnalytics - Component Name
 *
 * Brief description
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-08
 *
 * Following C++ Core Guidelines:
 * - [List applicable guidelines]
 * - Trailing return type syntax throughout
 */
```

**MUST validate before commit:**
```bash
./scripts/validate_code.sh src/your_new_file.cpp
```

### When Modifying Existing Files:

**MUST:**
1. Preserve existing authorship
2. Follow existing code style (trailing return syntax)
3. Maintain module structure
4. Run validation after changes
5. Ensure build still succeeds
6. Verify tests still pass

### When Fixing Build Errors:

**Process:**
1. Read error message carefully
2. Check module dependencies
3. Verify global module fragment structure
4. Fix the issue
5. Run `./scripts/validate_code.sh`
6. Rebuild with `ninja`
7. Only commit if validation passes

---

## Common Issues & Solutions

### Issue: "module 'bigbrother.X' not found"

**Solution:** Check CMakeLists.txt - ensure module X's library is linked:
```cmake
target_link_libraries(your_lib
    PUBLIC
    dependency_lib  # Contains the module
)
```

### Issue: "declaration in global module follows declaration in module"

**Solution:** Move #includes to global module fragment:
```cpp
module;           // Global fragment
#include <vector> // Std library here

module bigbrother.component;  // Then module declaration
```

### Issue: "no viable conversion" or template errors

**Solution:** Check parameter types match exactly between declaration and usage

### Issue: clang-tidy errors

**Solution:** Follow C++ Core Guidelines:
- Add const to member functions that don't modify state
- Use [[nodiscard]] on query functions
- Implement Rule of Five properly

### Issue: cppcheck warnings

**Solution:** Fix logic issues:
- Initialize all variables
- Check for null before dereferencing
- Avoid unused variables

---

## Validation Reports

**Generate full validation report:**
```bash
./scripts/validate_code.sh src/ 2>&1 | tee validation_report.txt
```

**Check specific module:**
```bash
./scripts/validate_code.sh src/risk_management/
```

**Quick check single file:**
```bash
clang-tidy src/file.cpp -- -std=c++23 -I./src
cppcheck --std=c++23 src/file.cpp
```

---

## Performance Targets

**Validation Time:**
- clang-tidy: ~2-5 sec per file
- cppcheck: ~1-3 sec per file
- Build: ~60-120 sec (full clean build)
- Tests: ~5-10 sec

**Total workflow:** ~2-3 minutes for complete validation

---

## Integration with Git Hooks

Pre-commit hook automatically runs subset of checks:
- Trailing return syntax: ~1 sec
- Module structure: ~1 sec
- [[nodiscard]]: ~1 sec
- **clang-tidy: ~10-30 sec** (on staged files only)
- **cppcheck: ~5-15 sec** (on staged files only)

Total pre-commit time: ~20-50 seconds (fast feedback!)

---

## CI/CD Integration

**Local (Pre-commit):**
- Fast checks on changed files only
- Immediate feedback
- Blocks commit if critical issues

**GitHub Actions (PR/Schedule):**
- Comprehensive checks on entire codebase
- CodeQL security analysis (2x daily)
- Full validation suite
- Prevents merge if failing

---

## Questions?

See:
- `docs/CODING_STANDARDS.md` - Complete coding standards
- `BUILD_AND_TEST_INSTRUCTIONS.md` - Build procedures
- `.github/workflows/code-quality.yml` - CI/CD details
- `.githooks/pre-commit` - Local enforcement details

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** 2025-11-08
**Version:** 1.0.0
