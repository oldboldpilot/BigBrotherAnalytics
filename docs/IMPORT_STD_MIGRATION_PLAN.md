# Import std; Migration Plan

**Author:** Olumuyiwa Oluwasanmi  
**Date:** 2025-11-09  
**Status:** Planning Phase

---

## Overview

Migrate from `#include <header>` to `import std;` for cleaner C++23 module syntax.

**Current Pattern:**
```cpp
module;
#include <vector>
#include <string>
#include <memory>

export module bigbrother.component;
```

**Target Pattern:**
```cpp
module;

import std;  // Single import for all standard library

export module bigbrother.component;
```

---

## Prerequisites ✅ COMPLETE

**libc++ with Module Support:**
- ✅ LLVM/libc++ 21.1.5 source downloaded
- ✅ Built with -DLIBCXX_ENABLE_MODULES=ON
- ✅ Installed to /opt/libc++_modules
- ✅ std.cppm module file available

**Files:**
- `/opt/libc++_modules/share/libc++/v1/std.cppm` - Main std module
- `/opt/libc++_modules/share/libc++/v1/std.compat.cppm` - C compatibility
- `/opt/libc++_modules/lib/libc++.modules.json` - Module map

---

## Migration Steps

### Phase 1: Precompile std Module (1 hour)

**Step 1.1: Verify libc++ installation**
```bash
ls -lh /opt/libc++_modules/share/libc++/v1/std.cppm
ls -lh /opt/libc++_modules/lib/libc++.so
```

**Step 1.2: Precompile std module**
```bash
/home/linuxbrew/.linuxbrew/bin/clang++ \
  -std=c++23 \
  -stdlib=libc++ \
  -nostdinc++ \
  -isystem /opt/libc++_modules/include/c++/v1 \
  -fmodule-output \
  -c /opt/libc++_modules/share/libc++/v1/std.cppm \
  -o /opt/libc++_modules/lib/std.pcm
```

**Step 1.3: Test std module**
```cpp
// test_std_import.cppm
module;

export module test;

import std;

export namespace test {
    auto testVector() -> int {
        std::vector<int> v{1,2,3};
        return v.size();
    }
}
```

Compile test:
```bash
clang++ -std=c++23 -stdlib=libc++ \
  -nostdinc++ \
  -isystem /opt/libc++_modules/include/c++/v1 \
  -fprebuilt-module-path=/opt/libc++_modules/lib \
  -c test_std_import.cppm
```

---

### Phase 2: Update CMake Configuration (2 hours)

**Step 2.1: Add libc++ module flags to CMakeLists.txt**
```cmake
# C++23 Standard Library Modules
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++")
include_directories(SYSTEM /opt/libc++_modules/include/c++/v1)
link_directories(/opt/libc++_modules/lib)

# Prebuilt module path for import std;
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprebuilt-module-path=/opt/libc++_modules/lib")
```

**Step 2.2: Add std module as dependency**
```cmake
# Ensure std.pcm is available
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/std.pcm
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++23 -stdlib=libc++ 
            -fmodule-output 
            -c /opt/libc++_modules/share/libc++/v1/std.cppm
            -o ${CMAKE_BINARY_DIR}/std.pcm
    COMMENT "Precompiling std module"
)
add_custom_target(std_module DEPENDS ${CMAKE_BINARY_DIR}/std.pcm)
```

**Step 2.3: Make all targets depend on std_module**
```cmake
add_dependencies(utils std_module)
add_dependencies(correlation_engine std_module)
# ... for all libraries
```

---

### Phase 3: Convert Modules One-by-One (8-12 hours)

**Conversion Priority:**
1. **Small utility modules first** (test conversions, low risk)
2. **Independent modules** (no circular dependencies)
3. **Core modules** (correlation, options, risk)
4. **Integration modules** (strategies, decision engine)

**Module Conversion Checklist:**

For each .cppm file:
- [ ] Remove all `#include <header>` from global module fragment
- [ ] Add `import std;` after `module;`
- [ ] Keep project module imports (`import bigbrother.*`)
- [ ] Build and verify
- [ ] Run tests
- [ ] Commit if successful

**Example Conversion:**

**Before:**
```cpp
module;

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <expected>

export module bigbrother.utils.types;

export namespace bigbrother::types {
    // ... implementation
}
```

**After:**
```cpp
module;

import std;  // Replaces all standard library includes

export module bigbrother.utils.types;

export namespace bigbrother::types {
    // ... implementation (unchanged)
}
```

---

### Phase 4: Test Conversions (Priority Order)

**Batch 1: Utilities (Low Risk)**
- [ ] src/utils/types.cppm
- [ ] src/utils/logger.cppm
- [ ] src/utils/math.cppm
- [ ] src/utils/timer.cppm

**Batch 2: Core Engines (Medium Risk)**
- [ ] src/correlation_engine/correlation.cppm
- [ ] src/correlation_engine/options_pricing.cppm
- [ ] src/correlation_engine/trinomial_tree.cppm
- [ ] src/correlation_engine/black_scholes.cppm

**Batch 3: Business Logic (Higher Risk)**
- [ ] src/risk_management/risk_management.cppm
- [ ] src/trading_decision/strategy.cppm
- [ ] src/trading_decision/strategies.cppm

**Batch 4: Applications**
- [ ] src/main.cpp (may need `import std;` + module imports)
- [ ] src/backtest_main.cpp

---

### Phase 5: Verification & Testing (2 hours)

**For each batch:**
1. Convert 3-5 modules
2. Run: `cd build && ninja`
3. Verify: All libraries compile
4. Test: Run test executables
5. Check: clang-tidy still 0 errors, 0 warnings
6. Commit: If all pass

**Full verification:**
```bash
# Clean build
rm -rf build/*
cd build && cmake .. && ninja

# Run all tests
./run_tests.sh

# Verify clang-tidy
./scripts/run_clang_tidy.sh

# Test Python bindings
LD_LIBRARY_PATH=/opt/libc++_modules/lib:$LD_LIBRARY_PATH \
  uv run python -c "import bigbrother_options; print('OK')"
```

---

## Potential Issues & Solutions

### Issue 1: std Module Not Found

**Symptom:**
```
error: module 'std' not found
```

**Solution:**
```bash
# Ensure std.pcm is in module path
export CMAKE_CXX_FLAGS="-fprebuilt-module-path=/opt/libc++_modules/lib"
```

### Issue 2: Header Conflicts

**Symptom:**
```
error: cannot mix module and non-module includes
```

**Solution:**
- Remove ALL `#include <std*>` from global module fragment
- Keep only external library includes (DuckDB, spdlog, pybind11)

### Issue 3: Symbol Not Found

**Symptom:**
```
error: use of undeclared identifier 'std::vector'
```

**Solution:**
- Verify `import std;` comes BEFORE module declaration
- Check std.pcm is being found

---

## Benefits of import std;

**Compile Time:**
- Precompiled std module: Faster compilation
- No header parsing: Reduced preprocessor overhead
- Incremental builds: Only recompile changed modules

**Code Clarity:**
- Single `import std;` vs dozens of `#include`
- Cleaner module interface
- True module isolation

**Maintenance:**
- No include guard issues
- No header order dependencies
- Easier to see module dependencies

---

## Timeline Estimate

**Total: 12-16 hours**
- Phase 1: Precompile std module - 1 hour
- Phase 2: CMake configuration - 2 hours
- Phase 3: Module conversions - 8-12 hours (25 modules)
- Phase 4: Verification - 2 hours

**Risk Level:** MEDIUM
- Can break build if not careful
- Incremental approach reduces risk
- Easy to revert if issues found

**Recommendation:** Start with Phase 1-2, test thoroughly, then proceed with Phase 3 in small batches.

---

## Success Criteria

- [ ] All 25 .cppm modules use `import std;`
- [ ] Build: 100% SUCCESS maintained
- [ ] clang-tidy: 0 errors, 0 warnings maintained
- [ ] All tests passing
- [ ] Python bindings working
- [ ] No performance regression

---

## Rollback Plan

If migration fails:
1. Git revert the conversion commits
2. Return to `#include` pattern
3. Document issues for future attempt
4. Wait for Clang 22/23 with better support

---

**Next Steps:**
1. Execute Phase 1 (precompile std module)
2. Test with one simple module (types.cppm)
3. If successful, proceed to Phase 2
4. Batch convert in Phase 3

**Status:** Ready to start when build system is stable.
