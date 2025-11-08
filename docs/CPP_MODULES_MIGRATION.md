# C++23 Modules Migration Plan

## Overview

Migrating from traditional headers to C++23 modules for **2-10x faster compilation** and better dependency management.

**Current Status:**
- ✅ Clang 21.1.5 with full C++23 modules support installed
- ⏳ One module file exists: `src/utils/utils.cppm`
- ⏳ Most code still using traditional headers

**Benefits of Modules:**
- **Faster compilation:** 2-10x speedup (only parse module interface once)
- **Better isolation:** No macro leakage, cleaner boundaries
- **Faster incremental builds:** Module interface changes don't force full rebuild
- **Better tooling:** IDEs can provide faster autocomplete

---

## Migration Strategy

### Phase 1: Core Utilities (Week 1)

**Convert to Modules:**
```
src/utils/types.hpp          → src/utils/types.cppm
src/utils/logger.hpp         → src/utils/logger.cppm
src/utils/timer.hpp          → src/utils/timer.cppm
src/utils/config.hpp         → src/utils/config.cppm
src/utils/database.hpp       → src/utils/database.cppm
```

**Module Structure:**
```cpp
// src/utils/types.cppm
export module bigbrother.utils.types;

import <string>;
import <vector>;
import <variant>;
import <chrono>;

export namespace bigbrother::types {
    using Price = double;
    using Quantity = int64_t;
    using Timestamp = std::chrono::system_clock::time_point;
    // ... rest of types
}
```

### Phase 2: Correlation Engine (Week 2)

```
src/correlation_engine/correlation.hpp     → correlation.cppm
src/correlation_engine/options_pricing.hpp → options_pricing.cppm
src/correlation_engine/greeks.hpp          → greeks.cppm
```

**Module with Dependencies:**
```cpp
// src/correlation_engine/options_pricing.cppm
export module bigbrother.correlation.options_pricing;

import bigbrother.utils.types;  // Import our module
import <vector>;
import <memory>;

export namespace bigbrother::correlation {
    class BlackScholesModel {
        // ... implementation
    };
}
```

### Phase 3: Trading Decision Engine (Week 3)

```
src/trading_decision/strategy.hpp  → strategy.cppm
src/trading_decision/...           → *.cppm
```

### Phase 4: Complete Migration (Week 4)

```
All remaining .hpp files → .cppm modules
```

---

## CMakeLists.txt Changes

### Enable C++23 Modules

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)  # Modules support
project(BigBrotherAnalytics VERSION 1.0.0 LANGUAGES CXX)

# C++23 with modules
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable modules
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Compiler flags for modules
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fmodules-ts -fprebuilt-module-path=${CMAKE_BINARY_DIR}/modules)
endif()

# Module library
add_library(utils_module)
target_sources(utils_module
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/utils/types.cppm
            src/utils/logger.cppm
            src/utils/timer.cppm
            src/utils/config.cppm
            src/utils/database.cppm
)
```

---

## Example Module Conversion

### Before (Header):

```cpp
// src/utils/types.hpp
#pragma once

#include <string>
#include <vector>
#include <chrono>

namespace bigbrother::types {
    using Price = double;
    using Quantity = int64_t;
    using Timestamp = std::chrono::system_clock::time_point;
}
```

### After (Module):

```cpp
// src/utils/types.cppm
export module bigbrother.utils.types;

import <string>;
import <vector>;
import <chrono>;

export namespace bigbrother::types {
    using Price = double;
    using Quantity = int64_t;
    using Timestamp = std::chrono::system_clock::time_point;
}
```

### Usage:

```cpp
// Old way (headers):
#include "utils/types.hpp"
#include "utils/logger.hpp"

// New way (modules):
import bigbrother.utils.types;
import bigbrother.utils.logger;
```

---

## Module Naming Convention

```
Namespace:           bigbrother::utils::logger
Module name:         bigbrother.utils.logger
File name:           src/utils/logger.cppm
Import statement:    import bigbrother.utils.logger;
```

**Pattern:** Replace `::` with `.` and use `.cppm` extension

---

## Standard Library Modules

Clang 21 supports standard library modules:

```cpp
// Instead of:
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Use:
import std;  // Import entire standard library
// or
import std.vector;
import std.string;
import std.memory;
```

**Note:** As of Clang 21, `import std;` is experimental. Better to use individual headers with `import <header>;` syntax.

---

## Build System Updates

### CMake Module Support

```cmake
# Create module library for each component

# Utils modules
add_library(utils_modules)
target_sources(utils_modules
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/utils/types.cppm
            src/utils/logger.cppm
            src/utils/timer.cppm
            src/utils/config.cppm
)

# Correlation engine modules
add_library(correlation_modules)
target_sources(correlation_modules
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/correlation_engine/correlation.cppm
            src/correlation_engine/options_pricing.cppm
)
target_link_libraries(correlation_modules PRIVATE utils_modules)

# Main library depends on modules
add_library(bigbrother_lib
    src/main.cpp
    src/trading_engine.cpp
)
target_link_libraries(bigbrother_lib
    PRIVATE
        utils_modules
        correlation_modules
)
```

---

## Compilation Performance Comparison

### Without Modules (Current):
```
Clean build:           ~2-3 minutes (with Clang 21)
Incremental build:     ~30-60 seconds (change in header forces recompilation)
Header parsing:        Repeated for each translation unit
```

### With Modules (Target):
```
Clean build:           ~1-2 minutes (faster, modules precompiled)
Incremental build:     ~5-15 seconds (only changed modules recompile)
Module parsing:        Once per module (not per TU)
Estimated speedup:     3-5x for clean, 5-10x for incremental
```

---

## Migration Checklist

### Tier 1 Implementation (Immediate)

- [ ] Update CMakeLists.txt for module support (CMAKE_CXX_SCAN_FOR_MODULES)
- [ ] Convert core utilities to modules:
  - [ ] `types.hpp` → `types.cppm`
  - [ ] `logger.hpp` → `logger.cppm`
  - [ ] `timer.hpp` → `timer.cppm`
  - [ ] `config.hpp` → `config.cppm`
- [ ] Test module compilation with Clang 21
- [ ] Measure build time improvements
- [ ] Document module usage in TIER1_IMPLEMENTATION_TASKS.md

### Tier 2 (After Tier 1 POC Proven)

- [ ] Convert correlation engine to modules
- [ ] Convert trading decision engine to modules
- [ ] Convert risk management to modules
- [ ] Full migration of all headers
- [ ] Setup module precompilation cache
- [ ] CI/CD optimization with module cache

---

## Example: Logger Module

```cpp
// src/utils/logger.cppm
export module bigbrother.utils.logger;

import bigbrother.utils.types;
import <string>;
import <memory>;
import <format>;

// Forward declare spdlog (don't expose in module interface)
namespace spdlog {
    class logger;
}

export namespace bigbrother::utils {

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static auto getInstance() -> Logger&;

    auto setLevel(LogLevel level) -> void;

    template<typename... Args>
    auto info(std::string_view fmt, Args&&... args) -> void {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    auto error(std::string_view fmt, Args&&... args) -> void {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
    }

    // ... other methods

private:
    Logger();
    ~Logger();

    template<typename... Args>
    auto log(LogLevel level, std::string_view fmt, Args&&... args) -> void;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace bigbrother::utils
```

---

## Current Clang 21 Module Support

**Clang 21.1.5 Capabilities:**
- ✅ Full C++20 modules support
- ✅ C++23 modules features
- ✅ Standard library header units (`import <vector>;`)
- ✅ Named modules (`export module name;`)
- ✅ Module partitions
- ⚠️ `import std;` - experimental, prefer header units

**Recommended Approach for Tier 1:**
1. Use header units for standard library: `import <vector>;`
2. Create named modules for our code: `export module bigbrother.utils;`
3. Keep implementation files as `.cpp`
4. Module interface files as `.cppm`

---

## Quick Start: Enable Modules Now

### 1. Update CMakeLists.txt

```cmake
# Require CMake 3.28+ for module support
cmake_minimum_required(VERSION 3.28)

# Enable module scanning
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Clang-specific module flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -fmodules
        -fbuiltin-module-map
        -fimplicit-module-maps
    )
endif()
```

### 2. Create First Module (Logger)

```bash
# Convert logger.hpp to logger.cppm
mv src/utils/logger.hpp src/utils/logger.cppm.new
# Edit to add export module declaration
# Update CMakeLists.txt to build as module
```

### 3. Update Consumers

```cpp
// Old:
#include "utils/logger.hpp"

// New:
import bigbrother.utils.logger;
```

---

## Estimated Impact

**Current Build Time:**
- Clean: ~120 seconds
- Incremental (1 header change): ~45 seconds

**After Full Module Migration:**
- Clean: ~30-50 seconds (2-4x faster)
- Incremental (1 module change): ~5-10 seconds (5-10x faster)

**Developer Productivity:**
- Faster iteration cycles
- Less waiting for compilation
- Better IDE performance

---

## Risks & Mitigation

**Risks:**
1. **Module support still maturing** - Some third-party libraries don't support modules
   - **Mitigation:** Keep header compatibility, gradual migration

2. **Build system complexity** - CMake module support is new
   - **Mitigation:** Use CMake 3.28+, test thoroughly

3. **Debugger support** - Some debuggers have issues with modules
   - **Mitigation:** Clang 21 has good debug info for modules

**Recommendation:**
- Start migration in Tier 1
- Convert utils first (low risk, high benefit)
- Keep headers as fallback during transition
- Full migration in Tier 2 after validation

---

## Action Items for Next Session

**Immediate (Next 1-2 hours):**
1. Update CMakeLists.txt to enable module scanning
2. Convert logger.hpp to logger.cppm
3. Update 2-3 files to import logger module
4. Test compilation with modules
5. Measure build time improvement

**Tier 1 (Next 2-4 weeks):**
1. Convert all utils/*.hpp to modules
2. Convert correlation engine to modules
3. Document module usage patterns
4. Update all consumers to use import

**Tier 2 (After POC):**
1. Complete migration of all components
2. Optimize module precompilation
3. Setup CI/CD module caching
4. Benchmark full build pipeline

---

This migration will significantly improve development velocity during the 16-week Tier 1 implementation phase.
