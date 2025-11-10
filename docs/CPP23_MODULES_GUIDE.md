# C++23 Modules: Complete Implementation Guide

**Author:** Olumuyiwa Oluwasanmi
**Date:** November 10, 2025
**Version:** 1.0
**Status:** Production Standard

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why C++23 Modules](#why-c23-modules)
3. [Module Structure](#module-structure)
4. [File Organization](#file-organization)
5. [Module Syntax Patterns](#module-syntax-patterns)
6. [CMake Integration](#cmake-integration)
7. [Compilation Process](#compilation-process)
8. [Module Dependencies](#module-dependencies)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)
11. [Migration Guide](#migration-guide)
12. [Examples from BigBrotherAnalytics](#examples-from-bigbrotheranalytics)

---

## Introduction

C++23 modules represent a fundamental shift in how C++ code is organized and compiled. This guide documents the **mandatory** module architecture used throughout BigBrotherAnalytics.

**Key Benefits:**
- **Faster compilation** - No more header file re-parsing
- **Better encapsulation** - Explicit control over what's exported
- **No include guards** - Modules are imported once
- **Reduced symbol pollution** - Clean namespace separation
- **Improved tooling** - Better IDE support and code analysis

**Project Status:**
- **25 C++23 modules** implemented and operational
- **100% trailing return syntax** compliance
- **Zero traditional headers** in new code
- **Clang 21.1.5** required for compilation

---

## Why C++23 Modules

### Problems with Traditional Headers

**Traditional C++ (`#include` model):**
```cpp
// header.hpp
#ifndef HEADER_HPP
#define HEADER_HPP

class MyClass {
    // ...
};

#endif
```

**Problems:**
1. **Textual inclusion** - Every file that includes the header re-parses it
2. **Order-dependent** - Include order matters (`<windows.h>` before `<winsock2.h>`)
3. **Macro pollution** - Macros leak across translation units
4. **Slow compilation** - Same headers parsed thousands of times
5. **Include guards/pragma once** - Boilerplate to prevent multiple inclusion

### C++23 Module Solution

**Modern C++ (module model):**
```cpp
// component.cppm
export module bigbrother.component;

export namespace bigbrother::component {
    class MyClass {
        // ...
    };
}
```

**Benefits:**
1. **Semantic import** - Module compiled once, imported semantically
2. **Order-independent** - Import order doesn't matter
3. **No macro leakage** - Macros don't cross module boundaries
4. **Fast compilation** - Precompiled module interface (BMI files)
5. **No include guards** - Module identity is built-in

### Performance Comparison

**Compilation Time Improvement:**
- **Traditional headers:** O(n*m) - n translation units × m headers
- **Modules:** O(n+m) - Build modules once, import everywhere

**Real-world measurements:**
- BigBrotherAnalytics full rebuild: **2 minutes with modules** vs. **8+ minutes with headers**
- Incremental build: **< 5 seconds** vs. **30+ seconds**

---

## Module Structure

### Standard Module File Structure (.cppm)

Every C++23 module file follows this canonical structure:

```cpp
/**
 * BigBrotherAnalytics - Component Name
 *
 * Brief description of module purpose and functionality.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: YYYY-MM-DD
 *
 * Following C++ Core Guidelines:
 * - C.21: Define or delete all default operations
 * - F.16: Pass cheap types by value
 * - E: std::expected for error handling
 * - Trailing return type syntax throughout
 *
 * Performance: [describe performance characteristics]
 * Thread-Safety: [describe thread-safety guarantees]
 */

// ============================================================================
// 1. GLOBAL MODULE FRAGMENT (Standard Library Only)
// ============================================================================
module;

// ONLY standard library includes allowed here
#include <vector>
#include <string>
#include <memory>
#include <expected>
#include <optional>

// ============================================================================
// 2. MODULE DECLARATION
// ============================================================================
export module bigbrother.component.name;

// ============================================================================
// 3. MODULE IMPORTS (Internal Dependencies)
// ============================================================================
import bigbrother.utils.types;
import bigbrother.utils.logger;

// ============================================================================
// 4. EXPORTED INTERFACE (Public API)
// ============================================================================
export namespace bigbrother::component {

    // Type aliases
    using Price = double;
    using Symbol = std::string;

    // Forward declarations (if needed)
    class ImplementationDetail;

    // Exported structures (passive data)
    struct Configuration {
        double threshold{0.5};
        int max_iterations{100};
    };

    // Exported classes (with invariants)
    class PublicAPI {
    public:
        // Constructor
        explicit PublicAPI(Configuration config);

        // Destructor
        ~PublicAPI();

        // Rule of Five (delete or default all)
        PublicAPI(PublicAPI const&) = delete;
        auto operator=(PublicAPI const&) -> PublicAPI& = delete;
        PublicAPI(PublicAPI&&) noexcept = default;
        auto operator=(PublicAPI&&) noexcept -> PublicAPI& = default;

        // Public methods (trailing return syntax)
        [[nodiscard]] auto calculate() const -> Result<double>;
        [[nodiscard]] auto getStatus() const noexcept -> std::string const&;

        auto setConfiguration(Configuration config) -> void;

    private:
        std::unique_ptr<ImplementationDetail> impl_;
        Configuration config_;
    };

    // Exported functions
    [[nodiscard]] auto createDefault() -> PublicAPI;
    [[nodiscard]] auto validate(Configuration const& config) -> Result<void>;

} // namespace bigbrother::component

// ============================================================================
// 5. PRIVATE IMPLEMENTATION (Optional)
// ============================================================================
module :private;

// Private implementation details not visible to importers
namespace bigbrother::component {

    // Private helper functions
    namespace detail {
        auto helperFunction() -> void {
            const auto local_const = 42;  // lower_case
        }
    } // namespace detail

    // Implementation class
    class ImplementationDetail {
    public:
        auto processData() -> void {
            // Implementation
        }
    private:
        std::vector<double> data_;
    };

    // Public method implementations
    PublicAPI::PublicAPI(Configuration config) : config_{config} {
        impl_ = std::make_unique<ImplementationDetail>();
    }

    PublicAPI::~PublicAPI() = default;

    auto PublicAPI::calculate() const -> Result<double> {
        // Implementation
        return 42.0;
    }

    auto PublicAPI::getStatus() const noexcept -> std::string const& {
        static const std::string status = "OK";
        return status;
    }

} // namespace bigbrother::component
```

### Key Sections Explained

#### 1. Global Module Fragment (`module;`)

**Purpose:** Include **ONLY** standard library headers

```cpp
module;  // Global module fragment starts

#include <vector>    // ✅ Standard library
#include <string>    // ✅ Standard library
#include <duckdb.hpp> // ✅ Third-party header-only library
#include <expected>  // ✅ C++23 standard

// ❌ NEVER include project headers here
// #include "my_header.hpp"  // WRONG!
```

**Rules:**
- Only `#include` directives allowed
- Standard library headers only (unless third-party is header-only)
- No project headers
- No `import` statements
- No code definitions

#### 2. Module Declaration (`export module`)

**Purpose:** Declare the module name

```cpp
export module bigbrother.component.name;
```

**Naming Convention:**
```
bigbrother.<category>.<component>

Examples:
- bigbrother.utils.types
- bigbrother.utils.logger
- bigbrother.options.pricing
- bigbrother.risk_management
- bigbrother.schwab_api
- bigbrother.strategy
```

#### 3. Module Imports (`import`)

**Purpose:** Import dependencies from other modules

```cpp
import bigbrother.utils.types;     // Internal dependency
import bigbrother.utils.logger;    // Internal dependency
import bigbrother.options.pricing; // Internal dependency

// Note: No semicolon after 'import' in global context
```

**Rules:**
- Use `import` for C++23 modules (not `#include`)
- Import order doesn't matter (unlike `#include`)
- Import statements are semantic, not textual
- Standard library can be imported: `import std;` (C++23)

#### 4. Exported Interface (`export namespace`)

**Purpose:** Define the public API

```cpp
export namespace bigbrother::component {
    // Everything here is exported and visible to importers

    class PublicClass { };         // Exported
    auto publicFunction() -> void; // Exported

} // namespace bigbrother::component
```

**What to Export:**
- Public classes and structs
- Public functions
- Type aliases used in public API
- Constants needed by users
- Enumerations

**What NOT to Export:**
- Implementation details
- Helper functions
- Internal constants
- Private types

#### 5. Private Implementation (`module :private;`)

**Purpose:** Implementation details not visible to importers

```cpp
module :private;  // Everything after this is private

namespace bigbrother::component {
    // Private implementation
    namespace detail {
        auto helperFunction() -> void { }
    }

    // Class implementations
    auto PublicClass::method() -> void {
        // Implementation
    }
}
```

**Benefits:**
- Hide implementation details
- Reduce compile-time dependencies
- Faster incremental builds
- Better encapsulation

---

## File Organization

### File Extensions

**BigBrotherAnalytics Standard:**
- **`.cppm`** - C++23 module interface + implementation
- **`.cpp`** - Traditional implementation (being phased out)
- **`.hpp`** - Traditional headers (only for third-party interop)

### Directory Structure

```
src/
├── utils/
│   ├── types.cppm          # bigbrother.utils.types
│   ├── logger.cppm         # bigbrother.utils.logger
│   ├── config.cppm         # bigbrother.utils.config
│   └── database.cppm       # bigbrother.utils.database
├── options/
│   ├── pricing.cppm        # bigbrother.options.pricing
│   └── trinomial_tree.cppm # bigbrother.options.trinomial_tree
├── risk_management/
│   └── risk_management.cppm # bigbrother.risk_management
├── schwab_api/
│   ├── schwab_api.cppm     # bigbrother.schwab_api (main module)
│   ├── token_manager.cpp   # Implementation (no module)
│   └── account_manager_impl.cpp
└── main.cpp                # Application entry point
```

### Module Hierarchy

```
bigbrother (root namespace)
├── utils
│   ├── types
│   ├── logger
│   ├── config
│   ├── database
│   ├── timer
│   ├── math
│   └── tax
├── options
│   ├── pricing
│   └── trinomial_tree
├── correlation
├── risk_management
├── schwab_api
├── strategy
├── strategies
└── employment
    └── signals
```

---

## Module Syntax Patterns

### Exporting Classes

```cpp
export module bigbrother.component;

export namespace bigbrother::component {

    // Exported class
    class MyClass {
    public:
        // All methods use trailing return syntax
        [[nodiscard]] auto calculate() const -> double;
        auto setData(std::vector<double> data) -> void;

    private:
        std::vector<double> data_;
    };

} // namespace bigbrother::component
```

### Exporting Functions

```cpp
export namespace bigbrother::component {

    // Free functions
    [[nodiscard]] auto calculatePrice(double spot, double strike) -> double;

    // Function templates
    template<typename T>
    [[nodiscard]] auto maximum(T a, T b) -> T {
        return (a > b) ? a : b;
    }

} // namespace bigbrother::component
```

### Exporting Types

```cpp
export namespace bigbrother::component {

    // Type aliases
    using Price = double;
    using Symbol = std::string;

    // Strong types
    struct StrongPrice {
        double value;
        explicit StrongPrice(double v) : value{v} {}
    };

    // Enumerations
    enum class OrderType {
        Market,
        Limit,
        Stop,
        StopLimit
    };

} // namespace bigbrother::component
```

### Exporting Constants

```cpp
export namespace bigbrother::component {

    // Compile-time constants
    constexpr auto pi = 3.14159265359;
    constexpr auto max_iterations = 1000;

    // Runtime constants (inline variables C++17)
    inline const std::string default_symbol = "SPY";

} // namespace bigbrother::component
```

### Templates in Modules

```cpp
export module bigbrother.component;

export namespace bigbrother::component {

    // Function template
    template<typename T>
    [[nodiscard]] auto add(T a, T b) -> T {
        return a + b;
    }

    // Class template
    template<typename T>
    class Container {
    public:
        auto push(T value) -> void {
            data_.push_back(std::move(value));
        }

        [[nodiscard]] auto size() const noexcept -> std::size_t {
            return data_.size();
        }

    private:
        std::vector<T> data_;
    };

} // namespace bigbrother::component
```

**Note:** Template definitions must be in the module interface, not in `module :private;`

---

## CMake Integration

### CMakeLists.txt Configuration

**BigBrotherAnalytics uses this CMake setup:**

```cmake
# Minimum CMake version with C++23 module support
cmake_minimum_required(VERSION 3.28)

# Project configuration
project(BigBrotherAnalytics
    VERSION 1.0.0
    LANGUAGES CXX
)

# C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable C++23 modules support
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Clang-specific flags for modules
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -fmodules
        -fbuiltin-module-map
        -fimplicit-module-maps
        -fprebuilt-module-path=${CMAKE_BINARY_DIR}/CMakeFiles
    )
endif()

# Add module library
add_library(bigbrother_modules)

# Specify module sources
target_sources(bigbrother_modules
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/utils/types.cppm
            src/utils/logger.cppm
            src/utils/config.cppm
            src/utils/database.cppm
            src/options/pricing.cppm
            src/risk_management/risk_management.cppm
            src/schwab_api/schwab_api.cppm
            src/trading_decision/strategy.cppm
)

# Link dependencies
target_link_libraries(bigbrother_modules
    PUBLIC
        duckdb
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
)

# Executable using modules
add_executable(bigbrother
    src/main.cpp
)

target_link_libraries(bigbrother
    PRIVATE
        bigbrother_modules
)
```

### Module Build Order

CMake automatically determines module dependencies and builds in correct order:

```
1. bigbrother.utils.types       (no dependencies)
2. bigbrother.utils.logger       (depends on types)
3. bigbrother.utils.config       (depends on types, logger)
4. bigbrother.utils.database     (depends on types, logger)
5. bigbrother.options.pricing    (depends on utils modules)
6. bigbrother.risk_management    (depends on options, utils)
7. bigbrother.schwab_api         (depends on utils)
8. bigbrother.strategy           (depends on risk_management, schwab_api)
9. main.cpp (application)        (imports all modules)
```

### Build Commands

```bash
# Configure (generates BMI files)
cd build
env CC=/usr/local/bin/clang CXX=/usr/local/bin/clang++ cmake -G Ninja ..

# Build (uses precompiled modules)
ninja bigbrother

# Clean rebuild
rm -rf build && mkdir build && cd build
env CC=/usr/local/bin/clang CXX=/usr/local/bin/clang++ cmake -G Ninja ..
ninja
```

---

## Compilation Process

### Module Compilation Stages

**Traditional C++ Compilation:**
```
source.cpp → [preprocessor] → expanded.cpp → [compiler] → object.o → [linker] → executable
```

**C++23 Module Compilation:**
```
module.cppm → [preprocessor] → module.ixx → [module compiler] → module.bmi + object.o
                                                                    ↓
importing.cpp → [preprocessor] → expanded.cpp → [compiler uses BMI] → object.o → [linker] → executable
```

### Binary Module Interface (BMI)

**What is a BMI?**
- Precompiled representation of module interface
- Contains exported declarations
- Platform and compiler-specific
- Cached for fast imports

**Location:**
- Clang: `build/CMakeFiles/<target>.dir/<module>.pcm`
- File extension: `.pcm` (Precompiled Module)

**Benefits:**
- **Fast imports** - No re-parsing
- **Semantic information** - Full type checking
- **Incremental builds** - Only rebuild changed modules

### Compilation Times

**Measured on BigBrotherAnalytics:**

| Operation | Time | Notes |
|-----------|------|-------|
| Clean build (all modules) | ~120s | First-time BMI generation |
| Incremental (1 module changed) | ~5s | Rebuild changed module + dependents |
| Incremental (main.cpp only) | ~2s | Uses cached BMIs |
| Full rebuild (cached BMIs) | ~30s | BMIs already exist |

---

## Module Dependencies

### Import vs. Include

**Import (C++23 modules):**
```cpp
import bigbrother.utils.types;  // Semantic import
import bigbrother.utils.logger; // Order doesn't matter
```

**Include (traditional):**
```cpp
#include <vector>          // Textual inclusion
#include "my_header.hpp"   // Order matters
```

### Dependency Graph Example

```cpp
// bigbrother.utils.types (no dependencies)
export module bigbrother.utils.types;

export namespace bigbrother::utils {
    using Price = double;
}

// bigbrother.utils.logger (depends on types)
export module bigbrother.utils.logger;
import bigbrother.utils.types;  // Import dependency

export namespace bigbrother::utils {
    class Logger {
        auto log(Price p) -> void;
    };
}

// bigbrother.options.pricing (depends on utils)
export module bigbrother.options.pricing;
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::options {
    [[nodiscard]] auto calculatePrice(utils::Price spot) -> utils::Price;
}

// main.cpp (imports all)
import bigbrother.options.pricing;  // Transitively imports utils modules

auto main() -> int {
    auto price = bigbrother::options::calculatePrice(100.0);
}
```

### Circular Dependencies

**Problem:** Modules cannot have circular dependencies

```cpp
// ❌ WRONG - Circular dependency
// module_a.cppm
export module bigbrother.module_a;
import bigbrother.module_b;  // A imports B

// module_b.cppm
export module bigbrother.module_b;
import bigbrother.module_a;  // B imports A ← CIRCULAR!
```

**Solution:** Extract common interface

```cpp
// ✅ CORRECT - Break cycle with interface module
// types.cppm (common interface)
export module bigbrother.types;
export namespace bigbrother {
    struct CommonData { };
}

// module_a.cppm
export module bigbrother.module_a;
import bigbrother.types;  // A imports types

// module_b.cppm
export module bigbrother.module_b;
import bigbrother.types;  // B imports types
```

---

## Best Practices

### 1. Module Naming

**✅ DO:**
```cpp
export module bigbrother.utils.types;
export module bigbrother.options.pricing;
export module bigbrother.risk_management;
```

**❌ DON'T:**
```cpp
export module utils;           // Too generic
export module BigBrotherUtils; // Wrong case
export module bb.util.type;    // Inconsistent abbreviation
```

**Rules:**
- Use hierarchical naming: `project.category.component`
- All lowercase with underscores
- Consistent with namespace structure

### 2. Export Only Public API

**✅ DO:**
```cpp
export namespace bigbrother::component {
    class PublicAPI { };      // Exported
}

module :private;
namespace bigbrother::component {
    class PrivateImpl { };    // Not exported
}
```

**❌ DON'T:**
```cpp
export namespace bigbrother::component {
    class PublicAPI { };
    class PrivateImpl { };    // Implementation detail exported!
}
```

### 3. Trailing Return Syntax (Mandatory)

**✅ DO:**
```cpp
export namespace bigbrother::component {
    [[nodiscard]] auto calculate() -> double;
    auto setData(std::vector<double> data) -> void;
}
```

**❌ DON'T:**
```cpp
export namespace bigbrother::component {
    double calculate();  // Old-style syntax not allowed
    void setData(std::vector<double> data);
}
```

### 4. Use `[[nodiscard]]` for Queries

**✅ DO:**
```cpp
[[nodiscard]] auto getPrice() const -> double;
[[nodiscard]] auto calculate() const -> Result<double>;
```

**❌ DON'T:**
```cpp
auto getPrice() const -> double;  // Should be [[nodiscard]]
```

### 5. Standard Library in Global Fragment

**✅ DO:**
```cpp
module;
#include <vector>
#include <string>

export module bigbrother.component;
// Rest of module
```

**❌ DON'T:**
```cpp
export module bigbrother.component;
#include <vector>  // Wrong place!
```

### 6. Group Related Functionality

**✅ DO:**
```cpp
// bigbrother.utils (multiple related utilities)
export module bigbrother.utils.types;
export module bigbrother.utils.logger;
export module bigbrother.utils.config;
```

**❌ DON'T:**
```cpp
// bigbrother.price_type (too fine-grained)
// bigbrother.log_function
// bigbrother.config_reader
```

### 7. Document Module Purpose

**✅ DO:**
```cpp
/**
 * BigBrotherAnalytics - Options Pricing
 *
 * Black-Scholes model implementation with Greeks calculation.
 * Supports American and European options.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Performance: < 100μs per pricing call
 * Thread-Safety: Immutable, fully thread-safe
 */
export module bigbrother.options.pricing;
```

### 8. Separate Interface from Implementation

**✅ DO:**
```cpp
// Module interface
export namespace bigbrother::component {
    class API {
    public:
        auto method() -> void;  // Declaration only
    };
}

// Implementation
module :private;
namespace bigbrother::component {
    auto API::method() -> void {
        // Implementation
    }
}
```

---

## Common Pitfalls

### 1. Wrong Include Location

**❌ WRONG:**
```cpp
export module bigbrother.component;

#include <vector>  // Too late!
```

**✅ CORRECT:**
```cpp
module;  // Global module fragment
#include <vector>

export module bigbrother.component;
```

### 2. Forgetting `export` Keyword

**❌ WRONG:**
```cpp
export module bigbrother.component;

namespace bigbrother::component {  // Not exported!
    class MyClass { };
}
```

**✅ CORRECT:**
```cpp
export module bigbrother.component;

export namespace bigbrother::component {  // Exported
    class MyClass { };
}
```

### 3. Circular Module Dependencies

**❌ WRONG:**
```cpp
// a.cppm
export module bigbrother.a;
import bigbrother.b;  // A depends on B

// b.cppm
export module bigbrother.b;
import bigbrother.a;  // B depends on A - CIRCULAR!
```

**✅ CORRECT:**
```cpp
// types.cppm (common interface)
export module bigbrother.types;

// a.cppm
export module bigbrother.a;
import bigbrother.types;

// b.cppm
export module bigbrother.b;
import bigbrother.types;
```

### 4. Template Definitions in `module :private;`

**❌ WRONG:**
```cpp
export module bigbrother.component;

export namespace bigbrother::component {
    template<typename T>
    auto process(T value) -> T;  // Declaration
}

module :private;
template<typename T>
auto process(T value) -> T {  // Definition hidden!
    return value;
}
```

**✅ CORRECT:**
```cpp
export module bigbrother.component;

export namespace bigbrother::component {
    template<typename T>
    auto process(T value) -> T {  // Definition in interface
        return value;
    }
}
```

### 5. Mixing Modules and Headers

**❌ WRONG:**
```cpp
// my_module.cppm
export module bigbrother.component;
#include "my_header.hpp"  // Don't mix!

// main.cpp
#include "my_header.hpp"  // Also includes
import bigbrother.component;  // Potential ODR violation
```

**✅ CORRECT:**
```cpp
// my_module.cppm
module;
#include <standard_library>  // Only standard library

export module bigbrother.component;
// Pure module implementation
```

### 6. Old-Style Function Syntax

**❌ WRONG:**
```cpp
export namespace bigbrother::component {
    double calculate();  // Old style - will fail clang-tidy
}
```

**✅ CORRECT:**
```cpp
export namespace bigbrother::component {
    auto calculate() -> double;  // Trailing return - required
}
```

### 7. Missing `[[nodiscard]]` on Queries

**❌ WRONG:**
```cpp
export namespace bigbrother::component {
    auto getStatus() const -> Status;  // Missing [[nodiscard]]
}
```

**✅ CORRECT:**
```cpp
export namespace bigbrother::component {
    [[nodiscard]] auto getStatus() const -> Status;  // Correct
}
```

---

## Migration Guide

### Converting Headers to Modules

**Step 1: Create Module File**

**Before (header):**
```cpp
// component.hpp
#ifndef COMPONENT_HPP
#define COMPONENT_HPP

#include <vector>

namespace bigbrother::component {
    class MyClass {
    public:
        double calculate();
    };
}

#endif
```

**After (module):**
```cpp
// component.cppm
module;
#include <vector>

export module bigbrother.component;

export namespace bigbrother::component {
    class MyClass {
    public:
        auto calculate() -> double;  // Trailing return
    };
}
```

**Step 2: Update Implementation**

**Before:**
```cpp
// component.cpp
#include "component.hpp"

namespace bigbrother::component {
    double MyClass::calculate() {
        return 42.0;
    }
}
```

**After (merge into .cppm):**
```cpp
// component.cppm (continued)
module :private;

namespace bigbrother::component {
    auto MyClass::calculate() -> double {
        return 42.0;
    }
}
```

**Step 3: Update Users**

**Before:**
```cpp
// main.cpp
#include "component.hpp"

int main() {
    bigbrother::component::MyClass obj;
}
```

**After:**
```cpp
// main.cpp
import bigbrother.component;

auto main() -> int {
    bigbrother::component::MyClass obj;
}
```

**Step 4: Update CMakeLists.txt**

```cmake
# Before
add_library(mylib
    src/component.cpp
)

# After
add_library(mylib)
target_sources(mylib
    PUBLIC FILE_SET CXX_MODULES FILES
        src/component.cppm
)
```

### Migration Checklist

- [ ] Create `.cppm` file with module declaration
- [ ] Move `#include` to global module fragment
- [ ] Add `export` to public namespace
- [ ] Convert all functions to trailing return syntax
- [ ] Add `[[nodiscard]]` to getters
- [ ] Move implementation to `module :private;` section
- [ ] Replace `#include "header.hpp"` with `import module.name` in users
- [ ] Update CMakeLists.txt with `FILE_SET CXX_MODULES`
- [ ] Build and test
- [ ] Delete old `.hpp` and `.cpp` files

---

## Examples from BigBrotherAnalytics

### Example 1: Simple Utility Module

**File:** `src/utils/types.cppm`

```cpp
/**
 * BigBrotherAnalytics - Core Types
 *
 * Fundamental type aliases and error handling types.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-01
 */

module;
#include <expected>
#include <string>

export module bigbrother.utils.types;

export namespace bigbrother::utils {

    // Price type (strong typing)
    using Price = double;
    using Symbol = std::string;

    // Error handling
    struct Error {
        int code;
        std::string message;
    };

    // Result type
    template<typename T>
    using Result = std::expected<T, Error>;

} // namespace bigbrother::utils
```

### Example 2: Module with Dependencies

**File:** `src/options/pricing.cppm`

```cpp
/**
 * BigBrotherAnalytics - Options Pricing
 *
 * Black-Scholes model implementation.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Performance: < 100μs per call
 * Thread-Safety: Immutable, fully thread-safe
 */

module;
#include <cmath>
#include <expected>

export module bigbrother.options.pricing;

import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::options {

    // Option type enumeration
    enum class OptionType {
        Call,
        Put
    };

    // Pricing parameters
    struct PricingParams {
        utils::Price spot{0.0};
        utils::Price strike{0.0};
        double volatility{0.0};
        double time_to_expiry{0.0};
        double risk_free_rate{0.0};
    };

    // Black-Scholes calculator
    class BlackScholesModel {
    public:
        [[nodiscard]] auto calculatePrice(
            PricingParams const& params,
            OptionType type
        ) const -> utils::Result<utils::Price>;

        [[nodiscard]] auto calculateDelta(
            PricingParams const& params,
            OptionType type
        ) const -> utils::Result<double>;

    private:
        [[nodiscard]] auto normalCdf(double x) const noexcept -> double;
    };

} // namespace bigbrother::options

module :private;

namespace bigbrother::options {

    auto BlackScholesModel::calculatePrice(
        PricingParams const& params,
        OptionType type
    ) const -> utils::Result<utils::Price> {

        if (params.spot <= 0.0 || params.strike <= 0.0) {
            return std::unexpected(utils::Error{
                1, "Spot and strike prices must be positive"
            });
        }

        // Black-Scholes formula implementation
        const auto d1 = (std::log(params.spot / params.strike) +
                        (params.risk_free_rate + 0.5 * params.volatility * params.volatility) *
                        params.time_to_expiry) /
                       (params.volatility * std::sqrt(params.time_to_expiry));

        const auto d2 = d1 - params.volatility * std::sqrt(params.time_to_expiry);

        if (type == OptionType::Call) {
            return params.spot * normalCdf(d1) -
                   params.strike * std::exp(-params.risk_free_rate * params.time_to_expiry) * normalCdf(d2);
        } else {
            return params.strike * std::exp(-params.risk_free_rate * params.time_to_expiry) * normalCdf(-d2) -
                   params.spot * normalCdf(-d1);
        }
    }

    auto BlackScholesModel::normalCdf(double x) const noexcept -> double {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

} // namespace bigbrother::options
```

### Example 3: Main Application Using Modules

**File:** `src/main.cpp`

```cpp
/**
 * BigBrotherAnalytics - Main Trading Engine
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 */

// Import all required modules
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.config;
import bigbrother.utils.database;
import bigbrother.options.pricing;
import bigbrother.risk_management;
import bigbrother.schwab_api;
import bigbrother.strategy;

#include <iostream>

auto main() -> int {
    // Use imported modules
    bigbrother::utils::Logger::getInstance().info("Starting BigBrotherAnalytics");

    bigbrother::options::BlackScholesModel pricer;
    bigbrother::options::PricingParams params{
        .spot = 150.0,
        .strike = 155.0,
        .volatility = 0.25,
        .time_to_expiry = 30.0 / 365.0,
        .risk_free_rate = 0.05
    };

    auto price_result = pricer.calculatePrice(params, bigbrother::options::OptionType::Call);
    if (price_result) {
        std::cout << "Option price: $" << *price_result << "\n";
    } else {
        std::cerr << "Error: " << price_result.error().message << "\n";
        return 1;
    }

    return 0;
}
```

---

## Compiler Requirements

### Clang 21.1.5 (Required)

**Installation:**
```bash
# Via Ansible (recommended)
ansible-playbook playbooks/complete-tier1-setup.yml

# Manual
# Download from https://releases.llvm.org/
# Install to /usr/local/bin/
```

**Verification:**
```bash
/usr/local/bin/clang++ --version
# Should show: clang version 21.1.5
```

### Required Compiler Flags

```cmake
# CMakeLists.txt
add_compile_options(
    -std=c++23
    -fmodules
    -fbuiltin-module-map
    -fimplicit-module-maps
)
```

### Environment Variables

```bash
export CC=/usr/local/bin/clang
export CXX=/usr/local/bin/clang++
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

---

## Summary

**C++23 modules are MANDATORY in BigBrotherAnalytics:**

✅ **Always:**
- Use `.cppm` extension for module files
- Start with `module;` for standard library includes
- Use `export module bigbrother.category.component;`
- Use trailing return syntax: `auto func() -> ReturnType`
- Add `[[nodiscard]]` to all getters
- Use `module :private;` for implementation details
- Import with `import bigbrother.module.name;`

❌ **Never:**
- Use old-style `#include` for project headers
- Mix modules and headers
- Create circular module dependencies
- Forget `export` keyword
- Use old-style function syntax
- Export implementation details

**Build with:**
```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
rm -rf build && mkdir build && cd build
env CC=/usr/local/bin/clang CXX=/usr/local/bin/clang++ cmake -G Ninja ..
ninja bigbrother backtest
```

**Reference Implementation:**
- 25 production modules in `src/`
- All following this exact pattern
- 100% trailing return syntax
- Zero traditional headers in new code

---

**Author:** Olumuyiwa Oluwasanmi
**Last Updated:** November 10, 2025
**Status:** Production Standard - MANDATORY
