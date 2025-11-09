# GoogleTest libc++ Build Guide

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-09
**Version:** 1.0.0

---

## Overview

This document explains why GoogleTest needed to be rebuilt with libc++ and provides exact instructions for reproducing the build. This is critical for maintaining ABI compatibility with BigBrotherAnalytics' C++23 module-based architecture.

---

## Why GoogleTest Needed Rebuilding

### The Problem

BigBrotherAnalytics uses:
- **Clang 21** as the compiler (`/usr/local/bin/clang++`)
- **libc++** (LLVM's C++ standard library) instead of libstdc++
- **C++23 modules** with `-stdlib=libc++` flag

The system-installed GoogleTest from `libgtest-dev` was compiled with:
- GCC's libstdc++
- Different ABI (Application Binary Interface)

### ABI Incompatibility Issues

When linking test executables that use:
1. Our libraries (compiled with Clang 21 + libc++)
2. System GoogleTest (compiled with GCC + libstdc++)

This causes:
- **Symbol conflicts** between libstdc++ and libc++
- **Undefined behavior** due to different standard library implementations
- **Linker errors** from incompatible vtables and RTTI
- **Runtime crashes** from mixing C++ runtime libraries

### The Solution

Rebuild GoogleTest from source using the **exact same toolchain**:
- Same compiler: Clang 21
- Same standard library: libc++
- Same C++ standard: C++23
- Same flags: `-stdlib=libc++`

This ensures **ABI compatibility** across the entire codebase.

---

## Build Instructions

### Prerequisites

Ensure you have:
- Clang 21 installed at `/usr/local/bin/clang` and `/usr/local/bin/clang++`
- libc++ development files installed
- CMake 3.28 or newer
- Git

**Verify Clang installation:**
```bash
/usr/local/bin/clang++ --version
# Should output: clang version 21.1.5 or similar
```

### Step 1: Clone GoogleTest v1.15.2

```bash
cd /tmp
git clone --branch v1.15.2 --depth 1 https://github.com/google/googletest.git
cd googletest
```

**Why v1.15.2?**
- Latest stable release as of November 2024
- Full C++23 support
- Modern CMake configuration
- Well-tested with libc++

### Step 2: Configure with Clang 21 and libc++

```bash
mkdir build
cd build

cmake .. \
  -DCMAKE_C_COMPILER=/usr/local/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
  -DCMAKE_CXX_STANDARD=23 \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
  -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
```

**Flag Explanation:**
- `-DCMAKE_C_COMPILER` / `-DCMAKE_CXX_COMPILER`: Use Clang 21
- `-DCMAKE_CXX_STANDARD=23`: Use C++23 (same as our project)
- `-DCMAKE_CXX_FLAGS="-stdlib=libc++"`: Use libc++ instead of libstdc++
- `-DCMAKE_EXE_LINKER_FLAGS`: Link against libc++ and libc++abi
- `-DCMAKE_SHARED_LINKER_FLAGS`: Same for shared libraries
- `-DCMAKE_BUILD_TYPE=Release`: Optimized build
- `-DBUILD_SHARED_LIBS=OFF`: Build static libraries (easier linking)
- `-DCMAKE_INSTALL_PREFIX=/usr/local`: Install to standard location

### Step 3: Build GoogleTest

```bash
cmake --build . -j$(nproc)
```

**This builds:**
- `lib/libgtest.a` - Main GoogleTest library
- `lib/libgtest_main.a` - Main function for tests
- `lib/libgmock.a` - Google Mock library
- `lib/libgmock_main.a` - Main function for mock tests

**Expected output:**
```
[ 12%] Building CXX object googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
[ 25%] Linking CXX static library ../lib/libgtest.a
[ 37%] Building CXX object googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
[ 50%] Linking CXX static library ../lib/libgtest_main.a
...
[100%] Built target gmock_main
```

### Step 4: Install to /usr/local

```bash
sudo cmake --install .
```

**This installs:**
- **Libraries:** `/usr/local/lib/libgtest.a`, `/usr/local/lib/libgtest_main.a`
- **Headers:** `/usr/local/include/gtest/`
- **CMake files:** `/usr/local/lib/cmake/GTest/`

**Verify installation:**
```bash
ls -lh /usr/local/lib/libgtest*.a
# Expected output:
# -rw-r--r-- 1 root root 1.2M /usr/local/lib/libgtest.a
# -rw-r--r-- 1 root root  35K /usr/local/lib/libgtest_main.a
```

### Step 5: Clean Up

```bash
cd /tmp
rm -rf googletest
```

---

## Verify the Installation

### Check Library Symbols

Verify that GoogleTest was built with libc++:

```bash
nm /usr/local/lib/libgtest.a | grep "std::" | head -5
```

**Expected:** Symbols should reference libc++ implementation details, not libstdc++.

### Check for libc++ Dependencies

```bash
readelf -d /usr/local/lib/libgtest.a 2>/dev/null || echo "Static library - no dynamic dependencies"
```

**Expected:** Static libraries don't have runtime dependencies, which is good.

### Build a Simple Test

Create `/tmp/test_gtest.cpp`:
```cpp
#include <gtest/gtest.h>

TEST(SampleTest, BasicAssertion) {
    EXPECT_EQ(1 + 1, 2);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

Compile and run:
```bash
/usr/local/bin/clang++ \
  -std=c++23 \
  -stdlib=libc++ \
  /tmp/test_gtest.cpp \
  -I/usr/local/include \
  -L/usr/local/lib \
  -lgtest \
  -lc++abi \
  -pthread \
  -o /tmp/test_gtest

/tmp/test_gtest
```

**Expected output:**
```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from SampleTest
[ RUN      ] SampleTest.BasicAssertion
[       OK ] SampleTest.BasicAssertion (0 ms)
[----------] 1 test from SampleTest (0 ms total)

[==========] 1 test from 1 test suite ran. (0 ms total)
[  PASSED  ] 1 test.
```

---

## CMakeLists.txt Integration

### How tests/cpp/CMakeLists.txt Was Updated

The test configuration was modified to use the custom-built GoogleTest:

**File:** `/home/muyiwa/Development/BigBrotherAnalytics/tests/cpp/CMakeLists.txt`

```cmake
# Use GoogleTest built with libc++ from /usr/local
set(GTEST_ROOT "/usr/local")
set(GTEST_LIBRARY "/usr/local/lib/libgtest.a")
set(GTEST_MAIN_LIBRARY "/usr/local/lib/libgtest_main.a")
set(GTEST_INCLUDE_DIRS "/usr/include/gtest")

# Mark as found
set(GTest_FOUND TRUE)

# Create imported targets (only if not already exists)
if(NOT TARGET GTest::GTest)
    add_library(GTest::GTest STATIC IMPORTED)
    set_target_properties(GTest::GTest PROPERTIES
        IMPORTED_LOCATION "${GTEST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
    )
endif()

if(NOT TARGET GTest::Main)
    add_library(GTest::Main STATIC IMPORTED)
    set_target_properties(GTest::Main PROPERTIES
        IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "GTest::GTest"
    )
endif()
```

**Why these paths?**
- **Library paths:** Point to `/usr/local/lib` where we installed the custom build
- **Include paths:** System headers at `/usr/include/gtest` (standard location)
- **STATIC IMPORTED:** Tells CMake these are pre-built static libraries
- **INTERFACE_LINK_LIBRARIES:** Ensures `GTest::Main` depends on `GTest::GTest`

### How Tests Link Against GoogleTest

Test targets link like this:

```cmake
add_executable(test_options_pricing
    test_black_scholes.cpp
    test_greeks.cpp
    test_implied_volatility.cpp
)

target_link_libraries(test_options_pricing
    PRIVATE
    options_pricing       # Our library (built with Clang + libc++)
    GTest::GTest          # GoogleTest (now also built with Clang + libc++)
    GTest::Main           # GoogleTest main (same)
)
```

**Result:** All components share the same ABI - no conflicts!

---

## Troubleshooting

### Issue: "undefined reference to `std::__throw_bad_alloc()@GLIBCXX_3.4'"

**Symptom:** Linker error mentioning GLIBCXX (libstdc++) symbols.

**Cause:** Mixing libstdc++ and libc++ libraries.

**Solution:**
1. Verify GoogleTest was built with libc++:
   ```bash
   nm /usr/local/lib/libgtest.a | grep GLIBCXX
   ```
   Should return **nothing**. If it shows GLIBCXX symbols, GoogleTest wasn't built with libc++.

2. Rebuild GoogleTest with correct flags (see Step 2).

### Issue: "cannot find -lgtest"

**Symptom:** CMake or linker can't find GoogleTest libraries.

**Cause:** Libraries not installed to expected location.

**Solution:**
1. Check if libraries exist:
   ```bash
   ls -la /usr/local/lib/libgtest*
   ```

2. Update `GTEST_LIBRARY` path in `tests/cpp/CMakeLists.txt` if installed elsewhere.

3. Update linker search path:
   ```bash
   sudo ldconfig /usr/local/lib
   ```

### Issue: Tests Compile But Crash at Runtime

**Symptom:** Segfault or "pure virtual method called" errors when running tests.

**Cause:** ABI mismatch between application and GoogleTest.

**Solution:**
1. Confirm all components use the same compiler:
   ```bash
   # Check main project
   grep CMAKE_CXX_COMPILER /home/muyiwa/Development/BigBrotherAnalytics/CMakeLists.txt
   # Should show: /usr/local/bin/clang++
   ```

2. Rebuild GoogleTest ensuring `-stdlib=libc++` is used.

3. Clean and rebuild everything:
   ```bash
   cd /home/muyiwa/Development/BigBrotherAnalytics/build
   rm -rf *
   cmake .. -G Ninja
   ninja
   ```

### Issue: "fatal error: 'gtest/gtest.h' file not found"

**Symptom:** Compiler can't find GoogleTest headers.

**Cause:** Headers not installed or wrong include path.

**Solution:**
1. Check header installation:
   ```bash
   ls -la /usr/local/include/gtest/
   ls -la /usr/include/gtest/
   ```

2. Update `GTEST_INCLUDE_DIRS` in `tests/cpp/CMakeLists.txt`:
   ```cmake
   set(GTEST_INCLUDE_DIRS "/usr/local/include/gtest")
   ```

### Issue: CMake Find Package Conflicts

**Symptom:** CMake finds system GoogleTest instead of custom build.

**Cause:** CMake's `find_package(GTest)` searching in default locations.

**Solution:**
Our `tests/cpp/CMakeLists.txt` manually sets:
```cmake
set(GTest_FOUND TRUE)
set(GTEST_ROOT "/usr/local")
```

This **overrides** automatic detection and forces use of our custom build.

---

## CI/CD Setup

### For GitHub Actions

Add to `.github/workflows/build.yml`:

```yaml
- name: Build GoogleTest with libc++
  run: |
    cd /tmp
    git clone --branch v1.15.2 --depth 1 https://github.com/google/googletest.git
    cd googletest
    mkdir build && cd build
    cmake .. \
      -DCMAKE_C_COMPILER=/usr/local/bin/clang \
      -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
      -DCMAKE_CXX_STANDARD=23 \
      -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
      -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
      -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build . -j$(nproc)
    sudo cmake --install .
```

### For Docker Builds

Add to `Dockerfile`:

```dockerfile
# Install Clang 21 first (see separate documentation)

# Build GoogleTest with libc++
RUN cd /tmp && \
    git clone --branch v1.15.2 --depth 1 https://github.com/google/googletest.git && \
    cd googletest && mkdir build && cd build && \
    cmake .. \
      -DCMAKE_C_COMPILER=/usr/local/bin/clang \
      -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
      -DCMAKE_CXX_STANDARD=23 \
      -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
      -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
      -DCMAKE_SHARED_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build . -j$(nproc) && \
    cmake --install . && \
    cd /tmp && rm -rf googletest
```

---

## Alternative: System Package with libc++ (Not Recommended)

Some distributions provide `libgtest-dev` variants built with libc++, but:

**Issues:**
- Not available on all systems
- Version may be outdated
- Build flags may not match our project
- Less control over configuration

**When acceptable:**
- Quick development on compatible system
- Non-production environments
- If exact ABI match is verified

**To use system package:**

1. Check if available:
   ```bash
   apt-cache search gtest | grep libc++
   ```

2. Install:
   ```bash
   sudo apt-get install libgtest-dev-libc++  # If it exists
   ```

3. Update `tests/cpp/CMakeLists.txt`:
   ```cmake
   find_package(GTest REQUIRED)
   ```

4. **Verify ABI compatibility** before proceeding!

---

## Summary

### What We Did

1. **Identified** ABI incompatibility between system GoogleTest (GCC + libstdc++) and our project (Clang + libc++)
2. **Rebuilt** GoogleTest v1.15.2 from source using Clang 21 with libc++
3. **Installed** to `/usr/local/lib` for system-wide availability
4. **Updated** `tests/cpp/CMakeLists.txt` to use custom-built GoogleTest
5. **Verified** tests compile and run without ABI conflicts

### Key Takeaways

- **ABI compatibility is critical** when mixing C++ libraries
- **Same toolchain** = same compiler + same standard library + same flags
- **Static linking** (`.a` libraries) avoids runtime dependency issues
- **Manual CMake configuration** gives precise control over library selection

### Maintenance

**When to rebuild GoogleTest:**
- Upgrading Clang version
- Changing C++ standard (e.g., C++23 to C++26)
- Modifying compiler flags in main CMakeLists.txt
- Updating to new GoogleTest version

**Monthly check:**
```bash
# Check if new GoogleTest version available
curl -s https://api.github.com/repos/google/googletest/releases/latest | grep tag_name
```

---

## References

- **GoogleTest Repository:** https://github.com/google/googletest
- **libc++ Documentation:** https://libcxx.llvm.org/
- **C++ ABI Compatibility:** https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
- **BigBrotherAnalytics Build Workflow:** `/home/muyiwa/Development/BigBrotherAnalytics/docs/BUILD_WORKFLOW.md`

---

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-09
**Version:** 1.0.0
**Last Updated:** 2025-11-09
