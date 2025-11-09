# Schwab API Build Script Flow Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    build_and_test_schwab.sh                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. PARSE ARGUMENTS                                             │
│     ├─ --clean      → FORCE_CLEAN=true                          │
│     ├─ --no-tests   → RUN_TESTS=false                           │
│     ├─ --verbose    → VERBOSE=true                              │
│     └─ --help       → Show help and exit                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. DISPLAY BUILD INFO                                          │
│     ├─ Project root directory                                   │
│     ├─ Build directory location                                 │
│     ├─ Compiler versions                                        │
│     └─ Build type                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SETUP ENVIRONMENT                                           │
│     ├─ export CC=/usr/local/bin/clang                           │
│     ├─ export CXX=/usr/local/bin/clang++                        │
│     ├─ export SKIP_CLANG_TIDY=1                                 │
│     ├─ export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH  │
│     └─ Verify compilers exist                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. CLEAN BUILD (if --clean)                                    │
│     └─ Remove build/ directory                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. CONFIGURE CMAKE                                             │
│     ├─ Create build directory                                   │
│     ├─ cd build                                                 │
│     └─ cmake .. -DCMAKE_BUILD_TYPE=Release \                    │
│              -DCMAKE_C_COMPILER=$CC \                           │
│              -DCMAKE_CXX_COMPILER=$CXX \                        │
│              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. BUILD TARGETS (sequential)                                  │
│     ┌───────────────────────────────────────────┐               │
│     │  Target 1: schwab_api                     │               │
│     │  cmake --build . --target schwab_api -j N │               │
│     └───────────────────────────────────────────┘               │
│                     │ Success                                   │
│                     ▼                                           │
│     ┌───────────────────────────────────────────┐               │
│     │  Target 2: test_order_manager_integration │               │
│     │  cmake --build . --target ... -j N        │               │
│     └───────────────────────────────────────────┘               │
│                     │ Success                                   │
│                     ▼                                           │
│     ┌───────────────────────────────────────────┐               │
│     │  Target 3: test_account_manager_...       │               │
│     │  cmake --build . --target ... -j N        │               │
│     └───────────────────────────────────────────┘               │
│                     │ Success                                   │
│                     ▼                                           │
│     ┌───────────────────────────────────────────┐               │
│     │  Target 4: test_schwab_e2e_workflow       │               │
│     │  cmake --build . --target ... -j N        │               │
│     └───────────────────────────────────────────┘               │
│                                                                 │
│     N = nproc (parallel jobs)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Build Success?  │
                    └─────────────────┘
                         /        \
                    Yes /          \ No
                       /            \
                      ▼              ▼
          ┌──────────────────┐   ┌────────────┐
          │  Continue        │   │  Exit 1    │
          └──────────────────┘   └────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. RUN TESTS (if not --no-tests)                               │
│     ┌───────────────────────────────────────────┐               │
│     │  Test 1: test_order_manager_integration   │               │
│     │  ./bin/test_order_manager_integration \   │               │
│     │      --gtest_color=yes                    │               │
│     └───────────────────────────────────────────┘               │
│                     │                                           │
│                     ▼                                           │
│     ┌───────────────────────────────────────────┐               │
│     │  Test 2: test_account_manager_integration │               │
│     │  ./bin/test_account_manager_integration \ │               │
│     │      --gtest_color=yes                    │               │
│     └───────────────────────────────────────────┘               │
│                     │                                           │
│                     ▼                                           │
│     ┌───────────────────────────────────────────┐               │
│     │  Test 3: test_schwab_e2e_workflow         │               │
│     │  ./bin/test_schwab_e2e_workflow \         │               │
│     │      --gtest_color=yes                    │               │
│     └───────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. DISPLAY TEST SUMMARY                                        │
│     ┌─────────────────────────────────────────┐                 │
│     │  Test Results:                          │                 │
│     │    ✓ test_order_manager_integration     │                 │
│     │    ✓ test_account_manager_integration   │                 │
│     │    ✗ test_schwab_e2e_workflow           │                 │
│     │                                         │                 │
│     │  Statistics:                            │                 │
│     │    Total Tests:  3                      │                 │
│     │    Passed:       2                      │                 │
│     │    Failed:       1                      │                 │
│     │                                         │                 │
│     │  Failed tests:                          │                 │
│     │    ✗ test_schwab_e2e_workflow           │                 │
│     └─────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ All Tests Pass? │
                    └─────────────────┘
                         /        \
                    Yes /          \ No
                       /            \
                      ▼              ▼
          ┌──────────────────┐   ┌────────────┐
          │  Continue        │   │  Exit 2    │
          └──────────────────┘   └────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  9. DISPLAY BUILD SUMMARY                                       │
│     ├─ Time elapsed: Xm Ys                                      │
│     ├─ Success/failure message                                  │
│     └─ Exit with appropriate code                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Exit Code:     │
                    │  0 = Success    │
                    │  1 = Build fail │
                    │  2 = Test fail  │
                    │  3 = Config fail│
                    └─────────────────┘
```

## Detailed Function Call Flow

```
main()
  │
  ├─→ parse_args()
  │    └─ Parse --clean, --no-tests, --verbose, --help
  │
  ├─→ display_build_info()
  │    └─ Show project paths and compiler info
  │
  ├─→ setup_environment()
  │    ├─ Set CC, CXX, SKIP_CLANG_TIDY, LD_LIBRARY_PATH
  │    └─ Verify compiler exists
  │
  ├─→ clean_build()
  │    └─ Remove build/ directory if --clean
  │
  ├─→ configure_cmake()
  │    ├─ Create build directory
  │    ├─ cd build
  │    └─ Run cmake with flags
  │
  ├─→ build_targets()
  │    ├─ For each target in BUILD_TARGETS:
  │    │    ├─ log_step("Building target: $target")
  │    │    ├─ cmake --build . --target $target -j $(nproc)
  │    │    ├─ Check exit code
  │    │    └─ log_success() or exit 1
  │    └─ log_success("All targets built")
  │
  ├─→ run_tests()
  │    ├─ If RUN_TESTS == false, return early
  │    ├─ For each test in TEST_EXECUTABLES:
  │    │    ├─ log_step("Running test: $test")
  │    │    ├─ ./build/bin/$test --gtest_color=yes
  │    │    ├─ Track pass/fail
  │    │    └─ Store result
  │    ├─ Display test summary
  │    ├─ Show statistics
  │    └─ Return 0 (pass) or 2 (fail)
  │
  ├─→ Calculate elapsed time
  │
  ├─→ Display final summary
  │
  └─→ Exit with appropriate code
```

## Logging Function Hierarchy

```
Logging Functions:
├─ log_header()    - Section headers (cyan, bold)
├─ log_info()      - Informational (blue)
├─ log_success()   - Success messages (green)
├─ log_warning()   - Warnings (yellow)
├─ log_error()     - Errors (red)
└─ log_step()      - Step indicators (magenta, bold)
```

## Color Scheme

```
CYAN      - Section headers, banner
BLUE      - Info messages
GREEN     - Success messages, passed tests
YELLOW    - Warning messages
RED       - Error messages, failed tests
MAGENTA   - Step indicators
BOLD      - Emphasis
RESET     - Clear formatting
```

## Build Target Dependencies

```
schwab_api
    │
    ├─→ utils
    ├─→ options_pricing
    ├─→ CURL::libcurl
    └─→ nlohmann_json
          │
          └─→ (used by tests)
                │
                ├─→ test_order_manager_integration
                │     └─ Tests: Market/Limit/Option orders
                │
                ├─→ test_account_manager_integration
                │     └─ Tests: Account data, positions, balances
                │
                └─→ test_schwab_e2e_workflow
                      └─ Tests: Complete workflows + risk mgmt
```

## Error Handling Flow

```
Any Error Occurs
    │
    ├─→ trap ERR catches error
    │    └─ log_error("Script failed at line $LINENO")
    │
    ├─→ Function checks result
    │    ├─ if ! command; then
    │    │    log_error("Description")
    │    │    exit N
    │    └─ fi
    │
    └─→ Exit with code:
         ├─ 1: Build failed
         ├─ 2: Tests failed
         └─ 3: Configuration failed
```

## Parallel Processing

```
Build Phase (Parallel within each target):
    Target 1: schwab_api
        └─ cmake --build . -j $(nproc)
             └─ Compiles N files in parallel

    Target 2: test_order_manager_integration
        └─ cmake --build . -j $(nproc)
             └─ Compiles N files in parallel

    (Sequential across targets for dependencies)

Test Phase (Sequential):
    Test 1: test_order_manager_integration
        └─ Runs to completion

    Test 2: test_account_manager_integration
        └─ Runs to completion

    Test 3: test_schwab_e2e_workflow
        └─ Runs to completion
```

## Output Examples

### Successful Build

```
╔═══════════════════════════════════════════════════════════════╗
║       Schwab API Build and Test Script                       ║
║       BigBrotherAnalytics Trading System                      ║
╚═══════════════════════════════════════════════════════════════╝

========================================
Setting Up Build Environment
========================================

[INFO] CC          = /usr/local/bin/clang
[INFO] CXX         = /usr/local/bin/clang++
[SUCCESS] Environment configured successfully

========================================
Configuring CMake
========================================

>>> Running CMake configuration

[SUCCESS] CMake configuration completed successfully

========================================
Building Schwab API Module and Tests
========================================

>>> Building target: schwab_api

[SUCCESS] Target 'schwab_api' built successfully

========================================
Running Schwab API Integration Tests
========================================

>>> Running test: test_order_manager_integration

[SUCCESS] Test 'test_order_manager_integration' PASSED

========================================
Test Summary
========================================

Test Results:
  ✓ test_order_manager_integration
  ✓ test_account_manager_integration
  ✓ test_schwab_e2e_workflow

Statistics:
  Total Tests:  3
  Passed:       3
  Failed:       0

[SUCCESS] All tests passed!

========================================
Build Complete
========================================

Time Elapsed: 2m 34s

✓ Build and test cycle completed successfully!
```

### Failed Test

```
========================================
Test Summary
========================================

Test Results:
  ✓ test_order_manager_integration
  ✗ test_account_manager_integration
  ✓ test_schwab_e2e_workflow

Statistics:
  Total Tests:  3
  Passed:       2
  Failed:       1

[ERROR] The following tests failed:
  ✗ test_account_manager_integration

========================================
Build Complete
========================================

Time Elapsed: 2m 45s

✗ Build succeeded but tests failed
```

## Script Statistics

- **Total Lines**: 429
- **Functions**: 11
- **Color Definitions**: 8
- **Build Targets**: 4
- **Test Executables**: 3
- **Command Line Options**: 4
- **Exit Codes**: 4
- **Logging Levels**: 6

## Key Variables

```bash
SCRIPT_DIR          # Script location
PROJECT_ROOT        # Project root directory
BUILD_DIR           # Build directory path
FORCE_CLEAN         # Clean build flag
RUN_TESTS           # Run tests flag
VERBOSE             # Verbose output flag
BUILD_TARGETS[]     # Array of build targets
TEST_EXECUTABLES[]  # Array of test executables
```

## Integration Points

```
Script Interfaces:
├─ Input:
│   ├─ Command line arguments
│   ├─ Environment variables (optional overrides)
│   └─ CMakeLists.txt (build configuration)
│
├─ Output:
│   ├─ Console output (colored)
│   ├─ Exit codes (0/1/2/3)
│   └─ Build artifacts (build/bin/, build/lib/)
│
└─ External Tools:
    ├─ cmake (configuration and build)
    ├─ clang/clang++ (compilation)
    ├─ GTest (test framework)
    └─ nproc (CPU detection)
```
