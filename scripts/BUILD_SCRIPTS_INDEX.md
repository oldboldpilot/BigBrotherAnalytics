# Build Scripts Index

## Schwab API Build and Test Scripts

This directory contains automated build and test scripts for the BigBrotherAnalytics Schwab API integration.

### Main Build Script

**File**: `build_and_test_schwab.sh`
**Purpose**: Comprehensive build and test automation for Schwab API module
**Type**: Bash script (executable)
**Size**: 429 lines, 13 KB

### Documentation Files

| File | Purpose | Size |
|------|---------|------|
| `README_BUILD_SCRIPT.md` | Complete documentation and reference | 8 KB |
| `SCHWAB_BUILD_QUICKSTART.md` | Quick start guide and common commands | 3.6 KB |
| `BUILD_SCRIPTS_INDEX.md` | This file - overview of all scripts | - |

## Quick Start

```bash
# Run from project root
cd /home/muyiwa/Development/BigBrotherAnalytics

# Standard build and test
./scripts/build_and_test_schwab.sh

# Clean build
./scripts/build_and_test_schwab.sh --clean

# Build without tests
./scripts/build_and_test_schwab.sh --no-tests

# Get help
./scripts/build_and_test_schwab.sh --help
```

## What Gets Built

The script builds and tests the following components:

1. **schwab_api** - Core Schwab API C++23 module library
   - Token management
   - Order management
   - Account management
   - Market data integration
   - Error handling and retry logic

2. **test_order_manager_integration** - Order Management Tests
   - Market order placement
   - Limit order handling
   - Option order strategies
   - Order modification
   - Order cancellation
   - Order status tracking

3. **test_account_manager_integration** - Account Management Tests
   - Account data retrieval
   - Position tracking
   - Balance queries
   - Account preferences
   - Trading permissions

4. **test_schwab_e2e_workflow** - End-to-End Workflow Tests
   - Complete trading workflows
   - Risk management integration
   - Order validation
   - Multi-leg strategies
   - Error recovery

## Script Features

### Environment Setup
- Configures Clang compiler paths
- Sets library paths for runtime linking
- Skips clang-tidy for faster builds
- Validates compiler availability

### Build Management
- Clean build support
- Parallel compilation (uses all CPU cores)
- Sequential target building for proper dependencies
- CMake configuration with Release optimization

### Testing
- Runs all Schwab API integration tests
- Colored GTest output
- Individual test pass/fail tracking
- Comprehensive summary with statistics

### Output and Logging
- Color-coded console output
- Build progress indicators
- Error messages with line numbers
- Timing statistics
- Test result summaries

### Error Handling
- Robust error detection
- Clear error messages
- Proper exit codes for CI/CD
- Trap for unexpected failures

## Command Line Options

| Option | Description |
|--------|-------------|
| `--clean` | Remove build directory before building |
| `--no-tests` | Build only, skip test execution |
| `--verbose` | Enable verbose CMake and build output |
| `--help` | Display usage information |

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success - all tests passed | Continue |
| 1 | Build failed | Check build errors |
| 2 | Tests failed | Check test output |
| 3 | Configuration failed | Check CMake configuration |

## Performance Metrics

Typical execution times on modern hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Incremental build | 30-60s | With existing build cache |
| Clean build | 2-5m | Full rebuild from scratch |
| Test execution | 10-30s | All three test suites |
| Total (clean + test) | 3-6m | Complete build cycle |

Performance scales with:
- CPU cores (parallel compilation)
- Storage speed (SSD recommended)
- Build cache state

## Directory Structure

```
BigBrotherAnalytics/
├── scripts/
│   ├── build_and_test_schwab.sh          # Main build script
│   ├── README_BUILD_SCRIPT.md            # Full documentation
│   ├── SCHWAB_BUILD_QUICKSTART.md        # Quick reference
│   └── BUILD_SCRIPTS_INDEX.md            # This file
│
├── src/
│   └── schwab_api/
│       └── schwab_api.cppm               # Main API module
│
├── tests/cpp/
│   ├── test_order_manager_integration.cpp
│   ├── test_account_manager_integration.cpp
│   └── test_schwab_e2e_workflow.cpp
│
└── build/                                 # Created by script
    ├── bin/
    │   ├── test_order_manager_integration
    │   ├── test_account_manager_integration
    │   └── test_schwab_e2e_workflow
    └── lib/
        └── libschwab_api.so
```

## Integration with Development Workflow

### Standard Workflow

```bash
# 1. Make code changes
vim src/schwab_api/schwab_api.cppm

# 2. Build and test
./scripts/build_and_test_schwab.sh

# 3. If successful, commit
git add src/schwab_api/schwab_api.cppm
git commit -m "feat: add order validation"
git push
```

### After Major Changes

```bash
# Clean build recommended
./scripts/build_and_test_schwab.sh --clean
```

### Rapid Iteration

```bash
# Build without tests for quick feedback
./scripts/build_and_test_schwab.sh --no-tests

# Run specific test manually
./build/bin/test_order_manager_integration --gtest_filter=OrderTest.Market
```

### Debugging Build Issues

```bash
# Maximum verbosity
./scripts/build_and_test_schwab.sh --clean --verbose 2>&1 | tee build.log

# Review logs
less build.log
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Schwab API Build and Test

on:
  push:
    branches: [ master, schwab-api ]
  pull_request:
    branches: [ master ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Test Schwab API
        run: ./scripts/build_and_test_schwab.sh --clean
```

### GitLab CI Example

```yaml
schwab-api-test:
  stage: test
  script:
    - ./scripts/build_and_test_schwab.sh --clean
  only:
    - schwab-api
    - master
  artifacts:
    when: on_failure
    paths:
      - build/
```

## Troubleshooting

### Common Issues

#### Compiler Not Found
```bash
# Verify Clang installation
which clang++
ls -l /usr/local/bin/clang++

# Update script if needed
export CXX=/usr/bin/clang++
```

#### Build Fails
```bash
# Try clean build with verbose output
./scripts/build_and_test_schwab.sh --clean --verbose

# Check for missing dependencies
cmake --version  # Should be 3.28+
clang++ --version  # Should be 21+
```

#### Tests Fail
```bash
# Run individual test with filter
./build/bin/test_order_manager_integration --gtest_filter=TestName

# Check test logs
./build/bin/test_order_manager_integration --gtest_output=xml:results.xml
```

#### Configuration Fails
```bash
# Check CMake configuration
cd build
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON

# Verify all dependencies
cmake .. --debug-output
```

## Best Practices

1. **Run tests before committing**
   ```bash
   ./scripts/build_and_test_schwab.sh
   ```

2. **Clean build after major changes**
   ```bash
   ./scripts/build_and_test_schwab.sh --clean
   ```

3. **Use verbose mode for debugging**
   ```bash
   ./scripts/build_and_test_schwab.sh --verbose
   ```

4. **Check exit codes in automation**
   ```bash
   if ./scripts/build_and_test_schwab.sh; then
       echo "Success"
   else
       echo "Failed with code $?"
       exit 1
   fi
   ```

5. **Run specific tests for focused work**
   ```bash
   ./build/bin/test_order_manager_integration --gtest_filter=OrderTest.*
   ```

## Related Scripts

| Script | Purpose |
|--------|---------|
| `build.sh` | Main project build script |
| `build_schwab_tests.sh` | Alternative Schwab test builder |
| `run_schwab_tests.sh` | Test runner only |
| `quick_test_schwab.sh` | Fast test iteration |
| `run_clang_tidy.sh` | Code quality checks |

## Dependencies

### Required
- Clang 21+ (C++23 support)
- CMake 3.28+
- Google Test (GTest)
- CURL library
- nlohmann/json
- OpenMP
- Threads (pthread)

### Optional
- spdlog (logging)
- yaml-cpp (configuration)
- DuckDB (database)

## Environment Variables

The script sets these automatically:

```bash
CC=/usr/local/bin/clang
CXX=/usr/local/bin/clang++
SKIP_CLANG_TIDY=1
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

Override if needed:

```bash
export CXX=/custom/path/clang++
./scripts/build_and_test_schwab.sh
```

## Support and Documentation

For more information:

- **Complete Guide**: `scripts/README_BUILD_SCRIPT.md`
- **Quick Reference**: `scripts/SCHWAB_BUILD_QUICKSTART.md`
- **Main CMake**: `CMakeLists.txt`
- **Test CMake**: `tests/cpp/CMakeLists.txt`
- **API Documentation**: `src/schwab_api/schwab_api.cppm`

## Contributing

When adding new tests or modifying the build:

1. Update the script's `BUILD_TARGETS` array if adding new targets
2. Update the `TEST_EXECUTABLES` array if adding new tests
3. Update documentation files
4. Test with both `--clean` and incremental builds
5. Verify exit codes work correctly

## Version History

- **v1.0** (2025-11-09): Initial comprehensive build script
  - Environment setup
  - Clean build support
  - Parallel compilation
  - Test integration
  - Colored output
  - Error handling
  - Documentation

## License

Part of BigBrotherAnalytics trading system.

## Contact

See main project README for contact information.
