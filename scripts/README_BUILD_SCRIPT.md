# Schwab API Build and Test Script

## Overview

The `build_and_test_schwab.sh` script provides a comprehensive, automated build and test workflow for the Schwab API integration module in the BigBrotherAnalytics trading system.

## Location

```
/home/muyiwa/Development/BigBrotherAnalytics/scripts/build_and_test_schwab.sh
```

## Features

- **Automated Environment Setup**: Configures all necessary environment variables
- **Clean Build Support**: Option to start from a clean state
- **Parallel Building**: Utilizes all available CPU cores for faster builds
- **Comprehensive Testing**: Runs all Schwab API integration tests
- **Colored Output**: Clear, color-coded console output for easy reading
- **Error Handling**: Robust error detection and reporting
- **Build Metrics**: Tracks and reports build time and test results

## Usage

### Basic Usage

```bash
# Build and test everything
./scripts/build_and_test_schwab.sh

# Clean build from scratch
./scripts/build_and_test_schwab.sh --clean

# Build only, skip tests
./scripts/build_and_test_schwab.sh --no-tests

# Verbose output for debugging
./scripts/build_and_test_schwab.sh --verbose

# Combine options
./scripts/build_and_test_schwab.sh --clean --verbose
```

### Display Help

```bash
./scripts/build_and_test_schwab.sh --help
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--clean` | Force a clean build by removing the build directory |
| `--no-tests` | Build only, skip running tests |
| `--verbose` | Enable verbose output for CMake and build process |
| `--help` | Display usage information |

## Environment Variables

The script automatically configures:

- `CC=/usr/local/bin/clang` - C compiler
- `CXX=/usr/local/bin/clang++` - C++ compiler
- `SKIP_CLANG_TIDY=1` - Skip clang-tidy for faster builds
- `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` - Runtime library path

## Build Targets

The script builds the following targets in order:

1. **schwab_api** - Core Schwab API library
2. **test_order_manager_integration** - Order management integration tests
3. **test_account_manager_integration** - Account management integration tests
4. **test_schwab_e2e_workflow** - End-to-end workflow tests

## Test Executables

The script runs these test suites:

- `test_order_manager_integration` - Tests order placement, modification, cancellation
- `test_account_manager_integration` - Tests account data retrieval and management
- `test_schwab_e2e_workflow` - Tests complete trading workflows

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - all tests passed |
| 1 | Build failed |
| 2 | Tests failed |
| 3 | Configuration failed |

## Output

The script provides:

- **Build Progress**: Clear indicators of current build step
- **Test Results**: Individual test pass/fail status with colored output
- **Summary**: Complete statistics of build and test results
- **Timing**: Total time elapsed for build and test cycle

### Example Output

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║       Schwab API Build and Test Script                       ║
║       BigBrotherAnalytics Trading System                      ║
║                                                               ║
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

[...]

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

## Error Handling

The script includes comprehensive error handling:

- Verifies compiler existence before starting
- Checks for build failures at each stage
- Reports line numbers where errors occur
- Provides clear error messages with color coding
- Returns appropriate exit codes for CI/CD integration

## Integration with CI/CD

The script is designed for easy integration with CI/CD pipelines:

```bash
# In CI/CD pipeline
./scripts/build_and_test_schwab.sh --clean
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Build and test successful"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "Build failed"
    exit 1
elif [ $EXIT_CODE -eq 2 ]; then
    echo "Tests failed"
    exit 2
else
    echo "Configuration failed"
    exit 3
fi
```

## Troubleshooting

### Compiler Not Found

If you see "C++ compiler not found":

```bash
# Verify Clang installation
ls -l /usr/local/bin/clang++

# If not found, update the script to point to your Clang installation
export CXX=/path/to/your/clang++
```

### Build Failures

For build failures:

1. Run with `--verbose` flag to see detailed output
2. Check CMake configuration errors
3. Verify all dependencies are installed
4. Try a clean build with `--clean`

### Test Failures

If tests fail:

1. Run individual test executables directly:
   ```bash
   ./build/bin/test_order_manager_integration --gtest_filter=TestName
   ```
2. Check test output for specific failure messages
3. Verify Schwab API credentials and configuration
4. Ensure test environment is properly configured

## Best Practices

1. **Clean Builds**: Use `--clean` after major changes or when switching branches
2. **Regular Testing**: Run tests before committing code
3. **Verbose Mode**: Use `--verbose` when debugging build issues
4. **CI Integration**: Integrate script into continuous integration pipeline
5. **Build Isolation**: Script runs in isolated environment for reproducibility

## Dependencies

Required system packages:

- Clang 21+ (C++23 support)
- CMake 3.28+
- Google Test (GTest)
- CURL library
- nlohmann/json
- All other BigBrotherAnalytics dependencies

## Performance

Typical build times (on modern hardware):

- **Incremental Build**: 30-60 seconds
- **Clean Build**: 2-5 minutes
- **Test Execution**: 10-30 seconds
- **Total (Clean + Tests)**: 3-6 minutes

Build time scales with:
- Number of CPU cores (uses all available)
- Storage speed (SSD recommended)
- Prior build cache

## Advanced Usage

### Custom Build Directory

```bash
# Edit script to change BUILD_DIR variable
BUILD_DIR="/custom/path/build"
```

### Selective Test Execution

```bash
# Run specific test only
./build/bin/test_order_manager_integration --gtest_filter=OrderManagerTest.PlaceMarketOrder
```

### Debugging Build Issues

```bash
# Maximum verbosity
./scripts/build_and_test_schwab.sh --clean --verbose 2>&1 | tee build.log
```

## Maintenance

The script is self-contained and requires minimal maintenance. Update when:

- New test targets are added to CMakeLists.txt
- Compiler paths change
- Additional build flags are needed
- New environment variables are required

## See Also

- [CMakeLists.txt](../CMakeLists.txt) - Main build configuration
- [tests/cpp/CMakeLists.txt](../tests/cpp/CMakeLists.txt) - Test configuration
- [Schwab API Documentation](../docs/) - API integration guide
