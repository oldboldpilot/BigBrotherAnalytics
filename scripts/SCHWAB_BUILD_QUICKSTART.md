# Schwab API Build Script - Quick Start Guide

## TL;DR

```bash
# Quick build and test
./scripts/build_and_test_schwab.sh

# Clean build
./scripts/build_and_test_schwab.sh --clean

# Build without tests
./scripts/build_and_test_schwab.sh --no-tests
```

## What Gets Built

1. `schwab_api` - Core Schwab API library
2. `test_order_manager_integration` - Order management tests
3. `test_account_manager_integration` - Account management tests
4. `test_schwab_e2e_workflow` - End-to-end workflow tests

## What Gets Tested

- Order placement, modification, and cancellation
- Account data retrieval and management
- Complete trading workflows with risk management
- Integration with Schwab's REST API

## Common Commands

```bash
# Standard workflow
./scripts/build_and_test_schwab.sh

# After major changes
./scripts/build_and_test_schwab.sh --clean

# Debug build issues
./scripts/build_and_test_schwab.sh --clean --verbose

# Build only (for quick iteration)
./scripts/build_and_test_schwab.sh --no-tests

# Get help
./scripts/build_and_test_schwab.sh --help
```

## Exit Codes

- **0** = Success (all good!)
- **1** = Build failed
- **2** = Tests failed
- **3** = Configuration failed

## Environment

The script automatically sets:
- Uses Clang compiler (`/usr/local/bin/clang++`)
- Skips clang-tidy for speed
- Sets proper library paths
- Configures for Release build

## Time Estimates

- **Incremental build**: ~30-60 seconds
- **Clean build**: ~2-5 minutes
- **Test execution**: ~10-30 seconds

## Troubleshooting

### Build fails?
```bash
./scripts/build_and_test_schwab.sh --clean --verbose
```

### Test fails?
```bash
# Run specific test
./build/bin/test_order_manager_integration --gtest_filter=TestName
```

### Need more info?
```bash
# See full documentation
cat scripts/README_BUILD_SCRIPT.md
```

## Directory Structure

```
scripts/
├── build_and_test_schwab.sh          # Main build script (executable)
├── README_BUILD_SCRIPT.md            # Full documentation
└── SCHWAB_BUILD_QUICKSTART.md        # This file

build/                                 # Created by script
├── bin/
│   ├── test_order_manager_integration
│   ├── test_account_manager_integration
│   └── test_schwab_e2e_workflow
└── lib/
    └── libschwab_api.so
```

## Integration with Development Workflow

```bash
# 1. Make changes to Schwab API code
vim src/schwab_api/schwab_api.cppm

# 2. Build and test
./scripts/build_and_test_schwab.sh

# 3. If tests pass, commit
git add .
git commit -m "feat: improve Schwab API error handling"

# 4. Push
git push
```

## CI/CD Integration

```yaml
# Example .gitlab-ci.yml or .github/workflows/build.yml
build-and-test:
  script:
    - ./scripts/build_and_test_schwab.sh --clean
  only:
    - schwab-api
    - master
```

## Key Features

- **Smart Dependencies**: Builds in correct order
- **Parallel Building**: Uses all CPU cores
- **Color Output**: Easy to read results
- **Error Recovery**: Clear error messages
- **Fast Iteration**: Incremental builds
- **Test Integration**: Runs GTest with colored output

## Requirements

- Clang 21+ (C++23 support)
- CMake 3.28+
- Google Test
- All BigBrotherAnalytics dependencies

## Pro Tips

1. Use `--clean` after switching branches
2. Use `--no-tests` for rapid code iteration
3. Use `--verbose` when debugging
4. Check exit codes in scripts: `if [ $? -eq 0 ]`
5. Run individual tests for focused debugging

## Support

For detailed documentation, see:
- `scripts/README_BUILD_SCRIPT.md` - Complete guide
- `tests/cpp/CMakeLists.txt` - Test configuration
- `CMakeLists.txt` - Main build configuration
