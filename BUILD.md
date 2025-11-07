# Build Instructions

This document provides instructions for building the C++23 components of BigBrotherAnalytics.

## Prerequisites

### Required
- GCC 15+ with C++23 support (already installed)
- CMake 3.20+ (already installed)
- Python 3.13+ (already installed)
- pybind11 (already installed via uv)
- OpenMP (already installed)

### Optional
- Intel MKL (for optimized BLAS/LAPACK)
- MPI (for distributed correlation calculations)
- libcurl (for Schwab API HTTP requests)
- Google Test (for C++ unit tests)

## Installing Optional Dependencies

```bash
# Install libcurl
sudo apt-get install libcurl4-openssl-dev

# Install Google Test
sudo apt-get install libgtest-dev

# Intel MKL (optional, for performance)
# Follow: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
```

## Build Process

### 1. Create Build Directory

```bash
mkdir -p build
cd build
```

### 2. Configure with CMake

```bash
# Basic configuration
cmake ..

# Or with specific options
cmake -DCMAKE_BUILD_TYPE=Release ..

# With Intel MKL (if installed)
cmake -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/intel/oneapi/mkl/latest ..
```

### 3. Build

```bash
# Build all targets
make -j$(nproc)

# Or build specific targets
make options_pricing          # Options pricing library
make options_pricing_py       # Python bindings for options pricing
make correlation_engine       # Correlation engine library
make correlation_engine_py    # Python bindings for correlation engine
make schwab_api              # Schwab API library
make schwab_api_py           # Python bindings for Schwab API
```

### 4. Run Tests (Optional)

```bash
# Run C++ tests
make test

# Or use ctest directly
ctest --output-on-failure
```

### 5. Install (Optional)

```bash
# Install libraries and Python modules
sudo make install
```

## Quick Build Script

A convenience script is provided for common build operations:

```bash
# Build everything in Release mode
./scripts/build.sh

# Build with Debug symbols
./scripts/build.sh debug

# Clean and rebuild
./scripts/build.sh clean

# Run tests after building
./scripts/build.sh test
```

## Using the Python Modules

After building, the Python bindings will be available in the `src/correlation_engine/` directory:

```python
# Import the C++ modules
from src.correlation_engine import options_pricing_py
from src.correlation_engine import correlation_engine_py
from src.schwab_api import schwab_api_py

# Example: Calculate Black-Scholes price
price = options_pricing_py.black_scholes(
    S=100.0,      # Stock price
    K=105.0,      # Strike price
    T=0.5,        # Time to expiration (years)
    r=0.05,       # Risk-free rate
    sigma=0.2,    # Volatility
    option_type='call'
)
```

## Build Output

After a successful build, you'll find:
- **Libraries**: `build/lib/*.so`
- **Python modules**: `src/correlation_engine/*.so` and `src/schwab_api/*.so`
- **Tests**: `build/tests/cpp/test_*`

## Troubleshooting

### CMake can't find pybind11
```bash
# Ensure pybind11 is installed via uv
uv pip list | grep pybind11

# Or install manually
uv pip install pybind11
```

### OpenMP not found
```bash
# Install OpenMP
sudo apt-get install libomp-dev
```

### MKL not found (optional)
This is not an error - the system will fall back to standard BLAS/LAPACK. For better performance, install Intel MKL.

### Python module import errors
Make sure you're running Python with uv:
```bash
uv run python your_script.py
```

## Performance Notes

- **Release builds** are highly optimized with `-O3 -march=native`
- **Intel MKL** provides 2-5x speedup for matrix operations
- **MPI** enables multi-node correlation calculations (Tier 2+)
- **OpenMP** provides multi-threading within a single node

## Development Workflow

1. Make changes to C++ source files in `src/*/cpp/`
2. Rebuild: `cd build && make -j$(nproc)`
3. Test in Python: `uv run python test_script.py`
4. Run C++ tests: `make test`
5. Iterate

## Next Steps

After building the C++ components:
1. Test the Python bindings work correctly
2. Implement the actual C++ source files (currently placeholders)
3. Write comprehensive tests
4. Benchmark performance vs pure Python
5. Optimize hot paths identified through profiling
