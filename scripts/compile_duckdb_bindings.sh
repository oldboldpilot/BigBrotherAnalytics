#!/bin/bash
#
# Compile DuckDB Python Bindings
#
# This script compiles the native C++ DuckDB bindings for Python.
# The bindings provide 5-10x performance improvement over pure Python DuckDB.
#
# Features:
# - GIL-free query execution
# - Zero-copy NumPy array transfer
# - Native DuckDB C++ API integration
#
# Note: Must use C++20 (not C++23) due to DuckDB compatibility with Clang 21
#

set -e

echo "======================================================================="
echo "Compiling DuckDB Native C++ Python Bindings"
echo "======================================================================="
echo

# Configuration
CLANG_CXX="/home/linuxbrew/.linuxbrew/bin/clang++"
PYTHON_INCLUDE="/home/muyiwa/.local/share/uv/python/cpython-3.13.8-linux-x86_64-gnu/include/python3.13"
PYBIND11_INCLUDE="/usr/include/pybind11"
DUCKDB_INCLUDE="/usr/local/include"
DUCKDB_LIB="/usr/local/lib/libduckdb.so"
SOURCE="src/python_bindings/duckdb_bindings.cpp"
OUTPUT="python/bigbrother_duckdb.cpython-313-x86_64-linux-gnu.so"

echo "Configuration:"
echo "  Compiler: $CLANG_CXX"
echo "  Source: $SOURCE"
echo "  Output: $OUTPUT"
echo

# Check if source file exists
if [ ! -f "$SOURCE" ]; then
    echo "ERROR: Source file not found: $SOURCE"
    exit 1
fi

# Check if DuckDB library exists
if [ ! -f "$DUCKDB_LIB" ]; then
    echo "ERROR: DuckDB library not found: $DUCKDB_LIB"
    echo "Install with: sudo apt install libduckdb-dev"
    exit 1
fi

echo "Compiling..."

# Compile with C++20 (DuckDB v1.1.3 has issues with C++23)
$CLANG_CXX \
  -O3 -DNDEBUG -std=c++20 -fPIC \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -shared \
  -I"$PYBIND11_INCLUDE" \
  -I"$PYTHON_INCLUDE" \
  -I"$DUCKDB_INCLUDE" \
  -o "$OUTPUT" \
  "$SOURCE" \
  "$DUCKDB_LIB"

if [ $? -eq 0 ]; then
    echo
    echo "✓ Compilation successful!"
    echo
    ls -lh "$OUTPUT"
    echo
    echo "Test with:"
    echo "  python3 test_duckdb_bindings.py"
    echo
else
    echo
    echo "✗ Compilation failed!"
    exit 1
fi
