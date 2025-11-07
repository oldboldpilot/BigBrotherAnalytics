#!/bin/bash

# Install C++ dependencies for BigBrotherAnalytics
# Run with sudo if needed

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   BigBrotherAnalytics C++ Dependencies Installer          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Warning: Not running as root. May need sudo for some installations.${NC}"
    SUDO="sudo"
else
    SUDO=""
fi

# Update package lists
echo -e "${YELLOW}Updating package lists...${NC}"
$SUDO apt-get update

# Install core build tools (if not already installed)
echo -e "${YELLOW}Installing core build tools...${NC}"
$SUDO apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git

# Install libcurl for HTTP requests
echo -e "${YELLOW}Installing libcurl...${NC}"
$SUDO apt-get install -y libcurl4-openssl-dev

# Install JSON library
echo -e "${YELLOW}Installing nlohmann-json...${NC}"
$SUDO apt-get install -y nlohmann-json3-dev

# Install YAML parser
echo -e "${YELLOW}Installing yaml-cpp...${NC}"
$SUDO apt-get install -y libyaml-cpp-dev

# Install spdlog for logging
echo -e "${YELLOW}Installing spdlog...${NC}"
$SUDO apt-get install -y libspdlog-dev

# Install Google Test for C++ testing
echo -e "${YELLOW}Installing Google Test...${NC}"
$SUDO apt-get install -y libgtest-dev
cd /usr/src/gtest || true
$SUDO cmake CMakeLists.txt || true
$SUDO make || true
$SUDO cp lib/*.a /usr/lib || $SUDO cp *.a /usr/lib || true
cd - > /dev/null

# Install websocketpp for WebSocket support
echo -e "${YELLOW}Installing websocketpp...${NC}"
$SUDO apt-get install -y libwebsocketpp-dev

# Install Boost (required by websocketpp)
echo -e "${YELLOW}Installing Boost...${NC}"
$SUDO apt-get install -y libboost-all-dev

# ============================================================================
# DuckDB C++ Library
# ============================================================================

echo -e "${YELLOW}Installing DuckDB C++ library...${NC}"

DUCKDB_VERSION="1.1.3"
DUCKDB_URL="https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/libduckdb-linux-amd64.zip"

cd /tmp
wget -q "$DUCKDB_URL" -O libduckdb.zip
unzip -q -o libduckdb.zip
$SUDO cp duckdb.h duckdb.hpp /usr/local/include/
$SUDO cp libduckdb.so /usr/local/lib/
$SUDO ldconfig
rm -f libduckdb.zip duckdb.h duckdb.hpp libduckdb.so
cd - > /dev/null

echo -e "${GREEN}✓ DuckDB installed${NC}"

# ============================================================================
# ONNX Runtime C++ Library
# ============================================================================

echo -e "${YELLOW}Installing ONNX Runtime C++ library...${NC}"

ONNX_VERSION="1.18.0"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"

cd /tmp
wget -q "$ONNX_URL" -O onnxruntime.tgz
tar -xzf onnxruntime.tgz
ONNX_DIR="onnxruntime-linux-x64-${ONNX_VERSION}"
$SUDO cp -r "$ONNX_DIR/include/"* /usr/local/include/
$SUDO cp -r "$ONNX_DIR/lib/"* /usr/local/lib/
$SUDO ldconfig
rm -rf onnxruntime.tgz "$ONNX_DIR"
cd - > /dev/null

echo -e "${GREEN}✓ ONNX Runtime installed${NC}"

# ============================================================================
# Verify installations
# ============================================================================

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Verifying Installations                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for headers and libraries
check_header() {
    if [ -f "/usr/include/$1" ] || [ -f "/usr/local/include/$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 not found"
    fi
}

check_lib() {
    if ldconfig -p | grep -q "$1"; then
        echo -e "${GREEN}✓${NC} lib$1"
    else
        echo -e "${RED}✗${NC} lib$1 not found"
    fi
}

echo "Headers:"
check_header "curl/curl.h"
check_header "nlohmann/json.hpp"
check_header "yaml-cpp/yaml.h"
check_header "spdlog/spdlog.h"
check_header "gtest/gtest.h"
check_header "websocketpp/config/asio_client.hpp"
check_header "duckdb.hpp"
check_header "onnxruntime_cxx_api.h"

echo ""
echo "Libraries:"
check_lib "curl"
check_lib "yaml-cpp"
check_lib "spdlog"
check_lib "gtest"
check_lib "boost_system"
check_lib "duckdb"
check_lib "onnxruntime"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Installation Complete!                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "1. cd build"
echo "2. cmake .."
echo "3. make -j\$(nproc)"
echo ""
