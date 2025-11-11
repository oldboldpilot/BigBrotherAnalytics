#!/bin/bash
# BigBrotherAnalytics - Complete System Bootstrap
# Deploys entire system from scratch on a fresh machine
#
# Author: Olumuyiwa Oluwasanmi
# Date: November 10, 2025
#
# Usage:
#   ./scripts/bootstrap.sh [--skip-ansible] [--skip-build] [--skip-tests]
#
# This script performs:
#   1. Prerequisites check (ansible, uv, git)
#   2. Ansible playbook deployment (compilers, libraries, dependencies)
#   3. C++ project compilation
#   4. Python environment setup
#   5. Database initialization
#   6. System verification
#   7. Phase 5 readiness check

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse command line arguments
SKIP_ANSIBLE=false
SKIP_BUILD=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ansible)
            SKIP_ANSIBLE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help|-h)
            echo "BigBrotherAnalytics Bootstrap Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-ansible    Skip ansible playbook deployment"
            echo "  --skip-build      Skip C++ compilation"
            echo "  --skip-tests      Skip test execution"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo ""
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Start bootstrap
print_header "BigBrotherAnalytics - Complete System Bootstrap"

echo "Project Root: $PROJECT_ROOT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 1: Check Prerequisites
print_header "Step 1: Prerequisites Check"

MISSING_DEPS=()

print_step "Checking for git..."
if command -v git &> /dev/null; then
    print_success "git $(git --version | awk '{print $3}')"
else
    print_error "git not found"
    MISSING_DEPS+=("git")
fi

print_step "Checking for ansible..."
if command -v ansible-playbook &> /dev/null; then
    print_success "ansible $(ansible --version | head -1 | awk '{print $2}')"
else
    print_warning "ansible not found (required for full deployment)"
    if [ "$SKIP_ANSIBLE" = false ]; then
        MISSING_DEPS+=("ansible")
    fi
fi

print_step "Checking for uv (Python package manager)..."
if command -v uv &> /dev/null; then
    print_success "uv $(uv --version | awk '{print $2}')"
else
    print_error "uv not found"
    MISSING_DEPS+=("uv")
    print_info "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

print_step "Checking for ninja build system..."
if command -v ninja &> /dev/null; then
    print_success "ninja $(ninja --version)"
else
    print_warning "ninja not found (will be installed by ansible)"
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_error "Missing required dependencies: ${MISSING_DEPS[*]}"
    echo ""
    print_info "Install missing dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        case $dep in
            git)
                echo "  Ubuntu/Debian: sudo apt-get install git"
                echo "  RHEL/CentOS:   sudo yum install git"
                echo "  macOS:         brew install git"
                ;;
            ansible)
                echo "  Ubuntu/Debian: sudo apt-get install ansible"
                echo "  RHEL/CentOS:   sudo yum install ansible"
                echo "  macOS:         brew install ansible"
                echo "  pip:           pip install ansible"
                ;;
            uv)
                echo "  All systems:   curl -LsSf https://astral.sh/uv/install.sh | sh"
                ;;
        esac
    done
    exit 1
fi

print_success "All required prerequisites installed"

# Step 2: Run Ansible Playbook
if [ "$SKIP_ANSIBLE" = false ]; then
    print_header "Step 2: Ansible Playbook Deployment"

    PLAYBOOK="$PROJECT_ROOT/playbooks/complete-tier1-setup.yml"

    if [ ! -f "$PLAYBOOK" ]; then
        print_warning "Ansible playbook not found: $PLAYBOOK"
        print_info "Skipping ansible deployment"
    else
        print_step "Running Tier 1 deployment playbook..."
        print_info "This will install:"
        echo "  - Clang/LLVM 21 compiler toolchain"
        echo "  - libc++ standard library"
        echo "  - OpenMP, MPI (Open MPI 5.0)"
        echo "  - DuckDB, ninja, cmake"
        echo "  - Python development headers"
        echo ""

        # Check if running as root or with sudo
        if [ "$EUID" -eq 0 ]; then
            ansible-playbook "$PLAYBOOK"
        else
            print_info "Ansible requires sudo privileges for system packages"
            sudo ansible-playbook "$PLAYBOOK"
        fi

        print_success "Ansible deployment complete"

        # Source the environment file
        if [ -f /etc/profile.d/bigbrother_env.sh ]; then
            print_step "Loading environment variables..."
            source /etc/profile.d/bigbrother_env.sh
            print_success "Environment loaded"
        fi
    fi
else
    print_info "Skipping ansible playbook deployment (--skip-ansible)"
fi

# Step 3: Build C++ Project
if [ "$SKIP_BUILD" = false ]; then
    print_header "Step 3: Build C++ Project"

    cd "$PROJECT_ROOT"

    # Clean build directory
    print_step "Cleaning build directory..."
    rm -rf build
    mkdir -p build
    print_success "Build directory ready"

    # Run CMake
    print_step "Running CMake configuration..."
    cd build

    # Auto-detect compilers (portable approach)
    if [ -n "${CC:-}" ]; then
        export CMAKE_C_COMPILER="$CC"
    elif [ -f /usr/local/bin/clang ]; then
        export CMAKE_C_COMPILER=/usr/local/bin/clang
    fi

    if [ -n "${CXX:-}" ]; then
        export CMAKE_CXX_COMPILER="$CXX"
    elif [ -f /usr/local/bin/clang++ ]; then
        export CMAKE_CXX_COMPILER=/usr/local/bin/clang++
    fi

    cmake -G Ninja .. || {
        print_error "CMake configuration failed"
        exit 1
    }

    print_success "CMake configuration complete"

    # Build project
    print_step "Compiling C++ project (this may take 5-10 minutes)..."
    ninja || {
        print_error "Build failed"
        exit 1
    }

    print_success "C++ compilation complete"

    # Run tests if not skipped
    if [ "$SKIP_TESTS" = false ]; then
        print_step "Running C++ tests..."

        # Set library path for tests
        if [ -d /usr/local/lib/x86_64-unknown-linux-gnu ]; then
            export LD_LIBRARY_PATH=/usr/local/lib/x86_64-unknown-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH:-}
        elif [ -d /usr/local/lib ]; then
            export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
        fi

        ninja test || {
            print_warning "Some tests failed (continuing anyway)"
        }

        print_success "Tests complete"
    else
        print_info "Skipping tests (--skip-tests)"
    fi

    cd "$PROJECT_ROOT"
else
    print_info "Skipping C++ build (--skip-build)"
fi

# Step 4: Python Environment Setup
print_header "Step 4: Python Environment Setup"

cd "$PROJECT_ROOT"

print_step "Installing Python dependencies with uv..."
# uv automatically creates/uses .venv
uv sync || {
    print_error "Failed to install Python dependencies"
    exit 1
}

print_success "Python environment ready"

# Step 5: Database Initialization
print_header "Step 5: Database Initialization"

print_step "Creating data directories..."
mkdir -p data data/backups logs
print_success "Directories created"

print_step "Initializing tax database..."
uv run python scripts/monitoring/setup_tax_database.py || {
    print_warning "Tax database initialization failed (may already exist)"
}

print_step "Configuring tax rates (California, Married Filing Jointly)..."
uv run python scripts/monitoring/update_tax_rates_california.py || {
    print_warning "Tax configuration failed (may already be configured)"
}

print_success "Database initialization complete"

# Step 6: System Verification
print_header "Step 6: System Verification"

print_step "Running Phase 5 setup verification..."
uv run python scripts/phase5_setup.py --quick --skip-oauth || {
    print_warning "Phase 5 verification had warnings (check output above)"
}

print_success "System verification complete"

# Final Summary
print_header "Bootstrap Complete"

print_success "BigBrotherAnalytics is ready for deployment!"
echo ""
print_info "Next Steps:"
echo ""
echo "1. Configure API keys:"
echo "   - Copy api_keys.yaml.example to api_keys.yaml"
echo "   - Add your FRED API key: https://fred.stlouisfed.org"
echo "   - Add Schwab API credentials (if ready for live trading)"
echo ""
echo "2. Run OAuth authentication (for Schwab API):"
echo "   uv run python scripts/run_schwab_oauth_interactive.py"
echo ""
echo "3. Start Phase 5 paper trading:"
echo "   Morning:  uv run python scripts/phase5_setup.py --quick --start-all"
echo "   Evening:  uv run python scripts/phase5_shutdown.py"
echo ""
echo "4. Access dashboard:"
echo "   uv run streamlit run dashboard/app.py"
echo "   Then open: http://localhost:8501"
echo ""
echo "5. Manual trading engine start:"
echo "   cd build && LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/bigbrother"
echo ""

print_info "Documentation:"
echo "  - Phase 5 Guide: docs/PHASE5_SETUP_GUIDE.md"
echo "  - README:        README.md"
echo "  - Status:        docs/CURRENT_STATUS.md"
echo ""

print_success "Bootstrap completed successfully!"
echo ""
