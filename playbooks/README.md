# BigBrotherAnalytics - Ansible Playbooks

This directory contains Ansible playbooks for automated deployment and configuration of the BigBrotherAnalytics platform.

## Available Playbooks

### 1. Tier 1 Setup Playbook
**File:** `tier1-setup.yml`

Complete Tier 1 development environment setup including:
- Homebrew installation and configuration
- GCC 15+ with C++23 support
- Python 3.14+ with uv package manager
- OpenMPI 5.x
- **UPC++ and Berkeley Distributed Components (PGAS)**
- PostgreSQL 16+ with extensions
- Redis cache
- Project environment setup

**Usage:**
```bash
ansible-playbook playbooks/tier1-setup.yml
```

### 2. UPC++ and Berkeley Components Installation
**File:** `install-upcxx-berkeley.yml`

**CRITICAL:** Installs complete Berkeley PGAS (Partitioned Global Address Space) stack for high-performance distributed computing.

**Components Installed:**
- **GASNet-EX 2024.5.0** - High-performance communication layer
- **UPC++ 2024.3.0** - Modern C++ PGAS programming model
- **Berkeley UPC (BUPC)** - Optional, for legacy code support
- **OpenSHMEM 1.5.2** - Optional PGAS alternative

**Installation Details:**
- Based on proven installation procedures from [ClusterSetupAndConfigs](https://github.com/oldboldpilot/ClusterSetupAndConfigs)
- Builds from source with optimizations
- Configured for MPI conduit (highest performance)
- Automatically sets environment variables
- Includes verification and testing

**Usage:**
```bash
# Standalone installation
ansible-playbook playbooks/install-upcxx-berkeley.yml

# With options
ansible-playbook playbooks/install-upcxx-berkeley.yml \
  -e "install_berkeley_upc=true" \
  -e "install_openshmem=true" \
  -e "num_build_jobs=32"
```

**Environment Variables Set:**
```bash
export BERKELEY_PREFIX=/opt/berkeley
export GASNET_HOME=/opt/berkeley/gasnet
export UPCXX_INSTALL=/opt/berkeley/upcxx
export PATH=/opt/berkeley/upcxx/bin:$PATH
export LD_LIBRARY_PATH=/opt/berkeley/upcxx/lib:$LD_LIBRARY_PATH
export UPCXX_GASNET_CONDUIT=mpi
export UPCXX_NETWORK=mpi
```

**After Installation:**
```bash
# Source environment
source /etc/profile.d/berkeley_components.sh

# Compile UPC++ program
upcxx -std=c++23 myprogram.cpp -o myprogram

# Run with 4 ranks
upcxx-run -n 4 ./myprogram
```

### 3. Complete Platform Setup (Future)
**File:** `complete-platform-setup.yml` (To be created)

Will include:
- All Tier 1 components
- All three tool deployments
- Database initialization
- Service configuration

## Reference Documentation

### Complete Installation Guide
For comprehensive cluster setup, advanced configurations, and troubleshooting:

**Repository:** https://github.com/oldboldpilot/ClusterSetupAndConfigs

**Key Documents:**
- `DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `cluster_modules/` - Modular manager implementations
- `cluster_modules/REFACTORING_DOCUMENTATION.md` - Architecture details

### PGAS Programming Models

**UPC++ (Recommended for BigBrotherAnalytics):**
- Modern C++23-based PGAS model
- One-sided communication (RDMA)
- Asynchronous operations
- Excellent for distributed correlation calculations
- Best performance for our use case

**GASNet-EX:**
- Communication substrate for UPC++
- Supports multiple network types (MPI, UDP, IBV)
- Optimized for RDMA and one-sided operations
- Low-latency, high-bandwidth

**Berkeley UPC (Optional):**
- C-based PGAS (legacy)
- Only needed for legacy codebases
- Skip unless specifically required

## Deployment Order

**Recommended order for complete platform setup:**

1. **Base system setup**
   ```bash
   # Install OS (RHEL 9+ or Ubuntu 22.04)
   # Update system packages
   ```

2. **Homebrew and toolchain**
   ```bash
   ansible-playbook playbooks/tier1-setup.yml
   ```
   This includes UPC++ installation automatically.

3. **Verify installation**
   ```bash
   source /etc/profile.d/berkeley_components.sh
   gcc-15 --version
   mpirun --version
   upcxx --version
   ```

4. **Build BigBrotherAnalytics components**
   ```bash
   cd /opt/bigbrother
   source .venv/bin/activate
   cd src/cpp
   cmake -B build -G Ninja \
       -DCMAKE_CXX_COMPILER=g++-15 \
       -DCMAKE_CXX_STANDARD=23 \
       -DENABLE_OPENMP=ON \
       -DENABLE_MPI=ON \
       -DENABLE_UPCXX=ON \
       -DENABLE_CUDA=ON
   cmake --build build -j $(nproc)
   ```

## Troubleshooting

### UPC++ Installation Issues

**Problem:** GASNet-EX configure fails
**Solution:** Ensure MPI is installed first
```bash
brew install open-mpi
which mpicc  # Verify MPI compiler available
```

**Problem:** UPC++ compilation fails with C++23
**Solution:** Verify GCC 15+ is installed and set as default
```bash
brew install gcc@15
export CC=$(brew --prefix)/bin/gcc-15
export CXX=$(brew --prefix)/bin/g++-15
```

**Problem:** upcxx-run fails with network errors
**Solution:** Check GASNet conduit configuration
```bash
export UPCXX_GASNET_CONDUIT=mpi  # Use MPI conduit (most compatible)
export GASNET_VERBOSEENV=1       # Debug output
```

### Getting Help

1. **BigBrotherAnalytics Issues:** https://github.com/oldboldpilot/BigBrotherAnalytics/issues
2. **Cluster Setup Issues:** https://github.com/oldboldpilot/ClusterSetupAndConfigs/issues
3. **UPC++ Documentation:** https://upcxx.lbl.gov/
4. **GASNet-EX Documentation:** https://gasnet.lbl.gov/

## Performance Optimization

After installation, optimize for BigBrotherAnalytics workloads:

```bash
# Enable hugepages for better memory performance
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages

# Set CPU governor to performance mode
sudo cpupower frequency-set -g performance

# Disable CPU C-states for lower latency
sudo cpupower idle-set -D 0
```

## Cluster Configuration

For multi-node cluster setups:
- See [ClusterSetupAndConfigs](https://github.com/oldboldpilot/ClusterSetupAndConfigs) for complete cluster management
- Includes SSH configuration, NFS setup, and distributed job scheduling
- BigBrotherAnalytics currently targets single-node deployment (Tier 1/2)
- Multi-node expansion possible in Tier 3

---

**Maintained by:** Olumuyiwa Oluwasanmi
**Last Updated:** November 6, 2025
**Related Repositories:**
- BigBrotherAnalytics: https://github.com/oldboldpilot/BigBrotherAnalytics
- ClusterSetupAndConfigs: https://github.com/oldboldpilot/ClusterSetupAndConfigs
