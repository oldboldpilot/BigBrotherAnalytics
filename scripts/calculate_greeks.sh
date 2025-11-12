#!/bin/bash
# Wrapper script to calculate option Greeks with proper library paths
# Sets LD_LIBRARY_PATH before running Python to ensure OpenMP library is found

# Library paths from Ansible playbook (playbooks/complete-tier1-setup.yml:881)
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-unknown-linux-gnu:/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH

# Run the Python script
uv run python "$(dirname "$0")/calculate_option_greeks.py" "$@"
