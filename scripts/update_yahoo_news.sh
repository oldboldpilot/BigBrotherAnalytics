#!/bin/bash
# Wrapper script to fetch Yahoo Finance news with C++ sentiment analyzer
# Sets LD_LIBRARY_PATH before running Python to ensure C++ modules are found

# Library paths from Ansible playbook (playbooks/complete-tier1-setup.yml:881)
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-unknown-linux-gnu:/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH

# Run the Python script with --news-only flag
uv run python "$(dirname "$0")/update_yahoo_prices.py" --news-only "$@"
