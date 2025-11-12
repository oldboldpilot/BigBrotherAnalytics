#!/usr/bin/env python3
"""
Test script for risk_bindings.cpp

Tests the wired C++ implementation by attempting to load the module
and documenting what works vs. what needs library dependencies.
"""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

print("=" * 80)
print("Risk Bindings Test - Implementation Verification")
print("=" * 80)
print()

# Try to import and test
try:
    print("Attempting to load bigbrother_risk module...")
    import bigbrother_risk as risk
    print("✅ Module loaded successfully!")
    print()

    print("Module documentation:")
    print(risk.__doc__)
    print()

    print("Testing kelly_criterion...")
    kelly = risk.kelly_criterion(0.65, 2.0)
    print(f"✅ Kelly Criterion (0.65, 2.0) = {kelly:.4f}")
    print()

    print("Testing position_size...")
    size = risk.position_size(30000, kelly, 0.05)
    print(f"✅ Position size ($30000, {kelly:.4f}, 5%) = ${size:.2f}")
    print()

    print("Testing monte_carlo...")
    result = risk.monte_carlo(100, 0.25, 0.05, 10000)
    print(f"✅ Monte Carlo simulation completed")
    print(f"   Expected Value: ${result.expected_value:.2f}")
    print(f"   Std Deviation: ${result.std_deviation:.2f}")
    print(f"   Probability of Profit: {result.probability_of_profit:.1%}")
    print(f"   95% VaR: ${result.var_95:.2f}")
    print(f"   99% VaR: ${result.var_99:.2f}")
    print(f"   Max Profit: ${result.max_profit:.2f}")
    print(f"   Max Loss: ${result.max_loss:.2f}")
    print()

    print("=" * 80)
    print("SUCCESS: All functions are properly wired to C++ implementation!")
    print("=" * 80)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print()
    print("Required libraries:")
    print("  - librisk_management.so")
    print("  - libutils.so")
    print("  - liboptions_pricing.so")
    print()
    print("Library search path needs:")
    print("  export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH")
    print()
    print("Or rebuild the project completely to generate all libraries.")
    print()

except Exception as e:
    print(f"❌ Runtime error: {e}")
    print()
    import traceback
    traceback.print_exc()
