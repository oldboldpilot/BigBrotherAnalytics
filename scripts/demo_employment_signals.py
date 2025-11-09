#!/usr/bin/env python3
"""
BigBrotherAnalytics: Employment Signal Demo

Quick demonstration of employment signal generation for decision engine.

Usage:
    python scripts/demo_employment_signals.py

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import sys
sys.path.insert(0, 'scripts')

from employment_signals import generate_employment_signals, generate_rotation_signals
import json


def main():
    print("\n" + "="*80)
    print("  BigBrotherAnalytics - Employment Signal Demo")
    print("="*80 + "\n")

    # Generate employment signals
    print("Generating employment signals from DuckDB...\n")
    signals = generate_employment_signals('data/bigbrother.duckdb')

    if signals:
        print(f"Generated {len(signals)} employment signals:\n")
        for sig in signals:
            direction = "BULLISH ↑" if sig['bullish'] else "BEARISH ↓"
            print(f"  • {sig['sector_name']} ({sig['sector_code']})")
            print(f"    Type: {sig['type']}")
            print(f"    Signal: {sig['signal_strength']:+.2f} | Confidence: {sig['confidence']:.0%}")
            print(f"    Direction: {direction}")
            print(f"    Rationale: {sig['rationale']}\n")
    else:
        print("  No employment signals above threshold.\n")

    # Generate rotation signals
    print("\nGenerating sector rotation signals...\n")
    rotation = generate_rotation_signals('data/bigbrother.duckdb')

    print(f"Top 5 Sectors (by employment score):\n")
    for i, sig in enumerate(rotation[:5], 1):
        print(f"  {i}. {sig['sector_name']} ({sig['sector_etf']})")
        print(f"     Employment Score: {sig['employment_score']:+.3f}")
        print(f"     Composite Score: {sig['composite_score']:+.3f}")
        print(f"     Action: {sig['action']} → {sig['target_allocation']:.1f}% allocation\n")

    print("\nBottom 3 Sectors:\n")
    for i, sig in enumerate(rotation[-3:], 1):
        print(f"  {i}. {sig['sector_name']} ({sig['sector_etf']})")
        print(f"     Employment Score: {sig['employment_score']:+.3f}")
        print(f"     Composite Score: {sig['composite_score']:+.3f}")
        print(f"     Action: {sig['action']} → {sig['target_allocation']:.1f}% allocation\n")

    # Example C++ integration output
    print("\n" + "="*80)
    print("  Example C++ Integration")
    print("="*80 + "\n")

    print("C++ Code:")
    print("-" * 80)
    print("""
    #include <bigbrother.employment.signals>

    // Initialize signal generator
    EmploymentSignalGenerator generator("scripts", "data/bigbrother.duckdb");

    // Generate signals
    auto signals = generator.generateSignals();
    auto rotation = generator.generateRotationSignals();

    // Process signals
    for (auto const& signal : signals) {
        if (signal.isActionable()) {
            std::cout << signal.sector_name << ": "
                     << signal.signal_strength << "\\n";
        }
    }

    // Use in strategy
    SectorRotationStrategy strategy;
    auto trading_signals = strategy.generateSignals(context);
    """)
    print("-" * 80)

    print("\nJSON Output (for C++ parsing):")
    print("-" * 80)
    print(json.dumps(rotation[:2], indent=2))
    print("-" * 80)

    print("\n" + "="*80)
    print("  Demo Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
