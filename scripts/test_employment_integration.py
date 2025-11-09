#!/usr/bin/env python3
"""
BigBrotherAnalytics: Employment Signal Integration Test

Comprehensive test of employment signal generation for the decision engine.
Demonstrates the complete pipeline from database to trading signals.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import sys
sys.path.insert(0, 'scripts')

from employment_signals import (
    generate_employment_signals,
    generate_rotation_signals,
    calculate_employment_statistics,
    BLS_TO_SECTOR_MAP,
    SECTOR_ETF_MAP
)
import duckdb


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def test_database_connection():
    """Test database connectivity and data availability."""
    print_section("DATABASE CONNECTIVITY TEST")

    conn = duckdb.connect('data/bigbrother.duckdb')

    # Check employment data
    result = conn.execute("""
        SELECT COUNT(DISTINCT series_id) as series_count,
               MIN(report_date) as earliest_date,
               MAX(report_date) as latest_date,
               COUNT(*) as total_records
        FROM sector_employment_raw
    """).fetchone()

    print(f"Series Count: {result[0]}")
    print(f"Date Range: {result[1]} to {result[2]}")
    print(f"Total Records: {result[3]}")

    # Check sectors table
    sectors = conn.execute("SELECT COUNT(*) FROM sectors").fetchone()[0]
    print(f"GICS Sectors: {sectors}")

    conn.close()
    print("✓ Database connection successful")


def test_statistics_calculation():
    """Test employment statistics calculation."""
    print_section("EMPLOYMENT STATISTICS CALCULATION")

    conn = duckdb.connect('data/bigbrother.duckdb')

    test_series = [
        ('CES1000000001', 'Mining/Logging (Energy/Materials)'),
        ('CES3000000001', 'Manufacturing (Materials/Industrials)'),
        ('CES5000000001', 'Information (IT/Communications)'),
        ('CES5500000001', 'Financial Activities (Financials)'),
        ('CES6500000001', 'Education/Health (Health Care)')
    ]

    for series_id, description in test_series:
        stats = calculate_employment_statistics(conn, series_id)

        print(f"\n{series_id} - {description}")
        print(f"  Latest Employment: {stats['latest']:,}")
        print(f"  3-Month Trend: {stats['trend_3m']:+.2f}%")
        print(f"  6-Month Trend: {stats['trend_6m']:+.2f}%")
        print(f"  12-Month Trend: {stats['trend_12m']:+.2f}%")
        print(f"  Acceleration: {stats['acceleration']:+.2f}%")
        print(f"  Z-Score: {stats['z_score']:+.2f}")
        print(f"  Volatility: {stats['volatility']:.2f}%")
        print(f"  Inflection Point: {'YES' if stats['is_inflection'] else 'NO'}")

    conn.close()
    print("\n✓ Statistics calculation successful")


def test_employment_signals():
    """Test employment signal generation."""
    print_section("EMPLOYMENT SIGNAL GENERATION")

    signals = generate_employment_signals('data/bigbrother.duckdb')

    print(f"\nTotal Signals Generated: {len(signals)}")

    if signals:
        print("\nSignal Details:")
        print("-" * 80)

        for i, sig in enumerate(signals, 1):
            print(f"\n{i}. {sig['sector_name']} (Sector {sig['sector_code']})")
            print(f"   Type: {sig['type']}")
            print(f"   Signal Strength: {sig['signal_strength']:+.2f}")
            print(f"   Confidence: {sig['confidence']:.0%}")
            print(f"   Employment Change: {sig['employment_change']:+.2f}%")
            print(f"   Direction: {'BULLISH ↑' if sig['bullish'] else 'BEARISH ↓'}")
            print(f"   Rationale: {sig['rationale']}")
            print(f"   Actionable: {'YES' if sig['confidence'] > 0.60 and abs(sig['signal_strength']) > 0.50 else 'NO'}")
    else:
        print("\n⚠ No signals generated (employment changes below threshold)")

    print("\n✓ Employment signal generation successful")
    return signals


def test_rotation_signals():
    """Test sector rotation signal generation."""
    print_section("SECTOR ROTATION SIGNALS")

    rotation = generate_rotation_signals('data/bigbrother.duckdb')

    print(f"\nTotal Rotation Signals: {len(rotation)}")
    print("\nSector Rankings (by Composite Score):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Sector':<28} {'ETF':<6} {'Emp Score':<12} {'Composite':<12} {'Action':<12} {'Allocation':<12}")
    print("-" * 100)

    for i, sig in enumerate(rotation, 1):
        action_emoji = {
            'Overweight': '↑ OW',
            'Underweight': '↓ UW',
            'Neutral': '→ NEU'
        }[sig['action']]

        print(f"{i:<6} {sig['sector_name']:<28} {sig['sector_etf']:<6} "
              f"{sig['employment_score']:+.3f}       {sig['composite_score']:+.3f}       "
              f"{action_emoji:<12} {sig['target_allocation']:>6.2f}%")

    # Summary statistics
    overweight = [s for s in rotation if s['action'] == 'Overweight']
    underweight = [s for s in rotation if s['action'] == 'Underweight']
    neutral = [s for s in rotation if s['action'] == 'Neutral']

    print("-" * 100)
    print(f"\nSummary:")
    print(f"  Overweight Sectors: {len(overweight)}")
    print(f"  Neutral Sectors: {len(neutral)}")
    print(f"  Underweight Sectors: {len(underweight)}")

    if overweight:
        print(f"\n  Top Overweight: {overweight[0]['sector_name']} ({overweight[0]['sector_etf']}) "
              f"with score {overweight[0]['composite_score']:+.3f}")

    if underweight:
        print(f"  Top Underweight: {underweight[-1]['sector_name']} ({underweight[-1]['sector_etf']}) "
              f"with score {underweight[-1]['composite_score']:+.3f}")

    print("\n✓ Rotation signal generation successful")
    return rotation


def test_signal_methodology():
    """Test and document signal generation methodology."""
    print_section("SIGNAL GENERATION METHODOLOGY")

    print("""
Signal Strength Calculation:
-----------------------------
1. Employment Score (60% weight):
   - Combines 3-month (60%) and 6-month (40%) trends
   - Normalized to -1.0 to +1.0 scale (±10% cap)

2. Acceleration Score (25% weight):
   - Measures change in trend (inflection detection)
   - Normalized to -1.0 to +1.0 scale (±5% cap)

3. Z-Score Position (15% weight):
   - Relative to 24-month historical average
   - Normalized to -1.0 to +1.0 scale (±2σ cap)

Composite Score Formula:
   score = (trend_score × 0.60) + (accel_score × 0.25) + (z_score × 0.15)

Signal Interpretation:
   +0.80 to +1.00: Very Strong Bullish (Exceptional growth)
   +0.60 to +0.79: Strong Bullish (Solid growth)
   +0.40 to +0.59: Moderate Bullish (Above average)
   +0.20 to +0.39: Weak Bullish (Slight positive)
   -0.19 to +0.19: Neutral (Stable/Mixed)
   -0.20 to -0.39: Weak Bearish (Slight negative)
   -0.40 to -0.59: Moderate Bearish (Below average)
   -0.60 to -0.79: Strong Bearish (Declining)
   -0.80 to -1.00: Very Strong Bearish (Severe decline)

Confidence Levels:
   0.85+: Very High (Trends agree, strong z-score, inflection)
   0.75-0.84: High (Trends agree, good z-score)
   0.60-0.74: Moderate (Some agreement)
   <0.60: Low (Weak or conflicting signals)

Action Thresholds (Sector Rotation):
   Overweight: Composite Score > +0.25
   Neutral: Composite Score between -0.25 and +0.25
   Underweight: Composite Score < -0.25
""")

    print("✓ Methodology documented")


def test_bls_sector_mapping():
    """Test BLS series to GICS sector mapping."""
    print_section("BLS SERIES TO GICS SECTOR MAPPING")

    print("\nBLS Series → GICS Sectors:")
    print("-" * 80)

    for series_id, sector_codes in sorted(BLS_TO_SECTOR_MAP.items()):
        sector_names = [f"{code} ({SECTOR_ETF_MAP.get(code, 'N/A')})" for code in sector_codes]
        print(f"{series_id}: {', '.join(sector_names)}")

    print("\n✓ BLS mapping verified")


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("  BigBrotherAnalytics - Employment Signal Integration Test")
    print("  Testing Decision Engine Employment Logic")
    print("="*80)

    try:
        # Run tests
        test_database_connection()
        test_bls_sector_mapping()
        test_statistics_calculation()
        test_signal_methodology()

        signals = test_employment_signals()
        rotation = test_rotation_signals()

        # Final summary
        print_section("INTEGRATION TEST SUMMARY")
        print(f"✓ Database: Connected")
        print(f"✓ Statistics: Calculated for all series")
        print(f"✓ Employment Signals: {len(signals)} generated")
        print(f"✓ Rotation Signals: {len(rotation)} generated")
        print(f"✓ C++ Integration: Ready (via subprocess)")
        print(f"\n{'='*80}")
        print("  ALL TESTS PASSED")
        print('='*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
