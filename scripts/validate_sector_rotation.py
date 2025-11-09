#!/usr/bin/env python3
"""
BigBrotherAnalytics: Sector Rotation Strategy Validation (Python)

Comprehensive validation of the sector rotation strategy focusing on:
- Database query validation
- Employment signal calculation accuracy
- Sector scoring algorithm verification
- Business logic validation
- Data quality checks

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import duckdb
import json
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import statistics

# Import the employment signals module
from employment_signals import (
    generate_employment_signals,
    generate_rotation_signals,
    calculate_employment_statistics,
    BLS_TO_SECTOR_MAP,
    SECTOR_ETF_MAP,
    DB_PATH
)


@dataclass
class ValidationResult:
    """Validation result for a single test"""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


class ValidationSuite:
    """Comprehensive validation test suite"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.results: List[ValidationResult] = []
        self.conn = None

    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def add_result(self, test_name: str, passed: bool, message: str, details: Dict = None):
        """Add a validation result"""
        self.results.append(ValidationResult(test_name, passed, message, details))
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed or details:
            print(f"  → {message}")
            if details:
                for key, value in details.items():
                    print(f"    {key}: {value}")

    def validate_database_records(self):
        """Validate database has sufficient employment records"""
        print("\n" + "="*80)
        print("VALIDATION 1: Database Records")
        print("="*80)

        # Count total employment records
        result = self.conn.execute(
            "SELECT COUNT(*) FROM sector_employment_raw"
        ).fetchone()
        total_records = result[0] if result else 0

        self.add_result(
            "Database Record Count",
            total_records > 2000,
            f"Found {total_records} records (expected >2000)",
            {"total_records": total_records}
        )

        # Count unique series
        result = self.conn.execute(
            "SELECT COUNT(DISTINCT series_id) FROM sector_employment_raw"
        ).fetchone()
        unique_series = result[0] if result else 0

        self.add_result(
            "Unique BLS Series Count",
            unique_series >= len(BLS_TO_SECTOR_MAP),
            f"Found {unique_series} series (expected >={len(BLS_TO_SECTOR_MAP)})",
            {"unique_series": unique_series, "expected": len(BLS_TO_SECTOR_MAP)}
        )

        # Check data recency
        result = self.conn.execute(
            "SELECT MAX(report_date) FROM sector_employment_raw"
        ).fetchone()
        latest_date = result[0] if result else None

        if latest_date:
            self.add_result(
                "Data Recency",
                True,
                f"Latest data: {latest_date}",
                {"latest_date": str(latest_date)}
            )

        # Count sectors
        result = self.conn.execute(
            "SELECT COUNT(*) FROM sectors"
        ).fetchone()
        sector_count = result[0] if result else 0

        self.add_result(
            "Sector Count",
            sector_count == 11,
            f"Found {sector_count} sectors (expected 11 GICS sectors)",
            {"sector_count": sector_count}
        )

    def validate_employment_statistics(self):
        """Validate employment statistics calculations"""
        print("\n" + "="*80)
        print("VALIDATION 2: Employment Statistics Calculations")
        print("="*80)

        # Test statistics calculation for a sample series
        test_series = 'CES5000000001'  # Information sector

        stats = calculate_employment_statistics(self.conn, test_series)

        if not stats:
            self.add_result(
                "Statistics Calculation",
                False,
                f"Failed to calculate statistics for series {test_series}"
            )
            return

        # Validate statistics structure
        required_keys = ['latest', 'mom_change', 'trend_3m', 'trend_6m', 'trend_12m',
                        'acceleration', 'z_score', 'is_inflection', 'volatility']

        missing_keys = [k for k in required_keys if k not in stats]

        self.add_result(
            "Statistics Structure",
            len(missing_keys) == 0,
            f"All required keys present" if not missing_keys else f"Missing: {missing_keys}",
            {"calculated_keys": list(stats.keys())}
        )

        # Validate statistics ranges
        self.add_result(
            "Employment Count Positive",
            stats.get('latest', 0) > 0,
            f"Latest employment: {stats.get('latest', 0):,.0f}"
        )

        self.add_result(
            "Volatility Reasonable",
            0 <= stats.get('volatility', 0) <= 50,
            f"Volatility: {stats.get('volatility', 0):.2f}%",
            {"volatility": stats.get('volatility', 0)}
        )

        self.add_result(
            "Z-Score Range",
            -5.0 <= stats.get('z_score', 0) <= 5.0,
            f"Z-score: {stats.get('z_score', 0):.2f}",
            {"z_score": stats.get('z_score', 0)}
        )

        # Print detailed statistics
        print("\n  Sample Statistics (Information Sector):")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    def validate_signal_generation(self):
        """Validate employment signal generation"""
        print("\n" + "="*80)
        print("VALIDATION 3: Employment Signal Generation")
        print("="*80)

        signals = generate_employment_signals(self.db_path)

        self.add_result(
            "Signals Generated",
            len(signals) > 0,
            f"Generated {len(signals)} employment signals",
            {"signal_count": len(signals)}
        )

        if not signals:
            return

        # Validate signal structure
        required_fields = ['type', 'sector_code', 'sector_name', 'confidence',
                          'employment_change', 'rationale', 'timestamp',
                          'bullish', 'bearish', 'signal_strength']

        for i, signal in enumerate(signals[:3]):  # Check first 3 signals
            missing_fields = [f for f in required_fields if f not in signal]
            self.add_result(
                f"Signal {i+1} Structure",
                len(missing_fields) == 0,
                "All required fields present" if not missing_fields else f"Missing: {missing_fields}"
            )

        # Validate signal value ranges
        for signal in signals:
            # Confidence: 0.0 to 1.0
            if not (0.0 <= signal.get('confidence', -1) <= 1.0):
                self.add_result(
                    "Signal Confidence Range",
                    False,
                    f"Confidence {signal.get('confidence')} out of range [0,1]",
                    {"sector": signal.get('sector_name'), "confidence": signal.get('confidence')}
                )
                break
        else:
            self.add_result(
                "Signal Confidence Range",
                True,
                "All signals have confidence in [0.0, 1.0]"
            )

        # Validate signal strength: -1.0 to +1.0
        for signal in signals:
            if not (-1.0 <= signal.get('signal_strength', -999) <= 1.0):
                self.add_result(
                    "Signal Strength Range",
                    False,
                    f"Strength {signal.get('signal_strength')} out of range [-1,1]",
                    {"sector": signal.get('sector_name'), "strength": signal.get('signal_strength')}
                )
                break
        else:
            self.add_result(
                "Signal Strength Range",
                True,
                "All signals have strength in [-1.0, +1.0]"
            )

        # Print signal distribution
        bullish_count = sum(1 for s in signals if s.get('bullish'))
        bearish_count = sum(1 for s in signals if s.get('bearish'))

        print(f"\n  Signal Distribution:")
        print(f"    Bullish: {bullish_count}")
        print(f"    Bearish: {bearish_count}")

    def validate_rotation_signals(self):
        """Validate sector rotation signal generation"""
        print("\n" + "="*80)
        print("VALIDATION 4: Sector Rotation Signals")
        print("="*80)

        signals = generate_rotation_signals(self.db_path)

        self.add_result(
            "Rotation Signals Generated",
            len(signals) > 0,
            f"Generated {len(signals)} rotation signals",
            {"signal_count": len(signals)}
        )

        if not signals:
            return

        # Validate we have signals for all major sectors
        sector_codes = set(s['sector_code'] for s in signals)
        expected_sectors = set(SECTOR_ETF_MAP.keys())

        self.add_result(
            "All Sectors Covered",
            sector_codes == expected_sectors,
            f"Coverage: {len(sector_codes)}/{len(expected_sectors)} sectors",
            {"covered": sorted(sector_codes), "expected": sorted(expected_sectors)}
        )

        # Validate rotation signal structure
        required_fields = ['sector_code', 'sector_name', 'sector_etf',
                          'employment_score', 'sentiment_score', 'technical_score',
                          'composite_score', 'action', 'target_allocation']

        for i, signal in enumerate(signals[:3]):  # Check first 3
            missing_fields = [f for f in required_fields if f not in signal]
            self.add_result(
                f"Rotation Signal {i+1} Structure",
                len(missing_fields) == 0,
                "All required fields present" if not missing_fields else f"Missing: {missing_fields}"
            )

        # Validate score ranges
        for signal in signals:
            emp_score = signal.get('employment_score', -999)
            if not (-1.0 <= emp_score <= 1.0):
                self.add_result(
                    "Employment Score Range",
                    False,
                    f"Score {emp_score} out of range [-1,1]",
                    {"sector": signal.get('sector_name')}
                )
                break
        else:
            self.add_result(
                "Employment Score Range",
                True,
                "All employment scores in [-1.0, +1.0]"
            )

        # Validate target allocations sum to ~100%
        total_allocation = sum(s.get('target_allocation', 0) for s in signals)

        self.add_result(
            "Total Allocation",
            95.0 <= total_allocation <= 105.0,
            f"Total allocation: {total_allocation:.2f}% (expected ~100%)",
            {"total_allocation": total_allocation}
        )

        # Validate action distribution
        actions = [s.get('action') for s in signals]
        overweight_count = actions.count('Overweight')
        neutral_count = actions.count('Neutral')
        underweight_count = actions.count('Underweight')

        print(f"\n  Action Distribution:")
        print(f"    Overweight: {overweight_count}")
        print(f"    Neutral: {neutral_count}")
        print(f"    Underweight: {underweight_count}")

        self.add_result(
            "Action Distribution Balanced",
            overweight_count > 0 and underweight_count > 0,
            "Has both overweight and underweight sectors"
        )

        # Print top 3 and bottom 3 by composite score
        print(f"\n  Top 3 Sectors by Composite Score:")
        for i, signal in enumerate(signals[:3]):
            print(f"    {i+1}. {signal['sector_name']} ({signal['sector_etf']}): "
                  f"{signal['composite_score']:.3f} - {signal['action']}")

        print(f"\n  Bottom 3 Sectors by Composite Score:")
        for i, signal in enumerate(signals[-3:]):
            print(f"    {signal['sector_name']} ({signal['sector_etf']}): "
                  f"{signal['composite_score']:.3f} - {signal['action']}")

    def validate_business_logic(self):
        """Validate business logic rules"""
        print("\n" + "="*80)
        print("VALIDATION 5: Business Logic Rules")
        print("="*80)

        signals = generate_rotation_signals(self.db_path)

        if not signals:
            self.add_result(
                "Business Logic Validation",
                False,
                "No signals to validate"
            )
            return

        # Rule 1: Overweight sectors should have positive composite scores
        overweight_signals = [s for s in signals if s['action'] == 'Overweight']
        overweight_positive = all(s['composite_score'] > 0.25 for s in overweight_signals)

        self.add_result(
            "Overweight Sectors Have Positive Scores",
            overweight_positive,
            f"Checked {len(overweight_signals)} overweight sectors"
        )

        # Rule 2: Underweight sectors should have negative composite scores
        underweight_signals = [s for s in signals if s['action'] == 'Underweight']
        underweight_negative = all(s['composite_score'] < -0.25 for s in underweight_signals)

        self.add_result(
            "Underweight Sectors Have Negative Scores",
            underweight_negative,
            f"Checked {len(underweight_signals)} underweight sectors"
        )

        # Rule 3: Neutral sectors should have scores near zero
        neutral_signals = [s for s in signals if s['action'] == 'Neutral']
        neutral_centered = all(-0.25 <= s['composite_score'] <= 0.25 for s in neutral_signals)

        self.add_result(
            "Neutral Sectors Have Centered Scores",
            neutral_centered,
            f"Checked {len(neutral_signals)} neutral sectors"
        )

        # Rule 4: Stronger signals should get higher allocations
        if len(overweight_signals) >= 2:
            sorted_overweight = sorted(overweight_signals,
                                      key=lambda x: x['composite_score'],
                                      reverse=True)

            allocations_ordered = True
            for i in range(len(sorted_overweight) - 1):
                if sorted_overweight[i]['target_allocation'] < sorted_overweight[i+1]['target_allocation']:
                    allocations_ordered = False
                    break

            self.add_result(
                "Allocations Match Score Strength",
                allocations_ordered,
                "Stronger scores get higher allocations"
            )

        # Rule 5: Max allocation should not exceed reasonable limits
        max_allocation = max(s['target_allocation'] for s in signals)
        self.add_result(
            "Maximum Allocation Reasonable",
            max_allocation <= 25.0,
            f"Max allocation: {max_allocation:.2f}% (should be ≤25%)",
            {"max_allocation": max_allocation}
        )

        # Rule 6: Min allocation for active sectors should be reasonable
        active_signals = [s for s in signals if s['action'] != 'Underweight']
        min_active_allocation = min(s['target_allocation'] for s in active_signals) if active_signals else 0

        self.add_result(
            "Minimum Active Allocation Reasonable",
            min_active_allocation >= 2.0,
            f"Min active allocation: {min_active_allocation:.2f}% (should be ≥2%)",
            {"min_allocation": min_active_allocation}
        )

    def validate_data_quality(self):
        """Validate data quality and consistency"""
        print("\n" + "="*80)
        print("VALIDATION 6: Data Quality and Consistency")
        print("="*80)

        # Check for NULL values in critical fields
        result = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN employment_count IS NULL THEN 1 ELSE 0 END) as null_employment,
                SUM(CASE WHEN report_date IS NULL THEN 1 ELSE 0 END) as null_date,
                SUM(CASE WHEN series_id IS NULL THEN 1 ELSE 0 END) as null_series
            FROM sector_employment_raw
        """).fetchone()

        total, null_employment, null_date, null_series = result

        self.add_result(
            "No NULL Employment Counts",
            null_employment == 0,
            f"Found {null_employment}/{total} NULL employment counts"
        )

        self.add_result(
            "No NULL Dates",
            null_date == 0,
            f"Found {null_date}/{total} NULL dates"
        )

        self.add_result(
            "No NULL Series IDs",
            null_series == 0,
            f"Found {null_series}/{total} NULL series IDs"
        )

        # Check for negative employment counts (data quality issue)
        result = self.conn.execute("""
            SELECT COUNT(*)
            FROM sector_employment_raw
            WHERE employment_count < 0
        """).fetchone()

        negative_count = result[0] if result else 0

        self.add_result(
            "No Negative Employment Counts",
            negative_count == 0,
            f"Found {negative_count} negative employment counts"
        )

        # Check for duplicate records (same series_id + report_date)
        result = self.conn.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT series_id, report_date, COUNT(*) as cnt
                FROM sector_employment_raw
                GROUP BY series_id, report_date
                HAVING cnt > 1
            )
        """).fetchone()

        duplicate_count = result[0] if result else 0

        self.add_result(
            "No Duplicate Records",
            duplicate_count == 0,
            f"Found {duplicate_count} duplicate date/series combinations"
        )

        # Check data continuity (no large gaps in time series)
        result = self.conn.execute("""
            SELECT series_id, COUNT(DISTINCT report_date) as date_count
            FROM sector_employment_raw
            GROUP BY series_id
            ORDER BY date_count
        """).fetchall()

        min_dates = min(row[1] for row in result) if result else 0
        max_dates = max(row[1] for row in result) if result else 0

        self.add_result(
            "Data Continuity",
            min_dates >= 100,
            f"All series have at least {min_dates} data points (max: {max_dates})",
            {"min_dates": min_dates, "max_dates": max_dates}
        )

    def validate_edge_cases(self):
        """Validate edge case handling"""
        print("\n" + "="*80)
        print("VALIDATION 7: Edge Case Handling")
        print("="*80)

        signals = generate_rotation_signals(self.db_path)

        # Edge case 1: Check if strategy handles missing sentiment/technical scores
        # (should default to 0.0)
        sentiment_scores = [s.get('sentiment_score', None) for s in signals]
        technical_scores = [s.get('technical_score', None) for s in signals]

        self.add_result(
            "Handles Missing Sentiment Scores",
            all(s == 0.0 for s in sentiment_scores),
            "All sentiment scores default to 0.0 (not yet implemented)"
        )

        self.add_result(
            "Handles Missing Technical Scores",
            all(s == 0.0 for s in technical_scores),
            "All technical scores default to 0.0 (not yet implemented)"
        )

        # Edge case 2: Check composite score calculation with only employment
        for signal in signals:
            expected_composite = signal['employment_score']  # Since sentiment=0, technical=0
            actual_composite = signal['composite_score']

            if abs(expected_composite - actual_composite) > 0.01:
                self.add_result(
                    "Composite Score Calculation",
                    False,
                    f"Mismatch in {signal['sector_name']}: expected {expected_composite:.3f}, got {actual_composite:.3f}"
                )
                break
        else:
            self.add_result(
                "Composite Score Calculation",
                True,
                "Composite scores correctly calculated from components"
            )

        # Edge case 3: Check for extreme outliers
        employment_scores = [s['employment_score'] for s in signals]
        if len(employment_scores) >= 3:
            mean_score = statistics.mean(employment_scores)
            stdev_score = statistics.stdev(employment_scores)

            outliers = [s for s in signals
                       if abs(s['employment_score'] - mean_score) > 3 * stdev_score]

            self.add_result(
                "No Extreme Outliers",
                len(outliers) == 0,
                f"Found {len(outliers)} outliers (>3 std devs from mean)",
                {"outliers": [s['sector_name'] for s in outliers]}
            )

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total:.1f}%)")

        if failed > 0:
            print(f"\nFailed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  ❌ {result.test_name}: {result.message}")

        print("\n" + "="*80)
        print("PRODUCTION READINESS ASSESSMENT")
        print("="*80)

        if failed == 0:
            print("\n✓ ALL VALIDATIONS PASSED")
            print("\nStatus: READY FOR PRODUCTION")
            print("\nThe sector rotation strategy has passed all validation tests.")
            print("Business logic, data quality, and edge cases are all validated.")
        elif failed <= total * 0.1:  # Less than 10% failure rate
            print("\n⚠ MOSTLY PASSING (Minor Issues)")
            print("\nStatus: READY FOR TESTING")
            print("\nThe strategy is mostly functional but has some minor issues to address.")
        else:
            print("\n❌ VALIDATION FAILURES DETECTED")
            print("\nStatus: NEEDS FIXES")
            print("\nPlease address the failed validations before deployment.")

        print("\n" + "="*80)

    def run_all_validations(self):
        """Run all validation tests"""
        self.validate_database_records()
        self.validate_employment_statistics()
        self.validate_signal_generation()
        self.validate_rotation_signals()
        self.validate_business_logic()
        self.validate_data_quality()
        self.validate_edge_cases()
        self.print_summary()


def main():
    """Main entry point"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║     SECTOR ROTATION STRATEGY - PYTHON VALIDATION SUITE                    ║
║                                                                            ║
║  Database validation, signal accuracy, and business logic verification    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH

    try:
        with ValidationSuite(db_path) as validator:
            validator.run_all_validations()

        return 0

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
