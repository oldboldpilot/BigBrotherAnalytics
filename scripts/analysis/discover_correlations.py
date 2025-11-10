#!/usr/bin/env python3
"""
BigBrotherAnalytics: Time-Lagged Correlation Discovery

Discovers lead-lag relationships between GICS sectors using employment data.
Calculates Pearson and Spearman correlations at multiple time lags.

Key Features:
- Analyzes all 55 unique sector pairs (11 choose 2)
- Tests lags: 0, 7, 14, 30, 60, 90 days
- Computes statistical significance (p-values)
- Stores results in DuckDB for signal enhancement

Author: Agent 6 - Correlation Discovery Agent
Date: 2025-11-10
"""

import duckdb
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from itertools import combinations
from scipy import stats
from pathlib import Path

# Configuration
DB_PATH = "data/bigbrother.duckdb"
LAGS_TO_TEST = [0, 7, 14, 30, 60, 90]  # Days
MIN_CORRELATION = 0.5  # Only store correlations with |r| > 0.5
SIGNIFICANCE_LEVEL = 0.05  # p < 0.05
MIN_SAMPLE_SIZE = 20  # Minimum data points required

class CorrelationDiscovery:
    """Discovers time-lagged correlations between sectors."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.sectors = self._load_sectors()
        self.employment_data = self._load_employment_data()

    def _load_sectors(self) -> List[Tuple[int, str, str]]:
        """Load all GICS sectors."""
        sectors = self.conn.execute("""
            SELECT sector_code, sector_name, sector_etf
            FROM sectors
            ORDER BY sector_code
        """).fetchall()
        print(f"Loaded {len(sectors)} GICS sectors")
        return sectors

    def _load_employment_data(self) -> pd.DataFrame:
        """Load employment data for all sectors."""
        query = """
            SELECT
                sector_code,
                report_date,
                employment_count,
                ROW_NUMBER() OVER (PARTITION BY sector_code, report_date ORDER BY created_at DESC) as rn
            FROM sector_employment
            ORDER BY sector_code, report_date
        """
        df = self.conn.execute(query).df()
        # Remove duplicates, keep most recent
        df = df[df['rn'] == 1].drop('rn', axis=1)
        print(f"Loaded employment data: {len(df)} records")
        return df

    def _calculate_growth_rates(self, sector_code: int) -> pd.DataFrame:
        """Calculate employment growth rates for a sector."""
        df = self.employment_data[self.employment_data['sector_code'] == sector_code].copy()
        df = df.sort_values('report_date')

        # Calculate month-over-month growth rate
        df['growth_rate'] = df['employment_count'].pct_change() * 100

        # Remove NaN values
        df = df.dropna(subset=['growth_rate'])

        return df[['report_date', 'growth_rate']]

    def _calculate_correlation(
        self,
        series1: pd.DataFrame,
        series2: pd.DataFrame,
        lag_days: int,
        method: str = 'pearson'
    ) -> Tuple[float, float, int]:
        """
        Calculate correlation between two time series with lag.

        Args:
            series1: Leading series (date, value)
            series2: Lagging series (date, value)
            lag_days: Number of days to lag series2
            method: 'pearson' or 'spearman'

        Returns:
            (correlation, p_value, sample_size)
        """
        # Shift series2 by lag_days
        series2_shifted = series2.copy()
        series2_shifted['report_date'] = series2_shifted['report_date'] - pd.Timedelta(days=lag_days)

        # Merge on date
        merged = pd.merge(
            series1,
            series2_shifted,
            on='report_date',
            suffixes=('_1', '_2')
        )

        if len(merged) < MIN_SAMPLE_SIZE:
            return 0.0, 1.0, len(merged)

        values1 = merged['growth_rate_1'].values
        values2 = merged['growth_rate_2'].values

        # Calculate correlation
        if method == 'pearson':
            corr, p_value = stats.pearsonr(values1, values2)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(values1, values2)
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr, p_value, len(merged)

    def discover_all_correlations(self) -> List[Dict]:
        """
        Discover correlations for all sector pairs at all lags.

        Returns:
            List of correlation results
        """
        results = []
        sector_pairs = list(combinations(self.sectors, 2))
        total_pairs = len(sector_pairs)

        print(f"\n{'='*80}")
        print(f"CORRELATION DISCOVERY")
        print(f"{'='*80}")
        print(f"Analyzing {total_pairs} sector pairs at {len(LAGS_TO_TEST)} time lags")
        print(f"Total correlations to compute: {total_pairs * len(LAGS_TO_TEST) * 2}")
        print(f"{'='*80}\n")

        for idx, (sector1, sector2) in enumerate(sector_pairs, 1):
            code1, name1, etf1 = sector1
            code2, name2, etf2 = sector2

            print(f"[{idx}/{total_pairs}] {name1} ({etf1}) vs {name2} ({etf2})")

            # Get growth rates
            growth1 = self._calculate_growth_rates(code1)
            growth2 = self._calculate_growth_rates(code2)

            if len(growth1) < MIN_SAMPLE_SIZE or len(growth2) < MIN_SAMPLE_SIZE:
                print(f"  ⚠ Insufficient data (need {MIN_SAMPLE_SIZE} points)")
                continue

            # Test each lag
            for lag in LAGS_TO_TEST:
                # Test both directions
                for method in ['pearson', 'spearman']:
                    # Direction 1: sector1 leads sector2
                    corr, p_val, n = self._calculate_correlation(
                        growth1, growth2, lag, method
                    )

                    if abs(corr) > MIN_CORRELATION and p_val < SIGNIFICANCE_LEVEL:
                        direction = "leads" if lag > 0 else "contemporaneous"
                        print(f"  ✓ {method.capitalize()}: r={corr:+.3f} (p={p_val:.4f}, lag={lag}d, n={n}) - {name1} {direction} {name2}")

                        results.append({
                            'sector_code_1': code1,
                            'sector_code_2': code2,
                            'lag_days': lag,
                            'correlation_coefficient': corr,
                            'correlation_type': method,
                            'p_value': p_val,
                            'sample_size': n
                        })

                    # Direction 2: sector2 leads sector1 (only if lag > 0)
                    if lag > 0:
                        corr, p_val, n = self._calculate_correlation(
                            growth2, growth1, lag, method
                        )

                        if abs(corr) > MIN_CORRELATION and p_val < SIGNIFICANCE_LEVEL:
                            print(f"  ✓ {method.capitalize()}: r={corr:+.3f} (p={p_val:.4f}, lag={lag}d, n={n}) - {name2} leads {name1}")

                            results.append({
                                'sector_code_1': code2,
                                'sector_code_2': code1,
                                'lag_days': lag,
                                'correlation_coefficient': corr,
                                'correlation_type': method,
                                'p_value': p_val,
                                'sample_size': n
                            })

        return results

    def store_correlations(self, results: List[Dict]) -> int:
        """
        Store correlation results in DuckDB.

        Returns:
            Number of correlations stored
        """
        if not results:
            print("No correlations to store")
            return 0

        # Clear existing correlations
        self.conn.execute("DELETE FROM sector_correlations")

        # Insert new correlations one by one
        for i, result in enumerate(results, 1):
            self.conn.execute("""
                INSERT INTO sector_correlations
                (id, sector_code_1, sector_code_2, lag_days, correlation_coefficient,
                 correlation_type, p_value, sample_size, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                i,
                result['sector_code_1'],
                result['sector_code_2'],
                result['lag_days'],
                result['correlation_coefficient'],
                result['correlation_type'],
                result['p_value'],
                result['sample_size'],
                datetime.now()
            ])

        print(f"\n✓ Stored {len(results)} correlations in DuckDB")
        return len(results)

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of discovered correlations."""
        stats_dict = {}

        # Total correlations
        total = self.conn.execute("SELECT COUNT(*) FROM sector_correlations").fetchone()[0]
        stats_dict['total_correlations'] = total

        # By correlation type
        by_type = self.conn.execute("""
            SELECT correlation_type, COUNT(*) as count
            FROM sector_correlations
            GROUP BY correlation_type
        """).fetchall()
        stats_dict['by_type'] = {t: c for t, c in by_type}

        # By lag
        by_lag = self.conn.execute("""
            SELECT lag_days, COUNT(*) as count
            FROM sector_correlations
            GROUP BY lag_days
            ORDER BY lag_days
        """).fetchall()
        stats_dict['by_lag'] = {lag: count for lag, count in by_lag}

        # Strength distribution
        strength = self.conn.execute("""
            SELECT
                COUNT(CASE WHEN ABS(correlation_coefficient) > 0.7 THEN 1 END) as strong,
                COUNT(CASE WHEN ABS(correlation_coefficient) BETWEEN 0.5 AND 0.7 THEN 1 END) as moderate
            FROM sector_correlations
        """).fetchone()
        stats_dict['strength'] = {'strong': strength[0], 'moderate': strength[1]}

        # Top correlations
        top = self.conn.execute("""
            SELECT
                s1.sector_name as sector_1,
                s2.sector_name as sector_2,
                correlation_coefficient,
                lag_days,
                correlation_type
            FROM sector_correlations sc
            JOIN sectors s1 ON sc.sector_code_1 = s1.sector_code
            JOIN sectors s2 ON sc.sector_code_2 = s2.sector_code
            ORDER BY ABS(correlation_coefficient) DESC
            LIMIT 10
        """).fetchall()
        stats_dict['top_10'] = top

        return stats_dict

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main execution function."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SECTOR CORRELATION DISCOVERY AGENT                        ║
║                                                                              ║
║  Discovering time-lagged correlations between 11 GICS sectors               ║
║  Using employment growth rates as primary signal                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize discovery engine
    discovery = CorrelationDiscovery()

    try:
        # Discover correlations
        results = discovery.discover_all_correlations()

        # Store results
        count = discovery.store_correlations(results)

        # Get summary statistics
        stats = discovery.get_summary_statistics()

        # Print summary
        print(f"\n{'='*80}")
        print(f"DISCOVERY SUMMARY")
        print(f"{'='*80}")
        print(f"Total correlations found: {stats['total_correlations']}")
        print(f"\nBy correlation type:")
        for ctype, count in stats['by_type'].items():
            print(f"  {ctype.capitalize()}: {count}")

        print(f"\nBy time lag:")
        for lag, count in stats['by_lag'].items():
            print(f"  {lag} days: {count}")

        print(f"\nBy strength:")
        print(f"  Strong (|r| > 0.7): {stats['strength']['strong']}")
        print(f"  Moderate (0.5 < |r| ≤ 0.7): {stats['strength']['moderate']}")

        print(f"\nTop 10 Strongest Correlations:")
        print(f"{'-'*80}")
        for i, (s1, s2, corr, lag, ctype) in enumerate(stats['top_10'], 1):
            lag_str = f"lag={lag}d" if lag > 0 else "contemporaneous"
            print(f"{i:2}. {s1:<25} → {s2:<25} r={corr:+.3f} ({lag_str}, {ctype})")

        print(f"\n{'='*80}")
        print(f"✓ Correlation discovery complete!")
        print(f"✓ Results stored in: {DB_PATH}")
        print(f"{'='*80}")

        return stats

    finally:
        discovery.close()


if __name__ == "__main__":
    main()
