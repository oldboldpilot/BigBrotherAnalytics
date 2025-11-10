#!/usr/bin/env python3
"""
BigBrotherAnalytics: Correlation-Based Signal Enhancement

Integration functions for using sector correlations to enhance trading signals.
Enables predictive signals based on leading sector relationships.

Key Functions:
- get_leading_sectors(): Find sectors that lead a given sector
- get_lagging_sectors(): Find sectors that lag a given sector
- get_correlation_strength(): Get correlation coefficient between sectors
- enhance_signal(): Amplify/dampen signals based on correlated sectors

Author: Agent 6 - Correlation Discovery Agent
Date: 2025-11-10
"""

import duckdb
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

DB_PATH = "data/bigbrother.duckdb"


@dataclass
class CorrelationSignal:
    """Represents a correlation-based signal."""
    sector_code: int
    sector_name: str
    sector_etf: str
    correlation: float
    lag_days: int
    correlation_type: str
    p_value: float
    sample_size: int
    signal_strength: str  # 'strong', 'moderate', 'weak'


class CorrelationSignalEngine:
    """Engine for generating trading signals based on sector correlations."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def get_leading_sectors(
        self,
        sector_code: int,
        min_correlation: float = 0.5,
        min_lag: int = 7
    ) -> List[CorrelationSignal]:
        """
        Get sectors that lead the given sector.

        These sectors' movements predict future movements in the target sector.
        Useful for generating early-warning signals.

        Args:
            sector_code: Target sector code
            min_correlation: Minimum absolute correlation (default 0.5)
            min_lag: Minimum lag in days (default 7)

        Returns:
            List of CorrelationSignal objects for leading sectors
        """
        query = """
            SELECT
                sc.sector_code_1 as leading_sector_code,
                s1.sector_name as leading_sector_name,
                s1.sector_etf as leading_sector_etf,
                sc.correlation_coefficient,
                sc.lag_days,
                sc.correlation_type,
                sc.p_value,
                sc.sample_size,
                CASE
                    WHEN ABS(sc.correlation_coefficient) > 0.7 THEN 'strong'
                    WHEN ABS(sc.correlation_coefficient) > 0.5 THEN 'moderate'
                    ELSE 'weak'
                END as signal_strength
            FROM sector_correlations sc
            JOIN sectors s1 ON sc.sector_code_1 = s1.sector_code
            WHERE sc.sector_code_2 = ?
              AND sc.lag_days >= ?
              AND ABS(sc.correlation_coefficient) >= ?
              AND sc.p_value < 0.05
            ORDER BY ABS(sc.correlation_coefficient) DESC, sc.lag_days ASC
        """

        results = self.conn.execute(query, [sector_code, min_lag, min_correlation]).fetchall()

        signals = [
            CorrelationSignal(
                sector_code=row[0],
                sector_name=row[1],
                sector_etf=row[2],
                correlation=row[3],
                lag_days=row[4],
                correlation_type=row[5],
                p_value=row[6],
                sample_size=row[7],
                signal_strength=row[8]
            )
            for row in results
        ]

        return signals

    def get_lagging_sectors(
        self,
        sector_code: int,
        min_correlation: float = 0.5,
        min_lag: int = 7
    ) -> List[CorrelationSignal]:
        """
        Get sectors that lag the given sector.

        These sectors follow movements in the target sector.
        Useful for confirming trends and sector rotation strategies.

        Args:
            sector_code: Target sector code
            min_correlation: Minimum absolute correlation (default 0.5)
            min_lag: Minimum lag in days (default 7)

        Returns:
            List of CorrelationSignal objects for lagging sectors
        """
        query = """
            SELECT
                sc.sector_code_2 as lagging_sector_code,
                s2.sector_name as lagging_sector_name,
                s2.sector_etf as lagging_sector_etf,
                sc.correlation_coefficient,
                sc.lag_days,
                sc.correlation_type,
                sc.p_value,
                sc.sample_size,
                CASE
                    WHEN ABS(sc.correlation_coefficient) > 0.7 THEN 'strong'
                    WHEN ABS(sc.correlation_coefficient) > 0.5 THEN 'moderate'
                    ELSE 'weak'
                END as signal_strength
            FROM sector_correlations sc
            JOIN sectors s2 ON sc.sector_code_2 = s2.sector_code
            WHERE sc.sector_code_1 = ?
              AND sc.lag_days >= ?
              AND ABS(sc.correlation_coefficient) >= ?
              AND sc.p_value < 0.05
            ORDER BY ABS(sc.correlation_coefficient) DESC, sc.lag_days ASC
        """

        results = self.conn.execute(query, [sector_code, min_lag, min_correlation]).fetchall()

        signals = [
            CorrelationSignal(
                sector_code=row[0],
                sector_name=row[1],
                sector_etf=row[2],
                correlation=row[3],
                lag_days=row[4],
                correlation_type=row[5],
                p_value=row[6],
                sample_size=row[7],
                signal_strength=row[8]
            )
            for row in results
        ]

        return signals

    def get_correlation_strength(
        self,
        sector_code_1: int,
        sector_code_2: int,
        lag_days: int = 0
    ) -> Optional[float]:
        """
        Get correlation coefficient between two sectors at a specific lag.

        Args:
            sector_code_1: First sector code
            sector_code_2: Second sector code
            lag_days: Time lag in days (default 0 for contemporaneous)

        Returns:
            Correlation coefficient or None if not found
        """
        query = """
            SELECT correlation_coefficient
            FROM sector_correlations
            WHERE sector_code_1 = ?
              AND sector_code_2 = ?
              AND lag_days = ?
              AND correlation_type = 'pearson'
            LIMIT 1
        """

        result = self.conn.execute(
            query,
            [sector_code_1, sector_code_2, lag_days]
        ).fetchone()

        return result[0] if result else None

    def get_all_correlations(self, sector_code: int) -> List[CorrelationSignal]:
        """
        Get all significant correlations for a sector (both leading and lagging).

        Args:
            sector_code: Target sector code

        Returns:
            List of all CorrelationSignal objects
        """
        query = """
            SELECT
                sc.sector_code_1,
                s1.sector_name,
                s1.sector_etf,
                sc.correlation_coefficient,
                sc.lag_days,
                sc.correlation_type,
                sc.p_value,
                sc.sample_size,
                CASE
                    WHEN ABS(sc.correlation_coefficient) > 0.7 THEN 'strong'
                    WHEN ABS(sc.correlation_coefficient) > 0.5 THEN 'moderate'
                    ELSE 'weak'
                END as signal_strength
            FROM sector_correlations sc
            JOIN sectors s1 ON sc.sector_code_1 = s1.sector_code
            WHERE sc.sector_code_2 = ?
            UNION ALL
            SELECT
                sc.sector_code_2,
                s2.sector_name,
                s2.sector_etf,
                sc.correlation_coefficient,
                sc.lag_days,
                sc.correlation_type,
                sc.p_value,
                sc.sample_size,
                CASE
                    WHEN ABS(sc.correlation_coefficient) > 0.7 THEN 'strong'
                    WHEN ABS(sc.correlation_coefficient) > 0.5 THEN 'moderate'
                    ELSE 'weak'
                END as signal_strength
            FROM sector_correlations sc
            JOIN sectors s2 ON sc.sector_code_2 = s2.sector_code
            WHERE sc.sector_code_1 = ?
            ORDER BY ABS(correlation_coefficient) DESC
        """

        results = self.conn.execute(query, [sector_code, sector_code]).fetchall()

        signals = [
            CorrelationSignal(
                sector_code=row[0],
                sector_name=row[1],
                sector_etf=row[2],
                correlation=row[3],
                lag_days=row[4],
                correlation_type=row[5],
                p_value=row[6],
                sample_size=row[7],
                signal_strength=row[8]
            )
            for row in results
        ]

        return signals

    def enhance_signal(
        self,
        sector_code: int,
        base_signal: float,
        lookback_days: int = 14
    ) -> Tuple[float, Dict[str, any]]:
        """
        Enhance a trading signal using correlated sector analysis.

        Args:
            sector_code: Target sector code
            base_signal: Base signal strength (-1 to +1)
            lookback_days: Days to look back for leading indicators

        Returns:
            (enhanced_signal, metadata)
            enhanced_signal: Adjusted signal strength
            metadata: Dictionary with enhancement details
        """
        # Get leading sectors that could predict this sector
        leading = self.get_leading_sectors(
            sector_code,
            min_correlation=0.5,
            min_lag=1
        )

        if not leading:
            return base_signal, {
                'enhancement': 'none',
                'reason': 'no_leading_sectors',
                'leading_signals': []
            }

        # Calculate weighted enhancement based on leading sectors
        total_weight = 0.0
        weighted_adjustment = 0.0

        leading_signals = []

        for signal in leading[:3]:  # Use top 3 leading sectors
            # Weight by correlation strength
            weight = abs(signal.correlation)

            # Assume positive movement in leading sector
            # (in real implementation, would check actual employment data)
            adjustment = signal.correlation * weight

            total_weight += weight
            weighted_adjustment += adjustment

            leading_signals.append({
                'sector': signal.sector_name,
                'etf': signal.sector_etf,
                'correlation': signal.correlation,
                'lag': signal.lag_days,
                'weight': weight
            })

        # Calculate enhancement factor (0.5 to 1.5)
        if total_weight > 0:
            enhancement_factor = 1.0 + (weighted_adjustment / total_weight) * 0.5
        else:
            enhancement_factor = 1.0

        enhanced_signal = base_signal * enhancement_factor

        # Clamp to valid range
        enhanced_signal = max(-1.0, min(1.0, enhanced_signal))

        metadata = {
            'enhancement': 'applied',
            'factor': enhancement_factor,
            'leading_signals': leading_signals,
            'original_signal': base_signal,
            'enhanced_signal': enhanced_signal
        }

        return enhanced_signal, metadata

    def close(self):
        """Close database connection."""
        self.conn.close()


# Convenience functions for easy integration

def get_leading_sectors(sector_code: int, min_correlation: float = 0.5) -> List[CorrelationSignal]:
    """Get sectors that lead the given sector."""
    engine = CorrelationSignalEngine()
    try:
        return engine.get_leading_sectors(sector_code, min_correlation)
    finally:
        engine.close()


def get_lagging_sectors(sector_code: int, min_correlation: float = 0.5) -> List[CorrelationSignal]:
    """Get sectors that lag the given sector."""
    engine = CorrelationSignalEngine()
    try:
        return engine.get_lagging_sectors(sector_code, min_correlation)
    finally:
        engine.close()


def enhance_sector_signal(sector_code: int, base_signal: float) -> float:
    """Enhance a sector trading signal using correlations."""
    engine = CorrelationSignalEngine()
    try:
        enhanced, _ = engine.enhance_signal(sector_code, base_signal)
        return enhanced
    finally:
        engine.close()


def main():
    """Demo the correlation signal engine."""
    print("=" * 80)
    print("CORRELATION SIGNAL ENGINE DEMO")
    print("=" * 80)
    print()

    engine = CorrelationSignalEngine()

    try:
        # Example: XLK (Information Technology, code 45)
        sector_code = 45
        sector_name = "Information Technology"

        print(f"Analyzing correlations for {sector_name} (XLK, code {sector_code})")
        print("-" * 80)

        # Get leading sectors
        leading = engine.get_leading_sectors(sector_code, min_correlation=0.5, min_lag=0)
        if leading:
            print(f"\nLeading Sectors (predict XLK movements):")
            for sig in leading:
                print(f"  {sig.sector_name:30} ({sig.sector_etf}): "
                      f"r={sig.correlation:+.3f}, lag={sig.lag_days}d, {sig.signal_strength}")
        else:
            print("\nNo leading sectors found")

        # Get lagging sectors
        lagging = engine.get_lagging_sectors(sector_code, min_correlation=0.5, min_lag=0)
        if lagging:
            print(f"\nLagging Sectors (follow XLK movements):")
            for sig in lagging:
                print(f"  {sig.sector_name:30} ({sig.sector_etf}): "
                      f"r={sig.correlation:+.3f}, lag={sig.lag_days}d, {sig.signal_strength}")
        else:
            print("\nNo lagging sectors found")

        # Demo signal enhancement
        print("\n" + "-" * 80)
        print("Signal Enhancement Demo:")
        base_signal = 0.6
        enhanced, metadata = engine.enhance_signal(sector_code, base_signal)
        print(f"  Base signal: {base_signal:+.3f}")
        print(f"  Enhanced signal: {enhanced:+.3f}")
        print(f"  Enhancement factor: {metadata.get('factor', 1.0):.3f}")

        print("\n" + "=" * 80)

    finally:
        engine.close()


if __name__ == "__main__":
    main()
