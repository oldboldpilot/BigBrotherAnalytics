#!/usr/bin/env python3
"""
BigBrotherAnalytics: Employment Signal Generator (Python Backend)

Generates trading signals from BLS employment data for C++ integration.
Called by C++ EmploymentSignalGenerator via subprocess.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import duckdb
import json
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Map BLS series to GICS sectors
BLS_TO_SECTOR_MAP = {
    'CES1000000001': [10, 15],  # Mining/Logging → Energy, Materials
    'CES2000000001': [20, 60],  # Construction → Industrials, Real Estate
    'CES3000000001': [15, 20],  # Manufacturing → Materials, Industrials
    'CES4200000001': [25, 30],  # Retail → Consumer Discretionary, Staples
    'CES4300000001': [20],      # Transport → Industrials
    'CES4422000001': [55],      # Utilities → Utilities
    'CES5000000001': [45, 50],  # Information → IT, Communications
    'CES5500000001': [40],      # Financial Activities → Financials
    'CES6500000001': [35],      # Education/Health → Health Care
    'CES7000000001': [25],      # Leisure/Hospitality → Consumer Discretionary
}

# Sector ETF mapping
SECTOR_ETF_MAP = {
    10: 'XLE',   # Energy
    15: 'XLB',   # Materials
    20: 'XLI',   # Industrials
    25: 'XLY',   # Consumer Discretionary
    30: 'XLP',   # Consumer Staples
    35: 'XLV',   # Health Care
    40: 'XLF',   # Financials
    45: 'XLK',   # Information Technology
    50: 'XLC',   # Communication Services
    55: 'XLU',   # Utilities
    60: 'XLRE',  # Real Estate
}

DB_PATH = "data/bigbrother.duckdb"


def get_sector_name(conn: duckdb.DuckDBPyConnection, sector_code: int) -> str:
    """Get sector name from code."""
    result = conn.execute(
        "SELECT sector_name FROM sectors WHERE sector_code = ?",
        [sector_code]
    ).fetchone()
    return result[0] if result else f"Sector {sector_code}"


def calculate_employment_change(
    conn: duckdb.DuckDBPyConnection,
    series_id: str,
    months: int
) -> Optional[float]:
    """
    Calculate employment % change over specified months.

    Returns:
        % change (e.g., 5.2 for 5.2% increase), or None if insufficient data
    """
    query = f"""
        WITH recent_data AS (
            SELECT
                report_date,
                employment_count,
                ROW_NUMBER() OVER (ORDER BY report_date DESC) as rn
            FROM sector_employment_raw
            WHERE series_id = '{series_id}'
            ORDER BY report_date DESC
            LIMIT {months + 1}
        )
        SELECT
            (SELECT employment_count FROM recent_data WHERE rn = 1) as latest,
            (SELECT employment_count FROM recent_data WHERE rn = {months + 1}) as baseline
    """

    result = conn.execute(query).fetchone()
    if not result or not result[0] or not result[1]:
        return None

    latest, baseline = result
    return ((latest - baseline) / baseline) * 100.0


def calculate_employment_statistics(
    conn: duckdb.DuckDBPyConnection,
    series_id: str
) -> Dict:
    """
    Calculate comprehensive employment statistics for a series.

    Returns:
        Dict with trend, acceleration, z-score, volatility metrics
    """
    query = f"""
        WITH employment_data AS (
            SELECT
                report_date,
                employment_count,
                LAG(employment_count, 1) OVER (ORDER BY report_date) as prev_1m,
                LAG(employment_count, 3) OVER (ORDER BY report_date) as prev_3m,
                LAG(employment_count, 6) OVER (ORDER BY report_date) as prev_6m,
                LAG(employment_count, 12) OVER (ORDER BY report_date) as prev_12m
            FROM sector_employment_raw
            WHERE series_id = '{series_id}'
            ORDER BY report_date DESC
            LIMIT 24
        ),
        stats AS (
            SELECT
                AVG(employment_count) as mean,
                STDDEV(employment_count) as stddev,
                MIN(employment_count) as min_val,
                MAX(employment_count) as max_val
            FROM employment_data
        )
        SELECT
            ed.employment_count as latest,
            ed.prev_1m,
            ed.prev_3m,
            ed.prev_6m,
            ed.prev_12m,
            s.mean,
            s.stddev,
            s.min_val,
            s.max_val
        FROM employment_data ed, stats s
        WHERE ed.report_date = (SELECT MAX(report_date) FROM employment_data)
    """

    result = conn.execute(query).fetchone()
    if not result:
        return {}

    latest, prev_1m, prev_3m, prev_6m, prev_12m, mean, stddev, min_val, max_val = result

    # Calculate month-over-month change
    mom_change = ((latest - prev_1m) / prev_1m * 100.0) if prev_1m else 0.0

    # Calculate 3-month and 6-month trends
    trend_3m = ((latest - prev_3m) / prev_3m * 100.0) if prev_3m else 0.0
    trend_6m = ((latest - prev_6m) / prev_6m * 100.0) if prev_6m else 0.0
    trend_12m = ((latest - prev_12m) / prev_12m * 100.0) if prev_12m else 0.0

    # Calculate acceleration (change in trend)
    acceleration = trend_3m - trend_6m if trend_3m and trend_6m else 0.0

    # Calculate z-score (how many standard deviations from mean)
    z_score = ((latest - mean) / stddev) if stddev and stddev > 0 else 0.0

    # Detect inflection point (acceleration changing sign)
    is_inflection = abs(acceleration) > 0.5

    return {
        'latest': latest,
        'mom_change': mom_change,
        'trend_3m': trend_3m,
        'trend_6m': trend_6m,
        'trend_12m': trend_12m,
        'acceleration': acceleration,
        'z_score': z_score,
        'is_inflection': is_inflection,
        'mean': mean,
        'stddev': stddev,
        'volatility': (stddev / mean * 100.0) if mean else 0.0
    }


def generate_employment_signals(db_path: str = DB_PATH) -> List[Dict]:
    """
    Generate employment signals for all sectors.

    Returns:
        List of signal dictionaries with structure:
        {
            'type': str,  # Signal type name
            'sector_code': int,
            'sector_name': str,
            'confidence': float,
            'employment_change': float,
            'rationale': str,
            'timestamp': int,  # Unix timestamp
            'bullish': bool,
            'bearish': bool,
            'signal_strength': float  # -1.0 to +1.0
        }
    """
    conn = duckdb.connect(db_path)
    signals = []

    for series_id, sector_codes in BLS_TO_SECTOR_MAP.items():
        # Calculate comprehensive statistics
        stats = calculate_employment_statistics(conn, series_id)

        if not stats:
            continue

        for sector_code in sector_codes:
            sector_name = get_sector_name(conn, sector_code)

            # Determine signal type based on trends and acceleration
            signal_type = None
            rationale_parts = []

            # Strong trend detection (threshold: >2% over 3 months or >3% over 6 months)
            if abs(stats['trend_3m']) > 2.0 or abs(stats['trend_6m']) > 3.0:
                is_improving = stats['trend_3m'] > 0
                signal_type = 'EmploymentImproving' if is_improving else 'EmploymentDeclining'

                # Build detailed rationale
                rationale_parts.append(
                    f"3-month trend: {stats['trend_3m']:+.2f}%, "
                    f"6-month trend: {stats['trend_6m']:+.2f}%"
                )

                # Check for acceleration
                if stats['is_inflection']:
                    rationale_parts.append(
                        f"Acceleration: {stats['acceleration']:+.2f}% (inflection detected)"
                    )

                # Calculate confidence based on multiple factors
                confidence = 0.60  # Base confidence

                # Increase confidence if trends agree
                if (stats['trend_3m'] > 0) == (stats['trend_6m'] > 0):
                    confidence += 0.15

                # Increase confidence for stronger z-score
                if abs(stats['z_score']) > 1.0:
                    confidence += 0.10
                    rationale_parts.append(f"Z-score: {stats['z_score']:.2f}")

                # Increase confidence for consistent acceleration
                if stats['is_inflection']:
                    confidence += 0.05

                confidence = min(0.95, confidence)  # Cap at 0.95

                # Calculate signal strength: combination of trend and acceleration
                # Weight: 70% trend, 30% acceleration
                trend_component = stats['trend_3m'] * 0.7
                accel_component = stats['acceleration'] * 0.3
                raw_strength = trend_component + accel_component

                # Normalize to -1.0 to +1.0 (cap at ±10% for normalization)
                signal_strength = max(-1.0, min(1.0, raw_strength / 10.0))

                signals.append({
                    'type': signal_type,
                    'sector_code': sector_code,
                    'sector_name': sector_name,
                    'confidence': confidence,
                    'employment_change': stats['trend_3m'],
                    'rationale': ' | '.join(rationale_parts),
                    'timestamp': int(datetime.now().timestamp()),
                    'bullish': is_improving,
                    'bearish': not is_improving,
                    'signal_strength': signal_strength
                })

            # Detect inflection points even with smaller trends
            elif stats['is_inflection'] and abs(stats['acceleration']) > 1.0:
                # Inflection point signal
                is_accelerating = stats['acceleration'] > 0
                signal_type = 'EmploymentImproving' if is_accelerating else 'EmploymentDeclining'

                rationale_parts.append(
                    f"Employment inflection detected: acceleration {stats['acceleration']:+.2f}%"
                )
                rationale_parts.append(
                    f"Recent trend: {stats['trend_3m']:+.2f}%"
                )

                confidence = 0.65
                signal_strength = max(-1.0, min(1.0, stats['acceleration'] / 5.0))

                signals.append({
                    'type': signal_type,
                    'sector_code': sector_code,
                    'sector_name': sector_name,
                    'confidence': confidence,
                    'employment_change': stats['trend_3m'],
                    'rationale': ' | '.join(rationale_parts),
                    'timestamp': int(datetime.now().timestamp()),
                    'bullish': is_accelerating,
                    'bearish': not is_accelerating,
                    'signal_strength': signal_strength
                })

    conn.close()
    return signals


def generate_rotation_signals(db_path: str = DB_PATH) -> List[Dict]:
    """
    Generate sector rotation signals.

    Returns:
        List of rotation signal dictionaries with structure:
        {
            'sector_code': int,
            'sector_name': str,
            'sector_etf': str,
            'employment_score': float,
            'sentiment_score': float,
            'technical_score': float,
            'composite_score': float,
            'action': str,  # 'Overweight', 'Neutral', 'Underweight'
            'target_allocation': float
        }
    """
    conn = duckdb.connect(db_path)
    signals = []

    # Calculate employment scores for each sector using comprehensive statistics
    sector_scores: Dict[int, List[Tuple[float, float]]] = {}  # (score, z_score)

    for series_id, sector_codes in BLS_TO_SECTOR_MAP.items():
        stats = calculate_employment_statistics(conn, series_id)

        if not stats:
            continue

        # Calculate employment score combining multiple factors:
        # 1. Trend strength (60%): weighted average of 3m and 6m trends
        # 2. Acceleration (25%): is employment accelerating or decelerating?
        # 3. Z-score position (15%): relative to historical average

        trend_3m = stats['trend_3m']
        trend_6m = stats['trend_6m']
        acceleration = stats['acceleration']
        z_score = stats['z_score']

        # Weighted trend component
        weighted_trend = (trend_3m * 0.6) + (trend_6m * 0.4)

        # Normalize components to -1.0 to +1.0
        trend_score = max(-1.0, min(1.0, weighted_trend / 10.0))  # Cap at ±10%
        accel_score = max(-1.0, min(1.0, acceleration / 5.0))     # Cap at ±5%
        z_score_normalized = max(-1.0, min(1.0, z_score / 2.0))  # Cap at ±2 std devs

        # Combined employment score with weights
        employment_score = (
            trend_score * 0.60 +
            accel_score * 0.25 +
            z_score_normalized * 0.15
        )

        for sector_code in sector_codes:
            if sector_code not in sector_scores:
                sector_scores[sector_code] = []
            sector_scores[sector_code].append((employment_score, z_score))

    # Generate rotation signals for each sector
    total_sectors = len(sector_scores)
    if total_sectors == 0:
        conn.close()
        return signals

    for sector_code, score_tuples in sector_scores.items():
        # Average employment score if multiple series map to same sector
        employment_score = sum(s[0] for s in score_tuples) / len(score_tuples)
        avg_z_score = sum(s[1] for s in score_tuples) / len(score_tuples)

        # For now, sentiment and technical scores are 0.0 (future implementation)
        sentiment_score = 0.0
        technical_score = 0.0

        # Composite score: weighted average (employment: 100% for now)
        # Future: (0.50*employment + 0.30*sentiment + 0.20*technical)
        composite_score = employment_score

        # Determine action based on composite score with refined thresholds
        if composite_score > 0.25:
            action = 'Overweight'
            # Allocate more to strong sectors (10-18%)
            # Stronger signals get higher allocation
            target_allocation = 10.0 + (composite_score * 8.0)
        elif composite_score < -0.25:
            action = 'Underweight'
            # Allocate less to weak sectors (2-7%)
            target_allocation = max(2.0, 7.0 - (abs(composite_score) * 5.0))
        else:
            action = 'Neutral'
            # Equal weight allocation (~9.09% for 11 sectors)
            target_allocation = 100.0 / 11.0

        sector_name = get_sector_name(conn, sector_code)
        sector_etf = SECTOR_ETF_MAP.get(sector_code, 'SPY')

        signals.append({
            'sector_code': sector_code,
            'sector_name': sector_name,
            'sector_etf': sector_etf,
            'employment_score': employment_score,
            'sentiment_score': sentiment_score,
            'technical_score': technical_score,
            'composite_score': composite_score,
            'action': action,
            'target_allocation': target_allocation
        })

    conn.close()

    # Sort by composite score (strongest first)
    signals.sort(key=lambda x: x['composite_score'], reverse=True)

    return signals


def check_jobless_claims_spike(db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Check for jobless claims spike (>10% increase).

    Note: Current schema doesn't have jobless claims data.
    This is a placeholder for future implementation when weekly claims data is added.

    Returns:
        Signal dictionary if spike detected, None otherwise
    """
    # TODO: Implement when jobless claims data is available in database
    # For now, return None as we don't have this data yet

    # Future implementation would:
    # 1. Query latest weekly jobless claims from economic_data table
    # 2. Calculate 4-week moving average
    # 3. Check if current week > 1.10 * moving average
    # 4. Return RecessionWarning signal if spike detected

    return None


def main():
    """CLI entry point for testing and C++ integration."""
    if len(sys.argv) < 2:
        print("Usage: employment_signals.py <command>", file=sys.stderr)
        print("Commands: generate_signals, rotation_signals, check_jobless_claims", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else DB_PATH

    try:
        if command == 'generate_signals':
            signals = generate_employment_signals(db_path)
            print(json.dumps(signals, indent=2))

        elif command == 'rotation_signals':
            signals = generate_rotation_signals(db_path)
            print(json.dumps(signals, indent=2))

        elif command == 'check_jobless_claims':
            signal = check_jobless_claims_spike(db_path)
            if signal:
                print(json.dumps(signal, indent=2))
            else:
                print(json.dumps({'status': 'no_spike'}))

        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
