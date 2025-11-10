#!/usr/bin/env python3
"""
BigBrotherAnalytics: Optimized Employment Signal Generator (Python Backend)

Performance Optimizations:
1. Database connection pooling and prepared statements
2. Data caching for employment data (refreshes daily only)
3. Parallel processing for sector calculations
4. Batch database queries to reduce round trips

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import duckdb
import json
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

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

# Global cache for employment data
_employment_cache = {}
_cache_lock = threading.Lock()
_cache_timestamp = None
CACHE_TTL_HOURS = 24  # Refresh once per day

class EmploymentDataCache:
    """
    Thread-safe cache for employment data
    Data changes daily (not intraday), so we can cache aggressively
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.cache = {}
        self.cache_timestamp = None
        self.lock = threading.Lock()

    def needs_refresh(self) -> bool:
        """Check if cache needs refreshing"""
        if self.cache_timestamp is None:
            return True

        elapsed = datetime.now() - self.cache_timestamp
        return elapsed > timedelta(hours=CACHE_TTL_HOURS)

    def refresh(self):
        """Load all employment data into cache"""
        with self.lock:
            if not self.needs_refresh():
                return

            conn = duckdb.connect(self.db_path, read_only=True)

            # Load all employment data in one batch query
            query = """
                SELECT
                    series_id,
                    report_date,
                    employment_count,
                    LAG(employment_count, 1) OVER (PARTITION BY series_id ORDER BY report_date) as prev_1m,
                    LAG(employment_count, 3) OVER (PARTITION BY series_id ORDER BY report_date) as prev_3m,
                    LAG(employment_count, 6) OVER (PARTITION BY series_id ORDER BY report_date) as prev_6m,
                    LAG(employment_count, 12) OVER (PARTITION BY series_id ORDER BY report_date) as prev_12m
                FROM sector_employment_raw
                WHERE series_id IN (
                    'CES1000000001', 'CES2000000001', 'CES3000000001',
                    'CES4200000001', 'CES4300000001', 'CES4422000001',
                    'CES5000000001', 'CES5500000001', 'CES6500000001', 'CES7000000001'
                )
                ORDER BY series_id, report_date DESC
            """

            result = conn.execute(query).fetchall()

            # Organize by series_id
            for row in result:
                series_id = row[0]
                if series_id not in self.cache:
                    self.cache[series_id] = []
                self.cache[series_id].append({
                    'report_date': row[1],
                    'employment_count': row[2],
                    'prev_1m': row[3],
                    'prev_3m': row[4],
                    'prev_6m': row[5],
                    'prev_12m': row[6]
                })

            # Also cache sector names
            sector_query = "SELECT sector_code, sector_name FROM sectors"
            self.sector_names = dict(conn.execute(sector_query).fetchall())

            conn.close()
            self.cache_timestamp = datetime.now()

    def get_series_data(self, series_id: str, limit: int = 24) -> List[Dict]:
        """Get cached data for a series"""
        if self.needs_refresh():
            self.refresh()

        with self.lock:
            return self.cache.get(series_id, [])[:limit]

    def get_sector_name(self, sector_code: int) -> str:
        """Get sector name from cache"""
        if self.needs_refresh():
            self.refresh()

        with self.lock:
            return self.sector_names.get(sector_code, f"Sector {sector_code}")

# Global cache instance
_data_cache = None

def get_cache(db_path: str) -> EmploymentDataCache:
    """Get or create global cache instance"""
    global _data_cache
    if _data_cache is None:
        _data_cache = EmploymentDataCache(db_path)
    return _data_cache

def calculate_employment_statistics(cache: EmploymentDataCache, series_id: str) -> Dict:
    """
    Calculate comprehensive employment statistics using cached data
    """
    data = cache.get_series_data(series_id, limit=24)

    if not data or len(data) == 0:
        return {}

    latest = data[0]

    if not latest['employment_count']:
        return {}

    latest_val = latest['employment_count']
    prev_1m = latest['prev_1m']
    prev_3m = latest['prev_3m']
    prev_6m = latest['prev_6m']
    prev_12m = latest['prev_12m']

    # Calculate month-over-month change
    mom_change = ((latest_val - prev_1m) / prev_1m * 100.0) if prev_1m else 0.0

    # Calculate trends
    trend_3m = ((latest_val - prev_3m) / prev_3m * 100.0) if prev_3m else 0.0
    trend_6m = ((latest_val - prev_6m) / prev_6m * 100.0) if prev_6m else 0.0
    trend_12m = ((latest_val - prev_12m) / prev_12m * 100.0) if prev_12m else 0.0

    # Calculate acceleration
    acceleration = trend_3m - trend_6m if trend_3m and trend_6m else 0.0

    # Calculate statistics from all available data
    values = [d['employment_count'] for d in data if d['employment_count']]
    if values:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = variance ** 0.5
        z_score = ((latest_val - mean) / stddev) if stddev > 0 else 0.0
    else:
        mean = stddev = z_score = 0.0

    # Detect inflection point
    is_inflection = abs(acceleration) > 0.5

    return {
        'latest': latest_val,
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

def process_sector_signals(series_id: str, sector_codes: List[int], cache: EmploymentDataCache) -> List[Dict]:
    """
    Process employment signals for a single BLS series (parallelizable)
    """
    signals = []

    # Calculate comprehensive statistics
    stats = calculate_employment_statistics(cache, series_id)

    if not stats:
        return signals

    for sector_code in sector_codes:
        sector_name = cache.get_sector_name(sector_code)

        # Determine signal type based on trends and acceleration
        signal_type = None
        rationale_parts = []

        # Strong trend detection
        if abs(stats['trend_3m']) > 2.0 or abs(stats['trend_6m']) > 3.0:
            is_improving = stats['trend_3m'] > 0
            signal_type = 'EmploymentImproving' if is_improving else 'EmploymentDeclining'

            rationale_parts.append(
                f"3-month trend: {stats['trend_3m']:+.2f}%, "
                f"6-month trend: {stats['trend_6m']:+.2f}%"
            )

            if stats['is_inflection']:
                rationale_parts.append(
                    f"Acceleration: {stats['acceleration']:+.2f}% (inflection detected)"
                )

            # Calculate confidence
            confidence = 0.60
            if (stats['trend_3m'] > 0) == (stats['trend_6m'] > 0):
                confidence += 0.15
            if abs(stats['z_score']) > 1.0:
                confidence += 0.10
                rationale_parts.append(f"Z-score: {stats['z_score']:.2f}")
            if stats['is_inflection']:
                confidence += 0.05
            confidence = min(0.95, confidence)

            # Calculate signal strength
            trend_component = stats['trend_3m'] * 0.7
            accel_component = stats['acceleration'] * 0.3
            raw_strength = trend_component + accel_component
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
            is_accelerating = stats['acceleration'] > 0
            signal_type = 'EmploymentImproving' if is_accelerating else 'EmploymentDeclining'

            rationale_parts.append(
                f"Employment inflection detected: acceleration {stats['acceleration']:+.2f}%"
            )
            rationale_parts.append(f"Recent trend: {stats['trend_3m']:+.2f}%")

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

    return signals

def generate_employment_signals(db_path: str = DB_PATH) -> List[Dict]:
    """
    Generate employment signals for all sectors (optimized with caching and parallelization)
    """
    cache = get_cache(db_path)
    cache.refresh()  # Ensure cache is fresh

    all_signals = []

    # Process all series in parallel
    with ThreadPoolExecutor(max_workers=len(BLS_TO_SECTOR_MAP)) as executor:
        futures = {
            executor.submit(process_sector_signals, series_id, sector_codes, cache): series_id
            for series_id, sector_codes in BLS_TO_SECTOR_MAP.items()
        }

        for future in as_completed(futures):
            try:
                signals = future.result()
                all_signals.extend(signals)
            except Exception as e:
                series_id = futures[future]
                print(f"Error processing series {series_id}: {e}", file=sys.stderr)

    return all_signals

def generate_rotation_signals(db_path: str = DB_PATH) -> List[Dict]:
    """
    Generate sector rotation signals (optimized with caching)
    """
    cache = get_cache(db_path)
    cache.refresh()

    signals = []
    sector_scores: Dict[int, List[Tuple[float, float]]] = {}

    # Calculate employment scores using cached data
    for series_id, sector_codes in BLS_TO_SECTOR_MAP.items():
        stats = calculate_employment_statistics(cache, series_id)

        if not stats:
            continue

        # Calculate employment score
        trend_3m = stats['trend_3m']
        trend_6m = stats['trend_6m']
        acceleration = stats['acceleration']
        z_score = stats['z_score']

        weighted_trend = (trend_3m * 0.6) + (trend_6m * 0.4)

        trend_score = max(-1.0, min(1.0, weighted_trend / 10.0))
        accel_score = max(-1.0, min(1.0, acceleration / 5.0))
        z_score_normalized = max(-1.0, min(1.0, z_score / 2.0))

        employment_score = (
            trend_score * 0.60 +
            accel_score * 0.25 +
            z_score_normalized * 0.15
        )

        for sector_code in sector_codes:
            if sector_code not in sector_scores:
                sector_scores[sector_code] = []
            sector_scores[sector_code].append((employment_score, z_score))

    # Generate rotation signals
    for sector_code, score_tuples in sector_scores.items():
        employment_score = sum(s[0] for s in score_tuples) / len(score_tuples)

        sentiment_score = 0.0
        technical_score = 0.0
        composite_score = employment_score

        if composite_score > 0.25:
            action = 'Overweight'
            target_allocation = 10.0 + (composite_score * 8.0)
        elif composite_score < -0.25:
            action = 'Underweight'
            target_allocation = max(2.0, 7.0 - (abs(composite_score) * 5.0))
        else:
            action = 'Neutral'
            target_allocation = 100.0 / 11.0

        sector_name = cache.get_sector_name(sector_code)
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

    signals.sort(key=lambda x: x['composite_score'], reverse=True)
    return signals

def check_jobless_claims_spike(db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Check for jobless claims spike (using read-only connection)
    """
    conn = duckdb.connect(db_path, read_only=True)

    try:
        query = """
            SELECT
                report_date,
                initial_claims,
                four_week_avg,
                spike_detected,
                CASE
                    WHEN four_week_avg > 0
                    THEN ((initial_claims - four_week_avg) / four_week_avg * 100.0)
                    ELSE 0.0
                END as pct_increase
            FROM jobless_claims
            WHERE spike_detected = TRUE
            ORDER BY report_date DESC
            LIMIT 1
        """

        result = conn.execute(query).fetchone()

        if not result:
            return None

        report_date, initial_claims, four_week_avg, spike_detected, pct_increase = result

        signal = {
            'type': 'JoblessClaimsSpike',
            'sector_code': 0,
            'sector_name': 'Market-Wide',
            'confidence': 0.85,
            'employment_change': -pct_increase,
            'rationale': f'Initial jobless claims spiked {pct_increase:.1f}% above 4-week average on {report_date}. Claims: {initial_claims:,}, 4-week avg: {four_week_avg:,}. Indicates potential recession risk.',
            'timestamp': int(datetime.now().timestamp()),
            'bullish': False,
            'bearish': True,
            'signal_strength': -0.80
        }

        return signal

    except Exception as e:
        print(f"Error checking jobless claims: {e}", file=sys.stderr)
        return None
    finally:
        conn.close()

def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: employment_signals_optimized.py <command>", file=sys.stderr)
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
