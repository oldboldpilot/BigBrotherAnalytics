#!/usr/bin/env python3
"""
BigBrotherAnalytics: Employment Trends Visualization
=====================================================

Create text-based visualizations of employment trends for all sectors.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import sys
sys.path.insert(0, 'python')

import bigbrother_duckdb as db
from datetime import datetime

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

SERIES_NAMES = {
    'CES1000000001': 'Mining/Logging',
    'CES2000000001': 'Construction',
    'CES3000000001': 'Manufacturing',
    'CES4200000001': 'Retail Trade',
    'CES4300000001': 'Transport/Warehousing',
    'CES4422000001': 'Utilities',
    'CES5000000001': 'Information',
    'CES5500000001': 'Financial Activities',
    'CES6500000001': 'Education/Health',
    'CES7000000001': 'Leisure/Hospitality',
}


def create_sparkline(values, width=50, height=7):
    """Create a simple ASCII sparkline."""
    if not values or len(values) < 2:
        return ["No data"]

    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val

    if value_range == 0:
        return ["─" * width]

    # Normalize values to height levels
    normalized = []
    for v in values:
        level = int(((v - min_val) / value_range) * (height - 1))
        normalized.append(level)

    # Create chart
    lines = []
    for h in range(height - 1, -1, -1):
        line = ""
        for level in normalized:
            if level == h:
                line += "●"
            elif level > h:
                line += "│"
            else:
                line += " "
        lines.append(line)

    return lines


def main():
    """Main visualization function"""
    print("=" * 80)
    print("BigBrotherAnalytics: Employment Trends Visualization")
    print("=" * 80)
    print()

    conn = db.connect('data/bigbrother.duckdb')

    for series_id, sector_codes in BLS_TO_SECTOR_MAP.items():
        series_name = SERIES_NAMES.get(series_id, series_id)

        # Get last 24 months of data
        result = conn.execute(f"""
            SELECT
                report_date,
                employment_count,
                LAG(employment_count, 1) OVER (ORDER BY report_date) as prev_month,
                LAG(employment_count, 12) OVER (ORDER BY report_date) as prev_year
            FROM sector_employment_raw
            WHERE series_id = '{series_id}'
            ORDER BY report_date DESC
            LIMIT 24
        """)

        data = result.to_pandas_dict()

        if len(data['report_date']) == 0:
            continue

        # Reverse to chronological order
        dates = list(reversed(data['report_date']))
        employment = list(reversed(data['employment_count']))
        prev_month = list(reversed(data['prev_month']))
        prev_year = list(reversed(data['prev_year']))

        # Calculate changes
        latest = employment[-1]
        if prev_month[-1] is not None:
            mom_change = ((latest - prev_month[-1]) / prev_month[-1]) * 100
        else:
            mom_change = 0.0

        if prev_year[-1] is not None:
            yoy_change = ((latest - prev_year[-1]) / prev_year[-1]) * 100
        else:
            yoy_change = 0.0

        print("─" * 80)
        print(f"\n{series_name} ({series_id})")
        print(f"Latest: {latest:,.0f}k | MoM: {mom_change:+.2f}% | YoY: {yoy_change:+.2f}%")
        print()

        # Create sparkline for last 24 months
        sparkline = create_sparkline(employment, width=len(employment), height=5)

        # Add trend indicator
        if yoy_change > 1.0:
            trend = "📈 Strong Growth"
        elif yoy_change > 0:
            trend = "↗ Growing"
        elif yoy_change > -1.0:
            trend = "→ Stable"
        else:
            trend = "📉 Declining"

        print(f"24-Month Trend: {trend}")
        print()

        for line in sparkline:
            print("  " + line)

        # Print date range
        print(f"\n  {dates[0]} to {dates[-1]}")
        print(f"  Range: {min(employment):,.0f}k - {max(employment):,.0f}k")
        print()

    print("=" * 80)
    print("Visualization Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
