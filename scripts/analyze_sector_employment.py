#!/usr/bin/env python3
"""
BigBrotherAnalytics: Sector Employment Analysis

Maps BLS employment data to GICS sectors for trading signals.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import duckdb
from datetime import datetime
from typing import Dict, List

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

def analyze_sector_employment(db_path: str = "data/bigbrother.duckdb"):
    """Analyze employment trends by sector."""
    conn = duckdb.connect(db_path)
    
    print("=" * 80)
    print("SECTOR EMPLOYMENT ANALYSIS")
    print("=" * 80)
    print()
    
    # Get latest employment figures mapped to sectors
    print("Latest Employment by Sector (thousands):")
    print("-" * 80)
    
    for series_id, sectors in BLS_TO_SECTOR_MAP.items():
        result = conn.execute(f"""
            SELECT employment_count 
            FROM sector_employment_raw 
            WHERE series_id = '{series_id}'
            ORDER BY report_date DESC 
            LIMIT 1
        """).fetchone()
        
        if result:
            employment = result[0]
            sector_names = [
                conn.execute(f"SELECT sector_name FROM sectors WHERE sector_code = {sc}").fetchone()[0]
                for sc in sectors
            ]
            print(f"  {series_id}: {employment:,}k → {', '.join(sector_names)}")
    
    print()
    print("=" * 80)
    print("Portfolio Sector Allocation:")
    print("-" * 80)
    
    allocation = conn.execute("SELECT * FROM sector_diversification").fetchall()
    for row in allocation:
        print(f"  {row[0]:25s} ({row[1]:10s}): {row[2]} stocks ({row[3]}%)")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    analyze_sector_employment()
