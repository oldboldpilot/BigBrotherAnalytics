#!/usr/bin/env python3
"""
BigBrotherAnalytics: Correlation Visualization

Generate heatmaps and visualizations for sector correlations.

Outputs:
- 11x11 correlation matrix heatmap
- Lag analysis plots
- Correlation network diagram
- Statistical distribution charts

Author: Agent 6 - Correlation Discovery Agent
Date: 2025-11-10
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

DB_PATH = "data/bigbrother.duckdb"
OUTPUT_DIR = Path("reports/correlations")


def create_correlation_matrix(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create 11x11 correlation matrix from database."""
    # Get all sectors
    sectors = conn.execute("""
        SELECT sector_code, sector_etf
        FROM sectors
        ORDER BY sector_code
    """).fetchall()

    sector_codes = [s[0] for s in sectors]
    sector_etfs = [s[1] for s in sectors]

    # Initialize matrix with zeros
    n = len(sector_codes)
    matrix = np.zeros((n, n))

    # Fill diagonal with 1.0
    np.fill_diagonal(matrix, 1.0)

    # Get all correlations (lag=0, pearson)
    correlations = conn.execute("""
        SELECT sector_code_1, sector_code_2, correlation_coefficient
        FROM sector_correlations
        WHERE lag_days = 0
          AND correlation_type = 'pearson'
    """).fetchall()

    # Fill matrix
    for code1, code2, corr in correlations:
        try:
            i = sector_codes.index(code1)
            j = sector_codes.index(code2)
            matrix[i, j] = corr
            matrix[j, i] = corr  # Symmetric
        except ValueError:
            continue

    # Create DataFrame
    df = pd.DataFrame(matrix, index=sector_etfs, columns=sector_etfs)
    return df


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """Generate correlation heatmap."""
    plt.figure(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )

    plt.title('GICS Sector Employment Correlation Matrix\n(11 Sectors, Pearson, Lag=0)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sector ETF', fontsize=12, fontweight='bold')
    plt.ylabel('Sector ETF', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    plt.close()


def plot_lag_analysis(conn: duckdb.DuckDBPyConnection, output_path: Path):
    """Generate lag analysis plot."""
    # Get correlations by lag
    lag_data = conn.execute("""
        SELECT
            lag_days,
            AVG(ABS(correlation_coefficient)) as avg_abs_corr,
            COUNT(*) as count
        FROM sector_correlations
        WHERE correlation_type = 'pearson'
        GROUP BY lag_days
        ORDER BY lag_days
    """).fetchall()

    if not lag_data:
        print("⚠ No lag data available for plotting")
        return

    lags = [row[0] for row in lag_data]
    avg_corrs = [row[1] for row in lag_data]
    counts = [row[2] for row in lag_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Average correlation by lag
    ax1.bar(lags, avg_corrs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Time Lag (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average |Correlation|', fontsize=12, fontweight='bold')
    ax1.set_title('Average Absolute Correlation by Time Lag', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (lag, corr) in enumerate(zip(lags, avg_corrs)):
        ax1.text(lag, corr + 0.01, f'{corr:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Count of correlations by lag
    ax2.bar(lags, counts, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Time Lag (days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Significant Correlations', fontsize=12, fontweight='bold')
    ax2.set_title('Correlation Count by Time Lag (|r| > 0.5, p < 0.05)',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, (lag, count) in enumerate(zip(lags, counts)):
        ax2.text(lag, count + 0.2, str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved lag analysis: {output_path}")
    plt.close()


def plot_correlation_distribution(conn: duckdb.DuckDBPyConnection, output_path: Path):
    """Generate correlation distribution plot."""
    # Get all correlations
    correlations = conn.execute("""
        SELECT correlation_coefficient
        FROM sector_correlations
        WHERE correlation_type = 'pearson'
    """).fetchall()

    if not correlations:
        print("⚠ No correlation data available for distribution")
        return

    corr_values = [row[0] for row in correlations]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Histogram
    ax1.hist(corr_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
    ax1.axvline(x=-0.5, color='green', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Sector Correlations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Box plot
    ax2.boxplot(corr_values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax2.set_title('Correlation Distribution Statistics', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(['All Correlations'])

    # Add statistics
    stats_text = f"Mean: {np.mean(corr_values):.3f}\n"
    stats_text += f"Median: {np.median(corr_values):.3f}\n"
    stats_text += f"Std Dev: {np.std(corr_values):.3f}\n"
    stats_text += f"Min: {np.min(corr_values):.3f}\n"
    stats_text += f"Max: {np.max(corr_values):.3f}"
    ax2.text(1.2, np.mean(corr_values), stats_text,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot: {output_path}")
    plt.close()


def plot_top_correlations(conn: duckdb.DuckDBPyConnection, output_path: Path):
    """Generate bar chart of top correlations."""
    # Get top 15 correlations
    top_corrs = conn.execute("""
        SELECT
            s1.sector_etf || ' → ' || s2.sector_etf as pair,
            sc.correlation_coefficient,
            sc.lag_days
        FROM sector_correlations sc
        JOIN sectors s1 ON sc.sector_code_1 = s1.sector_code
        JOIN sectors s2 ON sc.sector_code_2 = s2.sector_code
        WHERE sc.correlation_type = 'pearson'
        ORDER BY ABS(sc.correlation_coefficient) DESC
        LIMIT 15
    """).fetchall()

    if not top_corrs:
        print("⚠ No correlations available for top chart")
        return

    pairs = [row[0] for row in top_corrs]
    corrs = [row[1] for row in top_corrs]
    lags = [row[2] for row in top_corrs]

    # Color by correlation strength
    colors = ['darkgreen' if abs(c) > 0.7 else 'steelblue' for c in corrs]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(pairs)), corrs, color=colors, alpha=0.7, edgecolor='black')

    plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.ylabel('Sector Pair', fontsize=12, fontweight='bold')
    plt.title('Top 15 Strongest Sector Correlations', fontsize=14, fontweight='bold')
    plt.yticks(range(len(pairs)), pairs)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)

    # Add correlation values
    for i, (corr, lag) in enumerate(zip(corrs, lags)):
        lag_text = f" (lag={lag}d)" if lag > 0 else ""
        plt.text(corr + 0.02 if corr > 0 else corr - 0.02,
                 i, f'{corr:.3f}{lag_text}',
                 va='center', ha='left' if corr > 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved top correlations: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("CORRELATION VISUALIZATION")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Connect to database
    conn = duckdb.connect(DB_PATH, read_only=True)

    try:
        # Generate visualizations
        print("Generating visualizations...")
        print("-" * 80)

        # 1. Correlation matrix heatmap
        print("\n1. Correlation Matrix Heatmap")
        df = create_correlation_matrix(conn)
        plot_correlation_heatmap(df, OUTPUT_DIR / "correlation_heatmap.png")

        # 2. Lag analysis
        print("\n2. Lag Analysis")
        plot_lag_analysis(conn, OUTPUT_DIR / "lag_analysis.png")

        # 3. Correlation distribution
        print("\n3. Correlation Distribution")
        plot_correlation_distribution(conn, OUTPUT_DIR / "correlation_distribution.png")

        # 4. Top correlations
        print("\n4. Top Correlations")
        plot_top_correlations(conn, OUTPUT_DIR / "top_correlations.png")

        print("\n" + "=" * 80)
        print("✓ All visualizations generated successfully!")
        print(f"✓ Saved to: {OUTPUT_DIR.absolute()}")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
