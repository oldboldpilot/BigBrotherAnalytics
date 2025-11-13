"""
BigBrotherAnalytics - Risk Exposure Dashboard

Real-time risk metrics and exposure monitoring for paper trading.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-12
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_risk_metrics(
    positions_df: pd.DataFrame, account_value: float = 30000.00
) -> Dict:
    """
    Calculate comprehensive risk metrics

    Args:
        positions_df: DataFrame with position data
        account_value: Total account value

    Returns:
        Dictionary of risk metrics
    """
    if positions_df.empty:
        return {
            "portfolio_heat": 0.0,
            "largest_position_pct": 0.0,
            "sector_concentration": {},
            "daily_loss": 0.0,
            "daily_loss_pct": 0.0,
            "var_95": 0.0,  # Value at Risk (95% confidence)
            "max_drawdown_pct": 0.0,
            "leverage_ratio": 0.0,
        }

    total_exposure = positions_df["market_value"].abs().sum()
    total_day_pnl = positions_df["day_pnl"].sum()

    # Portfolio heat (total exposure / account value)
    portfolio_heat = total_exposure / account_value if account_value > 0 else 0.0

    # Largest position as % of account
    largest_position = positions_df["market_value"].abs().max()
    largest_position_pct = largest_position / account_value if account_value > 0 else 0.0

    # Sector concentration (mock - would need sector data)
    sector_concentration = calculate_sector_exposure(positions_df)

    # Daily loss metrics
    daily_loss_pct = (total_day_pnl / account_value * 100) if account_value > 0 else 0.0

    # VaR calculation (simplified - would use historical data in production)
    var_95 = calculate_var_95(positions_df, account_value)

    # Leverage ratio (total exposure / equity)
    leverage_ratio = total_exposure / account_value if account_value > 0 else 0.0

    return {
        "portfolio_heat": portfolio_heat,
        "largest_position_pct": largest_position_pct,
        "sector_concentration": sector_concentration,
        "daily_loss": total_day_pnl,
        "daily_loss_pct": daily_loss_pct,
        "var_95": var_95,
        "max_drawdown_pct": 0.0,  # Would need historical data
        "leverage_ratio": leverage_ratio,
    }


def calculate_sector_exposure(positions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate exposure by sector (mock implementation)

    Args:
        positions_df: DataFrame with position data

    Returns:
        Dictionary mapping sector to exposure percentage
    """
    # Mock sector mapping (in production, fetch from database)
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Consumer Cyclical",
        "TSLA": "Consumer Cyclical",
        "JPM": "Financial",
        "BAC": "Financial",
        "XOM": "Energy",
        "CVX": "Energy",
    }

    if positions_df.empty:
        return {}

    sector_exposure = {}
    total_value = positions_df["market_value"].abs().sum()

    for _, row in positions_df.iterrows():
        symbol = row["symbol"]
        sector = sector_map.get(symbol, "Other")
        exposure = abs(row["market_value"]) / total_value * 100 if total_value > 0 else 0

        if sector in sector_exposure:
            sector_exposure[sector] += exposure
        else:
            sector_exposure[sector] = exposure

    return sector_exposure


def calculate_var_95(positions_df: pd.DataFrame, account_value: float) -> float:
    """
    Calculate Value at Risk at 95% confidence (simplified)

    Args:
        positions_df: DataFrame with position data
        account_value: Total account value

    Returns:
        VaR as dollar amount
    """
    # Simplified VaR: assume 2% daily volatility, 95% confidence = 1.65 std devs
    total_exposure = positions_df["market_value"].abs().sum()
    daily_volatility = 0.02  # 2% assumed daily volatility
    confidence_multiplier = 1.65  # 95% confidence (z-score)

    var = total_exposure * daily_volatility * confidence_multiplier
    return var


def create_risk_heatmap(risk_metrics: Dict) -> go.Figure:
    """
    Create risk heatmap visualization

    Args:
        risk_metrics: Dictionary of risk metrics

    Returns:
        Plotly figure
    """
    # Define risk categories and thresholds
    categories = [
        ("Portfolio Heat", risk_metrics["portfolio_heat"], 0.15),  # 15% max
        ("Largest Position", risk_metrics["largest_position_pct"], 0.067),  # 6.7% max
        ("Daily Loss", abs(risk_metrics["daily_loss_pct"] / 100), 0.03),  # 3% max
        ("Leverage", risk_metrics["leverage_ratio"], 1.0),  # 1.0 max
    ]

    names = [cat[0] for cat in categories]
    values = [cat[1] for cat in categories]
    limits = [cat[2] for cat in categories]

    # Calculate risk percentages (current / limit)
    risk_pcts = [(val / lim * 100) if lim > 0 else 0 for val, lim in zip(values, limits)]

    # Create color scale based on risk level
    colors = []
    for pct in risk_pcts:
        if pct < 50:
            colors.append("green")
        elif pct < 75:
            colors.append("yellow")
        elif pct < 90:
            colors.append("orange")
        else:
            colors.append("red")

    fig = go.Figure(
        data=[
            go.Bar(
                x=risk_pcts,
                y=names,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{pct:.1f}%" for pct in risk_pcts],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Risk Utilization (% of Limit)",
        xaxis_title="Utilization (%)",
        yaxis_title="Risk Category",
        height=400,
        showlegend=False,
    )

    # Add threshold line at 100%
    fig.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Limit")

    return fig


def create_sector_concentration_chart(sector_exposure: Dict[str, float]) -> go.Figure:
    """
    Create sector concentration pie chart

    Args:
        sector_exposure: Dictionary of sector exposures

    Returns:
        Plotly figure
    """
    if not sector_exposure:
        fig = go.Figure()
        fig.add_annotation(
            text="No Sector Data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    sectors = list(sector_exposure.keys())
    values = list(sector_exposure.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=sectors,
                values=values,
                hole=0.3,
                textinfo="label+percent",
                marker=dict(
                    colors=px.colors.qualitative.Set2[:len(sectors)]
                ),
            )
        ]
    )

    fig.update_layout(
        title="Sector Concentration",
        height=400,
    )

    return fig


def render_risk_exposure_view():
    """
    Render the risk exposure dashboard
    """
    st.title("⚠️ Risk Exposure Dashboard")
    st.markdown("Real-time risk monitoring and limit tracking")

    # Configuration
    account_value = 30000.00  # Starting capital
    daily_loss_limit = -900.00
    position_limit = 2000.00

    # Fetch positions
    try:
        positions_df = fetch_positions()
        if positions_df is None or positions_df.empty:
            st.warning("No positions currently held")
            positions_df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        positions_df = create_mock_positions()

    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(positions_df, account_value)

    # Summary metrics
    st.subheader("📊 Risk Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        heat_color = "normal" if risk_metrics["portfolio_heat"] <= 0.15 else "inverse"
        st.metric(
            label="Portfolio Heat",
            value=f"{risk_metrics['portfolio_heat']:.1%}",
            delta=f"Limit: 15%",
            delta_color=heat_color,
        )

    with col2:
        largest_color = (
            "normal" if risk_metrics["largest_position_pct"] <= 0.067 else "inverse"
        )
        st.metric(
            label="Largest Position",
            value=f"{risk_metrics['largest_position_pct']:.1%}",
            delta=f"Limit: 6.7%",
            delta_color=largest_color,
        )

    with col3:
        loss_color = "normal" if risk_metrics["daily_loss"] >= 0 else "inverse"
        st.metric(
            label="Daily P/L",
            value=f"${risk_metrics['daily_loss']:,.2f}",
            delta=f"Limit: ${daily_loss_limit:,.2f}",
            delta_color=loss_color,
        )

    with col4:
        st.metric(
            label="Value at Risk (95%)",
            value=f"${risk_metrics['var_95']:,.2f}",
            help="Maximum expected loss with 95% confidence",
        )

    # Risk heatmap
    st.subheader("🌡️ Risk Utilization")
    risk_heatmap = create_risk_heatmap(risk_metrics)
    st.plotly_chart(risk_heatmap, use_container_width=True)

    # Sector concentration
    st.subheader("🏢 Sector Exposure")
    sector_chart = create_sector_concentration_chart(risk_metrics["sector_concentration"])
    st.plotly_chart(sector_chart, use_container_width=True)

    # Position limits check
    st.subheader("🔍 Position Limit Compliance")

    if not positions_df.empty:
        positions_df["limit_pct"] = (
            positions_df["market_value"].abs() / position_limit * 100
        )
        positions_df["within_limit"] = positions_df["market_value"].abs() <= position_limit

        compliant = positions_df["within_limit"].sum()
        total = len(positions_df)

        if compliant == total:
            st.success(f"✅ All {total} positions within ${position_limit:,.2f} limit")
        else:
            violations = total - compliant
            st.error(f"🚨 {violations} positions exceed ${position_limit:,.2f} limit")

        # Show positions sorted by size
        display_df = positions_df[["symbol", "market_value", "limit_pct", "within_limit"]].copy()
        display_df = display_df.sort_values("limit_pct", ascending=False)

        def style_compliance(row):
            return [
                "background-color: lightcoral" if not row["within_limit"] else ""
            ] * len(row)

        styled_df = display_df.style.apply(style_compliance, axis=1)
        st.dataframe(styled_df, use_container_width=True)

    # Risk alerts
    st.subheader("🚨 Risk Alerts")

    alerts = []

    if risk_metrics["portfolio_heat"] > 0.15:
        alerts.append(
            ("CRITICAL", f"Portfolio heat {risk_metrics['portfolio_heat']:.1%} exceeds 15% limit")
        )

    if risk_metrics["daily_loss"] < daily_loss_limit:
        alerts.append(
            ("CRITICAL", f"Daily loss ${risk_metrics['daily_loss']:,.2f} exceeds ${daily_loss_limit:,.2f} limit")
        )

    if risk_metrics["largest_position_pct"] > 0.067:
        alerts.append(
            ("WARNING", f"Largest position {risk_metrics['largest_position_pct']:.1%} exceeds 6.7% limit")
        )

    if risk_metrics["leverage_ratio"] > 1.0:
        alerts.append(
            ("WARNING", f"Leverage ratio {risk_metrics['leverage_ratio']:.2f} exceeds 1.0")
        )

    if alerts:
        for severity, message in alerts:
            if severity == "CRITICAL":
                st.error(f"🚨 {message}")
            else:
                st.warning(f"⚠️ {message}")
    else:
        st.success("✅ No risk alerts - all metrics within limits")


def fetch_positions() -> pd.DataFrame:
    """Fetch current positions"""
    # TODO: Implement via C++ bindings or DuckDB
    return None


def create_mock_positions() -> pd.DataFrame:
    """Create mock position data"""
    mock_data = [
        {
            "symbol": "AAPL",
            "quantity": 50,
            "market_value": 8525.00,
            "day_pnl": 25.00,
        },
        {
            "symbol": "MSFT",
            "quantity": 30,
            "market_value": 11400.00,
            "day_pnl": 30.00,
        },
        {
            "symbol": "GOOGL",
            "quantity": 40,
            "market_value": 5600.00,
            "day_pnl": -20.00,
        },
    ]
    return pd.DataFrame(mock_data)


if __name__ == "__main__":
    render_risk_exposure_view()
