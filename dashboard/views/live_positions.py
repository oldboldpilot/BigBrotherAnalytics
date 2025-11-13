"""
BigBrotherAnalytics - Live Positions Monitoring View

Real-time position monitoring with P/L tracking and risk metrics.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-12
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_position_metrics(positions_df: pd.DataFrame) -> Dict:
    """
    Calculate aggregate position metrics

    Args:
        positions_df: DataFrame with position data

    Returns:
        Dictionary of metrics
    """
    if positions_df.empty:
        return {
            "total_positions": 0,
            "total_market_value": 0.0,
            "total_cost_basis": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_unrealized_pnl_pct": 0.0,
            "total_day_pnl": 0.0,
            "long_positions": 0,
            "short_positions": 0,
            "largest_position": 0.0,
            "largest_position_symbol": "N/A",
        }

    total_market_value = positions_df["market_value"].sum()
    total_cost_basis = positions_df["cost_basis"].sum()
    total_unrealized_pnl = positions_df["unrealized_pnl"].sum()
    total_day_pnl = positions_df["day_pnl"].sum()

    long_positions = len(positions_df[positions_df["quantity"] > 0])
    short_positions = len(positions_df[positions_df["quantity"] < 0])

    # Find largest position by market value
    if not positions_df.empty:
        largest_idx = positions_df["market_value"].abs().idxmax()
        largest_position = positions_df.loc[largest_idx, "market_value"]
        largest_symbol = positions_df.loc[largest_idx, "symbol"]
    else:
        largest_position = 0.0
        largest_symbol = "N/A"

    return {
        "total_positions": len(positions_df),
        "total_market_value": total_market_value,
        "total_cost_basis": total_cost_basis,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_unrealized_pnl_pct": (
            (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis != 0 else 0.0
        ),
        "total_day_pnl": total_day_pnl,
        "long_positions": long_positions,
        "short_positions": short_positions,
        "largest_position": largest_position,
        "largest_position_symbol": largest_symbol,
    }


def create_pnl_gauge(
    current_pnl: float, max_loss: float = -900.0, max_gain: float = 900.0
) -> go.Figure:
    """
    Create a gauge chart for P/L visualization

    Args:
        current_pnl: Current profit/loss
        max_loss: Maximum acceptable loss (negative)
        max_gain: Target maximum gain

    Returns:
        Plotly figure
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_pnl,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Today's P/L", "font": {"size": 24}},
            delta={"reference": 0, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
            gauge={
                "axis": {"range": [max_loss, max_gain], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [max_loss, max_loss * 0.5], "color": "lightcoral"},
                    {"range": [max_loss * 0.5, 0], "color": "lightyellow"},
                    {"range": [0, max_gain * 0.5], "color": "lightgreen"},
                    {"range": [max_gain * 0.5, max_gain], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_loss,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="white",
        font={"color": "black", "family": "Arial"},
        height=300,
    )

    return fig


def create_position_allocation_chart(positions_df: pd.DataFrame) -> go.Figure:
    """
    Create pie chart showing position allocation

    Args:
        positions_df: DataFrame with position data

    Returns:
        Plotly figure
    """
    if positions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Positions",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 20},
        )
        return fig

    # Group by symbol and sum market values
    allocation = (
        positions_df.groupby("symbol")["market_value"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=allocation.index,
                values=allocation.values,
                hole=0.3,
                textposition="auto",
                textinfo="label+percent",
            )
        ]
    )

    fig.update_layout(
        title="Position Allocation by Symbol",
        showlegend=True,
        height=400,
    )

    return fig


def render_live_positions_view():
    """
    Render the live positions monitoring view
    """
    st.title("📊 Live Positions Monitor")
    st.markdown("Real-time position tracking with P/L and risk metrics")

    # Add refresh button and auto-refresh toggle
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 Refresh Now"):
            st.rerun()

    with col2:
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

    with col3:
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"Last Update: {last_update}")

    # Auto-refresh logic
    if auto_refresh:
        st.markdown(
            """
            <meta http-equiv="refresh" content="60">
            """,
            unsafe_allow_html=True,
        )

    # Try to fetch live positions
    try:
        # Import position fetching functionality
        # This would connect to your C++ bindings or database
        positions_df = fetch_current_positions()

        if positions_df is None or positions_df.empty:
            st.warning("⚠️ No positions currently held")
            positions_df = pd.DataFrame()
        else:
            st.success(f"✅ {len(positions_df)} active positions")

    except Exception as e:
        st.error(f"❌ Error fetching positions: {e}")
        st.info("Using mock data for demonstration...")
        positions_df = create_mock_positions()

    # Calculate metrics
    metrics = calculate_position_metrics(positions_df)

    # Display summary metrics
    st.subheader("📈 Position Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Positions",
            value=metrics["total_positions"],
            delta=f"{metrics['long_positions']} Long, {metrics['short_positions']} Short",
        )

    with col2:
        st.metric(
            label="Market Value",
            value=f"${metrics['total_market_value']:,.2f}",
            delta=f"${metrics['total_cost_basis']:,.2f} Cost",
        )

    with col3:
        pnl_color = "normal" if metrics["total_unrealized_pnl"] >= 0 else "inverse"
        st.metric(
            label="Unrealized P/L",
            value=f"${metrics['total_unrealized_pnl']:,.2f}",
            delta=f"{metrics['total_unrealized_pnl_pct']:.2f}%",
            delta_color=pnl_color,
        )

    with col4:
        day_pnl_color = "normal" if metrics["total_day_pnl"] >= 0 else "inverse"
        st.metric(
            label="Today's P/L",
            value=f"${metrics['total_day_pnl']:,.2f}",
            delta_color=day_pnl_color,
        )

    # P/L Gauge Chart
    st.subheader("💰 P/L Gauge")
    pnl_gauge = create_pnl_gauge(
        current_pnl=metrics["total_day_pnl"],
        max_loss=-900.0,  # $900 max daily loss
        max_gain=900.0,
    )
    st.plotly_chart(pnl_gauge, use_container_width=True)

    # Position allocation
    if not positions_df.empty:
        st.subheader("🥧 Position Allocation")
        allocation_chart = create_position_allocation_chart(positions_df)
        st.plotly_chart(allocation_chart, use_container_width=True)

    # Detailed position table
    st.subheader("📋 Position Details")

    if not positions_df.empty:
        # Format the dataframe for display
        display_df = positions_df[
            [
                "symbol",
                "quantity",
                "current_price",
                "market_value",
                "unrealized_pnl",
                "unrealized_pnl_percent",
                "day_pnl",
            ]
        ].copy()

        # Style the dataframe
        def style_pnl(val):
            color = "green" if val >= 0 else "red"
            return f"color: {color}"

        styled_df = display_df.style.applymap(style_pnl, subset=["unrealized_pnl", "day_pnl"])

        st.dataframe(styled_df, use_container_width=True, height=400)

        # Risk warnings
        st.subheader("⚠️ Risk Checks")

        # Check if any position exceeds limits
        position_limit = 2000.00
        oversized_positions = display_df[display_df["market_value"].abs() > position_limit]

        if not oversized_positions.empty:
            st.error(
                f"🚨 {len(oversized_positions)} positions exceed ${position_limit:,.2f} limit:"
            )
            st.dataframe(oversized_positions)
        else:
            st.success("✅ All positions within $2,000 limit")

        # Check daily loss
        daily_loss_limit = -900.00
        if metrics["total_day_pnl"] < daily_loss_limit:
            st.error(f"🚨 Daily loss ${metrics['total_day_pnl']:,.2f} exceeds limit ${daily_loss_limit:,.2f}")
        else:
            remaining_loss = daily_loss_limit - metrics["total_day_pnl"]
            st.info(f"ℹ️ Daily loss buffer: ${remaining_loss:,.2f}")

    else:
        st.info("No positions to display")


def fetch_current_positions() -> Optional[pd.DataFrame]:
    """
    Fetch current positions from database or API

    Returns:
        DataFrame with position data or None
    """
    # TODO: Implement actual position fetching via C++ bindings or DuckDB
    # For now, return None to trigger mock data
    return None


def create_mock_positions() -> pd.DataFrame:
    """
    Create mock position data for demonstration

    Returns:
        DataFrame with mock positions
    """
    mock_data = [
        {
            "symbol": "AAPL",
            "quantity": 50,
            "current_price": 170.50,
            "average_cost": 168.00,
            "market_value": 8525.00,
            "cost_basis": 8400.00,
            "unrealized_pnl": 125.00,
            "unrealized_pnl_percent": 1.49,
            "day_pnl": 25.00,
        },
        {
            "symbol": "MSFT",
            "quantity": 30,
            "current_price": 380.00,
            "average_cost": 375.00,
            "market_value": 11400.00,
            "cost_basis": 11250.00,
            "unrealized_pnl": 150.00,
            "unrealized_pnl_percent": 1.33,
            "day_pnl": 30.00,
        },
        {
            "symbol": "GOOGL",
            "quantity": 40,
            "current_price": 140.00,
            "average_cost": 142.00,
            "market_value": 5600.00,
            "cost_basis": 5680.00,
            "unrealized_pnl": -80.00,
            "unrealized_pnl_percent": -1.41,
            "day_pnl": -20.00,
        },
    ]

    return pd.DataFrame(mock_data)


if __name__ == "__main__":
    # Test the view
    render_live_positions_view()
