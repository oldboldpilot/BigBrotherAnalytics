#!/usr/bin/env python3
"""
Live Trading Activity View - Real-time signal and trade monitoring

Shows:
- Real-time signal generation stream
- Signal flow through filters
- Rejection reasons analysis
- Key metrics and alerts
"""

import streamlit as st
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def load_todays_signals():
    """Load all signals generated today"""
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'bigbrother.duckdb')
    conn = duckdb.connect(db_path, read_only=True)

    query = """
        SELECT
            timestamp,
            strategy,
            symbol,
            confidence,
            expected_return,
            win_probability,
            estimated_cost,
            max_risk,
            status,
            rejection_reason,
            greeks_delta,
            greeks_theta,
            greeks_vega,
            iv_percentile,
            days_to_expiration
        FROM trading_signals
        WHERE DATE(timestamp) = CURRENT_DATE
        ORDER BY timestamp DESC
    """

    try:
        df = conn.execute(query).df()
    except:
        df = pd.DataFrame()
    finally:
        conn.close()

    return df

def create_signal_flow_sankey(df):
    """Create Sankey diagram showing signal flow through filters"""

    if df.empty:
        return None

    # Count signals by status
    total_generated = len(df)
    executed = len(df[df['status'] == 'EXECUTED'])
    filtered_conf = len(df[df['status'] == 'FILTERED_CONFIDENCE'])
    filtered_return = len(df[df['status'] == 'FILTERED_RETURN'])
    filtered_winprob = len(df[df['status'] == 'FILTERED_WIN_PROB'])
    filtered_budget = len(df[df['status'] == 'FILTERED_BUDGET'])
    rejected_risk = len(df[df['status'] == 'REJECTED_RISK'])

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = [
                "Signals Generated",
                "Confidence Filter",
                "Return Filter",
                "Win Prob Filter",
                "Budget Filter",
                "Risk Manager",
                "✅ Executed",
                "❌ Rejected"
            ],
            color = ["#4285F4", "#FBBC04", "#FBBC04", "#FBBC04", "#FBBC04",
                     "#FBBC04", "#34A853", "#EA4335"]
        ),
        link = dict(
            source = [0, 0, 0, 0, 0, 0],
            target = [6, 7, 7, 7, 7, 7],
            value = [executed, filtered_conf, filtered_return,
                     filtered_winprob, filtered_budget, rejected_risk],
            label = ["Passed all checks", "Low confidence", "Low return",
                     "Low win prob", "Over budget", "Risk rejected"]
        )
    )])

    fig.update_layout(
        title_text="Signal Flow Through Filters",
        font_size=12,
        height=400
    )

    return fig

def create_rejection_pie(df):
    """Create pie chart of rejection reasons"""

    rejection_df = df[df['status'] != 'EXECUTED']

    if rejection_df.empty:
        return None

    rejection_counts = rejection_df['status'].value_counts()

    # Map status to friendly names
    status_map = {
        'FILTERED_CONFIDENCE': 'Low Confidence (<60%)',
        'FILTERED_RETURN': 'Low Expected Return (<$50)',
        'FILTERED_WIN_PROB': 'Low Win Probability (<60%)',
        'FILTERED_BUDGET': 'Over Budget (>$500)',
        'REJECTED_RISK': 'Risk Manager Rejection'
    }

    labels = [status_map.get(s, s) for s in rejection_counts.index]

    fig = px.pie(
        values=rejection_counts.values,
        names=labels,
        title="Rejection Reasons",
        color_discrete_sequence=px.colors.sequential.RdBu
    )

    return fig

def show_live_activity():
    """Main function to display live trading activity"""

    st.title("📊 Live Trading Activity")
    st.caption(f"Real-time signal monitoring | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.rerun()
    with col2:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with col3:
        if auto_refresh:
            refresh_interval = st.selectbox("Interval", [30, 60, 120],
                                           format_func=lambda x: f"{x}s",
                                           index=1)

    # Load data
    df = load_todays_signals()

    if df.empty:
        st.info("ℹ️ No signals generated today yet. Waiting for market open...")
        st.caption("Signals will appear here as the trading engine generates them")
        return

    # Key metrics row
    st.subheader("Today's Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Signals", len(df),
                  help="All signals generated by strategies today")

    with col2:
        executed = len(df[df['status'] == 'EXECUTED'])
        exec_pct = (executed / len(df) * 100) if len(df) > 0 else 0
        st.metric("✅ Executed", executed,
                  delta=f"{exec_pct:.0f}%",
                  help="Signals that passed all filters and were traded")

    with col3:
        rejected = len(df[df['status'] != 'EXECUTED'])
        rej_pct = (rejected / len(df) * 100) if len(df) > 0 else 0
        st.metric("❌ Rejected", rejected,
                  delta=f"{rej_pct:.0f}%",
                  delta_color="inverse",
                  help="Signals that failed filters or risk checks")

    with col4:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}",
                  help="Average confidence across all signals")

    with col5:
        total_potential = df['expected_return'].sum()
        actual_potential = df[df['status'] == 'EXECUTED']['expected_return'].sum()
        st.metric("Total Potential EV", f"${total_potential:.0f}",
                  delta=f"${actual_potential:.0f} captured",
                  help="Sum of expected returns for all signals")

    st.divider()

    # Signal flow diagram
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Signal Flow Analysis")
        sankey_fig = create_signal_flow_sankey(df)
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info("Not enough data for flow diagram")

    with col2:
        st.subheader("Rejection Breakdown")
        pie_fig = create_rejection_pie(df)
        if pie_fig:
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.success("✅ No rejections today!")

    # Alert for common rejection reason
    rejection_df = df[df['status'] != 'EXECUTED']
    if not rejection_df.empty:
        most_common = rejection_df['status'].value_counts()
        if len(most_common) > 0:
            top_reason = most_common.index[0]
            top_count = most_common.values[0]

            if top_reason == 'FILTERED_BUDGET' and top_count >= 3:
                st.warning(f"""
                ⚠️ **Budget Constraint Alert**

                {top_count} signals ({top_count/len(df)*100:.0f}%) were rejected due to the $500 position limit.

                **Missed potential**: ${rejection_df[rejection_df['status'] == 'FILTERED_BUDGET']['expected_return'].sum():.0f}

                Consider increasing max_position_size in config.yaml to capture more opportunities.
                """)

    st.divider()

    # Strategy performance table
    st.subheader("Performance by Strategy")

    strategy_stats = df.groupby('strategy').agg({
        'signal_id': 'count',
        'confidence': 'mean',
        'expected_return': 'mean',
        'estimated_cost': 'mean'
    }).rename(columns={
        'signal_id': 'Signals',
        'confidence': 'Avg Confidence',
        'expected_return': 'Avg Expected Return',
        'estimated_cost': 'Avg Cost'
    })

    # Add execution rate
    exec_by_strategy = df[df['status'] == 'EXECUTED'].groupby('strategy').size()
    strategy_stats['Executed'] = exec_by_strategy
    strategy_stats['Executed'] = strategy_stats['Executed'].fillna(0).astype(int)
    strategy_stats['Execution Rate'] = (strategy_stats['Executed'] / strategy_stats['Signals'] * 100).round(1)

    st.dataframe(
        strategy_stats,
        column_config={
            "Avg Confidence": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg Expected Return": st.column_config.NumberColumn(format="$%.0f"),
            "Avg Cost": st.column_config.NumberColumn(format="$%.0f"),
            "Execution Rate": st.column_config.NumberColumn(format="%.1f%%"),
        },
        use_container_width=True
    )

    st.divider()

    # Recent signals timeline
    st.subheader("Recent Signals (Latest 20)")

    for idx, row in df.head(20).iterrows():
        status_emoji = "✅" if row['status'] == 'EXECUTED' else "❌"

        with st.expander(
            f"{status_emoji} {row['timestamp'].strftime('%H:%M:%S')} - "
            f"{row['symbol']} ({row['strategy']})",
            expanded=(idx == 0)  # Expand first one
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Confidence", f"{row['confidence']:.1%}")
                st.metric("Expected Return", f"${row['expected_return']:.0f}")
                st.metric("Win Probability", f"{row['win_probability']:.1%}")

            with col2:
                st.metric("Estimated Cost", f"${row['estimated_cost']:.0f}")
                st.metric("Max Risk", f"${row['max_risk']:.0f}")
                st.metric("Status", row['status'].replace('_', ' ').title())

            with col3:
                if pd.notna(row['days_to_expiration']):
                    st.metric("DTE", f"{int(row['days_to_expiration'])}")
                if pd.notna(row['iv_percentile']):
                    st.metric("IV Percentile", f"{row['iv_percentile']:.0f}")
                if row['rejection_reason']:
                    st.error(f"**Rejection**: {row['rejection_reason']}")

            # Greeks (if available)
            if pd.notna(row['greeks_delta']):
                st.caption("**Greeks at Signal Generation**")
                greeks_cols = st.columns(3)
                with greeks_cols[0]:
                    st.text(f"Delta (Δ): {row['greeks_delta']:.3f}")
                with greeks_cols[1]:
                    st.text(f"Theta (Θ): {row['greeks_theta']:.2f}")
                with greeks_cols[2]:
                    st.text(f"Vega (ν): {row['greeks_vega']:.1f}")

if __name__ == "__main__":
    show_live_activity()
