#!/usr/bin/env python3
"""
Rejection Analysis View - Deep dive into why signals aren't trading

Analyzes:
- Rejection reasons breakdown
- Cost distribution for budget rejections
- Optimization recommendations
- Missed opportunities
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def load_rejection_data(days=7):
    """Load rejection data for analysis"""
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'bigbrother.duckdb')
    conn = duckdb.connect(db_path, read_only=True)

    query = f"""
        SELECT
            DATE(timestamp) as date,
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
            days_to_expiration,
            iv_percentile
        FROM trading_signals
        WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
        AND status != 'EXECUTED'
        ORDER BY timestamp DESC
    """

    try:
        df = conn.execute(query).df()
    except:
        df = pd.DataFrame()
    finally:
        conn.close()

    return df

def load_executed_data(days=7):
    """Load executed signals for comparison"""
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'bigbrother.duckdb')
    conn = duckdb.connect(db_path, read_only=True)

    query = f"""
        SELECT
            DATE(timestamp) as date,
            strategy,
            COUNT(*) as executed_count,
            AVG(estimated_cost) as avg_cost
        FROM trading_signals
        WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
        AND status = 'EXECUTED'
        GROUP BY DATE(timestamp), strategy
        ORDER BY date DESC
    """

    try:
        df = conn.execute(query).df()
    except:
        df = pd.DataFrame()
    finally:
        conn.close()

    return df

def show_rejection_analysis():
    """Main function for rejection analysis"""

    st.title("🔍 Signal Rejection Analysis")
    st.caption("Deep dive into why signals aren't becoming trades")

    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        days = st.selectbox(
            "Time Range",
            [1, 7, 30],
            index=1,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Load data
    df = load_rejection_data(days)
    exec_df = load_executed_data(days)

    if df.empty:
        st.success("🎉 No rejected signals in this period! Perfect execution.")
        st.balloons()
        return

    # Overview metrics
    st.subheader("Rejection Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rejections", len(df),
                  help=f"Signals rejected in last {days} days")

    with col2:
        budget_rejections = len(df[df['status'] == 'FILTERED_BUDGET'])
        budget_pct = (budget_rejections / len(df) * 100) if len(df) > 0 else 0
        st.metric("Budget Rejections", budget_rejections,
                  delta=f"{budget_pct:.0f}%",
                  help="Rejected due to $500 position limit")

    with col3:
        missed_return = df['expected_return'].sum()
        st.metric("Missed Potential EV", f"${missed_return:.0f}",
                  delta_color="inverse",
                  help="Total expected value of rejected signals")

    with col4:
        avg_cost = df[df['status'] == 'FILTERED_BUDGET']['estimated_cost'].mean()
        if pd.notna(avg_cost):
            st.metric("Avg Rejected Cost", f"${avg_cost:.0f}",
                      help="Average cost of budget-rejected signals")
        else:
            st.metric("Avg Rejected Cost", "N/A")

    st.divider()

    # Rejection reasons breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rejection Distribution")

        rejection_counts = df['status'].value_counts()

        # Map to friendly names
        status_map = {
            'FILTERED_CONFIDENCE': 'Low Confidence',
            'FILTERED_RETURN': 'Low Expected Return',
            'FILTERED_WIN_PROB': 'Low Win Probability',
            'FILTERED_BUDGET': 'Over Budget',
            'REJECTED_RISK': 'Risk Manager'
        }

        labels = [status_map.get(s, s) for s in rejection_counts.index]
        colors = ['#EA4335', '#FBBC04', '#34A853', '#4285F4', '#9334E6']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=rejection_counts.values,
            hole=0.4,
            marker_colors=colors[:len(labels)]
        )])

        fig.update_layout(
            title="Why Signals Failed",
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rejection Trend")

        daily_rejections = df.groupby(['date', 'status']).size().reset_index(name='count')

        fig = px.bar(
            daily_rejections,
            x='date',
            y='count',
            color='status',
            title="Daily Rejection Count by Reason",
            color_discrete_map={
                'FILTERED_BUDGET': '#EA4335',
                'FILTERED_CONFIDENCE': '#FBBC04',
                'FILTERED_RETURN': '#34A853',
                'FILTERED_WIN_PROB': '#4285F4',
                'REJECTED_RISK': '#9334E6'
            }
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Rejections",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Insights and recommendations
    if budget_rejections > len(df) * 0.3:
        st.warning(f"""
        ⚠️ **Budget Constraint is Limiting Performance**

        **Issue**: {budget_rejections} signals ({budget_pct:.0f}%) rejected due to $500 position limit

        **Impact**: Missing ${df[df['status'] == 'FILTERED_BUDGET']['expected_return'].sum():.0f} in potential returns

        **Recommendation**: Consider increasing `max_position_size` in config.yaml

        Current limit: $500 → Suggested: $750-1000 (based on analysis below)
        """)

    st.divider()

    # Strategy-specific analysis
    st.subheader("Rejection by Strategy")

    strategy_rejections = df.groupby(['strategy', 'status']).size().reset_index(name='count')

    fig = px.bar(
        strategy_rejections,
        x='strategy',
        y='count',
        color='status',
        title="Which Strategies Are Being Rejected?",
        barmode='stack',
        text='count'
    )

    fig.update_layout(
        xaxis_title="Strategy",
        yaxis_title="Rejection Count",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show strategy-specific insights
    strategy_analysis = df.groupby('strategy').agg({
        'signal_id': 'count',
        'expected_return': 'sum',
        'status': lambda x: x.value_counts().index[0]  # Most common rejection
    }).rename(columns={
        'signal_id': 'Rejections',
        'expected_return': 'Missed EV',
        'status': 'Primary Rejection'
    })

    st.dataframe(
        strategy_analysis,
        column_config={
            "Missed EV": st.column_config.NumberColumn(format="$%.0f"),
        },
        use_container_width=True
    )

    st.divider()

    # Cost distribution analysis (critical for budget optimization)
    if budget_rejections > 0:
        st.subheader("💰 Budget Optimization Analysis")

        budget_df = df[df['status'] == 'FILTERED_BUDGET']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cost Distribution of Rejected Signals**")

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=budget_df['estimated_cost'],
                nbinsx=20,
                name="Rejected Signals",
                marker_color='indianred'
            ))

            # Add current limit line
            fig.add_vline(
                x=500,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text="Current $500 Limit",
                annotation_position="top"
            )

            # Add suggested optimal limit
            optimal_limit = budget_df['estimated_cost'].quantile(0.75)
            fig.add_vline(
                x=optimal_limit,
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text=f"75th percentile: ${optimal_limit:.0f}",
                annotation_position="bottom right"
            )

            fig.update_layout(
                xaxis_title="Signal Cost ($)",
                yaxis_title="Number of Signals",
                title="How much over budget?",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Impact Analysis**")

            # Calculate impact at different limits
            limits = [500, 600, 700, 750, 800, 900, 1000]
            impact_data = []

            for limit in limits:
                would_pass = budget_df[budget_df['estimated_cost'] <= limit]
                count = len(would_pass)
                missed_ev = would_pass['expected_return'].sum()
                pct = (count / len(budget_df) * 100) if len(budget_df) > 0 else 0

                impact_data.append({
                    'Limit': f"${limit}",
                    'Signals Captured': count,
                    '% Captured': f"{pct:.0f}%",
                    'Additional EV': f"${missed_ev:.0f}"
                })

            impact_df = pd.DataFrame(impact_data)

            st.dataframe(
                impact_df,
                hide_index=True,
                use_container_width=True
            )

            # Highlight optimal
            optimal_limit_rounded = round(optimal_limit, -1)  # Round to nearest 10
            st.success(f"""
            💡 **Recommended Action**

            Increase limit to **${optimal_limit_rounded:.0f}**

            This would capture 75% of currently rejected signals
            and add ${budget_df[budget_df['estimated_cost'] <= optimal_limit]['expected_return'].sum():.0f} in expected value.
            """)

    st.divider()

    # Recent rejections table
    st.subheader("Recent Rejections (Last 50)")

    display_df = df.head(50)[['date', 'timestamp', 'strategy', 'symbol',
                               'confidence', 'expected_return', 'estimated_cost',
                               'status', 'rejection_reason']]

    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')

    st.dataframe(
        display_df,
        column_config={
            "timestamp": "Time",
            "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            "expected_return": st.column_config.NumberColumn("Expected EV", format="$%.0f"),
            "estimated_cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
            "status": st.column_config.TextColumn("Status"),
        },
        use_container_width=True,
        hide_index=True
    )

    # Export functionality
    st.divider()
    st.subheader("📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Rejections as CSV",
            data=csv,
            file_name=f"rejections_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        if not exec_df.empty:
            exec_csv = exec_df.to_csv(index=False)
            st.download_button(
                label="Download Executions as CSV",
                data=exec_csv,
                file_name=f"executions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    show_rejection_analysis()
