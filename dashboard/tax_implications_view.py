"""
Tax Implications View for BigBrother Dashboard

Shows comprehensive tax calculations including:
- 3% trading fees
- Short-term vs long-term capital gains
- Federal, state, and Medicare surtax
- After-tax P&L
- Tax efficiency metrics

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def load_tax_summary(conn):
    """Load YTD tax summary"""
    try:
        query = "SELECT * FROM v_ytd_tax_summary LIMIT 1"
        result = conn.execute(query).fetchone()
        if result:
            return {
                'total_gross_pnl': result[0] or 0.0,
                'total_trading_fees': result[1] or 0.0,
                'total_pnl_after_fees': result[2] or 0.0,
                'total_tax_owed': result[3] or 0.0,
                'total_net_after_tax': result[4] or 0.0,
                'avg_effective_tax_rate': result[5] or 0.0,
                'total_trades': result[6] or 0,
                'wash_sales_count': result[7] or 0,
                'total_wash_sale_loss': result[8] or 0.0
            }
    except Exception as e:
        st.warning(f"Tax tables not initialized. Run: `uv run python scripts/monitoring/setup_tax_database.py`")
        return None

def load_tax_records(conn):
    """Load individual tax records"""
    try:
        query = """
        SELECT
            symbol,
            entry_time,
            exit_time,
            holding_period_days,
            gross_pnl,
            trading_fees,
            pnl_after_fees,
            tax_owed,
            net_pnl_after_tax,
            effective_tax_rate,
            is_long_term,
            wash_sale_disallowed
        FROM tax_records
        ORDER BY exit_time DESC
        LIMIT 100
        """
        return conn.execute(query).df()
    except:
        return pd.DataFrame()

def load_monthly_tax_summary(conn):
    """Load monthly tax breakdown"""
    try:
        query = "SELECT * FROM v_monthly_tax_summary LIMIT 12"
        return conn.execute(query).df()
    except:
        return pd.DataFrame()

def show_tax_implications(db_conn):
    """Display Tax Implications View with 3% fee"""
    st.header("💰 Tax Implications & After-Tax Performance")

    # Load data
    tax_summary = load_tax_summary(db_conn)

    if tax_summary is None:
        st.error("Tax tracking not set up. Please run the setup script:")
        st.code("uv run python scripts/monitoring/setup_tax_database.py", language="bash")
        st.code("uv run python scripts/monitoring/calculate_taxes.py", language="bash")
        return

    # YTD Summary Metrics
    st.subheader("📊 Year-to-Date Tax Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Gross P&L",
            f"${tax_summary['total_gross_pnl']:,.2f}",
            help="Total profit/loss before fees and taxes"
        )

    with col2:
        st.metric(
            "Trading Fees (3%)",
            f"${tax_summary['total_trading_fees']:,.2f}",
            help="3% trading fee on all transactions"
        )

    with col3:
        st.metric(
            "P&L After Fees",
            f"${tax_summary['total_pnl_after_fees']:,.2f}",
            help="P&L after deducting 3% trading fees"
        )

    with col4:
        st.metric(
            "Tax Owed",
            f"${tax_summary['total_tax_owed']:,.2f}",
            help="Federal + State + Medicare surtax"
        )

    with col5:
        st.metric(
            "Net After-Tax",
            f"${tax_summary['total_net_after_tax']:,.2f}",
            help="Final profit after fees and taxes",
            delta_color="normal" if tax_summary['total_net_after_tax'] >= 0 else "inverse"
        )

    st.divider()

    # Tax Rate Breakdown
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Effective Tax Rate",
            f"{tax_summary['avg_effective_tax_rate']*100:.1f}%",
            help="Average tax rate across all trades"
        )

    with col2:
        tax_efficiency = (tax_summary['total_net_after_tax'] / tax_summary['total_gross_pnl'] * 100) if tax_summary['total_gross_pnl'] > 0 else 0
        st.metric(
            "Tax Efficiency",
            f"{tax_efficiency:.1f}%",
            help="Net after-tax P&L / Gross P&L (higher is better)"
        )

    with col3:
        st.metric(
            "Total Trades",
            f"{tax_summary['total_trades']:,}",
            help="Number of closed trades analyzed"
        )

    # Wash Sales Warning
    if tax_summary['wash_sales_count'] > 0:
        st.warning(f"⚠️ {tax_summary['wash_sales_count']} wash sale(s) detected! ${tax_summary['total_wash_sale_loss']:,.2f} in losses disallowed.")
        with st.expander("What is a wash sale?"):
            st.markdown("""
            **IRS Wash Sale Rule:** If you sell a security at a loss and buy a substantially
            identical security within 30 days before or after the sale, you cannot deduct the loss.

            **Impact:** The disallowed loss is added to the cost basis of the replacement security,
            deferring the tax benefit to a future sale.

            **How to avoid:**
            - Wait 31 days before repurchasing the same security after a loss
            - Trade different but correlated securities
            - Use tax-loss harvesting strategically
            """)

    st.divider()

    # Waterfall Chart: Gross P&L → Net After-Tax
    st.subheader("💸 P&L Waterfall: From Gross to Net After-Tax")

    waterfall_data = [
        ("Gross P&L", tax_summary['total_gross_pnl'], "relative"),
        ("Trading Fees (3%)", -tax_summary['total_trading_fees'], "relative"),
        ("P&L After Fees", tax_summary['total_pnl_after_fees'], "total"),
        ("Taxes", -tax_summary['total_tax_owed'], "relative"),
        ("Net After-Tax", tax_summary['total_net_after_tax'], "total")
    ]

    fig = go.Figure(go.Waterfall(
        x=[item[0] for item in waterfall_data],
        y=[item[1] for item in waterfall_data],
        measure=[item[2] for item in waterfall_data],
        text=[f"${item[1]:,.2f}" for item in waterfall_data],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))

    fig.update_layout(
        title="P&L Waterfall: Impact of Fees and Taxes",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Tax Configuration
    st.subheader("⚙️ Current Tax Configuration")

    try:
        config = db_conn.execute("SELECT * FROM tax_config WHERE id = 1").fetchone()
        if config:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                **Federal Tax Rates:**
                - Short-term: {config[2]*100:.1f}% (≤365 days)
                - Long-term: {config[3]*100:.1f}% (>365 days)
                """)

            with col2:
                st.markdown(f"""
                **Additional Taxes:**
                - State tax: {config[4]*100:.1f}%
                - Medicare surtax: {config[5]*100:.2f}%
                - **Total short-term: {(config[2]+config[4]+config[5])*100:.1f}%**
                - **Total long-term: {(config[3]+config[4]+config[5])*100:.1f}%**
                """)

            with col3:
                st.markdown(f"""
                **Trading Fees:**
                - Per-trade fee: {config[6]*100:.1f}%
                - Pattern day trader: {'Yes' if config[7] else 'No'}
                - Wash sale tracking: {'Enabled' if config[9] else 'Disabled'}
                """)
    except:
        st.info("Tax configuration not loaded")

    st.divider()

    # Monthly Tax Trend
    monthly_df = load_monthly_tax_summary(db_conn)

    if not monthly_df.empty:
        st.subheader("📅 Monthly Tax Breakdown")

        # Format month-year labels
        monthly_df['month_year'] = monthly_df.apply(
            lambda row: f"{int(row['year'])}-{int(row['month']):02d}", axis=1
        )

        col1, col2 = st.columns(2)

        with col1:
            # Monthly Gross vs Net After-Tax
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_df['month_year'],
                y=monthly_df['gross_pnl'],
                name='Gross P&L',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=monthly_df['month_year'],
                y=monthly_df['net_after_tax'],
                name='Net After-Tax',
                marker_color='darkblue'
            ))
            fig.update_layout(
                title="Monthly: Gross P&L vs Net After-Tax",
                xaxis_title="Month",
                yaxis_title="P&L ($)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Monthly Tax Paid
            fig = px.bar(
                monthly_df,
                x='month_year',
                y='tax_owed',
                title='Monthly Tax Paid',
                labels={'tax_owed': 'Tax ($)', 'month_year': 'Month'},
                color='tax_owed',
                color_continuous_scale=['green', 'orange', 'red']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Individual Trade Tax Records
    tax_records_df = load_tax_records(db_conn)

    if not tax_records_df.empty:
        st.subheader("📋 Individual Trade Tax Records")

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)

        short_term_trades = len(tax_records_df[tax_records_df['is_long_term'] == False])
        long_term_trades = len(tax_records_df[tax_records_df['is_long_term'] == True])

        with col1:
            st.metric("Short-Term Trades", short_term_trades)

        with col2:
            st.metric("Long-Term Trades", long_term_trades)

        with col3:
            avg_holding = tax_records_df['holding_period_days'].mean()
            st.metric("Avg Holding Period", f"{avg_holding:.0f} days")

        with col4:
            avg_tax_rate = tax_records_df['effective_tax_rate'].mean()
            st.metric("Avg Tax Rate", f"{avg_tax_rate*100:.1f}%")

        # Trade records table
        display_df = tax_records_df.copy()
        display_df['holding_period_days'] = display_df['holding_period_days'].astype(int)
        display_df['term'] = display_df['is_long_term'].apply(lambda x: 'Long' if x else 'Short')

        st.dataframe(
            display_df,
            column_config={
                "symbol": "Symbol",
                "entry_time": st.column_config.DatetimeColumn("Entry", format="MM/DD/YY"),
                "exit_time": st.column_config.DatetimeColumn("Exit", format="MM/DD/YY"),
                "holding_period_days": "Days Held",
                "term": "Term",
                "gross_pnl": st.column_config.NumberColumn("Gross P&L", format="$%.2f"),
                "trading_fees": st.column_config.NumberColumn("Fees (3%)", format="$%.2f"),
                "pnl_after_fees": st.column_config.NumberColumn("P&L After Fees", format="$%.2f"),
                "tax_owed": st.column_config.NumberColumn("Tax Owed", format="$%.2f"),
                "net_pnl_after_tax": st.column_config.NumberColumn("Net After-Tax", format="$%.2f"),
                "effective_tax_rate": st.column_config.NumberColumn("Tax Rate", format="%.1f%%")
            },
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Action Items
    st.subheader("🎯 Tax Planning Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Optimize Tax Strategy:**
        - ✅ Hold positions >365 days for lower long-term rates
        - ✅ Harvest tax losses strategically (avoid wash sales)
        - ✅ Consider Section 1256 contracts (60/40 rule)
        - ✅ Make quarterly estimated tax payments
        """)

    with col2:
        st.markdown("""
        **Scripts to Run:**
        ```bash
        # Update tax calculations
        uv run python scripts/monitoring/calculate_taxes.py

        # View tax report
        uv run python scripts/monitoring/generate_tax_report.py
        ```
        """)
