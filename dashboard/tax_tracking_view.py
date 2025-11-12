"""
Tax Tracking View - Real-time cumulative tax calculations
Displays YTD tax liability, effective tax rates, and tax-adjusted returns
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta


def show_realtime_tax_tracking(get_db_connection):
    """Display real-time tax cumulative calculations with comprehensive tax lot analysis"""
    st.header("💰 Tax Tracking - Real-Time Cumulative Calculations")
    st.markdown("""
    Year-to-date tax liability tracking with advanced analytics. Track your realized gains/losses,
    tax obligations, and optimize your trading strategies for tax efficiency.
    """)

    conn = get_db_connection()

    # Tax rate constants
    SHORT_TERM_TAX_RATE = 0.371  # 37.1% (federal + CA + Medicare)
    LONG_TERM_TAX_RATE = 0.281   # 28.1%
    BASE_INCOME = 300000

    # Load tax lots data
    try:
        # Query all closed lots with realized gains/losses
        closed_lots_query = """
            SELECT
                id,
                account_id,
                symbol,
                asset_type,
                quantity,
                entry_price,
                entry_date,
                close_price,
                close_date,
                strategy,
                realized_pnl,
                holding_period_days,
                is_long_term,
                option_type,
                strike_price,
                expiration_date,
                underlying_symbol
            FROM tax_lots
            WHERE is_closed = true
            AND close_date IS NOT NULL
            ORDER BY close_date DESC
        """
        closed_lots_df = conn.execute(closed_lots_query).df()

        # Query all open lots (for unrealized position info)
        open_lots_query = """
            SELECT
                symbol,
                asset_type,
                quantity,
                entry_price,
                entry_date,
                strategy,
                holding_period_days,
                is_long_term
            FROM tax_lots
            WHERE is_closed = false
        """
        open_lots_df = conn.execute(open_lots_query).df()

    except Exception as e:
        st.error(f"Error loading tax lots: {e}")
        st.info("Run: `uv run python scripts/setup_tax_lots_table.py` to create the tax_lots table")
        return

    # Check if we have any closed lots
    has_closed_lots = not closed_lots_df.empty

    if not has_closed_lots:
        st.info("📭 No closed positions yet. Tax liability will appear once positions are closed.")
        st.markdown("""
        ### What You'll See Here

        Once you close positions, this view will display:
        - **YTD Tax Liability**: Total estimated tax owed on realized gains
        - **Short-term vs Long-term**: Breakdown of capital gains by holding period
        - **Effective Tax Rate**: Your actual tax rate on trading profits
        - **Tax-Adjusted Returns**: Net returns after tax obligations
        - **Strategy Analysis**: Which strategies are most tax-efficient
        - **Wash Sale Warnings**: Potential wash sale violations

        ### Tax Rates Used
        - **Short-term** (≤365 days): {:.1f}% (ordinary income rate)
        - **Long-term** (>365 days): {:.1f}% (preferential rate)
        - **Base Income**: ${:,} (affects tax brackets)
        """.format(SHORT_TERM_TAX_RATE * 100, LONG_TERM_TAX_RATE * 100, BASE_INCOME))

        # Show open positions summary
        if not open_lots_df.empty:
            st.divider()
            st.subheader("📈 Open Positions (Unrealized)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Open Lots", len(open_lots_df))
            with col2:
                long_term_count = len(open_lots_df[open_lots_df['is_long_term'] == True])
                st.metric("Long-term Holdings", long_term_count)
            with col3:
                short_term_count = len(open_lots_df[open_lots_df['is_long_term'] == False])
                st.metric("Short-term Holdings", short_term_count)

        return

    # Calculate tax metrics from closed lots
    # Separate short-term and long-term gains
    closed_lots_df['short_term_gain'] = closed_lots_df.apply(
        lambda x: x['realized_pnl'] if not x['is_long_term'] else 0, axis=1
    )
    closed_lots_df['long_term_gain'] = closed_lots_df.apply(
        lambda x: x['realized_pnl'] if x['is_long_term'] else 0, axis=1
    )

    # Calculate total gains/losses
    total_short_term_gain = closed_lots_df['short_term_gain'].sum()
    total_long_term_gain = closed_lots_df['long_term_gain'].sum()
    total_realized_pnl = closed_lots_df['realized_pnl'].sum()

    # Calculate tax liability (only on gains, not losses)
    short_term_tax = max(total_short_term_gain, 0) * SHORT_TERM_TAX_RATE
    long_term_tax = max(total_long_term_gain, 0) * LONG_TERM_TAX_RATE
    total_tax_liability = short_term_tax + long_term_tax

    # Calculate proceeds (total sale value)
    closed_lots_df['proceeds'] = closed_lots_df['quantity'] * closed_lots_df['close_price']
    total_proceeds = closed_lots_df['proceeds'].sum()

    # Calculate cost basis
    closed_lots_df['cost_basis'] = closed_lots_df['quantity'] * closed_lots_df['entry_price']
    total_cost_basis = closed_lots_df['cost_basis'].sum()

    # Effective tax rate
    effective_tax_rate = (total_tax_liability / total_realized_pnl * 100) if total_realized_pnl > 0 else 0

    # Tax-adjusted returns
    tax_adjusted_return = total_realized_pnl - total_tax_liability
    tax_adjusted_return_pct = (tax_adjusted_return / total_cost_basis * 100) if total_cost_basis > 0 else 0

    # Display top-level metrics
    st.subheader("📊 Year-to-Date Tax Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Tax Liability",
            f"${total_tax_liability:,.2f}",
            delta=f"{effective_tax_rate:.2f}% effective rate",
            help="Estimated tax owed on realized gains (federal + state + Medicare)"
        )

    with col2:
        st.metric(
            "Gross Realized P&L",
            f"${total_realized_pnl:,.2f}",
            delta=None,
            help="Total realized profit/loss before taxes"
        )

    with col3:
        st.metric(
            "Tax-Adjusted Return",
            f"${tax_adjusted_return:,.2f}",
            delta=f"{tax_adjusted_return_pct:.2f}%",
            help="Net profit after tax obligations"
        )

    with col4:
        st.metric(
            "Closed Positions",
            len(closed_lots_df),
            delta=None,
            help="Number of tax lots closed this year"
        )

    st.divider()

    # Capital gains breakdown
    st.subheader("📈 Capital Gains Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Short-term Gains",
            f"${total_short_term_gain:,.2f}",
            delta=f"${short_term_tax:,.2f} tax",
            delta_color="inverse" if short_term_tax > 0 else "off",
            help=f"Held ≤365 days | Tax rate: {SHORT_TERM_TAX_RATE*100:.1f}%"
        )

    with col2:
        st.metric(
            "Long-term Gains",
            f"${total_long_term_gain:,.2f}",
            delta=f"${long_term_tax:,.2f} tax",
            delta_color="inverse" if long_term_tax > 0 else "off",
            help=f"Held >365 days | Tax rate: {LONG_TERM_TAX_RATE*100:.1f}%"
        )

    with col3:
        # Calculate potential tax savings if all were long-term
        potential_short_term_as_long_term_tax = max(total_short_term_gain, 0) * LONG_TERM_TAX_RATE
        tax_savings_potential = short_term_tax - potential_short_term_as_long_term_tax
        st.metric(
            "Potential Savings",
            f"${tax_savings_potential:,.2f}",
            delta="if held >365 days",
            help="Tax savings if short-term gains were held long enough for LT treatment"
        )

    st.divider()

    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tax Breakdown",
        "📈 Cumulative Tax Over Time",
        "🎯 Strategy Analysis",
        "📋 Recent Closed Lots"
    ])

    with tab1:
        st.subheader("Tax Liability Composition")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart: Short-term vs Long-term gains
            gain_data = pd.DataFrame({
                'Type': ['Short-term (≤365d)', 'Long-term (>365d)'],
                'Gains': [total_short_term_gain, total_long_term_gain],
                'Tax': [short_term_tax, long_term_tax]
            })

            if total_realized_pnl > 0:
                fig = px.pie(
                    gain_data,
                    names='Type',
                    values='Gains',
                    title='Realized Gains Distribution',
                    color='Type',
                    color_discrete_map={
                        'Short-term (≤365d)': '#FF6B6B',
                        'Long-term (>365d)': '#51CF66'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No gains to display")

        with col2:
            # Bar chart: Tax liability breakdown
            fig = go.Figure(data=[
                go.Bar(
                    name='Short-term Tax',
                    x=['Tax Liability'],
                    y=[short_term_tax],
                    marker_color='#FF6B6B',
                    text=[f"${short_term_tax:,.2f}"],
                    textposition='auto'
                ),
                go.Bar(
                    name='Long-term Tax',
                    x=['Tax Liability'],
                    y=[long_term_tax],
                    marker_color='#51CF66',
                    text=[f"${long_term_tax:,.2f}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title='Tax Liability by Type',
                yaxis_title='Tax Amount ($)',
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Additional metrics table
        st.markdown("### Key Tax Metrics")
        metrics_data = {
            'Metric': [
                'Total Proceeds',
                'Total Cost Basis',
                'Gross Realized Gain',
                'Short-term Tax',
                'Long-term Tax',
                'Total Tax Liability',
                'Net After-Tax Profit',
                'Effective Tax Rate'
            ],
            'Value': [
                f"${total_proceeds:,.2f}",
                f"${total_cost_basis:,.2f}",
                f"${total_realized_pnl:,.2f}",
                f"${short_term_tax:,.2f}",
                f"${long_term_tax:,.2f}",
                f"${total_tax_liability:,.2f}",
                f"${tax_adjusted_return:,.2f}",
                f"{effective_tax_rate:.2f}%"
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Cumulative Tax Liability Over Time")

        # Create time series of cumulative tax
        closed_lots_df['close_date_only'] = pd.to_datetime(closed_lots_df['close_date']).dt.date
        closed_lots_df = closed_lots_df.sort_values('close_date')

        # Calculate cumulative tax
        closed_lots_df['short_term_tax'] = closed_lots_df['short_term_gain'].apply(lambda x: max(x, 0) * SHORT_TERM_TAX_RATE)
        closed_lots_df['long_term_tax'] = closed_lots_df['long_term_gain'].apply(lambda x: max(x, 0) * LONG_TERM_TAX_RATE)
        closed_lots_df['tax_liability'] = closed_lots_df['short_term_tax'] + closed_lots_df['long_term_tax']
        closed_lots_df['cumulative_tax'] = closed_lots_df['tax_liability'].cumsum()
        closed_lots_df['cumulative_pnl'] = closed_lots_df['realized_pnl'].cumsum()

        # Line chart: Cumulative tax over time
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=closed_lots_df['close_date'],
            y=closed_lots_df['cumulative_tax'],
            mode='lines+markers',
            name='Cumulative Tax Liability',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title='Cumulative Tax Liability Over Time',
            xaxis_title='Date',
            yaxis_title='Tax Liability ($)',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Dual axis: Tax vs P&L
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                closed_lots_df,
                x='close_date',
                y='cumulative_pnl',
                title='Cumulative Realized P&L',
                markers=True
            )
            fig.update_layout(yaxis_title='P&L ($)', xaxis_title='Date')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tax as percentage of P&L over time
            closed_lots_df['tax_rate_pct'] = (closed_lots_df['cumulative_tax'] / closed_lots_df['cumulative_pnl'] * 100).fillna(0)
            fig = px.line(
                closed_lots_df,
                x='close_date',
                y='tax_rate_pct',
                title='Effective Tax Rate Over Time',
                markers=True
            )
            fig.update_layout(yaxis_title='Tax Rate (%)', xaxis_title='Date')
            fig.add_hline(y=SHORT_TERM_TAX_RATE*100, line_dash="dash", line_color="red",
                         annotation_text=f"Short-term rate ({SHORT_TERM_TAX_RATE*100:.1f}%)")
            fig.add_hline(y=LONG_TERM_TAX_RATE*100, line_dash="dash", line_color="green",
                         annotation_text=f"Long-term rate ({LONG_TERM_TAX_RATE*100:.1f}%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Tax Efficiency by Strategy")

        # Group by strategy
        strategy_analysis = closed_lots_df.groupby('strategy').agg({
            'realized_pnl': 'sum',
            'short_term_gain': 'sum',
            'long_term_gain': 'sum',
            'tax_liability': 'sum',
            'id': 'count'
        }).reset_index()

        strategy_analysis.columns = ['Strategy', 'Total P&L', 'Short-term Gains', 'Long-term Gains', 'Tax Liability', 'Trade Count']

        # Calculate effective rate per strategy
        strategy_analysis['Effective Tax Rate (%)'] = (
            strategy_analysis['Tax Liability'] / strategy_analysis['Total P&L'] * 100
        ).fillna(0)

        strategy_analysis['After-Tax P&L'] = strategy_analysis['Total P&L'] - strategy_analysis['Tax Liability']

        # Format for display
        display_strategy = strategy_analysis.copy()
        display_strategy['Total P&L'] = display_strategy['Total P&L'].apply(lambda x: f"${x:,.2f}")
        display_strategy['Short-term Gains'] = display_strategy['Short-term Gains'].apply(lambda x: f"${x:,.2f}")
        display_strategy['Long-term Gains'] = display_strategy['Long-term Gains'].apply(lambda x: f"${x:,.2f}")
        display_strategy['Tax Liability'] = display_strategy['Tax Liability'].apply(lambda x: f"${x:,.2f}")
        display_strategy['After-Tax P&L'] = display_strategy['After-Tax P&L'].apply(lambda x: f"${x:,.2f}")
        display_strategy['Effective Tax Rate (%)'] = display_strategy['Effective Tax Rate (%)'].apply(lambda x: f"{x:.2f}%")

        st.dataframe(display_strategy, use_container_width=True, hide_index=True)

        # Bar chart: Tax liability by strategy
        fig = px.bar(
            strategy_analysis,
            x='Strategy',
            y='Tax Liability',
            title='Tax Liability by Strategy',
            color='Effective Tax Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Trade Count',
            hover_data=['Total P&L', 'After-Tax P&L']
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Most tax-efficient strategies
        st.markdown("### Tax Efficiency Rankings")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Most Efficient (Lowest Tax Rate)")
            efficient = strategy_analysis.nsmallest(5, 'Effective Tax Rate (%)')[['Strategy', 'Effective Tax Rate (%)', 'After-Tax P&L']]
            st.dataframe(efficient, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Least Efficient (Highest Tax Rate)")
            inefficient = strategy_analysis.nlargest(5, 'Effective Tax Rate (%)')[['Strategy', 'Effective Tax Rate (%)', 'Tax Liability']]
            st.dataframe(inefficient, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("📋 Top 20 Most Recent Closed Lots")

        # Create display name for better readability
        def create_display_name(row):
            if row['asset_type'] == 'OPTION':
                option_label = f"{row['underlying_symbol']} {row['option_type']} ${row['strike_price']}"
                if pd.notna(row['expiration_date']):
                    exp_str = pd.to_datetime(row['expiration_date']).strftime('%m/%d/%y')
                    return f"{option_label} exp {exp_str}"
                return option_label
            else:
                return row['symbol']

        closed_lots_df['display_name'] = closed_lots_df.apply(create_display_name, axis=1)

        # Top 20 most recent
        recent_lots = closed_lots_df.head(20).copy()

        # Format for display
        display_recent = recent_lots[[
            'display_name', 'asset_type', 'quantity', 'entry_price', 'close_price',
            'entry_date', 'close_date', 'holding_period_days', 'realized_pnl',
            'is_long_term', 'tax_liability', 'strategy'
        ]].copy()

        display_recent['entry_price'] = display_recent['entry_price'].apply(lambda x: f"${x:.2f}")
        display_recent['close_price'] = display_recent['close_price'].apply(lambda x: f"${x:.2f}")
        display_recent['realized_pnl'] = display_recent['realized_pnl'].apply(lambda x: f"${x:,.2f}")
        display_recent['tax_liability'] = display_recent['tax_liability'].apply(lambda x: f"${x:,.2f}")
        display_recent['tax_treatment'] = display_recent['is_long_term'].apply(
            lambda x: '🟢 Long-term' if x else '🔴 Short-term'
        )

        st.dataframe(
            display_recent[[
                'display_name', 'asset_type', 'quantity', 'entry_price', 'close_price',
                'entry_date', 'close_date', 'holding_period_days', 'tax_treatment',
                'realized_pnl', 'tax_liability', 'strategy'
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "display_name": "Position",
                "asset_type": "Type",
                "quantity": st.column_config.NumberColumn("Qty", format="%.2f"),
                "entry_price": "Entry $",
                "close_price": "Close $",
                "entry_date": st.column_config.DatetimeColumn("Opened", format="MM/DD/YY"),
                "close_date": st.column_config.DatetimeColumn("Closed", format="MM/DD/YY"),
                "holding_period_days": "Days Held",
                "tax_treatment": "Tax Treatment",
                "realized_pnl": "Realized P&L",
                "tax_liability": "Tax",
                "strategy": "Strategy"
            }
        )

    # Wash sale warnings section
    st.divider()
    st.subheader("⚠️ Wash Sale Analysis")

    # Check for potential wash sales (same symbol bought/sold within 30 days)
    wash_sale_candidates = []
    for idx, lot in closed_lots_df.iterrows():
        if lot['realized_pnl'] < 0:  # Only losses can trigger wash sales
            symbol = lot['symbol']
            close_date = pd.to_datetime(lot['close_date'])

            # Check if same symbol was purchased within 30 days before or after
            window_start = close_date - timedelta(days=30)
            window_end = close_date + timedelta(days=30)

            # Check in closed lots
            matching_purchases = closed_lots_df[
                (closed_lots_df['symbol'] == symbol) &
                (closed_lots_df['id'] != lot['id']) &
                (pd.to_datetime(closed_lots_df['entry_date']) >= window_start) &
                (pd.to_datetime(closed_lots_df['entry_date']) <= window_end)
            ]

            # Check in open lots
            matching_open = open_lots_df[
                (open_lots_df['symbol'] == symbol) &
                (pd.to_datetime(open_lots_df['entry_date']) >= window_start) &
                (pd.to_datetime(open_lots_df['entry_date']) <= window_end)
            ]

            if len(matching_purchases) > 0 or len(matching_open) > 0:
                wash_sale_candidates.append({
                    'Symbol': symbol,
                    'Close Date': close_date.strftime('%Y-%m-%d'),
                    'Loss': f"${lot['realized_pnl']:,.2f}",
                    'Potential Wash Sale': 'Yes',
                    'Reason': f"Purchased within 30-day window"
                })

    if wash_sale_candidates:
        st.warning(f"⚠️ Found {len(wash_sale_candidates)} potential wash sale(s)")
        st.markdown("""
        **Wash Sale Rule**: If you sell a security at a loss and buy the same or substantially identical
        security within 30 days before or after the sale, the loss may be disallowed for tax purposes.
        """)
        st.dataframe(pd.DataFrame(wash_sale_candidates), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No potential wash sales detected")

    # Additional info
    st.divider()
    st.markdown("""
    ### 📚 Tax Information

    **Tax Rates Used:**
    - **Short-term Capital Gains** (≤365 days): 37.1% (Federal 37% + CA + Medicare 3.8%)
    - **Long-term Capital Gains** (>365 days): 28.1% (Federal 20% + CA 13.3% + NIIT 3.8% - varies by bracket)
    - **Base Income**: $300,000 (affects tax bracket calculations)

    **Important Notes:**
    - This is an estimate. Actual tax liability may vary based on your total income, deductions, and tax situation.
    - Consult with a tax professional for accurate tax planning.
    - Wash sale rules may disallow losses if you repurchase the same security within 30 days.
    - Tax-loss harvesting can offset up to $3,000 of ordinary income per year.

    **Strategy Tips:**
    - Hold positions >365 days for preferential long-term capital gains rates
    - Harvest tax losses before year-end to offset gains
    - Use tax-optimized lot selection (HIFO for losses, LIFO for preservation)
    - Consider deferring gains into next tax year if near year-end
    """)
