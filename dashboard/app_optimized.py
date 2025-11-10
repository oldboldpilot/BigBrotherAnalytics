"""
BigBrother Analytics - Trading Dashboard (OPTIMIZED)
Real-time monitoring of trades, performance, and employment signals

Performance Optimizations:
1. Streamlit caching with appropriate TTL values
2. Lazy loading of chart data
3. Pagination for large datasets
4. Reduced refresh rate (5-10 seconds instead of 1 second)
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="BigBrother Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection
DB_PATH = Path(__file__).parent.parent / "data" / "bigbrother.duckdb"

@st.cache_resource
def get_db_connection():
    """Get cached DuckDB connection"""
    return duckdb.connect(str(DB_PATH), read_only=True)

# Cached data loading functions with appropriate TTL

@st.cache_data(ttl=300)  # Cache for 5 minutes (static data)
def load_sectors_cached():
    """Load sector information (cached - rarely changes)"""
    conn = get_db_connection()
    query = """
        SELECT
            sector_code,
            sector_name,
            sector_etf,
            category,
            description
        FROM sectors
        ORDER BY sector_code
    """
    return conn.execute(query).df()

@st.cache_data(ttl=10)  # Cache for 10 seconds (frequently updated)
def load_positions_cached():
    """Load active positions from database (cached)"""
    conn = get_db_connection()
    query = """
        SELECT
            id,
            symbol,
            quantity,
            avg_cost,
            current_price,
            market_value,
            unrealized_pnl,
            is_bot_managed,
            managed_by,
            bot_strategy,
            opened_at,
            opened_by,
            updated_at
        FROM positions
        ORDER BY unrealized_pnl DESC
    """
    return conn.execute(query).df()

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_positions_history_cached(limit=100):
    """Load positions history from database (cached and paginated)"""
    conn = get_db_connection()
    query = f"""
        SELECT
            timestamp,
            symbol,
            quantity,
            average_price,
            current_price,
            unrealized_pnl,
            is_bot_managed,
            strategy
        FROM positions_history
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    return conn.execute(query).df()

@st.cache_data(ttl=300)  # Cache for 5 minutes (employment data changes daily)
def load_sector_employment_cached():
    """Load sector employment data with growth calculations (cached)"""
    conn = get_db_connection()
    query = """
        WITH latest_data AS (
            SELECT
                se.sector_code,
                s.sector_name,
                s.sector_etf,
                s.category,
                se.report_date,
                se.employment_count,
                ROW_NUMBER() OVER (PARTITION BY se.sector_code ORDER BY se.report_date DESC) as rn
            FROM sector_employment se
            JOIN sectors s ON se.sector_code = s.sector_code
            WHERE se.employment_count IS NOT NULL
        ),
        current_data AS (
            SELECT * FROM latest_data WHERE rn = 1
        ),
        previous_data AS (
            SELECT * FROM latest_data WHERE rn = 4
        )
        SELECT
            c.sector_code,
            c.sector_name,
            c.sector_etf,
            c.category,
            c.report_date as latest_date,
            c.employment_count as current_employment,
            p.employment_count as previous_employment,
            CASE
                WHEN p.employment_count IS NOT NULL AND p.employment_count > 0
                THEN ((c.employment_count - p.employment_count) * 100.0 / p.employment_count)
                ELSE 0
            END as growth_rate_3m
        FROM current_data c
        LEFT JOIN previous_data p ON c.sector_code = p.sector_code
        ORDER BY growth_rate_3m DESC
    """
    return conn.execute(query).df()

# Original loading functions (fallback)
def load_positions():
    """Load active positions from database"""
    return load_positions_cached()

def load_positions_history():
    """Load positions history from database"""
    return load_positions_history_cached(100)

def load_sectors():
    """Load sector information"""
    return load_sectors_cached()

def load_sector_employment():
    """Load sector employment data with growth calculations"""
    return load_sector_employment_cached()

def calculate_signal(growth_rate):
    """Calculate investment signal based on growth rate"""
    if growth_rate >= 2.0:
        return "OVERWEIGHT", "green"
    elif growth_rate >= 0:
        return "MARKET WEIGHT", "yellow"
    else:
        return "AVOID", "red"

def format_pnl(value):
    """Format P&L value with color"""
    if pd.isna(value):
        return "N/A"
    if value > 0:
        return f"<span style='color: green;'>${value:,.2f}</span>"
    elif value < 0:
        return f"<span style='color: red;'>${value:,.2f}</span>"
    else:
        return f"${value:,.2f}"

def format_price(value):
    """Format price value"""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"

# Main Dashboard
def main():
    st.title("📊 BigBrother Trading Dashboard (Optimized)")
    st.markdown("Real-time monitoring of trades, performance, and employment signals")

    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")

        # Reduced auto-refresh options
        refresh_interval = st.selectbox(
            "Auto Refresh Interval",
            options=[0, 5, 10, 30, 60],
            format_func=lambda x: "Disabled" if x == 0 else f"{x} seconds",
            index=0  # Default: disabled
        )

        if refresh_interval > 0:
            st.info(f"Dashboard will refresh every {refresh_interval} seconds")

        st.divider()
        st.markdown("### Navigation")
        view = st.radio(
            "Select View",
            ["Overview", "Positions", "P&L Analysis", "Employment Signals", "Trade History"]
        )

        st.divider()
        st.markdown("### Database Status")
        if DB_PATH.exists():
            st.success("✅ Connected to DuckDB")
            st.caption(f"Path: {DB_PATH}")
        else:
            st.error("❌ Database not found")

        st.divider()
        st.markdown("### Performance Info")
        st.caption("⚡ Cached queries (TTL: 10s-5m)")
        st.caption("📊 Lazy loading enabled")

    # Main content area
    if view == "Overview":
        show_overview()
    elif view == "Positions":
        show_positions()
    elif view == "P&L Analysis":
        show_pnl_analysis()
    elif view == "Employment Signals":
        show_employment_signals()
    elif view == "Trade History":
        show_trade_history()

    # Auto-refresh
    if refresh_interval > 0:
        import time
        time.sleep(refresh_interval)
        st.rerun()

def show_overview():
    """Display overview dashboard"""
    st.header("📈 System Overview")

    # Load data (cached)
    positions_df = load_positions()
    sectors_df = load_sectors()
    employment_df = load_sector_employment()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Active Positions",
            len(positions_df),
            delta=None
        )

    with col2:
        total_pnl = positions_df['unrealized_pnl'].sum() if not positions_df.empty else 0
        st.metric(
            "Total Unrealized P&L",
            f"${total_pnl:,.2f}",
            delta=None,
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )

    with col3:
        bot_managed = len(positions_df[positions_df['is_bot_managed'] == True]) if not positions_df.empty else 0
        st.metric(
            "Bot-Managed Positions",
            bot_managed
        )

    with col4:
        st.metric(
            "Tracked Sectors",
            len(sectors_df)
        )

    st.divider()

    # Quick view of positions and signals
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 Positions by P&L")
        if not positions_df.empty:
            top_positions = positions_df.nlargest(5, 'unrealized_pnl')[['symbol', 'quantity', 'unrealized_pnl']]
            st.dataframe(top_positions, use_container_width=True, hide_index=True)
        else:
            st.info("No active positions")

    with col2:
        st.subheader("Top 5 Employment Growth Sectors")
        if not employment_df.empty:
            top_sectors = employment_df.nlargest(5, 'growth_rate_3m')[['sector_name', 'sector_etf', 'growth_rate_3m']]
            top_sectors['growth_rate_3m'] = top_sectors['growth_rate_3m'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(top_sectors, use_container_width=True, hide_index=True)
        else:
            st.info("No employment data available")

def show_positions():
    """Display Component 1: Real-Time Positions View"""
    st.header("📍 Real-Time Positions")

    positions_df = load_positions()

    if positions_df.empty:
        st.warning("No active positions found in the database.")
        st.info("Positions will appear here once trades are executed.")
        return

    # Add color coding column
    positions_df['Status'] = positions_df['unrealized_pnl'].apply(
        lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Break Even')
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_positions = len(positions_df)
        st.metric("Total Positions", total_positions)

    with col2:
        profitable = len(positions_df[positions_df['unrealized_pnl'] > 0])
        st.metric("Profitable", profitable, delta=f"{profitable/total_positions*100:.1f}%")

    with col3:
        losing = len(positions_df[positions_df['unrealized_pnl'] < 0])
        st.metric("Losing", losing, delta=f"-{losing/total_positions*100:.1f}%", delta_color="inverse")

    with col4:
        total_value = positions_df['market_value'].sum()
        st.metric("Total Market Value", f"${total_value:,.2f}")

    st.divider()

    # Display positions table
    st.subheader("Position Details")

    # Format the dataframe for display
    display_df = positions_df.copy()
    display_df['avg_cost'] = display_df['avg_cost'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    display_df['market_value'] = display_df['market_value'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    display_df['unrealized_pnl'] = display_df['unrealized_pnl'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

    # Select columns to display
    cols_to_display = ['symbol', 'quantity', 'avg_cost', 'current_price', 'market_value',
                       'unrealized_pnl', 'is_bot_managed', 'bot_strategy', 'opened_at']

    st.dataframe(
        display_df[cols_to_display],
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": "Symbol",
            "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
            "avg_cost": "Entry Price",
            "current_price": "Current Price",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L",
            "is_bot_managed": "Bot Managed",
            "bot_strategy": "Strategy",
            "opened_at": st.column_config.DatetimeColumn("Opened At", format="MM/DD/YY HH:mm")
        }
    )

    # Position breakdown chart (lazy loaded)
    with st.expander("📊 Position Breakdown", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # P&L distribution
            fig = px.pie(
                positions_df,
                names='Status',
                title='P&L Distribution',
                color='Status',
                color_discrete_map={'Profit': 'green', 'Loss': 'red', 'Break Even': 'gray'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top positions by market value
            top_10 = positions_df.nlargest(10, 'market_value')
            fig = px.bar(
                top_10,
                x='symbol',
                y='market_value',
                title='Top 10 Positions by Market Value',
                color='unrealized_pnl',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig, use_container_width=True)

def show_pnl_analysis():
    """Display Component 2: P&L Charts"""
    st.header("📊 P&L Analysis")

    positions_df = load_positions()
    history_df = load_positions_history()

    if positions_df.empty and history_df.empty:
        st.warning("No position data available for P&L analysis.")
        st.info("P&L charts will appear once trading activity is recorded.")
        return

    # Current P&L Summary
    st.subheader("Current P&L Summary")

    col1, col2, col3, col4 = st.columns(4)

    total_pnl = positions_df['unrealized_pnl'].sum() if not positions_df.empty else 0
    total_value = positions_df['market_value'].sum() if not positions_df.empty else 0
    avg_pnl = positions_df['unrealized_pnl'].mean() if not positions_df.empty else 0

    with col1:
        st.metric("Total Unrealized P&L", f"${total_pnl:,.2f}")

    with col2:
        st.metric("Total Market Value", f"${total_value:,.2f}")

    with col3:
        st.metric("Average P&L per Position", f"${avg_pnl:,.2f}")

    with col4:
        pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0
        st.metric("P&L %", f"{pnl_pct:.2f}%")

    st.divider()

    # Charts (lazy loaded)
    with st.expander("📈 P&L Charts", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Win/Loss Ratio Pie Chart
            st.subheader("Win/Loss Ratio")

            if not positions_df.empty:
                winners = len(positions_df[positions_df['unrealized_pnl'] > 0])
                losers = len(positions_df[positions_df['unrealized_pnl'] < 0])
                breakeven = len(positions_df[positions_df['unrealized_pnl'] == 0])

                fig = go.Figure(data=[go.Pie(
                    labels=['Winners', 'Losers', 'Break Even'],
                    values=[winners, losers, breakeven],
                    marker=dict(colors=['green', 'red', 'gray']),
                    hole=0.3
                )])
                fig.update_layout(title="Position Performance Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No position data available")

        with col2:
            # P&L by Symbol
            st.subheader("P&L by Symbol")

            if not positions_df.empty:
                pnl_by_symbol = positions_df.nlargest(10, 'unrealized_pnl')[['symbol', 'unrealized_pnl']]

                fig = px.bar(
                    pnl_by_symbol,
                    x='symbol',
                    y='unrealized_pnl',
                    title='Top 10 Positions by P&L',
                    color='unrealized_pnl',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig.update_layout(xaxis_title="Symbol", yaxis_title="Unrealized P&L ($)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No position data available")

    # Historical P&L (lazy loaded)
    if not history_df.empty:
        with st.expander("📉 Historical P&L Trend", expanded=False):
            # Aggregate P&L by date
            history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
            daily_pnl = history_df.groupby('date')['unrealized_pnl'].sum().reset_index()
            daily_pnl['cumulative_pnl'] = daily_pnl['unrealized_pnl'].cumsum()

            col1, col2 = st.columns(2)

            with col1:
                # Daily P&L
                fig = px.line(
                    daily_pnl,
                    x='date',
                    y='unrealized_pnl',
                    title='Daily P&L (Last 30 Days)',
                    markers=True
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Daily P&L ($)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Cumulative P&L
                fig = px.line(
                    daily_pnl,
                    x='date',
                    y='cumulative_pnl',
                    title='Cumulative P&L',
                    markers=True
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative P&L ($)")
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

def show_employment_signals():
    """Display Component 3: Employment Signals"""
    st.header("💼 Employment Signals - Sector Rotation Strategy")

    employment_df = load_sector_employment()

    if employment_df.empty:
        st.warning("No employment data available.")
        st.info("Employment signals will appear once data is populated from BLS API.")
        return

    st.markdown("""
    This view displays employment growth rates for each GICS sector over the last 3 months.
    The signal classification helps identify which sectors to overweight, maintain market weight, or avoid.
    """)

    # Add signal classification
    employment_df['Signal'], employment_df['Color'] = zip(*employment_df['growth_rate_3m'].apply(calculate_signal))

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        overweight = len(employment_df[employment_df['Signal'] == 'OVERWEIGHT'])
        st.metric("OVERWEIGHT Sectors", overweight, delta=None)

    with col2:
        market_weight = len(employment_df[employment_df['Signal'] == 'MARKET WEIGHT'])
        st.metric("MARKET WEIGHT Sectors", market_weight, delta=None)

    with col3:
        avoid = len(employment_df[employment_df['Signal'] == 'AVOID'])
        st.metric("AVOID Sectors", avoid, delta=None)

    with col4:
        avg_growth = employment_df['growth_rate_3m'].mean()
        st.metric("Avg Growth Rate", f"{avg_growth:.2f}%")

    st.divider()

    # Sector signals table
    st.subheader("Sector Employment Signals")

    # Format for display
    display_df = employment_df.copy()
    display_df['growth_rate_3m'] = display_df['growth_rate_3m'].apply(lambda x: f"{x:.2f}%")
    display_df['current_employment'] = display_df['current_employment'].apply(lambda x: f"{x:,}")

    # Sort by signal priority
    signal_order = {'OVERWEIGHT': 0, 'MARKET WEIGHT': 1, 'AVOID': 2}
    display_df['signal_order'] = display_df['Signal'].map(signal_order)
    display_df = display_df.sort_values('signal_order')

    st.dataframe(
        display_df[['sector_name', 'sector_etf', 'category', 'current_employment',
                    'growth_rate_3m', 'Signal', 'latest_date']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "sector_name": "Sector",
            "sector_etf": "ETF Symbol",
            "category": "Category",
            "current_employment": "Employment (000s)",
            "growth_rate_3m": "3-Month Growth",
            "Signal": st.column_config.TextColumn("Signal", width="medium"),
            "latest_date": st.column_config.DateColumn("Latest Data", format="YYYY-MM-DD")
        }
    )

    # Visualizations (lazy loaded)
    with st.expander("📊 Employment Growth Visualization", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Growth rate bar chart
            fig = px.bar(
                employment_df.sort_values('growth_rate_3m', ascending=False),
                x='sector_etf',
                y='growth_rate_3m',
                color='Signal',
                title='3-Month Employment Growth by Sector',
                color_discrete_map={'OVERWEIGHT': 'green', 'MARKET WEIGHT': 'yellow', 'AVOID': 'red'},
                hover_data=['sector_name', 'category']
            )
            fig.update_layout(xaxis_title="Sector ETF", yaxis_title="Growth Rate (%)")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Signal distribution pie chart
            signal_counts = employment_df['Signal'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                marker=dict(colors=['green', 'yellow', 'red'])
            )])
            fig.update_layout(title="Signal Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Category breakdown (lazy loaded)
    with st.expander("📈 Growth by Sector Category", expanded=False):
        category_growth = employment_df.groupby('category')['growth_rate_3m'].mean().reset_index()
        category_growth = category_growth.sort_values('growth_rate_3m', ascending=False)

        fig = px.bar(
            category_growth,
            x='category',
            y='growth_rate_3m',
            title='Average Growth Rate by Sector Category',
            color='growth_rate_3m',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Avg Growth Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

def show_trade_history():
    """Display Component 4: Trade History"""
    st.header("📜 Trade History")

    # Pagination support
    page_size = st.sidebar.slider("Records per page", 50, 500, 100, 50)
    history_df = load_positions_history_cached(page_size)

    if history_df.empty:
        st.warning("No trade history available.")
        st.info("Trade history will appear once positions are tracked over time.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_trades = len(history_df)
        st.metric("Total Records", total_trades)

    with col2:
        unique_symbols = history_df['symbol'].nunique()
        st.metric("Unique Symbols", unique_symbols)

    with col3:
        bot_trades = len(history_df[history_df['is_bot_managed'] == True])
        st.metric("Bot-Managed Records", bot_trades)

    with col4:
        avg_pnl = history_df['unrealized_pnl'].mean()
        st.metric("Avg Unrealized P&L", f"${avg_pnl:,.2f}")

    st.divider()

    # Filters
    st.subheader("Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbols = ['All'] + sorted(history_df['symbol'].unique().tolist())
        selected_symbol = st.selectbox("Symbol", symbols)

    with col2:
        strategies = ['All'] + sorted(history_df['strategy'].dropna().unique().tolist())
        selected_strategy = st.selectbox("Strategy", strategies)

    with col3:
        bot_filter = st.selectbox("Bot Managed", ['All', 'Yes', 'No'])

    # Apply filters
    filtered_df = history_df.copy()

    if selected_symbol != 'All':
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]

    if selected_strategy != 'All':
        filtered_df = filtered_df[filtered_df['strategy'] == selected_strategy]

    if bot_filter == 'Yes':
        filtered_df = filtered_df[filtered_df['is_bot_managed'] == True]
    elif bot_filter == 'No':
        filtered_df = filtered_df[filtered_df['is_bot_managed'] == False]

    # Display trade history table
    st.subheader(f"Recent Trades ({len(filtered_df)} records)")

    # Format for display
    display_df = filtered_df.copy()
    display_df['average_price'] = display_df['average_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    display_df['unrealized_pnl'] = display_df['unrealized_pnl'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

    st.dataframe(
        display_df[['timestamp', 'symbol', 'quantity', 'average_price', 'current_price',
                    'unrealized_pnl', 'is_bot_managed', 'strategy']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="MM/DD/YY HH:mm:ss"),
            "symbol": "Symbol",
            "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
            "average_price": "Avg Price",
            "current_price": "Current Price",
            "unrealized_pnl": "Unrealized P&L",
            "is_bot_managed": "Bot Managed",
            "strategy": "Strategy"
        }
    )

    # Trading activity over time (lazy loaded)
    if not filtered_df.empty:
        with st.expander("📊 Trading Activity Visualization", expanded=False):
            # Group by date
            filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
            daily_activity = filtered_df.groupby('date').size().reset_index(name='trade_count')

            fig = px.line(
                daily_activity,
                x='date',
                y='trade_count',
                title='Daily Trading Activity',
                markers=True
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Number of Records")
            st.plotly_chart(fig, use_container_width=True)

            # Symbol and strategy distribution
            col1, col2 = st.columns(2)

            with col1:
                symbol_counts = filtered_df['symbol'].value_counts().head(10)
                fig = px.bar(
                    x=symbol_counts.index,
                    y=symbol_counts.values,
                    title='Top 10 Most Traded Symbols',
                    labels={'x': 'Symbol', 'y': 'Trade Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                strategy_counts = filtered_df['strategy'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=strategy_counts.index,
                    values=strategy_counts.values
                )])
                fig.update_layout(title="Strategy Distribution")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
