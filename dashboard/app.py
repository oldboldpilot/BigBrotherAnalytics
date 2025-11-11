"""
BigBrother Analytics - Trading Dashboard
Real-time monitoring of trades, performance, and employment signals
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add scripts and dashboard directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

# Import tax implications view
from tax_implications_view import show_tax_implications

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

def load_positions():
    """Load active positions from database"""
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

def load_positions_history():
    """Load positions history from database"""
    conn = get_db_connection()
    query = """
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
        LIMIT 100
    """
    return conn.execute(query).df()

def load_sectors():
    """Load sector information"""
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

def load_sector_employment():
    """Load sector employment data with growth calculations"""
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
            SELECT * FROM latest_data WHERE rn = 4  -- 3 months ago (quarterly data)
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
    st.title("📊 BigBrother Trading Dashboard")
    st.markdown("Real-time monitoring of trades, performance, and employment signals")

    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        if auto_refresh:
            st.info("Dashboard will refresh every 30 seconds")

        st.divider()
        st.markdown("### Navigation")
        view = st.radio(
            "Select View",
            ["Overview", "Positions", "P&L Analysis", "Employment Signals", "Trade History", "News Feed", "Alerts", "System Health", "Tax Implications"]
        )

        st.divider()
        st.markdown("### Database Status")
        if DB_PATH.exists():
            st.success("✅ Connected to DuckDB")
            st.caption(f"Path: {DB_PATH}")
        else:
            st.error("❌ Database not found")

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
    elif view == "News Feed":
        show_news_feed()
    elif view == "Alerts":
        show_alerts()
    elif view == "System Health":
        show_system_health()
    elif view == "Tax Implications":
        show_tax_implications(get_db_connection())

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

def show_overview():
    """Display overview dashboard"""
    st.header("📈 System Overview")

    # Load data
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

    # Position breakdown chart
    st.subheader("Position Breakdown")

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

    # Charts
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

    # Historical P&L (if available)
    if not history_df.empty:
        st.subheader("Historical P&L Trend")

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

    # Visualizations
    st.subheader("Employment Growth Visualization")

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

    # Category breakdown
    st.subheader("Growth by Sector Category")

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

    history_df = load_positions_history()

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

    # Trading activity over time
    st.subheader("Trading Activity Over Time")

    if not filtered_df.empty:
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

        # Symbol distribution
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

def show_news_feed():
    """Display Component: News Feed with Sentiment Analysis"""
    st.header("📰 News Feed")

    conn = get_db_connection()

    # Check if news_articles table exists
    try:
        table_check = conn.execute("""
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_name = 'news_articles'
        """).fetchone()

        if not table_check or table_check[0] == 0:
            st.warning("📊 News database not initialized")
            st.info("Run `uv run python scripts/monitoring/setup_news_database.py` to create the news table")
            return
    except Exception as e:
        st.error(f"Error checking news table: {e}")
        return

    # Load news articles
    try:
        query = """
            SELECT
                article_id,
                symbol,
                title,
                description,
                url,
                source_name,
                author,
                published_at,
                sentiment_score,
                sentiment_label,
                positive_keywords,
                negative_keywords,
                newsapi_sentiment_score,
                newsapi_sentiment_label,
                alphavantage_sentiment_score,
                alphavantage_sentiment_label,
                alphavantage_relevance,
                sentiment_source,
                selection_reason,
                fetched_at
            FROM news_articles
            ORDER BY published_at DESC
            LIMIT 100
        """
        news_df = conn.execute(query).df()
    except Exception as e:
        st.error(f"Error loading news: {e}")
        st.info("Run `uv run python scripts/data_collection/news_ingestion_multi_source.py` to fetch news with multi-source sentiment")
        return

    if news_df.empty:
        st.info("📭 No news articles yet")
        st.markdown("""
        ### Get Started

        1. **Setup database**: `uv run python scripts/monitoring/setup_news_database.py`
        2. **Fetch news**: `uv run python scripts/data_collection/news_ingestion.py`
        3. **View results**: Refresh this page

        The system will fetch news for your traded symbols with sentiment analysis.
        """)
        return

    # Summary metrics at top
    st.subheader("📊 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Articles", len(news_df))

    with col2:
        unique_symbols = news_df['symbol'].nunique()
        st.metric("Symbols", unique_symbols)

    with col3:
        positive_count = len(news_df[news_df['sentiment_label'] == 'positive'])
        st.metric("Positive", positive_count, delta=f"{100*positive_count/len(news_df):.1f}%")

    with col4:
        negative_count = len(news_df[news_df['sentiment_label'] == 'negative'])
        st.metric("Negative", negative_count, delta=f"{100*negative_count/len(news_df):.1f}%", delta_color="inverse")

    with col5:
        avg_sentiment = news_df['sentiment_score'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:+.3f}")

    st.divider()

    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbols = ['All'] + sorted(news_df['symbol'].unique().tolist())
        selected_symbol = st.selectbox("Symbol", symbols, key='news_symbol')

    with col2:
        sentiments = ['All', 'positive', 'negative', 'neutral']
        selected_sentiment = st.selectbox("Sentiment", sentiments, key='news_sentiment')

    with col3:
        # Sentiment source filter
        if 'sentiment_source' in news_df.columns:
            sources = ['All'] + [s for s in news_df['sentiment_source'].dropna().unique().tolist() if s]
            selected_source = st.selectbox("Sentiment Source", sources, key='news_source')
        else:
            selected_source = 'All'

    with col4:
        limit = st.slider("Articles", 5, 50, 20, key='news_limit')

    # Apply filters
    filtered_df = news_df.copy()

    if selected_symbol != 'All':
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]

    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]

    if selected_source != 'All' and 'sentiment_source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sentiment_source'] == selected_source]

    filtered_df = filtered_df.head(limit)

    st.divider()

    # Sentiment distribution chart
    st.subheader("📈 Sentiment Distribution")

    sentiment_counts = news_df.groupby('sentiment_label').size().reset_index(name='count')

    fig = px.bar(
        sentiment_counts,
        x='sentiment_label',
        y='count',
        color='sentiment_label',
        color_discrete_map={
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        },
        title="Articles by Sentiment"
    )
    fig.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment by symbol
    if news_df['symbol'].nunique() > 1:
        st.subheader("📊 Average Sentiment by Symbol")

        sentiment_by_symbol = news_df.groupby('symbol')['sentiment_score'].agg(['mean', 'count']).reset_index()
        sentiment_by_symbol = sentiment_by_symbol.sort_values('mean', ascending=False)

        fig = px.bar(
            sentiment_by_symbol,
            x='symbol',
            y='mean',
            color='mean',
            color_continuous_scale=['red', 'yellow', 'green'],
            hover_data=['count'],
            title="Average Sentiment Score by Symbol"
        )
        fig.update_layout(xaxis_title="Symbol", yaxis_title="Avg Sentiment Score", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Display news articles
    st.subheader(f"📰 Recent Articles ({len(filtered_df)} shown)")

    for idx, row in filtered_df.iterrows():
        # Sentiment color
        if row['sentiment_label'] == 'positive':
            sentiment_color = 'green'
            sentiment_emoji = '📈'
        elif row['sentiment_label'] == 'negative':
            sentiment_color = 'red'
            sentiment_emoji = '📉'
        else:
            sentiment_color = 'gray'
            sentiment_emoji = '➖'

        # Article card
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### [{row['title']}]({row['url']})")
                st.caption(f"**{row['symbol']}** | {row['source_name']} | {pd.to_datetime(row['published_at']).strftime('%Y-%m-%d %H:%M')}")

            with col2:
                st.markdown(f"<h3 style='text-align: right; color: {sentiment_color};'>{sentiment_emoji} {row['sentiment_label'].upper()}</h3>", unsafe_allow_html=True)
                st.metric("Selected Score", f"{row['sentiment_score']:+.3f}", label_visibility="collapsed")

            if row['description']:
                st.write(row['description'])

            # Multi-source sentiment comparison (power-of-2-choices)
            if pd.notna(row.get('newsapi_sentiment_score')) and pd.notna(row.get('alphavantage_sentiment_score')):
                with st.expander("📊 Sentiment Comparison (Power-of-2-Choices)"):
                    comp_col1, comp_col2, comp_col3 = st.columns(3)

                    with comp_col1:
                        st.metric(
                            "NewsAPI (Keyword)",
                            f"{row['newsapi_sentiment_score']:+.3f}",
                            delta=row['newsapi_sentiment_label']
                        )

                    with comp_col2:
                        st.metric(
                            "AlphaVantage (AI)",
                            f"{row['alphavantage_sentiment_score']:+.3f}",
                            delta=row['alphavantage_sentiment_label']
                        )

                    with comp_col3:
                        source_badge = "🔤" if row.get('sentiment_source') == 'newsapi_keyword' else "🤖"
                        st.metric(
                            f"{source_badge} Selected",
                            f"{row['sentiment_score']:+.3f}",
                            delta=row['sentiment_label']
                        )

                    if pd.notna(row.get('selection_reason')):
                        st.caption(f"💡 Selection reason: {row['selection_reason']}")

                    if pd.notna(row.get('alphavantage_relevance')):
                        st.caption(f"🎯 Relevance to {row['symbol']}: {row['alphavantage_relevance']:.1%}")

            # Keywords
            if row['positive_keywords'] or row['negative_keywords']:
                kw_col1, kw_col2 = st.columns(2)

                with kw_col1:
                    if row['positive_keywords'] and len(row['positive_keywords']) > 0:
                        st.caption(f"✅ Positive: {', '.join(row['positive_keywords'][:5])}")

                with kw_col2:
                    if row['negative_keywords'] and len(row['negative_keywords']) > 0:
                        st.caption(f"❌ Negative: {', '.join(row['negative_keywords'][:5])}")

            if row['author']:
                st.caption(f"By {row['author']}")

            st.divider()

def show_alerts():
    """Display alert history and statistics"""
    st.header("🔔 Alert History")

    conn = get_db_connection()

    # Check if alerts table exists
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]

    if 'alerts' not in table_names:
        st.warning("⚠️ Alerts table not found in database. Run setup_alerts_database.py to create it.")
        st.code("uv run python scripts/monitoring/setup_alerts_database.py")
        return

    # Alert statistics
    st.subheader("Alert Statistics")

    col1, col2, col3, col4 = st.columns(4)

    # Total alerts
    total_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    with col1:
        st.metric("Total Alerts", f"{total_alerts:,}")

    # Unacknowledged alerts
    unack_alerts = conn.execute("""
        SELECT COUNT(*) FROM alerts
        WHERE acknowledged = false
          AND severity IN ('ERROR', 'CRITICAL')
          AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 48 HOURS
    """).fetchone()[0]
    with col2:
        st.metric("Unacknowledged", f"{unack_alerts:,}", delta=None if unack_alerts == 0 else f"-{unack_alerts}")

    # Alerts today
    today_alerts = conn.execute("""
        SELECT COUNT(*) FROM alerts
        WHERE DATE(timestamp) = CURRENT_DATE
    """).fetchone()[0]
    with col3:
        st.metric("Today", f"{today_alerts:,}")

    # Critical alerts (last 7 days)
    critical_alerts = conn.execute("""
        SELECT COUNT(*) FROM alerts
        WHERE severity = 'CRITICAL'
          AND timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
    """).fetchone()[0]
    with col4:
        st.metric("Critical (7d)", f"{critical_alerts:,}")

    st.divider()

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        severity_filter = st.multiselect(
            "Severity",
            options=['INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default=['WARNING', 'ERROR', 'CRITICAL']
        )

    with col2:
        type_filter = st.multiselect(
            "Alert Type",
            options=['trading', 'data', 'system', 'performance'],
            default=['trading', 'data', 'system', 'performance']
        )

    with col3:
        days_back = st.selectbox(
            "Time Period",
            options=[1, 7, 30, 90],
            index=1,
            format_func=lambda x: f"Last {x} days"
        )

    # Load alerts with filters
    severity_list = "', '".join(severity_filter)
    type_list = "', '".join(type_filter)

    query = f"""
        SELECT
            id,
            alert_type,
            alert_subtype,
            severity,
            message,
            context,
            timestamp,
            sent,
            acknowledged,
            acknowledged_by,
            acknowledged_at
        FROM alerts
        WHERE severity IN ('{severity_list}')
          AND alert_type IN ('{type_list}')
          AND timestamp >= CURRENT_DATE - INTERVAL {days_back} DAYS
        ORDER BY timestamp DESC
        LIMIT 200
    """

    df = conn.execute(query).df()

    if df.empty:
        st.info("No alerts found matching the selected filters.")
        return

    # Alert timeline
    st.subheader("Alert Timeline")

    # Group by date and severity
    timeline_df = df.copy()
    timeline_df['date'] = pd.to_datetime(timeline_df['timestamp']).dt.date

    timeline_pivot = timeline_df.groupby(['date', 'severity']).size().reset_index(name='count')

    fig = px.bar(
        timeline_pivot,
        x='date',
        y='count',
        color='severity',
        title='Alerts Over Time',
        color_discrete_map={
            'INFO': '#2196F3',
            'WARNING': '#FF9800',
            'ERROR': '#F44336',
            'CRITICAL': '#D32F2F'
        }
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Alert Count")
    st.plotly_chart(fig, use_container_width=True)

    # Alert type distribution
    col1, col2 = st.columns(2)

    with col1:
        type_counts = df['alert_type'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.3
        )])
        fig.update_layout(title='Alerts by Type')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        severity_counts = df['severity'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            hole=0.3,
            marker=dict(colors=['#2196F3', '#FF9800', '#F44336', '#D32F2F'])
        )])
        fig.update_layout(title='Alerts by Severity')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent alerts table
    st.subheader("Recent Alerts")

    # Format for display
    display_df = df[['timestamp', 'severity', 'alert_type', 'alert_subtype', 'message', 'sent', 'acknowledged']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Add color coding to severity
    def color_severity(val):
        colors = {
            'INFO': 'background-color: #E3F2FD',
            'WARNING': 'background-color: #FFF3E0',
            'ERROR': 'background-color: #FFEBEE',
            'CRITICAL': 'background-color: #FFCDD2'
        }
        return colors.get(val, '')

    # Show with pagination
    page_size = 25
    total_pages = len(display_df) // page_size + (1 if len(display_df) % page_size > 0 else 0)

    page = st.number_input('Page', min_value=1, max_value=total_pages, value=1) - 1
    start_idx = page * page_size
    end_idx = start_idx + page_size

    st.dataframe(
        display_df.iloc[start_idx:end_idx].style.applymap(
            color_severity,
            subset=['severity']
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": "Time",
            "severity": "Severity",
            "alert_type": "Type",
            "alert_subtype": "Subtype",
            "message": "Message",
            "sent": st.column_config.CheckboxColumn("Sent"),
            "acknowledged": st.column_config.CheckboxColumn("Ack'd")
        }
    )

    st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(display_df))} of {len(display_df)} alerts")

    # Alert details expander
    st.divider()
    with st.expander("📋 View Alert Details"):
        alert_id = st.number_input("Alert ID", min_value=1, max_value=int(df['id'].max()) if not df.empty else 1, value=1)

        alert_detail = conn.execute(f"""
            SELECT * FROM alerts WHERE id = {alert_id}
        """).fetchone()

        if alert_detail:
            st.json({
                'id': alert_detail[0],
                'type': alert_detail[1],
                'subtype': alert_detail[2],
                'severity': alert_detail[3],
                'message': alert_detail[4],
                'context': alert_detail[5],
                'timestamp': str(alert_detail[6]),
                'source': alert_detail[7],
                'sent': alert_detail[8],
                'acknowledged': alert_detail[10]
            })
        else:
            st.warning(f"Alert ID {alert_id} not found")

def show_system_health():
    """Display Component 5: System Health"""
    st.header("🏥 System Health Monitor")

    st.markdown("""
    Real-time monitoring of all system components including API connectivity,
    database health, data freshness, and system resources.
    """)

    # Add refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Run health check
    try:
        from monitoring.health_check import check_system_health, STATUS_HEALTHY, STATUS_DEGRADED, STATUS_DOWN, STATUS_STALE, STATUS_WARNING, STATUS_NO_DATA, STATUS_RUNNING, STATUS_NOT_RUNNING, STATUS_LOW, STATUS_HIGH

        health = check_system_health()

        # Status emoji mapping
        status_emoji = {
            STATUS_HEALTHY: "✅",
            STATUS_DEGRADED: "⚠️",
            STATUS_DOWN: "❌",
            STATUS_STALE: "🕒",
            STATUS_WARNING: "⚠️",
            STATUS_NO_DATA: "❓",
            STATUS_RUNNING: "✅",
            STATUS_NOT_RUNNING: "⭕",
            STATUS_LOW: "⚠️",
            STATUS_HIGH: "⚠️"
        }

        # Overall status
        st.subheader("Overall System Status")
        overall_status = health['overall_status']

        # Color coding
        status_color = {
            STATUS_HEALTHY: "green",
            STATUS_DEGRADED: "orange",
            STATUS_DOWN: "red",
            STATUS_WARNING: "orange"
        }

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "System Status",
                f"{status_emoji.get(overall_status, '❓')} {overall_status}",
                delta=None
            )

        with col2:
            st.metric(
                "Last Check",
                datetime.fromisoformat(health['timestamp']).strftime("%H:%M:%S")
            )

        with col3:
            # Count healthy vs unhealthy components
            components = health['components']
            healthy_count = sum(1 for c in components.values() if c.get('status') == STATUS_HEALTHY)
            total_count = len(components)
            st.metric(
                "Healthy Components",
                f"{healthy_count}/{total_count}"
            )

        st.divider()

        # Component details in tabs
        tab1, tab2, tab3 = st.tabs(["🌐 APIs & Data", "💾 System Resources", "📊 Detailed Status"])

        with tab1:
            st.subheader("APIs and Data Services")

            # Schwab API
            api = components.get('schwab_api', {})
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Schwab API", f"{status_emoji.get(api.get('status'), '❓')} {api.get('status', 'UNKNOWN')}")
            with col2:
                if 'message' in api:
                    st.info(api['message'])
                if 'error' in api:
                    st.error(api['error'])

            st.markdown("---")

            # Database
            db = components.get('database', {})
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Database", f"{status_emoji.get(db.get('status'), '❓')} {db.get('status', 'UNKNOWN')}")
            with col2:
                if 'size_mb' in db:
                    st.info(f"Size: {db['size_mb']} MB | Tables: {db.get('tables', 0)}")
                    if 'record_counts' in db:
                        st.caption("Record Counts: " + ", ".join([f"{k}: {v:,}" for k, v in db['record_counts'].items()]))
                if 'error' in db:
                    st.error(db['error'])

            st.markdown("---")

            # Signal Generation
            signals = components.get('signal_generation', {})
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Signal Generation", f"{status_emoji.get(signals.get('status'), '❓')} {signals.get('status', 'UNKNOWN')}")
            with col2:
                if 'last_signal_time' in signals and signals['last_signal_time']:
                    age_hours = signals.get('age_hours', 0)
                    age_days = signals.get('age_days', 0)
                    st.info(f"Last signal: {age_hours:.1f} hours ago ({age_days:.1f} days)")
                if 'message' in signals:
                    st.warning(signals['message'])

            st.markdown("---")

            # Data Freshness
            st.subheader("Data Freshness")
            data_fresh = components.get('data_freshness', {})

            if 'overall_status' in data_fresh:
                st.metric("Overall Data Status",
                         f"{status_emoji.get(data_fresh['overall_status'], '❓')} {data_fresh['overall_status']}")

            # Display each data type
            for data_type in ['employment_data', 'jobless_claims', 'stock_prices']:
                if data_type in data_fresh:
                    data_info = data_fresh[data_type]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.caption(data_type.replace('_', ' ').title())

                    with col2:
                        st.caption(f"{status_emoji.get(data_info.get('status'), '❓')} {data_info.get('status', 'UNKNOWN')}")

                    with col3:
                        if data_info.get('last_update'):
                            st.caption(f"{data_info.get('age_days', 0)} days old")
                        else:
                            st.caption("No data")

        with tab2:
            st.subheader("System Resources")

            # Disk Space
            disk = components.get('disk_space', {})
            st.markdown("#### Disk Space")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Status", f"{status_emoji.get(disk.get('status'), '❓')} {disk.get('status', 'UNKNOWN')}")
            with col2:
                if 'total_gb' in disk:
                    st.metric("Total", f"{disk['total_gb']} GB")
            with col3:
                if 'used_gb' in disk:
                    st.metric("Used", f"{disk['used_gb']} GB")
            with col4:
                if 'free_gb' in disk:
                    st.metric("Free", f"{disk['free_gb']} GB")

            if 'percent_used' in disk:
                st.progress(disk['percent_used'] / 100, text=f"Disk Usage: {disk['percent_used']}%")

            st.markdown("---")

            # Memory
            mem = components.get('memory', {})
            st.markdown("#### Memory")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Status", f"{status_emoji.get(mem.get('status'), '❓')} {mem.get('status', 'UNKNOWN')}")
            with col2:
                if 'total_gb' in mem:
                    st.metric("Total", f"{mem['total_gb']} GB")
            with col3:
                if 'used_gb' in mem:
                    st.metric("Used", f"{mem['used_gb']} GB")
            with col4:
                if 'available_gb' in mem:
                    st.metric("Available", f"{mem['available_gb']} GB")

            if 'percent_used' in mem:
                st.progress(mem['percent_used'] / 100, text=f"Memory Usage: {mem['percent_used']}%")

            st.markdown("---")

            # CPU
            cpu = components.get('cpu', {})
            st.markdown("#### CPU")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", f"{status_emoji.get(cpu.get('status'), '❓')} {cpu.get('status', 'UNKNOWN')}")
            with col2:
                if 'percent_used' in cpu:
                    st.metric("Usage", f"{cpu['percent_used']}%")
            with col3:
                if 'cpu_count' in cpu:
                    st.metric("CPU Count", cpu['cpu_count'])

            if 'percent_used' in cpu:
                st.progress(cpu['percent_used'] / 100, text=f"CPU Usage: {cpu['percent_used']}%")

            st.markdown("---")

            # Processes
            proc = components.get('process', {})
            st.markdown("#### BigBrother Processes")

            status = proc.get('status', 'UNKNOWN')
            st.metric("Process Status", f"{status_emoji.get(status, '❓')} {status}")

            if 'processes' in proc and proc['processes']:
                st.caption(f"Running processes: {proc['process_count']}")

                # Display process table
                proc_data = []
                for p in proc['processes']:
                    proc_data.append({
                        "PID": p['pid'],
                        "Name": p['name'],
                        "CPU %": f"{p['cpu_percent']}%",
                        "Memory (MB)": f"{p['memory_mb']:.2f}"
                    })

                st.dataframe(pd.DataFrame(proc_data), use_container_width=True, hide_index=True)
            elif 'message' in proc:
                st.info(proc['message'])

        with tab3:
            st.subheader("Detailed Component Status")

            # Display all components in a table
            status_data = []
            for component_name, component_data in components.items():
                status = component_data.get('status', 'UNKNOWN')
                status_data.append({
                    "Component": component_name.replace('_', ' ').title(),
                    "Status": f"{status_emoji.get(status, '❓')} {status}",
                    "Details": component_data.get('message', component_data.get('error', 'OK'))
                })

            st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Raw JSON data (expandable)
            with st.expander("View Raw Health Data (JSON)"):
                st.json(health)

        # Alert indicators
        st.divider()

        # Count issues
        critical_issues = []
        warnings = []

        for component_name, component_data in components.items():
            status = component_data.get('status')
            if status == STATUS_DOWN:
                critical_issues.append(f"{component_name}: DOWN")
            elif status in [STATUS_DEGRADED, STATUS_WARNING, STATUS_STALE]:
                warnings.append(f"{component_name}: {status}")

        if critical_issues:
            st.error(f"🚨 Critical Issues ({len(critical_issues)}): " + ", ".join(critical_issues))

        if warnings:
            st.warning(f"⚠️ Warnings ({len(warnings)}): " + ", ".join(warnings))

        if not critical_issues and not warnings:
            st.success("✅ All systems operational!")

    except ImportError as e:
        st.error("❌ Health check module not found. Please ensure scripts/monitoring/health_check.py exists.")
        st.exception(e)
    except Exception as e:
        st.error("❌ Error running health check")
        st.exception(e)

if __name__ == "__main__":
    main()
