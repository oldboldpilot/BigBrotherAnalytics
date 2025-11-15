"""
Options Analytics View - Comprehensive Options Trading Visualization

Features:
1. Options Chain Analysis - Display options data with IV, strikes, bid/ask
2. Greeks Display - Show delta, gamma, theta, vega, rho for active positions
3. Strategy Performance Charts - Real-time P&L and performance metrics
4. Options Pricing Visualization - Fair value vs market price

Author: BigBrotherAnalytics
Date: 2025-11-15
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm
import duckdb

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Database path
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bigbrother.duckdb"


def parse_occ_symbol(symbol: str) -> dict:
    """
    Parse OCC options symbol format

    Format: TICKER  YYMMDDCPPPPPPP
    Example: QS    270115C00005000 -> QS, 2027-01-15, Call, $5.00

    Returns:
        dict with underlying, expiration, option_type, strike
    """
    symbol = symbol.strip()

    # Check if it's an options symbol (contains date pattern)
    if len(symbol) < 15 or not any(c.isdigit() for c in symbol[6:]):
        return None

    # Split into components
    parts = symbol.split()
    if len(parts) < 2:
        return None

    underlying = parts[0].strip()
    option_part = ''.join(parts[1:]).strip()

    if len(option_part) < 15:
        return None

    try:
        # Parse date: YYMMDD
        year = 2000 + int(option_part[0:2])
        month = int(option_part[2:4])
        day = int(option_part[4:6])
        expiration = datetime(year, month, day)

        # Parse option type: C or P
        option_type = option_part[6]
        if option_type not in ['C', 'P']:
            return None

        # Parse strike: 8 digits with implied decimal
        strike_str = option_part[7:15]
        strike = float(strike_str) / 1000.0

        return {
            'underlying': underlying,
            'expiration': expiration,
            'option_type': 'Call' if option_type == 'C' else 'Put',
            'strike': strike,
            'days_to_expiration': (expiration - datetime.now()).days,
            'time_to_expiration': (expiration - datetime.now()).days / 365.0
        }
    except (ValueError, IndexError):
        return None


def calculate_black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'Call'):
    """
    Calculate Black-Scholes option price and Greeks

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'Call' or 'Put'

    Returns:
        dict with price and greeks
    """
    if T <= 0 or sigma <= 0:
        return {
            'price': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate price
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in r
    else:  # Put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    # Calculate Greeks (same for both call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'Call' else -d2)) / 365  # Per day
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


@st.cache_data(ttl=60)
def load_options_positions():
    """Load active options positions from database"""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
        SELECT
            id,
            symbol,
            quantity,
            avg_cost,
            current_price,
            market_value,
            unrealized_pnl,
            bot_strategy,
            opened_at
        FROM positions
        WHERE symbol LIKE '%C%' OR symbol LIKE '%P%'
        ORDER BY opened_at DESC
    """

    df = conn.execute(query).df()
    conn.close()

    return df


@st.cache_data(ttl=300)
def load_options_chain_data():
    """Load options chain data from database"""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
        SELECT
            symbol,
            date,
            strike,
            expiration,
            option_type,
            last_price,
            bid,
            ask,
            volume,
            open_interest,
            implied_volatility
        FROM options_data
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY symbol, expiration, strike
    """

    df = conn.execute(query).df()
    conn.close()

    return df


@st.cache_data(ttl=60)
def load_strategy_performance():
    """Load strategy performance metrics"""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Get positions grouped by strategy
    query = """
        SELECT
            bot_strategy,
            COUNT(*) as num_positions,
            SUM(market_value) as total_value,
            SUM(unrealized_pnl) as total_pnl,
            AVG(unrealized_pnl / NULLIF(avg_cost * quantity, 0)) * 100 as avg_return_pct
        FROM positions
        WHERE bot_strategy IS NOT NULL
        GROUP BY bot_strategy
        ORDER BY total_pnl DESC
    """

    df = conn.execute(query).df()
    conn.close()

    return df


@st.cache_data(ttl=300)
def get_risk_free_rate():
    """Get current risk-free rate from database"""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
        SELECT rate
        FROM risk_free_rates
        ORDER BY date DESC
        LIMIT 1
    """

    try:
        result = conn.execute(query).fetchone()
        conn.close()
        return result[0] / 100 if result else 0.045  # Default to 4.5%
    except:
        conn.close()
        return 0.045


def show_greeks_panel(options_positions: pd.DataFrame, risk_free_rate: float):
    """Display Greeks for active options positions"""
    st.header("Greeks for Active Positions")

    if options_positions.empty:
        st.info("No active options positions found")
        return

    # Parse options positions and calculate Greeks
    greeks_data = []

    for _, pos in options_positions.iterrows():
        parsed = parse_occ_symbol(pos['symbol'])
        if not parsed:
            continue

        # Estimate underlying price from current option price
        # In production, fetch actual underlying price from database
        S = pos['current_price'] * 10  # Rough approximation
        K = parsed['strike']
        T = parsed['time_to_expiration']
        sigma = 0.30  # Default IV, should fetch from database

        if T <= 0:
            continue

        greeks = calculate_black_scholes_greeks(S, K, T, risk_free_rate, sigma, parsed['option_type'])

        greeks_data.append({
            'Symbol': pos['symbol'],
            'Underlying': parsed['underlying'],
            'Type': parsed['option_type'],
            'Strike': f"${K:.2f}",
            'Expiration': parsed['expiration'].strftime('%Y-%m-%d'),
            'DTE': parsed['days_to_expiration'],
            'Quantity': int(pos['quantity']),
            'Delta': greeks['delta'] * pos['quantity'],
            'Gamma': greeks['gamma'] * pos['quantity'],
            'Theta': greeks['theta'] * pos['quantity'],
            'Vega': greeks['vega'] * pos['quantity'],
            'Rho': greeks['rho'] * pos['quantity'],
            'Fair Value': f"${greeks['price']:.2f}",
            'Market Price': f"${pos['current_price']:.2f}",
            'P&L': f"${pos['unrealized_pnl']:.2f}"
        })

    if not greeks_data:
        st.warning("Could not parse options positions")
        return

    greeks_df = pd.DataFrame(greeks_data)

    # Display summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_delta = greeks_df['Delta'].sum()
        st.metric("Portfolio Delta", f"{total_delta:.2f}",
                  help="Sensitivity to $1 change in underlying price")

    with col2:
        total_gamma = greeks_df['Gamma'].sum()
        st.metric("Portfolio Gamma", f"{total_gamma:.4f}",
                  help="Rate of change of delta")

    with col3:
        total_theta = greeks_df['Theta'].sum()
        st.metric("Portfolio Theta", f"${total_theta:.2f}/day",
                  delta=f"{total_theta:.2f}",
                  delta_color="inverse" if total_theta < 0 else "normal",
                  help="Time decay per day")

    with col4:
        total_vega = greeks_df['Vega'].sum()
        st.metric("Portfolio Vega", f"${total_vega:.2f}",
                  help="Sensitivity to 1% change in IV")

    with col5:
        total_rho = greeks_df['Rho'].sum()
        st.metric("Portfolio Rho", f"${total_rho:.2f}",
                  help="Sensitivity to 1% change in interest rate")

    st.divider()

    # Display detailed Greeks table
    st.subheader("Position Greeks Details")
    st.dataframe(
        greeks_df,
        use_container_width=True,
        height=400
    )

    # Greeks visualization
    st.subheader("Greeks Distribution")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=greeks_df['Symbol'],
        y=greeks_df['Delta'],
        name='Delta',
        marker_color='blue'
    ))

    fig.update_layout(
        title="Delta by Position",
        xaxis_title="Position",
        yaxis_title="Delta",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def show_options_chain(chain_data: pd.DataFrame):
    """Display options chain analysis"""
    st.header("Options Chain Analysis")

    if chain_data.empty:
        st.info("No options chain data available. Options data will be populated when the trading bot fetches option quotes.")
        return

    # Symbol selector
    symbols = chain_data['symbol'].unique()
    selected_symbol = st.selectbox("Select Underlying", symbols)

    # Filter for selected symbol
    symbol_chain = chain_data[chain_data['symbol'] == selected_symbol].copy()

    # Expiration selector
    expirations = sorted(symbol_chain['expiration'].unique())
    selected_expiration = st.selectbox("Select Expiration", expirations)

    # Filter for selected expiration
    exp_chain = symbol_chain[symbol_chain['expiration'] == selected_expiration].copy()

    # Separate calls and puts
    calls = exp_chain[exp_chain['option_type'] == 'call'].sort_values('strike')
    puts = exp_chain[exp_chain['option_type'] == 'put'].sort_values('strike')

    # Display chain
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calls")
        if not calls.empty:
            calls_display = calls[['strike', 'last_price', 'bid', 'ask', 'volume', 'open_interest', 'implied_volatility']].copy()
            calls_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
            st.dataframe(calls_display, use_container_width=True)
        else:
            st.info("No call options data")

    with col2:
        st.subheader("Puts")
        if not puts.empty:
            puts_display = puts[['strike', 'last_price', 'bid', 'ask', 'volume', 'open_interest', 'implied_volatility']].copy()
            puts_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
            st.dataframe(puts_display, use_container_width=True)
        else:
            st.info("No put options data")

    # IV Skew visualization
    st.subheader("Implied Volatility Skew")

    fig = go.Figure()

    if not calls.empty:
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=calls['implied_volatility'],
            mode='lines+markers',
            name='Calls',
            line=dict(color='green')
        ))

    if not puts.empty:
        fig.add_trace(go.Scatter(
            x=puts['strike'],
            y=puts['implied_volatility'],
            mode='lines+markers',
            name='Puts',
            line=dict(color='red')
        ))

    fig.update_layout(
        title=f"IV Skew - {selected_symbol} (Exp: {selected_expiration})",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def show_strategy_performance(performance_data: pd.DataFrame):
    """Display real-time strategy performance charts"""
    st.header("Strategy Performance")

    if performance_data.empty:
        st.info("No strategy performance data available yet")
        return

    # Performance metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        total_positions = performance_data['num_positions'].sum()
        st.metric("Total Positions", int(total_positions))

    with col2:
        total_value = performance_data['total_value'].sum()
        st.metric("Total Value", f"${total_value:,.2f}")

    with col3:
        total_pnl = performance_data['total_pnl'].sum()
        st.metric("Total P&L", f"${total_pnl:,.2f}",
                  delta=f"{total_pnl:,.2f}",
                  delta_color="normal" if total_pnl >= 0 else "inverse")

    st.divider()

    # Strategy breakdown
    st.subheader("Performance by Strategy")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=performance_data['bot_strategy'],
        y=performance_data['total_pnl'],
        marker_color=performance_data['total_pnl'].apply(lambda x: 'green' if x >= 0 else 'red'),
        text=performance_data['total_pnl'].apply(lambda x: f"${x:,.0f}"),
        textposition='outside'
    ))

    fig.update_layout(
        title="P&L by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Total P&L ($)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Return percentage chart
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=performance_data['bot_strategy'],
        y=performance_data['avg_return_pct'],
        marker_color=performance_data['avg_return_pct'].apply(lambda x: 'green' if x >= 0 else 'red'),
        text=performance_data['avg_return_pct'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside'
    ))

    fig2.update_layout(
        title="Average Return % by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Return (%)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Detailed table
    st.subheader("Strategy Details")
    st.dataframe(
        performance_data.style.format({
            'num_positions': '{:.0f}',
            'total_value': '${:,.2f}',
            'total_pnl': '${:,.2f}',
            'avg_return_pct': '{:.2f}%'
        }),
        use_container_width=True
    )


def show_pricing_visualization(options_positions: pd.DataFrame, risk_free_rate: float):
    """Display options pricing visualization"""
    st.header("Options Pricing Analysis")

    if options_positions.empty:
        st.info("No options positions to analyze")
        return

    pricing_data = []

    for _, pos in options_positions.iterrows():
        parsed = parse_occ_symbol(pos['symbol'])
        if not parsed:
            continue

        # Estimate underlying price
        S = pos['current_price'] * 10
        K = parsed['strike']
        T = parsed['time_to_expiration']
        sigma = 0.30

        if T <= 0:
            continue

        greeks = calculate_black_scholes_greeks(S, K, T, risk_free_rate, sigma, parsed['option_type'])

        market_price = pos['current_price']
        fair_value = greeks['price']
        mispricing = ((market_price - fair_value) / fair_value * 100) if fair_value > 0 else 0

        pricing_data.append({
            'Symbol': pos['symbol'],
            'Underlying': parsed['underlying'],
            'Type': parsed['option_type'],
            'Strike': K,
            'DTE': parsed['days_to_expiration'],
            'Market Price': market_price,
            'Fair Value': fair_value,
            'Mispricing %': mispricing,
            'Recommendation': 'Overvalued' if mispricing > 5 else 'Undervalued' if mispricing < -5 else 'Fair'
        })

    if not pricing_data:
        st.warning("Could not calculate pricing for options positions")
        return

    pricing_df = pd.DataFrame(pricing_data)

    # Mispricing chart
    fig = go.Figure()

    colors = pricing_df['Mispricing %'].apply(
        lambda x: 'red' if x > 5 else 'green' if x < -5 else 'gray'
    )

    fig.add_trace(go.Bar(
        x=pricing_df['Symbol'],
        y=pricing_df['Mispricing %'],
        marker_color=colors,
        text=pricing_df['Mispricing %'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside'
    ))

    # Add reference lines
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Overvalued threshold")
    fig.add_hline(y=-5, line_dash="dash", line_color="green", annotation_text="Undervalued threshold")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title="Options Mispricing Analysis",
        xaxis_title="Position",
        yaxis_title="Mispricing (%)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pricing table
    st.subheader("Detailed Pricing Analysis")
    st.dataframe(
        pricing_df.style.format({
            'Strike': '${:.2f}',
            'Market Price': '${:.2f}',
            'Fair Value': '${:.2f}',
            'Mispricing %': '{:.2f}%'
        }).apply(
            lambda row: ['background-color: #ffcccc' if row['Mispricing %'] > 5
                        else 'background-color: #ccffcc' if row['Mispricing %'] < -5
                        else '' for _ in row],
            axis=1
        ),
        use_container_width=True
    )


def show_options_analytics():
    """Main options analytics view"""
    st.title("Options Analytics Dashboard")
    st.markdown("Comprehensive options trading analysis and monitoring")

    # Load data
    options_positions = load_options_positions()
    chain_data = load_options_chain_data()
    performance_data = load_strategy_performance()
    risk_free_rate = get_risk_free_rate()

    # Display current risk-free rate
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Market Parameters")
    st.sidebar.metric("Risk-Free Rate", f"{risk_free_rate*100:.2f}%")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Options Chain",
        "📈 Greeks & Positions",
        "💰 Strategy Performance",
        "🎯 Pricing Analysis"
    ])

    with tab1:
        show_options_chain(chain_data)

    with tab2:
        show_greeks_panel(options_positions, risk_free_rate)

    with tab3:
        show_strategy_performance(performance_data)

    with tab4:
        show_pricing_visualization(options_positions, risk_free_rate)
