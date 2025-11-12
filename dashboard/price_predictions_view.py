"""
Price Predictions View for BigBrother Analytics Dashboard

Displays ML-based price predictions with:
- 1-day, 5-day, 20-day forecasts
- Confidence scores
- Trading signals (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- Feature importance display
- Multi-horizon forecast visualization

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-11
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def fetch_fred_rates_from_db(conn):
    """Fetch current FRED rates from database"""
    try:
        rates_df = conn.execute("""
            SELECT rate_name, rate_value
            FROM risk_free_rates
            ORDER BY last_updated DESC
            LIMIT 5
        """).df()

        if rates_df.empty:
            return None

        rates = {}
        for _, row in rates_df.iterrows():
            rate_name = row['rate_name']
            if rate_name == '3_month_treasury':
                rates['treasury_3m'] = row['rate_value']
            elif rate_name == '2_year_treasury':
                rates['treasury_2y'] = row['rate_value']
            elif rate_name == '10_year_treasury':
                rates['treasury_10y'] = row['rate_value']
            elif rate_name == 'fed_funds_rate':
                rates['fed_funds'] = row['rate_value']

        return rates
    except Exception:
        return None


def get_default_rates():
    """Get default rates when FRED is unavailable"""
    return {
        'treasury_3m': 0.0392,
        'treasury_2y': 0.0355,
        'treasury_10y': 0.0411,
        'fed_funds': 0.0387,
    }


def generate_features_for_symbol(symbol: str, rates: dict):
    """
    Generate feature vector for prediction

    NOTE: This is using placeholder features for demonstration.
    In production, these would come from:
    - Real-time market data (prices, volumes)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Sentiment analysis (news, social media)
    - Economic data (employment, GDP, inflation)
    - Sector correlations
    """

    # Placeholder features (in production, fetch from data sources)
    features = {
        # Technical indicators (10)
        'rsi_14': 55.3,                 # RSI slightly above neutral
        'macd': 0.42,                   # Positive momentum
        'macd_signal': 0.35,            # Below MACD (bullish)
        'macd_histogram': 0.07,         # Positive histogram
        'bb_upper': 152.50,             # Bollinger upper band
        'bb_middle': 150.00,            # Bollinger middle band
        'bb_lower': 147.50,             # Bollinger lower band
        'atr_14': 2.15,                 # Average True Range
        'volume_ratio': 1.23,           # Above average volume
        'momentum_5d': 0.018,           # 1.8% gain over 5 days

        # Sentiment (5)
        'news_sentiment': 0.35,         # Positive news sentiment
        'social_sentiment': 0.15,       # Slightly positive social
        'analyst_rating': 3.8,          # Buy rating (1-5 scale)
        'put_call_ratio': 0.85,         # Slightly bullish
        'vix_level': 16.5,              # Low fear

        # Economic indicators (5)
        'employment_change': 185000,    # Strong job growth
        'gdp_growth': 0.025,            # 2.5% quarterly
        'inflation_rate': 0.031,        # 3.1% CPI
        'fed_rate': rates.get('fed_funds', 0.04),
        'treasury_yield_10y': rates.get('treasury_10y', 0.04),

        # Sector correlation (5)
        'sector_momentum': 0.042,       # 4.2% sector gain
        'spy_correlation': 0.78,        # High correlation with market
        'sector_beta': 1.15,            # Slightly more volatile
        'peer_avg_return': 0.025,       # Peers up 2.5%
        'market_regime': 0.65,          # Bullish regime
    }

    return features


def predict_price_movements(symbol: str, features: dict):
    """
    Generate price predictions using feature-based scoring

    NOTE: This is a placeholder implementation using heuristics.
    In production, this would call the C++ CUDA-accelerated neural network
    trained on historical data.
    """

    # Calculate component scores
    technical_score = (
        (features['rsi_14'] - 50) / 50 * 0.3 +
        features['macd_histogram'] * 10 * 0.3 +
        (features['volume_ratio'] - 1.0) * 0.2
    )

    sentiment_score = (
        features['news_sentiment'] * 0.4 +
        features['social_sentiment'] * 0.3 +
        (features['analyst_rating'] - 3.0) / 2.0 * 0.3
    )

    economic_score = (
        features['gdp_growth'] * 20 * 0.4 +
        (1.0 - features['inflation_rate']) * 0.3 +
        features['sector_momentum'] * 5 * 0.3
    )

    overall_score = (technical_score * 0.4 + sentiment_score * 0.3 + economic_score * 0.3)

    # Generate predictions with different horizons
    predictions = {
        'symbol': symbol,
        'day_1_change': overall_score * 1.5,      # 1-day forecast
        'day_5_change': overall_score * 4.0,      # 5-day forecast
        'day_20_change': overall_score * 10.0,    # 20-day forecast
        'confidence_1d': min(0.95, 0.6 + abs(overall_score) * 0.2),
        'confidence_5d': min(0.90, 0.5 + abs(overall_score) * 0.2),
        'confidence_20d': min(0.85, 0.4 + abs(overall_score) * 0.2),
        'timestamp': datetime.now().isoformat(),
        'technical_score': technical_score,
        'sentiment_score': sentiment_score,
        'economic_score': economic_score,
        'features': features,
    }

    # Generate trading signals
    def get_signal(change: float) -> str:
        if change > 5.0: return "STRONG_BUY"
        if change > 2.0: return "BUY"
        if change < -5.0: return "STRONG_SELL"
        if change < -2.0: return "SELL"
        return "HOLD"

    predictions['signal_1d'] = get_signal(predictions['day_1_change'])
    predictions['signal_5d'] = get_signal(predictions['day_5_change'])
    predictions['signal_20d'] = get_signal(predictions['day_20_change'])

    return predictions


def get_signal_color(signal: str) -> str:
    """Get color for signal badge"""
    colors = {
        'STRONG_BUY': '#00C853',    # Dark green
        'BUY': '#69F0AE',           # Light green
        'HOLD': '#FFD54F',          # Yellow
        'SELL': '#FF8A65',          # Light red
        'STRONG_SELL': '#D32F2F',   # Dark red
    }
    return colors.get(signal, '#9E9E9E')


def get_signal_emoji(signal: str) -> str:
    """Get emoji for signal"""
    emojis = {
        'STRONG_BUY': '🚀',
        'BUY': '📈',
        'HOLD': '➖',
        'SELL': '📉',
        'STRONG_SELL': '⚠️',
    }
    return emojis.get(signal, '❓')


def show_price_predictions(conn):
    """Display price predictions view"""
    st.header("🔮 AI Price Predictions")

    st.markdown("""
    Machine learning-based price forecasts using:
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Sentiment analysis (news, social media)
    - Economic data (employment, GDP, FRED rates)
    - Sector correlations and market regime
    """)

    # Placeholder notice
    st.warning("⚠️ **Using Placeholder Model** - ML model training in progress. Predictions are based on simplified heuristics for demonstration purposes.")

    st.divider()

    # Symbol selection
    st.subheader("🎯 Select Symbol")

    # Get available symbols from positions
    try:
        symbols_df = conn.execute("""
            SELECT DISTINCT symbol
            FROM positions
            ORDER BY symbol
        """).df()

        if not symbols_df.empty:
            available_symbols = symbols_df['symbol'].tolist()
        else:
            available_symbols = []
    except Exception:
        available_symbols = []

    # Add some default symbols if no positions
    default_symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META']
    all_symbols = sorted(list(set(available_symbols + default_symbols)))

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_symbol = st.selectbox("Symbol", all_symbols, index=0)

    with col2:
        if st.button("🔄 Generate Prediction", use_container_width=True):
            st.rerun()

    st.divider()

    # Fetch FRED rates
    rates = fetch_fred_rates_from_db(conn)
    if rates is None:
        rates = get_default_rates()
        st.info("💡 Using default risk-free rates. Run FRED data collection to get live rates.")

    # Generate prediction
    with st.spinner(f"Generating prediction for {selected_symbol}..."):
        features = generate_features_for_symbol(selected_symbol, rates)
        prediction = predict_price_movements(selected_symbol, features)

    # Display timestamp
    pred_time = datetime.fromisoformat(prediction['timestamp'])
    st.caption(f"Prediction generated at: {pred_time.strftime('%Y-%m-%d %H:%M:%S')}")

    st.divider()

    # === PREDICTION CARDS ===
    st.subheader("📊 Price Forecasts")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1-Day Forecast")
        change_1d = prediction['day_1_change']
        signal_1d = prediction['signal_1d']
        conf_1d = prediction['confidence_1d']

        st.metric(
            "Expected Change",
            f"{change_1d:+.2f}%",
            delta=f"Confidence: {conf_1d:.1%}"
        )

        signal_color = get_signal_color(signal_1d)
        signal_emoji = get_signal_emoji(signal_1d)
        st.markdown(
            f"<div style='background-color: {signal_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
            f"{signal_emoji} {signal_1d}"
            f"</div>",
            unsafe_allow_html=True
        )

        # Confidence bar
        st.progress(conf_1d, text=f"Confidence: {conf_1d:.1%}")

    with col2:
        st.markdown("### 5-Day Forecast")
        change_5d = prediction['day_5_change']
        signal_5d = prediction['signal_5d']
        conf_5d = prediction['confidence_5d']

        st.metric(
            "Expected Change",
            f"{change_5d:+.2f}%",
            delta=f"Confidence: {conf_5d:.1%}"
        )

        signal_color = get_signal_color(signal_5d)
        signal_emoji = get_signal_emoji(signal_5d)
        st.markdown(
            f"<div style='background-color: {signal_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
            f"{signal_emoji} {signal_5d}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.progress(conf_5d, text=f"Confidence: {conf_5d:.1%}")

    with col3:
        st.markdown("### 20-Day Forecast")
        change_20d = prediction['day_20_change']
        signal_20d = prediction['signal_20d']
        conf_20d = prediction['confidence_20d']

        st.metric(
            "Expected Change",
            f"{change_20d:+.2f}%",
            delta=f"Confidence: {conf_20d:.1%}"
        )

        signal_color = get_signal_color(signal_20d)
        signal_emoji = get_signal_emoji(signal_20d)
        st.markdown(
            f"<div style='background-color: {signal_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
            f"{signal_emoji} {signal_20d}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.progress(conf_20d, text=f"Confidence: {conf_20d:.1%}")

    st.divider()

    # === OVERALL SIGNAL ===
    st.subheader("🎯 Overall Recommendation")

    # Calculate weighted average
    weighted_change = (
        prediction['day_1_change'] * prediction['confidence_1d'] * 0.5 +
        prediction['day_5_change'] * prediction['confidence_5d'] * 0.3 +
        prediction['day_20_change'] * prediction['confidence_20d'] * 0.2
    )

    if weighted_change > 5.0:
        overall_signal = "STRONG_BUY"
    elif weighted_change > 2.0:
        overall_signal = "BUY"
    elif weighted_change < -5.0:
        overall_signal = "STRONG_SELL"
    elif weighted_change < -2.0:
        overall_signal = "SELL"
    else:
        overall_signal = "HOLD"

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        signal_color = get_signal_color(overall_signal)
        signal_emoji = get_signal_emoji(overall_signal)

        st.markdown(
            f"<div style='background-color: {signal_color}; padding: 30px; border-radius: 10px; text-align: center; color: white;'>"
            f"<h1>{signal_emoji} {overall_signal}</h1>"
            f"<h3>Weighted Expected Change: {weighted_change:+.2f}%</h3>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # === CHARTS ===
    st.subheader("📈 Visualization")

    tab1, tab2, tab3, tab4 = st.tabs(["Multi-Horizon Forecast", "Confidence Scores", "Score Breakdown", "Input Features"])

    with tab1:
        # Multi-horizon forecast bar chart
        forecast_df = pd.DataFrame({
            'Horizon': ['1-Day', '5-Day', '20-Day'],
            'Expected Change (%)': [
                prediction['day_1_change'],
                prediction['day_5_change'],
                prediction['day_20_change']
            ],
            'Confidence': [
                prediction['confidence_1d'],
                prediction['confidence_5d'],
                prediction['confidence_20d']
            ],
            'Signal': [signal_1d, signal_5d, signal_20d]
        })

        fig = px.bar(
            forecast_df,
            x='Horizon',
            y='Expected Change (%)',
            color='Expected Change (%)',
            color_continuous_scale=['red', 'yellow', 'green'],
            color_continuous_midpoint=0,
            text='Expected Change (%)',
            title=f'{selected_symbol} - Multi-Horizon Price Forecast'
        )

        fig.update_traces(texttemplate='%{text:+.2f}%', textposition='outside')
        fig.update_layout(height=500, showlegend=False)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Horizon": "Time Horizon",
                "Expected Change (%)": st.column_config.NumberColumn("Expected Change", format="%.2f%%"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
                "Signal": "Trading Signal"
            }
        )

    with tab2:
        # Confidence gauge charts
        st.markdown("### Prediction Confidence Levels")

        col1, col2, col3 = st.columns(3)

        for col, horizon, conf in [
            (col1, "1-Day", prediction['confidence_1d']),
            (col2, "5-Day", prediction['confidence_5d']),
            (col3, "20-Day", prediction['confidence_20d'])
        ]:
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf * 100,
                    title={'text': f"{horizon} Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "lightyellow"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Score breakdown
        st.markdown("### Prediction Score Breakdown")

        score_df = pd.DataFrame({
            'Component': ['Technical', 'Sentiment', 'Economic'],
            'Score': [
                prediction['technical_score'],
                prediction['sentiment_score'],
                prediction['economic_score']
            ],
            'Weight': [0.4, 0.3, 0.3]
        })

        score_df['Weighted Score'] = score_df['Score'] * score_df['Weight']

        fig = px.bar(
            score_df,
            x='Component',
            y='Score',
            color='Score',
            color_continuous_scale=['red', 'yellow', 'green'],
            color_continuous_midpoint=0,
            text='Score',
            title='Component Scores'
        )

        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig, use_container_width=True)

        # Show breakdown table
        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Component": "Component",
                "Score": st.column_config.NumberColumn("Raw Score", format="%.3f"),
                "Weight": st.column_config.NumberColumn("Weight", format="%.1f"),
                "Weighted Score": st.column_config.NumberColumn("Weighted Score", format="%.3f")
            }
        )

        st.caption(f"Overall Score: {prediction['technical_score'] * 0.4 + prediction['sentiment_score'] * 0.3 + prediction['economic_score'] * 0.3:.3f}")

    with tab4:
        # Feature display
        st.markdown("### Input Features (25 Features)")

        st.markdown("These are the features used to generate the prediction. In production, these would be fetched from real-time data sources.")

        features_list = []

        # Technical indicators
        st.markdown("#### 📊 Technical Indicators (10)")
        tech_features = [
            ('RSI (14)', features['rsi_14'], 'Relative Strength Index'),
            ('MACD', features['macd'], 'Moving Average Convergence Divergence'),
            ('MACD Signal', features['macd_signal'], 'MACD Signal Line'),
            ('MACD Histogram', features['macd_histogram'], 'MACD Histogram'),
            ('BB Upper', features['bb_upper'], 'Bollinger Band Upper'),
            ('BB Middle', features['bb_middle'], 'Bollinger Band Middle'),
            ('BB Lower', features['bb_lower'], 'Bollinger Band Lower'),
            ('ATR (14)', features['atr_14'], 'Average True Range'),
            ('Volume Ratio', features['volume_ratio'], 'Current/Average Volume'),
            ('5-Day Momentum', features['momentum_5d'], '5-Day Price Momentum'),
        ]

        tech_df = pd.DataFrame(tech_features, columns=['Feature', 'Value', 'Description'])
        st.dataframe(tech_df, use_container_width=True, hide_index=True)

        # Sentiment
        st.markdown("#### 📰 Sentiment Indicators (5)")
        sent_features = [
            ('News Sentiment', features['news_sentiment'], 'Aggregated news sentiment'),
            ('Social Sentiment', features['social_sentiment'], 'Social media sentiment'),
            ('Analyst Rating', features['analyst_rating'], 'Average analyst rating (1-5)'),
            ('Put/Call Ratio', features['put_call_ratio'], 'Options put/call ratio'),
            ('VIX Level', features['vix_level'], 'Market fear gauge'),
        ]

        sent_df = pd.DataFrame(sent_features, columns=['Feature', 'Value', 'Description'])
        st.dataframe(sent_df, use_container_width=True, hide_index=True)

        # Economic
        st.markdown("#### 💰 Economic Indicators (5)")
        econ_features = [
            ('Employment Change', features['employment_change'], 'Monthly job creation'),
            ('GDP Growth', features['gdp_growth'], 'Quarterly GDP growth rate'),
            ('Inflation Rate', features['inflation_rate'], 'CPI inflation rate'),
            ('Fed Funds Rate', features['fed_rate'], 'Federal Reserve rate'),
            ('10Y Treasury Yield', features['treasury_yield_10y'], '10-year Treasury yield'),
        ]

        econ_df = pd.DataFrame(econ_features, columns=['Feature', 'Value', 'Description'])
        st.dataframe(econ_df, use_container_width=True, hide_index=True)

        # Sector
        st.markdown("#### 🏢 Sector Correlation (5)")
        sector_features = [
            ('Sector Momentum', features['sector_momentum'], 'Sector average return'),
            ('SPY Correlation', features['spy_correlation'], 'Correlation with S&P 500'),
            ('Sector Beta', features['sector_beta'], 'Volatility vs market'),
            ('Peer Avg Return', features['peer_avg_return'], 'Peer companies average'),
            ('Market Regime', features['market_regime'], 'Bullish/bearish regime'),
        ]

        sector_df = pd.DataFrame(sector_features, columns=['Feature', 'Value', 'Description'])
        st.dataframe(sector_df, use_container_width=True, hide_index=True)

    st.divider()

    # === DISCLAIMER ===
    st.info("""
    📝 **Disclaimer**:
    - These predictions are for informational purposes only and should not be considered financial advice.
    - The model is currently using placeholder heuristics. Full ML model training is in progress.
    - Always conduct your own research and consult with a financial advisor before making investment decisions.
    - Past performance does not guarantee future results.
    """)
