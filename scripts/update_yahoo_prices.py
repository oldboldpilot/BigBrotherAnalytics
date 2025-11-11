#!/usr/bin/env python3
"""
Update Prices and News from Yahoo Finance

Fetches real-time prices and news articles for all holdings and stores
in database for dashboard charts, position tracking, and news feed.

Yahoo Finance Free Tier Provides:
- Real-time prices (15-min delay)
- News articles for each ticker
- Historical OHLCV data
- No cost, no API key required

Usage:
    uv run python scripts/update_yahoo_prices.py
    uv run python scripts/update_yahoo_prices.py --news-only  # Fetch only news
    uv run python scripts/update_yahoo_prices.py --prices-only  # Fetch only prices
"""

import yfinance as yf
import duckdb
from pathlib import Path
from datetime import datetime
import pandas as pd
import hashlib
import sys
import os

# Set LD_LIBRARY_PATH for C++ sentiment analyzer
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/local/lib/x86_64-unknown-linux-gnu:/usr/lib/llvm-18/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Import C++ sentiment analyzer
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
try:
    import news_ingestion_py
    SENTIMENT_ANALYZER = news_ingestion_py.SentimentAnalyzer()
    USE_CPP_SENTIMENT = True
    print("✅ Using C++23 sentiment analyzer (60+ keywords, faster)")
except ImportError as e:
    print(f"⚠️  C++ sentiment analyzer not available: {e}")
    print("   Falling back to simple Python sentiment analysis")
    USE_CPP_SENTIMENT = False
    SENTIMENT_ANALYZER = None

def analyze_sentiment_cpp(text):
    """Use C++23 sentiment analyzer"""
    result = SENTIMENT_ANALYZER.analyze(text)
    return result.score, result.label

def analyze_sentiment_python(text):
    """Fallback Python-based sentiment analysis"""
    if not text:
        return 0.0, 'neutral'

    text_lower = text.lower()

    # Basic positive/negative keywords
    positive_words = ['surge', 'soar', 'rally', 'gain', 'profit', 'growth', 'beat', 'positive', 'strong']
    negative_words = ['plunge', 'crash', 'drop', 'loss', 'decline', 'fall', 'miss', 'negative', 'weak']

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    total = positive_count + negative_count
    if total == 0:
        return 0.0, 'neutral'

    score = (positive_count - negative_count) / total
    label = 'positive' if score > 0.2 else 'negative' if score < -0.2 else 'neutral'
    return score, label

def analyze_sentiment(text):
    """Route to C++ or Python sentiment analyzer"""
    if USE_CPP_SENTIMENT:
        return analyze_sentiment_cpp(text)
    else:
        return analyze_sentiment_python(text)

def update_news_from_yahoo(conn, symbols):
    """Fetch news articles from Yahoo Finance for given symbols"""

    print("\n📰 Fetching news from Yahoo Finance...")

    # Ensure news_articles table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            article_id VARCHAR PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            description TEXT,
            url VARCHAR,
            published_at TIMESTAMP NOT NULL,
            source_name VARCHAR,
            author VARCHAR,
            content TEXT,
            sentiment_score DOUBLE,
            sentiment_label VARCHAR,
            keywords VARCHAR,
            image_url VARCHAR,
            category VARCHAR,
            language VARCHAR DEFAULT 'en',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    total_articles = 0
    new_articles = 0

    for symbol in symbols:
        try:
            print(f"   {symbol:10s} ... ", end="", flush=True)

            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                print("No news")
                continue

            symbol_articles = 0

            for article in news:
                # Generate article ID from URL
                url = article.get('link', '')
                article_id = hashlib.sha256(url.encode()).hexdigest()[:16]

                # Check if already exists
                existing = conn.execute(
                    "SELECT 1 FROM news_articles WHERE article_id = ?",
                    [article_id]
                ).fetchone()

                if existing:
                    continue

                # Extract fields
                title = article.get('title', '')
                description = article.get('summary', '')
                published_timestamp = datetime.fromtimestamp(article.get('providerPublishTime', 0))

                # Source: Original publisher + "via Yahoo Finance"
                original_publisher = article.get('publisher', 'Unknown')
                source_name = f"{original_publisher} (via Yahoo Finance)"

                # Combine title + description for sentiment (C++ analyzer if available)
                text = f"{title} {description}"
                sentiment_score, sentiment_label = analyze_sentiment(text)

                # Extract thumbnail
                thumbnail = article.get('thumbnail', {})
                image_url = thumbnail.get('resolutions', [{}])[0].get('url') if thumbnail else None

                # Insert into database
                conn.execute("""
                    INSERT INTO news_articles
                    (article_id, symbol, title, description, url, published_at,
                     source_name, sentiment_score, sentiment_label, image_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    article_id, symbol, title, description, url,
                    published_timestamp, source_name,
                    sentiment_score, sentiment_label, image_url
                ])

                symbol_articles += 1
                new_articles += 1

            total_articles += len(news)
            print(f"✅ {len(news)} articles ({symbol_articles} new)")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n   📊 Total: {total_articles} articles, {new_articles} new")
    conn.commit()

def update_prices_from_yahoo():
    """Fetch current prices from Yahoo Finance and update database"""

    print("=" * 80)
    print("Updating Prices from Yahoo Finance (Free Tier)")
    print("=" * 80)

    # Connect to database
    db_path = Path('data/bigbrother.duckdb')
    conn = duckdb.connect(str(db_path))

    # Get all unique symbols from open positions
    print("\n📊 Fetching symbols from open positions...")
    symbols_query = """
        SELECT DISTINCT
            CASE
                WHEN asset_type = 'OPTION' THEN
                    -- Extract underlying symbol from option symbol
                    -- SPY   251219C00679000 -> SPY
                    TRIM(SUBSTRING(symbol, 1, POSITION(' ' IN symbol)))
                ELSE symbol
            END as underlying_symbol,
            asset_type
        FROM tax_lots
        WHERE is_closed = false
    """

    positions = conn.execute(symbols_query).fetchall()

    # Get unique underlying symbols
    symbols = list(set([p[0] for p in positions if p[0]]))
    print(f"   Found {len(symbols)} unique symbols: {', '.join(symbols)}")

    # Create price history table if not exists
    print("\n📦 Creating price_history table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source VARCHAR DEFAULT 'yahoo_finance',
            PRIMARY KEY (symbol, timestamp)
        )
    """)

    # Fetch prices for each symbol
    print("\n💰 Fetching current prices from Yahoo Finance...")
    success_count = 0
    failed_symbols = []

    for symbol in symbols:
        try:
            print(f"   {symbol:10s} ... ", end="", flush=True)

            # Fetch ticker data
            ticker = yf.Ticker(symbol)

            # Get current price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            if not current_price:
                print("❌ No price data")
                failed_symbols.append(symbol)
                continue

            # Get latest OHLCV (1 day history)
            hist = ticker.history(period="1d")

            if hist.empty:
                print("❌ No history")
                failed_symbols.append(symbol)
                continue

            # Get latest row
            latest = hist.iloc[-1]
            timestamp = latest.name.to_pydatetime()

            # Insert/update price history
            conn.execute("""
                INSERT OR REPLACE INTO price_history
                (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'yahoo_finance')
            """, [
                symbol,
                timestamp,
                float(latest['Open']),
                float(latest['High']),
                float(latest['Low']),
                float(latest['Close']),
                int(latest['Volume'])
            ])

            print(f"✅ ${current_price:8.2f}  (Vol: {latest['Volume']:,})")
            success_count += 1

        except Exception as e:
            print(f"❌ Error: {e}")
            failed_symbols.append(symbol)

    conn.commit()

    # Summary
    print("\n" + "=" * 80)
    print("UPDATE SUMMARY")
    print("=" * 80)
    print(f"   ✅ Successfully updated: {success_count}/{len(symbols)}")

    if failed_symbols:
        print(f"   ❌ Failed symbols: {', '.join(failed_symbols)}")

    # Show latest prices
    print("\n📈 Latest Prices in Database:")
    latest_prices = conn.execute("""
        SELECT
            symbol,
            timestamp,
            close,
            volume,
            source
        FROM price_history
        WHERE timestamp = (
            SELECT MAX(timestamp)
            FROM price_history p2
            WHERE p2.symbol = price_history.symbol
        )
        ORDER BY symbol
    """).fetchdf()

    print(latest_prices.to_string(index=False))

    print("\n💡 Tip: Use this data in dashboard for real-time price updates")
    print("   Example: SELECT close FROM price_history WHERE symbol = 'SPY' ORDER BY timestamp DESC LIMIT 1")
    print()

    conn.close()
    return 0

def main():
    """Main entry point with command-line argument handling"""

    # Parse arguments
    fetch_prices = True
    fetch_news = True

    if '--prices-only' in sys.argv:
        fetch_news = False
    elif '--news-only' in sys.argv:
        fetch_prices = False

    print("=" * 80)
    print("Yahoo Finance Data Update (Free Tier)")
    print("=" * 80)
    print(f"   Fetching prices: {'✅' if fetch_prices else '❌'}")
    print(f"   Fetching news:   {'✅' if fetch_news else '❌'}")

    # Connect to database
    db_path = Path('data/bigbrother.duckdb')
    conn = duckdb.connect(str(db_path))

    # Get symbols from open positions
    print("\n📊 Fetching symbols from open positions...")
    symbols_query = """
        SELECT DISTINCT
            CASE
                WHEN asset_type = 'OPTION' THEN
                    -- Extract underlying symbol from option symbol
                    TRIM(SUBSTRING(symbol, 1, POSITION(' ' IN symbol)))
                ELSE symbol
            END as underlying_symbol
        FROM tax_lots
        WHERE is_closed = false
    """
    positions = conn.execute(symbols_query).fetchall()
    symbols = list(set([p[0] for p in positions if p[0]]))

    print(f"   Found {len(symbols)} unique symbols: {', '.join(symbols)}")

    # Fetch data
    if fetch_news:
        update_news_from_yahoo(conn, symbols)

    if fetch_prices:
        # Close and let update_prices_from_yahoo handle its own connection
        conn.close()
        return update_prices_from_yahoo()

    conn.close()
    print("\n✅ Update complete")
    return 0

if __name__ == "__main__":
    exit(main())
