#!/usr/bin/env python3
"""
BigBrotherAnalytics: Multi-Source News Ingestion

Fetches news from multiple sources (NewsAPI + AlphaVantage) and applies
power-of-2-choices algorithm to select the most reliable sentiment scores.

Features:
- Dual-source news collection
- Power-of-2-choices sentiment selection (choose least extreme score)
- Article matching by URL and title similarity
- Stores both sentiment scores for analysis

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: Multi-Source News Ingestion
"""

import os
import sys
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import duckdb
import yaml

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import sentiment selection utilities
from scripts.data_collection.sentiment_selection import (
    create_multi_source_articles,
    convert_to_db_format,
    MultiSourceArticle
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_api_keys() -> Tuple[Optional[str], Optional[str]]:
    """
    Load API keys from config file

    Returns:
        Tuple of (newsapi_key, alphavantage_key)
    """
    try:
        import yaml
        config_path = BASE_DIR / 'configs' / 'api_keys.yaml'

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                newsapi_key = config.get('news_api', {}).get('api_key')
                alphavantage_key = config.get('alpha_vantage', {}).get('api_key')
                return newsapi_key, alphavantage_key
    except Exception as e:
        logger.error(f"Error loading API keys: {e}")

    return None, None


def get_traded_symbols(db_path: str, limit: int = 10) -> List[str]:
    """
    Get list of traded symbols from database

    Args:
        db_path: Path to DuckDB database
        limit: Maximum number of symbols

    Returns:
        List of stock symbols
    """
    try:
        conn = duckdb.connect(str(db_path), read_only=True)

        result = conn.execute("""
            SELECT DISTINCT symbol
            FROM stocks
            ORDER BY symbol
            LIMIT ?
        """, [limit]).fetchall()

        conn.close()

        symbols = [row[0] for row in result]
        logger.info(f"Fetched {len(symbols)} symbols from database")

        return symbols

    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return []


def fetch_newsapi_articles(api_key: str, symbol: str, from_date: str, to_date: str) -> List[Dict]:
    """
    Fetch articles from NewsAPI with keyword-based sentiment

    Args:
        api_key: NewsAPI key
        symbol: Stock symbol
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)

    Returns:
        List of article dictionaries
    """
    try:
        from newsapi import NewsApiClient
        from scripts.data_collection.news_ingestion import simple_sentiment

        newsapi = NewsApiClient(api_key=api_key)

        # Fetch from NewsAPI
        response = newsapi.get_everything(
            q=symbol,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt',
            page_size=50
        )

        if response.get('status') != 'ok':
            logger.warning(f"NewsAPI error for {symbol}: {response.get('message')}")
            return []

        articles = []
        for article in response.get('articles', []):
            # Sentiment analysis
            text_to_analyze = f"{article.get('title', '')} {article.get('description', '')}"
            score, label, pos_kw, neg_kw = simple_sentiment(text_to_analyze)

            # Parse published date
            published_at_str = article.get('publishedAt', '')
            try:
                published_dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                published_at = int(published_dt.timestamp())
            except:
                published_at = 0

            # Create article dict
            source = article.get('source', {})
            articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'source_name': source.get('name', ''),
                'source_id': source.get('id', ''),
                'author': article.get('author', ''),
                'published_at': published_at,
                'sentiment_score': score,
                'sentiment_label': label,
                'positive_keywords': pos_kw,
                'negative_keywords': neg_kw
            })

        logger.info(f"  NewsAPI: Fetched {len(articles)} articles for {symbol}")
        return articles

    except Exception as e:
        logger.error(f"  NewsAPI error for {symbol}: {e}")
        return []


def fetch_alphavantage_articles(api_key: str, symbol: str, from_date: str) -> List[Dict]:
    """
    Fetch articles from AlphaVantage with AI-powered sentiment

    Args:
        api_key: AlphaVantage API key
        symbol: Stock symbol
        from_date: Start date (YYYYMMDDTHHMM format for AlphaVantage)

    Returns:
        List of article dictionaries
    """
    try:
        import requests

        # Convert date format for AlphaVantage (YYYYMMDDTHHMM)
        try:
            dt = datetime.strptime(from_date, '%Y-%m-%d')
            alphavantage_date = dt.strftime('%Y%m%dT0000')
        except:
            alphavantage_date = ""

        # Call AlphaVantage NEWS_SENTIMENT API
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_key,
            'limit': 50
        }

        if alphavantage_date:
            params['time_from'] = alphavantage_date

        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            logger.warning(f"AlphaVantage HTTP error for {symbol}: {response.status_code}")
            return []

        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            logger.warning(f"AlphaVantage error for {symbol}: {data['Error Message']}")
            return []

        if 'Note' in data:
            logger.warning(f"AlphaVantage rate limit note: {data['Note']}")
            return []

        # Parse articles
        articles = []
        for item in data.get('feed', []):
            # Extract ticker-specific sentiment
            ticker_sentiment_list = item.get('ticker_sentiment', [])
            ticker_score = None
            ticker_label = None
            ticker_relevance = None

            for ts in ticker_sentiment_list:
                if ts.get('ticker', '').upper() == symbol.upper():
                    ticker_score = float(ts.get('ticker_sentiment_score', 0.0))
                    ticker_label = ts.get('ticker_sentiment_label', 'Neutral')
                    ticker_relevance = float(ts.get('relevance_score', 0.0))
                    break

            # Use overall sentiment if ticker-specific not found
            if ticker_score is None:
                ticker_score = float(item.get('overall_sentiment_score', 0.0))
                ticker_label = item.get('overall_sentiment_label', 'Neutral')

            # Parse timestamp
            time_published = item.get('time_published', '')
            try:
                dt = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                published_at = int(dt.timestamp())
            except:
                published_at = 0

            # Create article dict
            articles.append({
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'url': item.get('url', ''),
                'source': item.get('source', ''),
                'time_published': time_published,
                'published_at': published_at,
                'overall_sentiment_score': ticker_score,
                'overall_sentiment_label': ticker_label,
                'ticker_sentiment': ticker_sentiment_list,
                'relevance_score': ticker_relevance
            })

        logger.info(f"  AlphaVantage: Fetched {len(articles)} articles for {symbol}")
        return articles

    except Exception as e:
        logger.error(f"  AlphaVantage error for {symbol}: {e}")
        return []


def update_database_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Update news_articles table to support multi-source sentiment tracking

    Adds columns for:
    - newsapi_sentiment_score, newsapi_sentiment_label
    - alphavantage_sentiment_score, alphavantage_sentiment_label
    - sentiment_source (which source was selected)
    - selection_reason (why this sentiment was chosen)
    """
    try:
        # Check if new columns exist
        result = conn.execute("""
            SELECT COUNT(*) FROM pragma_table_info('news_articles')
            WHERE name = 'newsapi_sentiment_score'
        """).fetchone()

        if result[0] == 0:
            logger.info("Updating database schema for multi-source sentiment tracking...")

            # Add new columns
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS newsapi_sentiment_score DOUBLE
            """)
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS newsapi_sentiment_label VARCHAR
            """)
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS alphavantage_sentiment_score DOUBLE
            """)
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS alphavantage_sentiment_label VARCHAR
            """)
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS alphavantage_relevance DOUBLE
            """)
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS sentiment_source VARCHAR
            """)
            conn.execute("""
                ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS selection_reason VARCHAR
            """)

            logger.info("  Schema updated successfully")
        else:
            logger.info("  Schema already up to date")

    except Exception as e:
        logger.error(f"Error updating schema: {e}")


def store_multi_source_articles(conn: duckdb.DuckDBPyConnection,
                                  articles: List[MultiSourceArticle]) -> int:
    """
    Store multi-source articles in database

    Args:
        conn: DuckDB connection
        articles: List of MultiSourceArticle objects

    Returns:
        Number of articles stored
    """
    stored_count = 0

    for article in articles:
        try:
            # Convert to database format
            db_data = convert_to_db_format(article)

            # Insert with deduplication
            conn.execute("""
                INSERT INTO news_articles (
                    article_id, symbol, title, description, content, url,
                    source_name, published_at, fetched_at,
                    sentiment_score, sentiment_label, sentiment_source,
                    newsapi_sentiment_score, newsapi_sentiment_label,
                    positive_keywords, negative_keywords,
                    alphavantage_sentiment_score, alphavantage_sentiment_label,
                    alphavantage_relevance, selection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (article_id) DO UPDATE SET
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_label = EXCLUDED.sentiment_label,
                    sentiment_source = EXCLUDED.sentiment_source,
                    newsapi_sentiment_score = EXCLUDED.newsapi_sentiment_score,
                    newsapi_sentiment_label = EXCLUDED.newsapi_sentiment_label,
                    alphavantage_sentiment_score = EXCLUDED.alphavantage_sentiment_score,
                    alphavantage_sentiment_label = EXCLUDED.alphavantage_sentiment_label,
                    alphavantage_relevance = EXCLUDED.alphavantage_relevance,
                    selection_reason = EXCLUDED.selection_reason
            """, [
                db_data['article_id'], db_data['symbol'], db_data['title'],
                db_data['description'], db_data['content'], db_data['url'],
                db_data['source_name'], db_data['published_at'], db_data['fetched_at'],
                db_data['sentiment_score'], db_data['sentiment_label'], db_data['sentiment_source'],
                db_data['newsapi_sentiment_score'], db_data['newsapi_sentiment_label'],
                db_data['positive_keywords'], db_data['negative_keywords'],
                db_data['alphavantage_sentiment_score'], db_data['alphavantage_sentiment_label'],
                db_data['alphavantage_relevance'], db_data['selection_reason']
            ])

            stored_count += 1

        except Exception as e:
            logger.debug(f"  Error storing article: {e}")

    return stored_count


def run_multi_source_ingestion(newsapi_key: str, alphavantage_key: str,
                                 symbols: List[str], db_path: str) -> bool:
    """
    Run multi-source news ingestion with power-of-2-choices sentiment selection

    Args:
        newsapi_key: NewsAPI key
        alphavantage_key: AlphaVantage key
        symbols: List of symbols to fetch news for
        db_path: Path to DuckDB database

    Returns:
        True if successful
    """
    logger.info("Starting multi-source news ingestion...")
    logger.info("  Sources: NewsAPI (keyword sentiment) + AlphaVantage (AI sentiment)")
    logger.info("  Strategy: Power-of-2-Choices (select least extreme score)")

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        logger.info(f"  Date range: {from_date} to {to_date}")

        # Connect to database
        conn = duckdb.connect(str(db_path))

        # Update schema if needed
        update_database_schema(conn)

        total_articles = 0

        # Fetch news for each symbol
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")

            # Fetch from both sources
            newsapi_articles = fetch_newsapi_articles(newsapi_key, symbol, from_date, to_date)
            alphavantage_articles = fetch_alphavantage_articles(alphavantage_key, symbol, from_date)

            # Apply power-of-2-choices sentiment selection
            multi_source_articles = create_multi_source_articles(
                newsapi_articles, alphavantage_articles, symbol
            )

            # Store in database
            stored = store_multi_source_articles(conn, multi_source_articles)
            total_articles += stored

            logger.info(f"  Stored {stored} articles for {symbol}")

            # Rate limiting (be conservative)
            if i < len(symbols):
                import time
                time.sleep(2.0)  # 2 seconds between symbols

        conn.commit()
        conn.close()

        logger.info("=" * 80)
        logger.info(f"Multi-source ingestion complete!")
        logger.info(f"  Total articles stored: {total_articles}")
        logger.info(f"  Symbols processed: {len(symbols)}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Error during multi-source ingestion: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("BigBrotherAnalytics - Multi-Source News Ingestion")
    logger.info("Power-of-2-Choices Sentiment Selection")
    logger.info("=" * 80)

    # Load API keys
    newsapi_key, alphavantage_key = load_api_keys()

    # Check NewsAPI key
    if not newsapi_key:
        newsapi_key = os.getenv('NEWS_API_KEY')

    if not newsapi_key:
        logger.error("NewsAPI key not found!")
        logger.error("  Add to configs/api_keys.yaml or set NEWS_API_KEY environment variable")
        sys.exit(1)

    logger.info(f"NewsAPI key loaded: {newsapi_key[:10]}...")

    # Check AlphaVantage key
    if not alphavantage_key:
        alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY')

    if not alphavantage_key:
        logger.error("AlphaVantage API key not found!")
        logger.error("  Add to configs/api_keys.yaml or set ALPHAVANTAGE_API_KEY environment variable")
        sys.exit(1)

    logger.info(f"AlphaVantage key loaded: {alphavantage_key[:10]}...")

    # Database path
    db_path = str(BASE_DIR / 'data' / 'bigbrother.duckdb')

    # Get symbols from database
    symbols = get_traded_symbols(db_path, limit=10)

    if not symbols:
        logger.error("No symbols found in database")
        sys.exit(1)

    logger.info(f"Symbols to fetch: {', '.join(symbols)}")

    # Run multi-source ingestion
    success = run_multi_source_ingestion(newsapi_key, alphavantage_key, symbols, db_path)

    if not success:
        logger.error("Multi-source ingestion failed!")
        sys.exit(1)

    logger.info("Success!")


if __name__ == '__main__':
    main()
