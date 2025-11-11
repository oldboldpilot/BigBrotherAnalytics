#!/usr/bin/env python3
"""
BigBrotherAnalytics: Setup News Database Schema
Creates the news articles table and related structures in DuckDB

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: News Ingestion System
"""

import sys
from pathlib import Path
import logging

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import duckdb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_news_schema(db_path: str) -> bool:
    """
    Set up news database schema.

    Creates:
    - news_articles table: stores news articles with sentiment
    - indexes on symbol and published_at for fast queries

    Args:
        db_path: Path to DuckDB database

    Returns:
        True if successful
    """
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")

        # Create news_articles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                article_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                description TEXT,
                content TEXT,
                url VARCHAR,
                source_name VARCHAR,
                source_id VARCHAR,
                author VARCHAR,
                published_at TIMESTAMP NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sentiment_score DOUBLE,
                sentiment_label VARCHAR,
                positive_keywords TEXT[],
                negative_keywords TEXT[]
            )
        """)
        logger.info("Created/verified news_articles table")

        # Create indexes for performance
        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_symbol
                ON news_articles(symbol)
            """)
            logger.info("Created index on symbol")
        except Exception as e:
            logger.debug(f"Index on symbol may already exist: {e}")

        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_published
                ON news_articles(published_at DESC)
            """)
            logger.info("Created index on published_at")
        except Exception as e:
            logger.debug(f"Index on published_at may already exist: {e}")

        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_sentiment
                ON news_articles(sentiment_label, sentiment_score)
            """)
            logger.info("Created index on sentiment")
        except Exception as e:
            logger.debug(f"Index on sentiment may already exist: {e}")

        conn.commit()

        # Verify table created
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'news_articles'
        """).fetchall()

        if tables:
            logger.info(f"Table verified: {tables[0][0]}")

            # Show table structure
            columns = conn.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'news_articles'
                ORDER BY ordinal_position
            """).fetchall()

            logger.info("\nTable structure:")
            for col, dtype in columns:
                logger.info(f"  - {col}: {dtype}")

        # Check if we have any news data
        count = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        logger.info(f"\nCurrent news articles in database: {count}")

        conn.close()
        logger.info("\nNews database schema created successfully!")
        return True

    except Exception as e:
        logger.error(f"Error setting up news schema: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Setup news database schema')
    parser.add_argument(
        '--db',
        default=str(BASE_DIR / 'data' / 'bigbrother.duckdb'),
        help='Path to DuckDB database'
    )

    args = parser.parse_args()

    success = setup_news_schema(args.db)
    sys.exit(0 if success else 1)
