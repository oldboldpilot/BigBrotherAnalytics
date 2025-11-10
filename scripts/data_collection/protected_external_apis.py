"""
Protected External APIs with Circuit Breaker

Wraps BLS, FRED, and other external APIs with circuit breaker protection.
Provides cached data fallback when circuits are open.

Usage:
    from protected_external_apis import ProtectedFREDAPI

    fred = ProtectedFREDAPI(api_key="your_key")
    data = fred.get_jobless_claims_with_fallback()
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import duckdb

from circuit_breaker_wrapper import CircuitBreakerWrapper

logger = logging.getLogger(__name__)


class ProtectedFREDAPI:
    """
    Protected FRED API Client with Circuit Breaker

    Wraps Federal Reserve Economic Data API calls with circuit breaker
    protection and provides cached data fallback.
    """

    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize protected FRED API client

        Args:
            api_key: FRED API key
            failure_threshold: Circuit breaker failure threshold
            timeout_seconds: Circuit breaker timeout in seconds
            cache_dir: Directory for cached data (default: data/cache)
        """
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreakerWrapper(
            name="FRED_API",
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
            enable_logging=True,
        )

        logger.info("Protected FRED API client initialized")

    def get_series_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get FRED series data with circuit breaker protection

        Args:
            series_id: FRED series ID (e.g., 'ICSA' for jobless claims)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with series data

        Raises:
            Exception: If circuit is open or API call fails
        """
        def fetch():
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
            }

            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date

            response = requests.get(self.FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Cache successful response
            self._cache_data(f"fred_{series_id}", data)

            logger.info(f"Fetched FRED series: {series_id}")
            return data

        return self.circuit_breaker.call(fetch)

    def get_series_data_with_fallback(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get FRED series data with cached fallback

        Returns cached data if circuit is open.
        """
        try:
            return self.get_series_data(series_id, start_date, end_date)
        except Exception as e:
            if self.circuit_breaker.is_open:
                logger.warning(f"FRED API circuit open - using cached data for {series_id}")
                cached = self._get_cached_data(f"fred_{series_id}")
                if cached:
                    return cached
                else:
                    logger.error(f"No cached data available for {series_id}")
            raise

    def get_jobless_claims(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get jobless claims data (initial and continued claims)

        Returns:
            Dictionary with 'initial_claims' and 'continued_claims' data
        """
        result = {}

        try:
            # Get initial claims (ICSA)
            initial = self.get_series_data('ICSA', start_date, end_date)
            result['initial_claims'] = initial.get('observations', [])

            # Get continued claims (CCSA)
            continued = self.get_series_data('CCSA', start_date, end_date)
            result['continued_claims'] = continued.get('observations', [])

            logger.info("Fetched jobless claims data")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch jobless claims: {e}")
            raise

    def get_jobless_claims_with_fallback(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get jobless claims data with cached fallback
        """
        try:
            return self.get_jobless_claims(start_date, end_date)
        except Exception as e:
            if self.circuit_breaker.is_open:
                logger.warning("FRED API circuit open - using cached jobless claims")
                cached = self._get_cached_data("fred_jobless_claims")
                if cached:
                    return cached
                else:
                    logger.error("No cached jobless claims data available")
            raise

    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data to disk"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f, indent=2)
            logger.debug(f"Cached data: {key}")
        except Exception as e:
            logger.error(f"Failed to cache data {key}: {e}")

    def _get_cached_data(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """Get cached data from disk"""
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check cache age
            cache_time = datetime.fromisoformat(cached['timestamp'])
            age = datetime.now() - cache_time

            if age.total_seconds() / 3600 > max_age_hours:
                logger.warning(f"Cached data {key} is stale ({age.total_seconds() / 3600:.1f}h old)")
                # Return stale data anyway if circuit is open
                if self.circuit_breaker.is_open:
                    logger.warning(f"Using stale cached data for {key}")
                    return cached['data']
                return None

            logger.info(f"Using cached data: {key} (age: {age.total_seconds() / 3600:.1f}h)")
            return cached['data']

        except Exception as e:
            logger.error(f"Failed to read cached data {key}: {e}")
            return None

    def reset_circuit(self) -> None:
        """Manually reset circuit breaker"""
        self.circuit_breaker.reset()
        logger.info("FRED API circuit breaker manually reset")

    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        stats = self.circuit_breaker.get_stats()
        return {
            'state': stats.state.value,
            'total_calls': stats.total_calls,
            'success_count': stats.success_count,
            'failure_count': stats.failure_count,
            'consecutive_failures': stats.consecutive_failures,
            'success_rate': stats.success_rate,
            'last_error': stats.last_error,
        }


class ProtectedBLSAPI:
    """
    Protected BLS API Client with Circuit Breaker

    Wraps Bureau of Labor Statistics API calls with circuit breaker
    protection and provides cached data fallback.
    """

    BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize protected BLS API client

        Args:
            api_key: BLS API key (optional, increases rate limits)
            failure_threshold: Circuit breaker failure threshold
            timeout_seconds: Circuit breaker timeout in seconds
            cache_dir: Directory for cached data
        """
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreakerWrapper(
            name="BLS_API",
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
            enable_logging=True,
        )

        logger.info("Protected BLS API client initialized")

    def get_series_data(
        self,
        series_ids: List[str],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get BLS series data with circuit breaker protection

        Args:
            series_ids: List of BLS series IDs
            start_year: Start year (YYYY)
            end_year: End year (YYYY)

        Returns:
            Dictionary with series data
        """
        def fetch():
            payload = {
                'seriesid': series_ids,
            }

            if self.api_key:
                payload['registrationkey'] = self.api_key

            if start_year and end_year:
                payload['startyear'] = str(start_year)
                payload['endyear'] = str(end_year)

            headers = {'Content-type': 'application/json'}
            response = requests.post(
                self.BLS_BASE_URL,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Cache successful response
            cache_key = f"bls_{'_'.join(series_ids)}"
            self._cache_data(cache_key, data)

            logger.info(f"Fetched BLS series: {series_ids}")
            return data

        return self.circuit_breaker.call(fetch)

    def get_series_data_with_fallback(
        self,
        series_ids: List[str],
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get BLS series data with cached fallback"""
        try:
            return self.get_series_data(series_ids, start_year, end_year)
        except Exception as e:
            if self.circuit_breaker.is_open:
                cache_key = f"bls_{'_'.join(series_ids)}"
                logger.warning(f"BLS API circuit open - using cached data for {cache_key}")
                cached = self._get_cached_data(cache_key)
                if cached:
                    return cached
                else:
                    logger.error(f"No cached data available for {cache_key}")
            raise

    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data to disk"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f, indent=2)
            logger.debug(f"Cached data: {key}")
        except Exception as e:
            logger.error(f"Failed to cache data {key}: {e}")

    def _get_cached_data(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """Get cached data from disk"""
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check cache age
            cache_time = datetime.fromisoformat(cached['timestamp'])
            age = datetime.now() - cache_time

            if age.total_seconds() / 3600 > max_age_hours:
                logger.warning(f"Cached data {key} is stale ({age.total_seconds() / 3600:.1f}h old)")
                # Return stale data anyway if circuit is open
                if self.circuit_breaker.is_open:
                    logger.warning(f"Using stale cached data for {key}")
                    return cached['data']
                return None

            logger.info(f"Using cached data: {key} (age: {age.total_seconds() / 3600:.1f}h)")
            return cached['data']

        except Exception as e:
            logger.error(f"Failed to read cached data {key}: {e}")
            return None

    def reset_circuit(self) -> None:
        """Manually reset circuit breaker"""
        self.circuit_breaker.reset()
        logger.info("BLS API circuit breaker manually reset")

    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        stats = self.circuit_breaker.get_stats()
        return {
            'state': stats.state.value,
            'total_calls': stats.total_calls,
            'success_count': stats.success_count,
            'failure_count': stats.failure_count,
            'consecutive_failures': stats.consecutive_failures,
            'success_rate': stats.success_rate,
            'last_error': stats.last_error,
        }
