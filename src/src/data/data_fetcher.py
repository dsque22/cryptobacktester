"""
Data fetching module with Binance primary and Yahoo Finance backup.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict
import json
from pathlib import Path

from config import CACHE_DIR, CACHE_EXPIRY_HOURS
from utils import setup_logger

logger = setup_logger(__name__)

class DataFetcher:
    """Simple data fetcher with caching."""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
    
    def fetch(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch data with fallback mechanism.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data or None
        """
        # Check cache first
        cached_data = self._load_from_cache(symbol, interval)
        if cached_data is not None:
            logger.info(f"Loaded {symbol} from cache")
            return cached_data
        
        # Try Binance first (PRIMARY)
        logger.info(f"Fetching {symbol} from Binance...")
        data = self._fetch_binance(symbol, start_date, end_date, interval)
        
        if data is None or data.empty:
            # Try Yahoo Finance as fallback
            logger.info(f"Binance failed, trying Yahoo Finance...")
            data = self._fetch_yahoo(symbol, start_date, end_date, interval)
        
        if data is not None and not data.empty:
            # Save to cache
            self._save_to_cache(symbol, interval, data)
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
        
        logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    def _fetch_yahoo(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                return None
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return data[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return None
    
    def _fetch_binance(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Binance public API."""
        # Convert symbol format (BTC-USD -> BTCUSDT)
        binance_symbol = symbol.replace('-USD', 'USDT').replace('-', '')
        
        # Convert interval format - UPDATED WITH NEW TIMEFRAMES
        interval_map = {
            '5m': '5m',
            '15m': '15m', 
            '1h': '1h',
            '4h': '4h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'  # 1 month
        }
        
        if interval not in interval_map:
            logger.error(f"Unsupported interval for Binance: {interval}")
            return None
        
        try:
            # Binance API endpoint
            url = 'https://api.binance.com/api/v3/klines'
            
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            all_data = []
            current_start = start_ts
            
            while current_start < end_ts:
                params = {
                    'symbol': binance_symbol,
                    'interval': interval_map[interval],
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': 1000  # Max limit
                }
                
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    logger.error(f"Binance API error: {response.status_code}")
                    return None
                
                data = response.json()
                if not data:
                    break
                
                all_data.extend(data)
                
                # Update start time for next batch
                current_start = data[-1][0] + 1
            
            if not all_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                'buy_quote_volume', 'ignore'
            ])
            
            # Convert types and set index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return None
    
    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get cache file path."""
        safe_symbol = symbol.replace('/', '_').replace('-', '_')
        return self.cache_dir / f"{safe_symbol}_{interval}.parquet"
    
    def _load_from_cache(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(symbol, interval)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Cache expired for {symbol}")
            return None
        
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, interval: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol, interval)
        try:
            data.to_parquet(cache_path)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

# Convenience function
def fetch_crypto_data(symbol: str, period: str = '2y', interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Simple function to fetch crypto data.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC-USD')
        period: Time period (1y, 2y, etc.)
        interval: Time interval (5m, 15m, 1h, 4h, 8h, 12h, 1d, 3d, 1w, 1M)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Calculate dates
    end_date = datetime.now()
    
    period_map = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '3y': 1095
    }
    
    days = period_map.get(period, 365)
    start_date = end_date - timedelta(days=days)
    
    fetcher = DataFetcher()
    return fetcher.fetch(
        symbol,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        interval
    )