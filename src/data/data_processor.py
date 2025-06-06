"""
Data preprocessing and technical indicator calculation.
"""
import pandas as pd
import numpy as np
import ta
import ta.trend
import ta.momentum
import ta.volatility
from typing import Optional, List

from utils import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    """Process and prepare data for backtesting."""
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Remove any NaN values
        df = df.dropna()
        
        # Ensure proper column types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid price data
        df = df[(df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close']) &
                (df['close'] > 0)]
        
        logger.info(f"Cleaned data: {len(data)} -> {len(df)} rows")
        return df
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        df = data.copy()
        
        if len(df) < 50:
            logger.warning("Insufficient data for indicators")
            return df
        
        try:  # ADD ERROR HANDLING
            # Simple Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # ATR (for volatility)
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return original data if indicator calculation fails
            return data
        
        return df

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add return calculations."""
        df = data.copy()
        
        # Simple returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Cumulative returns
        df['cum_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Complete data preparation pipeline."""
    processor = DataProcessor()
    
    # Clean data
    df = processor.clean_data(data)
    
    # Add indicators
    df = processor.add_indicators(df)
    
    # Calculate returns
    df = processor.calculate_returns(df)
    
    # Drop any remaining NaN values from indicators
    df = df.dropna()
    
    return df