"""
Trading strategy implementations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

from .base_strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"SMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on SMA crossover."""
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Calculate SMAs
        sma_fast = data['close'].rolling(window=self.fast_period).mean()
        sma_slow = data['close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        signals[sma_fast > sma_slow] = 1.0   # Buy signal
        signals[sma_fast < sma_slow] = -1.0  # Sell signal
        
        # Only trade on crossovers (not every bar)
        signals = signals.diff()
        signals[signals > 0] = 1.0   # Buy on upward crossover
        signals[signals < 0] = -1.0  # Sell on downward crossover
        signals[signals == 0] = 0.0  # Hold otherwise
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }

class RSIStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy."""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI levels."""
        signals = pd.Series(0, index=data.index, dtype=float)

        if 'rsi' not in data.columns:
            # Calculate RSI if not present
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = data['rsi']

        # Generate signals - Fixed logic
        buy_signals = rsi <= self.oversold
        sell_signals = rsi >= self.overbought
        
        # Only signal on transitions (not sustained levels)
        buy_transitions = buy_signals & ~buy_signals.shift(1, fill_value=False)
        sell_transitions = sell_signals & ~sell_signals.shift(1, fill_value=False)
        
        signals[buy_transitions] = 1.0   # Buy when crossing into oversold
        signals[sell_transitions] = -1.0  # Sell when crossing into overbought

        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought
        }

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands."""
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Use pre-calculated BB if available
        if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            upper = data['bb_upper']
            lower = data['bb_lower']
            middle = data['bb_middle']
        else:
            # Calculate Bollinger Bands
            middle = data['close'].rolling(window=self.period).mean()
            std = data['close'].rolling(window=self.period).std()
            upper = middle + (std * self.std_dev)
            lower = middle - (std * self.std_dev)
        
        close = data['close']
        
        # Mean reversion signals
        # Buy when price touches lower band
        signals[close <= lower] = 1.0
        
        # Sell when price touches upper band
        signals[close >= upper] = -1.0
        
        # Exit positions when price crosses middle band
        long_exit = (close > middle) & (close.shift(1) <= middle)
        short_exit = (close < middle) & (close.shift(1) >= middle)
        
        signals[long_exit | short_exit] = 0.0
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'std_dev': self.std_dev
        }

class HMAWAEStrategy(BaseStrategy):
    """Hull Moving Average with Waddah Attar Explosion Strategy."""
    
    def __init__(self, hma_period: int = 21, sensitivity: float = 150):
        super().__init__(f"HMA_WAE_{hma_period}")
        self.hma_period = hma_period
        self.sensitivity = sensitivity
    
    def _wma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def _hma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self._wma(series, half_period)
        wma_full = self._wma(series, period)
        
        raw_hma = 2 * wma_half - wma_full
        return self._wma(raw_hma, sqrt_period)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on HMA direction and WAE momentum."""
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Calculate HMA
        hma = self._hma(data['close'], self.hma_period)
        hma_direction = hma.diff()
        
        # Simple momentum filter (simplified WAE)
        momentum = data['close'].pct_change(5).abs() * 100
        strong_momentum = momentum > (momentum.rolling(20).mean() * 1.5)
        
        # Generate signals
        # Buy: HMA turning up with strong momentum
        buy_signal = (hma_direction > 0) & (hma_direction.shift(1) <= 0) & strong_momentum
        signals[buy_signal] = 1.0
        
        # Sell: HMA turning down with strong momentum
        sell_signal = (hma_direction < 0) & (hma_direction.shift(1) >= 0) & strong_momentum
        signals[sell_signal] = -1.0
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'hma_period': self.hma_period,
            'sensitivity': self.sensitivity
        }

# Convenience functions
def create_sma_strategy(fast: int = 20, slow: int = 50) -> SMAStrategy:
    """Create SMA crossover strategy."""
    return SMAStrategy(fast, slow)

def create_rsi_strategy(period: int = 14, oversold: float = 30, overbought: float = 70) -> RSIStrategy:
    """Create RSI mean reversion strategy."""
    return RSIStrategy(period, oversold, overbought)

def create_bb_strategy(period: int = 20, std_dev: float = 2.0) -> BollingerBandsStrategy:
    """Create Bollinger Bands strategy."""
    return BollingerBandsStrategy(period, std_dev)

def create_hma_wae_strategy(hma_period: int = 21, sensitivity: float = 150) -> HMAWAEStrategy:
    """Create HMA-WAE strategy."""
    return HMAWAEStrategy(hma_period, sensitivity)