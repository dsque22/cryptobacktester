"""
Base strategy class for all trading strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str = "Strategy"):
        self.name = name
        self.data = None
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data: DataFrame with OHLCV and indicators
            
        Returns:
            Series with signals: 1 = buy, -1 = sell, 0 = hold
        """
        pass
    
    def backtest_prepare(self, data: pd.DataFrame):
        """Prepare strategy for backtesting."""
        self.data = data.copy()
        self.signals = self.generate_signals(data)
        return self.signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {}