"""
Utility functions for the backtester.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

def setup_logger(name: str) -> logging.Logger:
    """Set up a simple logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that dataframe has required columns."""
    return all(col in df.columns for col in required_columns)

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from price series."""
    return prices.pct_change()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio (annualized for crypto - 365 days)."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 365  # Daily risk-free rate for crypto
    if returns.std() == 0:
        return 0.0
    
    return np.sqrt(365) * excess_returns.mean() / returns.std()  # Use 365 for crypto

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve / peak) - 1
    return drawdown.min()

def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.2f}%"

def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"