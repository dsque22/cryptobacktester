"""
Simple configuration for crypto backtester v1.
Enhanced with better data period management.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories with parents
for dir_path in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Trading parameters
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION_RATE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005       # 0.05%

# Data settings
DEFAULT_SYMBOL = 'BTC-USD'
DEFAULT_TIMEFRAME = '1d'
CACHE_EXPIRY_HOURS = 24

# Available symbols
SYMBOLS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BNB': 'BNB-USD',
    'ADA': 'ADA-USD',
    'SOL': 'SOL-USD',
    'XRP': 'XRP-USD',
    'DOT': 'DOT-USD',
    'DOGE': 'DOGE-USD'
}

# Available timeframes
TIMEFRAMES = {
    '5m': '5m',
    '15m': '15m', 
    '1h': '1h',
    '4h': '4h',
    '8h': '8h',
    '12h': '12h',
    '1d': '1d',
    '3d': '3d',
    '1w': '1w',
    '1m': '1M'  # Note: Binance uses '1M' for 1 month
}

# Available data periods
DATA_PERIODS = {
    '1mo': 30,     # 1 month
    '3mo': 90,     # 3 months
    '6mo': 180,    # 6 months
    '1y': 365,     # 1 year
    '2y': 730,     # 2 years
    '3y': 1095     # 3 years
}

# Recommended combinations of timeframe and data period
# This helps users understand what makes sense for different timeframes
TIMEFRAME_RECOMMENDATIONS = {
    '5m': {
        'recommended_periods': ['1mo', '3mo'],
        'note': 'High frequency data - shorter periods recommended to avoid excessive data'
    },
    '15m': {
        'recommended_periods': ['1mo', '3mo', '6mo'],
        'note': 'Good for intraday strategies'
    },
    '1h': {
        'recommended_periods': ['3mo', '6mo', '1y'],
        'note': 'Balanced for short to medium-term strategies'
    },
    '4h': {
        'recommended_periods': ['6mo', '1y', '2y'],
        'note': 'Good for swing trading strategies'
    },
    '8h': {
        'recommended_periods': ['6mo', '1y', '2y'],
        'note': 'Medium-term trend following'
    },
    '12h': {
        'recommended_periods': ['1y', '2y', '3y'],
        'note': 'Long-term trend analysis'
    },
    '1d': {
        'recommended_periods': ['1y', '2y', '3y'],
        'note': 'Most common for strategy development'
    },
    '3d': {
        'recommended_periods': ['2y', '3y'],
        'note': 'Long-term position strategies'
    },
    '1w': {
        'recommended_periods': ['2y', '3y'],
        'note': 'Very long-term analysis'
    },
    '1m': {
        'recommended_periods': ['3y'],
        'note': 'Macro trend analysis only'
    }
}

# Legacy support - keeping this for backward compatibility but not using in main.py
TIMEFRAME_PERIODS = {
    '5m': '7d',    # 7 days for 5min data
    '15m': '30d',  # 30 days for 15min data
    '1h': '90d',   # 90 days for 1hr data
    '4h': '1y',    # 1 year for 4hr data
    '8h': '1y',    # 1 year for 8hr data
    '12h': '2y',   # 2 years for 12hr data
    '1d': '2y',    # 2 years for daily data
    '3d': '3y',    # 3 years for 3-day data
    '1w': '3y',    # 3 years for weekly data
    '1m': '3y'     # 3 years for monthly data
}