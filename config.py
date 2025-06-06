"""
Simple configuration for crypto backtester v1.
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

# Available timeframes - UPDATED
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

# Default configuration for different timeframes
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
