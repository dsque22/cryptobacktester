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