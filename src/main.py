"""
Main script demonstrating crypto backtester usage - Hull Strategy Focus.
"""
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# Import our modules
from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy
from src.backtesting import Backtester, calculate_metrics, format_metrics
from src.backtesting.metrics import save_performance_table, save_trades_table
from src.visualization import create_performance_report
from utils import setup_logger
from config import TIMEFRAMES, TIMEFRAME_PERIODS

logger = setup_logger(__name__)

def main():
    """Run Hull strategy backtest with configurable timeframe."""
    print("🚀 Crypto Trading Backtester v1.0 - Hull Strategy")
    print("=" * 60)
    
    # Configuration - EASILY CHANGEABLE
    SYMBOL = 'BTC-USD'
    TIMEFRAME = '8h'  # Change this to any timeframe: 5m, 15m, 1h, 4h, 8h, 12h, 1d, 3d, 1w, 1m
    
    # Validate timeframe
    if TIMEFRAME not in TIMEFRAMES:
        print(f"❌ Invalid timeframe: {TIMEFRAME}")
        print(f"Available timeframes: {list(TIMEFRAMES.keys())}")
        return
    
    # Get appropriate period for timeframe
    period = TIMEFRAME_PERIODS.get(TIMEFRAME, '1y')
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Timeframe: {TIMEFRAME}")
    print(f"   Period: {period}")
    
    # 1. Fetch data
    print(f"\n📊 Fetching {SYMBOL} data ({TIMEFRAME})...")
    data = fetch_crypto_data(SYMBOL, period=period, interval=TIMEFRAME)
    
    if data is None or data.empty:
        print("❌ Failed to fetch data. Please check your internet connection.")
        return
    
    print(f"✅ Fetched {len(data)} candles of data")
    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
    
    # 2. Prepare data
    print("\n🔧 Preparing data...")
    prepared_data = prepare_data(data)
    print(f"✅ Data prepared with {len(prepared_data.columns)} features")
    
    # 3. Create Hull strategy only
    print("\n🎯 Creating Hull Moving Average + WAE strategy...")
    strategy = create_hma_wae_strategy(hma_period=21, sensitivity=150)
    
    # 4. Run backtest
    print(f"\n⚡ Running backtest for {strategy.name}...")
    backtester = Backtester(
        initial_capital=10000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    # Generate signals
    signals = strategy.backtest_prepare(prepared_data)
    
    # Run backtest
    result = backtester.run(prepared_data, signals)
    
    # Calculate metrics
    metrics = calculate_metrics(result)
    
    # 5. Display results
    print(f"\n📊 Performance Results for {strategy.name}")
    print("=" * 60)
    print(format_metrics(metrics))
    
    # 6. Export performance metrics table
    print(f"\n📊 Exporting performance metrics...")
    save_performance_table(result, f"{strategy.name}_{TIMEFRAME}")
    
    # 7. Export trades table
    print(f"\n📋 Exporting trades details...")
    save_trades_table(result, f"{strategy.name}_{TIMEFRAME}")
    
    # 8. Create visualizations (equity curve only)
    print(f"\n📊 Creating equity curve...")
    create_performance_report(result, strategy_name=f"{strategy.name}_{TIMEFRAME}")
    
    print(f"\n✅ Backtest complete! Check the results folder for charts.")
    print(f"Strategy: {strategy.name}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Total Return: {result.total_return:.2%}")

if __name__ == "__main__":
    main()