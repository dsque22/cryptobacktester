"""
Main script demonstrating crypto backtester usage.
"""
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# Import our modules
from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_sma_strategy, create_rsi_strategy
from src.backtesting import Backtester, calculate_metrics, format_metrics
from src.visualization import create_performance_report
from utils import setup_logger

logger = setup_logger(__name__)

def main():
    """Run example backtest."""
    print("🚀 Crypto Trading Backtester v1.0")
    print("=" * 50)
    
    # 1. Fetch data
    print("\n📊 Fetching Bitcoin data...")
    data = fetch_crypto_data('BTC-USD', period='1y', interval='1d')
    
    if data is None or data.empty:
        print("❌ Failed to fetch data. Please check your internet connection.")
        return
    
    print(f"✅ Fetched {len(data)} days of data")
    print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # 2. Prepare data
    print("\n🔧 Preparing data...")
    prepared_data = prepare_data(data)
    print(f"✅ Data prepared with {len(prepared_data.columns)} features")
    
    # 3. Create strategies
    print("\n🎯 Creating strategies...")
    strategies = {
        'SMA_20_50': create_sma_strategy(20, 50),
        'RSI_14': create_rsi_strategy(14, 30, 70)
    }
    
    # 4. Run backtests
    print("\n⚡ Running backtests...")
    backtester = Backtester(
        initial_capital=10000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        
        # Generate signals
        signals = strategy.backtest_prepare(prepared_data)
        
        # Run backtest
        result = backtester.run(prepared_data, signals)
        results[name] = result
        
        # Calculate metrics
        metrics = calculate_metrics(result)
        
        # Display results
        print(format_metrics(metrics))
    
    # 5. Create visualizations
    print("\n📊 Creating performance reports...")
    for name, result in results.items():
        create_performance_report(result, strategy_name=name)
    
    print("\n✅ Backtest complete! Check the results folder for charts.")

if __name__ == "__main__":
    main()