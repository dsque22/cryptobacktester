"""
Main script demonstrating crypto backtester usage - Hull Strategy Focus.
Enhanced with easy data period selection.
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
from config import TIMEFRAMES

logger = setup_logger(__name__)

def main():
    """Run Hull strategy backtest with configurable timeframe and data period."""
    print("🚀 Crypto Trading Backtester v1.0 - Hull Strategy")
    print("=" * 60)
    
    # ========================= EASY CONFIGURATION =========================
    # Change these values to customize your backtest
    SYMBOL = 'BTC-USD'          # Crypto symbol to test
    TIMEFRAME = '8h'            # Chart timeframe: 5m, 15m, 1h, 4h, 8h, 12h, 1d, 3d, 1w, 1m
    DATA_PERIOD = '3mo'         # How much historical data to pull
    
    # Available DATA_PERIOD options:
    # '1mo'  = 1 month     | '3mo'  = 3 months    | '6mo'  = 6 months
    # '1y'   = 1 year      | '2y'   = 2 years     | '3y'   = 3 years
    # ======================================================================
    
    # Validate timeframe
    if TIMEFRAME not in TIMEFRAMES:
        print(f"❌ Invalid timeframe: {TIMEFRAME}")
        print(f"Available timeframes: {list(TIMEFRAMES.keys())}")
        return
    
    # Validate data period
    valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '3y']
    if DATA_PERIOD not in valid_periods:
        print(f"❌ Invalid data period: {DATA_PERIOD}")
        print(f"Available periods: {valid_periods}")
        return
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Timeframe: {TIMEFRAME}")
    print(f"   Data Period: {DATA_PERIOD}")
    
    # Provide recommendations based on timeframe
    recommended_periods = {
        '5m': ['1mo', '3mo'],
        '15m': ['1mo', '3mo', '6mo'],
        '1h': ['3mo', '6mo', '1y'],
        '4h': ['6mo', '1y', '2y'],
        '8h': ['6mo', '1y', '2y'],
        '12h': ['1y', '2y', '3y'],
        '1d': ['1y', '2y', '3y'],
        '3d': ['2y', '3y'],
        '1w': ['2y', '3y'],
        '1m': ['3y']
    }
    
    if DATA_PERIOD not in recommended_periods.get(TIMEFRAME, []):
        recommended = recommended_periods.get(TIMEFRAME, ['1y'])
        print(f"⚠️  Note: For {TIMEFRAME} timeframe, recommended periods are: {recommended}")
        print(f"   You selected {DATA_PERIOD} which may provide limited or excessive data.")
    
    # 1. Fetch data
    print(f"\n📊 Fetching {SYMBOL} data ({TIMEFRAME}, {DATA_PERIOD})...")
    data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    
    if data is None or data.empty:
        print("❌ Failed to fetch data. Please check your internet connection.")
        return
    
    print(f"✅ Fetched {len(data)} candles of data")
    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
    
    # Show data summary
    days_covered = (data.index[-1] - data.index[0]).days
    print(f"📈 Data summary: {days_covered} days covered")
    
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
    strategy_name = f"{strategy.name}_{TIMEFRAME}_{DATA_PERIOD}"
    save_performance_table(result, strategy_name)
    
    # 7. Export trades table
    print(f"\n📋 Exporting trades details...")
    save_trades_table(result, strategy_name)
    
    # 8. Create visualizations (equity curve only)
    print(f"\n📊 Creating equity curve...")
    create_performance_report(result, strategy_name=strategy_name)
    
    print(f"\n✅ Backtest complete! Check the results folder for charts.")
    print(f"Strategy: {strategy.name}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Total Return: {result.total_return:.2%}")
    
    # Show file outputs
    print(f"\n📁 Generated files:")
    print(f"   📊 Performance metrics: results/{strategy_name}_performance_metrics.csv")
    print(f"   📋 Trade details: results/{strategy_name}_trades_detailed.csv")
    print(f"   📈 Equity curve: results/{strategy_name}_equity.png")

if __name__ == "__main__":
    main()