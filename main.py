"""
Main script demonstrating crypto backtester usage - Enhanced HMA-WAE Strategy.
Updated with advanced strategy implementation.
"""
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# Import our modules
from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy, create_legacy_hma_wae_strategy
from src.backtesting import Backtester, calculate_metrics, format_metrics
from src.backtesting.metrics import save_performance_table, save_trades_table
from src.visualization import create_performance_report
from utils import setup_logger
from config import TIMEFRAMES

logger = setup_logger(__name__)

def main():
    """Run advanced HMA-WAE strategy backtest with configurable parameters."""
    print("🚀 Crypto Trading Backtester v1.1 - Advanced HMA-WAE Strategy")
    print("=" * 70)
    
    # ========================= EASY CONFIGURATION =========================
    # Change these values to customize your backtest
    SYMBOL = 'BTC-USD'          # Crypto symbol to test
    TIMEFRAME = '8h'            # Chart timeframe: 5m, 15m, 1h, 4h, 8h, 12h, 1d, 3d, 1w, 1m
    DATA_PERIOD = '3mo'          # How much historical data to pull
    
    # Advanced HMA-WAE Strategy Parameters
    STRATEGY_PARAMS = {
        'hma_length': 45,           # Hull Moving Average period
        'hma_mode': 'hma',          # HMA mode: 'hma', 'ehma', 'thma'
        'fast_length': 20,          # MACD fast EMA period
        'slow_length': 40,          # MACD slow EMA period
        'sensitivity': 150,         # WAE sensitivity multiplier
        'max_bars_lag': 3,          # Max bars after HMA flip for WAE confirmation
        'trade_direction': 'long'   # Trading direction: 'long', 'short', 'both'
    }
    
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
    print(f"   Strategy: Advanced HMA-WAE")
    print(f"   HMA Length: {STRATEGY_PARAMS['hma_length']}")
    print(f"   HMA Mode: {STRATEGY_PARAMS['hma_mode'].upper()}")
    print(f"   WAE Sensitivity: {STRATEGY_PARAMS['sensitivity']}")
    print(f"   Trade Direction: {STRATEGY_PARAMS['trade_direction'].title()}")
    
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
    
    # 3. Create advanced HMA-WAE strategy
    print(f"\n🎯 Creating Advanced HMA-WAE strategy...")
    print(f"   Parameters: {STRATEGY_PARAMS}")
    
    try:
        strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
        print(f"✅ Advanced strategy created: {strategy.name}")
    except Exception as e:
        print(f"⚠️  Advanced strategy failed, falling back to legacy version: {e}")
        # Fallback to legacy strategy
        strategy = create_legacy_hma_wae_strategy(
            hma_period=STRATEGY_PARAMS['hma_length'], 
            sensitivity=STRATEGY_PARAMS['sensitivity']
        )
        print(f"✅ Legacy strategy created: {strategy.name}")
    
    # 4. Run backtest
    print(f"\n⚡ Running backtest for {strategy.name}...")
    backtester = Backtester(
        initial_capital=10000,
        commission_rate=0.001,  # 0.1%
        slippage=0.0005        # 0.05%
    )
    
    # Generate signals
    print("🔄 Generating trading signals...")
    signals = strategy.backtest_prepare(prepared_data)
    signal_count = (signals != 0).sum()
    print(f"✅ Generated {signal_count} trading signals")
    
    # Run backtest
    print("🔄 Running backtest simulation...")
    result = backtester.run(prepared_data, signals)
    
    # Calculate metrics
    metrics = calculate_metrics(result)
    
    # 5. Display results
    print(f"\n📊 Performance Results for {strategy.name}")
    print("=" * 70)
    print(format_metrics(metrics))
    
    # Additional strategy-specific metrics
    if hasattr(strategy, 'get_parameters'):
        params = strategy.get_parameters()
        print(f"\n🎛️  Strategy Parameters:")
        print("=" * 30)
        for key, value in params.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
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
    
    # 9. Summary
    print(f"\n✅ Backtest complete!")
    print(f"Strategy: {strategy.name}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Return: {result.total_return:.2%}")
    
    if result.total_trades > 0:
        avg_trade_duration = sum([(t.exit_time - t.entry_time).total_seconds() / 3600 
                                 for t in result.trades]) / len(result.trades)
        print(f"Average Trade Duration: {avg_trade_duration:.1f} hours")
    
    # Show file outputs
    print(f"\n📁 Generated files:")
    print(f"   📊 Performance metrics: results/{strategy_name}_performance_metrics.csv")
    print(f"   📋 Trade details: results/{strategy_name}_trades_detailed.csv")
    print(f"   📈 Equity curve: results/{strategy_name}_equity.png")
    
    # Performance analysis
    print(f"\n🔍 Quick Analysis:")
    if result.total_return > 0:
        print(f"   ✅ Strategy was profitable (+{result.total_return:.2%})")
    else:
        print(f"   ❌ Strategy was unprofitable ({result.total_return:.2%})")
    
    if hasattr(result, 'win_rate') and result.win_rate > 0.5:
        print(f"   ✅ Good win rate ({result.win_rate:.1%})")
    elif hasattr(result, 'win_rate'):
        print(f"   ⚠️  Low win rate ({result.win_rate:.1%})")
    
    if hasattr(result, 'max_drawdown') and abs(result.max_drawdown) < 0.1:
        print(f"   ✅ Low drawdown ({result.max_drawdown:.2%})")
    elif hasattr(result, 'max_drawdown'):
        print(f"   ⚠️  High drawdown ({result.max_drawdown:.2%})")

def run_strategy_comparison():
    """Optional function to compare advanced vs legacy strategy"""
    print("\n🔄 Running Strategy Comparison...")
    print("=" * 50)
    
    # This could be extended to run both strategies and compare results
    print("💡 Tip: You can modify the STRATEGY_PARAMS to test different configurations:")
    print("   - hma_mode: 'hma' (standard), 'ehma' (exponential), 'thma' (triangular)")
    print("   - trade_direction: 'long' (buy only), 'short' (sell only), 'both'")
    print("   - sensitivity: Higher values = more sensitive to momentum")
    print("   - max_bars_lag: How long to wait for WAE confirmation after HMA flip")

if __name__ == "__main__":
    main()
    
    # Uncomment to see comparison tips
    # run_strategy_comparison()