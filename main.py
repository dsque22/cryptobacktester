"""
Main script for crypto backtester - Clean Production Version
Uses the fixed backtesting engine with HMA-WAE strategy.
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
    """Run HMA-WAE strategy backtest with configurable parameters."""
    print("🚀 Crypto Trading Backtester v1.2 - HMA-WAE Strategy")
    print("=" * 70)
    
    # ========================= CONFIGURATION =========================
    # Change these values to customize your backtest
    SYMBOL = 'BTC-USD'          # Crypto symbol to test
    TIMEFRAME = '8h'            # Chart timeframe: 5m, 15m, 1h, 4h, 8h, 12h, 1d, 3d, 1w, 1m
    DATA_PERIOD = '1y'         # How much historical data: 1mo, 3mo, 6mo, 1y, 2y, 3y
    
    # Parameter optimization (set to True to optimize strategy parameters)
    OPTIMIZE_PARAMETERS = False  # Set to True to run parameter optimization
    
    # HMA-WAE Strategy Parameters
    STRATEGY_PARAMS = {
        'hma_length': 45,           # Hull Moving Average period
        'hma_mode': 'hma',          # HMA mode: 'hma', 'ehma', 'thma'
        'fast_length': 20,          # MACD fast EMA period
        'slow_length': 40,          # MACD slow EMA period
        'sensitivity': 150,         # WAE sensitivity multiplier
        'bb_length': 20,            # Bollinger Bands period
        'bb_mult': 2.0,             # Bollinger Bands multiplier
        'dz_length': 20,            # Dead Zone length
        'dz_mult': 3,             # Dead Zone multiplier
        'max_bars_lag': 3,          # Max bars after HMA flip for WAE confirmation
        'trade_direction': 'long'   # Trading direction: 'long', 'short', 'both'
    }
    
    # Backtesting Parameters
    BACKTEST_PARAMS = {
        'initial_capital': 50000,   # Starting capital
        'commission_rate': 0.001,   # 0.1% commission
        'slippage': 0.0005         # 0.05% slippage
    }
    # ==================================================================
    
    # Validate configuration
    if not validate_config(SYMBOL, TIMEFRAME, DATA_PERIOD):
        return
    
    print_config(SYMBOL, TIMEFRAME, DATA_PERIOD, STRATEGY_PARAMS, BACKTEST_PARAMS)
    
    try:
        # 1. Fetch and prepare data
        print(f"\n📊 Fetching {SYMBOL} data ({TIMEFRAME}, {DATA_PERIOD})...")
        data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
        
        if data is None or data.empty:
            print("❌ Failed to fetch data. Please check your internet connection.")
            return
        
        print(f"✅ Fetched {len(data)} candles")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d %H:%M')} to {data.index[-1].strftime('%Y-%m-%d %H:%M')}")
        
        # 2. Prepare data with indicators
        print(f"\n🔧 Preparing data with technical indicators...")
        prepared_data = prepare_data(data)
        print(f"✅ Data prepared: {len(prepared_data)} rows with {len(prepared_data.columns)} features")
        
        # 3. Parameter optimization (if enabled)
        if OPTIMIZE_PARAMETERS:
            print(f"\n🔬 Running parameter optimization...")
            from src.optimization import ParameterOptimizer
            from src.strategy.strategies import HMAWAEStrategy
            
            # Define parameter ranges for optimization
            param_ranges = {
                'hma_length': range(20, 50, 10),           # [20, 30, 40]
                'wae_sensitivity': range(100, 200, 50),    # [100, 150]  
                'trade_direction': ['long', 'both']        # Test long-only and both
            }
            
            optimizer = ParameterOptimizer(
                strategy_class=HMAWAEStrategy,
                data=prepared_data,
                param_ranges=param_ranges,
                n_jobs=2
            )
            
            results = optimizer.optimize()
            
            print(f"\n📊 Optimization Results:")
            print(f"   🏆 Best Parameters: {results.best_params}")
            print(f"   📈 Best Sharpe: {results.best_sharpe:.3f}")
            print(f"   🎯 Deployment Score: {results.deployment_score:.3f}")
            print(f"   📋 Recommendation: {results.recommendation}")
            
            # Use optimized parameters
            STRATEGY_PARAMS.update(results.best_params)
            print(f"✅ Using optimized parameters for backtest")
        
        # 4. Create strategy
        print(f"\n🎯 Creating HMA-WAE strategy...")
        strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
        print(f"✅ Strategy created: {strategy.name}")
        print(f"📋 Trade direction: {STRATEGY_PARAMS['trade_direction'].title()}")
        
        # 5. Generate signals
        print(f"\n⚡ Generating trading signals...")
        signals = strategy.backtest_prepare(prepared_data)
        
        # Count signals
        long_signals = (signals == 1.0).sum()
        short_signals = (signals == -1.0).sum()
        exit_signals = (signals == 0.0).sum()
        
        print(f"✅ Signals generated:")
        print(f"   📈 Long entries: {long_signals}")
        print(f"   📉 Short entries: {short_signals}")
        print(f"   🔄 Exit/Hold: {exit_signals}")
        
        if long_signals + short_signals == 0:
            print("⚠️  No trading signals generated. Consider adjusting strategy parameters.")
            return
        
        # 6. Run backtest
        print(f"\n🔄 Running backtest...")
        backtester = Backtester(**BACKTEST_PARAMS)
        result = backtester.run(prepared_data, signals)
        
        # 7. Calculate and display results
        print(f"\n📊 Backtest Results")
        print("=" * 50)
        
        metrics = calculate_metrics(result)
        print(format_metrics(metrics))
        
        # Additional summary
        print(f"\n📋 Trade Summary:")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Winning Trades: {result.winning_trades}")
        print(f"   Losing Trades: {result.losing_trades}")
        
        if result.total_trades > 0:
            avg_duration = sum([(t.exit_time - t.entry_time).total_seconds() / 3600 
                               for t in result.trades]) / len(result.trades)
            print(f"   Average Duration: {avg_duration:.1f} hours")
            
            # Show individual trades
            print(f"\n📈 Trade Details:")
            for i, trade in enumerate(result.trades, 1):
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                print(f"   Trade {i}: {trade.side.upper()} | "
                      f"Entry: {trade.entry_time.strftime('%m-%d %H:%M')} | "
                      f"Exit: {trade.exit_time.strftime('%m-%d %H:%M')} | "
                      f"P&L: {trade.pnl_percent:.2%} | "
                      f"Duration: {duration:.1f}h")
        
        # 7. Export results
        strategy_name = f"{strategy.name}_{TIMEFRAME}_{DATA_PERIOD}"
        print(f"\n💾 Exporting results...")
        
        # Save performance metrics
        save_performance_table(result, strategy_name)
        
        # Save trade details
        save_trades_table(result, strategy_name)
        
        # Create equity curve chart
        create_performance_report(result, strategy_name=strategy_name)
        
        # 8. Final summary
        print(f"\n✅ Backtest Complete!")
        print(f"📊 Final Performance: {result.total_return:.2%}")
        print(f"🎯 Win Rate: {result.win_rate:.1%}")
        print(f"📁 Results saved to 'results/' folder")
        
        # Performance assessment
        assess_performance(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return None

def validate_config(symbol: str, timeframe: str, data_period: str) -> bool:
    """Validate configuration parameters."""
    # Validate timeframe
    if timeframe not in TIMEFRAMES:
        print(f"❌ Invalid timeframe: {timeframe}")
        print(f"Available timeframes: {list(TIMEFRAMES.keys())}")
        return False
    
    # Validate data period
    valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '3y']
    if data_period not in valid_periods:
        print(f"❌ Invalid data period: {data_period}")
        print(f"Available periods: {valid_periods}")
        return False
    
    return True

def print_config(symbol: str, timeframe: str, data_period: str, 
                strategy_params: dict, backtest_params: dict):
    """Print configuration summary."""
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data Period: {data_period}")
    print(f"   Initial Capital: ${backtest_params['initial_capital']:,}")
    print(f"   Commission: {backtest_params['commission_rate']:.3%}")
    print(f"   Slippage: {backtest_params['slippage']:.3%}")
    
    print(f"\n🎯 Strategy Parameters:")
    print(f"   HMA Length: {strategy_params['hma_length']}")
    print(f"   HMA Mode: {strategy_params['hma_mode'].upper()}")
    print(f"   WAE Sensitivity: {strategy_params['sensitivity']}")
    print(f"   Max Bars Lag: {strategy_params['max_bars_lag']}")
    print(f"   Trade Direction: {strategy_params['trade_direction'].title()}")

def assess_performance(result):
    """Provide performance assessment."""
    print(f"\n🔍 Performance Assessment:")
    
    if result.total_return > 0.05:  # > 5%
        print(f"   ✅ Excellent return: {result.total_return:.2%}")
    elif result.total_return > 0:
        print(f"   ✅ Profitable: {result.total_return:.2%}")
    else:
        print(f"   ❌ Loss: {result.total_return:.2%}")
    
    if hasattr(result, 'win_rate'):
        if result.win_rate > 0.6:  # > 60%
            print(f"   ✅ Strong win rate: {result.win_rate:.1%}")
        elif result.win_rate > 0.5:  # > 50%
            print(f"   ✅ Good win rate: {result.win_rate:.1%}")
        else:
            print(f"   ⚠️  Low win rate: {result.win_rate:.1%}")
    
    if result.total_trades < 5:
        print(f"   ⚠️  Few trades ({result.total_trades}). Consider longer data period or different parameters.")
    elif result.total_trades > 50:
        print(f"   ⚠️  Many trades ({result.total_trades}). Strategy may be overtrading.")
    else:
        print(f"   ✅ Good trade frequency: {result.total_trades} trades")

def quick_test():
    """Quick test function for development."""
    print("🧪 Running Quick Test...")
    print("=" * 30)
    
    # Quick test with smaller dataset
    test_params = {
        'symbol': 'BTC-USD',
        'timeframe': '4h',
        'data_period': '1mo',
        'strategy_params': {
            'hma_length': 21,
            'trade_direction': 'both',
            'sensitivity': 100
        }
    }
    
    print(f"Test config: {test_params['symbol']} {test_params['timeframe']} {test_params['data_period']}")
    # Implementation would go here for quick testing

if __name__ == "__main__":
    # Run main backtest
    main()
    
    # Uncomment below for quick testing during development
    # quick_test()