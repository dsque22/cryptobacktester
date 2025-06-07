"""
Test the strategy with the FIXED backtesting engine
"""
import warnings
warnings.filterwarnings('ignore')

from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy
from src.backtesting import Backtester
from src.backtesting.metrics import calculate_metrics, format_metrics
from utils import setup_logger

logger = setup_logger(__name__)

def test_fixed_backtester():
    """Test with the fixed backtesting engine"""
    print("🔧 Testing FIXED Backtesting Engine")
    print("=" * 50)
    
    # Same config as main.py
    SYMBOL = 'BTC-USD'
    TIMEFRAME = '8h'
    DATA_PERIOD = '3mo'
    
    STRATEGY_PARAMS = {
        'hma_length': 45,
        'hma_mode': 'hma',
        'fast_length': 20,
        'slow_length': 40,
        'sensitivity': 150,
        'bb_length': 20,
        'bb_mult': 2.0,
        'dz_length': 20,
        'dz_mult': 3.7,
        'max_bars_lag': 3,
        'trade_direction': 'long'  # LONG ONLY
    }
    
    print(f"📊 Config: {SYMBOL}, {TIMEFRAME}, {DATA_PERIOD}")
    print(f"🎯 Direction: {STRATEGY_PARAMS['trade_direction']}")
    
    # Get data and signals
    print(f"\n📊 Fetching data...")
    data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    if data is None or data.empty:
        print("❌ Failed to fetch data")
        return None
        
    prepared_data = prepare_data(data)
    print(f"✅ Data prepared: {len(prepared_data)} rows")
    
    print(f"📊 Creating strategy...")
    strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
    print(f"✅ Strategy created: {strategy.name}")
    
    print(f"📊 Generating signals...")
    signals = strategy.backtest_prepare(prepared_data)
    
    # Show our signals
    print(f"\n🎯 Our signals:")
    signal_counts = signals.value_counts().sort_index()
    for value, count in signal_counts.items():
        if value == 1.0:
            print(f"   LONG entries: {count}")
        elif value == -1.0:
            print(f"   SHORT entries: {count}")
        elif value == 0.0:
            print(f"   HOLD/EXIT: {count}")
    
    non_zero = signals[signals != 0]
    print(f"\n🎯 Signal events ({len(non_zero)} total):")
    for i, (date, signal) in enumerate(non_zero.items()):
        if i < 10:  # Show first 10
            signal_type = "LONG" if signal > 0 else "SHORT" 
            price = prepared_data.loc[date, 'close']
            print(f"   {date}: {signal_type} at ${price:.2f}")
        elif i == 10:
            print(f"   ... and {len(non_zero) - 10} more signals")
            break
    
    # Test with FIXED backtester
    print(f"\n⚡ Running FIXED backtester...")
    
    backtester = Backtester(
        initial_capital=10000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    try:
        result = backtester.run(prepared_data, signals)
        
        # Show results
        print(f"\n✅ FIXED BACKTESTER RESULTS:")
        print(f"📊 Total trades: {result.total_trades}")
        print(f"📊 Total return: {result.total_return:.2%}")
        print(f"📊 Winning trades: {result.winning_trades}")
        print(f"📊 Losing trades: {result.losing_trades}")
        
        if result.total_trades > 0:
            print(f"📊 Win rate: {result.win_rate:.2%}")
        
        if result.trades:
            print(f"\n📋 First 5 Trade Details:")
            for i, trade in enumerate(result.trades[:5], 1):
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                print(f"   Trade {i}: {trade.side.upper()}")
                print(f"      Entry: {trade.entry_time} at ${trade.entry_price:.2f}")
                print(f"      Exit:  {trade.exit_time} at ${trade.exit_price:.2f}")
                print(f"      P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2%})")
                print(f"      Duration: {duration:.1f} hours")
                print()
        
        # Compare with our signals
        print(f"\n🔄 Analysis:")
        print(f"   Our entry signals: {len(non_zero)} signals")
        print(f"   Backtester trades: {result.total_trades} trades")
        
        if result.total_trades == len(non_zero):
            print(f"   ✅ PERFECT! Trades exactly match our signals")
        elif result.total_trades > 0:
            ratio = result.total_trades / len(non_zero) if len(non_zero) > 0 else 0
            print(f"   📊 Trade to signal ratio: {ratio:.2f}")
            if ratio < 1:
                print(f"   ⚠️ Some signals weren't converted to trades")
            elif ratio > 1:
                print(f"   ⚠️ More trades than signals (unexpected)")
        else:
            print(f"   ❌ No trades generated despite having signals")
            
        return result
        
    except Exception as e:
        print(f"❌ Backtester failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_broken_results():
    """Compare with the broken backtester results"""
    print(f"\n📊 Comparison with Broken Results:")
    print(f"=" * 40)
    print(f"Broken backtester (from CSV):")
    print(f"   Total trades: 110")
    print(f"   Total return: -5.96%")
    print(f"   Win rate: 44.55%")
    print(f"   Avg trade duration: 8.0 hours")
    print(f"   Problem: Ignored our signals, made its own trades")
    print()
    print(f"Our signals should generate far fewer, more strategic trades")
    print(f"based on HMA flips + WAE momentum confirmation")

if __name__ == "__main__":
    result = test_fixed_backtester()
    compare_with_broken_results()