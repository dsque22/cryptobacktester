"""
Debug script to trace exactly what happens in main.py execution
"""
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# Import our modules
from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy
from src.backtesting import Backtester

def debug_main_execution():
    """Debug the main.py execution step by step"""
    print("🔍 Main.py Execution Debug")
    print("=" * 50)
    
    # Exact same config as main.py
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
        'trade_direction': 'long'
    }
    
    print(f"📊 Config: {SYMBOL}, {TIMEFRAME}, {DATA_PERIOD}")
    print(f"🎯 Strategy params: {STRATEGY_PARAMS}")
    
    # 1. Fetch data (same as main.py)
    print(f"\n📊 Step 1: Fetching data...")
    data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    print(f"✅ Data shape: {data.shape}")
    
    # 2. Prepare data (same as main.py)
    print(f"\n🔧 Step 2: Preparing data...")
    prepared_data = prepare_data(data)
    print(f"✅ Prepared data shape: {prepared_data.shape}")
    
    # 3. Create strategy (same as main.py)
    print(f"\n🎯 Step 3: Creating strategy...")
    strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
    print(f"✅ Strategy: {strategy.name}")
    print(f"📋 Trade direction: {strategy.trade_direction}")
    
    # 4. Generate signals using backtest_prepare (same as main.py)
    print(f"\n⚡ Step 4: Generating signals via backtest_prepare...")
    signals = strategy.backtest_prepare(prepared_data)
    
    print(f"✅ Signals generated")
    print(f"📊 Signal shape: {signals.shape}")
    print(f"📊 Signal type: {type(signals)}")
    print(f"📊 Signal dtype: {signals.dtype}")
    
    # Analyze the signals from backtest_prepare
    signal_counts = signals.value_counts().sort_index()
    print(f"\n📈 Signal distribution from backtest_prepare:")
    for value, count in signal_counts.items():
        if value == 1.0:
            print(f"   LONG (1.0): {count}")
        elif value == -1.0:
            print(f"   SHORT (-1.0): {count}")
        elif value == 0.0:
            print(f"   HOLD/EXIT (0.0): {count}")
        else:
            print(f"   OTHER ({value}): {count}")
    
    # Show non-zero signals
    non_zero_signals = signals[signals != 0.0]
    print(f"\n🎯 Non-zero signals from backtest_prepare:")
    for date, signal in non_zero_signals.items():
        price = prepared_data.loc[date, 'close']
        signal_type = "LONG" if signal > 0 else "SHORT" if signal < 0 else "EXIT"
        print(f"   {date}: {signal_type} (signal={signal}) at ${price:.2f}")
    
    # 5. Create backtester (same as main.py)
    print(f"\n🔄 Step 5: Creating backtester...")
    backtester = Backtester(
        initial_capital=10000,
        commission_rate=0.001,
        slippage=0.0005
    )
    print(f"✅ Backtester created")
    
    # 6. Check what backtester receives
    print(f"\n🧪 Step 6: Checking backtester input...")
    print(f"Data shape passed to backtester: {prepared_data.shape}")
    print(f"Signals shape passed to backtester: {signals.shape}")
    print(f"Data index type: {type(prepared_data.index)}")
    print(f"Signals index type: {type(signals.index)}")
    print(f"Index alignment: {prepared_data.index.equals(signals.index)}")
    
    # Check if indices match
    if not prepared_data.index.equals(signals.index):
        print(f"⚠️ INDEX MISMATCH!")
        print(f"Data index range: {prepared_data.index[0]} to {prepared_data.index[-1]}")
        print(f"Signals index range: {signals.index[0]} to {signals.index[-1]}")
    
    # 7. Test direct signal generation (bypass backtest_prepare)
    print(f"\n🧪 Step 7: Testing direct generate_signals...")
    direct_signals = strategy.generate_signals(prepared_data)
    
    # Compare backtest_prepare vs direct generate_signals
    print(f"\n🔄 Comparing backtest_prepare vs generate_signals:")
    differences = (signals != direct_signals).sum()
    print(f"Differences: {differences}")
    
    if differences > 0:
        print(f"🚨 FOUND THE PROBLEM! backtest_prepare ≠ generate_signals")
        
        # Show where they differ
        diff_mask = signals != direct_signals
        diff_data = prepared_data[diff_mask]
        
        print(f"Differences found at {differences} locations:")
        for i, (date, _) in enumerate(diff_data.iterrows()):
            if i < 10:  # Show first 10 differences
                print(f"   {date}: backtest_prepare={signals[date]}, generate_signals={direct_signals[date]}")
    else:
        print(f"✅ Both methods return identical signals")
    
    # 8. Check base strategy class
    print(f"\n🔍 Step 8: Checking base strategy...")
    print(f"Strategy data attribute: {hasattr(strategy, 'data')}")
    print(f"Strategy signals attribute: {hasattr(strategy, 'signals')}")
    
    if hasattr(strategy, 'signals'):
        print(f"Strategy.signals type: {type(strategy.signals)}")
        if strategy.signals is not None:
            print(f"Strategy.signals shape: {strategy.signals.shape}")
            strategy_signal_counts = strategy.signals.value_counts().sort_index()
            print(f"Strategy.signals distribution:")
            for value, count in strategy_signal_counts.items():
                print(f"   {value}: {count}")
    
    # 9. Actually run the backtest to see what happens
    print(f"\n⚡ Step 9: Running actual backtest...")
    try:
        result = backtester.run(prepared_data, signals)
        print(f"✅ Backtest completed")
        print(f"📊 Total trades: {result.total_trades}")
        print(f"📊 Total return: {result.total_return:.2%}")
        
        # Check first few trades
        if result.trades:
            print(f"\n📋 First 5 trades:")
            for i, trade in enumerate(result.trades[:5]):
                print(f"   Trade {i+1}: {trade.side} from {trade.entry_time} to {trade.exit_time}")
                print(f"      Entry: ${trade.entry_price:.2f}, Exit: ${trade.exit_price:.2f}, P&L: ${trade.pnl:.2f}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
    
    return signals, prepared_data

if __name__ == "__main__":
    signals, data = debug_main_execution()