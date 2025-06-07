"""
Debug the backtesting engine to see why it's ignoring our signals
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy
from src.backtesting import Backtester

def debug_backtesting_engine():
    """Debug what the backtesting engine is actually doing"""
    print("🔍 Backtesting Engine Debug")
    print("=" * 50)
    
    # Same setup as main
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
    
    # Get data and signals
    data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    prepared_data = prepare_data(data)
    strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
    signals = strategy.backtest_prepare(prepared_data)
    
    print(f"📊 Our signals summary:")
    print(f"   Data range: {prepared_data.index[0]} to {prepared_data.index[-1]}")
    print(f"   Signal range: {signals.index[0]} to {signals.index[-1]}")
    print(f"   Total signals: {len(signals)}")
    print(f"   LONG signals: {(signals == 1.0).sum()}")
    print(f"   SHORT signals: {(signals == -1.0).sum()}")
    print(f"   HOLD signals: {(signals == 0.0).sum()}")
    
    # Show our signal events
    print(f"\n🎯 Our signal events:")
    non_zero = signals[signals != 0]
    for date, signal in non_zero.items():
        print(f"   {date}: {'LONG' if signal > 0 else 'SHORT'} (signal={signal})")
    
    # Create backtester
    backtester = Backtester(
        initial_capital=10000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    # Let's manually trace what the backtester does
    print(f"\n🔍 Manual backtester simulation:")
    print(f"Data shape: {prepared_data.shape}")
    print(f"Signals shape: {signals.shape}")
    
    # Check if indices match
    data_index = prepared_data.index
    signal_index = signals.index
    
    print(f"Index match: {data_index.equals(signal_index)}")
    
    if not data_index.equals(signal_index):
        print(f"❌ INDEX MISMATCH!")
        print(f"Data index: {data_index[0]} to {data_index[-1]} ({len(data_index)} items)")
        print(f"Signal index: {signal_index[0]} to {signal_index[-1]} ({len(signal_index)} items)")
        
        # Check for overlap
        overlap = data_index.intersection(signal_index)
        print(f"Overlap: {len(overlap)} dates")
        
    # Manually check what backtester would see
    print(f"\n🔍 What backtester receives:")
    for i in range(min(10, len(signals))):
        date = signals.index[i]
        signal = signals.iloc[i]
        if date in prepared_data.index:
            price = prepared_data.loc[date, 'close']
            print(f"   {date}: signal={signal}, price=${price:.2f}")
        else:
            print(f"   {date}: signal={signal}, price=NOT_IN_DATA")
    
    # Check specific dates where our signals are
    print(f"\n🎯 Checking our signal dates in data:")
    our_signal_dates = [
        pd.Timestamp('2025-04-11 08:00:00'),
        pd.Timestamp('2025-05-08 00:00:00'), 
        pd.Timestamp('2025-05-26 00:00:00')
    ]
    
    for date in our_signal_dates:
        if date in prepared_data.index:
            price = prepared_data.loc[date, 'close']
            signal = signals.loc[date] if date in signals.index else "NOT_IN_SIGNALS"
            print(f"   {date}: signal={signal}, price=${price:.2f} ✅")
        else:
            print(f"   {date}: NOT IN DATA ❌")
    
    # Now let's run the actual backtester and see what it does
    print(f"\n⚡ Running backtester...")
    
    # Create a simple test with just our 3 signals
    test_signals = pd.Series(0.0, index=prepared_data.index)
    test_signals.loc[our_signal_dates[0]] = 1.0  # LONG
    test_signals.loc[our_signal_dates[1]] = 1.0  # LONG  
    test_signals.loc[our_signal_dates[2]] = 1.0  # LONG
    
    print(f"🧪 Testing with simplified signals:")
    print(f"   Test signals: {(test_signals != 0).sum()} non-zero")
    
    # Run with test signals
    try:
        test_result = backtester.run(prepared_data, test_signals)
        print(f"✅ Test backtest: {test_result.total_trades} trades")
        
        if test_result.trades:
            for i, trade in enumerate(test_result.trades[:3]):
                print(f"   Test Trade {i+1}: {trade.side} from {trade.entry_time} to {trade.exit_time}")
    except Exception as e:
        print(f"❌ Test backtest failed: {e}")
    
    # Run with our actual signals
    try:
        actual_result = backtester.run(prepared_data, signals)
        print(f"✅ Actual backtest: {actual_result.total_trades} trades")
        
        if actual_result.trades:
            print(f"\n📋 First 3 actual trades:")
            for i, trade in enumerate(actual_result.trades[:3]):
                print(f"   Trade {i+1}: {trade.side} from {trade.entry_time} to {trade.exit_time}")
                print(f"      Entry: ${trade.entry_price:.2f}, Exit: ${trade.exit_price:.2f}")
                
                # Check if this trade corresponds to our signals
                entry_signal = signals.loc[trade.entry_time] if trade.entry_time in signals.index else "N/A"
                print(f"      Our signal at entry: {entry_signal}")
        
    except Exception as e:
        print(f"❌ Actual backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_backtesting_engine()