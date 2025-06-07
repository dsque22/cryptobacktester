"""
Debug script to check exactly what signals are being generated and passed to the backtester.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy

def debug_signals():
    """Debug the signal generation process"""
    print("🔍 Signal Generation Debug")
    print("=" * 50)
    
    # Same config as main
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
    
    # Fetch and prepare data
    data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    prepared_data = prepare_data(data)
    
    print(f"✅ Data prepared: {prepared_data.shape}")
    
    # Create strategy
    strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
    print(f"✅ Strategy created: {strategy.name}")
    print(f"📋 Trade direction: {strategy.trade_direction}")
    
    # Generate signals
    print(f"\n🔄 Generating signals...")
    signals = strategy.generate_signals(prepared_data)
    
    # Analyze the signal series
    print(f"\n📊 Signal Analysis:")
    print(f"Total signals generated: {len(signals)}")
    
    signal_counts = signals.value_counts().sort_index()
    print(f"Signal distribution:")
    for value, count in signal_counts.items():
        if value == 1.0:
            print(f"   LONG (1.0): {count}")
        elif value == -1.0:
            print(f"   SHORT (-1.0): {count}")
        elif value == 0.0:
            print(f"   HOLD/EXIT (0.0): {count}")
        else:
            print(f"   OTHER ({value}): {count}")
    
    # Show all non-zero signals with dates
    print(f"\n🎯 All Non-Zero Signals:")
    non_zero_signals = signals[signals != 0.0]
    
    if len(non_zero_signals) == 0:
        print("   ❌ No non-zero signals found!")
        return
    
    for date, signal in non_zero_signals.items():
        price = prepared_data.loc[date, 'close']
        if signal > 0:
            signal_type = "🟢 LONG ENTRY"
        elif signal < 0:
            signal_type = "🔴 SHORT ENTRY"
        else:
            signal_type = "🟠 EXIT"
        
        print(f"   {date}: {signal_type} (signal={signal}) at ${price:.2f}")
    
    # Check if there are any consecutive signals
    print(f"\n🔍 Consecutive Signal Check:")
    prev_signal = 0
    consecutive_count = 0
    
    for i, (date, signal) in enumerate(signals.items()):
        if signal != 0:
            if signal == prev_signal:
                consecutive_count += 1
                print(f"   ⚠️ Consecutive signal at {date}: {signal} (count: {consecutive_count})")
            else:
                consecutive_count = 0
            prev_signal = signal
        else:
            prev_signal = 0
            consecutive_count = 0
    
    # Test with backtest_prepare method
    print(f"\n🧪 Testing backtest_prepare method:")
    try:
        backtest_signals = strategy.backtest_prepare(prepared_data)
        
        print(f"Backtest signals shape: {backtest_signals.shape}")
        backtest_counts = backtest_signals.value_counts().sort_index()
        print(f"Backtest signal distribution:")
        for value, count in backtest_counts.items():
            if value == 1.0:
                print(f"   LONG (1.0): {count}")
            elif value == -1.0:
                print(f"   SHORT (-1.0): {count}")
            elif value == 0.0:
                print(f"   HOLD/EXIT (0.0): {count}")
            else:
                print(f"   OTHER ({value}): {count}")
        
        # Compare the two signal series
        print(f"\n🔄 Comparing generate_signals vs backtest_prepare:")
        differences = (signals != backtest_signals).sum()
        print(f"Differences found: {differences}")
        
        if differences > 0:
            print(f"🚨 PROBLEM: backtest_prepare returns different signals!")
            diff_indices = signals[signals != backtest_signals].index
            for idx in diff_indices[:10]:  # Show first 10 differences
                print(f"   {idx}: generate_signals={signals[idx]}, backtest_prepare={backtest_signals[idx]}")
        else:
            print(f"✅ Both methods return identical signals")
            
    except Exception as e:
        print(f"❌ Error in backtest_prepare: {e}")
        import traceback
        traceback.print_exc()
    
    return signals, prepared_data

if __name__ == "__main__":
    signals, data = debug_signals()