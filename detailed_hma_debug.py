"""
Detailed HMA debugging to understand why flips aren't being detected.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy

def debug_hma_calculation():
    """Debug HMA calculation step by step"""
    print("🔍 Detailed HMA Calculation Debug")
    print("=" * 50)
    
    # Fetch data
    SYMBOL = 'BTC-USD'
    TIMEFRAME = '8h' 
    DATA_PERIOD = '1mo'
    
    print(f"📊 Fetching {SYMBOL} data ({TIMEFRAME}, {DATA_PERIOD})...")
    data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    prepared_data = prepare_data(data)
    
    print(f"✅ Data shape: {prepared_data.shape}")
    print(f"📅 Date range: {prepared_data.index[0]} to {prepared_data.index[-1]}")
    
    # Create strategy
    strategy = create_hma_wae_strategy(
        hma_length=45,
        trade_direction='long'
    )
    
    # Step-by-step HMA calculation
    print(f"\n🔢 HMA Calculation (Length = {strategy.hma_length}):")
    
    # 1. Calculate WMA components
    close_prices = prepared_data['close']
    print(f"Close prices range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
    
    # Manual HMA calculation to debug
    def debug_wma(series, length):
        """Debug WMA calculation"""
        weights = np.arange(1, length + 1)
        result = series.rolling(length).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan, 
            raw=True
        )
        return result
    
    def debug_hma(series, length):
        """Debug HMA calculation"""
        print(f"   Calculating HMA with length {length}")
        l2 = int(length / 2)  # 22
        lsqr = int(np.sqrt(length))  # 6
        print(f"   Half period: {l2}, Square root period: {lsqr}")
        
        wma_half = debug_wma(series, l2)
        wma_full = debug_wma(series, length)
        
        print(f"   WMA half ({l2}) - First valid: {wma_half.first_valid_index()}")
        print(f"   WMA full ({length}) - First valid: {wma_full.first_valid_index()}")
        
        raw_hma = 2 * wma_half - wma_full
        hma_result = debug_wma(raw_hma, lsqr)
        
        print(f"   HMA result - First valid: {hma_result.first_valid_index()}")
        print(f"   HMA result - Non-null count: {hma_result.count()}")
        
        return hma_result
    
    # Calculate HMA manually
    hma_manual = debug_hma(close_prices, 45)
    
    # Compare with strategy's HMA
    hma_strategy = strategy.get_hma(close_prices)
    
    print(f"\n📊 HMA Comparison:")
    print(f"Manual HMA non-null: {hma_manual.count()}")
    print(f"Strategy HMA non-null: {hma_strategy.count()}")
    
    # Show recent HMA values
    print(f"\n📈 Recent HMA Values (last 10 bars):")
    recent_data = prepared_data.tail(10).copy()
    recent_data['hma_manual'] = hma_manual.tail(10)
    recent_data['hma_strategy'] = hma_strategy.tail(10)
    
    for idx, row in recent_data.iterrows():
        print(f"{idx.strftime('%m-%d %H:%M')}: Close=${row['close']:.2f}, "
              f"HMA_manual={row['hma_manual']:.2f}, HMA_strategy={row['hma_strategy']:.2f}")
    
    # Check HMA direction and flips
    print(f"\n🔄 HMA Direction Analysis:")
    
    # Test different shift values for comparison
    for shift_val in [1, 2, 3]:
        print(f"\n   Testing HMA shift = {shift_val}:")
        
        hma_shifted = hma_strategy.shift(shift_val)
        valid_mask = hma_strategy.notna() & hma_shifted.notna()
        
        if valid_mask.sum() > 0:
            hma_up = pd.Series(False, index=prepared_data.index)
            hma_up[valid_mask] = hma_strategy[valid_mask] > hma_shifted[valid_mask]
            
            # Calculate flips
            hma_up_prev = hma_up.shift(1).fillna(False)
            hma_flip_up = hma_up & ~hma_up_prev
            hma_flip_down = ~hma_up & hma_up_prev
            
            print(f"      Valid comparisons: {valid_mask.sum()}")
            print(f"      HMA up periods: {hma_up.sum()}")
            print(f"      HMA flip up: {hma_flip_up.sum()}")
            print(f"      HMA flip down: {hma_flip_down.sum()}")
            
            # Show flip dates
            if hma_flip_up.sum() > 0:
                flip_dates = prepared_data.index[hma_flip_up]
                print(f"      Flip up dates: {[d.strftime('%m-%d %H:%M') for d in flip_dates]}")
            
            if hma_flip_down.sum() > 0:
                flip_dates = prepared_data.index[hma_flip_down]
                print(f"      Flip down dates: {[d.strftime('%m-%d %H:%M') for d in flip_dates]}")
    
    # Test with shorter HMA period to see if it detects flips
    print(f"\n🧪 Testing with shorter HMA periods:")
    for test_length in [21, 30, 35]:
        test_hma = debug_hma(close_prices, test_length)
        test_shifted = test_hma.shift(2)
        valid = test_hma.notna() & test_shifted.notna()
        
        if valid.sum() > 0:
            up_trend = test_hma[valid] > test_shifted[valid]
            up_prev = up_trend.shift(1).fillna(False)
            flips_up = (up_trend & ~up_prev).sum()
            flips_down = (~up_trend & up_prev).sum()
            
            print(f"   HMA({test_length}): {flips_up} up flips, {flips_down} down flips")
    
    # Show raw price movement for context
    print(f"\n📊 Price Movement Context:")
    price_changes = close_prices.pct_change() * 100
    print(f"Max price change: {price_changes.max():.2f}%")
    print(f"Min price change: {price_changes.min():.2f}%")
    print(f"Price volatility (std): {price_changes.std():.2f}%")
    
    return prepared_data, hma_strategy

def test_simple_hma():
    """Test with a very simple HMA calculation"""
    print(f"\n🔬 Simple HMA Test:")
    
    # Create simple test data with clear trend
    dates = pd.date_range('2024-01-01', periods=100, freq='8H')
    # Create trending data
    trend_data = np.linspace(100, 200, 100)  # Clear uptrend
    noise = np.random.normal(0, 2, 100)  # Small noise
    prices = trend_data + noise
    
    test_df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    strategy = create_hma_wae_strategy(hma_length=21, trade_direction='long')
    hma_test = strategy.get_hma(test_df['close'])
    
    # Check for flips
    hma_shifted = hma_test.shift(2)
    valid = hma_test.notna() & hma_shifted.notna()
    
    if valid.sum() > 0:
        up_trend = hma_test[valid] > hma_shifted[valid]
        up_prev = up_trend.shift(1).fillna(False)
        flips_up = (up_trend & ~up_prev).sum()
        flips_down = (~up_trend & up_prev).sum()
        
        print(f"Test data - Up flips: {flips_up}, Down flips: {flips_down}")
        print(f"✅ HMA calculation is working on test data")
    else:
        print(f"❌ HMA calculation failed on test data")

if __name__ == "__main__":
    data, hma = debug_hma_calculation()
    test_simple_hma()