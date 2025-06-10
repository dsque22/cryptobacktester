"""
HMA-WAE Strategy Parameter Optimization

Standalone example for optimizing Hull Moving Average + Waddah Attar Explosion strategy.
This is the primary strategy optimization example for deployment validation.
"""

import pandas as pd
import numpy as np
import logging
from src.optimization import ParameterOptimizer
from src.strategy.strategies import HMAWAEStrategy
from src.data import fetch_crypto_data, prepare_data
from config import SYMBOLS, TIMEFRAMES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def optimize_hma_wae():
    """Optimize HMA-WAE strategy parameters for deployment readiness."""
    
    print("🎯 HMA-WAE Strategy Parameter Optimization")
    print("=" * 60)
    
    # 1. Use the same real market data as main.py
    print("📊 Fetching real market data (same as main.py)...")
    
    # Configuration matching main.py
    SYMBOL = 'BTC-USD'
    TIMEFRAME = '8h'
    DATA_PERIOD = '1y'
    
    print(f"🎯 Symbol: {SYMBOL}")
    print(f"📅 Timeframe: {TIMEFRAME}")
    print(f"⏰ Period: {DATA_PERIOD}")
    
    # Fetch real market data
    raw_data = fetch_crypto_data(SYMBOL, period=DATA_PERIOD, interval=TIMEFRAME)
    if raw_data is None or raw_data.empty:
        print("❌ Failed to fetch market data")
        return None
        
    # Prepare data with technical indicators
    data = prepare_data(raw_data)
    
    print(f"✅ Real market data loaded: {len(data)} bars")
    print(f"📈 Price range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
    print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 2. Define HMA-WAE parameter optimization space
    print("\n🔧 Defining HMA-WAE parameter space...")
    param_ranges = {
        'hma_length': [21, 35, 45],                    # Hull MA periods
        'sensitivity': [100, 150, 200],                # WAE sensitivity levels (correct param name)
        'trade_direction': ['long', 'both']            # Trading directions
    }
    
    total_combinations = len(param_ranges['hma_length']) * len(param_ranges['sensitivity']) * len(param_ranges['trade_direction'])
    print(f"🎲 Parameter combinations: {total_combinations}")
    for param, values in param_ranges.items():
        print(f"   • {param}: {values}")
    
    # 3. Run HMA-WAE optimization
    print("\n⚡ Running HMA-WAE parameter optimization...")
    
    optimizer = ParameterOptimizer(
        strategy_class=HMAWAEStrategy,
        data=data,
        param_ranges=param_ranges,
        n_jobs=2,
        initial_capital=10000
    )
    
    try:
        results = optimizer.optimize()
        
        # 4. Display comprehensive results
        print("\n" + "="*60)
        print("📊 HMA-WAE OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"🏆 Best Parameters:")
        for param, value in results.best_params.items():
            print(f"   • {param}: {value}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   • Best Sharpe Ratio: {results.best_sharpe:.3f}")
        print(f"   • Deployment Score: {results.deployment_score:.3f}")
        print(f"   • Stability (test/train): {results.stability:.3f}")
        print(f"   • Test Sharpe: {results.test_sharpe:.3f}")
        
        print(f"\n🎯 Deployment Analysis:")
        print(f"   • Recommendation: {results.recommendation}")
        print(f"   • Successful Tests: {len(results.all_results)}/{results.total_combinations}")
        
        # 5. Deployment decision
        print(f"\n" + "="*60)
        if results.recommendation == 'DEPLOY':
            print("🟢 DEPLOYMENT READY!")
            print("✅ HMA-WAE strategy meets all deployment criteria")
            print("🚀 Recommended for live trading with these parameters")
        elif results.recommendation == 'TEST_MORE':
            print("🟡 NEEDS MORE VALIDATION")
            print("⚠️  Strategy shows promise but requires additional testing")
            print("🔬 Consider longer data periods or parameter refinement")
        else:
            print("🔴 NOT RECOMMENDED FOR DEPLOYMENT")
            print("❌ Strategy does not meet minimum performance standards")
            print("🔧 Consider different parameter ranges or strategy modifications")
        
        # 6. Parameter recommendations
        print(f"\n📋 PARAMETER RECOMMENDATIONS:")
        print(f"   Use these parameters in main.py:")
        print(f"   STRATEGY_PARAMS = {{")
        for param, value in results.best_params.items():
            print(f"       '{param}': {repr(value)},")
        print(f"   }}")
        
        # 7. Analysis and insights
        print(f"\n🔍 OPTIMIZATION INSIGHTS:")
        
        # Analyze best performing parameters
        best_hma = results.best_params.get('hma_length', 'N/A')
        best_sens = results.best_params.get('sensitivity', 'N/A')
        best_direction = results.best_params.get('trade_direction', 'N/A')
        
        print(f"   🎯 Best HMA Length: {best_hma}")
        if best_hma == 21:
            print(f"      → Faster trend detection (more signals)")
        elif best_hma == 35:
            print(f"      → Balanced responsiveness and stability")
        elif best_hma == 45:
            print(f"      → Smoother trends (fewer false signals)")
        
        print(f"   ⚡ Best WAE Sensitivity: {best_sens}")
        if best_sens == 100:
            print(f"      → Conservative momentum filter")
        elif best_sens == 150:
            print(f"      → Balanced momentum detection")
        elif best_sens == 200:
            print(f"      → Aggressive momentum filter")
        
        print(f"   📈 Best Trade Direction: {best_direction.title()}")
        if best_direction == 'long':
            print(f"      → Long-only strategy recommended")
        elif best_direction == 'both':
            print(f"      → Bidirectional trading optimal")
        
        print(f"\n💡 DEPLOYMENT NOTES:")
        print(f"   • Test Sharpe Ratio: {results.test_sharpe:.3f}")
        print(f"   • Strategy Stability: {results.stability:.3f}")
        if results.stability > 0.8:
            print(f"   • ✅ High stability - consistent performance")
        elif results.stability > 0.6:
            print(f"   • ⚠️  Moderate stability - monitor performance")
        else:
            print(f"   • ❌ Low stability - high variance between train/test")
        
        return results
        
    except Exception as e:
        print(f"\n❌ HMA-WAE optimization failed: {e}")
        print("🔧 This indicates HMA-WAE strategy has dependency issues")
        return None

if __name__ == "__main__":
    results = optimize_hma_wae()
    
    if results:
        print(f"\n✅ HMA-WAE optimization completed successfully!")
        print(f"🎯 Ready for deployment validation")
    else:
        print(f"\n🔧 HMA-WAE optimization requires debugging")
        print(f"💡 Check strategy dependencies and indicator calculations")