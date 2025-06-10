# Parameter Optimization Usage Guide

## Quick Start

```python
from src.optimization import ParameterOptimizer
from src.strategy.strategies import HMAWAEStrategy

# Define parameter ranges
param_ranges = {
    'hma_length': [21, 35, 45, 55, 60, 75],
    'sensitivity': [100, 125, 150, 175, 200], 
    'trade_direction': ['long', 'both',]
}

# Run optimization
optimizer = ParameterOptimizer(
    strategy_class=HMAWAEStrategy,
    data=your_data,
    param_ranges=param_ranges
)

results = optimizer.optimize()
print(f"Best params: {results.best_params}")
print(f"Recommendation: {results.recommendation}")
```

## Standalone Example

Run the complete HMA-WAE optimization:

```bash
python optimize_hma_wae.py
```

## Parameter Ranges

```python
# Simple ranges
param_ranges = {
    'period': [10, 20, 30],              # List of values
    'threshold': range(5, 20, 5),        # Range object
    'direction': ['long', 'short', 'both'] # String options
}

# Advanced ranges
param_ranges = {
    'fast_ma': list(range(5, 25, 5)),    # [5, 10, 15, 20]
    'slow_ma': [50, 100, 200],           # Specific values
    'risk_level': [0.01, 0.02, 0.05]     # Float values
}
```

## Results Interpretation

### Deployment Recommendations

- **🟢 DEPLOY**: Strategy meets all criteria, ready for live trading
- **🟡 TEST_MORE**: Promising but needs additional validation
- **🔴 REJECT**: Does not meet minimum performance standards

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Deployment Score**: Combined performance metric (>0.7 for deployment)
- **Stability**: Train/test consistency (>0.8 is stable)

### Using Results

```python
# Get optimized parameters
best_params = results.best_params

# Update main.py configuration
STRATEGY_PARAMS = {
    'hma_length': best_params['hma_length'],
    'sensitivity': best_params['sensitivity'],
    'trade_direction': best_params['trade_direction']
}
```

## Performance Notes

- **Parallel Processing**: Uses all available CPU cores by default
- **Memory Usage**: ~1MB per parameter combination
- **Runtime**: ~1-2 seconds per combination on modern hardware
- **Recommended**: <100 combinations for interactive use