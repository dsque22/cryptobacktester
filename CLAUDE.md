# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Run the main backtesting system (HMA-WAE strategy on BTC-USD, 8h timeframe)
python main.py

# Run specific test validation
python test_fixed_backtester.py

# Install dependencies
pip install -r requirements.txt

# Check data cache status
ls -la data/cache/

# View recent results
ls -la results/
```

## System Architecture

This is a professional-grade cryptocurrency backtesting framework with a sophisticated modular design:

### Core Data Flow
```
Data Sources (Binance/Yahoo) → Cache (Parquet) → Data Processor → Strategy Signals → Backtesting Engine → Performance Analysis → Visualization/Reports
```

### Key Components & Interactions

**Configuration Layer** (`config.py`):
- Central hub for symbols, timeframes, data periods, and trading parameters
- Defines directory structure and caching behavior
- Provides timeframe recommendations for different trading styles

**Data Pipeline** (`src/data/`):
- `data_fetcher.py`: Multi-source fetching with intelligent fallback (Binance → Yahoo Finance)
- `data_processor.py`: Data cleaning and technical indicator calculation
- Uses Parquet caching for performance optimization

**Strategy Framework** (`src/strategy/`):
- `base_strategy.py`: Abstract BaseStrategy interface using Strategy Pattern
- `strategies.py`: Concrete implementations including sophisticated HMA-WAE strategy
- Factory functions for easy strategy instantiation

**Backtesting Engine** (`src/backtesting/`):
- `engine.py`: "Fixed" backtesting engine that properly respects trading signals
- `metrics.py`: Comprehensive performance analysis (20+ metrics)
- Realistic simulation with commission, slippage, and position sizing

**Visualization** (`src/visualization/`):
- Professional charts and equity curve generation
- CSV export for detailed analysis

### Critical Architecture Notes

1. **The Backtesting Engine is "Fixed"**: Previous versions had signal processing issues. The current `Backtester` class in `engine.py` has been specifically corrected to properly execute trades based on strategy signals.

2. **HMA-WAE Strategy Focus**: The primary strategy combines Hull Moving Average (trend) with Waddah Attar Explosion (momentum) - this is a sophisticated technical analysis approach requiring proper lag tolerance and signal confirmation.

3. **Data Integrity**: The system validates data quality and handles missing/invalid data gracefully. Cache expiry ensures fresh data when needed.

## Strategy Development Patterns

### Adding New Strategies
1. Inherit from `BaseStrategy` in `src/strategy/base_strategy.py`
2. Implement `generate_signals()` method returning pandas Series with values: 1 (buy), -1 (sell), 0 (hold)
3. Add factory function in `strategies.py`
4. Update `main.py` to use new strategy

### Strategy Signal Format
- Signals must be pandas Series with same index as input data
- Values: `1.0` for buy, `-1.0` for sell, `0.0` for hold/close
- The backtesting engine processes these signals sequentially and manages position state

### Configuration Patterns
- Modify `main.py` for symbol/timeframe changes
- Use factory functions with parameters for strategy customization
- Leverage `config.py` constants for consistent behavior

## Key Files & Responsibilities

**`main.py`**: Primary entry point and orchestrator
- Configures backtest parameters (symbol, timeframe, strategy settings)
- Coordinates entire workflow from data fetching to results generation
- Currently optimized for HMA-WAE strategy demonstration

**`src/backtesting/engine.py`**: Core backtesting logic
- `Backtester` class: Fixed engine that respects provided signals
- `Trade` dataclass: Comprehensive trade record keeping
- `BacktestResults` dataclass: Complete performance encapsulation

**`src/strategy/strategies.py`**: Strategy implementations
- `HMAWAEStrategy`: Primary advanced strategy with sophisticated signal logic
- Alternative strategies: SMA crossover, RSI mean reversion, Bollinger Bands
- Factory functions for easy instantiation and parameter configuration

**`test_fixed_backtester.py`**: Validation and testing
- Validates backtesting engine accuracy
- Compares strategies and provides debugging information
- Essential for verifying new strategy implementations

## Data Management

### Cache System
- Parquet files in `data/cache/` for fast access
- Automatic cache validation and refresh
- Filename format: `{SYMBOL}_{TIMEFRAME}.parquet`

### Supported Assets & Timeframes
- Major cryptocurrencies: BTC, ETH, ADA, DOT, AVAX, LTC, LINK, UNI
- Timeframes: 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M
- Data periods: 1mo, 3mo, 6mo, 1y, 2y, 3y

## Testing & Validation

### Strategy Validation Process
1. Run `test_fixed_backtester.py` to validate engine behavior
2. Check signal generation logic matches expected patterns
3. Verify performance metrics are reasonable for the strategy type
4. Compare results against benchmark (buy-and-hold)

### Common Issues & Debugging
- **No trades executed**: Check signal generation - may need to adjust strategy parameters
- **Poor performance**: Verify strategy logic and parameter tuning
- **Data issues**: Check cache validity and data source availability
- **Signal lag**: HMA-WAE strategy includes lag tolerance - review signal timing

## File Templates

### New Strategy Template
```python
class NewStrategy(BaseStrategy):
    def __init__(self, param1=default1, param2=default2):
        super().__init__("NewStrategy")
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index, dtype=float)
        # Implement signal logic
        return signals

def create_new_strategy(param1=default1, param2=default2) -> NewStrategy:
    return NewStrategy(param1=param1, param2=param2)
```

### Results Analysis Pattern
```python
# Standard result processing
results = backtester.run_backtest(strategy, data)
print_performance_summary(results)
save_results_to_csv(results, filename)
create_equity_curve_chart(results, filename)
```

## Important Notes

- The project has evolved from multi-strategy comparison to focus on the sophisticated HMA-WAE strategy
- Performance metrics are comprehensive and include risk-adjusted measures
- The backtesting engine accounts for realistic trading costs (commission: 0.1%, slippage: 0.05%)
- Position sizing is fixed at 35% of available capital per trade
- All timestamps use the data's original timezone (typically UTC for crypto data)