"""
Test the strategy with the FIXED backtesting engine
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.data import fetch_crypto_data, prepare_data
from src.strategy import create_hma_wae_strategy
from src.backtesting.metrics import calculate_metrics, format_metrics
from utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    commission: float

@dataclass
class BacktestResults:
    """Container for backtest results."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    
    # Summary metrics
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

class FixedBacktester:
    """Fixed backtesting engine that properly respects provided signals."""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        self.trades = []
        self.equity = [self.initial_capital]
        self.timestamps = []
    
    def run(self, data: pd.DataFrame, signals: pd.Series) -> BacktestResults:
        """
        Run backtest on data with given signals - FIXED VERSION.
        
        Args:
            data: OHLCV data
            signals: Trading signals (1 = buy, -1 = sell, 0 = hold/close)
            
        Returns:
            BacktestResults object
        """
        self.reset()
        
        print(f"🔍 FIXED BACKTESTER: Starting with {len(signals)} signals")
        
        # Debug: Count signal types
        long_signals = (signals == 1.0).sum()
        short_signals = (signals == -1.0).sum()
        hold_signals = (signals == 0.0).sum()
        
        print(f"📊 Signal distribution: {long_signals} long, {short_signals} short, {hold_signals} hold")
        
        # Show signal events
        non_zero_signals = signals[signals != 0.0]
        print(f"🎯 Signal events: {len(non_zero_signals)}")
        for date, signal in non_zero_signals.items():
            signal_type = "LONG" if signal > 0 else "SHORT" if signal < 0 else "EXIT"
            print(f"   {date}: {signal_type} (signal={signal})")
        
        # Process each timestamp
        for timestamp in data.index:
            if timestamp not in signals.index:
                continue
            
            row = data.loc[timestamp]
            current_price = row['close']
            signal = signals.loc[timestamp]
            
            # Store timestamp
            self.timestamps.append(timestamp)
            
            # Process signal - FIXED LOGIC
            if signal == 1.0 and self.position == 0:
                # LONG entry signal and we're flat
                self._open_position(timestamp, current_price, 1)
                print(f"🟢 OPENED LONG: {timestamp} at ${current_price:.2f}")
                
            elif signal == -1.0 and self.position == 0:
                # SHORT entry signal and we're flat
                self._open_position(timestamp, current_price, -1)
                print(f"🔴 OPENED SHORT: {timestamp} at ${current_price:.2f}")
                
            elif signal == 0.0 and self.position != 0:
                # Exit signal and we have a position
                self._close_position(timestamp, current_price)
                print(f"🟠 CLOSED POSITION: {timestamp} at ${current_price:.2f}")
            
            # Update equity
            equity_value = self._calculate_equity(current_price)
            self.equity.append(equity_value)
        
        # Close any remaining position
        if self.position != 0:
            last_timestamp = data.index[-1]
            last_price = data.loc[last_timestamp, 'close']
            self._close_position(last_timestamp, last_price)
            print(f"🟠 CLOSED FINAL POSITION: {last_timestamp} at ${last_price:.2f}")
        
        # Create results
        results = self._create_results()
        
        print(f"✅ FIXED BACKTEST COMPLETE: {results.total_trades} trades, "
                   f"{results.total_return:.2%} return")
        
        return results
    
    def _open_position(self, timestamp: pd.Timestamp, price: float, direction: int):
        """Open a new position."""
        # Apply slippage
        if direction > 0:  # Buy (long)
            execution_price = price * (1 + self.slippage)
        else:  # Sell (short)
            execution_price = price * (1 - self.slippage)
        
        # Calculate position size and commission
        position_value = self.capital * 0.35  # Use 35% of capital
        commission = position_value * self.commission_rate
        
        # Update state
        self.position = direction
        self.position_size = (position_value - commission) / execution_price
        self.entry_price = execution_price
        self.entry_time = timestamp
        self.capital -= position_value
    
    def _close_position(self, timestamp: pd.Timestamp, price: float):
        """Close current position."""
        if self.position == 0:
            return
        
        # Apply slippage
        if self.position > 0:  # Closing long
            execution_price = price * (1 - self.slippage)
        else:  # Closing short
            execution_price = price * (1 + self.slippage)
        
        # Calculate P&L
        if self.position > 0:  # Long
            gross_pnl = self.position_size * (execution_price - self.entry_price)
        else:  # Short
            gross_pnl = self.position_size * (self.entry_price - execution_price)
        
        # Calculate commission
        position_value = abs(self.position_size * execution_price)
        commission = position_value * self.commission_rate
        
        # Net P&L
        net_pnl = gross_pnl - commission
        pnl_percent = net_pnl / (self.position_size * self.entry_price)
        
        # Create trade record
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=timestamp,
            entry_price=self.entry_price,
            exit_price=execution_price,
            position_size=self.position_size,
            side='long' if self.position > 0 else 'short',
            pnl=net_pnl,
            pnl_percent=pnl_percent,
            commission=commission * 2  # Entry + exit commission
        )
        
        self.trades.append(trade)
        
        # Update capital
        self.capital += position_value + net_pnl
        
        # Reset position
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity value."""
        if self.position == 0:
            return self.capital
        
        # Calculate unrealized P&L
        if self.position > 0:  # Long
            market_value = self.position_size * current_price
            unrealized_pnl = self.position_size * (current_price - self.entry_price)
        else:  # Short
            market_value = self.position_size * self.entry_price
            unrealized_pnl = self.position_size * (self.entry_price - current_price)
        
        return self.capital + market_value + unrealized_pnl
    
    def _create_results(self) -> BacktestResults:
        """Create BacktestResults object."""
        results = BacktestResults()
        
        # Store trades
        results.trades = self.trades
        
        # Create equity curve
        equity_series = pd.Series(self.equity[1:], index=self.timestamps)
        results.equity_curve = equity_series
        
        # Calculate returns
        results.returns = equity_series.pct_change().fillna(0)
        
        # Summary metrics
        results.total_return = (equity_series.iloc[-1] / self.initial_capital) - 1
        results.total_trades = len(self.trades)
        
        if results.total_trades > 0:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]
            
            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = results.winning_trades / results.total_trades
            
            if winning_trades:
                results.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                total_wins = sum(t.pnl for t in winning_trades)
            else:
                results.avg_win = 0
                total_wins = 0
            
            if losing_trades:
                results.avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
                total_losses = abs(sum(t.pnl for t in losing_trades))
            else:
                results.avg_loss = 0
                total_losses = 0
            
            if total_losses > 0:
                results.profit_factor = total_wins / total_losses
            else:
                results.profit_factor = float('inf') if total_wins > 0 else 0
        
        return results

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
    prepared_data = prepare_data(data)
    
    print(f"📊 Creating strategy...")
    strategy = create_hma_wae_strategy(**STRATEGY_PARAMS)
    
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
    print(f"\n🎯 Signal events:")
    for date, signal in non_zero.items():
        signal_type = "LONG" if signal > 0 else "SHORT" 
        print(f"   {date}: {signal_type}")
    
    # Test with FIXED backtester
    print(f"\n⚡ Running FIXED backtester...")
    
    fixed_backtester = FixedBacktester(
        initial_capital=10000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    result = fixed_backtester.run(prepared_data, signals)
    
    # Show results
    print(f"\n✅ FIXED BACKTESTER RESULTS:")
    print(f"📊 Total trades: {result.total_trades}")
    print(f"📊 Total return: {result.total_return:.2%}")
    print(f"📊 Winning trades: {result.winning_trades}")
    print(f"📊 Losing trades: {result.losing_trades}")
    print(f"📊 Win rate: {result.win_rate:.2%}")
    
    if result.trades:
        print(f"\n📋 Trade Details:")
        for i, trade in enumerate(result.trades, 1):
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            print(f"   Trade {i}: {trade.side.upper()}")
            print(f"      Entry: {trade.entry_time} at ${trade.entry_price:.2f}")
            print(f"      Exit:  {trade.exit_time} at ${trade.exit_price:.2f}")
            print(f"      P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2%})")
            print(f"      Duration: {duration:.1f} hours")
            print()
    
    # Compare with broken backtester
    print(f"\n🔄 Comparison:")
    print(f"   Fixed backtester: {result.total_trades} trades")
    print(f"   Broken backtester: 110 trades (ignoring our signals)")
    print(f"   Our signals: {len(non_zero)} entry signals")
    
    if result.total_trades == len(non_zero):
        print(f"   ✅ FIXED! Trades match our signals")
    else:
        print(f"   ⚠️ Still a mismatch")
        
    return result

if __name__ == "__main__":
    result = test_fixed_backtester()