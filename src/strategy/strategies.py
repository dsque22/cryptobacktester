"""
Trading strategy implementations.
Updated with advanced HMA-WAE strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, fall back to basic implementation if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using basic implementations.")

from .base_strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"SMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on SMA crossover."""
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Calculate SMAs
        sma_fast = data['close'].rolling(window=self.fast_period).mean()
        sma_slow = data['close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        signals[sma_fast > sma_slow] = 1.0   # Buy signal
        signals[sma_fast < sma_slow] = -1.0  # Sell signal
        
        # Only trade on crossovers (not every bar)
        signals = signals.diff()
        signals[signals > 0] = 1.0   # Buy on upward crossover
        signals[signals < 0] = -1.0  # Sell on downward crossover
        signals[signals == 0] = 0.0  # Hold otherwise
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }

class RSIStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy."""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI levels."""
        signals = pd.Series(0, index=data.index, dtype=float)

        if 'rsi' not in data.columns:
            # Calculate RSI if not present
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = data['rsi']

        # Generate signals - Fixed logic
        buy_signals = rsi <= self.oversold
        sell_signals = rsi >= self.overbought
        
        # Only signal on transitions (not sustained levels)
        buy_transitions = buy_signals & ~buy_signals.shift(1, fill_value=False)
        sell_transitions = sell_signals & ~sell_signals.shift(1, fill_value=False)
        
        signals[buy_transitions] = 1.0   # Buy when crossing into oversold
        signals[sell_transitions] = -1.0  # Sell when crossing into overbought

        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought
        }

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands."""
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Use pre-calculated BB if available
        if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            upper = data['bb_upper']
            lower = data['bb_lower']
            middle = data['bb_middle']
        else:
            # Calculate Bollinger Bands
            middle = data['close'].rolling(window=self.period).mean()
            std = data['close'].rolling(window=self.period).std()
            upper = middle + (std * self.std_dev)
            lower = middle - (std * self.std_dev)
        
        close = data['close']
        
        # Mean reversion signals
        # Buy when price touches lower band
        signals[close <= lower] = 1.0
        
        # Sell when price touches upper band
        signals[close >= upper] = -1.0
        
        # Exit positions when price crosses middle band
        long_exit = (close > middle) & (close.shift(1) <= middle)
        short_exit = (close < middle) & (close.shift(1) >= middle)
        
        signals[long_exit | short_exit] = 0.0
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'std_dev': self.std_dev
        }

class HMAWAEStrategy(BaseStrategy):
    """
    Advanced HMA-WAE Hybrid Trading Strategy
    
    Combines Hull Moving Average (HMA) trend direction with 
    Waddah Attar Explosion (WAE) momentum confirmation.
    
    Entry: HMA flip + WAE momentum confirmation (within lag period)
    Exit: HMA opposite flip only
    """
    
    def __init__(self, 
                 # HMA Parameters
                 hma_length: int = 21,
                 hma_length_mult: float = 1.0,
                 hma_mode: str = "hma",  # "hma", "ehma", "thma"
                 
                 # WAE Parameters (matching TradingView exactly)
                 fast_length: int = 20,
                 slow_length: int = 40,
                 sensitivity: int = 150,
                 bb_length: int = 20,    # chanLen in Pine Script
                 bb_mult: float = 2.0,   # bbMult in Pine Script  
                 dz_length: int = 20,    # dzLen in Pine Script
                 dz_mult: float = 3.7,   # dzMult in Pine Script (was 3.0, now 3.7)
                 
                 # Strategy Parameters
                 max_bars_lag: int = 3,
                 trade_direction: str = "both"):  # "long", "short", "both"
        
        super().__init__(f"HMA_WAE_{int(hma_length * hma_length_mult)}")
        
        # Store parameters
        self.hma_length = int(hma_length * hma_length_mult)
        self.hma_mode = hma_mode.lower()
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.sensitivity = sensitivity
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.dz_length = dz_length
        self.dz_mult = dz_mult
        self.max_bars_lag = max_bars_lag
        self.trade_direction = trade_direction.lower()
    
    def wma(self, data: pd.Series, length: int) -> pd.Series:
        """Weighted Moving Average - Fixed calculation"""
        if len(data) < length or length < 1:
            return pd.Series(index=data.index, dtype=float)
        
        weights = np.arange(1, length + 1)
        
        def wma_calc(x):
            if len(x) < length:
                return np.nan
            return np.dot(x, weights) / weights.sum()
        
        return data.rolling(window=length, min_periods=length).apply(wma_calc, raw=True)
    
    def ema(self, data: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=length).mean()
    
    def sma(self, data: pd.Series, length: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(length).mean()
    
    def hma(self, data: pd.Series, length: int) -> pd.Series:
        """Hull Moving Average - Fixed calculation"""
        if len(data) < length:
            return pd.Series(index=data.index, dtype=float)
        
        l2 = int(length / 2)
        lsqr = int(np.sqrt(length))
        
        # Ensure we have enough data
        if len(data) < length or l2 < 1 or lsqr < 1:
            return pd.Series(index=data.index, dtype=float)
        
        wma1 = self.wma(data, l2)
        wma2 = self.wma(data, length)
        
        # Check if we have valid WMA values
        if wma1.isna().all() or wma2.isna().all():
            return pd.Series(index=data.index, dtype=float)
        
        # Calculate raw HMA
        raw_hma = 2 * wma1 - wma2
        
        # Final HMA calculation
        hma_result = self.wma(raw_hma, lsqr)
        
        return hma_result
    
    def ehma(self, data: pd.Series, length: int) -> pd.Series:
        """Exponential Hull Moving Average"""
        l2 = int(length / 2)
        lsqr = int(np.sqrt(length))
        ema1 = self.ema(data, l2)
        ema2 = self.ema(data, length)
        return self.ema(2 * ema1 - ema2, lsqr)
    
    def thma(self, data: pd.Series, length: int) -> pd.Series:
        """Triangular Hull Moving Average"""
        l3 = int(length / 3)
        l2 = int(length / 2)
        wma1 = self.wma(data, l3)
        wma2 = self.wma(data, l2)
        wma3 = self.wma(data, length)
        return self.wma(wma1 * 3 - wma2 - wma3, length)
    
    def get_hma(self, data: pd.Series) -> pd.Series:
        """Get Hull Moving Average based on mode"""
        if self.hma_mode == "ehma":
            return self.ehma(data, self.hma_length)
        elif self.hma_mode == "thma":
            return self.thma(data, self.hma_length)
        else:  # default to hma
            return self.hma(data, self.hma_length)
    
    def true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range with proper handling"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # Create DataFrame to find max across columns
        tr_df = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2, 
            'tr3': tr3
        })
        
        return tr_df.max(axis=1)
    
    def rma(self, data: pd.Series, length: int) -> pd.Series:
        """Running Moving Average (like Pine Script's ta.rma)"""
        alpha = 1.0 / length
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_wae(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Waddah Attar Explosion indicators with error handling"""
        try:
            # MACD calculation
            ema_fast = self.ema(df['close'], self.fast_length)
            ema_slow = self.ema(df['close'], self.slow_length)
            macd = ema_fast - ema_slow
            macd_diff = (macd - macd.shift(1)) * self.sensitivity
            
            # Bollinger Bands calculation
            bb_basis = self.sma(df['close'], self.bb_length)
            bb_std = df['close'].rolling(self.bb_length).std()
            bb_upper = bb_basis + (self.bb_mult * bb_std)
            bb_lower = bb_basis - (self.bb_mult * bb_std)
            e1 = bb_upper - bb_lower
            
            # Dead Zone calculation
            tr = self.true_range(df['high'], df['low'], df['close'])
            dead_zone = self.rma(tr, self.dz_length) * self.dz_mult
            
            # WAE signals with proper NaN handling
            trend_up = np.maximum(macd_diff.fillna(0), 0)
            trend_down = np.maximum(-macd_diff.fillna(0), 0)
            
            # Create boolean masks properly
            wae_up = (trend_up > 0) & (e1 > dead_zone)
            wae_down = (trend_down > 0) & (e1 > dead_zone)
            
            # Fill any remaining NaN values
            wae_up = wae_up.fillna(False)
            wae_down = wae_down.fillna(False)
            dead_zone = dead_zone.fillna(0)
            
            return wae_up, wae_down, dead_zone
            
        except Exception as e:
            print(f"Warning: WAE calculation error: {e}")
            # Return safe default values
            false_series = pd.Series(False, index=df.index)
            zero_series = pd.Series(0.0, index=df.index)
            return false_series, false_series, zero_series
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate HMA-WAE hybrid signals matching TradingView Pine Script exactly"""
        signals = pd.Series(0.0, index=data.index, dtype=float)
        
        # Check if we have enough data for HMA calculation
        min_required_bars = self.hma_length + 10  # Add buffer
        if len(data) < min_required_bars:
            print(f"⚠️ Insufficient data: {len(data)} bars, need {min_required_bars} for HMA({self.hma_length})")
            print(f"💡 Consider using shorter HMA period or longer data period")
            return signals
        
        # Calculate HMA (matching Pine Script exactly)
        hma_values = self.get_hma(data['close'])
        
        # Pine Script: SHULL = H[2], hullUp = MHULL > SHULL
        hma_shifted_2 = hma_values.shift(2)  # This matches H[2] in Pine Script
        
        # Only proceed where we have valid HMA values
        valid_mask = hma_values.notna() & hma_shifted_2.notna()
        
        if valid_mask.sum() == 0:
            print(f"❌ No valid HMA values calculated")
            return signals
        
        print(f"✅ Valid HMA values: {valid_mask.sum()}/{len(data)}")
        
        # HMA direction (matching Pine Script)
        hull_up = pd.Series(False, index=data.index)
        hull_up[valid_mask] = hma_values[valid_mask] > hma_shifted_2[valid_mask]
        
        # HMA flips (matching Pine Script exactly)
        hull_up_prev = hull_up.shift(1).fillna(False)
        hull_flip_up = hull_up & ~hull_up_prev      # hullFlipUp = hullUp and not hullUp[1]
        hull_flip_down = ~hull_up & hull_up_prev    # hullFlipDn = not hullUp and hullUp[1]
        
        print(f"📈 HMA flip up events: {hull_flip_up.sum()}")
        print(f"📉 HMA flip down events: {hull_flip_down.sum()}")
        
        # Calculate WAE (matching Pine Script)
        try:
            wae_up, wae_down, dead_zone = self.calculate_wae(data)
            print(f"🟢 WAE up signals: {wae_up.sum()}")
            print(f"🔴 WAE down signals: {wae_down.sum()}")
        except Exception as e:
            print(f"Warning: WAE calculation failed: {e}")
            return signals
        
        # Track bars from HMA flip (matching Pine Script logic exactly)
        bars_from_flip_up = pd.Series(np.nan, index=data.index)
        bars_from_flip_down = pd.Series(np.nan, index=data.index)
        
        # Pine Script var tracking logic
        flip_up_counter = None
        flip_down_counter = None
        
        for i in range(len(data)):
            if hull_flip_up.iloc[i]:
                flip_up_counter = 0
                flip_down_counter = None
            elif hull_flip_down.iloc[i]:
                flip_down_counter = 0
                flip_up_counter = None
            else:
                if flip_up_counter is not None:
                    flip_up_counter += 1
                if flip_down_counter is not None:
                    flip_down_counter += 1
            
            bars_from_flip_up.iloc[i] = flip_up_counter
            bars_from_flip_down.iloc[i] = flip_down_counter
        
        # WAE confirmation within lag (matching Pine Script)
        wae_confirm_long = (wae_up & 
                           bars_from_flip_up.notna() & 
                           (bars_from_flip_up <= self.max_bars_lag))
        
        wae_confirm_short = (wae_down & 
                            bars_from_flip_down.notna() & 
                            (bars_from_flip_down <= self.max_bars_lag))
        
        print(f"🎯 WAE confirm long: {wae_confirm_long.sum()}")
        print(f"🎯 WAE confirm short: {wae_confirm_short.sum()}")
        
        # ⚠️ CRITICAL FIX: Position tracking to prevent consecutive entries
        # This matches Pine Script logic: strategy.position_size == 0
        
        current_position = 0  # 0 = flat, 1 = long, -1 = short
        
        for i in range(len(data)):
            current_signal = 0.0
            
            # Entry conditions - ONLY when position is flat (position_size == 0)
            if current_position == 0:  # No position currently
                
                # Long entry condition
                if (self.trade_direction in ["both", "long"] and 
                    wae_confirm_long.iloc[i]):
                    current_signal = 1.0
                    current_position = 1
                    print(f"🟢 LONG ENTRY: {data.index[i]}")
                
                # Short entry condition  
                elif (self.trade_direction in ["both", "short"] and 
                      wae_confirm_short.iloc[i]):
                    current_signal = -1.0
                    current_position = -1
                    print(f"🔴 SHORT ENTRY: {data.index[i]}")
            
            # Exit conditions - ONLY when we have a position
            elif current_position != 0:
                
                # Long exit condition (HMA flip down)
                if (current_position > 0 and hull_flip_down.iloc[i]):
                    current_signal = 0.0
                    current_position = 0
                    print(f"🟠 LONG EXIT: {data.index[i]}")
                
                # Short exit condition (HMA flip up)
                elif (current_position < 0 and hull_flip_up.iloc[i]):
                    current_signal = 0.0
                    current_position = 0
                    print(f"🟠 SHORT EXIT: {data.index[i]}")
            
            # Set the signal for this bar
            signals.iloc[i] = current_signal
        
        # Count and log final signals
        final_long_entries = (signals == 1.0).sum()
        final_short_entries = (signals == -1.0).sum()
        final_exits = (signals == 0.0).sum()
        
        print(f"🎯 Final signal counts:")
        print(f"   Long entries: {final_long_entries}")
        print(f"   Short entries: {final_short_entries}")
        print(f"   Exits/Holds: {final_exits}")
        
        # Log actual signal events
        non_zero_signals = signals[signals != 0.0]
        print(f"🎯 Final signal events:")
        for date, signal in non_zero_signals.items():
            if signal > 0:
                print(f"🟢 LONG ENTRY: {date}")
            elif signal < 0:
                print(f"🔴 SHORT ENTRY: {date}")
            else:
                print(f"🟠 EXIT: {date}")
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'hma_length': self.hma_length,
            'hma_mode': self.hma_mode,
            'fast_length': self.fast_length,
            'slow_length': self.slow_length,
            'sensitivity': self.sensitivity,
            'bb_length': self.bb_length,
            'bb_mult': self.bb_mult,
            'dz_length': self.dz_length,
            'dz_mult': self.dz_mult,
            'max_bars_lag': self.max_bars_lag,
            'trade_direction': self.trade_direction
        }

# Legacy HMA strategy (simpler version for backwards compatibility)
class LegacyHMAWAEStrategy(BaseStrategy):
    """Simplified Hull Moving Average with WAE Strategy (original version)."""
    
    def __init__(self, hma_period: int = 21, sensitivity: float = 150):
        super().__init__(f"Legacy_HMA_WAE_{hma_period}")
        self.hma_period = hma_period
        self.sensitivity = sensitivity
    
    def _wma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def _hma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self._wma(series, half_period)
        wma_full = self._wma(series, period)
        
        raw_hma = 2 * wma_half - wma_full
        return self._wma(raw_hma, sqrt_period)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on HMA direction and WAE momentum."""
        signals = pd.Series(0, index=data.index, dtype=float)
        
        # Calculate HMA
        hma = self._hma(data['close'], self.hma_period)
        hma_direction = hma.diff()
        
        # Simple momentum filter (simplified WAE)
        momentum = data['close'].pct_change(5).abs() * 100
        strong_momentum = momentum > (momentum.rolling(20).mean() * 1.5)
        
        # Generate signals
        # Buy: HMA turning up with strong momentum
        buy_signal = (hma_direction > 0) & (hma_direction.shift(1) <= 0) & strong_momentum
        signals[buy_signal] = 1.0
        
        # Sell: HMA turning down with strong momentum
        sell_signal = (hma_direction < 0) & (hma_direction.shift(1) >= 0) & strong_momentum
        signals[sell_signal] = -1.0
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'hma_period': self.hma_period,
            'sensitivity': self.sensitivity
        }

# Convenience functions
def create_sma_strategy(fast: int = 20, slow: int = 50) -> SMAStrategy:
    """Create SMA crossover strategy."""
    return SMAStrategy(fast, slow)

def create_rsi_strategy(period: int = 14, oversold: float = 30, overbought: float = 70) -> RSIStrategy:
    """Create RSI mean reversion strategy."""
    return RSIStrategy(period, oversold, overbought)

def create_bb_strategy(period: int = 20, std_dev: float = 2.0) -> BollingerBandsStrategy:
    """Create Bollinger Bands strategy."""
    return BollingerBandsStrategy(period, std_dev)

def create_hma_wae_strategy(hma_length: int = 21, 
                           hma_mode: str = "hma",
                           fast_length: int = 20,
                           slow_length: int = 40,
                           sensitivity: int = 150,
                           bb_length: int = 20,        # NEW parameter
                           bb_mult: float = 2.0,       # NEW parameter
                           dz_length: int = 20,        # NEW parameter
                           dz_mult: float = 3.7,       # NEW parameter
                           max_bars_lag: int = 3,
                           trade_direction: str = "both") -> HMAWAEStrategy:
    """Create advanced HMA-WAE strategy with all parameters."""
    return HMAWAEStrategy(
        hma_length=hma_length,
        hma_mode=hma_mode,
        fast_length=fast_length,
        slow_length=slow_length,
        sensitivity=sensitivity,
        bb_length=bb_length,           # Pass the new parameters
        bb_mult=bb_mult,
        dz_length=dz_length,
        dz_mult=dz_mult,
        max_bars_lag=max_bars_lag,
        trade_direction=trade_direction
    )

def create_legacy_hma_wae_strategy(hma_period: int = 21, sensitivity: float = 150) -> LegacyHMAWAEStrategy:
    """Create legacy HMA-WAE strategy (backwards compatibility)."""
    return LegacyHMAWAEStrategy(hma_period, sensitivity)