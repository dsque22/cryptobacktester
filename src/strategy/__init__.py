"""Trading strategies."""
from .base_strategy import BaseStrategy
from .strategies import (
    create_sma_strategy, 
    create_rsi_strategy, 
    create_bb_strategy, 
    create_hma_wae_strategy,
    create_legacy_hma_wae_strategy,
    HMAWAEStrategy,
    LegacyHMAWAEStrategy
)

__all__ = [
    'BaseStrategy', 
    'create_sma_strategy', 
    'create_rsi_strategy', 
    'create_bb_strategy', 
    'create_hma_wae_strategy',
    'create_legacy_hma_wae_strategy',
    'HMAWAEStrategy',
    'LegacyHMAWAEStrategy'
]