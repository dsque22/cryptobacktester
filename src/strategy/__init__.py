"""HMA-WAE Trading Strategy."""
from .base_strategy import BaseStrategy
from .strategies import (
    create_hma_wae_strategy,
    HMAWAEStrategy
)

__all__ = [
    'BaseStrategy', 
    'create_hma_wae_strategy',
    'HMAWAEStrategy'
]