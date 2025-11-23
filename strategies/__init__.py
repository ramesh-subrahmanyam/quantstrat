"""
Trading Strategies

Base classes and concrete strategy implementations.
"""

from .base import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy
)
from .vol_normalized_buy_and_hold import VolNormalizedBuyAndHold

__all__ = [
    'BaseStrategy',
    'BuyAndHoldStrategy',
    'MovingAverageCrossoverStrategy',
    'VolNormalizedBuyAndHold',
]
