"""
Trading Strategies

Base classes and concrete strategy implementations.
"""

from .base import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy
)

__all__ = [
    'BaseStrategy',
    'BuyAndHoldStrategy',
    'MovingAverageCrossoverStrategy',
]
