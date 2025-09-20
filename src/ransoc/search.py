# search.py
"""
Legacy search module for backward compatibility.

This module provides the original hit function API while using the improved
RANSOC implementation internally. New code should use the RANSOC class directly.
"""

from typing import List, Tuple
import warnings

from .planet import Planet
from .star import Star
from .ransoc import hit as new_hit

def hit(planets: List[Planet], star: Star, alpha: float = 0.1) -> Tuple[Planet, List[Planet]]:
    """
    Legacy function to calculate hit planet and update masses.

    .. deprecated::
        This function is deprecated. Use the RANSOC class directly for new code.

    Args:
        planets: List of Planet objects
        star: Star object representing the query
        alpha: Learning rate parameter (0 < alpha < 1)

    Returns:
        Tuple of (hit_planet, updated_planets)
    """
    warnings.warn(
        "ransoc.search.hit() is deprecated. Use RANSOC class directly for new code.",
        DeprecationWarning,
        stacklevel=2
    )

    return new_hit(planets, star, alpha)