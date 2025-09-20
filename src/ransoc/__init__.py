"""
RANSOC: Real-time Adaptive Normalization for the Satisfaction of Curiosity

This package implements the RANSOC algorithm for balancing relevance and exploration
in search results by dynamically adjusting result weights based on user interactions.
"""

__version__ = "2.0.0"
__author__ = "RANSOC Development Team"
__description__ = "Real-time Adaptive Normalization for the Satisfaction of Curiosity"

# Core classes
from .planet import Planet
from .star import Star
from .ransoc import RANSOC, RANSOCConfig, QueryResult, DistanceMetric
from .ransoc import hit as legacy_hit

# Visualization
from .visualization import RANSOCVisualizer, plot_ransoc_overview, plot_query_analysis

# Legacy compatibility
from .search import hit

# Exception classes
from .ransoc import RANSOCError, ValidationError, ConfigurationError

__all__ = [
    # Core classes
    'Planet',
    'Star',
    'RANSOC',
    'RANSOCConfig',
    'QueryResult',
    'DistanceMetric',

    # Visualization
    'RANSOCVisualizer',
    'plot_ransoc_overview',
    'plot_query_analysis',

    # Functions
    'hit',
    'legacy_hit',

    # Exceptions
    'RANSOCError',
    'ValidationError',
    'ConfigurationError',

    # Metadata
    '__version__',
    '__author__',
    '__description__'
]