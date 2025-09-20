# ransoc.py
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import numpy.typing as npt
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .planet import Planet
from .star import Star

logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """Enumeration of supported distance metrics."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    CHEBYSHEV = "chebyshev"

class RANSOCError(Exception):
    """Base exception class for RANSOC-related errors."""
    pass

class ValidationError(RANSOCError):
    """Raised when validation fails."""
    pass

class ConfigurationError(RANSOCError):
    """Raised when configuration is invalid."""
    pass

@dataclass
class RANSOCConfig:
    """Configuration class for RANSOC algorithm parameters."""

    alpha: float = 0.1
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    enable_history_tracking: bool = True
    max_history_length: int = 1000
    enable_logging: bool = True
    log_level: str = "INFO"
    epsilon: float = 1e-10  # Small value to prevent division by zero

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ConfigurationError(f"Alpha must be between 0 and 1, got {self.alpha}")
        if self.max_history_length <= 0:
            raise ConfigurationError("max_history_length must be positive")
        if self.epsilon <= 0:
            raise ConfigurationError("epsilon must be positive")

@dataclass
class QueryResult:
    """Container for RANSOC query results."""

    hit_planet: Planet
    rankings: List[Tuple[Planet, float]]
    query: Star
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    total_mass: float = 0.0
    entropy: float = 0.0

    def get_rank(self, planet: Planet) -> Optional[int]:
        """Get the rank of a specific planet in the results."""
        for i, (p, _) in enumerate(self.rankings):
            if p.id == planet.id:
                return i + 1
        return None

    def get_top_k(self, k: int) -> List[Tuple[Planet, float]]:
        """Get top k results."""
        return self.rankings[:k]

    def get_weights_array(self) -> npt.NDArray[np.float64]:
        """Get weights as numpy array."""
        return np.array([weight for _, weight in self.rankings])

class DistanceCalculator:
    """Handles distance calculations with different metrics."""

    @staticmethod
    def calculate(
        point1: npt.NDArray[np.float64],
        point2: npt.NDArray[np.float64],
        metric: DistanceMetric
    ) -> float:
        """Calculate distance between two points using specified metric."""
        if metric == DistanceMetric.EUCLIDEAN:
            return float(np.linalg.norm(point1 - point2))
        elif metric == DistanceMetric.MANHATTAN:
            return float(np.sum(np.abs(point1 - point2)))
        elif metric == DistanceMetric.COSINE:
            dot_product = np.dot(point1, point2)
            norm1 = np.linalg.norm(point1)
            norm2 = np.linalg.norm(point2)
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance for cosine
            return float(1 - dot_product / (norm1 * norm2))
        elif metric == DistanceMetric.CHEBYSHEV:
            return float(np.max(np.abs(point1 - point2)))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

class RANSOC:
    """
    Real-time Adaptive Normalization for the Satisfaction of Curiosity (RANSOC) algorithm.

    This algorithm balances between returning relevant search results and encouraging
    exploration of the full search space by dynamically adjusting weights of results.
    """

    def __init__(
        self,
        positions: Union[List[List[float]], npt.NDArray[np.float64]],
        config: Optional[RANSOCConfig] = None,
        planet_ids: Optional[List[str]] = None,
        planet_metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize RANSOC algorithm.

        Args:
            positions: Planet positions in n-dimensional space
            config: RANSOC configuration parameters
            planet_ids: Optional list of planet IDs
            planet_metadata: Optional list of planet metadata dictionaries

        Raises:
            ValidationError: If input parameters are invalid
            ConfigurationError: If configuration is invalid
        """
        self.config = config or RANSOCConfig()

        # Setup logging
        if self.config.enable_logging:
            logging.basicConfig(level=getattr(logging, self.config.log_level))
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

        # Convert positions to numpy array
        try:
            self.positions = np.array(positions, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Positions must be convertible to numpy array: {e}")

        if self.positions.ndim != 2:
            raise ValidationError("Positions must be a 2D array")
        if self.positions.shape[0] == 0:
            raise ValidationError("At least one planet position required")
        if self.positions.shape[1] == 0:
            raise ValidationError("Planet positions must have at least one dimension")

        self.dimension = self.positions.shape[1]
        self.n_planets = self.positions.shape[0]

        # Initialize planets
        self.planets: List[Planet] = []
        for i in range(self.n_planets):
            planet_id = planet_ids[i] if planet_ids and i < len(planet_ids) else None
            metadata = planet_metadata[i] if planet_metadata and i < len(planet_metadata) else None
            planet = Planet(self.positions[i], id=planet_id, metadata=metadata)
            self.planets.append(planet)

        # Algorithm state
        self.iteration = 0
        self.query_history: List[QueryResult] = []
        self.mass_history: List[npt.NDArray[np.float64]] = []
        self.distance_calculator = DistanceCalculator()

        self.logger.info(f"Initialized RANSOC with {self.n_planets} planets in {self.dimension}D space")

    def compute_distances(self, star: Star) -> npt.NDArray[np.float64]:
        """Compute distances between all planets and the query star."""
        distances = np.array([
            self.distance_calculator.calculate(planet.position, star.position, self.config.distance_metric)
            for planet in self.planets
        ])
        return distances

    def compute_weights(self, distances: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute weights for all planets based on distances and masses."""
        masses = np.array([planet.mass for planet in self.planets])

        # Prevent division by zero
        safe_distances = np.maximum(distances, self.config.epsilon)

        weights = masses / safe_distances
        return weights

    def update_masses(self, hit_index: int) -> None:
        """Update masses of all planets based on the hit planet."""
        n_planets = len(self.planets)

        # Update hit planet
        hit_planet = self.planets[hit_index]
        old_mass = hit_planet.mass
        new_mass = old_mass * (1 - self.config.alpha)
        hit_planet.update_mass(new_mass)
        hit_planet.increment_hit_count()

        # Update other planets
        mass_increase = (1 + self.config.alpha / (n_planets - 1))
        for i, planet in enumerate(self.planets):
            if i != hit_index:
                planet.update_mass(planet.mass * mass_increase)

        # Record mass history
        if self.config.enable_history_tracking:
            current_masses = np.array([p.mass for p in self.planets])
            self.mass_history.append(current_masses)
            if len(self.mass_history) > self.config.max_history_length:
                self.mass_history.pop(0)

        self.logger.debug(f"Updated masses: hit planet {hit_index} mass {old_mass:.3f} -> {new_mass:.3f}")

    def query(self, star: Star) -> QueryResult:
        """
        Process a query and return the most relevant planet along with all rankings.

        Args:
            star: Query star containing the search position

        Returns:
            QueryResult containing hit planet, rankings, and metadata
        """
        if star.position.shape[0] != self.dimension:
            raise ValidationError(
                f"Query dimension {star.position.shape[0]} doesn't match "
                f"planet dimension {self.dimension}"
            )

        star.increment_query_count()
        self.iteration += 1

        # Compute distances and weights
        distances = self.compute_distances(star)
        weights = self.compute_weights(distances)

        # Find hit planet
        hit_index = np.argmax(weights)
        hit_planet = self.planets[hit_index]

        # Update masses
        self.update_masses(hit_index)

        # Create sorted rankings
        rankings = [(planet, weight) for planet, weight in zip(self.planets, weights)]
        rankings.sort(key=lambda x: x[1], reverse=True)

        # Calculate additional metrics
        total_mass = sum(p.mass for p in self.planets)
        masses = np.array([p.mass for p in self.planets])
        entropy = float(-np.sum(masses * np.log(masses + self.config.epsilon)))

        # Create result
        result = QueryResult(
            hit_planet=hit_planet,
            rankings=rankings,
            query=star,
            iteration=self.iteration,
            total_mass=total_mass,
            entropy=entropy
        )

        # Record in history
        if self.config.enable_history_tracking:
            self.query_history.append(result)
            if len(self.query_history) > self.config.max_history_length:
                self.query_history.pop(0)

        self.logger.info(
            f"Query {self.iteration}: hit planet '{hit_planet.id}' "
            ".3f"
        )

        return result

    def batch_query(self, stars: List[Star]) -> List[QueryResult]:
        """
        Process multiple queries in batch.

        Args:
            stars: List of query stars

        Returns:
            List of QueryResult objects
        """
        results = []
        for star in stars:
            result = self.query(star)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the RANSOC instance."""
        if not self.planets:
            return {}

        masses = np.array([p.mass for p in self.planets])
        hit_counts = np.array([p.hit_count for p in self.planets])

        stats = {
            'n_planets': self.n_planets,
            'dimension': self.dimension,
            'iteration': self.iteration,
            'total_queries': len(self.query_history),
            'config': {
                'alpha': self.config.alpha,
                'distance_metric': self.config.distance_metric.value,
                'enable_history_tracking': self.config.enable_history_tracking,
                'max_history_length': self.config.max_history_length
            },
            'mass_stats': {
                'mean': float(np.mean(masses)),
                'std': float(np.std(masses)),
                'min': float(np.min(masses)),
                'max': float(np.max(masses)),
                'total': float(np.sum(masses))
            },
            'hit_stats': {
                'mean': float(np.mean(hit_counts)),
                'std': float(np.std(hit_counts)),
                'min': int(np.min(hit_counts)),
                'max': int(np.max(hit_counts)),
                'total': int(np.sum(hit_counts))
            }
        }

        if self.query_history:
            recent_results = self.query_history[-10:]  # Last 10 queries
            avg_entropy = np.mean([r.entropy for r in recent_results])
            stats['recent_performance'] = {
                'avg_entropy': float(avg_entropy),
                'unique_hits': len(set(r.hit_planet.id for r in recent_results))
            }

        return stats

    def reset(self) -> None:
        """Reset the algorithm to initial state."""
        for planet in self.planets:
            planet.mass = 1.0
            planet.hit_count = 0
            planet.mass_history = [1.0]

        self.iteration = 0
        self.query_history.clear()
        self.mass_history.clear()

        self.logger.info("RANSOC instance reset to initial state")

    def get_planet_by_id(self, planet_id: str) -> Optional[Planet]:
        """Get a planet by its ID."""
        for planet in self.planets:
            if planet.id == planet_id:
                return planet
        return None

    def get_planets_sorted_by_mass(self) -> List[Planet]:
        """Get planets sorted by mass (descending)."""
        return sorted(self.planets, key=lambda p: p.mass, reverse=True)

    def get_planets_sorted_by_hits(self) -> List[Planet]:
        """Get planets sorted by hit count (descending)."""
        return sorted(self.planets, key=lambda p: p.hit_count, reverse=True)

# Backward compatibility function
def hit(planets: List[Planet], star: Star, alpha: float = 0.1) -> Tuple[Planet, List[Planet]]:
    """
    Legacy function for backward compatibility.

    This function maintains the original API while using the improved RANSOC class internally.
    """
    # Extract positions from planets
    positions = [planet.position for planet in planets]

    # Create RANSOC instance with legacy configuration
    config = RANSOCConfig(alpha=alpha, enable_history_tracking=False)
    ransoc = RANSOC(positions, config=config)

    # Copy planet properties
    for old_planet, new_planet in zip(planets, ransoc.planets):
        new_planet.mass = old_planet.mass
        new_planet.id = old_planet.id
        new_planet.metadata = old_planet.metadata
        new_planet.hit_count = old_planet.hit_count

    # Process query
    result = ransoc.query(star)

    # Update original planets with new masses
    for old_planet, new_planet in zip(planets, ransoc.planets):
        old_planet.mass = new_planet.mass
        old_planet.hit_count = new_planet.hit_count

    return result.hit_planet, planets