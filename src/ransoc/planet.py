# planet.py
from typing import Union, List, Dict, Any, Optional
import numpy as np
import numpy.typing as npt

class Planet:
    """
    Represents a searchable item (planet) in the RANSOC algorithm.

    A planet has a position in n-dimensional space and a mass that influences
    its ranking in search results. The mass is dynamically adjusted to promote
    exploration of the search space.
    """

    def __init__(
        self,
        position: Union[List[float], npt.NDArray[np.float64]],
        mass: float = 1.0,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Planet.

        Args:
            position: The planet's position in n-dimensional space
            mass: Initial mass of the planet (must be positive)
            id: Optional unique identifier for the planet
            metadata: Optional dictionary of additional planet information

        Raises:
            ValueError: If mass is not positive or position is empty
            TypeError: If position cannot be converted to numpy array
        """
        if mass <= 0:
            raise ValueError(f"Planet mass must be positive, got {mass}")

        try:
            self.position = np.array(position, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Position must be convertible to numpy array: {e}")

        if self.position.size == 0:
            raise ValueError("Planet position cannot be empty")

        self.mass = float(mass)
        # Set id first to avoid issues with hash
        self.id = id or f"planet_{hash((tuple(self.position.flatten()), self.mass))}"
        self.metadata = metadata or {}
        self.hit_count = 0
        self.mass_history: List[float] = [self.mass]

    def distance_to(self, other_position: npt.NDArray[np.float64]) -> float:
        """
        Calculate Euclidean distance to another position.

        Args:
            other_position: Position to calculate distance to

        Returns:
            Euclidean distance between positions
        """
        return float(np.linalg.norm(self.position - other_position))

    def update_mass(self, new_mass: float) -> None:
        """
        Update the planet's mass and record in history.

        Args:
            new_mass: New mass value (must be positive)

        Raises:
            ValueError: If new_mass is not positive
        """
        if new_mass <= 0:
            raise ValueError(f"Planet mass must be positive, got {new_mass}")

        self.mass = float(new_mass)
        self.mass_history.append(self.mass)

    def increment_hit_count(self) -> None:
        """Increment the hit counter when this planet is selected."""
        self.hit_count += 1

    def get_weight(self, query_position: npt.NDArray[np.float64]) -> float:
        """
        Calculate the weight of this planet for a given query.

        Weight is inversely proportional to distance, scaled by mass.

        Args:
            query_position: Position of the query point

        Returns:
            Weight value for ranking
        """
        distance = self.distance_to(query_position)
        return self.mass / distance if distance > 0 else float('inf')

    def __repr__(self) -> str:
        return f"Planet(id='{self.id}', position={self.position}, mass={self.mass:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Planet):
            return False
        return (np.array_equal(self.position, other.position) and
                self.id == other.id and
                abs(self.mass - other.mass) < 1e-10)

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.position.flatten())))