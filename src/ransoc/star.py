# star.py
from typing import Union, List, Dict, Any, Optional
import numpy as np
import numpy.typing as npt
from datetime import datetime

class Star:
    """
    Represents a query (star) in the RANSOC algorithm.

    A star has a position in n-dimensional space and represents a user query
    or search request. It can contain metadata about the query context.
    """

    def __init__(
        self,
        position: Union[List[float], npt.NDArray[np.float64]],
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize a Star.

        Args:
            position: The star's position in n-dimensional space
            id: Optional unique identifier for the query
            metadata: Optional dictionary of query context information
            timestamp: Optional timestamp of when the query was made

        Raises:
            ValueError: If position is empty
            TypeError: If position cannot be converted to numpy array
        """
        try:
            self.position = np.array(position, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Position must be convertible to numpy array: {e}")

        if self.position.size == 0:
            raise ValueError("Star position cannot be empty")

        # Set id first to avoid issues with hash
        self.id = id or f"query_{hash((tuple(self.position.flatten()), float('nan')))}"
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        self.query_count = 0

    def increment_query_count(self) -> None:
        """Increment the query counter when this star is used."""
        self.query_count += 1

    def distance_to(self, other_position: npt.NDArray[np.float64]) -> float:
        """
        Calculate Euclidean distance to another position.

        Args:
            other_position: Position to calculate distance to

        Returns:
            Euclidean distance between positions
        """
        return float(np.linalg.norm(self.position - other_position))

    def get_query_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this query.

        Returns:
            Dictionary containing query information
        """
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'dimension': len(self.position),
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'query_count': self.query_count
        }

    def __repr__(self) -> str:
        return f"Star(id='{self.id}', position={self.position}, queries={self.query_count})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Star):
            return False
        return (np.array_equal(self.position, other.position) and
                self.id == other.id)

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.position.flatten())))