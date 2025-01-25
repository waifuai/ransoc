# star.py
import numpy as np

class Star:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)  # Store position as a NumPy array

    def __repr__(self):
        return f"Star(position={self.position})"