# planet.py
import numpy as np

class Planet:
    def __init__(self, position, mass=1.0):
        self.position = np.array(position, dtype=float)  # Store position as a NumPy array
        self.mass = mass

    def __repr__(self):
        return f"Planet(position={self.position}, mass={self.mass})"