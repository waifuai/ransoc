# search.py
import numpy as np
from .planet import Planet  # Import Planet from the current package

def hit(planets, star, alpha=0.1):  # Add alpha as a parameter
    """
    Calculates the hit planet based on RANSOC and updates planet masses.
    """
    distances = np.array([np.linalg.norm(planet.position - star.position) for planet in planets])
    weights = np.array([planet.mass / distance for planet, distance in zip(planets, distances)])
    hit_index = np.argmax(weights)
    hit_planet = planets[hit_index]

    # Update masses
    n = len(planets)
    for i, planet in enumerate(planets):
        if i == hit_index:
            planet.mass *= (1 - alpha)
        else:
            planet.mass *= (1 + alpha / (n - 1))

    return hit_planet, planets