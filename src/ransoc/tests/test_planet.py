import unittest
import numpy as np
from planet import Planet

class TestPlanet(unittest.TestCase):

    def test_init(self):
        planet = Planet([1, 2, 3], 5.0)
        self.assertTrue(np.array_equal(planet.position, np.array([1, 2, 3], dtype=float)))
        self.assertEqual(planet.mass, 5.0)

    def test_repr(self):
        planet = Planet([1, 2, 3], 5.0)
        self.assertEqual(repr(planet), "Planet(position=[1. 2. 3.], mass=5.0)")

if __name__ == '__main__':
    unittest.main()