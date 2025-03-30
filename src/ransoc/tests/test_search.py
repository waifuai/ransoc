import unittest
import numpy as np
from ransoc.planet import Planet
from ransoc.star import Star
from ransoc.search import hit

class TestSearch(unittest.TestCase):

    def test_hit(self):
        star = Star([0, 0, 0])
        planets = [
            Planet([1, 1, 1], mass=2.0),
            Planet([2, 2, 2], mass=1.0),
            Planet([3, 3, 3], mass=0.5)
        ]
        
        hit_planet, updated_planets = hit(planets, star, alpha=0.2)
        
        # Check if the correct planet is hit (planet 0 in this case)
        self.assertEqual(hit_planet, planets[0])
        
        # Check if masses are updated correctly
        self.assertAlmostEqual(updated_planets[0].mass, 2.0 * (1 - 0.2))
        self.assertAlmostEqual(updated_planets[1].mass, 1.0 * (1 + 0.2 / 2))
        self.assertAlmostEqual(updated_planets[2].mass, 0.5 * (1 + 0.2 / 2))

if __name__ == '__main__':
    unittest.main()