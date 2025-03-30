import unittest
import numpy as np
from ransoc.star import Star

class TestStar(unittest.TestCase):

    def test_init(self):
        star = Star([1, 2, 3])
        self.assertTrue(np.array_equal(star.position, np.array([1, 2, 3], dtype=float)))

    def test_repr(self):
        star = Star([1, 2, 3])
        self.assertEqual(repr(star), "Star(position=[1. 2. 3.])")

if __name__ == '__main__':
    unittest.main()