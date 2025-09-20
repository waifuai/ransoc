import unittest
import numpy as np
from ransoc.planet import Planet

class TestPlanet(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.position = [1, 2, 3]
        self.mass = 5.0
        self.planet = Planet(self.position, self.mass)

    def test_init_with_list(self):
        """Test initialization with list position."""
        self.assertTrue(np.array_equal(self.planet.position, np.array(self.position, dtype=float)))
        self.assertEqual(self.planet.mass, self.mass)
        self.assertIsNotNone(self.planet.id)
        self.assertEqual(self.planet.hit_count, 0)
        self.assertEqual(len(self.planet.mass_history), 1)
        self.assertEqual(self.planet.mass_history[0], self.mass)

    def test_init_with_numpy_array(self):
        """Test initialization with numpy array position."""
        position_array = np.array([4, 5, 6])
        planet = Planet(position_array, 3.0)
        self.assertTrue(np.array_equal(planet.position, position_array))
        self.assertEqual(planet.mass, 3.0)

    def test_init_with_custom_id(self):
        """Test initialization with custom ID."""
        custom_id = "test_planet_123"
        planet = Planet([1, 2], 2.0, id=custom_id)
        self.assertEqual(planet.id, custom_id)

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"category": "test", "priority": 1}
        planet = Planet([1, 2], 2.0, metadata=metadata)
        self.assertEqual(planet.metadata, metadata)

    def test_init_invalid_mass(self):
        """Test initialization with invalid mass."""
        with self.assertRaises(ValueError):
            Planet([1, 2], -1.0)

        with self.assertRaises(ValueError):
            Planet([1, 2], 0.0)

    def test_init_empty_position(self):
        """Test initialization with empty position."""
        with self.assertRaises(ValueError):
            Planet([], 1.0)

    def test_init_invalid_position(self):
        """Test initialization with invalid position."""
        with self.assertRaises(TypeError):
            Planet("invalid", 1.0)

    def test_distance_to(self):
        """Test distance calculation."""
        other_position = np.array([4, 5, 6])
        distance = self.planet.distance_to(other_position)

        # Expected distance: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) = 3*sqrt(3)
        expected = np.sqrt(27)
        self.assertAlmostEqual(distance, expected, places=10)

    def test_update_mass(self):
        """Test mass update functionality."""
        new_mass = 7.5
        self.planet.update_mass(new_mass)

        self.assertEqual(self.planet.mass, new_mass)
        self.assertEqual(len(self.planet.mass_history), 2)
        self.assertEqual(self.planet.mass_history[-1], new_mass)

    def test_update_mass_invalid(self):
        """Test mass update with invalid values."""
        with self.assertRaises(ValueError):
            self.planet.update_mass(-1.0)

        with self.assertRaises(ValueError):
            self.planet.update_mass(0.0)

    def test_increment_hit_count(self):
        """Test hit count increment."""
        initial_count = self.planet.hit_count
        self.planet.increment_hit_count()

        self.assertEqual(self.planet.hit_count, initial_count + 1)

    def test_get_weight(self):
        """Test weight calculation."""
        query_position = np.array([1, 2, 6])  # Distance 3 along z-axis
        weight = self.planet.get_weight(query_position)

        expected_weight = self.mass / 3.0
        self.assertAlmostEqual(weight, expected_weight, places=10)

    def test_get_weight_zero_distance(self):
        """Test weight calculation with zero distance."""
        query_position = np.array([1, 2, 3])  # Same position
        weight = self.planet.get_weight(query_position)

        self.assertEqual(weight, float('inf'))

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.planet)
        self.assertIn("Planet", repr_str)
        self.assertIn(self.planet.id, repr_str)
        self.assertIn("mass=", repr_str)

    def test_equality(self):
        """Test planet equality."""
        planet1 = Planet([1, 2, 3], 5.0, id="test")
        planet2 = Planet([1, 2, 3], 5.0, id="test")
        planet3 = Planet([1, 2, 4], 5.0, id="test")
        planet4 = Planet([1, 2, 3], 6.0, id="test")

        self.assertEqual(planet1, planet2)
        self.assertNotEqual(planet1, planet3)
        self.assertNotEqual(planet1, planet4)

    def test_hash(self):
        """Test planet hash."""
        planet1 = Planet([1, 2, 3], 5.0, id="test")
        planet2 = Planet([1, 2, 3], 5.0, id="test")
        planet3 = Planet([1, 2, 4], 5.0, id="test")

        self.assertEqual(hash(planet1), hash(planet2))
        self.assertNotEqual(hash(planet1), hash(planet3))

    def test_hash_consistency(self):
        """Test that hash is consistent across calls."""
        hash1 = hash(self.planet)
        hash2 = hash(self.planet)
        self.assertEqual(hash1, hash2)

if __name__ == '__main__':
    unittest.main()