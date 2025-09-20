import unittest
import numpy as np
from datetime import datetime
from ransoc.star import Star

class TestStar(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.position = [1, 2, 3]
        self.star = Star(self.position)

    def test_init_with_list(self):
        """Test initialization with list position."""
        self.assertTrue(np.array_equal(self.star.position, np.array(self.position, dtype=float)))
        self.assertIsNotNone(self.star.id)
        self.assertIsInstance(self.star.timestamp, datetime)
        self.assertEqual(self.star.query_count, 0)
        self.assertEqual(self.star.metadata, {})

    def test_init_with_numpy_array(self):
        """Test initialization with numpy array position."""
        position_array = np.array([4, 5, 6])
        star = Star(position_array)
        self.assertTrue(np.array_equal(star.position, position_array))

    def test_init_with_custom_id(self):
        """Test initialization with custom ID."""
        custom_id = "test_star_123"
        star = Star([1, 2], id=custom_id)
        self.assertEqual(star.id, custom_id)

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"source": "test", "priority": 2}
        star = Star([1, 2], metadata=metadata)
        self.assertEqual(star.metadata, metadata)

    def test_init_with_custom_timestamp(self):
        """Test initialization with custom timestamp."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        star = Star([1, 2], timestamp=custom_time)
        self.assertEqual(star.timestamp, custom_time)

    def test_init_empty_position(self):
        """Test initialization with empty position."""
        with self.assertRaises(ValueError):
            Star([])

    def test_init_invalid_position(self):
        """Test initialization with invalid position."""
        with self.assertRaises(TypeError):
            Star("invalid")

    def test_increment_query_count(self):
        """Test query count increment."""
        initial_count = self.star.query_count
        self.star.increment_query_count()

        self.assertEqual(self.star.query_count, initial_count + 1)

    def test_distance_to(self):
        """Test distance calculation."""
        other_position = np.array([4, 5, 6])
        distance = self.star.distance_to(other_position)

        # Expected distance: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) = 3*sqrt(3)
        expected = np.sqrt(27)
        self.assertAlmostEqual(distance, expected, places=10)

    def test_get_query_info(self):
        """Test query info retrieval."""
        info = self.star.get_query_info()

        self.assertIsInstance(info, dict)
        self.assertEqual(info['id'], self.star.id)
        self.assertEqual(info['position'], self.position)
        self.assertEqual(info['dimension'], len(self.position))
        self.assertEqual(info['query_count'], 0)
        self.assertIn('timestamp', info)
        self.assertIn('metadata', info)

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.star)
        self.assertIn("Star", repr_str)
        self.assertIn(self.star.id, repr_str)
        self.assertIn(str(self.star.query_count), repr_str)

    def test_equality(self):
        """Test star equality."""
        star1 = Star([1, 2, 3], id="test")
        star2 = Star([1, 2, 3], id="test")
        star3 = Star([1, 2, 4], id="test")
        star4 = Star([1, 2, 3], id="different")

        self.assertEqual(star1, star2)
        self.assertNotEqual(star1, star3)
        self.assertNotEqual(star1, star4)

    def test_hash(self):
        """Test star hash."""
        star1 = Star([1, 2, 3], id="test")
        star2 = Star([1, 2, 3], id="test")
        star3 = Star([1, 2, 4], id="test")

        self.assertEqual(hash(star1), hash(star2))
        self.assertNotEqual(hash(star1), hash(star3))

    def test_hash_consistency(self):
        """Test that hash is consistent across calls."""
        hash1 = hash(self.star)
        hash2 = hash(self.star)
        self.assertEqual(hash1, hash2)

if __name__ == '__main__':
    unittest.main()