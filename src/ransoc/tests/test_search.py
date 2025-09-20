import unittest
import numpy as np
import warnings
from ransoc.planet import Planet
from ransoc.star import Star
from ransoc.search import hit
from ransoc.ransoc import RANSOC, RANSOCConfig, DistanceMetric, ValidationError

class TestLegacySearch(unittest.TestCase):
    """Test the legacy search function for backward compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.star = Star([0, 0, 0])
        self.planets = [
            Planet([1, 1, 1], mass=2.0),
            Planet([2, 2, 2], mass=1.0),
            Planet([3, 3, 3], mass=0.5)
        ]

    def test_hit_function(self):
        """Test the legacy hit function for backward compatibility."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            hit_planet, updated_planets = hit(self.planets, self.star, alpha=0.2)

            # Check if the correct planet is hit (planet 0 in this case)
            self.assertEqual(hit_planet, self.planets[0])

            # Check if masses are updated correctly
            self.assertAlmostEqual(updated_planets[0].mass, 2.0 * (1 - 0.2))
            self.assertAlmostEqual(updated_planets[1].mass, 1.0 * (1 + 0.2 / 2))
            self.assertAlmostEqual(updated_planets[2].mass, 0.5 * (1 + 0.2 / 2))

    def test_hit_function_default_alpha(self):
        """Test legacy hit function with default alpha."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            hit_planet, updated_planets = hit(self.planets, self.star)

            # Check if the correct planet is hit
            self.assertEqual(hit_planet, self.planets[0])

            # Check if masses are updated correctly with default alpha=0.1
            self.assertAlmostEqual(updated_planets[0].mass, 2.0 * (1 - 0.1))
            self.assertAlmostEqual(updated_planets[1].mass, 1.0 * (1 + 0.1 / 2))
            self.assertAlmostEqual(updated_planets[2].mass, 0.5 * (1 + 0.1 / 2))

class TestRANSOC(unittest.TestCase):
    """Test the new RANSOC class."""

    def setUp(self):
        """Set up test fixtures."""
        self.positions = [[1, 1], [2, 2], [3, 3]]
        self.config = RANSOCConfig(alpha=0.2, enable_logging=False)

    def test_init_with_list_positions(self):
        """Test initialization with list of positions."""
        ransoc = RANSOC(self.positions, self.config)
        self.assertEqual(ransoc.n_planets, 3)
        self.assertEqual(ransoc.dimension, 2)
        self.assertEqual(ransoc.config.alpha, 0.2)

    def test_init_with_numpy_positions(self):
        """Test initialization with numpy array positions."""
        positions_array = np.array(self.positions)
        ransoc = RANSOC(positions_array, self.config)
        self.assertEqual(ransoc.n_planets, 3)
        self.assertEqual(ransoc.dimension, 2)

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = RANSOCConfig(
            alpha=0.3,
            distance_metric=DistanceMetric.MANHATTAN,
            enable_history_tracking=False
        )
        ransoc = RANSOC(self.positions, config)
        self.assertEqual(ransoc.config.alpha, 0.3)
        self.assertEqual(ransoc.config.distance_metric, DistanceMetric.MANHATTAN)
        self.assertFalse(ransoc.config.enable_history_tracking)

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with self.assertRaises(Exception):  # Should raise ConfigurationError
            config = RANSOCConfig(alpha=1.5)
            RANSOC(self.positions, config)

    def test_init_empty_positions(self):
        """Test initialization with empty positions."""
        with self.assertRaises(Exception):  # Should raise ValidationError
            RANSOC([], self.config)

    def test_query_basic(self):
        """Test basic query functionality."""
        ransoc = RANSOC(self.positions, self.config)
        star = Star([0, 0])

        result = ransoc.query(star)

        self.assertIsNotNone(result.hit_planet)
        self.assertIsInstance(result.rankings, list)
        self.assertEqual(len(result.rankings), 3)
        self.assertEqual(result.query, star)
        self.assertEqual(result.iteration, 1)

    def test_query_updates_masses(self):
        """Test that query updates planet masses."""
        ransoc = RANSOC(self.positions, self.config)
        star = Star([0, 0])

        # Store initial masses
        initial_masses = [p.mass for p in ransoc.planets]

        # Process query
        result = ransoc.query(star)

        # Check that at least one mass changed
        masses_changed = any(
            abs(old - new) > 1e-10
            for old, new in zip(initial_masses, [p.mass for p in ransoc.planets])
        )
        self.assertTrue(masses_changed)

    def test_batch_query(self):
        """Test batch query functionality."""
        ransoc = RANSOC(self.positions, self.config)
        stars = [Star([0, 0]), Star([1, 1]), Star([2, 2])]

        results = ransoc.batch_query(stars)

        self.assertEqual(len(results), 3)
        self.assertEqual(ransoc.iteration, 3)

        for i, result in enumerate(results):
            self.assertEqual(result.iteration, i + 1)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        ransoc = RANSOC(self.positions, self.config)
        star = Star([0, 0])

        # Process a few queries
        for _ in range(3):
            ransoc.query(star)

        stats = ransoc.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['n_planets'], 3)
        self.assertEqual(stats['dimension'], 2)
        self.assertEqual(stats['iteration'], 3)
        self.assertIn('mass_stats', stats)
        self.assertIn('hit_stats', stats)

    def test_reset(self):
        """Test reset functionality."""
        ransoc = RANSOC(self.positions, self.config)
        star = Star([0, 0])

        # Process some queries
        for _ in range(5):
            ransoc.query(star)

        # Reset
        ransoc.reset()

        # Check reset state
        self.assertEqual(ransoc.iteration, 0)
        self.assertEqual(len(ransoc.query_history), 0)
        self.assertEqual(len(ransoc.mass_history), 0)

        for planet in ransoc.planets:
            self.assertEqual(planet.mass, 1.0)
            self.assertEqual(planet.hit_count, 0)

    def test_different_distance_metrics(self):
        """Test different distance metrics."""
        star = Star([0, 0])

        for metric in DistanceMetric:
            config = RANSOCConfig(
                alpha=0.1,
                distance_metric=metric,
                enable_logging=False
            )
            ransoc = RANSOC(self.positions, config)

            result = ransoc.query(star)

            # Basic validation that query works with different metrics
            self.assertIsNotNone(result.hit_planet)
            self.assertEqual(len(result.rankings), 3)
            self.assertEqual(result.query, star)

    def test_query_dimension_mismatch(self):
        """Test query with mismatched dimensions."""
        ransoc = RANSOC(self.positions, self.config)  # 2D planets
        star_3d = Star([0, 0, 0])  # 3D query

        with self.assertRaises(ValidationError):
            ransoc.query(star_3d)

if __name__ == '__main__':
    unittest.main()