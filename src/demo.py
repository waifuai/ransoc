#!/usr/bin/env python3
"""
Enhanced RANSOC Demo

This demo showcases the improved RANSOC implementation with:
- Enhanced Planet and Star classes
- Comprehensive configuration options
- Visualization capabilities
- Batch processing
- Statistics and analytics
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt

# Import the new RANSOC classes
from ransoc import (
    RANSOC, RANSOCConfig, Planet, Star,
    DistanceMetric, plot_ransoc_overview, plot_query_analysis
)

def create_demo_planets() -> List[List[float]]:
    """Create a set of demo planet positions."""
    # Create planets in a 2D space forming interesting clusters
    np.random.seed(42)  # For reproducible results

    # Cluster 1: Around (1, 1)
    cluster1 = np.random.normal([1, 1], 0.2, (5, 2))

    # Cluster 2: Around (4, 4)
    cluster2 = np.random.normal([4, 4], 0.3, (4, 2))

    # Cluster 3: Around (2, 5)
    cluster3 = np.random.normal([2, 5], 0.25, (3, 2))

    # Outliers
    outliers = np.array([[0, 0], [6, 6], [1, 6]])

    # Combine all positions
    all_positions = np.vstack([cluster1, cluster2, cluster3, outliers])

    return all_positions.tolist()

def demo_basic_usage():
    """Demonstrate basic RANSOC usage."""
    print("=" * 60)
    print("RANSOC Basic Usage Demo")
    print("=" * 60)

    # Create planet positions
    positions = create_demo_planets()
    print(f"Created {len(positions)} planets in 2D space")

    # Create RANSOC instance with custom configuration
    config = RANSOCConfig(
        alpha=0.15,
        distance_metric=DistanceMetric.EUCLIDEAN,
        enable_logging=True,
        log_level="INFO"
    )

    ransoc = RANSOC(positions, config=config)
    print(f"Initialized RANSOC with {ransoc.n_planets} planets in {ransoc.dimension}D")

    # Create a query star
    query_star = Star([3, 3], id="demo_query_1")
    print(f"Created query star at position {query_star.position}")

    # Process the query
    result = ransoc.query(query_star)

    print("\nQuery Results:")
    print(f"  Hit Planet: {result.hit_planet.id}")
    print(".3f")
    print(f"  Iteration: {result.iteration}")
    print(".3f")
    print(".3f")

    # Show top 5 rankings
    print("\nTop 5 Rankings:")
    for i, (planet, weight) in enumerate(result.get_top_k(5), 1):
        print(".3f")

    # Show statistics
    stats = ransoc.get_statistics()
    print("\nCurrent Statistics:")
    print(".3f")
    print(".3f")

    return ransoc, result

def demo_advanced_features():
    """Demonstrate advanced RANSOC features."""
    print("\n" + "=" * 60)
    print("RANSOC Advanced Features Demo")
    print("=" * 60)

    # Create a new RANSOC instance
    positions = create_demo_planets()
    config = RANSOCConfig(
        alpha=0.1,
        distance_metric=DistanceMetric.MANHATTAN,  # Try different distance metric
        enable_history_tracking=True,
        max_history_length=100
    )

    ransoc = RANSOC(positions, config=config)

    # Create multiple queries
    query_positions = [[0, 0], [3, 3], [5, 5], [1, 4], [4, 1]]

    print(f"Processing {len(query_positions)} queries...")

    results = []
    for i, pos in enumerate(query_positions):
        star = Star(pos, id=f"query_{i+1}")
        result = ransoc.query(star)
        results.append(result)
        print(f"  Query {i+1}: Hit planet {result.hit_planet.id} (position {pos})")

    # Show evolution of statistics
    stats = ransoc.get_statistics()
    print("\nFinal Statistics:")
    print(f"  Total Queries: {stats['iteration']}")
    print(".3f")
    print(".3f")
    print(f"  Hit Count Range: {stats['hit_stats']['min']} - {stats['hit_stats']['max']}")

    # Show planets sorted by different criteria
    print("\nPlanets by Mass (descending):")
    for i, planet in enumerate(ransoc.get_planets_sorted_by_mass()[:3], 1):
        print(".3f")

    print("\nPlanets by Hit Count (descending):")
    for i, planet in enumerate(ransoc.get_planets_sorted_by_hits()[:3], 1):
        print(f"  {i}. {planet.id}: {planet.hit_count} hits")

    return ransoc, results

def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "=" * 60)
    print("RANSOC Batch Processing Demo")
    print("=" * 60)

    positions = create_demo_planets()
    ransoc = RANSOC(positions)

    # Create batch of query stars
    batch_stars = [
        Star([1, 1], id="batch_1"),
        Star([2, 2], id="batch_2"),
        Star([3, 3], id="batch_3"),
        Star([4, 4], id="batch_4"),
        Star([5, 5], id="batch_5")
    ]

    print(f"Processing {len(batch_stars)} queries in batch...")

    # Process batch
    batch_results = ransoc.batch_query(batch_stars)

    print("\nBatch Results:")
    for i, result in enumerate(batch_results, 1):
        print(f"  Query {i}: {result.hit_planet.id} "
              ".3f")

    # Show statistics after batch processing
    stats = ransoc.get_statistics()
    print("\nPost-batch Statistics:")
    print(f"  Total Queries: {stats['iteration']}")
    print(".3f")

    return ransoc, batch_results

def demo_different_distance_metrics():
    """Demonstrate different distance metrics."""
    print("\n" + "=" * 60)
    print("RANSOC Distance Metrics Demo")
    print("=" * 60)

    positions = create_demo_planets()
    query_pos = [3, 3]

    metrics = [
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
        DistanceMetric.COSINE
    ]

    results = {}

    for metric in metrics:
        print(f"\nTesting {metric.value} distance:")

        config = RANSOCConfig(
            alpha=0.1,
            distance_metric=metric,
            enable_logging=False  # Reduce log noise
        )

        ransoc = RANSOC(positions, config=config)
        star = Star(query_pos, id=f"test_{metric.value}")

        result = ransoc.query(star)
        results[metric.value] = result

        print(f"  Hit Planet: {result.hit_planet.id}")
        print(".3f")

    # Compare results
    print("\nComparison of Results:")
    for metric_name, result in results.items():
        print(f"  {metric_name}: {result.hit_planet.id} "
              ".3f")

    return results

def main():
    """Main demo function."""
    print("RANSOC Enhanced Demo")
    print("====================")

    try:
        # Run all demos
        ransoc1, result1 = demo_basic_usage()
        ransoc2, results2 = demo_advanced_features()
        ransoc3, batch_results = demo_batch_processing()
        metric_results = demo_different_distance_metrics()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

        # Optional: Create visualizations (if matplotlib is available)
        try:
            print("\nGenerating visualizations...")

            # Create overview plot
            plot_ransoc_overview(ransoc2, "ransoc_demo_overview.png")
            print("  Created overview plot: ransoc_demo_overview.png")

            # Create query analysis plot
            plot_query_analysis(result1, "ransoc_query_analysis.png")
            print("  Created query analysis plot: ransoc_query_analysis.png")

            print("  Visualization files saved in current directory")

        except ImportError as e:
            print(f"\nVisualization skipped: {e}")
        except Exception as e:
            print(f"\nVisualization error: {e}")

        print("\nThank you for exploring RANSOC!")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()