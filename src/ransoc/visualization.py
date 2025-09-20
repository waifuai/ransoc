# visualization.py
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
import warnings

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .ransoc import RANSOC, QueryResult
from .planet import Planet
from .star import Star

class RANSOCVisualizer:
    """
    Visualization utilities for RANSOC algorithm.

    Provides static plots for analyzing algorithm behavior, planet distributions,
    and query results without animation.
    """

    def __init__(self, style: str = 'seaborn'):
        """Initialize the visualizer with a matplotlib style."""
        try:
            plt.style.use(style)
        except OSError:
            warnings.warn(f"Style '{style}' not found, using default")
        self.style = style

    def plot_planet_positions(
        self,
        ransoc: RANSOC,
        star: Optional[Star] = None,
        highlight_planet: Optional[Planet] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Figure:
        """
        Plot planet positions in 2D space.

        Args:
            ransoc: RANSOC instance
            star: Optional query star to plot
            highlight_planet: Optional planet to highlight
            figsize: Figure size tuple

        Returns:
            Matplotlib figure object
        """
        if ransoc.dimension != 2:
            raise ValueError("Position plotting only supported for 2D space")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot planets
        positions = np.array([p.position for p in ransoc.planets])
        masses = np.array([p.mass for p in ransoc.planets])
        hit_counts = np.array([p.hit_count for p in ransoc.planets])

        scatter = ax.scatter(
            positions[:, 0], positions[:, 1],
            s=masses * 100,  # Size based on mass
            c=hit_counts,    # Color based on hit count
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )

        # Highlight specific planet if requested
        if highlight_planet:
            highlight_pos = highlight_planet.position
            ax.scatter(
                highlight_pos[0], highlight_pos[1],
                s=200,
                marker='*',
                c='red',
                label=f'Highlighted: {highlight_planet.id}',
                edgecolors='black'
            )

        # Plot query star if provided
        if star:
            ax.scatter(
                star.position[0], star.position[1],
                marker='*',
                s=300,
                c='red',
                label='Query Star',
                edgecolors='black'
            )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Hit Count')

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('Planet Positions and Hit Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_mass_distribution(
        self,
        ransoc: RANSOC,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Figure:
        """Plot the distribution of planet masses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        masses = [p.mass for p in ransoc.planets]

        # Histogram
        if HAS_SEABORN:
            sns.histplot(masses, bins=20, ax=ax1)
        else:
            ax1.hist(masses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Mass')
        ax1.set_ylabel('Count')
        ax1.set_title('Mass Distribution')

        # Box plot
        ax2.boxplot(masses, patch_artist=True)
        ax2.set_ylabel('Mass')
        ax2.set_title('Mass Distribution (Box Plot)')
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = """.3f"""f"""
Mass Statistics:
Mean: {np.mean(masses):.3f}
Std: {np.std(masses):.3f}
Min: {np.min(masses):.3f}
Max: {np.max(masses):.3f}
Total: {np.sum(masses):.3f}
"""
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        return fig

    def plot_mass_history(
        self,
        ransoc: RANSOC,
        planet_ids: Optional[List[str]] = None,
        max_history: int = 100,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """Plot mass history for selected planets."""
        fig, ax = plt.subplots(figsize=figsize)

        if planet_ids is None:
            # Plot top 5 planets by current mass
            planets_to_plot = sorted(ransoc.planets, key=lambda p: p.mass, reverse=True)[:5]
        else:
            planets_to_plot = [p for p in ransoc.planets if p.id in planet_ids]

        for planet in planets_to_plot:
            history = planet.mass_history[-max_history:]  # Last max_history values
            iterations = range(len(history))
            ax.plot(iterations, history, label=f'Planet {planet.id}', marker='o', markersize=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mass')
        ax.set_title('Mass History Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_query_rankings(
        self,
        result: QueryResult,
        top_k: int = 10,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """Plot the rankings from a query result."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Get top k results
        top_results = result.get_top_k(top_k)
        planet_ids = [p.id for p, _ in top_results]
        weights = [w for _, w in top_results]

        # Bar plot
        bars = ax1.bar(range(len(planet_ids)), weights, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Weight')
        ax1.set_title(f'Top {top_k} Query Rankings')
        ax1.set_xticks(range(len(planet_ids)))
        ax1.set_xticklabels(planet_ids, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Highlight the hit planet
        hit_rank = result.get_rank(result.hit_planet)
        if hit_rank and hit_rank <= top_k:
            bars[hit_rank - 1].set_color('red')
            bars[hit_rank - 1].set_label(f'Hit: {result.hit_planet.id}')

        # Cumulative weight distribution
        cumulative_weights = np.cumsum(weights)
        ax2.plot(range(1, len(cumulative_weights) + 1), cumulative_weights,
                marker='o', color='green', linewidth=2)
        ax2.set_xlabel('Number of Results')
        ax2.set_ylabel('Cumulative Weight')
        ax2.set_title('Cumulative Weight Distribution')
        ax2.grid(True, alpha=0.3)

        # Add query info
        info_text = f"""
Query Info:
ID: {result.query.id}
Iteration: {result.iteration}
Hit Planet: {result.hit_planet.id}
Total Mass: {result.total_mass:.3f}
Entropy: {result.entropy:.3f}
"""
        fig.text(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        return fig

    def plot_hit_distribution(
        self,
        ransoc: RANSOC,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Figure:
        """Plot the distribution of hit counts across planets."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        hit_counts = [p.hit_count for p in ransoc.planets]
        planet_ids = [p.id for p in ransoc.planets]

        # Bar plot of hit counts
        bars = ax1.bar(range(len(planet_ids)), hit_counts, color='lightcoral', edgecolor='black')
        ax1.set_xlabel('Planet')
        ax1.set_ylabel('Hit Count')
        ax1.set_title('Hit Count Distribution')
        ax1.set_xticks(range(len(planet_ids)))
        ax1.set_xticklabels(planet_ids, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Pie chart of hit distribution
        ax2.pie(hit_counts, labels=planet_ids, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Hit Count Proportion')

        plt.tight_layout()
        return fig

    def plot_comprehensive_analysis(
        self,
        ransoc: RANSOC,
        figsize: Tuple[int, int] = (16, 12)
    ) -> Figure:
        """Create a comprehensive analysis dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('RANSOC Comprehensive Analysis', fontsize=16)

        # Mass distribution
        masses = [p.mass for p in ransoc.planets]
        if HAS_SEABORN:
            sns.histplot(masses, bins=15, ax=axes[0, 0])
        else:
            axes[0, 0].hist(masses, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Mass Distribution')

        # Hit count distribution
        hit_counts = [p.hit_count for p in ransoc.planets]
        axes[0, 1].bar(range(len(hit_counts)), hit_counts, color='lightcoral')
        axes[0, 1].set_title('Hit Counts')
        axes[0, 1].set_xlabel('Planet Index')

        # Mass vs Hit count scatter
        axes[0, 2].scatter(masses, hit_counts, alpha=0.7, s=50)
        axes[0, 2].set_xlabel('Mass')
        axes[0, 2].set_ylabel('Hit Count')
        axes[0, 2].set_title('Mass vs Hit Count')

        # Recent mass history (if available)
        if ransoc.mass_history:
            recent_history = np.array(ransoc.mass_history[-50:])  # Last 50 iterations
            for i in range(min(5, recent_history.shape[1])):  # Plot first 5 planets
                axes[1, 0].plot(recent_history[:, i], label=f'Planet {i}')
            axes[1, 0].set_title('Recent Mass History')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Mass')
            axes[1, 0].legend()

        # Query statistics over time
        if ransoc.query_history:
            iterations = [r.iteration for r in ransoc.query_history]
            entropies = [r.entropy for r in ransoc.query_history]
            axes[1, 1].plot(iterations, entropies, marker='o', markersize=3)
            axes[1, 1].set_title('Entropy Over Time')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Entropy')

        # Statistics summary
        stats = ransoc.get_statistics()
        stats_text = ".3f"".3f"".3f"".3f"".3f"".3f"".3f"".3f"".3f"".3f"f"""
Statistics:
Planets: {stats.get('n_planets', 0)}
Dimension: {stats.get('dimension', 0)}
Iterations: {stats.get('iteration', 0)}

Mass Stats:
  Mean: {stats['mass_stats']['mean']:.3f}
  Std: {stats['mass_stats']['std']:.3f}
  Min: {stats['mass_stats']['min']:.3f}
  Max: {stats['mass_stats']['max']:.3f}

Hit Stats:
  Mean: {stats['hit_stats']['mean']:.3f}
  Std: {stats['hit_stats']['std']:.3f}
  Min: {stats['hit_stats']['min']:.3f}
  Max: {stats['hit_stats']['max']:.3f}
"""
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('Summary Statistics')
        axes[1, 2].axis('off')

        plt.tight_layout()
        return fig

    def save_plot(self, fig: Figure, filename: str, dpi: int = 300) -> None:
        """Save a matplotlib figure to file."""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

def plot_ransoc_overview(ransoc: RANSOC, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive overview plot of RANSOC state.

    Args:
        ransoc: RANSOC instance to visualize
        save_path: Optional path to save the plot
    """
    visualizer = RANSOCVisualizer()

    if ransoc.dimension == 2:
        fig = visualizer.plot_comprehensive_analysis(ransoc)
    else:
        fig = visualizer.plot_mass_distribution(ransoc)

    if save_path:
        visualizer.save_plot(fig, save_path)
    else:
        plt.show()

def plot_query_analysis(result: QueryResult, save_path: Optional[str] = None) -> None:
    """
    Create detailed analysis plot of a query result.

    Args:
        result: QueryResult to analyze
        save_path: Optional path to save the plot
    """
    visualizer = RANSOCVisualizer()
    fig = visualizer.plot_query_rankings(result)

    if save_path:
        visualizer.save_plot(fig, save_path)
    else:
        plt.show()