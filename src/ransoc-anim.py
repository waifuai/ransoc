import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy.typing as npt
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque

@dataclass
class Planet:
    """Represents a searchable item in the RANSOC algorithm."""
    position: npt.NDArray[np.float64]
    mass: float = 1.0
    mass_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        self.mass_history.append(self.mass)

@dataclass
class Star:
    """Represents a query in the RANSOC algorithm."""
    position: npt.NDArray[np.float64]
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)

class RANSOCVisualizer:
    """Handles visualization of RANSOC algorithm dynamics."""
    
    def __init__(self, ransoc_instance: 'RANSOC'):
        self.ransoc = ransoc_instance
        self.fig = None
        self.history_length = 50  # Number of past states to show in plots
    
    def plot_current_state(self) -> None:
        """Create a comprehensive visualization of the current state."""
        plt.style.use('seaborn')
        self.fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('RANSOC Algorithm State Visualization', fontsize=16)
        
        # Plot 1: Planet Positions and Masses (2D only)
        ax1 = axes[0, 0]
        if self.ransoc.dimension == 2:
            positions = np.array([p.position for p in self.ransoc.planets])
            masses = np.array([p.mass for p in self.ransoc.planets])
            scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                                s=masses*100, alpha=0.6, 
                                c=masses, cmap='viridis')
            ax1.set_title('Planet Positions and Masses')
            plt.colorbar(scatter, ax=ax1, label='Mass')
        else:
            ax1.text(0.5, 0.5, f'Cannot visualize {self.ransoc.dimension}D space',
                    ha='center', va='center')
        
        # Plot 2: Mass Distribution
        ax2 = axes[0, 1]
        masses = [p.mass for p in self.ransoc.planets]
        sns.histplot(masses, bins=10, ax=ax2)
        ax2.set_title('Mass Distribution')
        ax2.set_xlabel('Mass')
        ax2.set_ylabel('Count')
        
        # Plot 3: Mass History
        ax3 = axes[1, 0]
        for i, planet in enumerate(self.ransoc.planets):
            history = planet.mass_history[-self.history_length:]
            ax3.plot(history, label=f'Planet {i}')
        ax3.set_title('Mass History')
        ax3.set_xlabel('Query Number')
        ax3.set_ylabel('Mass')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Rank Changes
        ax4 = axes[1, 1]
        current_weights = self.ransoc.compute_weights(
            self.ransoc.compute_distances(self.ransoc.last_query)
        )
        ranked_indices = np.argsort(current_weights)[::-1]
        ax4.bar(range(len(ranked_indices)), current_weights[ranked_indices])
        ax4.set_title('Current Rankings by Weight')
        ax4.set_xlabel('Rank')
        ax4.set_ylabel('Weight')
        
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, n_frames: int, query_star: Star) -> FuncAnimation:
        """Create an animation of the algorithm's behavior over multiple queries."""
        if self.ransoc.dimension != 2:
            raise ValueError("Animation only supported for 2D space")
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            self.ransoc.query(query_star)
            positions = np.array([p.position for p in self.ransoc.planets])
            masses = np.array([p.mass for p in self.ransoc.planets])
            
            # Plot planets
            scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                               s=masses*100, alpha=0.6,
                               c=masses, cmap='viridis')
            
            # Plot query star
            ax.scatter(query_star.position[0], query_star.position[1],
                      marker='*', s=200, c='red', label='Query Star')
            
            ax.set_title(f'Query {frame + 1}')
            ax.legend()
            return scatter,
            
        anim = FuncAnimation(fig, update, frames=n_frames, interval=500, blit=True)
        plt.close()
        return anim

class RANSOC:
    """
    Realtime Adaptive Normalization for the Satisfaction of Curiosity (RANSOC) algorithm.
    
    This algorithm balances between returning relevant search results and encouraging
    exploration of the full search space by dynamically adjusting weights of results.
    """
    
    def __init__(self, positions: npt.NDArray[np.float64], alpha: float = 0.1):
        """Initialize RANSOC algorithm."""
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        self.dimension = positions.shape[1]
        self.alpha = alpha
        self.planets = [Planet(pos) for pos in positions]
        self.last_query = None
        self.visualizer = RANSOCVisualizer(self)
    
    def compute_distances(self, star: Star) -> npt.NDArray[np.float64]:
        """Compute Euclidean distances between all planets and the query star."""
        positions = np.array([p.position for p in self.planets])
        return np.linalg.norm(positions - star.position, axis=1)
    
    def compute_weights(self, distances: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute weights for all planets based on distances and masses."""
        masses = np.array([p.mass for p in self.planets])
        return masses / distances
    
    def update_masses(self, hit_index: int) -> None:
        """Update masses of all planets based on the hit planet."""
        n_planets = len(self.planets)
        
        # Update hit planet
        self.planets[hit_index].mass *= (1 - self.alpha)
        
        # Update other planets
        mass_increase = (1 + self.alpha / (n_planets - 1))
        for i, planet in enumerate(self.planets):
            if i != hit_index:
                planet.mass *= mass_increase
            
            # Record mass history
            planet.mass_history.append(planet.mass)
    
    def query(self, star: Star) -> Tuple[Planet, List[Tuple[Planet, float]]]:
        """Process a query and return the most relevant planet along with all rankings."""
        self.last_query = star  # Store for visualization purposes
        
        # Compute distances and weights
        distances = self.compute_distances(star)
        weights = self.compute_weights(distances)
        
        # Find hit planet
        hit_index = np.argmax(weights)
        
        # Update masses
        self.update_masses(hit_index)
        
        # Create sorted rankings
        rankings = [(planet, weight) for planet, weight in zip(self.planets, weights)]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return self.planets[hit_index], rankings
    
    def visualize(self) -> None:
        """Generate visualization of current state."""
        self.visualizer.plot_current_state()
    
    def animate(self, n_frames: int, query_star: Star) -> FuncAnimation:
        """Generate animation of algorithm behavior."""
        return self.visualizer.create_animation(n_frames, query_star)

def demo_ransoc_visualization():
    """Demonstrate RANSOC with visualizations."""
    # Create sample planet positions in 2D space
    positions = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [-1, -1],
        [-2, -2],
        [1, -1],
        [-1, 1],
        [0, 2],
        [2, 0]
    ])
    
    # Initialize RANSOC
    ransoc = RANSOC(positions, alpha=0.1)
    
    # Create a query star
    query_star = Star(np.array([0.5, 0.5]))
    
    # Perform multiple queries
    print("Running queries and generating visualizations...")
    for _ in range(10):
        hit_planet, _ = ransoc.query(query_star)
    
    # Generate static visualization
    ransoc.visualize()
    
    # Generate animation
    anim = ransoc.animate(20, query_star)
    return anim  # Return animation object for display in notebooks

if __name__ == "__main__":
    demo_ransoc_visualization()