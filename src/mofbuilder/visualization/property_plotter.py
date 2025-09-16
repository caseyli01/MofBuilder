"""
Property plotter for MOF analysis results.

Copyright (C) 2024 MofBuilder Contributors

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class PropertyPlotter:
    """
    Plotter for MOF property analysis results.
    
    This class provides methods to visualize analysis results such as
    pore size distributions, surface area data, and other properties.
    """
    
    def __init__(self):
        """Initialize the property plotter."""
        pass
    
    def plot_pore_size_distribution(self, pore_data: Dict,
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot pore size distribution.
        
        Args:
            pore_data: Dictionary with pore analysis results.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pore_sizes = pore_data.get('pore_sizes', [])
        
        if not pore_sizes:
            ax.text(0.5, 0.5, 'No pore size data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Pore Size Distribution')
            return fig
        
        # Create histogram
        bins = np.linspace(min(pore_sizes), max(pore_sizes), 30)
        n, bins, patches = ax.hist(pore_sizes, bins=bins, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_size = pore_data.get('mean_pore_size', 0)
        max_size = pore_data.get('max_pore_size', 0)
        
        ax.axvline(mean_size, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_size:.2f} Å')
        ax.axvline(max_size, color='green', linestyle='--', linewidth=2,
                  label=f'Max: {max_size:.2f} Å')
        
        ax.set_xlabel('Pore Diameter (Å)')
        ax.set_ylabel('Frequency')
        ax.set_title('Pore Size Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_coordination_analysis(self, coord_data: Dict,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot coordination number analysis.
        
        Args:
            coord_data: Dictionary with coordination analysis results.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        per_atom_data = coord_data.get('per_atom', {})
        
        if not per_atom_data:
            ax1.text(0.5, 0.5, 'No coordination data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No coordination data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Extract data
        coord_numbers = [data['coordination_number'] for data in per_atom_data.values()]
        elements = [data['atom'] for data in per_atom_data.values()]
        
        # Plot 1: Coordination number distribution
        unique_coords, counts = np.unique(coord_numbers, return_counts=True)
        ax1.bar(unique_coords, counts, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_xlabel('Coordination Number')
        ax1.set_ylabel('Number of Atoms')
        ax1.set_title('Coordination Number Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coordination by element
        element_coord = {}
        for element, coord in zip(elements, coord_numbers):
            if element not in element_coord:
                element_coord[element] = []
            element_coord[element].append(coord)
        
        for element in element_coord:
            element_coord[element] = np.mean(element_coord[element])
        
        if element_coord:
            elem_names = list(element_coord.keys())
            avg_coords = list(element_coord.values())
            
            bars = ax2.bar(elem_names, avg_coords, alpha=0.7, color='lightgreen', 
                          edgecolor='black')
            ax2.set_xlabel('Element')
            ax2.set_ylabel('Average Coordination Number')
            ax2.set_title('Average Coordination by Element')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, coord in zip(bars, avg_coords):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{coord:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_surface_area_comparison(self, surface_data: List[Dict],
                                   labels: List[str],
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot comparison of surface area data.
        
        Args:
            surface_data: List of dictionaries with surface area results.
            labels: Labels for each dataset.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if not surface_data or not labels:
            ax.text(0.5, 0.5, 'No surface area data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract surface area values
        geometric_areas = []
        accessible_areas = []
        bet_areas = []
        
        for data in surface_data:
            geometric_areas.append(data.get('geometric_surface_area', 0))
            accessible_areas.append(data.get('accessible_surface_area', 0))
            bet_areas.append(data.get('bet_surface_area', 0))
        
        x = np.arange(len(labels))
        width = 0.25
        
        # Create grouped bar chart
        if any(geometric_areas):
            ax.bar(x - width, geometric_areas, width, label='Geometric', alpha=0.8)
        if any(accessible_areas):
            ax.bar(x, accessible_areas, width, label='Accessible', alpha=0.8)
        if any(bet_areas):
            ax.bar(x + width, bet_areas, width, label='BET', alpha=0.8)
        
        ax.set_xlabel('Frameworks')
        ax.set_ylabel('Surface Area (m²/g)')
        ax.set_title('Surface Area Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_property_correlation(self, x_data: List[float], y_data: List[float],
                                x_label: str, y_label: str,
                                point_labels: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot correlation between two properties.
        
        Args:
            x_data: X-axis data values.
            y_data: Y-axis data values.
            x_label: Label for X-axis.
            y_label: Label for Y-axis.
            point_labels: Optional labels for data points.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if not x_data or not y_data:
            ax.text(0.5, 0.5, 'No data available for correlation plot', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.7, s=50, edgecolors='black')
        
        # Add point labels if provided
        if point_labels:
            for i, label in enumerate(point_labels):
                if i < len(x_data) and i < len(y_data):
                    ax.annotate(label, (x_data[i], y_data[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.8)
        
        # Add correlation coefficient
        if len(x_data) > 1:
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add trend line
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "r--", alpha=0.8)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} vs {x_label}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_ring_analysis(self, ring_data: Dict,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot ring size analysis.
        
        Args:
            ring_data: Dictionary with ring analysis results.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ring_counts = ring_data.get('ring_counts', {})
        
        if not ring_counts:
            ax.text(0.5, 0.5, 'No ring data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ring Size Distribution')
            return fig
        
        ring_sizes = list(ring_counts.keys())
        counts = list(ring_counts.values())
        
        bars = ax.bar(ring_sizes, counts, alpha=0.7, color='gold', edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        ax.set_xlabel('Ring Size')
        ax.set_ylabel('Number of Rings')
        ax.set_title('Ring Size Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add total count
        total_rings = ring_data.get('total_rings', 0)
        ax.text(0.95, 0.95, f'Total rings: {total_rings}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save a plot to file.
        
        Args:
            fig: Matplotlib figure to save.
            filename: Output filename.
            dpi: Resolution in dots per inch.
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)