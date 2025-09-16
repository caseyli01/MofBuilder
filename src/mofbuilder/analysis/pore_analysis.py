"""
Pore analysis tools for MOF structures.

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

from typing import Dict, List, Tuple

import numpy as np

from ..core import Framework
from ..utils import get_atomic_radius


class PoreAnalyzer:
    """
    Analyzer for pore properties in MOF structures.
    
    This class provides methods to analyze pore size distribution,
    accessible volume, and pore connectivity in MOF frameworks.
    """
    
    def __init__(self, probe_radius: float = 1.4):
        """
        Initialize the pore analyzer.
        
        Args:
            probe_radius: Radius of the probe molecule in Angstroms (default: 1.4 for N2).
        """
        self.probe_radius = probe_radius
    
    def calculate_accessible_volume(self, framework: Framework, 
                                  grid_spacing: float = 0.2) -> Dict[str, float]:
        """
        Calculate accessible volume using a grid-based approach.
        
        Args:
            framework: Framework to analyze.
            grid_spacing: Grid spacing for volume calculation in Angstroms.
            
        Returns:
            Dictionary with volume information.
        """
        # Create a 3D grid within the unit cell
        lattice = framework.lattice
        
        # Determine grid dimensions
        nx = int(lattice.a / grid_spacing) + 1
        ny = int(lattice.b / grid_spacing) + 1
        nz = int(lattice.c / grid_spacing) + 1
        
        accessible_points = 0
        total_points = nx * ny * nz
        
        # Check each grid point
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Fractional coordinates
                    frac_coords = np.array([i/nx, j/ny, k/nz])
                    
                    # Convert to Cartesian
                    cart_coords = lattice.fractional_to_cartesian(frac_coords)
                    
                    # Check if point is accessible
                    if self._is_point_accessible(cart_coords, framework):
                        accessible_points += 1
        
        # Calculate volumes
        unit_cell_volume = lattice.volume
        accessible_fraction = accessible_points / total_points
        accessible_volume = accessible_fraction * unit_cell_volume
        
        return {
            "total_volume": unit_cell_volume,
            "accessible_volume": accessible_volume,
            "accessible_fraction": accessible_fraction,
            "probe_radius": self.probe_radius,
        }
    
    def _is_point_accessible(self, point: np.ndarray, framework: Framework) -> bool:
        """
        Check if a point is accessible to the probe molecule.
        
        Args:
            point: 3D coordinates of the point.
            framework: Framework to check against.
            
        Returns:
            True if the point is accessible, False otherwise.
        """
        for atom in framework.atoms:
            distance = np.linalg.norm(point - atom.position)
            atom_radius = get_atomic_radius(atom.element)
            
            # Check if probe would overlap with atom
            if distance < (atom_radius + self.probe_radius):
                return False
        
        return True
    
    def find_largest_sphere(self, framework: Framework, 
                           max_iterations: int = 1000) -> Dict[str, any]:
        """
        Find the largest sphere that can fit in the pore.
        
        Args:
            framework: Framework to analyze.
            max_iterations: Maximum number of optimization iterations.
            
        Returns:
            Dictionary with largest sphere information.
        """
        # Start from center of unit cell
        lattice = framework.lattice
        center = lattice.fractional_to_cartesian(np.array([0.5, 0.5, 0.5]))
        
        # Find the largest sphere using a simple search
        max_radius = 0.0
        best_center = center.copy()
        
        # Sample multiple starting points
        for frac_x in np.linspace(0.1, 0.9, 5):
            for frac_y in np.linspace(0.1, 0.9, 5):
                for frac_z in np.linspace(0.1, 0.9, 5):
                    test_center = lattice.fractional_to_cartesian(
                        np.array([frac_x, frac_y, frac_z])
                    )
                    
                    radius = self._find_sphere_radius_at_point(test_center, framework)
                    
                    if radius > max_radius:
                        max_radius = radius
                        best_center = test_center.copy()
        
        return {
            "radius": max_radius,
            "center": best_center,
            "diameter": 2.0 * max_radius,
        }
    
    def _find_sphere_radius_at_point(self, center: np.ndarray, 
                                   framework: Framework) -> float:
        """
        Find the largest sphere radius at a given point.
        
        Args:
            center: Center point for the sphere.
            framework: Framework to check against.
            
        Returns:
            Maximum sphere radius.
        """
        min_distance = float('inf')
        
        for atom in framework.atoms:
            distance = np.linalg.norm(center - atom.position)
            atom_radius = get_atomic_radius(atom.element)
            
            # Distance to atom surface
            surface_distance = distance - atom_radius
            
            if surface_distance < min_distance:
                min_distance = surface_distance
        
        return max(0.0, min_distance)
    
    def pore_size_distribution(self, framework: Framework, 
                             grid_spacing: float = 0.2) -> Dict[str, any]:
        """
        Calculate pore size distribution.
        
        Args:
            framework: Framework to analyze.
            grid_spacing: Grid spacing for analysis.
            
        Returns:
            Dictionary with pore size distribution data.
        """
        lattice = framework.lattice
        
        # Create grid
        nx = int(lattice.a / grid_spacing) + 1
        ny = int(lattice.b / grid_spacing) + 1
        nz = int(lattice.c / grid_spacing) + 1
        
        pore_sizes = []
        
        # Calculate local pore size at each accessible point
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Fractional coordinates
                    frac_coords = np.array([i/nx, j/ny, k/nz])
                    
                    # Convert to Cartesian
                    cart_coords = lattice.fractional_to_cartesian(frac_coords)
                    
                    # Check if point is accessible
                    if self._is_point_accessible(cart_coords, framework):
                        # Calculate local pore size
                        local_size = self._find_sphere_radius_at_point(cart_coords, framework)
                        if local_size > 0:
                            pore_sizes.append(local_size * 2)  # Convert to diameter
        
        if not pore_sizes:
            return {
                "mean_pore_size": 0.0,
                "max_pore_size": 0.0,
                "min_pore_size": 0.0,
                "pore_sizes": [],
            }
        
        pore_sizes = np.array(pore_sizes)
        
        return {
            "mean_pore_size": np.mean(pore_sizes),
            "max_pore_size": np.max(pore_sizes),
            "min_pore_size": np.min(pore_sizes),
            "std_pore_size": np.std(pore_sizes),
            "pore_sizes": pore_sizes.tolist(),
        }