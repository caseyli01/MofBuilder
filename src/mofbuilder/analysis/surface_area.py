"""
Surface area calculation tools for MOF structures.

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

from typing import Dict

import numpy as np

from ..core import Framework
from ..utils import get_atomic_radius


class SurfaceAreaCalculator:
    """
    Calculator for surface area properties of MOF structures.
    
    This class provides methods to calculate BET surface area,
    geometric surface area, and other surface-related properties.
    """
    
    def __init__(self, probe_radius: float = 1.4):
        """
        Initialize the surface area calculator.
        
        Args:
            probe_radius: Radius of the probe molecule in Angstroms (default: 1.4 for N2).
        """
        self.probe_radius = probe_radius
    
    def calculate_geometric_surface_area(self, framework: Framework, 
                                       grid_spacing: float = 0.2) -> Dict[str, float]:
        """
        Calculate geometric surface area using a grid-based approach.
        
        Args:
            framework: Framework to analyze.
            grid_spacing: Grid spacing for surface calculation in Angstroms.
            
        Returns:
            Dictionary with surface area information.
        """
        lattice = framework.lattice
        
        # Create a 3D grid within the unit cell
        nx = int(lattice.a / grid_spacing) + 1
        ny = int(lattice.b / grid_spacing) + 1
        nz = int(lattice.c / grid_spacing) + 1
        
        surface_points = 0
        
        # Check each grid point
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Fractional coordinates
                    frac_coords = np.array([i/nx, j/ny, k/nz])
                    
                    # Convert to Cartesian
                    cart_coords = lattice.fractional_to_cartesian(frac_coords)
                    
                    # Check if point is on the surface
                    if self._is_surface_point(cart_coords, framework, grid_spacing):
                        surface_points += 1
        
        # Calculate surface area
        grid_area = grid_spacing ** 2
        total_surface_area = surface_points * grid_area
        
        # Normalize by unit cell volume for specific surface area
        specific_surface_area = total_surface_area / lattice.volume
        
        return {
            "geometric_surface_area": total_surface_area,
            "specific_surface_area": specific_surface_area,
            "probe_radius": self.probe_radius,
        }
    
    def _is_surface_point(self, point: np.ndarray, framework: Framework, 
                         grid_spacing: float) -> bool:
        """
        Check if a point is on the accessible surface.
        
        Args:
            point: 3D coordinates of the point.
            framework: Framework to check against.
            grid_spacing: Grid spacing used.
            
        Returns:
            True if the point is on the surface, False otherwise.
        """
        # Check if the point itself is accessible
        if not self._is_point_accessible(point, framework):
            return False
        
        # Check neighboring points - if any are inaccessible, this is a surface point
        directions = [
            [grid_spacing, 0, 0], [-grid_spacing, 0, 0],
            [0, grid_spacing, 0], [0, -grid_spacing, 0],
            [0, 0, grid_spacing], [0, 0, -grid_spacing]
        ]
        
        for direction in directions:
            neighbor_point = point + np.array(direction)
            if not self._is_point_accessible(neighbor_point, framework):
                return True
        
        return False
    
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
    
    def calculate_bet_surface_area(self, framework: Framework, 
                                 adsorption_data: Dict = None) -> Dict[str, float]:
        """
        Calculate BET surface area from adsorption data.
        
        Args:
            framework: Framework to analyze.
            adsorption_data: Dictionary with pressure and adsorption amount data.
            
        Returns:
            Dictionary with BET surface area information.
            
        Note:
            This is a placeholder implementation. Real BET calculation
            requires experimental or simulated adsorption isotherms.
        """
        if adsorption_data is None:
            # Return estimated surface area based on geometric calculation
            geometric_data = self.calculate_geometric_surface_area(framework)
            
            # Rough estimate: BET surface area is typically 1.5-3x geometric surface area
            estimated_bet = geometric_data["specific_surface_area"] * 2.0
            
            return {
                "bet_surface_area": estimated_bet,
                "estimated": True,
                "method": "geometric_estimation",
            }
        
        # Placeholder for actual BET calculation
        # In practice, this would implement the BET equation:
        # 1/[V(P0/P - 1)] = 1/(VmC) + (C-1)/(VmC) * (P/P0)
        
        return {
            "bet_surface_area": 0.0,
            "estimated": False,
            "method": "bet_analysis",
            "note": "BET calculation not implemented - requires adsorption isotherm data",
        }
    
    def calculate_accessible_surface_area(self, framework: Framework) -> Dict[str, float]:
        """
        Calculate accessible surface area using a probe-rolling algorithm.
        
        Args:
            framework: Framework to analyze.
            
        Returns:
            Dictionary with accessible surface area information.
        """
        # This is a simplified implementation
        # A full implementation would use algorithms like the Shrake-Rupley method
        
        geometric_data = self.calculate_geometric_surface_area(framework)
        
        # Accessible surface area is typically smaller than geometric surface area
        accessible_area = geometric_data["geometric_surface_area"] * 0.8
        
        return {
            "accessible_surface_area": accessible_area,
            "probe_radius": self.probe_radius,
            "method": "simplified_calculation",
        }