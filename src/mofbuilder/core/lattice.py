"""
Lattice class for representing crystallographic lattices in MOF structures.

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

from typing import Tuple

import numpy as np


class Lattice:
    """
    Represents a crystallographic lattice for MOF structures.
    
    Attributes:
        matrix (np.ndarray): 3x3 matrix representing the lattice vectors.
        a (float): Length of lattice vector a.
        b (float): Length of lattice vector b.
        c (float): Length of lattice vector c.
        alpha (float): Angle between b and c in degrees.
        beta (float): Angle between a and c in degrees.
        gamma (float): Angle between a and b in degrees.
    """
    
    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
    ):
        """
        Initialize a Lattice object.
        
        Args:
            a: Length of lattice vector a in Angstroms.
            b: Length of lattice vector b in Angstroms.
            c: Length of lattice vector c in Angstroms.
            alpha: Angle between b and c in degrees (default: 90.0).
            beta: Angle between a and c in degrees (default: 90.0).
            gamma: Angle between a and b in degrees (default: 90.0).
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.matrix = self._compute_matrix()
    
    def _compute_matrix(self) -> np.ndarray:
        """
        Compute the lattice matrix from lattice parameters.
        
        Returns:
            3x3 numpy array representing the lattice vectors.
        """
        # Convert angles to radians
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)
        
        # Compute lattice vectors
        # a vector along x-axis
        a_vec = np.array([self.a, 0.0, 0.0])
        
        # b vector in xy-plane
        b_vec = np.array([
            self.b * np.cos(gamma_rad),
            self.b * np.sin(gamma_rad),
            0.0
        ])
        
        # c vector
        c_x = self.c * np.cos(beta_rad)
        c_y = self.c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        c_z = self.c * np.sqrt(1.0 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 
                               2.0 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        c_vec = np.array([c_x, c_y, c_z])
        
        return np.column_stack([a_vec, b_vec, c_vec])
    
    def __repr__(self) -> str:
        """Return string representation of the lattice."""
        return (
            f"Lattice(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, "
            f"alpha={self.alpha:.1f}, beta={self.beta:.1f}, gamma={self.gamma:.1f})"
        )
    
    @property
    def volume(self) -> float:
        """
        Calculate the volume of the unit cell.
        
        Returns:
            Volume in cubic Angstroms.
        """
        return float(np.abs(np.linalg.det(self.matrix)))
    
    def fractional_to_cartesian(self, fractional_coords: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to Cartesian coordinates.
        
        Args:
            fractional_coords: Fractional coordinates as numpy array.
            
        Returns:
            Cartesian coordinates as numpy array.
        """
        return np.dot(self.matrix, fractional_coords)
    
    def cartesian_to_fractional(self, cartesian_coords: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian coordinates to fractional coordinates.
        
        Args:
            cartesian_coords: Cartesian coordinates as numpy array.
            
        Returns:
            Fractional coordinates as numpy array.
        """
        return np.linalg.solve(self.matrix, cartesian_coords)
    
    def get_reciprocal_lattice(self) -> "Lattice":
        """
        Get the reciprocal lattice.
        
        Returns:
            Reciprocal Lattice object.
        """
        reciprocal_matrix = 2 * np.pi * np.linalg.inv(self.matrix).T
        
        # Extract reciprocal lattice parameters
        a_star = np.linalg.norm(reciprocal_matrix[:, 0])
        b_star = np.linalg.norm(reciprocal_matrix[:, 1])
        c_star = np.linalg.norm(reciprocal_matrix[:, 2])
        
        alpha_star = np.degrees(np.arccos(
            np.dot(reciprocal_matrix[:, 1], reciprocal_matrix[:, 2]) / (b_star * c_star)
        ))
        beta_star = np.degrees(np.arccos(
            np.dot(reciprocal_matrix[:, 0], reciprocal_matrix[:, 2]) / (a_star * c_star)
        ))
        gamma_star = np.degrees(np.arccos(
            np.dot(reciprocal_matrix[:, 0], reciprocal_matrix[:, 1]) / (a_star * b_star)
        ))
        
        return Lattice(a_star, b_star, c_star, alpha_star, beta_star, gamma_star)