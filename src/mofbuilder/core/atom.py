"""
Atom class for representing atoms in MOF structures.

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

from typing import List, Optional, Tuple

import numpy as np


class Atom:
    """
    Represents an atom in a MOF structure.
    
    Attributes:
        element (str): The chemical element symbol.
        position (np.ndarray): The 3D coordinates of the atom.
        charge (float): The partial charge of the atom.
        atom_id (Optional[int]): Unique identifier for the atom.
    """
    
    def __init__(
        self,
        element: str,
        position: Tuple[float, float, float],
        charge: float = 0.0,
        atom_id: Optional[int] = None,
    ):
        """
        Initialize an Atom object.
        
        Args:
            element: Chemical element symbol (e.g., 'C', 'N', 'O').
            position: 3D coordinates as (x, y, z) tuple.
            charge: Partial charge of the atom (default: 0.0).
            atom_id: Unique identifier for the atom.
        """
        self.element = element
        self.position = np.array(position, dtype=float)
        self.charge = charge
        self.atom_id = atom_id
    
    def __repr__(self) -> str:
        """Return string representation of the atom."""
        return (
            f"Atom(element='{self.element}', "
            f"position={self.position.tolist()}, "
            f"charge={self.charge}, "
            f"atom_id={self.atom_id})"
        )
    
    def distance_to(self, other: "Atom") -> float:
        """
        Calculate the distance to another atom.
        
        Args:
            other: Another Atom object.
            
        Returns:
            Distance between atoms in Angstroms.
        """
        return float(np.linalg.norm(self.position - other.position))
    
    def translate(self, vector: Tuple[float, float, float]) -> None:
        """
        Translate the atom by a given vector.
        
        Args:
            vector: Translation vector as (x, y, z) tuple.
        """
        self.position += np.array(vector, dtype=float)