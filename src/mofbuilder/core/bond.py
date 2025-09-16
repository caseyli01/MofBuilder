"""
Bond class for representing bonds between atoms in MOF structures.

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

from typing import Optional

from .atom import Atom


class Bond:
    """
    Represents a bond between two atoms in a MOF structure.
    
    Attributes:
        atom1 (Atom): First atom in the bond.
        atom2 (Atom): Second atom in the bond.
        bond_order (float): Bond order (1.0 for single, 2.0 for double, etc.).
        bond_type (str): Type of bond (e.g., 'covalent', 'coordination').
        bond_id (Optional[int]): Unique identifier for the bond.
    """
    
    def __init__(
        self,
        atom1: Atom,
        atom2: Atom,
        bond_order: float = 1.0,
        bond_type: str = "covalent",
        bond_id: Optional[int] = None,
    ):
        """
        Initialize a Bond object.
        
        Args:
            atom1: First atom in the bond.
            atom2: Second atom in the bond.
            bond_order: Bond order (default: 1.0).
            bond_type: Type of bond (default: 'covalent').
            bond_id: Unique identifier for the bond.
        """
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_order = bond_order
        self.bond_type = bond_type
        self.bond_id = bond_id
    
    def __repr__(self) -> str:
        """Return string representation of the bond."""
        return (
            f"Bond(atom1={self.atom1.element}, "
            f"atom2={self.atom2.element}, "
            f"bond_order={self.bond_order}, "
            f"bond_type='{self.bond_type}', "
            f"bond_id={self.bond_id})"
        )
    
    @property
    def length(self) -> float:
        """
        Calculate the bond length.
        
        Returns:
            Bond length in Angstroms.
        """
        return self.atom1.distance_to(self.atom2)
    
    def contains_atom(self, atom: Atom) -> bool:
        """
        Check if the bond contains a specific atom.
        
        Args:
            atom: Atom to check for.
            
        Returns:
            True if the bond contains the atom, False otherwise.
        """
        return atom is self.atom1 or atom is self.atom2
    
    def get_other_atom(self, atom: Atom) -> Atom:
        """
        Get the other atom in the bond.
        
        Args:
            atom: One of the atoms in the bond.
            
        Returns:
            The other atom in the bond.
            
        Raises:
            ValueError: If the provided atom is not part of this bond.
        """
        if atom is self.atom1:
            return self.atom2
        elif atom is self.atom2:
            return self.atom1
        else:
            raise ValueError("Provided atom is not part of this bond")