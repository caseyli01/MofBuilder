"""
Framework class for representing complete MOF structures.

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

from typing import Dict, List, Optional, Set

import numpy as np

from .atom import Atom
from .bond import Bond
from .lattice import Lattice


class Framework:
    """
    Represents a complete MOF framework structure.
    
    Attributes:
        name (str): Name of the framework.
        lattice (Lattice): Crystallographic lattice.
        atoms (List[Atom]): List of atoms in the framework.
        bonds (List[Bond]): List of bonds in the framework.
        properties (Dict): Dictionary to store framework properties.
    """
    
    def __init__(
        self,
        name: str,
        lattice: Lattice,
        atoms: Optional[List[Atom]] = None,
        bonds: Optional[List[Bond]] = None,
    ):
        """
        Initialize a Framework object.
        
        Args:
            name: Name of the framework.
            lattice: Crystallographic lattice.
            atoms: List of atoms (default: empty list).
            bonds: List of bonds (default: empty list).
        """
        self.name = name
        self.lattice = lattice
        self.atoms = atoms or []
        self.bonds = bonds or []
        self.properties: Dict = {}
    
    def __repr__(self) -> str:
        """Return string representation of the framework."""
        return (
            f"Framework(name='{self.name}', "
            f"atoms={len(self.atoms)}, "
            f"bonds={len(self.bonds)})"
        )
    
    def add_atom(self, atom: Atom) -> None:
        """
        Add an atom to the framework.
        
        Args:
            atom: Atom to add.
        """
        self.atoms.append(atom)
    
    def add_bond(self, bond: Bond) -> None:
        """
        Add a bond to the framework.
        
        Args:
            bond: Bond to add.
        """
        self.bonds.append(bond)
    
    def remove_atom(self, atom: Atom) -> None:
        """
        Remove an atom and all associated bonds from the framework.
        
        Args:
            atom: Atom to remove.
        """
        # Remove all bonds containing this atom
        self.bonds = [bond for bond in self.bonds if not bond.contains_atom(atom)]
        # Remove the atom
        if atom in self.atoms:
            self.atoms.remove(atom)
    
    def get_atoms_by_element(self, element: str) -> List[Atom]:
        """
        Get all atoms of a specific element.
        
        Args:
            element: Chemical element symbol.
            
        Returns:
            List of atoms with the specified element.
        """
        return [atom for atom in self.atoms if atom.element == element]
    
    def get_coordination_number(self, atom: Atom) -> int:
        """
        Get the coordination number of an atom.
        
        Args:
            atom: Atom to analyze.
            
        Returns:
            Number of bonds involving the atom.
        """
        return sum(1 for bond in self.bonds if bond.contains_atom(atom))
    
    def get_neighbors(self, atom: Atom) -> List[Atom]:
        """
        Get all neighboring atoms of a given atom.
        
        Args:
            atom: Atom to find neighbors for.
            
        Returns:
            List of neighboring atoms.
        """
        neighbors = []
        for bond in self.bonds:
            if bond.contains_atom(atom):
                neighbors.append(bond.get_other_atom(atom))
        return neighbors
    
    @property
    def composition(self) -> Dict[str, int]:
        """
        Get the elemental composition of the framework.
        
        Returns:
            Dictionary with element symbols as keys and counts as values.
        """
        composition = {}
        for atom in self.atoms:
            composition[atom.element] = composition.get(atom.element, 0) + 1
        return composition
    
    @property
    def formula(self) -> str:
        """
        Get the chemical formula of the framework.
        
        Returns:
            Chemical formula as a string.
        """
        comp = self.composition
        if not comp:
            return ""
        
        # Sort elements alphabetically, but put C and H first if present
        elements = list(comp.keys())
        priority = ["C", "H"]
        sorted_elements = []
        
        for elem in priority:
            if elem in elements:
                sorted_elements.append(elem)
                elements.remove(elem)
        
        sorted_elements.extend(sorted(elements))
        
        formula_parts = []
        for element in sorted_elements:
            count = comp[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        
        return "".join(formula_parts)
    
    def translate(self, vector: np.ndarray) -> None:
        """
        Translate all atoms in the framework.
        
        Args:
            vector: Translation vector.
        """
        for atom in self.atoms:
            atom.translate(vector)
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculate the center of mass of the framework.
        
        Returns:
            Center of mass coordinates.
        """
        if not self.atoms:
            return np.zeros(3)
        
        # Simple implementation assuming equal masses
        positions = np.array([atom.position for atom in self.atoms])
        return np.mean(positions, axis=0)
    
    def center_at_origin(self) -> None:
        """Center the framework at the origin."""
        com = self.get_center_of_mass()
        self.translate(-com)