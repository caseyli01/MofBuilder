"""
Unit tests for core module.

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

import numpy as np
import pytest

from mofbuilder.core import Atom, Bond, Framework, Lattice


class TestAtom:
    """Test cases for Atom class."""
    
    def test_atom_creation(self):
        """Test basic atom creation."""
        atom = Atom("C", (0.0, 0.0, 0.0))
        assert atom.element == "C"
        assert np.allclose(atom.position, [0.0, 0.0, 0.0])
        assert atom.charge == 0.0
        assert atom.atom_id is None
    
    def test_atom_with_charge_and_id(self):
        """Test atom creation with charge and ID."""
        atom = Atom("O", (1.0, 2.0, 3.0), charge=-0.5, atom_id=42)
        assert atom.element == "O"
        assert np.allclose(atom.position, [1.0, 2.0, 3.0])
        assert atom.charge == -0.5
        assert atom.atom_id == 42
    
    def test_distance_calculation(self):
        """Test distance calculation between atoms."""
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (3.0, 4.0, 0.0))
        
        distance = atom1.distance_to(atom2)
        assert np.isclose(distance, 5.0)
    
    def test_atom_translation(self):
        """Test atom translation."""
        atom = Atom("H", (1.0, 1.0, 1.0))
        atom.translate((2.0, -1.0, 3.0))
        
        assert np.allclose(atom.position, [3.0, 0.0, 4.0])
    
    def test_atom_repr(self):
        """Test atom string representation."""
        atom = Atom("C", (0.0, 1.0, 2.0), charge=0.1, atom_id=1)
        repr_str = repr(atom)
        
        assert "Atom" in repr_str
        assert "C" in repr_str
        assert "0.1" in repr_str


class TestBond:
    """Test cases for Bond class."""
    
    def test_bond_creation(self):
        """Test basic bond creation."""
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        
        bond = Bond(atom1, atom2)
        assert bond.atom1 is atom1
        assert bond.atom2 is atom2
        assert bond.bond_order == 1.0
        assert bond.bond_type == "covalent"
    
    def test_bond_length(self):
        """Test bond length calculation."""
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (3.0, 4.0, 0.0))
        
        bond = Bond(atom1, atom2)
        assert np.isclose(bond.length, 5.0)
    
    def test_bond_contains_atom(self):
        """Test checking if bond contains an atom."""
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        atom3 = Atom("O", (3.0, 0.0, 0.0))
        
        bond = Bond(atom1, atom2)
        assert bond.contains_atom(atom1)
        assert bond.contains_atom(atom2)
        assert not bond.contains_atom(atom3)
    
    def test_get_other_atom(self):
        """Test getting the other atom in a bond."""
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        
        bond = Bond(atom1, atom2)
        assert bond.get_other_atom(atom1) is atom2
        assert bond.get_other_atom(atom2) is atom1
    
    def test_get_other_atom_error(self):
        """Test error when atom is not in bond."""
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        atom3 = Atom("O", (3.0, 0.0, 0.0))
        
        bond = Bond(atom1, atom2)
        with pytest.raises(ValueError):
            bond.get_other_atom(atom3)


class TestLattice:
    """Test cases for Lattice class."""
    
    def test_cubic_lattice(self):
        """Test cubic lattice creation."""
        lattice = Lattice(10.0, 10.0, 10.0)
        
        assert lattice.a == 10.0
        assert lattice.b == 10.0
        assert lattice.c == 10.0
        assert lattice.alpha == 90.0
        assert lattice.beta == 90.0
        assert lattice.gamma == 90.0
        assert np.isclose(lattice.volume, 1000.0)
    
    def test_non_cubic_lattice(self):
        """Test non-cubic lattice creation."""
        lattice = Lattice(5.0, 6.0, 7.0, 80.0, 85.0, 95.0)
        
        assert lattice.a == 5.0
        assert lattice.b == 6.0
        assert lattice.c == 7.0
        assert lattice.alpha == 80.0
        assert lattice.beta == 85.0
        assert lattice.gamma == 95.0
        assert lattice.volume > 0  # Should be positive
    
    def test_coordinate_conversion(self):
        """Test coordinate conversion."""
        lattice = Lattice(10.0, 10.0, 10.0)
        
        # Test fractional to Cartesian
        frac_coords = np.array([0.5, 0.5, 0.5])
        cart_coords = lattice.fractional_to_cartesian(frac_coords)
        assert np.allclose(cart_coords, [5.0, 5.0, 5.0])
        
        # Test Cartesian to fractional
        frac_back = lattice.cartesian_to_fractional(cart_coords)
        assert np.allclose(frac_back, frac_coords)
    
    def test_reciprocal_lattice(self):
        """Test reciprocal lattice calculation."""
        lattice = Lattice(10.0, 10.0, 10.0)
        reciprocal = lattice.get_reciprocal_lattice()
        
        # For cubic lattice, reciprocal parameters should be 2π/a
        expected = 2 * np.pi / 10.0
        assert np.isclose(reciprocal.a, expected, rtol=1e-3)
        assert np.isclose(reciprocal.b, expected, rtol=1e-3)
        assert np.isclose(reciprocal.c, expected, rtol=1e-3)


class TestFramework:
    """Test cases for Framework class."""
    
    def test_framework_creation(self):
        """Test basic framework creation."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        assert framework.name == "test_mof"
        assert framework.lattice is lattice
        assert len(framework.atoms) == 0
        assert len(framework.bonds) == 0
    
    def test_add_atoms(self):
        """Test adding atoms to framework."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        
        framework.add_atom(atom1)
        framework.add_atom(atom2)
        
        assert len(framework.atoms) == 2
        assert atom1 in framework.atoms
        assert atom2 in framework.atoms
    
    def test_add_bonds(self):
        """Test adding bonds to framework."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        bond = Bond(atom1, atom2)
        
        framework.add_atom(atom1)
        framework.add_atom(atom2)
        framework.add_bond(bond)
        
        assert len(framework.bonds) == 1
        assert bond in framework.bonds
    
    def test_composition(self):
        """Test framework composition calculation."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        framework.add_atom(Atom("C", (0.0, 0.0, 0.0)))
        framework.add_atom(Atom("C", (1.0, 0.0, 0.0)))
        framework.add_atom(Atom("N", (2.0, 0.0, 0.0)))
        framework.add_atom(Atom("O", (3.0, 0.0, 0.0)))
        framework.add_atom(Atom("O", (4.0, 0.0, 0.0)))
        
        composition = framework.composition
        assert composition["C"] == 2
        assert composition["N"] == 1
        assert composition["O"] == 2
    
    def test_formula(self):
        """Test chemical formula generation."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        framework.add_atom(Atom("C", (0.0, 0.0, 0.0)))
        framework.add_atom(Atom("H", (1.0, 0.0, 0.0)))
        framework.add_atom(Atom("H", (2.0, 0.0, 0.0)))
        framework.add_atom(Atom("N", (3.0, 0.0, 0.0)))
        
        formula = framework.formula
        # Should be CH2N (C and H first, then alphabetical)
        assert formula == "CH2N"
    
    def test_get_atoms_by_element(self):
        """Test getting atoms by element."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        c_atom = Atom("C", (0.0, 0.0, 0.0))
        n_atom = Atom("N", (1.0, 0.0, 0.0))
        
        framework.add_atom(c_atom)
        framework.add_atom(n_atom)
        
        carbon_atoms = framework.get_atoms_by_element("C")
        assert len(carbon_atoms) == 1
        assert carbon_atoms[0] is c_atom
        
        oxygen_atoms = framework.get_atoms_by_element("O")
        assert len(oxygen_atoms) == 0
    
    def test_center_of_mass(self):
        """Test center of mass calculation."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        framework.add_atom(Atom("C", (0.0, 0.0, 0.0)))
        framework.add_atom(Atom("C", (2.0, 0.0, 0.0)))
        
        com = framework.get_center_of_mass()
        assert np.allclose(com, [1.0, 0.0, 0.0])
    
    def test_remove_atom(self):
        """Test removing atom and associated bonds."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        atom1 = Atom("C", (0.0, 0.0, 0.0))
        atom2 = Atom("N", (1.5, 0.0, 0.0))
        bond = Bond(atom1, atom2)
        
        framework.add_atom(atom1)
        framework.add_atom(atom2)
        framework.add_bond(bond)
        
        framework.remove_atom(atom1)
        
        assert len(framework.atoms) == 1
        assert atom1 not in framework.atoms
        assert len(framework.bonds) == 0  # Bond should be removed too