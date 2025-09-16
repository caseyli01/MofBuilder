"""
Unit tests for utils module.

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

from mofbuilder.utils import (
    get_atomic_mass,
    get_atomic_radius,
    element_info,
    distance_matrix,
    find_neighbors,
    rotation_matrix,
)


class TestPeriodicTable:
    """Test cases for periodic table utilities."""
    
    def test_get_atomic_mass(self):
        """Test atomic mass retrieval."""
        assert np.isclose(get_atomic_mass("C"), 12.011)
        assert np.isclose(get_atomic_mass("N"), 14.007)
        assert np.isclose(get_atomic_mass("O"), 15.999)
    
    def test_get_atomic_mass_unknown(self):
        """Test error for unknown element."""
        with pytest.raises(KeyError):
            get_atomic_mass("Xx")
    
    def test_get_atomic_radius(self):
        """Test atomic radius retrieval."""
        assert np.isclose(get_atomic_radius("C"), 0.73)
        assert np.isclose(get_atomic_radius("N"), 0.71)
        assert np.isclose(get_atomic_radius("O"), 0.66)
    
    def test_get_atomic_radius_unknown(self):
        """Test error for unknown element."""
        with pytest.raises(KeyError):
            get_atomic_radius("Xx")
    
    def test_element_info(self):
        """Test comprehensive element information."""
        info = element_info("C")
        
        assert info["symbol"] == "C"
        assert info["atomic_number"] == 6
        assert np.isclose(info["atomic_mass"], 12.011)
        assert np.isclose(info["covalent_radius"], 0.73)
    
    def test_element_info_unknown(self):
        """Test error for unknown element."""
        with pytest.raises(KeyError):
            element_info("Xx")


class TestGeometry:
    """Test cases for geometry utilities."""
    
    def test_distance_matrix(self):
        """Test distance matrix calculation."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        distances = distance_matrix(positions)
        
        assert distances.shape == (3, 3)
        assert np.isclose(distances[0, 1], 1.0)
        assert np.isclose(distances[0, 2], 1.0)
        assert np.isclose(distances[1, 2], np.sqrt(2))
        
        # Check symmetry
        assert np.allclose(distances, distances.T)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(distances), 0.0)
    
    def test_find_neighbors(self):
        """Test neighbor finding."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # center
            [0.5, 0.0, 0.0],  # close
            [2.0, 0.0, 0.0],  # far
            [0.0, 0.8, 0.0],  # close
        ])
        
        neighbors = find_neighbors(positions, 0, 1.0)
        
        assert 1 in neighbors  # Close neighbor
        assert 3 in neighbors  # Close neighbor
        assert 2 not in neighbors  # Far neighbor
    
    def test_rotation_matrix(self):
        """Test rotation matrix creation."""
        # Rotation about z-axis by 90 degrees
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2
        
        R = rotation_matrix(axis, angle)
        
        # Check it's a proper rotation matrix
        assert np.allclose(np.linalg.det(R), 1.0)
        assert np.allclose(R @ R.T, np.eye(3))
        
        # Check rotation of x-axis gives y-axis
        x_axis = np.array([1.0, 0.0, 0.0])
        rotated = R @ x_axis
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(rotated, expected, atol=1e-10)
    
    def test_rotation_matrix_arbitrary_axis(self):
        """Test rotation about arbitrary axis."""
        axis = np.array([1.0, 1.0, 1.0])  # Will be normalized
        angle = np.pi / 3
        
        R = rotation_matrix(axis, angle)
        
        # Check it's a proper rotation matrix
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestConstants:
    """Test cases for constants."""
    
    def test_atomic_symbols_present(self):
        """Test that common atomic symbols are present."""
        from mofbuilder.utils.constants import ATOMIC_SYMBOLS
        
        # Check some common elements
        assert 1 in ATOMIC_SYMBOLS
        assert ATOMIC_SYMBOLS[1] == "H"
        assert ATOMIC_SYMBOLS[6] == "C"
        assert ATOMIC_SYMBOLS[7] == "N"
        assert ATOMIC_SYMBOLS[8] == "O"
    
    def test_atomic_masses_present(self):
        """Test that atomic masses are present."""
        from mofbuilder.utils.constants import ATOMIC_MASSES
        
        # Check some common elements
        assert "H" in ATOMIC_MASSES
        assert "C" in ATOMIC_MASSES
        assert "N" in ATOMIC_MASSES
        assert "O" in ATOMIC_MASSES
        
        # Check reasonable values
        assert 0.5 < ATOMIC_MASSES["H"] < 2.0
        assert 10.0 < ATOMIC_MASSES["C"] < 15.0
    
    def test_atomic_radii_present(self):
        """Test that atomic radii are present."""
        from mofbuilder.utils.constants import ATOMIC_RADII
        
        # Check some common elements
        assert "H" in ATOMIC_RADII
        assert "C" in ATOMIC_RADII
        assert "N" in ATOMIC_RADII
        assert "O" in ATOMIC_RADII
        
        # Check reasonable values (in Angstroms)
        assert 0.1 < ATOMIC_RADII["H"] < 1.0
        assert 0.5 < ATOMIC_RADII["C"] < 1.5