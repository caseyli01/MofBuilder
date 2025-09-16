"""
Integration tests for MofBuilder package.

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

import tempfile
from pathlib import Path

import pytest

from mofbuilder.core import Atom, Framework, Lattice
from mofbuilder.io import CifWriter, XyzWriter, XyzReader
from mofbuilder.analysis import PoreAnalyzer, SurfaceAreaCalculator, TopologyAnalyzer


class TestIOIntegration:
    """Integration tests for IO operations."""
    
    def test_xyz_round_trip(self):
        """Test XYZ file round-trip."""
        # Create a simple framework
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_mof", lattice)
        
        framework.add_atom(Atom("C", (0.0, 0.0, 0.0)))
        framework.add_atom(Atom("N", (1.5, 0.0, 0.0)))
        framework.add_atom(Atom("O", (0.0, 1.5, 0.0)))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_file = Path(tmpdir) / "test.xyz"
            
            # Write XYZ file
            writer = XyzWriter()
            writer.write(framework, xyz_file)
            
            # Read it back
            reader = XyzReader()
            loaded_framework = reader.read(xyz_file, lattice)
            
            # Check that we got the same data
            assert len(loaded_framework.atoms) == 3
            assert loaded_framework.atoms[0].element == "C"
            assert loaded_framework.atoms[1].element == "N"
            assert loaded_framework.atoms[2].element == "O"
    
    def test_cif_writing(self):
        """Test CIF file writing."""
        # Create a simple framework
        lattice = Lattice(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        framework = Framework("test_mof", lattice)
        
        framework.add_atom(Atom("C", (0.0, 0.0, 0.0)))
        framework.add_atom(Atom("O", (2.5, 2.5, 2.5)))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_file = Path(tmpdir) / "test.cif"
            
            # Write CIF file
            writer = CifWriter()
            writer.write(framework, cif_file)
            
            # Check that file was created and has content
            assert cif_file.exists()
            content = cif_file.read_text()
            
            # Check for expected CIF sections
            assert "data_test_mof" in content
            assert "_cell_length_a" in content
            assert "_atom_site_label" in content
            assert "C1" in content
            assert "O2" in content


class TestAnalysisIntegration:
    """Integration tests for analysis modules."""
    
    def create_simple_framework(self):
        """Create a simple test framework."""
        lattice = Lattice(10.0, 10.0, 10.0)
        framework = Framework("test_framework", lattice)
        
        # Create a simple cubic arrangement
        positions = [
            (2.0, 2.0, 2.0),
            (8.0, 2.0, 2.0),
            (2.0, 8.0, 2.0),
            (8.0, 8.0, 2.0),
            (2.0, 2.0, 8.0),
            (8.0, 2.0, 8.0),
            (2.0, 8.0, 8.0),
            (8.0, 8.0, 8.0),
        ]
        
        for i, pos in enumerate(positions):
            element = "C" if i % 2 == 0 else "N"
            framework.add_atom(Atom(element, pos, atom_id=i))
        
        return framework
    
    def test_pore_analysis_integration(self):
        """Test pore analysis on a framework."""
        framework = self.create_simple_framework()
        
        analyzer = PoreAnalyzer(probe_radius=1.0)
        
        # Test accessible volume calculation
        volume_data = analyzer.calculate_accessible_volume(framework, grid_spacing=1.0)
        
        assert "total_volume" in volume_data
        assert "accessible_volume" in volume_data
        assert "accessible_fraction" in volume_data
        assert abs(volume_data["total_volume"] - 1000.0) < 1e-6  # 10^3
        assert 0.0 <= volume_data["accessible_fraction"] <= 1.0
        
        # Test largest sphere calculation
        sphere_data = analyzer.find_largest_sphere(framework)
        
        assert "radius" in sphere_data
        assert "diameter" in sphere_data
        assert "center" in sphere_data
        assert sphere_data["radius"] >= 0.0
        assert sphere_data["diameter"] == 2.0 * sphere_data["radius"]
    
    def test_surface_area_integration(self):
        """Test surface area calculation on a framework."""
        framework = self.create_simple_framework()
        
        calculator = SurfaceAreaCalculator(probe_radius=1.0)
        
        # Test geometric surface area
        surface_data = calculator.calculate_geometric_surface_area(framework, grid_spacing=1.0)
        
        assert "geometric_surface_area" in surface_data
        assert "specific_surface_area" in surface_data
        assert surface_data["geometric_surface_area"] >= 0.0
        assert surface_data["specific_surface_area"] >= 0.0
        
        # Test accessible surface area
        accessible_data = calculator.calculate_accessible_surface_area(framework)
        
        assert "accessible_surface_area" in accessible_data
        assert accessible_data["accessible_surface_area"] >= 0.0
    
    def test_topology_analysis_integration(self):
        """Test topology analysis on a framework."""
        framework = self.create_simple_framework()
        
        analyzer = TopologyAnalyzer(bond_tolerance=0.5)
        
        # Test coordination analysis
        coord_data = analyzer.analyze_coordination(framework)
        
        assert "per_atom" in coord_data
        assert "statistics" in coord_data
        assert len(coord_data["per_atom"]) == len(framework.atoms)
        
        stats = coord_data["statistics"]
        assert "mean_coordination" in stats
        assert "max_coordination" in stats
        assert "min_coordination" in stats
        
        # Test ring finding
        ring_data = analyzer.find_rings(framework, max_ring_size=8)
        
        assert "rings" in ring_data
        assert "ring_counts" in ring_data
        assert "total_rings" in ring_data
        assert ring_data["total_rings"] >= 0
        
        # Test SBU identification
        sbu_data = analyzer.identify_sbu(framework)
        
        assert "metal_nodes" in sbu_data
        assert "num_metal_nodes" in sbu_data
        assert "total_metals" in sbu_data


class TestWorkflowIntegration:
    """Integration tests for complete workflows."""
    
    def test_analysis_workflow(self):
        """Test complete analysis workflow."""
        # Create framework
        lattice = Lattice(8.0, 8.0, 8.0)
        framework = Framework("workflow_test", lattice)
        
        # Add atoms in a pattern that creates some pore space
        framework.add_atom(Atom("C", (1.0, 1.0, 1.0)))
        framework.add_atom(Atom("C", (7.0, 1.0, 1.0)))
        framework.add_atom(Atom("C", (1.0, 7.0, 1.0)))
        framework.add_atom(Atom("C", (7.0, 7.0, 1.0)))
        framework.add_atom(Atom("N", (4.0, 4.0, 4.0)))
        
        # Run all analyses
        pore_analyzer = PoreAnalyzer()
        surface_calculator = SurfaceAreaCalculator()
        topology_analyzer = TopologyAnalyzer()
        
        # Get results
        pore_results = pore_analyzer.calculate_accessible_volume(framework)
        surface_results = surface_calculator.calculate_geometric_surface_area(framework)
        coord_results = topology_analyzer.analyze_coordination(framework)
        
        # Verify we got meaningful results
        assert pore_results["accessible_fraction"] > 0.0
        assert surface_results["geometric_surface_area"] > 0.0
        assert len(coord_results["per_atom"]) == 5
        
        # Test framework properties
        assert framework.formula in ["C4N", "NC4"]  # Either order is fine
        assert len(framework.composition) == 2
        assert framework.composition["C"] == 4
        assert framework.composition["N"] == 1
    
    def test_file_analysis_workflow(self):
        """Test workflow involving file I/O and analysis."""
        # Create and save a framework
        lattice = Lattice(6.0, 6.0, 6.0)
        framework = Framework("file_test", lattice)
        
        framework.add_atom(Atom("C", (1.0, 1.0, 1.0)))
        framework.add_atom(Atom("O", (5.0, 5.0, 5.0)))
        framework.add_atom(Atom("N", (3.0, 3.0, 3.0)))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write to XYZ
            xyz_file = Path(tmpdir) / "test.xyz"
            xyz_writer = XyzWriter()
            xyz_writer.write(framework, xyz_file)
            
            # Read it back
            xyz_reader = XyzReader()
            loaded_framework = xyz_reader.read(xyz_file, lattice)
            
            # Analyze the loaded framework
            analyzer = PoreAnalyzer()
            results = analyzer.calculate_accessible_volume(loaded_framework)
            
            # Should get reasonable results
            assert abs(results["total_volume"] - 216.0) < 1e-6  # 6^3
            assert 0.0 <= results["accessible_fraction"] <= 1.0