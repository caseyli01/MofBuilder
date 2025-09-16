"""
Basic example of creating and analyzing a simple MOF structure.

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
from pathlib import Path

from mofbuilder.core import Atom, Bond, Framework, Lattice
from mofbuilder.analysis import PoreAnalyzer, SurfaceAreaCalculator, TopologyAnalyzer
from mofbuilder.io import XyzWriter, CifWriter
from mofbuilder.visualization import StructureViewer


def create_simple_cubic_framework():
    """Create a simple cubic MOF framework for demonstration."""
    # Create a cubic lattice
    lattice = Lattice(a=10.0, b=10.0, c=10.0)
    
    # Create framework
    framework = Framework("SimpleCubic", lattice)
    
    # Add atoms in a cubic pattern
    positions = [
        (2.0, 2.0, 2.0), (8.0, 2.0, 2.0), (2.0, 8.0, 2.0), (8.0, 8.0, 2.0),
        (2.0, 2.0, 8.0), (8.0, 2.0, 8.0), (2.0, 8.0, 8.0), (8.0, 8.0, 8.0),
        (5.0, 5.0, 2.0), (5.0, 5.0, 8.0),  # Linker atoms
        (5.0, 2.0, 5.0), (5.0, 8.0, 5.0),  # More linkers
        (2.0, 5.0, 5.0), (8.0, 5.0, 5.0),  # Final linkers
    ]
    
    elements = ["Zn"] * 8 + ["C"] * 6  # Metal nodes and organic linkers
    
    for i, (pos, element) in enumerate(zip(positions, elements)):
        atom = Atom(element, pos, atom_id=i)
        framework.add_atom(atom)
    
    # Add some bonds between nearby atoms
    for i, atom1 in enumerate(framework.atoms):
        for j, atom2 in enumerate(framework.atoms[i+1:], i+1):
            distance = atom1.distance_to(atom2)
            if 2.0 <= distance <= 4.0:  # Reasonable bonding distance
                bond = Bond(atom1, atom2, bond_id=len(framework.bonds))
                framework.add_bond(bond)
    
    return framework


def analyze_framework(framework):
    """Perform comprehensive analysis of the framework."""
    print(f"Analyzing framework: {framework.name}")
    print(f"Chemical formula: {framework.formula}")
    print(f"Number of atoms: {len(framework.atoms)}")
    print(f"Number of bonds: {len(framework.bonds)}")
    print(f"Unit cell volume: {framework.lattice.volume:.2f} Å³")
    print()
    
    # Composition analysis
    composition = framework.composition
    print("Elemental composition:")
    for element, count in composition.items():
        print(f"  {element}: {count}")
    print()
    
    # Pore analysis
    print("--- Pore Analysis ---")
    pore_analyzer = PoreAnalyzer(probe_radius=1.4)  # N2 probe
    
    volume_data = pore_analyzer.calculate_accessible_volume(framework, grid_spacing=0.5)
    print(f"Total volume: {volume_data['total_volume']:.2f} Å³")
    print(f"Accessible volume: {volume_data['accessible_volume']:.2f} Å³")
    print(f"Accessible fraction: {volume_data['accessible_fraction']:.3f}")
    
    sphere_data = pore_analyzer.find_largest_sphere(framework)
    print(f"Largest sphere diameter: {sphere_data['diameter']:.2f} Å")
    
    psd_data = pore_analyzer.pore_size_distribution(framework, grid_spacing=0.5)
    if psd_data['pore_sizes']:
        print(f"Mean pore size: {psd_data['mean_pore_size']:.2f} Å")
        print(f"Max pore size: {psd_data['max_pore_size']:.2f} Å")
    print()
    
    # Surface area analysis
    print("--- Surface Area Analysis ---")
    surface_calculator = SurfaceAreaCalculator(probe_radius=1.4)
    
    surface_data = surface_calculator.calculate_geometric_surface_area(framework, grid_spacing=0.5)
    print(f"Geometric surface area: {surface_data['geometric_surface_area']:.2f} Å²")
    print(f"Specific surface area: {surface_data['specific_surface_area']:.2f} Å²/Å³")
    
    accessible_data = surface_calculator.calculate_accessible_surface_area(framework)
    print(f"Accessible surface area: {accessible_data['accessible_surface_area']:.2f} Å²")
    print()
    
    # Topology analysis
    print("--- Topology Analysis ---")
    topology_analyzer = TopologyAnalyzer(bond_tolerance=0.3)
    
    coord_data = topology_analyzer.analyze_coordination(framework)
    stats = coord_data['statistics']
    print(f"Mean coordination number: {stats['mean_coordination']:.2f}")
    print(f"Coordination range: {stats['min_coordination']}-{stats['max_coordination']}")
    
    # Print coordination for each element
    element_coords = {}
    for atom_data in coord_data['per_atom'].values():
        element = atom_data['atom']
        coord = atom_data['coordination_number']
        if element not in element_coords:
            element_coords[element] = []
        element_coords[element].append(coord)
    
    for element, coords in element_coords.items():
        avg_coord = np.mean(coords)
        print(f"Average coordination for {element}: {avg_coord:.1f}")
    
    ring_data = topology_analyzer.find_rings(framework, max_ring_size=12)
    print(f"Total rings found: {ring_data['total_rings']}")
    if ring_data['ring_counts']:
        print("Ring size distribution:")
        for size, count in sorted(ring_data['ring_counts'].items()):
            print(f"  {size}-rings: {count}")
    
    sbu_data = topology_analyzer.identify_sbu(framework)
    print(f"Metal nodes identified: {sbu_data['num_metal_nodes']}")
    print(f"Total metal atoms: {sbu_data['total_metals']}")
    print()


def save_files(framework, output_dir="output"):
    """Save framework to various file formats."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as XYZ
    xyz_writer = XyzWriter()
    xyz_file = output_path / f"{framework.name}.xyz"
    xyz_writer.write(framework, xyz_file)
    print(f"Saved XYZ file: {xyz_file}")
    
    # Save as CIF
    cif_writer = CifWriter()
    cif_file = output_path / f"{framework.name}.cif"
    cif_writer.write(framework, cif_file, title="Example MOF structure")
    print(f"Saved CIF file: {cif_file}")


def visualize_framework(framework, output_dir="output"):
    """Create visualizations of the framework."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    viewer = StructureViewer()
    
    # Create 3D visualization
    try:
        fig_3d = viewer.plot_structure_3d(framework, show_bonds=True, show_unit_cell=True)
        image_3d = output_path / f"{framework.name}_3d.png"
        fig_3d.savefig(image_3d, dpi=300, bbox_inches='tight')
        print(f"Saved 3D visualization: {image_3d}")
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")
    
    # Create 2D projection
    try:
        fig_2d = viewer.plot_structure_2d(framework, projection='xy', show_bonds=True)
        image_2d = output_path / f"{framework.name}_2d.png"
        fig_2d.savefig(image_2d, dpi=300, bbox_inches='tight')
        print(f"Saved 2D visualization: {image_2d}")
    except Exception as e:
        print(f"Could not create 2D visualization: {e}")
    
    # Create element legend
    try:
        fig_legend = viewer.create_element_legend(framework)
        legend_file = output_path / f"{framework.name}_legend.png"
        fig_legend.savefig(legend_file, dpi=300, bbox_inches='tight')
        print(f"Saved element legend: {legend_file}")
    except Exception as e:
        print(f"Could not create element legend: {e}")


def main():
    """Main example function."""
    print("MofBuilder Basic Example")
    print("=" * 50)
    print()
    
    # Create a simple framework
    framework = create_simple_cubic_framework()
    
    # Analyze the framework
    analyze_framework(framework)
    
    # Save files
    print("--- Saving Files ---")
    save_files(framework)
    print()
    
    # Create visualizations
    print("--- Creating Visualizations ---")
    visualize_framework(framework)
    print()
    
    print("Example completed successfully!")
    print("Check the 'output' directory for generated files.")


if __name__ == "__main__":
    main()