"""
Command line interface for MofBuilder.

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

import argparse
import sys
from pathlib import Path

from .core import Framework
from .io import CifReader, CifWriter, XyzReader, XyzWriter
from .analysis import PoreAnalyzer, SurfaceAreaCalculator, TopologyAnalyzer
from .visualization import StructureViewer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MofBuilder: MOF structure analysis and visualization tool"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between file formats')
    convert_parser.add_argument('input', help='Input file path')
    convert_parser.add_argument('output', help='Output file path')
    convert_parser.add_argument('--format', choices=['cif', 'xyz'], 
                               help='Output format (auto-detected if not specified)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze MOF structure')
    analyze_parser.add_argument('input', help='Input file path (CIF or XYZ)')
    analyze_parser.add_argument('--pore', action='store_true', help='Perform pore analysis')
    analyze_parser.add_argument('--surface', action='store_true', help='Calculate surface area')
    analyze_parser.add_argument('--topology', action='store_true', help='Analyze topology')
    analyze_parser.add_argument('--probe-radius', type=float, default=1.4,
                               help='Probe radius for analysis (default: 1.4 Å)')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize MOF structure')
    viz_parser.add_argument('input', help='Input file path (CIF or XYZ)')
    viz_parser.add_argument('--output', help='Output image file path')
    viz_parser.add_argument('--type', choices=['2d', '3d'], default='3d',
                           help='Visualization type (default: 3d)')
    viz_parser.add_argument('--projection', choices=['xy', 'xz', 'yz'], default='xy',
                           help='2D projection plane (default: xy)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'convert':
            convert_files(args)
        elif args.command == 'analyze':
            analyze_structure(args)
        elif args.command == 'visualize':
            visualize_structure(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def convert_files(args):
    """Convert between file formats."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input file
    if input_path.suffix.lower() == '.cif':
        reader = CifReader()
        framework = reader.read(input_path)
    elif input_path.suffix.lower() == '.xyz':
        reader = XyzReader()
        framework = reader.read(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
    # Write output file
    output_format = args.format or output_path.suffix.lower().lstrip('.')
    
    if output_format == 'cif':
        writer = CifWriter()
        writer.write(framework, output_path)
    elif output_format == 'xyz':
        writer = XyzWriter()
        writer.write(framework, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    print(f"Converted {input_path} to {output_path}")


def analyze_structure(args):
    """Analyze MOF structure."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read structure
    if input_path.suffix.lower() == '.cif':
        reader = CifReader()
        framework = reader.read(input_path)
    elif input_path.suffix.lower() == '.xyz':
        reader = XyzReader()
        framework = reader.read(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    print(f"Analyzing structure: {framework.name}")
    print(f"Formula: {framework.formula}")
    print(f"Number of atoms: {len(framework.atoms)}")
    print(f"Unit cell volume: {framework.lattice.volume:.2f} Å³")
    
    # Pore analysis
    if args.pore:
        print("\n--- Pore Analysis ---")
        analyzer = PoreAnalyzer(probe_radius=args.probe_radius)
        
        volume_data = analyzer.calculate_accessible_volume(framework)
        print(f"Accessible volume: {volume_data['accessible_volume']:.2f} Å³")
        print(f"Accessible fraction: {volume_data['accessible_fraction']:.3f}")
        
        sphere_data = analyzer.find_largest_sphere(framework)
        print(f"Largest sphere diameter: {sphere_data['diameter']:.2f} Å")
    
    # Surface area analysis
    if args.surface:
        print("\n--- Surface Area Analysis ---")
        calculator = SurfaceAreaCalculator(probe_radius=args.probe_radius)
        
        surface_data = calculator.calculate_geometric_surface_area(framework)
        print(f"Geometric surface area: {surface_data['geometric_surface_area']:.2f} Å²")
        print(f"Specific surface area: {surface_data['specific_surface_area']:.2f} Å²/Å³")
    
    # Topology analysis
    if args.topology:
        print("\n--- Topology Analysis ---")
        analyzer = TopologyAnalyzer()
        
        coord_data = analyzer.analyze_coordination(framework)
        stats = coord_data['statistics']
        print(f"Mean coordination number: {stats['mean_coordination']:.2f}")
        print(f"Coordination range: {stats['min_coordination']}-{stats['max_coordination']}")
        
        ring_data = analyzer.find_rings(framework)
        print(f"Total rings found: {ring_data['total_rings']}")
        if ring_data['ring_counts']:
            print("Ring sizes:", dict(ring_data['ring_counts']))


def visualize_structure(args):
    """Visualize MOF structure."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read structure
    if input_path.suffix.lower() == '.cif':
        reader = CifReader()
        framework = reader.read(input_path)
    elif input_path.suffix.lower() == '.xyz':
        reader = XyzReader()
        framework = reader.read(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Create visualization
    viewer = StructureViewer()
    
    if args.type == '3d':
        fig = viewer.plot_structure_3d(framework)
    else:
        fig = viewer.plot_structure_2d(framework, projection=args.projection)
    
    if args.output:
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()