"""
Example showing file I/O operations with MOF structures.

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

from mofbuilder.core import Atom, Framework, Lattice
from mofbuilder.io import CifReader, CifWriter, XyzReader, XyzWriter


def create_sample_xyz_file(filepath):
    """Create a sample XYZ file for testing."""
    content = """4
Methane molecule
C    0.000000    0.000000    0.000000
H    1.089000    0.000000    0.000000
H   -0.363000    1.026804    0.000000
H   -0.363000   -0.513402   -0.889165
"""
    with open(filepath, 'w') as f:
        f.write(content)


def create_sample_framework():
    """Create a sample framework programmatically."""
    # Create lattice
    lattice = Lattice(a=15.0, b=15.0, c=15.0, alpha=90.0, beta=90.0, gamma=90.0)
    
    # Create framework
    framework = Framework("SampleMOF", lattice)
    
    # Add atoms to create a simple structure
    atoms_data = [
        ("Zn", (5.0, 5.0, 5.0)),
        ("Zn", (10.0, 10.0, 10.0)),
        ("O", (6.0, 5.0, 5.0)),
        ("O", (9.0, 10.0, 10.0)),
        ("O", (5.0, 6.0, 5.0)),
        ("O", (10.0, 9.0, 10.0)),
        ("C", (7.5, 5.0, 5.0)),
        ("C", (7.5, 10.0, 10.0)),
        ("C", (5.0, 7.5, 5.0)),
        ("C", (10.0, 7.5, 10.0)),
        ("N", (7.5, 7.5, 7.5)),  # Central linker
    ]
    
    for i, (element, position) in enumerate(atoms_data):
        atom = Atom(element, position, atom_id=i)
        framework.add_atom(atom)
    
    return framework


def demonstrate_xyz_io():
    """Demonstrate XYZ file I/O operations."""
    print("=== XYZ File I/O Demo ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a sample XYZ file
        xyz_input = tmpdir / "sample_input.xyz"
        create_sample_xyz_file(xyz_input)
        print(f"Created sample XYZ file: {xyz_input}")
        
        # Read the XYZ file
        reader = XyzReader()
        lattice = Lattice(20.0, 20.0, 20.0)  # Large box for molecule
        framework = reader.read(xyz_input, lattice)
        
        print(f"Read framework: {framework.name}")
        print(f"Number of atoms: {len(framework.atoms)}")
        print(f"Formula: {framework.formula}")
        print("Atoms:")
        for i, atom in enumerate(framework.atoms):
            print(f"  {i+1}: {atom.element} at {atom.position}")
        
        # Write back to XYZ
        xyz_output = tmpdir / "output.xyz"
        writer = XyzWriter()
        writer.write(framework, xyz_output, comment="Processed with MofBuilder")
        
        print(f"Wrote framework to: {xyz_output}")
        
        # Read the output file to verify
        with open(xyz_output, 'r') as f:
            content = f.read()
            print("Output file content:")
            print(content)
    
    print()


def demonstrate_cif_io():
    """Demonstrate CIF file I/O operations."""
    print("=== CIF File I/O Demo ===")
    
    # Create a framework programmatically
    framework = create_sample_framework()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write framework to CIF
        cif_output = tmpdir / "sample_mof.cif"
        writer = CifWriter()
        writer.write(framework, cif_output, title="Sample MOF for demonstration")
        
        print(f"Created CIF file: {cif_output}")
        print(f"Framework: {framework.name}")
        print(f"Formula: {framework.formula}")
        print(f"Unit cell volume: {framework.lattice.volume:.2f} Å³")
        
        # Read the CIF file content
        with open(cif_output, 'r') as f:
            content = f.read()
            print("CIF file content (first 20 lines):")
            lines = content.split('\n')
            for i, line in enumerate(lines[:20]):
                print(f"  {line}")
            if len(lines) > 20:
                print(f"  ... ({len(lines) - 20} more lines)")
        
        # Try to read it back (simplified - real CIF parsing is complex)
        try:
            reader = CifReader()
            loaded_framework = reader.read(cif_output)
            print(f"Successfully read back framework: {loaded_framework.name}")
            print(f"Number of atoms read: {len(loaded_framework.atoms)}")
        except Exception as e:
            print(f"Note: CIF reading is simplified in this example: {e}")
    
    print()


def demonstrate_format_conversion():
    """Demonstrate conversion between file formats."""
    print("=== Format Conversion Demo ===")
    
    # Create a framework
    framework = create_sample_framework()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save as CIF
        cif_file = tmpdir / "original.cif"
        cif_writer = CifWriter()
        cif_writer.write(framework, cif_file)
        print(f"Saved framework as CIF: {cif_file}")
        
        # Save as XYZ
        xyz_file = tmpdir / "converted.xyz"
        xyz_writer = XyzWriter()
        xyz_writer.write(framework, xyz_file)
        print(f"Converted framework to XYZ: {xyz_file}")
        
        # Compare file sizes
        cif_size = cif_file.stat().st_size
        xyz_size = xyz_file.stat().st_size
        
        print(f"CIF file size: {cif_size} bytes")
        print(f"XYZ file size: {xyz_size} bytes")
        
        # Show content comparison
        print("\nXYZ content:")
        with open(xyz_file, 'r') as f:
            xyz_content = f.read()
            print(xyz_content)
    
    print()


def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple structures."""
    print("=== Batch Processing Demo ===")
    
    # Create multiple frameworks
    frameworks = []
    
    for i in range(3):
        lattice = Lattice(a=10.0 + i, b=10.0 + i, c=10.0 + i)
        framework = Framework(f"MOF_{i+1}", lattice)
        
        # Add some atoms
        framework.add_atom(Atom("C", (2.0, 2.0, 2.0)))
        framework.add_atom(Atom("N", (5.0, 5.0, 5.0)))
        framework.add_atom(Atom("O", (8.0, 8.0, 8.0)))
        
        frameworks.append(framework)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Process each framework
        for framework in frameworks:
            # Save as both XYZ and CIF
            xyz_file = tmpdir / f"{framework.name}.xyz"
            cif_file = tmpdir / f"{framework.name}.cif"
            
            xyz_writer = XyzWriter()
            xyz_writer.write(framework, xyz_file)
            
            cif_writer = CifWriter()
            cif_writer.write(framework, cif_file)
            
            print(f"Processed {framework.name}:")
            print(f"  Formula: {framework.formula}")
            print(f"  Volume: {framework.lattice.volume:.1f} Å³")
            print(f"  Files: {xyz_file.name}, {cif_file.name}")
        
        # List all created files
        print(f"\nTotal files created: {len(list(tmpdir.glob('*')))}")
        for file_path in sorted(tmpdir.glob('*')):
            print(f"  {file_path.name} ({file_path.stat().st_size} bytes)")
    
    print()


def main():
    """Main function to run all I/O demonstrations."""
    print("MofBuilder File I/O Examples")
    print("=" * 50)
    print()
    
    demonstrate_xyz_io()
    demonstrate_cif_io()
    demonstrate_format_conversion()
    demonstrate_batch_processing()
    
    print("All I/O examples completed successfully!")


if __name__ == "__main__":
    main()