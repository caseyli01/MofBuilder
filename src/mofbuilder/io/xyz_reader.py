"""
XYZ file reader for MOF structures.

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

from pathlib import Path
from typing import List

from ..core import Atom, Framework, Lattice


class XyzReader:
    """
    Reader for XYZ coordinate files.
    
    This class provides functionality to read atomic coordinates from XYZ files
    and convert them to Framework objects.
    """
    
    def __init__(self):
        """Initialize the XYZ reader."""
        pass
    
    def read(self, filepath: Path, lattice: Lattice = None) -> Framework:
        """
        Read an XYZ file and return a Framework object.
        
        Args:
            filepath: Path to the XYZ file.
            lattice: Optional lattice for the framework (default: cubic).
            
        Returns:
            Framework object containing the structure.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the XYZ file is malformed.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"XYZ file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        return self._parse_xyz_content(lines, filepath.stem, lattice)
    
    def _parse_xyz_content(self, lines: List[str], name: str, 
                          lattice: Lattice = None) -> Framework:
        """
        Parse XYZ file content and create a Framework.
        
        Args:
            lines: List of lines from the XYZ file.
            name: Name for the framework.
            lattice: Optional lattice for the framework.
            
        Returns:
            Framework object.
        """
        if len(lines) < 2:
            raise ValueError("XYZ file must have at least 2 lines")
        
        try:
            num_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("First line must contain the number of atoms")
        
        comment = lines[1].strip() if len(lines) > 1 else ""
        
        if len(lines) < num_atoms + 2:
            raise ValueError(f"Expected {num_atoms + 2} lines, got {len(lines)}")
        
        atoms = []
        for i in range(2, num_atoms + 2):
            if i >= len(lines):
                break
            
            line = lines[i].strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            try:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                
                atom = Atom(element, (x, y, z), atom_id=i-2)
                atoms.append(atom)
            except (ValueError, IndexError):
                continue
        
        # Use provided lattice or create a default cubic one
        if lattice is None:
            # Create a cubic lattice large enough to contain all atoms
            if atoms:
                coords = [atom.position for atom in atoms]
                max_coord = max(max(coord) for coord in coords)
                min_coord = min(min(coord) for coord in coords)
                size = max(20.0, abs(max_coord - min_coord) + 10.0)
            else:
                size = 20.0
            
            lattice = Lattice(size, size, size)
        
        framework = Framework(name, lattice, atoms)
        framework.properties['comment'] = comment
        
        return framework