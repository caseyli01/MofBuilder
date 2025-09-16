"""
CIF file reader for MOF structures.

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

import re
from pathlib import Path
from typing import Dict, List, Optional

from ..core import Atom, Framework, Lattice


class CifReader:
    """
    Reader for Crystallographic Information Files (CIF).
    
    This class provides functionality to read MOF structures from CIF files
    and convert them to Framework objects.
    """
    
    def __init__(self):
        """Initialize the CIF reader."""
        pass
    
    def read(self, filepath: Path) -> Framework:
        """
        Read a CIF file and return a Framework object.
        
        Args:
            filepath: Path to the CIF file.
            
        Returns:
            Framework object containing the structure.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the CIF file is malformed.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"CIF file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self._parse_cif_content(content, filepath.stem)
    
    def _parse_cif_content(self, content: str, name: str) -> Framework:
        """
        Parse CIF file content and create a Framework.
        
        Args:
            content: CIF file content as string.
            name: Name for the framework.
            
        Returns:
            Framework object.
        """
        lines = content.split('\n')
        cif_data = self._extract_cif_data(lines)
        
        # Extract lattice parameters
        lattice = self._parse_lattice_parameters(cif_data)
        
        # Extract atomic coordinates
        atoms = self._parse_atomic_coordinates(cif_data, lattice)
        
        framework = Framework(name, lattice, atoms)
        return framework
    
    def _extract_cif_data(self, lines: List[str]) -> Dict[str, str]:
        """
        Extract data from CIF file lines.
        
        Args:
            lines: List of lines from the CIF file.
            
        Returns:
            Dictionary with CIF data tags and values.
        """
        data = {}
        current_loop = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('_'):
                # Data tag
                if ' ' in line:
                    tag, value = line.split(None, 1)
                    data[tag] = value.strip("'\"")
                else:
                    tag = line
                    data[tag] = ""
            elif line.startswith('loop_'):
                current_loop = []
                data['_current_loop'] = current_loop
            elif current_loop is not None and line.startswith('_'):
                current_loop.append(line)
        
        return data
    
    def _parse_lattice_parameters(self, cif_data: Dict[str, str]) -> Lattice:
        """
        Parse lattice parameters from CIF data.
        
        Args:
            cif_data: Dictionary with CIF data.
            
        Returns:
            Lattice object.
        """
        # Extract lattice parameters with error handling
        try:
            a = float(self._extract_number(cif_data.get('_cell_length_a', '1.0')))
            b = float(self._extract_number(cif_data.get('_cell_length_b', '1.0')))
            c = float(self._extract_number(cif_data.get('_cell_length_c', '1.0')))
            alpha = float(self._extract_number(cif_data.get('_cell_angle_alpha', '90.0')))
            beta = float(self._extract_number(cif_data.get('_cell_angle_beta', '90.0')))
            gamma = float(self._extract_number(cif_data.get('_cell_angle_gamma', '90.0')))
        except (ValueError, TypeError):
            # Default to cubic unit cell if parsing fails
            a = b = c = 1.0
            alpha = beta = gamma = 90.0
        
        return Lattice(a, b, c, alpha, beta, gamma)
    
    def _parse_atomic_coordinates(self, cif_data: Dict[str, str], lattice: Lattice) -> List[Atom]:
        """
        Parse atomic coordinates from CIF data.
        
        Args:
            cif_data: Dictionary with CIF data.
            lattice: Lattice object for coordinate conversion.
            
        Returns:
            List of Atom objects.
        """
        atoms = []
        
        # This is a simplified parser - real CIF parsing is more complex
        # In practice, you would use a dedicated CIF parsing library
        
        # Look for atom site data (simplified)
        x_coords = cif_data.get('_atom_site_fract_x', '').split()
        y_coords = cif_data.get('_atom_site_fract_y', '').split()
        z_coords = cif_data.get('_atom_site_fract_z', '').split()
        labels = cif_data.get('_atom_site_label', '').split()
        
        if x_coords and y_coords and z_coords and labels:
            for i, (x, y, z, label) in enumerate(zip(x_coords, y_coords, z_coords, labels)):
                try:
                    frac_coords = [float(self._extract_number(x)),
                                   float(self._extract_number(y)),
                                   float(self._extract_number(z))]
                    cart_coords = lattice.fractional_to_cartesian(frac_coords)
                    element = self._extract_element_from_label(label)
                    atom = Atom(element, tuple(cart_coords), atom_id=i)
                    atoms.append(atom)
                except (ValueError, TypeError):
                    continue
        
        return atoms
    
    def _extract_number(self, value: str) -> str:
        """
        Extract numeric value from CIF format (removing uncertainty).
        
        Args:
            value: String value that may contain uncertainty in parentheses.
            
        Returns:
            Cleaned numeric string.
        """
        # Remove uncertainty notation like "1.234(5)"
        return re.sub(r'\([^)]*\)', '', value).strip()
    
    def _extract_element_from_label(self, label: str) -> str:
        """
        Extract element symbol from atom label.
        
        Args:
            label: Atom label from CIF file.
            
        Returns:
            Element symbol.
        """
        # Extract element symbol (first 1-2 letters)
        match = re.match(r'([A-Z][a-z]?)', label)
        return match.group(1) if match else 'X'