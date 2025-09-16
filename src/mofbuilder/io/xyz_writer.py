"""
XYZ file writer for MOF structures.

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
from typing import Optional

from ..core import Framework


class XyzWriter:
    """
    Writer for XYZ coordinate files.
    
    This class provides functionality to write Framework objects to XYZ files.
    """
    
    def __init__(self):
        """Initialize the XYZ writer."""
        pass
    
    def write(self, framework: Framework, filepath: Path, 
              comment: Optional[str] = None) -> None:
        """
        Write a Framework object to an XYZ file.
        
        Args:
            framework: Framework object to write.
            filepath: Path where to write the XYZ file.
            comment: Optional comment for the second line.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            self._write_xyz_content(f, framework, comment)
    
    def _write_xyz_content(self, file, framework: Framework, 
                          comment: Optional[str] = None) -> None:
        """
        Write XYZ content to file.
        
        Args:
            file: File object to write to.
            framework: Framework object.
            comment: Optional comment.
        """
        # Write number of atoms
        file.write(f"{len(framework.atoms)}\n")
        
        # Write comment line
        if comment:
            file.write(f"{comment}\n")
        elif 'comment' in framework.properties:
            file.write(f"{framework.properties['comment']}\n")
        else:
            file.write(f"Structure: {framework.name}, Formula: {framework.formula}\n")
        
        # Write atomic coordinates
        for atom in framework.atoms:
            file.write(f"{atom.element:<3} {atom.position[0]:12.6f} "
                      f"{atom.position[1]:12.6f} {atom.position[2]:12.6f}\n")