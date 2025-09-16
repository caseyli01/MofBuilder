"""
Structure viewer for MOF frameworks.

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

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ..core import Framework
from ..utils import get_atomic_radius


class StructureViewer:
    """
    Viewer for MOF structure visualization using matplotlib.
    
    This class provides methods to visualize MOF frameworks in 2D and 3D,
    including atoms, bonds, and unit cell boundaries.
    """
    
    def __init__(self):
        """Initialize the structure viewer."""
        self.element_colors = self._get_default_colors()
    
    def _get_default_colors(self) -> Dict[str, str]:
        """
        Get default colors for chemical elements.
        
        Returns:
            Dictionary mapping element symbols to colors.
        """
        return {
            'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red', 'F': 'green',
            'P': 'orange', 'S': 'yellow', 'Cl': 'lime', 'Br': 'brown', 'I': 'purple',
            'Li': 'violet', 'Na': 'violet', 'K': 'violet', 'Mg': 'darkgreen',
            'Ca': 'darkgreen', 'Al': 'gray', 'Si': 'goldenrod', 'Fe': 'darkorange',
            'Co': 'deeppink', 'Ni': 'lightgreen', 'Cu': 'tan', 'Zn': 'steelblue',
            'Ag': 'silver', 'Au': 'gold', 'Pt': 'lightgray', 'Pd': 'darkkhaki',
        }
    
    def plot_structure_3d(self, framework: Framework, 
                         show_bonds: bool = True,
                         show_unit_cell: bool = True,
                         atom_scale: float = 0.5,
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a 3D plot of the MOF structure.
        
        Args:
            framework: Framework to visualize.
            show_bonds: Whether to show bonds between atoms.
            show_unit_cell: Whether to show unit cell boundaries.
            atom_scale: Scaling factor for atom sizes.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms
        for atom in framework.atoms:
            color = self.element_colors.get(atom.element, 'gray')
            radius = get_atomic_radius(atom.element) * atom_scale * 50  # Scale for visibility
            
            ax.scatter(atom.position[0], atom.position[1], atom.position[2],
                      c=color, s=radius, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Plot bonds
        if show_bonds and framework.bonds:
            for bond in framework.bonds:
                pos1 = bond.atom1.position
                pos2 = bond.atom2.position
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                       'k-', alpha=0.6, linewidth=1)
        
        # Plot unit cell
        if show_unit_cell:
            self._plot_unit_cell(ax, framework.lattice)
        
        # Set labels and title
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'MOF Structure: {framework.name}')
        
        # Set equal aspect ratio
        max_range = max(framework.lattice.a, framework.lattice.b, framework.lattice.c)
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)
        
        return fig
    
    def plot_structure_2d(self, framework: Framework,
                         projection: str = 'xy',
                         show_bonds: bool = True,
                         atom_scale: float = 0.5,
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create a 2D projection plot of the MOF structure.
        
        Args:
            framework: Framework to visualize.
            projection: Projection plane ('xy', 'xz', or 'yz').
            show_bonds: Whether to show bonds between atoms.
            atom_scale: Scaling factor for atom sizes.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine coordinates based on projection
        if projection == 'xy':
            x_coords = [atom.position[0] for atom in framework.atoms]
            y_coords = [atom.position[1] for atom in framework.atoms]
            xlabel, ylabel = 'X (Å)', 'Y (Å)'
            cell_x, cell_y = framework.lattice.a, framework.lattice.b
        elif projection == 'xz':
            x_coords = [atom.position[0] for atom in framework.atoms]
            y_coords = [atom.position[2] for atom in framework.atoms]
            xlabel, ylabel = 'X (Å)', 'Z (Å)'
            cell_x, cell_y = framework.lattice.a, framework.lattice.c
        elif projection == 'yz':
            x_coords = [atom.position[1] for atom in framework.atoms]
            y_coords = [atom.position[2] for atom in framework.atoms]
            xlabel, ylabel = 'Y (Å)', 'Z (Å)'
            cell_x, cell_y = framework.lattice.b, framework.lattice.c
        else:
            raise ValueError("Projection must be 'xy', 'xz', or 'yz'")
        
        # Plot atoms
        for i, atom in enumerate(framework.atoms):
            color = self.element_colors.get(atom.element, 'gray')
            radius = get_atomic_radius(atom.element) * atom_scale * 100
            
            ax.scatter(x_coords[i], y_coords[i], c=color, s=radius, 
                      alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Plot bonds
        if show_bonds and framework.bonds:
            for bond in framework.bonds:
                if projection == 'xy':
                    x1, y1 = bond.atom1.position[0], bond.atom1.position[1]
                    x2, y2 = bond.atom2.position[0], bond.atom2.position[1]
                elif projection == 'xz':
                    x1, y1 = bond.atom1.position[0], bond.atom1.position[2]
                    x2, y2 = bond.atom2.position[0], bond.atom2.position[2]
                elif projection == 'yz':
                    x1, y1 = bond.atom1.position[1], bond.atom1.position[2]
                    x2, y2 = bond.atom2.position[1], bond.atom2.position[2]
                
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=1)
        
        # Plot unit cell boundary
        ax.plot([0, cell_x, cell_x, 0, 0], [0, 0, cell_y, cell_y, 0], 
               'r--', linewidth=2, alpha=0.7, label='Unit cell')
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'MOF Structure: {framework.name} ({projection.upper()} projection)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def _plot_unit_cell(self, ax, lattice) -> None:
        """
        Plot unit cell boundaries in 3D.
        
        Args:
            ax: 3D axes object.
            lattice: Lattice object.
        """
        # Define unit cell vertices
        origin = np.array([0, 0, 0])
        a_vec = lattice.matrix[:, 0]
        b_vec = lattice.matrix[:, 1]
        c_vec = lattice.matrix[:, 2]
        
        vertices = [
            origin,
            a_vec,
            b_vec,
            c_vec,
            a_vec + b_vec,
            a_vec + c_vec,
            b_vec + c_vec,
            a_vec + b_vec + c_vec
        ]
        
        # Define edges of the unit cell
        edges = [
            [0, 1], [0, 2], [0, 3],  # from origin
            [1, 4], [1, 5],          # from a
            [2, 4], [2, 6],          # from b
            [3, 5], [3, 6],          # from c
            [4, 7], [5, 7], [6, 7]  # to opposite corner
        ]
        
        # Plot edges
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                   'r--', alpha=0.7, linewidth=1)
    
    def create_element_legend(self, framework: Framework) -> plt.Figure:
        """
        Create a legend showing element colors.
        
        Args:
            framework: Framework to create legend for.
            
        Returns:
            Matplotlib figure with element legend.
        """
        elements = list(set(atom.element for atom in framework.atoms))
        elements.sort()
        
        fig, ax = plt.subplots(figsize=(6, max(2, len(elements) * 0.5)))
        
        for i, element in enumerate(elements):
            color = self.element_colors.get(element, 'gray')
            ax.scatter([0], [i], c=color, s=200, edgecolors='black', linewidth=1)
            ax.text(0.1, i, element, fontsize=12, va='center')
        
        ax.set_xlim(-0.1, 0.5)
        ax.set_ylim(-0.5, len(elements) - 0.5)
        ax.set_title('Element Legend')
        ax.axis('off')
        
        return fig
    
    def save_structure_image(self, framework: Framework, filename: str,
                           view_type: str = '3d', **kwargs) -> None:
        """
        Save structure visualization to file.
        
        Args:
            framework: Framework to visualize.
            filename: Output filename.
            view_type: Type of view ('3d' or '2d').
            **kwargs: Additional arguments for plotting functions.
        """
        if view_type == '3d':
            fig = self.plot_structure_3d(framework, **kwargs)
        elif view_type == '2d':
            fig = self.plot_structure_2d(framework, **kwargs)
        else:
            raise ValueError("view_type must be '3d' or '2d'")
        
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)