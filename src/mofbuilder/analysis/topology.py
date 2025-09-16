"""
Topology analysis tools for MOF structures.

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

from typing import Dict, List, Set, Tuple

import numpy as np

from ..core import Atom, Framework
from ..utils import get_atomic_radius


class TopologyAnalyzer:
    """
    Analyzer for topological properties of MOF structures.
    
    This class provides methods to analyze coordination numbers,
    connectivity patterns, and topological classifications.
    """
    
    def __init__(self, bond_tolerance: float = 0.3):
        """
        Initialize the topology analyzer.
        
        Args:
            bond_tolerance: Tolerance for bond detection in Angstroms.
        """
        self.bond_tolerance = bond_tolerance
    
    def analyze_coordination(self, framework: Framework) -> Dict[str, any]:
        """
        Analyze coordination numbers and environments.
        
        Args:
            framework: Framework to analyze.
            
        Returns:
            Dictionary with coordination analysis results.
        """
        coordination_data = {}
        
        for i, atom in enumerate(framework.atoms):
            neighbors = self._find_neighbors(atom, framework)
            coord_number = len(neighbors)
            
            # Classify coordination environment
            coord_env = self._classify_coordination_environment(atom, neighbors)
            
            coordination_data[i] = {
                "atom": atom.element,
                "coordination_number": coord_number,
                "environment": coord_env,
                "neighbors": [neighbor.element for neighbor in neighbors],
            }
        
        # Calculate statistics
        coord_numbers = [data["coordination_number"] for data in coordination_data.values()]
        
        return {
            "per_atom": coordination_data,
            "statistics": {
                "mean_coordination": np.mean(coord_numbers) if coord_numbers else 0.0,
                "max_coordination": max(coord_numbers) if coord_numbers else 0,
                "min_coordination": min(coord_numbers) if coord_numbers else 0,
            },
        }
    
    def _find_neighbors(self, atom: Atom, framework: Framework) -> List[Atom]:
        """
        Find neighboring atoms within bonding distance.
        
        Args:
            atom: Central atom.
            framework: Framework containing all atoms.
            
        Returns:
            List of neighboring atoms.
        """
        neighbors = []
        atom_radius = get_atomic_radius(atom.element)
        
        for other_atom in framework.atoms:
            if other_atom is atom:
                continue
            
            distance = atom.distance_to(other_atom)
            other_radius = get_atomic_radius(other_atom.element)
            
            # Check if atoms are bonded (sum of covalent radii + tolerance)
            bond_distance = atom_radius + other_radius + self.bond_tolerance
            
            if distance <= bond_distance:
                neighbors.append(other_atom)
        
        return neighbors
    
    def _classify_coordination_environment(self, atom: Atom, 
                                         neighbors: List[Atom]) -> str:
        """
        Classify the coordination environment based on geometry.
        
        Args:
            atom: Central atom.
            neighbors: List of neighboring atoms.
            
        Returns:
            String describing the coordination environment.
        """
        coord_num = len(neighbors)
        
        if coord_num == 0:
            return "isolated"
        elif coord_num == 1:
            return "terminal"
        elif coord_num == 2:
            return "linear"
        elif coord_num == 3:
            return "trigonal"
        elif coord_num == 4:
            # Could be tetrahedral or square planar
            return "tetrahedral"  # Simplified classification
        elif coord_num == 5:
            return "trigonal_bipyramidal"
        elif coord_num == 6:
            return "octahedral"
        elif coord_num == 7:
            return "pentagonal_bipyramidal"
        elif coord_num == 8:
            return "cubic"
        else:
            return f"coordination_{coord_num}"
    
    def find_rings(self, framework: Framework, max_ring_size: int = 12) -> Dict[str, any]:
        """
        Find ring structures in the framework.
        
        Args:
            framework: Framework to analyze.
            max_ring_size: Maximum ring size to search for.
            
        Returns:
            Dictionary with ring analysis results.
        """
        # Build connectivity graph
        graph = self._build_connectivity_graph(framework)
        
        # Find rings using depth-first search
        rings = []
        visited_edges = set()
        
        for start_atom in range(len(framework.atoms)):
            rings.extend(
                self._find_rings_from_atom(graph, start_atom, max_ring_size, visited_edges)
            )
        
        # Classify rings by size
        ring_sizes = [len(ring) for ring in rings]
        ring_counts = {}
        for size in ring_sizes:
            ring_counts[size] = ring_counts.get(size, 0) + 1
        
        return {
            "rings": rings,
            "ring_counts": ring_counts,
            "total_rings": len(rings),
        }
    
    def _build_connectivity_graph(self, framework: Framework) -> Dict[int, List[int]]:
        """
        Build a connectivity graph from the framework.
        
        Args:
            framework: Framework to analyze.
            
        Returns:
            Dictionary representing the connectivity graph.
        """
        graph = {i: [] for i in range(len(framework.atoms))}
        
        for i, atom in enumerate(framework.atoms):
            neighbors = self._find_neighbors(atom, framework)
            
            for neighbor in neighbors:
                # Find neighbor index
                for j, other_atom in enumerate(framework.atoms):
                    if other_atom is neighbor:
                        graph[i].append(j)
                        break
        
        return graph
    
    def _find_rings_from_atom(self, graph: Dict[int, List[int]], start_atom: int,
                            max_ring_size: int, visited_edges: Set[Tuple[int, int]]) -> List[List[int]]:
        """
        Find rings starting from a specific atom using DFS.
        
        Args:
            graph: Connectivity graph.
            start_atom: Starting atom index.
            max_ring_size: Maximum ring size to search for.
            visited_edges: Set of already visited edges.
            
        Returns:
            List of rings (each ring is a list of atom indices).
        """
        rings = []
        
        def dfs(current_path: List[int], current_atom: int, target_atom: int, depth: int):
            if depth > max_ring_size:
                return
            
            if depth > 2 and current_atom == target_atom:
                # Found a ring
                ring = current_path[:-1]  # Remove the duplicate end atom
                if len(ring) >= 3:
                    # Check if this ring is already found (avoid duplicates)
                    canonical_ring = self._canonicalize_ring(ring)
                    if canonical_ring not in [self._canonicalize_ring(r) for r in rings]:
                        rings.append(ring)
                return
            
            for neighbor in graph.get(current_atom, []):
                edge = tuple(sorted([current_atom, neighbor]))
                
                if depth > 0 and neighbor in current_path[:-1]:
                    continue  # Avoid revisiting atoms (except target)
                
                if edge not in visited_edges or (depth > 2 and neighbor == target_atom):
                    dfs(current_path + [neighbor], neighbor, target_atom, depth + 1)
        
        dfs([start_atom], start_atom, start_atom, 0)
        
        # Mark edges as visited
        for ring in rings:
            for i in range(len(ring)):
                j = (i + 1) % len(ring)
                edge = tuple(sorted([ring[i], ring[j]]))
                visited_edges.add(edge)
        
        return rings
    
    def _canonicalize_ring(self, ring: List[int]) -> Tuple[int, ...]:
        """
        Create a canonical representation of a ring.
        
        Args:
            ring: List of atom indices forming a ring.
            
        Returns:
            Canonical tuple representation.
        """
        # Find the minimum starting point and direction
        min_idx = min(ring)
        start_pos = ring.index(min_idx)
        
        # Try both directions
        forward = ring[start_pos:] + ring[:start_pos]
        backward = ring[start_pos::-1] + ring[:start_pos:-1]
        
        return tuple(min(forward, backward))
    
    def identify_sbu(self, framework: Framework) -> Dict[str, any]:
        """
        Identify Secondary Building Units (SBUs) in the framework.
        
        Args:
            framework: Framework to analyze.
            
        Returns:
            Dictionary with SBU identification results.
        """
        # This is a simplified SBU identification
        # Real implementation would require more sophisticated algorithms
        
        metal_atoms = [i for i, atom in enumerate(framework.atoms) 
                      if self._is_metal(atom.element)]
        
        # Find metal clusters/nodes
        metal_clusters = []
        visited_metals = set()
        
        for metal_idx in metal_atoms:
            if metal_idx in visited_metals:
                continue
            
            cluster = self._find_metal_cluster(framework, metal_idx, visited_metals)
            if cluster:
                metal_clusters.append(cluster)
        
        return {
            "metal_nodes": metal_clusters,
            "num_metal_nodes": len(metal_clusters),
            "total_metals": len(metal_atoms),
        }
    
    def _is_metal(self, element: str) -> bool:
        """Check if an element is a metal."""
        metals = {
            "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
            "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
            "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
            "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
            "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        }
        return element in metals
    
    def _find_metal_cluster(self, framework: Framework, start_metal: int, 
                           visited: Set[int]) -> List[int]:
        """
        Find a cluster of connected metal atoms.
        
        Args:
            framework: Framework to analyze.
            start_metal: Starting metal atom index.
            visited: Set of already visited metal atoms.
            
        Returns:
            List of metal atom indices in the cluster.
        """
        cluster = []
        to_visit = [start_metal]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            cluster.append(current)
            
            # Find connected metal atoms
            current_atom = framework.atoms[current]
            neighbors = self._find_neighbors(current_atom, framework)
            
            for neighbor in neighbors:
                # Find neighbor index
                for i, atom in enumerate(framework.atoms):
                    if atom is neighbor and self._is_metal(atom.element) and i not in visited:
                        to_visit.append(i)
                        break
        
        return cluster