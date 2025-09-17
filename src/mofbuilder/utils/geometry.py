"""
Geometry utility functions for MOF structures.

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

from typing import List, Tuple

import numpy as np


def distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Calculate distance matrix between all pairs of positions.
    
    Args:
        positions: Array of shape (n, 3) containing 3D coordinates.
        
    Returns:
        Array of shape (n, n) containing pairwise distances.
    """
    n = positions.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def find_neighbors(
    positions: np.ndarray, 
    center_idx: int, 
    max_distance: float
) -> List[int]:
    """
    Find neighbors within a specified distance from a center point.
    
    Args:
        positions: Array of shape (n, 3) containing 3D coordinates.
        center_idx: Index of the center point.
        max_distance: Maximum distance for neighbors.
        
    Returns:
        List of indices of neighboring points.
    """
    center = positions[center_idx]
    neighbors = []
    
    for i, pos in enumerate(positions):
        if i == center_idx:
            continue
        
        distance = np.linalg.norm(pos - center)
        if distance <= max_distance:
            neighbors.append(i)
    
    return neighbors


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix for rotation about an arbitrary axis.
    
    Args:
        axis: Rotation axis (will be normalized).
        angle: Rotation angle in radians.
        
    Returns:
        3x3 rotation matrix.
    """
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Cross product matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Rotation matrix
    R = (np.eye(3) + 
         sin_angle * K + 
         (1 - cos_angle) * np.dot(K, K))
    
    return R


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors in radians.
    
    Args:
        v1: First vector.
        v2: Second vector.
        
    Returns:
        Angle between vectors in radians.
    """
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)


def dihedral_angle(p1: np.ndarray, p2: np.ndarray, 
                  p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate dihedral angle between four points.
    
    Args:
        p1, p2, p3, p4: Four points defining the dihedral angle.
        
    Returns:
        Dihedral angle in radians.
    """
    # Vectors along the bonds
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    # Normal vectors to the planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    # Dihedral angle
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Check sign
    if np.dot(np.cross(n1, n2), v2) < 0:
        angle = -angle
    
    return angle


def center_of_mass(positions: np.ndarray, masses: np.ndarray = None) -> np.ndarray:
    """
    Calculate center of mass of a set of points.
    
    Args:
        positions: Array of shape (n, 3) containing 3D coordinates.
        masses: Optional array of masses. If None, assumes equal masses.
        
    Returns:
        Center of mass coordinates.
    """
    if masses is None:
        return np.mean(positions, axis=0)
    else:
        return np.average(positions, axis=0, weights=masses)


def moment_of_inertia_tensor(positions: np.ndarray, 
                           masses: np.ndarray = None) -> np.ndarray:
    """
    Calculate moment of inertia tensor.
    
    Args:
        positions: Array of shape (n, 3) containing 3D coordinates.
        masses: Optional array of masses. If None, assumes equal masses.
        
    Returns:
        3x3 moment of inertia tensor.
    """
    if masses is None:
        masses = np.ones(len(positions))
    
    # Center at origin
    com = center_of_mass(positions, masses)
    centered_positions = positions - com
    
    # Calculate tensor elements
    I = np.zeros((3, 3))
    
    for i, (pos, mass) in enumerate(zip(centered_positions, masses)):
        x, y, z = pos
        I[0, 0] += mass * (y**2 + z**2)
        I[1, 1] += mass * (x**2 + z**2)
        I[2, 2] += mass * (x**2 + y**2)
        I[0, 1] -= mass * x * y
        I[0, 2] -= mass * x * z
        I[1, 2] -= mass * y * z
    
    # Make symmetric
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    
    return I