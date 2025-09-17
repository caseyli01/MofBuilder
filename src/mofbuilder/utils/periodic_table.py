"""
Periodic table utilities for element properties.

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

from typing import Dict, Optional

from .constants import ATOMIC_MASSES, ATOMIC_RADII, ATOMIC_SYMBOLS


def get_atomic_mass(element: str) -> float:
    """
    Get atomic mass for an element.
    
    Args:
        element: Element symbol (e.g., 'C', 'N', 'O').
        
    Returns:
        Atomic mass in amu.
        
    Raises:
        KeyError: If element is not found.
    """
    if element not in ATOMIC_MASSES:
        raise KeyError(f"Unknown element: {element}")
    return ATOMIC_MASSES[element]


def get_atomic_radius(element: str) -> float:
    """
    Get covalent radius for an element.
    
    Args:
        element: Element symbol (e.g., 'C', 'N', 'O').
        
    Returns:
        Covalent radius in Angstroms.
        
    Raises:
        KeyError: If element is not found.
    """
    if element not in ATOMIC_RADII:
        raise KeyError(f"Unknown element: {element}")
    return ATOMIC_RADII[element]


def get_atomic_number(element: str) -> Optional[int]:
    """
    Get atomic number for an element.
    
    Args:
        element: Element symbol (e.g., 'C', 'N', 'O').
        
    Returns:
        Atomic number or None if not found.
    """
    for atomic_num, symbol in ATOMIC_SYMBOLS.items():
        if symbol == element:
            return atomic_num
    return None


def get_element_symbol(atomic_number: int) -> Optional[str]:
    """
    Get element symbol from atomic number.
    
    Args:
        atomic_number: Atomic number.
        
    Returns:
        Element symbol or None if not found.
    """
    return ATOMIC_SYMBOLS.get(atomic_number)


def element_info(element: str) -> Dict[str, any]:
    """
    Get comprehensive information about an element.
    
    Args:
        element: Element symbol (e.g., 'C', 'N', 'O').
        
    Returns:
        Dictionary with element information.
        
    Raises:
        KeyError: If element is not found.
    """
    if element not in ATOMIC_MASSES:
        raise KeyError(f"Unknown element: {element}")
    
    atomic_number = get_atomic_number(element)
    
    info = {
        "symbol": element,
        "atomic_number": atomic_number,
        "atomic_mass": ATOMIC_MASSES[element],
        "covalent_radius": ATOMIC_RADII.get(element, 1.0),
    }
    
    # Add some basic properties based on atomic number
    if atomic_number:
        if atomic_number == 1:
            info["period"] = 1
            info["group"] = 1
            info["block"] = "s"
        elif atomic_number == 2:
            info["period"] = 1
            info["group"] = 18
            info["block"] = "s"
        elif 3 <= atomic_number <= 10:
            info["period"] = 2
            info["block"] = "s" if atomic_number <= 4 else "p"
        elif 11 <= atomic_number <= 18:
            info["period"] = 3
            info["block"] = "s" if atomic_number <= 12 else "p"
        # Add more periods as needed
    
    return info


def is_metal(element: str) -> bool:
    """
    Check if an element is a metal.
    
    Args:
        element: Element symbol.
        
    Returns:
        True if the element is a metal, False otherwise.
    """
    metals = {
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
        "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
        "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
        "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
    }
    return element in metals


def is_nonmetal(element: str) -> bool:
    """
    Check if an element is a nonmetal.
    
    Args:
        element: Element symbol.
        
    Returns:
        True if the element is a nonmetal, False otherwise.
    """
    nonmetals = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "At"}
    return element in nonmetals


def is_metalloid(element: str) -> bool:
    """
    Check if an element is a metalloid.
    
    Args:
        element: Element symbol.
        
    Returns:
        True if the element is a metalloid, False otherwise.
    """
    metalloids = {"B", "Si", "Ge", "As", "Sb", "Te"}
    return element in metalloids