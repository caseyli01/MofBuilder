"""
MofBuilder: A Python library for building and analyzing Metal-Organic Frameworks (MOFs)

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

__version__ = "0.1.0"
__author__ = "MofBuilder Contributors"
__email__ = "chenxili@kth.se"
__license__ = "LGPL-3.0-or-later"

# Import main modules for convenient access
from . import analysis, core, io, utils, visualization,md

__all__ = ["core", "io", "utils", "analysis", "visualization", "md", "__version__"]