# MofBuilder

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/mofbuilder.svg)](https://badge.fury.io/py/mofbuilder)

A comprehensive Python library for building, analyzing, and visualizing Metal-Organic Framework (MOF) structures.

## Overview

MofBuilder provides a modular and extensible framework for working with MOF structures, featuring:

- **Core Structure Representation**: Atoms, bonds, lattices, and complete framework objects
- **File I/O Support**: Read and write CIF, XYZ, and other crystallographic formats
- **Analysis Tools**: Pore analysis, surface area calculation, topology analysis
- **Visualization**: 2D/3D structure plotting and property visualization
- **Utilities**: Periodic table data, geometric calculations, and helper functions

## Features

### 🏗️ Core Components
- **Atom**: Represent individual atoms with positions, elements, and properties
- **Bond**: Model chemical bonds between atoms
- **Lattice**: Handle crystallographic unit cells and coordinate transformations
- **Framework**: Complete MOF structure with atoms, bonds, and lattice

### 📁 File I/O
- **CIF Support**: Read and write Crystallographic Information Files
- **XYZ Support**: Handle coordinate files for molecular structures
- **Format Conversion**: Convert between different file formats
- **Batch Processing**: Handle multiple structures efficiently

### 🔬 Analysis Tools
- **Pore Analysis**: Calculate accessible volume, pore size distribution, largest sphere
- **Surface Area**: Geometric and accessible surface area calculations
- **Topology Analysis**: Coordination numbers, ring finding, secondary building units (SBUs)
- **Property Correlation**: Analyze relationships between structural properties

### 📊 Visualization
- **Structure Viewing**: 3D and 2D visualization of MOF frameworks
- **Property Plotting**: Histograms, correlations, and analysis result visualization
- **Customizable**: Flexible color schemes and visualization options

## Installation

### From PyPI (recommended)

```bash
pip install mofbuilder
```

### From Source

```bash
git clone https://github.com/caseyli01/MofBuilder.git
cd MofBuilder
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/caseyli01/MofBuilder.git
cd MofBuilder
pip install -e ".[dev]"
```

### Optional Dependencies

For enhanced functionality, install optional dependencies:

```bash
# For advanced analysis
pip install "mofbuilder[analysis]"

# For visualization
pip install "mofbuilder[visualization]"

# For documentation
pip install "mofbuilder[docs]"

# All optional dependencies
pip install "mofbuilder[dev,analysis,visualization,docs]"
```

## Quick Start

### Basic Usage

```python
from mofbuilder.core import Atom, Framework, Lattice
from mofbuilder.analysis import PoreAnalyzer
from mofbuilder.visualization import StructureViewer

# Create a simple framework
lattice = Lattice(a=10.0, b=10.0, c=10.0)
framework = Framework("MyMOF", lattice)

# Add atoms
framework.add_atom(Atom("Zn", (0.0, 0.0, 0.0)))
framework.add_atom(Atom("O", (2.0, 0.0, 0.0)))

# Analyze pore properties
analyzer = PoreAnalyzer()
pore_data = analyzer.calculate_accessible_volume(framework)
print(f"Accessible volume: {pore_data['accessible_volume']:.2f} Å³")

# Visualize structure
viewer = StructureViewer()
fig = viewer.plot_structure_3d(framework)
fig.show()
```

### File I/O Operations

```python
from mofbuilder.io import CifReader, XyzWriter

# Read a CIF file
reader = CifReader()
framework = reader.read("my_mof.cif")

# Write to XYZ format
writer = XyzWriter()
writer.write(framework, "my_mof.xyz")
```

### Command Line Interface

MofBuilder includes a command-line interface for common operations:

```bash
# Convert between file formats
mofbuilder convert input.cif output.xyz

# Analyze a structure
mofbuilder analyze my_mof.cif --pore --surface --topology

# Create visualizations
mofbuilder visualize my_mof.cif --output structure.png --type 3d
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_example.py`: Create and analyze a simple MOF structure
- `file_io_example.py`: Demonstrate file I/O operations
- `analysis_workflow.py`: Complete analysis workflow
- `visualization_gallery.py`: Various visualization examples

Run an example:

```bash
cd examples
python basic_example.py
```

## API Documentation

### Core Module

The core module provides fundamental classes for MOF representation:

```python
from mofbuilder.core import Atom, Bond, Framework, Lattice

# Create lattice
lattice = Lattice(a=12.0, b=12.0, c=12.0, alpha=90, beta=90, gamma=90)

# Create atoms
atom1 = Atom("C", position=(0.0, 0.0, 0.0), charge=0.0)
atom2 = Atom("N", position=(1.5, 0.0, 0.0), charge=-0.3)

# Create bonds
bond = Bond(atom1, atom2, bond_order=1.0, bond_type="covalent")

# Create framework
framework = Framework("Example", lattice)
framework.add_atom(atom1)
framework.add_atom(atom2)
framework.add_bond(bond)
```

### Analysis Module

Comprehensive analysis tools for MOF characterization:

```python
from mofbuilder.analysis import PoreAnalyzer, SurfaceAreaCalculator, TopologyAnalyzer

# Pore analysis
pore_analyzer = PoreAnalyzer(probe_radius=1.4)  # N2 probe
volume_data = pore_analyzer.calculate_accessible_volume(framework)
sphere_data = pore_analyzer.find_largest_sphere(framework)
psd_data = pore_analyzer.pore_size_distribution(framework)

# Surface area calculation
surface_calc = SurfaceAreaCalculator(probe_radius=1.4)
surface_data = surface_calc.calculate_geometric_surface_area(framework)

# Topology analysis
topo_analyzer = TopologyAnalyzer()
coord_data = topo_analyzer.analyze_coordination(framework)
ring_data = topo_analyzer.find_rings(framework)
sbu_data = topo_analyzer.identify_sbu(framework)
```

### Visualization Module

Create publication-quality visualizations:

```python
from mofbuilder.visualization import StructureViewer, PropertyPlotter

# Structure visualization
viewer = StructureViewer()
fig_3d = viewer.plot_structure_3d(framework, show_bonds=True, show_unit_cell=True)
fig_2d = viewer.plot_structure_2d(framework, projection='xy')

# Property visualization
plotter = PropertyPlotter()
pore_fig = plotter.plot_pore_size_distribution(psd_data)
coord_fig = plotter.plot_coordination_analysis(coord_data)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mofbuilder --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest -m "not slow"        # Skip slow tests
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/caseyli01/MofBuilder.git
cd MofBuilder
pip install -e ".[dev]"
pre-commit install
```

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Documentation

Full documentation is available at [https://mofbuilder.readthedocs.io](https://mofbuilder.readthedocs.io)

Build documentation locally:

```bash
cd docs/
make html
```

## License

This project is licensed under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later). See the [LICENSE](LICENSE) file for details.

## Citation

If you use MofBuilder in your research, please cite:

```bibtex
@software{mofbuilder2024,
  title={MofBuilder: A Python Library for MOF Structure Analysis},
  author={MofBuilder Contributors},
  year={2024},
  url={https://github.com/caseyli01/MofBuilder},
  license={LGPL-3.0-or-later}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/caseyli01/MofBuilder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/caseyli01/MofBuilder/discussions)
- **Documentation**: [ReadTheDocs](https://mofbuilder.readthedocs.io)

## Acknowledgments

MofBuilder builds upon the excellent work of the computational chemistry and materials science communities. We thank all contributors and users who help improve this software.

## Roadmap

Planned features for future releases:

- **Advanced File Formats**: Support for VASP, Quantum ESPRESSO, and other formats
- **Machine Learning Integration**: Structure-property relationship prediction
- **High-Throughput Screening**: Automated analysis of large MOF databases
- **Web Interface**: Browser-based MOF analysis and visualization
- **Database Integration**: Direct access to MOF databases (COD, CCDC, etc.)

## Related Projects

- **pymatgen**: Comprehensive materials analysis library
- **ASE**: Atomic Simulation Environment
- **Open Babel**: Chemical file format conversion
- **MDAnalysis**: Molecular dynamics analysis
- **RDKit**: Cheminformatics toolkit