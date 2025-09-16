MofBuilder Documentation
========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials
   api_reference
   examples
   contributing

Welcome to MofBuilder
---------------------

MofBuilder is a comprehensive Python library for building, analyzing, and visualizing Metal-Organic Framework (MOF) structures.

Features
--------

* **Core Structure Representation**: Atoms, bonds, lattices, and complete framework objects
* **File I/O Support**: Read and write CIF, XYZ, and other crystallographic formats  
* **Analysis Tools**: Pore analysis, surface area calculation, topology analysis
* **Visualization**: 2D/3D structure plotting and property visualization
* **Utilities**: Periodic table data, geometric calculations, and helper functions

Quick Example
-------------

.. code-block:: python

   from mofbuilder.core import Atom, Framework, Lattice
   from mofbuilder.analysis import PoreAnalyzer

   # Create a simple framework
   lattice = Lattice(a=10.0, b=10.0, c=10.0)
   framework = Framework("MyMOF", lattice)
   framework.add_atom(Atom("Zn", (0.0, 0.0, 0.0)))
   framework.add_atom(Atom("O", (2.0, 0.0, 0.0)))

   # Analyze pore properties
   analyzer = PoreAnalyzer()
   pore_data = analyzer.calculate_accessible_volume(framework)
   print(f"Accessible volume: {pore_data['accessible_volume']:.2f} Å³")

Installation
------------

Install MofBuilder using pip:

.. code-block:: bash

   pip install mofbuilder

Or from source:

.. code-block:: bash

   git clone https://github.com/caseyli01/MofBuilder.git
   cd MofBuilder
   pip install -e .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`