API Reference
=============

This section contains the API reference for all MofBuilder modules.

Core Module
-----------
.. automodule:: mofbuilder.core
   :members:

.. autosummary::
   :toctree: core
   :recursive:

   mofbuilder.core.builder
   mofbuilder.core.topology
   mofbuilder.core.optimizer
   mofbuilder.core.node_placement
   mofbuilder.core.rotations

I/O Module
----------
.. automodule:: mofbuilder.io
   :members:

.. autosummary::
   :toctree: io
   :recursive:

   mofbuilder.io.cif_reader
   mofbuilder.io.pdb_reader
   mofbuilder.io.gro_reader
   mofbuilder.io.xyz_reader
   mofbuilder.io.itp_parser
   mofbuilder.io.mdp_parser
   mofbuilder.io.output_writer
   mofbuilder.io.mapping_writer

Graph Module
------------
.. automodule:: mofbuilder.graph
   :members:

.. autosummary::
   :toctree: graph
   :recursive:

   mofbuilder.graph.supercell
   mofbuilder.graph.process_graph
   mofbuilder.graph.edge_impl
   mofbuilder.graph.frag_recognizer
   mofbuilder.graph.learn_template

Forcefield Module
-----------------
.. automodule:: mofbuilder.forcefield
   :members:

.. autosummary::
   :toctree: forcefield
   :recursive:

   mofbuilder.forcefield.map_forcefield
   mofbuilder.forcefield.atoms2c
   mofbuilder.forcefield.gro_itps
   mofbuilder.forcefield.prepare_class

Analysis Module
---------------
.. automodule:: mofbuilder.analysis
   :members:

.. autosummary::
   :toctree: analysis
   :recursive:

   mofbuilder.analysis.porosity
   mofbuilder.analysis.surface_area
   mofbuilder.analysis.filter_distance
   mofbuilder.analysis.filter_atoms
   mofbuilder.analysis.isolated_node_cleaner
   mofbuilder.analysis.terminations
   mofbuilder.analysis.cluster

MD Module
---------
.. automodule:: mofbuilder.md
   :members:

.. autosummary::
   :toctree: md
   :recursive:

   mofbuilder.md.setup
   mofbuilder.md.run
   mofbuilder.md.traj_io
   mofbuilder.md.analysis
   mofbuilder.md.utils

Utils Module
------------
.. automodule:: mofbuilder.utils
   :members:

.. autosummary::
   :toctree: utils
   :recursive:

   mofbuilder.utils.display
   mofbuilder.utils.fetchfile
   mofbuilder.utils.config
   mofbuilder.utils.logger

Workflows Module
----------------
.. automodule:: mofbuilder.workflows
   :members:

.. autosummary::
   :toctree: workflows
   :recursive:

   mofbuilder.workflows.make_supergraph
   mofbuilder.workflows.scale_cif_optimizer
   mofbuilder.workflows.make_eG
   mofbuilder.workflows.vlx_integration
