import numpy as np
import networkx as nx
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.environment import get_data_path
import mpi4py.MPI as MPI
import sys
import re

from ..utils.geometry import (
    unit_cell_to_cartesian_matrix,
    fractional_to_cartesian,
    find_edge_pairings,
)
from ..utils.fetch import fetch_pdbfile
from pathlib import Path
from .net import FrameNet
from .node import FrameNode
from .linker import FrameLinker
from .termination import FrameTermination
from .moftoplibrary import MofTopLibrary
from .optimizer import NetOptimizer
from ..io.basic import pname, is_list_A_in_B
from ..utils.geometry import cartesian_to_fractional, fractional_to_cartesian
from .superimpose import superimpose_rotation_only


#sG:scaled and rotated G
#eG: edge graph with only edge and V node, and XOO atoms linked to the edge
#superG: supercell of sG
class Framework:

    def __init__(self, comm=None, ostream=None, mof_family=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)

        self.mof_family = mof_family
        self.spacegroup = None
        self.frame_unit_cell = None
        self.frame_supercell = None
        self.frame_nodes = FrameNode(comm=self.comm, ostream=self.ostream)
        self.frame_linker = FrameLinker(comm=self.comm, ostream=self.ostream)
        self.frame_terminations = FrameTermination(comm=self.comm,
                                                   ostream=self.ostream)
        self.frame_net = FrameNet(comm=self.comm, ostream=self.ostream)
        self.mof_top_library = MofTopLibrary(comm=self.comm,
                                             ostream=self.ostream)
        self.net_optimizer = NetOptimizer(comm=self.comm, ostream=self.ostream)
        self.data_path = None  # todo: set default data path

        self.node_metal_type = None
        self.dummy_atom_node = None
        self.dummy_atom_node_dict = None
        self.linker_xyzfile = None
        self.linker_topic = None
        self.linker_molecule = None
        self.linker_length = None
        self.termination_filename = None
        self.termination = False  # default use termination but need user to set the termination_filename

        self.target_directory = None

        self.node_atom = None
        self.node_ccoords = None
        self.node_x_ccoords = None

        self.linker_atom = None
        self.linker_ccoords = None
        self.linker_x_ccoords = None
        #self.linker_length = np.linalg.norm(self.linker_x_ccoords[0] -
        #                                    self.linker_x_ccoords[1])

        self.ec_atom = None
        self.ec_ccoords = None
        self.ec_x_ccoords = None
        self.constant_length = 1.54  # C-X bond length in Angstrom, default 1.54A

        #self.node_max_degree = self.node_x_ccoords.shape[0]
        self.saved_optimized_rotations = None  #should be h5 file

        self.to_save_optimized_rotations_filename = "rotations_opt"

        self.use_saved_rotations_as_initial_guess = True
        self.save_files = False
        self._debug = False

        #specific settings
        self.linker_length_search_range = []  #in Angstrom, [min, max]

    def _read_net(self):
        if self.data_path is None:
            self.data_path = get_data_path()
        self.mof_top_library._debug = self._debug
        self.mof_top_library.data_path = self.data_path
        self.frame_net.cif_file = self.mof_top_library.fetch(
            mof_family=self.mof_family)
        assert_msg_critical(
            self.frame_net.cif_file is not None,
            "Template cif file is not set in mof_top_library.")
        self.frame_net.edge_length_range = self.linker_length_search_range
        self.frame_net.create_net()
        #check if the max_degree of the net matches the node_connectivity
        assert_msg_critical(
            self.frame_net.max_degree ==
            self.mof_top_library.node_connectivity,
            "Max degree of the net does not match the node connectivity.")
        self.node_connectivity = self.frame_net.max_degree
        self.spacegroup = self.frame_net.cifreader.spacegroup
        self.cell_info = self.frame_net.cell_info
        self.G = self.frame_net.G.copy()
        self.unit_cell = self.frame_net.unit_cell
        self.unit_cell_inv = self.frame_net.unit_cell_inv
        self.linker_topic = self.frame_net.linker_topic
        self.sorted_nodes = self.frame_net.sorted_nodes
        self.sorted_edges = self.frame_net.sorted_edges
        self.pair_vertex_edge = self.frame_net.pair_vertex_edge

    def _read_linker(self):
        self.frame_linker.linker_topic = self.linker_topic
        if self.linker_molecule is not None:
            self.frame_linker.create(molecule=self.linker_molecule)
        else:
            self.frame_linker.filename = self.linker_xyzfile
            if self.save_files:  #TODO: check if the target directory is set
                self.frame_linker.target_directory = self.target_directory
            self.frame_linker.create()

        #pass linker data
        self.linker_center_data = self.frame_linker.linker_center_data
        self.linker_center_X_data = self.frame_linker.linker_center_X_data
        if self.frame_linker.linker_topic > 2:
            self.linker_outer_data = self.frame_linker.linker_outer_data
            self.linker_outer_X_data = self.frame_linker.linker_outer_X_data
            self.linker_length = np.linalg.norm(
                self.linker_outer_X_data[0, 5:8].astype(float) -
                self.linker_outer_X_data[1, 5:8].astype(float))
        else:
            self.linker_length = np.linalg.norm(
                self.linker_center_X_data[0, 5:8].astype(float) -
                self.linker_center_X_data[1, 5:8].astype(float))

    def _read_node(self):
        assert_msg_critical(self.node_connectivity is not None,
                            "node_connectivity is not set")
        assert_msg_critical(self.node_metal_type is not None,
                            "node_metal_type is not set")

        nodes_database_path = Path(self.data_path, "nodes_database")

        keywords = [str(self.node_connectivity) + "c", self.node_metal_type]
        nokeywords = ["dummy"]

        selected_node_pdb_filename = fetch_pdbfile(nodes_database_path,
                                                   keywords, nokeywords,
                                                   self.ostream)[0]
        self.frame_nodes.filename = Path(nodes_database_path,
                                         selected_node_pdb_filename)
        self.frame_nodes.node_metal_type = self.node_metal_type
        self.frame_nodes.dummy_node = self.dummy_atom_node
        self.frame_nodes.create()

        #pass node data
        self.node_data = self.frame_nodes.node_data
        self.node_X_data = self.frame_nodes.node_X_data
        self.dummy_node_dict = self.frame_nodes.dummy_node_split_dict

    def _read_termination(self):
        if not self.termination:
            return
        #try to get a valid termination file
        if self.termination_filename is None:
            assert_msg_critical(
                False,
                "Termination is set to True but termination_filename is None.")
        #termination_filename can be a file path or a name in the termination database
        #check if the termination_filename is a valid file path
        if not (Path(self.termination_filename).is_file()):
            #check if the termination is a name in the termination database
            if self._debug:
                self.ostream.print_info(
                    f"Termination file {self.termination_filename} is not a valid file path. Searching in termination database."
                )
                self.ostream.flush()
            keywords = [self.termination_filename]
            nokeywords = []
            terminations_database_path = Path(self.data_path,
                                              "terminations_database")
            selected_termination_pdb_filename = fetch_pdbfile(
                terminations_database_path, keywords, nokeywords,
                self.ostream)[0]
            assert_msg_critical(
                selected_termination_pdb_filename is not None,
                f"Termination file {self.termination_filename} does not exist in the termination database."
            )
            self.termination_filename = str(
                Path(terminations_database_path,
                     selected_termination_pdb_filename))
        if self._debug:
            self.ostream.print_info(
                f"Using termination file: {self.termination_filename}")
            self.ostream.flush()
        self.frame_terminations.filename = self.termination_filename
        self.frame_terminations.create()

        #pass termination data
        self.termination_data = self.frame_terminations.termination_data
        self.termination_X_data = self.frame_terminations.termination_X_data

    def read_framework(self):
        self._read_net()
        self._read_linker()
        self._read_node()
        self._read_termination()
        if self._debug:
            self.ostream.print_info(f"Framework components read:")
            self.ostream.print_info(
                f"  Net: {self.mof_family}, spacegroup: {self.spacegroup}, cell: {self.cell_info}"
            )
            self.ostream.print_info(
                f"  Node: {self.frame_nodes.filename} with metal type {self.node_metal_type}"
            )
            self.ostream.print_info(
                f"  Linker: {self.frame_linker.linker_topic}")
            if self.termination:
                self.ostream.print_info(
                    f"  Termination: {self.termination_filename}")
            else:
                self.ostream.print_info(f"  Termination: None")
            self.ostream.print_info("Finished reading framework components.")
            self.ostream.flush()

    def optimize_framework(self):
        self.net_optimizer._debug = self._debug
        self.net_optimizer.G = self.G.copy()
        self.net_optimizer.cell_info = self.cell_info
        self.net_optimizer.V_data = self.frame_nodes.node_data
        self.net_optimizer.V_X_data = self.frame_nodes.node_X_data
        if self.frame_net.linker_topic > 2:
            self.net_optimizer.EC_data = self.frame_linker.linker_center_data
            self.net_optimizer.EC_X_data = self.frame_linker.linker_center_X_data
            self.net_optimizer.E_data = self.frame_linker.linker_outer_data
            self.net_optimizer.E_X_data = self.frame_linker.linker_outer_X_data
        else:
            self.net_optimizer.E_data = self.frame_linker.linker_center_data
            self.net_optimizer.E_X_data = self.frame_linker.linker_center_X_data
            self.net_optimizer.EC_data = None
            self.net_optimizer.EC_X_data = None
        self.net_optimizer.constant_length = self.constant_length
        self.net_optimizer.sorted_nodes = self.frame_net.sorted_nodes
        self.net_optimizer.sorted_edges = self.frame_net.sorted_edges
        self.net_optimizer.linker_length = self.linker_length

        self.ostream.print_separator()
        self.ostream.print_info(
            "Start to optimize the node rotations and cell parameters")
        self.ostream.flush()
        self.net_optimizer.rotation_and_cell_optimization()
        self.ostream.print_info("--------------------------------")
        self.ostream.print_info(
            "Finished optimizing the node rotations and cell parameters")
        self.ostream.print_separator()
        self.net_optimizer._debug = False
        self.net_optimizer.place_edge_in_net()
        #here we can get the unit cell with nodes and edges placed
        self.net_optimizer.sG  #scaled and rotated G

        # save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_node_positions)


if __name__ == "__main__":
    mof = Framework(mof_family="UiO-66")
    mof.data_path = 'tests/database'
    mof.linker_xyzfile = 'tests/database/linker4test/bdc.xyz'
    mof.termination = True
    mof.termination_filename = 'methyl'
    mof.frame_net._debug = False
    mof.frame_linker._debug = False
    mof.frame_nodes._debug = False
    mof.frame_terminations._debug = False
    mof.node_metal_type = "Zr"
    mof.dummy_atom_node = True
    mof.read_framework()
    mof.optimize_framework()

    #optimize the roations and scale the net of unit cell to fit the linker length
    #mof.optimize_framework()
    #mof.update_framework() #defects and supercell
    #mof.terminate_framework()
    #mof.write_framework()
