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
from .optimizer import OptimizationDriver
from ..io.basic import pname,is_list_A_in_B
from ..utils.geometry import cartesian_to_fractional,fractional_to_cartesian
from .superimpose import superimpose_rotation_only


#sG:scaled and rotated G
#eG: edge graph with only edge and V node, and XOO atoms linked to the edge
#superG: supercell of sG
class Framework:
    def __init__(self, comm=None, ostream=None, MOFname=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)

        self.spacegroup = None
        self.frame_unit_cell = None
        self.frame_supercell = None
        self.frame_nodes = FrameNode()
        self.frame_linker = FrameLinker()
        self.frame_terminations = FrameTermination()
        self.frame_net = FrameNet()
        self.mof_top_library = MofTopLibrary()
        self.optimization_driver = OptimizationDriver()
        self.data_path = get_data_path()  # todo: set default data path

        self.node_metal_type = None
        self.dummy_atom_node = None
        self.linker_molecule = None
        self.linker_length = None
        self.termination_filename = None
        self.termination = True #default use termination but need user to set the termination_filename

        self.target_directory = None


        self.node_atom = None
        self.node_ccoords = None
        self.node_x_ccoords = None

        self.linker_atom = None
        self.linker_ccoords = None
        self.linker_x_ccoords = None
        self.linker_length = np.linalg.norm(self.linker_x_ccoords[0] -
                                            self.linker_x_ccoords[1])

        self.ec_atom = None
        self.ec_ccoords = None
        self.ec_x_ccoords = None
        self.constant_length = 1.54  # C-X bond length in Angstrom, default 1.54A

        self.node_max_degree = self.node_x_ccoords.shape[0]
        self.saved_optimized_rotations = None #should be h5 file

        self.to_save_optimized_rotations_filename = "rotations_opt"

        self.use_saved_rotations_as_initial_guess = True
        self.save_files = True
        self._debug = False


        #specific settings
        self.linker_length_search_range = [] #in Angstrom, [min, max]

    def _fetch_X_atoms_ind_array(self, array, column, X):
        # array: input array
        # column: column index to check for label
        # X: label to search for

        ind = [
            k for k in range(len(array)) if re.sub(r"\d", "", array[k, column]) == X
        ]
        x_array = array[ind]
        return ind, x_array

    def _fetch_net_file(self):
        self.mof_top_library.data_path = self.data_path
        self.mof_top_library.select_mof_family(self.mof_family)
        self.frame_net.cif_file = self.mof_top_library.selected_template_cif_file

    def _read_net(self):
        self._fetch_net_file()
        self.frame_net.edge_length_range = self.linker_length_search_range
        self.frame_net.create_net()
        #check if the max_degree of the net matches the node_connectivity
        assert_msg_critical(self.frame_net.max_degree == self.mof_top_library.node_connectivity,
                             "Max degree of the net does not match the node connectivity.")
        self.node_connectivity = self.frame_net.max_degree
        self.spacegroup = self.frame_net.spacegroup
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
            if self.save_files:#TODO: check if the target directory is set
                self.frame_linker.target_directory = self.target_directory
            self.frame_linker.create()
        
        self.linker_data = self.frame_linker.linker_data
        self.center_fragment_lines = np.vstack(self.frame_linker.lines)
        self.outer_fragment_lines = np.vstack(self.frame_linker.rows)

    def _fetch_node(self):
        assert_msg_critical(self.node_connectivity is not None,
                            "node_connectivity is not set")
        assert_msg_critical(self.node_metal_type is not None,
                            "node_metal_type is not set")

        nodes_database_path = Path(self.data_path, "nodes_database")
        if self.dummy_atom_node:
            keywords = [
                str(self.node_connectivity) + "c", self.node_metal_type, "dummy"
            ]
            nokeywords = []
        else:
            keywords = [str(self.node_connectivity) + "c", self.node_metal_type]
            nokeywords = ["dummy"]

        selected_node_pdb_filename = fetch_pdbfile(nodes_database_path, keywords,
                                               nokeywords)[0]
        self.frame_nodes.filename = Path(nodes_database_path, selected_node_pdb_filename)
        self.frame_nodes.node_metal_type = self.node_metal_type
        self.frame_nodes.dummy_node = self.dummy_atom_node
        self.node_data = self.frame_nodes.node_data

    def _read_node(self):
        self._fetch_node() #to fetch the node pdb file from database
        self.frame_nodes.create()
        self.node_lines = self.frame_nodes.all_atom_lines

    def _fetch_termination(self):
        if self.termination_filename is None:
            self.ostream.print_info("No termination file will be used for the nodes in the framework.")
        else:
            if not (Path(self.termination_filename).is_file()):
                #check if the termination is a name in the termination database
                keywords = [self.termination_filename]
                nokeywords = []
                terminations_database_path = Path(self.data_path, "terminations_database")
                selected_termination_pdb_filename = fetch_pdbfile(terminations_database_path, keywords,
                                                                   nokeywords)[0]
                if selected_termination_pdb_filename is not None:
                    self.termination_filename = str(
                        Path(terminations_database_path, selected_termination_pdb_filename))
                    self.frame_terminations.filename = self.termination_filename
                    return
            elif Path(self.termination_filename).is_file():
                self.frame_terminations.filename = self.termination_filename
                return

            self.ostream.print_warning(f"Termination file {self.termination_filename} does not exist, no termination will be used for the nodes in the framework.")
  

    def _read_termination(self):
        if not self.termination:
            return
        self._fetch_termination()
        if self.frame_terminations.filename is not None:
            self.frame_terminations.create()
        else:
            self.ostream.print_info("No termination file will be used for the nodes in the framework.")
            self.frame_terminations.termination_data = None
            self.termination = False

        
    def optimize_framework(self):
        self.optimization_driver.constant_length = self.constant_length
        self.optimization_driver.optimize()
        self.sorted_nodes = self.optimization_driver.sorted_nodes
        self.sorted_edges = self.optimization_driver.sorted_edges
        self.sorted_edges_of_sortednodeidx = self.optimization_driver.sorted_edges_of_sortednodeidx
        self.optimized_rotations = self.optimization_driver.optimized_rotations
        self.optimized_params = self.optimization_driver.optimized_params
        self.new_edge_length = self.optimization_driver.new_edge_length
        self.optimized_pair = self.optimization_driver.optimized_pair
        self.scaled_rotated_node_positions = self.optimization_driver.scaled_rotated_node_positions
        self.scaled_rotated_Xatoms_positions = self.optimization_driver.scaled_rotated_Xatoms_positions
        self.frame_unit_cell = self.optimization_driver.sc_unit_cell
        self.frame_unit_cell_inv = self.optimization_driver.sc_unit_cell_inv
        self.sG_node = self.optimization_driver.sG_node
        self.nodes_atom = self.optimization_driver.nodes_atom
        self.rotated_node_positions = self.optimization_driver.rotated_node_positions
        self.Xatoms_positions_dict = self.optimization_driver.Xatoms_positions_dict
        self.node_positions_dict = self.optimization_driver.node_positions_dict



        # save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_node_positions)

    def place_edge_in_net(self):
        """
        based on the optimized rotations and cell parameters, use optimized pair to find connected X-X pair in optimized cell,
        and place the edge in the target MOF cell

        return:
            sG (networkx graph):graph of the target MOF cell, with scaled and rotated node and edge positions
        """
        # linker_middle_point = np.mean(linker_x_vecs,axis=0)
        linker_xx_vec = self.linker_x_ccoords
        linker_length = self.linker_length
        optimized_pair = self.optimized_pair
        scaled_rotated_Xatoms_positions = self.scaled_rotated_Xatoms_positions
        scaled_rotated_node_positions = self.scaled_rotated_node_positions
        sorted_nodes = self.sorted_nodes
        sG_node = self.sG_node
        sc_unit_cell_inv = self.sc_unit_cell_inv
        nodes_atom = self.nodes_atom

        sG = sG_node.copy()
        scalar = (linker_length + 2 * self.constant_length) / linker_length
        extended_linker_xx_vec = [i * scalar for i in linker_xx_vec]
        norm_xx_vector_record = []
        rot_record = []

        # edges = {}
        for (i, j), pair in optimized_pair.items():
            x_idx_i, x_idx_j = pair
            reindex_i = sorted_nodes.index(i)
            reindex_j = sorted_nodes.index(j)
            x_i = scaled_rotated_Xatoms_positions[reindex_i][x_idx_i][1:]
            x_j = scaled_rotated_Xatoms_positions[reindex_j][x_idx_j][1:]
            x_i_x_j_middle_point = np.mean([x_i, x_j], axis=0)
            xx_vector = np.vstack(
                [x_i - x_i_x_j_middle_point, x_j - x_i_x_j_middle_point])
            norm_xx_vector = xx_vector / np.linalg.norm(xx_vector)

            # print(i, j, reindex_i, reindex_j, x_idx_i, x_idx_j)
            # use superimpose to get the rotation matrix
            # use record to record the rotation matrix for get rid of the repeat calculation
            indices = [
                index for index, value in enumerate(norm_xx_vector_record)
                if is_list_A_in_B(norm_xx_vector, value)
            ]
            if len(indices) == 1:
                rot = rot_record[indices[0]]
                # rot = reorthogonalize_matrix(rot)
            else:
                _, rot, _ = superimpose_rotation_only(extended_linker_xx_vec,
                                                      xx_vector)
                # rot = reorthogonalize_matrix(rot)
                norm_xx_vector_record.append(norm_xx_vector)
                # the rot may be opposite, so we need to check the angle between the two vectors
                # if the angle is larger than 90 degree, we need to reverse the rot
                roted_xx = np.dot(extended_linker_xx_vec, rot)

                if np.dot(roted_xx[1] - roted_xx[0],
                          xx_vector[1] - xx_vector[0]) < 0:
                    ##rotate 180 around the axis of the cross product of the two vectors
                    axis = np.cross(roted_xx[1] - roted_xx[0],
                                    xx_vector[1] - xx_vector[0])
                    # if 001 not linear to the two vectors
                    if np.linalg.norm(axis) == 0:
                        check_z_axis = np.cross(roted_xx[1] - roted_xx[0],
                                                [0, 0, 1])
                        if np.linalg.norm(check_z_axis) == 0:
                            axis = np.array([1, 0, 0])
                        else:
                            axis = np.array([0, 0, 1])

                    axis = axis / np.linalg.norm(axis)
                    flip_matrix = np.eye(3) - 2 * np.outer(
                        axis, axis)  # Householder matrix for reflection
                    rot = np.dot(rot, flip_matrix)
                # Flip the last column of the rotation matrix if the determinant is negative
                rot_record.append(rot)

            # use the rotation matrix to rotate the linker x coords
            placed_edge_ccoords = (np.dot(self.linker_ccoords, rot) +
                                   x_i_x_j_middle_point)

            placed_edge = np.hstack(
                (np.asarray(self.linker_atom), placed_edge_ccoords))
            sG.edges[(i, j)]["coords"] = x_i_x_j_middle_point
            sG.edges[(i, j)]["c_points"] = placed_edge

            sG.edges[(i, j)]["f_points"] = np.hstack((
                placed_edge[:, 0:2],
                cartesian_to_fractional(placed_edge[:, 2:5], sc_unit_cell_inv),
            ))  # NOTE: modified add the atom type and atom name

            _, sG.edges[(i, j)]["x_coords"] = self._fetch_X_atoms_ind_array(
                placed_edge, 0, "X")
        for k, v in scaled_rotated_node_positions.items():
            # print(k,v)
            # placed_node[k] = np.hstack((nodes_atom[k],v))
            sG.nodes[k]["c_points"] = np.hstack((nodes_atom[k], v))
            sG.nodes[k]["f_points"] = np.hstack(
                (nodes_atom[k], cartesian_to_fractional(v, sc_unit_cell_inv)))
            # find the atoms starts with "x" and extract the coordinates
            _, sG.nodes[k]["x_coords"] = self._fetch_X_atoms_ind_array(
                sG.nodes[k]["c_points"], 0, "X")
        self.sG = sG
        return sG

    def set_supercell(self, supercell):
        """
        set the supercell of the target MOF model
        """
        self.supercell = supercell

    def make_supercell_multitopic(self):
        """
        make the supercell of the multitopic linker MOF
        """
        sG = self.sG
        self.multiedge_bundlings = bundle_multiedge(sG)
        # self.dv_v_pairs, sG = replace_DV_with_corresponding_V(sG) #debug
        superG = update_supercell_node_fpoints_loose(sG, self.supercell)
        superG = update_supercell_edge_fpoints(sG, superG, self.supercell)
        # self.prim_multiedge_bundlings = replace_bundle_dvnode_with_vnode(  #debug
        #    self.dv_v_pairs, self.multiedge_bundlings
        # )
        self.prim_multiedge_bundlings = self.multiedge_bundlings
        self.super_multiedge_bundlings = make_super_multiedge_bundlings(
            self.prim_multiedge_bundlings, self.supercell)
        superG = update_supercell_bundle(superG, self.super_multiedge_bundlings)
        superG = check_multiedge_bundlings_insuperG(
            self.super_multiedge_bundlings, superG)
        self.superG = superG
        return superG

    def make_supercell_ditopic(self):
        """
        make the supercell of the ditopic linker MOF
        """

        sG = self.sG
        # self.dv_v_pairs, sG = replace_DV_with_corresponding_V(sG)
        superG = update_supercell_node_fpoints_loose(sG, self.supercell)
        superG = update_supercell_edge_fpoints(sG, superG, self.supercell)
        self.superG = superG
        return superG

    def set_virtual_edge(self, bool_x=False, range=0.5, max_neighbor=2):
        """
        set the virtual edge addition for the bridge type nodes,
        range is the range to search the virtual edge between two Vnodes directly, should <= 0.5,
        max_neighbor is the maximum number of neighbors of the node with virtual edge
        """

        self.add_virtual_edge = bool(bool_x)
        self.vir_edge_range = range
        self.vir_edge_max_neighbor = max_neighbor

    def add_virtual_edge_for_bridge_node(self, superG):
        """
        after setting the virtual edge search, add the virtual edge to the target supercell superG MOF
        """
        if self.add_virtual_edge:
            add_superG = add_virtual_edge(
                self.sc_unit_cell,
                superG,
                self.vir_edge_range,
                self.vir_edge_max_neighbor,
            )
            print("add virtual edge")
            return add_superG
        else:
            return superG

    def set_remove_node_list(self, remove_node_list):
        """
        make defect in the target MOF model by removing nodes
        """
        self.remove_node_list = remove_node_list

    def set_remove_edge_list(self, remove_edge_list):
        """
        make defect in the target MOF model by removing edges
        """
        self.remove_edge_list = remove_edge_list

    def make_eG_from_supereG_multitopic(self):
        """
        make the target MOF cell graph with only EDGE and V, link the XOO atoms to the EDGE
        always need to execute with make_supercell_multitopic
        """

        eG, _ = superG_to_eG_multitopic(self.superG, self.sc_unit_cell)
        self.eG = eG
        return eG

    def add_xoo_to_edge_multitopic(self):
        eG = self.eG
        eG, unsaturated_linker, matched_vnode_xind, xoo_dict = addxoo2edge_multitopic(
            eG, self.sc_unit_cell)
        self.unsaturated_linker = unsaturated_linker
        self.matched_vnode_xind = matched_vnode_xind
        self.xoo_dict = xoo_dict
        self.eG = eG
        return eG

    def make_eG_from_supereG_ditopic(self):
        """
        make the target MOF cell graph with only EDGE and V, link the XOO atoms to the EDGE
        always execute with make_supercell_ditopic
        """

        eG, _ = superG_to_eG_ditopic(self.superG)
        self.eG = eG
        return eG

    def add_xoo_to_edge_ditopic(self):
        """
        analyze eG and link the XOO atoms to the EDGE, update eG, for ditopic linker MOF
        """
        eG = self.eG
        eG, unsaturated_linker, matched_vnode_xind, xoo_dict = addxoo2edge_ditopic(
            eG, self.sc_unit_cell)
        self.unsaturated_linker = unsaturated_linker
        self.matched_vnode_xind = matched_vnode_xind
        self.xoo_dict = xoo_dict
        self.eG = eG
        return eG

    def main_frag_eG(self):
        """
        only keep the main fragment of the target MOF cell, remove the other fragments, to avoid the disconnected fragments
        """
        eG = self.eG
        self.eG = [eG.subgraph(c).copy() for c in nx.connected_components(eG)
                  ][0]
        print("main fragment of the MOF cell is kept"
             )  # ,len(self.eG.nodes()),'nodes')
        # print('fragment size list:',[len(c) for c in nx.connected_components(eG)]) #debug
        return self.eG

    def make_supercell_range_cleaved_eG(self, buffer_plus=0, buffer_minus=0):
        supercell = self.supercell
        new_eG = self.eG.copy()
        eG = self.eG
        removed_edges = []
        removed_nodes = []
        for n in eG.nodes():
            if pname(n) != "EDGE":
                if check_supercell_box_range(eG.nodes[n]["fcoords"], supercell,
                                             buffer_plus, buffer_minus):
                    pass
                else:
                    new_eG.remove_node(n)
                    removed_nodes.append(n)
            elif pname(n) == "EDGE":
                if (arr_dimension(eG.nodes[n]["fcoords"]) == 2
                   ):  # ditopic linker have two points in the fcoords
                    edge_coords = np.mean(eG.nodes[n]["fcoords"], axis=0)
                elif (
                        arr_dimension(eG.nodes[n]["fcoords"]) == 1
                ):  # multitopic linker have one point in the fcoords from EC
                    edge_coords = eG.nodes[n]["fcoords"]

                if check_supercell_box_range(edge_coords, supercell,
                                             buffer_plus, buffer_minus):
                    pass
                else:
                    new_eG.remove_node(n)
                    removed_edges.append(n)

        matched_vnode_xind = self.matched_vnode_xind
        self.matched_vnode_xind = update_matched_nodes_xind(
            removed_nodes,
            removed_edges,
            matched_vnode_xind,
        )

        self.eG = new_eG
        return new_eG, removed_edges, removed_nodes

    def set_node_topic(self, node_topic):
        """
        manually set the node topic, normally should be the same as the maximum degree of the node in the template
        """
        self.node_topic = node_topic

    def find_unsaturated_node_eG(self):
        """
        use the eG to find the unsaturated nodes, whose degree is less than the node topic
        """
        eG = self.eG
        if hasattr(self, "node_topic"):
            node_topic = self.node_topic
        else:
            node_topic = self.node_max_degree
        unsaturated_node = find_unsaturated_node(eG, node_topic)
        self.unsaturated_node = unsaturated_node
        return unsaturated_node

    def find_unsaturated_linker_eG(eG, linker_topics):
        """
        use the eG to find the unsaturated linkers, whose degree is less than linker topic
        """
        new_unsaturated_linker = find_unsaturated_linker(eG, linker_topics)
        return new_unsaturated_linker

    def set_node_terminamtion(self, term_file):
        """
        pdb file, set the node termination file, which contains the information of the node terminations, should have X of connected atom (normally C),
        Y of two connected O atoms (if in carboxylate group) to assist the placement of the node terminations
        """

        term_data = termpdb(term_file)
        term_info = term_data[:, :-3]
        term_coords = term_data[:, -3:]
        xterm, _ = Xpdb(term_data, "X")
        oterm, _ = Xpdb(term_data, "Y")
        term_xvecs = xterm[:, -3:]
        term_ovecs = oterm[:, -3:]
        term_coords = term_coords.astype("float")
        term_xvecs = term_xvecs.astype("float")
        term_ovecs = term_ovecs.astype("float")

        term_ovecs_c = np.mean(np.asarray(term_ovecs), axis=0)
        term_coords = term_coords - term_ovecs_c
        term_xoovecs = np.vstack((term_xvecs, term_ovecs))
        term_xoovecs = term_xoovecs - term_ovecs_c
        self.node_termination = term_file
        self.term_info = term_info
        self.term_coords = term_coords
        self.term_xoovecs = term_xoovecs

    # Function to add node terminations
    def add_terminations_to_unsaturated_node(self):
        """
        use the node terminations to add terminations to the unsaturated nodes

        """
        unsaturated_node = [
            n for n in self.unsaturated_node if n in self.eG.nodes()
        ]
        xoo_dict = self.xoo_dict
        matched_vnode_xind = self.matched_vnode_xind
        eG = self.eG
        sc_unit_cell = self.sc_unit_cell
        (
            unsaturated_vnode_xind_dict,
            unsaturated_vnode_xoo_dict,
            self.matched_vnode_xind_dict,
        ) = make_unsaturated_vnode_xoo_dict(unsaturated_node, xoo_dict,
                                            matched_vnode_xind, eG,
                                            sc_unit_cell)
        # term_file: path to the termination file
        # ex_node_cxo_cc: exposed node coordinates

        node_oovecs_record = []
        for n in eG.nodes():
            eG.nodes[n]["term_c_points"] = {}
        for exvnode_xind_key in unsaturated_vnode_xoo_dict.keys():
            exvnode_x_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "x_cpoints"]
            exvnode_oo_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "oo_cpoints"]
            node_xoo_ccoords = np.vstack(
                [exvnode_x_ccoords, exvnode_oo_ccoords])
            # make the beginning point of the termination to the center of the oo atoms
            node_oo_center_cvec = np.mean(
                exvnode_oo_ccoords[:, 2:5].astype(float),
                axis=0)  # NOTE: modified add the atom type and atom name
            node_xoo_cvecs = (node_xoo_ccoords[:, 2:5].astype(float) -
                              node_oo_center_cvec
                             )  # NOTE: modified add the atom type and atom name
            node_xoo_cvecs = node_xoo_cvecs.astype("float")
            # use record to record the rotation matrix for get rid of the repeat calculation

            indices = [
                index for index, value in enumerate(node_oovecs_record)
                if is_list_A_in_B(node_xoo_cvecs, value[0])
            ]
            if len(indices) == 1:
                rot = node_oovecs_record[indices[0]][1]
            else:
                _, rot, _ = superimpose(self.term_xoovecs, node_xoo_cvecs)
                node_oovecs_record.append((node_xoo_cvecs, rot))
            adjusted_term_vecs = np.dot(self.term_coords,
                                        rot) + node_oo_center_cvec
            adjusted_term = np.hstack((
                np.asarray(self.term_info[:, 0:1]),
                np.asarray(self.term_info[:, 2:3]),
                adjusted_term_vecs,
            ))
            # add the adjusted term to the terms, add index, add the node name
            unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "node_term_c_points"] = (adjusted_term)
            eG.nodes[exvnode_xind_key[0]]["term_c_points"][
                exvnode_xind_key[1]] = (adjusted_term)

        self.unsaturated_vnode_xoo_dict = unsaturated_vnode_xoo_dict
        self.eG = eG
        return eG

    def remove_xoo_from_node(self):
        """
        remove the XOO atoms from the node after adding the terminations, add ['noxoo_f_points'] to the node in eG
        """
        eG = self.eG
        xoo_dict = self.xoo_dict

        all_xoo_indices = []
        for x_ind, oo_ind in xoo_dict.items():
            all_xoo_indices.append(x_ind)
            all_xoo_indices.extend(oo_ind)

        for n in eG.nodes():
            if pname(n) != "EDGE":
                all_f_points = eG.nodes[n]["f_points"]
                noxoo_f_points = np.delete(all_f_points,
                                           all_xoo_indices,
                                           axis=0)
                eG.nodes[n]["noxoo_f_points"] = noxoo_f_points
        self.eG = eG

        return eG

    def get_node_edge_term_grolines(self, eG, sc_unit_cell):
        nodes_eG, edges_eG, terms_eG, node_res_num, edge_res_num, term_res_num = (
            extract_node_edge_term(eG, sc_unit_cell))
        merged_node_edge_term = merge_node_edge_term(nodes_eG, edges_eG,
                                                     terms_eG, node_res_num,
                                                     edge_res_num)
        print("node_res_num: ", node_res_num)
        print("edge_res_num: ", edge_res_num)
        print("term_res_num: ", term_res_num)
        return merged_node_edge_term

    def extract_node_edge_term(self):
        self.nodes_eG, self.edges_eG, self.terms_eG, self.node_res_num, self.edge_res_num, self.term_res_num = (
            extract_node_edge_term(self.eG, self.sc_unit_cell))
        print("node_res_num: ", self.node_res_num)
        print("edge_res_num: ", self.edge_res_num)
        print("term_res_num: ", self.term_res_num)

    def write_node_edge_term_gro(self, gro_name):
        """
        write the node, edge, term to the gro file
        """

        merged_node_edge_term = merge_node_edge_term(self.nodes_eG,
                                                     self.edges_eG,
                                                     self.terms_eG,
                                                     self.node_res_num,
                                                     self.edge_res_num)
        dir_name = "output_gros"
        save_node_edge_term_gro(merged_node_edge_term, gro_name, dir_name)
        print(str(gro_name) + ".gro is saved in folder " + str(dir_name))

        self.merged_node_edge_term = merged_node_edge_term

