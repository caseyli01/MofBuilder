import sys
from pathlib import Path

import numpy as np
import networkx as nx
import mpi4py.MPI as MPI
import h5py
import re

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.molecule import Molecule

from ..io.basic import nn, nl,pname,is_list_A_in_B,lname,arr_dimension
from ..utils.geometry import (
    unit_cell_to_cartesian_matrix, fractional_to_cartesian, cartesian_to_fractional,
    locate_min_idx, reorthogonalize_matrix, find_optimal_pairings, find_edge_pairings, Carte_points_generator
)
from .other import fetch_X_atoms_ind_array, find_pair_x_edge_fc, order_edge_array
from .superimpose import superimpose_rotation_only,superimpose



class DefectGenerator:
    """
    find unstatuirated node 
    remove node
    remove linker
    exchange node by superimpose
    exchange linker by superimpose
    """
    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)

        #need to be set before use
        self.cleaved_eG = None 
        self.node_topics = None 
        self.linker_coord_sites = None
        self.nodes_idx2rm = []
        self.linkers_idx2rm = []
        self.nodes_idx2ex = []
        self.linkers_idx2ex = []


        #will be set after use
        self.unsaturated_node = None #list of unsaturated node name
        self.unsaturated_linker = None #list of unsaturated linker name
        self.matched_vnode_xind = None #list of tuples/lists of the form (node, xind, edge)
        self.xoo_dict = None #dict of xoo atom coordinates, key is the node name, value is a dict, key is the x index, value is a dict with keys 'x_cpoints' and 'oo_cpoints'

    def remove_items(self):
        self.nodes_name2rm = extract_node_name_from_gro_resindex(self.nodes_idx2rm, nodes_dict)
        self.linkers_name2rm = extract_node_name_from_gro_resindex(self.linkers_idx2rm, linkers_dict)
        #cleave
        #xoo

    def exchange_items(self):
        self.nodes_name2ex = extract_node_name_from_gro_resindex(self.nodes_idx2ex, nodes_dict)
        self.linkers_name2ex = extract_node_name_from_gro_resindex(self.linkers_idx2ex, linkers_dict)

    def _find_unsaturated_items(self):
        self.unsaturated_nodes = self._find_unsaturated_node(self.cleaved_eG, self.node_topics)
        self.unsaturated_linkers = self._find_unsaturated_linker(self.cleaved_eG, self.linker_coord_sites)
        self.ostream.print_info(f"unsaturated nodes: {len(self.unsaturated_nodes)}, unsaturated linkers: {len(self.unsaturated_linkers)}")
        self.ostream.print_info(f"unsaturated nodes: {self.unsaturated_nodes}")
        self.ostream.print_info(f"unsaturated linkers: {self.unsaturated_linkers}")

    def _find_unsaturated_node(self, eG, node_topics):
        # find unsaturated node V in eG
        unsaturated_node = []
        for n in eG.nodes():
            if pname(n) != "EDGE":
                real_neighbor = []
                for cn in eG.neighbors(n):
                    if eG.edges[(n, cn)]["type"] == "real":
                        real_neighbor.append(cn)
                if len(real_neighbor) < node_topics:
                    unsaturated_node.append(n)
        return unsaturated_node

    def _find_unsaturated_linker(self, eG, linker_topics):
        # find unsaturated linker in eG
        unsaturated_linker = []
        for n in eG.nodes():
            if pname(n) == "EDGE" and len(list(eG.neighbors(n))) < linker_topics:
                unsaturated_linker.append(n)
        return unsaturated_linker


     # functions for defects are under construction
    
    def remove(self,
               nodes=[],
               linkers=[],
               update_node_termination=False,
               clean_unsaturated_linkers=False):


        #self.defective_net.eG = self.archive_eG.copy()
        remove_node_list = nodes
        remove_edge_list = linkers

        remove_edge_list = [
            str(int(i) - len(self.nodes_eG)) for i in remove_edge_list
        ]  # TODO: check if it is correct

        self.to_remove_nodes_name = extract_node_name_from_gro_resindex(
            remove_node_list, self.net.nodes_eG)
        self.to_remove_edges_name = extract_node_name_from_gro_resindex(
            remove_edge_list, self.net.edges_eG)

        # cleave the eG to the custom supercell box
        if self.linker_topic == 2:
            self.defective_net.add_xoo_to_edge_ditopic()
        elif self.linker_topic > 2:
            self.defective_net.add_xoo_to_edge_multitopic()
        # clean all unsaturated linkers
        if clean_unsaturated_linkers:
            self.to_remove_edges_name.update(self.saved_eG_unsaturated_linker)

        for node_name in self.to_remove_nodes_name:
            if node_name in self.defective_net.eG.nodes():
                self.defective_net.eG.remove_node(node_name)
        for edge_name in self.to_remove_edges_name:
            neighbors = list(self.defective_net.eG.neighbors(edge_name))
            if len(neighbors) == 2:  # ditopic linker case
                self.defective_net.eG.remove_edge(neighbors[0], neighbors[1])
            if edge_name in self.defective_net.eG.nodes():
                self.defective_net.eG.remove_node(edge_name)


        #print(self.to_remove_edges_name, " will be removed edge")
        #print(self.to_remove_nodes_name, " will be removed node")
        # update the matched_vnode_xind
        self.defective_net.matched_vnode_xind = update_matched_nodes_xind(
            self.to_remove_nodes_name,
            self.to_remove_edges_name,
            self.defective_net.matched_vnode_xind,
        )
        # sort subgraph by connectivity

        self.defective_net.make_supercell_range_cleaved_eG()

        if update_node_termination:
            self.defective_net.find_unsaturated_node_eG()
        else:
            self.defective_net.unsaturated_node = self.saved_eG_unsaturated_node
            self.defective_net.matched_vnode_xind = self.saved_eG_matched_vnode_xind

        #self.defective_net.add_terminations_to_unsaturated_node()
        #self.defective_net.remove_xoo_from_node()

        #self.defective_mofG = self.defective_net.eG.copy()
        #self.defective_net.extract_node_edge_term()
        #self.defective_mof_nodes_eG = self.defective_net.nodes_eG.copy()
        #self.defective_mof_edges_eG = self.defective_net.edges_eG.copy()
        #self.defective_mof_terms_eG = self.defective_net.terms_eG.copy()

    def exchange_linkers(self, linkers=[], exchange_linker_pdb=None):

        defective_net.eG = replace_edges_by_callname(
            exchange_edges_name,
            defective_net.eG,
            defective_net.sc_unit_cell_inv,
            exchange_linker_pdb,
            prefix="R",
        )

        self.defective_net = defective_net
        self.defective_mofG = self.defective_net.eG.copy()


    def terminate_nodes(self):

        self._add_termination_to_unsaturated_node_eG()
        self._remove_xoo_from_node() 



class MofWriter:
    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)

        #write the eG to the file
        #need to be set before use
        self.termination_data = None #data of the termination object
        self.termindation_X_data = None #data of the termination object X atoms
        self.cleaved_eG = None
        self.sc_cell_info = None #a list of 6 elements, a,b,c,alpha,beta,gamma
        self.sc_unit_cell= None #3x3 matrix of the supercell unit cell
        self.sc_unit_cell_inv = None #inverse of the supercell unit cell matrix

     # Function to add node terminations
    def _add_terminations_to_unsaturated_node(self):
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
        ) = self._make_unsaturated_vnode_xoo_dict(unsaturated_node, xoo_dict,
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

    def _make_unsaturated_vnode_xoo_dict(self, unsaturated_node, xoo_dict, matched_vnode_xind, eG, sc_unit_cell):
        """
        make a dictionary of the unsaturated node and the exposed X connected atom index and the corresponding O connected atoms
        """

        # process matched_vnode_xind make it to a dictionary
        matched_vnode_xind_dict = {}
        for [k, v, e] in matched_vnode_xind:
            if k in matched_vnode_xind_dict.keys():
                matched_vnode_xind_dict[k].append(v)
            else:
                matched_vnode_xind_dict[k] = [v]

        unsaturated_vnode_xind_dict = {}
        xoo_keys = list(xoo_dict.keys())
        # for each unsaturated node, get the upmatched x index and xoo atoms
        for unsat_v in unsaturated_node:
            if unsat_v in matched_vnode_xind_dict.keys():
                unsaturated_vnode_xind_dict[unsat_v] = [
                    i for i in xoo_keys if i not in matched_vnode_xind_dict[unsat_v]
                ]
                # print(unsaturated_vnode_xind_dict[unsat_v],'unsaturated_vnode_xind_dict[unsat_v]') #DEBUG
            else:
                unsaturated_vnode_xind_dict[unsat_v] = xoo_keys

        # based on the unsaturated_vnode_xind_dict, add termination to the unsaturated node xoo
        # loop over unsaturated nodes, and find all exposed X atoms and use paied xoo atoms to form a termination
        unsaturated_vnode_xoo_dict = {}
        for vnode, exposed_x_indices in unsaturated_vnode_xind_dict.items():
            for xind in exposed_x_indices:
                x_fpoints = eG.nodes[vnode]["f_points"][xind]
                x_cpoints = np.hstack(
                    (x_fpoints[0:2],
                    fractional_to_cartesian(x_fpoints[2:5], sc_unit_cell)
                    ))  # NOTE: modified add the atom type and atom name
                oo_ind_in_vnode = xoo_dict[xind]
                oo_fpoints_in_vnode = [
                    eG.nodes[vnode]["f_points"][i] for i in oo_ind_in_vnode
                ]
                oo_fpoints_in_vnode = np.vstack(oo_fpoints_in_vnode)
                oo_cpoints = np.hstack((
                    oo_fpoints_in_vnode[:, 0:2],
                    fractional_to_cartesian(oo_fpoints_in_vnode[:, 2:5],
                                            sc_unit_cell),
                ))  # NOTE: modified add the atom type and atom name

                unsaturated_vnode_xoo_dict[(vnode, xind)] = {
                    "xind": xind,
                    "oo_ind": oo_ind_in_vnode,
                    "x_fpoints": x_fpoints,
                    "x_cpoints": x_cpoints,
                    "oo_fpoints": oo_fpoints_in_vnode,
                    "oo_cpoints": oo_cpoints,
                }

        return (
            unsaturated_vnode_xind_dict,
            unsaturated_vnode_xoo_dict,
            matched_vnode_xind_dict,
        )


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
        
        
 

    def _find_unsaturated_node(eG, node_topics):
        # find unsaturated node V in eG
        unsaturated_node = []
        for n in eG.nodes():
            if pname(n) != "EDGE":
                real_neighbor = []
                for cn in eG.neighbors(n):
                    if eG.edges[(n, cn)]["type"] == "real":
                        real_neighbor.append(cn)
                if len(real_neighbor) < node_topics:
                    unsaturated_node.append(n)
        return unsaturated_node


    def _find_unsaturated_linker(eG, linker_topics):
        # find unsaturated linker in eG
        unsaturated_linker = []
        for n in eG.nodes():
            if pname(n) == "EDGE" and len(list(eG.neighbors(n))) < linker_topics:
                unsaturated_linker.append(n)
        return unsaturated_linker





# functions for write
# write gro file
def extract_node_edge_term(tG, sc_unit_cell):
    nodes_tG = []
    terms_tG = []
    edges_tG = []
    node_res_num = 0
    term_res_num = 0
    edge_res_num = 0
    nodes_check_set = set()
    nodes_name_set = set()
    edges_check_set = set()
    edges_name_set = set()
    terms_check_set = set()
    terms_name_set = set()
    for n in tG.nodes():
        if pname(n) != "EDGE":
            postions = tG.nodes[n]["noxoo_f_points"]
            name = tG.nodes[n]["name"]
            nodes_check_set.add(len(postions))
            nodes_name_set.add(name)
            if len(nodes_check_set) > len(nodes_name_set):
                raise ValueError(
                    "node index is not continuous, MOF have too many mixed nodes?"
                )
            node_res_num += 1
            nodes_tG.append(
                np.hstack((
                    np.tile(
                        np.array([node_res_num, name]),
                        (len(postions), 1)),  # residue number and residue name
                    postions[:, 1:2],  # atom type (element)
                    fractional_to_cartesian(
                        postions[:,
                                 2:5], sc_unit_cell),  # Cartesian coordinates
                    postions[:, 0:1],  # atom name
                    np.tile(np.array([n]), (len(postions), 1)),
                )))  # node name in eG is added to the last column
            if "term_c_points" in tG.nodes[n]:
                for term_ind_key, c_positions in tG.nodes[n][
                        "term_c_points"].items():
                    terms_check_set.add(len(c_positions))
                    name = "T" + tG.nodes[n]["name"]
                    terms_name_set.add(name)
                    if len(terms_check_set) > len(terms_name_set):
                        raise ValueError("term index is not continuous")

                    term_res_num += 1
                    terms_tG.append(
                        np.hstack((
                            np.tile(
                                np.array([term_res_num, name]),
                                (len(c_positions), 1),
                            ),  # residue number and residue name
                            c_positions[:, 1:2],  # atom type (element)
                            c_positions[:, 2:5],  # Cartesian coordinates
                            c_positions[:, 0:1],  # atom name
                            np.tile(np.array([term_ind_key]),
                                    (len(c_positions), 1)),
                        )))  # term name in eG is added to the last column

        elif pname(n) == "EDGE":
            postions = np.vstack(
                (tG.nodes[n]["f_points"], tG.nodes[n]["xoo_f_points"]))
            name = tG.nodes[n]["name"]
            edges_check_set.add(len(postions))
            edges_name_set.add(name)
            if len(edges_check_set) > len(edges_name_set):
                print(edges_check_set)
                # raise ValueError('edge atom number is not continuous')
                print(
                    "edge atom number is not continuous,ERROR edge name:",
                    len(edges_check_set),
                    len(edges_name_set),
                )
            edge_res_num += 1
            edges_tG.append(
                np.hstack((
                    np.tile(
                        np.array([edge_res_num, name]),
                        (len(postions), 1)),  # residue number and residue name
                    postions[:, 1:2],  # atom type (element)
                    fractional_to_cartesian(
                        postions[:,
                                 2:5], sc_unit_cell),  # Cartesian coordinates
                    postions[:, 0:1],  # atom name
                    np.tile(np.array([n]), (len(postions), 1)),
                )))  # edge name in eG is added to the last column

    return nodes_tG, edges_tG, terms_tG, node_res_num, edge_res_num, term_res_num


def check_supercell_box_range(point, supercell, buffer_plus, buffer_minus):
    # to cleave eG to supercell box

    supercell_x = supercell[0] + buffer_plus
    supercell_y = supercell[1] + buffer_plus
    supercell_z = supercell[2] + buffer_plus
    if (point[0] >= 0 + buffer_minus and point[0] <= supercell_x and
            point[1] >= 0 + buffer_minus and point[1] <= supercell_y and
            point[2] >= 0 + buffer_minus and point[2] <= supercell_z):
        return True
    else:
        # print(point, 'out of supercell box range:  [',supercell_x,supercell_y,supercell_z, '],   will be excluded') #debug
        return False


def replace_edges_by_callname(edge_n_list,
                              eG,
                              sc_unit_cell_inv,
                              new_linker_pdb,
                              prefix="R"):
    new_linker_atoms, new_linker_ccoords, new_linker_x_ccoords = process_node_pdb(
        new_linker_pdb, "X")
    for edge_n in edge_n_list:
        # check if edge_n is in eG
        if edge_n not in eG.nodes():
            print("this linker is not in MOF, will be skipped", edge_n)
            continue
        edge_n = edge_n
        edge_f_points = eG.nodes[edge_n]["f_points"]
        x_indices = [
            i for i in range(len(edge_f_points))
            if nn(edge_f_points[i][0]) == "X"
        ]
        edge_x_points = edge_f_points[x_indices]
        edge_com = np.mean(edge_x_points[:, 2:5].astype(float), axis=0)
        edge_x_fcoords = edge_x_points[:, 2:5].astype(float) - edge_com

        new_linker_x_fcoords = cartesian_to_fractional(new_linker_x_ccoords,
                                                       sc_unit_cell_inv)
        new_linker_fcoords = cartesian_to_fractional(new_linker_ccoords,
                                                     sc_unit_cell_inv)

        _, rot, trans = superimpose(new_linker_x_fcoords, edge_x_fcoords)
        replaced_linker_fcoords = np.dot(new_linker_fcoords, rot) + edge_com
        replaced_linker_f_points = np.hstack(
            (new_linker_atoms, replaced_linker_fcoords))

        eG.nodes[edge_n]["f_points"] = replaced_linker_f_points
        eG.nodes[edge_n]["name"] = prefix + edge_n

    return eG


# the following functions are used for the split node to metal, hho,ho,o and update name and residue number


def extract_node_name_from_gro_resindex(res_index, node_array_list):
    node_array = np.vstack(node_array_list)
    nodes_name = set()
    for node_ind in res_index:
        node_name = node_array[node_array[:, 0] == str(node_ind)][:, -1]
        name_set = set(node_name)
        nodes_name = nodes_name.union(name_set)
    return nodes_name


def make_dummy_split_node_dict(dummy_node_name):
    node_split_dict = {}
    dict_path = dummy_node_name.split(".")[0] + "_dict"
    with open(dict_path, "r") as f:
        lines = f.readlines()
    # node_res_counts = 0
    for li in lines:
        li = li.strip("\n")
        key = li[:20].strip(" ")
        value = li[-4:].strip(" ")
        node_split_dict[key] = int(value)
    return node_split_dict


def chunk_array(chunk_list, array, chunk_num, chunksize):
    chunk_list.extend(
        array[i * chunksize:(i + 1) * chunksize] for i in range(chunk_num))
    return chunk_list


def rename_node_arr(node_split_dict, node_arr):
    metal_count = node_split_dict["METAL_count"]
    dummy_len = int(node_split_dict["dummy_res_len"])
    metal_num = metal_count * dummy_len
    hho_num = node_split_dict["HHO_count"] * 3
    ho_num = node_split_dict["HO_count"] * 2
    o_num = node_split_dict["O_count"] * 1
    metal_range = metal_num
    hho_range = metal_range + hho_num
    ho_range = hho_range + ho_num
    o_range = ho_range + o_num
    # print(metal_range,hho_range,ho_range,o_range) #debug

    metals_list = []
    hhos_list = []
    hos_list = []
    os_list = []
    for idx in set(node_arr[:, 0]):
        idx_arr = node_arr[node_arr[:, 0] == idx]
        if metal_num > 0:
            metal = idx_arr[0:metal_range].copy()
            metal[:, 1] = "METAL"
            metals_list = chunk_array(metals_list, metal,
                                      node_split_dict["METAL_count"], dummy_len)
        if hho_num > 0:
            hho = idx_arr[metal_range:hho_range].copy()
            hho[:, 1] = "HHO"
            hhos_list = chunk_array(hhos_list, hho,
                                    node_split_dict["HHO_count"], 3)
        if ho_num > 0:
            ho = idx_arr[hho_range:ho_range].copy()
            ho[:, 1] = "HO"
            hos_list = chunk_array(hos_list, ho, node_split_dict["HO_count"], 2)
        if o_num > 0:
            o = idx_arr[ho_range:o_range].copy()
            o[:, 1] = "O"
            os_list = chunk_array(os_list, o, node_split_dict["O_count"], 1)

    return metals_list, hhos_list, hos_list, os_list


def merge_metal_list_to_node_array(merged_node_edge_term, metals_list, line_num,
                                   res_count):
    if any([len(metal) == 0 for metal in metals_list]):
        return merged_node_edge_term, line_num, res_count
    for i in range(len(metals_list)):
        metal = metals_list[i]
        metal[:, 0] = i + 1
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            metal, line_num, res_count)
        merged_node_edge_term += formatted_gro_lines
    res_count += len(metals_list)
    return merged_node_edge_term, line_num, res_count


def convert_node_array_to_gro_lines(array, line_num_start, res_num_start):
    formatted_gro_lines = []

    for i in range(len(array)):
        line = array[i]
        ind_inres = i + 1
        name = line[1]
        value_atom_number_in_gro = int(ind_inres + line_num_start)  # atom_number
        value_label = re.sub(r"\d", "", line[2]) + str(ind_inres)  # atom_label
        value_resname = str(name)[0:3]  # +str(eG.nodes[n]['index'])  # residue_name
        value_resnumber = int(res_num_start + int(line[0]))  # residue number
        value_x = 0.1 * float(line[3])  # x
        value_y = 0.1 * float(line[4])  # y
        value_z = 0.1 * float(line[5])  # z
        formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
            value_resnumber,
            value_resname,
            value_label,
            value_atom_number_in_gro,
            value_x,
            value_y,
            value_z,
        )
        formatted_gro_lines.append(formatted_line + "\n")
    return formatted_gro_lines, value_atom_number_in_gro


def merge_node_edge_term(nodes_tG, edges_tG, terms_tG, node_res_num,
                         edge_res_num):
    merged_node_edge_term = []
    line_num = 0
    for node in nodes_tG:
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            node, line_num, 0)
        merged_node_edge_term += formatted_gro_lines
    for edge in edges_tG:
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            edge, line_num, node_res_num)
        merged_node_edge_term += formatted_gro_lines
    for term in terms_tG:
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            term, line_num, node_res_num + edge_res_num)
        merged_node_edge_term += formatted_gro_lines
    return merged_node_edge_term


def save_node_edge_term_gro(merged_node_edge_term,
                            gro_name,
                            dir_name="output_gros"):
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    gro_name = str(Path(dir_name, gro_name))
    with open(gro_name + ".gro", "w") as f:
        head = []
        head.append("eG_NET\n")
        head.append(str(len(merged_node_edge_term)) + "\n")
        f.writelines(head)
        f.writelines(merged_node_edge_term)
        tail = ["20 20 20 \n"]
        f.writelines(tail)


#################below are from display.py######################


def gro_string_show(gro_lines_list, w=800, h=600, res_id=True, res_name=True):
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=w, height=h)
        lines = gro_lines_list

        viewer.addModel("".join(lines), "gro")
        # viewer.setStyle({"stick": {}})

        viewer.setViewStyle({"style": "outline", "width": 0.05})
        viewer.setStyle({"stick": {}, "sphere": {"scale": 0.20}})
        if res_id or res_name:
            for i in range(2, len(lines) - 1):
                if lines[i].strip() == "":
                    continue
                if lines[i - 1][0:5] == lines[i][0:5]:
                    continue

                value_resnumber = int((lines[i])[0:5])
                value_resname = lines[i][5:10]
                if value_resname.strip() == "TNO":
                    continue
                # value_label = lines[i][10:15]
                # value_atom_number = int(lines[i][15:20])
                value_x = float(lines[i][20:28]) * 10  # x
                value_y = float(lines[i][28:36]) * 10  # y
                value_z = float(lines[i][36:44]) * 10  # z

                text = ""
                if res_name:
                    text += str(value_resname)
                if res_id:
                    text += str(value_resnumber)

                viewer.addLabel(
                    text,
                    {
                        "position": {
                            "x": value_x,
                            "y": value_y,
                            "z": value_z,
                        },
                        "alignment": "center",
                        "fontColor": "white",
                        "font": "Arial",
                        "fontSize": 12,
                        "backgroundColor": "black",
                        "backgroundOpacity": 0.5,
                    },
                )
        viewer.render()
        viewer.zoomTo()
        viewer.show()
    except ImportError:
        raise ImportError("Unable to import py3Dmol")



    def _make_supercell_range_cleaved_eG(self, buffer_plus=0, buffer_minus=0):
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