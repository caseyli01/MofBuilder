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

from ..io.basic import nn, nl, pname, is_list_A_in_B, lname, arr_dimension
from ..utils.geometry import (unit_cell_to_cartesian_matrix,
                              fractional_to_cartesian, cartesian_to_fractional,
                              locate_min_idx, reorthogonalize_matrix,
                              find_optimal_pairings, find_edge_pairings,
                              Carte_points_generator)
from .other import fetch_X_atoms_ind_array, find_pair_x_edge_fc, order_edge_array
from .superimpose import superimpose_rotation_only, superimpose



class TerminationDefectGenerator:
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
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)

        #need to be set before use
        self.cleaved_eG = None
        self.node_connectivity = None
        self.linker_connectivity = None
        self.eG_index_name_dict = None
        self.sc_unit_cell = None
        self.sc_unit_cell_inv = None
        #for remove
        self.res_idx2rm = []
        self.matched_vnode_xind = None
        self.xoo_dict = None
        self.unsaturated_node = []  # list of unsaturated node name
        self.unsaturated_linker = []  # list of unsaturated linker name
        self.termination_data = []  # list of termination XOO vectors
        self.termination_X_data = []  # list of termination X atoms
        self.termination_Y_data = []  # list of termination Y atoms

        self.use_termination = False

        self.clean_unsaturated_linkers = False
        self.update_node_termination = True

        #for exchange
        self.nodes_idx2ex = []
        self.linkers_idx2ex = []
        self.exchange_node_data = None
        self.exchange_node_X_data = None
        self.exchange_linker_data = None
        self.exchange_linker_X_data = None

        #will be set after use
        self.saved_eG_matched_vnode_xind = self.matched_vnode_xind  #saved matched_vnode_xind before remove items
        self.defectG = None  #eG after removing nodes or linkers
        self.termG = None  #eG after adding terminations
        self.finalG = None  #eG after removing xoo from node

        #debug
        self._debug = False
    

    def remove_items_or_terminate(self, res_idx2rm, cleaved_eG):
        #cleave
        #xoo
        if not res_idx2rm:
            defectG = cleaved_eG.copy()
            if not self.use_termination:
                return defectG
            else:
                termG, unsaturated_vnode_xoo_dict = self._add_terminations_to_unsaturated_node(defectG)
                return termG

            
        self.ostream.print_info(f"trying removing items: {res_idx2rm}")
        nodes_names2rm = self._extract_node_name_from_eG_dict(res_idx2rm, self.eG_index_name_dict)
        self.ostream.print_info(f"removing nodes/linkers: {nodes_names2rm}")
        self.ostream.flush()

        defectG = cleaved_eG.copy()
        # clean all unsaturated linkers
        #add all unsaturated linkers to the to_remove_edges_name
        if self.clean_unsaturated_linkers:
            nodes_names2rm.append(self.saved_unsaturated_linker) if self.saved_unsaturated_linker else None
            self.ostream.print_info(
                f"clean unsaturated linkers: {self.saved_unsaturated_linker}"
            )

        for node_name in nodes_names2rm:
            if node_name in defectG.nodes():
                defectG.remove_node(node_name)
                self.ostream.print_info(f"node {node_name} removed")

        new_unsaturated_node = self._find_unsaturated_node(defectG, self.node_connectivity)
        new_unsaturated_linker = self._find_unsaturated_linker(defectG, self.linker_connectivity)
        old_unsaturated_node = self.unsaturated_node
        old_unsaturated_linker = self.unsaturated_linker

        if not self.use_termination:
            if self._debug:
                self.ostream.print_info("no termination, return the defectG")
            return defectG

        if self.update_node_termination:
            self.unsaturated_node = new_unsaturated_node
            self.matched_vnode_xind = self._update_matched_nodes_xind(
                nodes_names2rm, self.matched_vnode_xind)
            self.saved_eG_matched_vnode_xind = self.matched_vnode_xind
            #add termination to the new unsaturated node
            termG, unsaturated_vnode_xoo_dict = self._add_terminations_to_unsaturated_node(defectG)
            return termG

        else:
            self.unsaturated_node = old_unsaturated_node
            self.matched_vnode_xind = self.saved_eG_matched_vnode_xind
            #add termination to the old unsaturated node
            termG, unsaturated_vnode_xoo_dict = self._add_terminations_to_unsaturated_node(defectG)
            return termG


    def exchange_items(self, res_idx2ex, G):
        nodes_name2ex = self._extract_node_name_from_eG_dict(
            res_idx2ex, self.eG_index_name_dict)
        #split the node or edge name
        exchange_nodes_name = [n for n in nodes_name2ex if pname(n) != "EDGE"]
        exchange_edges_name = [n for n in nodes_name2ex if pname(n) == "EDGE"]
        if exchange_nodes_name:
            self.ostream.print_info(f"exchange nodes: {exchange_nodes_name}")
            #check if the exchange_node_data is set
            if self.exchange_node_data is None:
                self.ostream.print_warning("exchange_node_data is not set, skip exchanging nodes")
                #skip exchanging nodes
                exG = G.copy()
            else:
                exG = self._exchange_items_in_G(exchange_nodes_name, G,
                                                self.exchange_node_data,
                                                self.exchange_node_X_data,
                                                self.sc_unit_cell_inv)
        if exchange_edges_name:
            self.ostream.print_info(f"exchange linkers: {exchange_edges_name}")
            #check if the exchange_linker_data is set
            if self.exchange_linker_data is None:
                self.ostream.print_warning("exchange_linker_data is not set, skip exchanging linkers")
                #skip exchanging linkers
                exG = exG.copy()
            else:
                exG = self._exchange_items_in_G(exchange_edges_name, exG,
                                            self.exchange_linker_data,
                                            self.exchange_linker_X_data,
                                            self.sc_unit_cell_inv)
        return exG

    def _find_unsaturated_node(self, eG, node_connectivity):
        # find unsaturated node V in eG
        unsaturated_node = []
        for n in eG.nodes():
            if pname(n) != "EDGE":
                real_neighbor = []
                for cn in eG.neighbors(n):
                    if eG.edges[(n, cn)]["type"] == "real":
                        real_neighbor.append(cn)
                if len(real_neighbor) < node_connectivity:
                    unsaturated_node.append(n)
        return unsaturated_node

    def _find_unsaturated_linker(self, eG, linker_topics):
        # find unsaturated linker in eG
        unsaturated_linker = []
        for n in eG.nodes():
            if pname(n) == "EDGE" and len(list(
                    eG.neighbors(n))) < linker_topics:
                unsaturated_linker.append(n)
        return unsaturated_linker

    def _extract_node_name_from_eG_dict(self, idx_list, eG_dict):
        return [eG_dict[i] for i in idx_list if i in eG_dict]

    def _update_matched_nodes_xind(self, nodes_name_list,
                                   old_matched_vnode_xind):
        # if linked edge is removed and the connected node is not removed, then remove this line from matched_vnode_xind
        # add remove the middle xind of the node to matched_vnode_xind_dict[node] list
        update_matched_vnode_xind = []
        for i in range(len(old_matched_vnode_xind)):
            # format: old_matched_vnode_xind[i]: old_matched_vnode_xind[i]
            if (old_matched_vnode_xind[i][0]
                    in nodes_name_list) or (old_matched_vnode_xind[i][-1]
                                            in nodes_name_list):
                if self._debug:
                    self.ostream.print_info(
                        f"matched_vnode_xind {old_matched_vnode_xind[i]} removed"
                    )
                    continue
            update_matched_vnode_xind.append(old_matched_vnode_xind[i])

        return update_matched_vnode_xind

    def _add_terminations_to_unsaturated_node(self, eG):
        """
        use the node terminations to add terminations to the unsaturated nodes
        """
        #generate term_xoovecs
        term_xoovecs = np.vstack((self.termination_X_data, self.termination_Y_data))[:,5:8].astype(float)
        term_coords = self.termination_data[:,5:8].astype(float)
        term_atoms = self.termination_data[:,0:2]

        termG = eG.copy()
        unsaturated_node = [
            n for n in self.unsaturated_node if n in termG.nodes()
        ]

        (
            self.unsaturated_vnode_xind_dict,
            unsaturated_vnode_xoo_dict,  #will be updated after adding terminations
            self.matched_vnode_xind_dict,
        ) = self._make_unsaturated_vnode_xoo_dict(unsaturated_node,
                                                  self.xoo_dict,
                                                  self.matched_vnode_xind,
                                                  termG, self.sc_unit_cell)
        # term_file: path to the termination file
        # ex_node_cxo_cc: exposed node coordinates

        node_oovecs_record = []
        for n in termG.nodes():
            termG.nodes[n]["term_c_points"] = {}
        for exvnode_xind_key in unsaturated_vnode_xoo_dict.keys():
            exvnode_x_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "x_cpoints"]
            exvnode_oo_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "oo_cpoints"]
            node_xoo_ccoords = np.vstack(
                [exvnode_x_ccoords, exvnode_oo_ccoords])

            # make the beginning point of the termination to the center of the oo atoms
            node_oo_center_cvec = np.mean(exvnode_oo_ccoords[:, 2:5].astype(float), axis=0)
            node_xoo_cvecs = (node_xoo_ccoords[:, 2:5].astype(float) - node_oo_center_cvec)
            node_xoo_cvecs = node_xoo_cvecs.astype("float")
            # use record to record the rotation matrix for get rid of the repeat calculation

            indices = [
                index for index, value in enumerate(node_oovecs_record)
                if is_list_A_in_B(node_xoo_cvecs, value[0])
            ]
            if len(indices) == 1:
                rot = node_oovecs_record[indices[0]][1]
            else:
                _, rot, _ = superimpose(term_xoovecs, node_xoo_cvecs)
                node_oovecs_record.append((node_xoo_cvecs, rot))
            adjusted_term_vecs = np.dot(term_coords,
                                        rot) + node_oo_center_cvec
            adjusted_term = np.hstack((
                term_atoms,
                adjusted_term_vecs,
            ))
            # add the adjusted term to the terms, add index, add the node name
            unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "node_term_c_points"] = (adjusted_term)
            termG.nodes[exvnode_xind_key[0]]["term_c_points"][
                exvnode_xind_key[1]] = (adjusted_term)

        return termG, unsaturated_vnode_xoo_dict

    def _make_unsaturated_vnode_xoo_dict(self, unsaturated_node, xoo_dict,
                                         matched_vnode_xind, eG, sc_unit_cell):
        """
            Build data structures for unsaturated nodes:
              - unsaturated_vnode_xind_dict: for each unsaturated node -> list of exposed X indices
              - unsaturated_vnode_xoo_dict: for each (node, xind) -> metadata including fractional and cartesian points
              - matched_vnode_xind_dict: mapping node -> list of matched x indices
            """
        # Normalize matched_vnode_xind into a dict: node -> [xind, ...]
        matched_vnode_xind = matched_vnode_xind or []
        matched_vnode_xind_dict = {}
        for entry in matched_vnode_xind:
            try:
                n, xind, _ = entry  # entry expected as [node, xind, edge] (edge ignored here)
            except Exception:
                # malformed entry: skip
                continue
            matched_vnode_xind_dict.setdefault(n, []).append(xind)

        # Prepare exposed x-index list for each unsaturated vnode
        xoo_keys = list(xoo_dict.keys())
        unsaturated_vnode_xind_dict = {}
        for vnode in unsaturated_node:
            if vnode not in eG.nodes():
                # node not present in graph (may have been removed) -> treat as no exposed indices
                unsaturated_vnode_xind_dict[vnode] = []
                continue

            if vnode in matched_vnode_xind_dict:
                # exposed indices are those X keys not already matched for this node
                matched_list = matched_vnode_xind_dict[vnode]
                unsaturated_vnode_xind_dict[vnode] = [
                    i for i in xoo_keys if i not in matched_list
                ]
            else:
                # no matched indices -> all X keys are exposed
                # copy to avoid accidental mutation of xoo_keys
                unsaturated_vnode_xind_dict[vnode] = xoo_keys.copy()
                self.ostream.print_warning(
                    f"Node is isolated {vnode} whose index is {eG.nodes[vnode]['index']}"
                )

        # Build the detailed X/OO dictionary for each exposed (node, xind)
        unsaturated_vnode_xoo_dict = {}
        for vnode, exposed_x_indices in unsaturated_vnode_xind_dict.items():
            # skip nodes without exposed indices
            if not exposed_x_indices:
                continue

            fpoints = eG.nodes[vnode].get("f_points")
            if fpoints is None:
                # missing data: skip this node
                continue

            try:
                fpoints_len = len(fpoints)
            except Exception:
                fpoints_len = 0

            for xind in exposed_x_indices:
                # validate x index
                if not (0 <= xind < fpoints_len):
                    # invalid index -> skip
                    continue

                # fractional f_points for this X
                x_fpoints = fpoints[xind]
                # build Cartesian coords for X
                x_cpoints = np.hstack((
                    x_fpoints[0:2],
                    fractional_to_cartesian(x_fpoints[2:5], sc_unit_cell),
                ))

                # get associated OO indices (may be missing -> empty list)
                oo_ind_in_vnode = xoo_dict.get(xind, [])
                if oo_ind_in_vnode:
                    # gather fractional arrays for those OO atoms and compute Cartesian coords in one call
                    try:
                        oo_fpoints_in_vnode = np.vstack(
                            [fpoints[i] for i in oo_ind_in_vnode])
                    except Exception:
                        # if any index invalid, skip this xind
                        continue
                    oo_cpoints = np.hstack((
                        oo_fpoints_in_vnode[:, 0:2],
                        fractional_to_cartesian(oo_fpoints_in_vnode[:, 2:5],
                                                sc_unit_cell),
                    ))
                else:
                    # no OO partners found
                    oo_fpoints_in_vnode = np.empty((0, 0))
                    oo_cpoints = np.empty((0, 3))

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

    def _remove_xoo_from_node(self, G):
        """
        remove the XOO atoms from the node after adding the terminations, add ['noxoo_f_points'] to the node in eG
        """

        xoo_dict = self.xoo_dict
        eG = G.copy()

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

        return eG

    def _exchange_items_in_G(self,
                             edge_n_list,
                             G,
                             new_n_data,
                             new_n_X_data,
                             sc_unit_cell_inv,
                             newname=None):
        new_n_atoms = new_n_data[:, 0:2]
        new_n_ccoords = new_n_data[:, 5:8].astype(float)
        new_n_x_ccoords = new_n_X_data[:, 5:8].astype(float)

        for n in edge_n_list:
            if n not in G.nodes():
                self.ostream.print_warning(
                    f"this linker is not in MOF, will be skipped {n}")
                continue
            n_f_points = G.nodes[n]["f_points"]
            x_indices = [
                i for i in range(len(n_f_points))
                if nn(n_f_points[i][0]) == "X"
            ]
            #if x atoms in new linker is not equal to the x atoms in the processed edge print warning and skip
            if len(x_indices) != len(new_n_x_ccoords):
                self.ostream.print_warning(
                    f"the number of X atoms in the new fragment does not match the origin connectivity of {n}, skipping"
                )
                continue
            old_n_x_points = n_f_points[x_indices]
            old_n_com = np.mean(old_n_x_points[:, 2:5].astype(float), axis=0)
            old_n_x_fcoords = old_n_x_points[:, 2:5].astype(float) - old_n_com

            new_n_x_fcoords = cartesian_to_fractional(new_n_x_ccoords,
                                                      sc_unit_cell_inv)
            new_n_fcoords = cartesian_to_fractional(new_n_ccoords,
                                                    sc_unit_cell_inv)

            _, rot, trans = superimpose(new_n_x_fcoords, old_n_x_fcoords)
            replaced_linker_fcoords = np.dot(new_n_fcoords, rot) + old_n_com
            replaced_linker_f_points = np.hstack(
                (new_n_atoms, replaced_linker_fcoords))

            G.nodes[n]["f_points"] = replaced_linker_f_points
            G.nodes[n]["name"] = newname if newname else "R" + G.nodes[n]["name"]
            self.ostream.print_info(f"linker {n} exchanged")

        return G
