import sys
from pathlib import Path

import numpy as np
import networkx as nx
import mpi4py.MPI as MPI
import h5py
import re

try:
    from scipy.optimize import minimize
except ImportError:
    pass

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.molecule import Molecule

from ..io.basic import nn, nl,pname
from ..utils.geometry import (
    unit_cell_to_cartesian_matrix, fractional_to_cartesian, cartesian_to_fractional,
    locate_min_idx, reorthogonalize_matrix, find_optimal_pairings, find_edge_pairings
)
from ..utils.superimpose import superimpose_rotation_only

class OptimizationDriver:
    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)

        self.sorted_nodes = None
        self.sorted_edges = None
        self.pname_set_dict = None

        self.initial_rotations = None
        self.initial_set_rotations = None
        self.optimized_rotations = None
        self.optimized_set_rotations = None
        self.static_atom_positions = None

        # Optimization parameters
        self.opt_method = 'L-BFGS-B'
        self.maxfun = 15000
        self.maxiter = 15000
        self.display = True
        self.eps = 1e-8
        self.iprint = -1

        self._debug = False

    def _expand_set_rots(self, pname_set_dict, set_rotations, sorted_nodes):
        """
        Expand set rotations to all nodes based on the set dictionary.
        """
        set_rotations = set_rotations.reshape(len(pname_set_dict), 3, 3)
        rotations = np.empty((len(sorted_nodes), 3, 3))
        idx = 0
        for name in pname_set_dict:
            for k in pname_set_dict[name]["ind_ofsortednodes"]:
                rotations[k] = set_rotations[idx]
            idx += 1
        return rotations

    def _objective_function_pre(self, params, G, static_atom_positions):
        """
        Objective function to minimize distances between paired node to paired node_com along edges.

        Parameters:
            params (numpy.ndarray): Flattened array of rotation matrices.
            G (networkx.Graph): Graph structure.
            atom_positions (dict): Original positions of X atoms for each node.


        Returns:
            float: Total distance metric to minimize.
        """
        # num_nodes = len(G.nodes())
        sorted_nodes = self.sorted_nodes
        sorted_edges = self.sorted_edges
        pname_set_dict = self.pname_set_dict

        set_rotation_matrices = params.reshape(len(pname_set_dict), 3, 3)
        rotation_matrices = self._expand_set_rots(pname_set_dict, set_rotation_matrices, sorted_nodes)
        total_distance = 0.0

        for i, j in sorted_edges:
            R_i = reorthogonalize_matrix(rotation_matrices[i])

            com_i = G.nodes[sorted_nodes[i]]["ccoords"]
            com_j = G.nodes[sorted_nodes[j]]["ccoords"]
            # Rotate positions around their mass center
            rotated_i_positions = (
                np.dot(static_atom_positions[i][:, 1:] - com_i, R_i.T) + com_i)

            dist_matrix = np.empty((len(rotated_i_positions), 1))
            for idx_i in range(len(rotated_i_positions)):
                dist = np.linalg.norm(rotated_i_positions[idx_i] - com_j)
                dist_matrix[idx_i, 0] = dist
                # total_distance += dist ** 2
            if np.argmin(dist_matrix) > 1:
                total_distance += 1e4  # penalty for the distance difference
            else:
                total_distance += np.min(dist_matrix)**2
            #
            for idx_i in range(len(rotated_i_positions)):
                # second min and min distance difference not max
                if len(dist_matrix[idx_i, :]) > 1:
                    second_min_dist = np.partition(dist_matrix[idx_i, :], 1)[1]
                else:
                    second_min_dist = np.partition(dist_matrix[idx_i, :], 0)[0]
                diff = second_min_dist - np.min(dist_matrix[idx_i, :])

                if diff < 4:
                    total_distance += 1e4

            total_distance += 1e3 / (np.max(dist_matrix) - np.min(dist_matrix)
                                    )  # reward for the distance difference

        return total_distance


    def _objective_function_after(self, params, G, static_atom_positions):
        """
        Objective function to minimize distances between paired atoms along edges. just use minimum distance

        Parameters:
            params (numpy.ndarray): Flattened array of rotation matrices.
            G (networkx.Graph): Graph structure.
            atom_positions (dict): Original positions of X atoms for each node.
            edge_pairings (dict): Precomputed pairings for each edge.

        Returns:
            float: Total distance metric to minimize.
        """
        # num_nodes = len(G.nodes())
        set_rotation_matrices = params.reshape(len(pname_set_dict), 3, 3)
        rotation_matrices = self._expand_set_rots(pname_set_dict, set_rotation_matrices,
                                        sorted_nodes)
        total_distance = 0.0

        for i, j in sorted_edges:
            R_i = reorthogonalize_matrix(rotation_matrices[i])
            R_j = reorthogonalize_matrix(rotation_matrices[j])

            com_i = G.nodes[sorted_nodes[i]]["ccoords"]
            com_j = G.nodes[sorted_nodes[j]]["ccoords"]

            # Rotate positions around their mass center
            rotated_i_positions = (
                np.dot(static_atom_positions[i][:, 1:] - com_i, R_i.T) + com_i)
            rotated_j_positions = (
                np.dot(static_atom_positions[j][:, 1:] - com_j, R_j.T) + com_j)

            dist_matrix = np.empty(
                (len(rotated_i_positions), len(rotated_j_positions)))
            for idx_i in range(len(rotated_i_positions)):
                for idx_j in range(len(rotated_j_positions)):
                    dist = np.linalg.norm(rotated_i_positions[idx_i] -
                                        rotated_j_positions[idx_j])
                    dist_matrix[idx_i, idx_j] = dist

            if np.argmin(dist_matrix) > 1:
                total_distance += 1e4  # penalty for the distance difference

            total_distance += np.min(dist_matrix)**2

        return total_distance



    def _optimize_rotations_pre(self,num_nodes, G, atom_positions, initial_set_rotations):
        """
        Optimize rotations for all nodes in the graph.

        Parameters:
            G (networkx.Graph): Graph structure with edges between nodes.
            atom_positions (dict): Positions of X atoms for each node.

        Returns:
            list: Optimized rotation matrices for all nodes.
        """

        assert_msg_critical("scipy" in sys.modules,
                            "scipy is required for optimize_rotations_pre.")


        self.ostream.print_title(f"Rotation Optimization (pre step)")
        self.ostream.print_info(f"Rotations optimization information:")
        self.ostream.print_info(f"opt_method:, {self.opt_method}")
        self.ostream.print_info(f"maxfun:, {self.maxfun}")
        self.ostream.print_info(f"maxiter:, {self.maxiter}")
        self.ostream.print_info(f"display:, {self.display}")
        self.ostream.print_info(f"eps:, {self.eps}")
        self.ostream.print_info(f"iprint:, {self.iprint}")
        self.ostream.print_info(f"Number of nodes to optimize:, {num_nodes}")
        self.ostream.print_info("\n")
        self.ostream.flush()

        
        # initial_rotations = np.tile(np.eye(3), (num_nodes, 1)).flatten()
        # get a better initial guess, use random rotation matrix combination
        # initial_rotations  = np.array([reorthogonalize_matrix(np.random.rand(3,3)) for i in range(num_nodes)]).flatten()
        static_atom_positions = atom_positions.copy()
        # Precompute edge-specific pairings
        # edge_pairings = find_edge_pairings(sorted_edges, atom_positions).


        result = minimize(
            self._objective_function_pre,
            initial_set_rotations.flatten(),
            args=(G, static_atom_positions),
            method=self.opt_method,
            options={
                "maxfun": self.maxfun,
                "maxiter": self.maxiter,
                "disp": self.display,
                "eps": self.eps,
                "iprint": self.iprint,
                "maxls": 50,
            },
        )

        optimized_rotations = result.x

        return optimized_rotations, static_atom_positions


    def _optimize_rotations_after(self, num_nodes, G, atom_positions, initial_rotations):
        """
        Optimize rotations for all nodes in the graph.

        Parameters:
            G (networkx.Graph): Graph structure with edges between nodes.
            atom_positions (dict): Positions of X atoms for each node.

        Returns:
            list: Optimized rotation matrices for all nodes.
        """

        assert_msg_critical("scipy" in sys.modules,
                            "scipy is required for optimize_rotations_after.")

        self.ostream.print_title(f"Rotation Optimization (after step)")
        self.ostream.print_info(f"Rotations optimization information:")
        self.ostream.print_info(f"opt_method:, {self.opt_method}")
        self.ostream.print_info(f"maxfun:, {self.maxfun}")
        self.ostream.print_info(f"maxiter:, {self.maxiter}")
        self.ostream.print_info(f"display:, {self.display}")
        self.ostream.print_info(f"eps:, {self.eps}")
        self.ostream.print_info(f"iprint:, {self.iprint}")
        self.ostream.print_info(f"Number of nodes to optimize:, {num_nodes}")
        self.ostream.print_info("\n")
        self.ostream.flush()

        # get a better initial guess, use random rotation matrix combination
        # initial_rotations  = np.array([reorthogonalize_matrix(np.random.rand(3,3)) for i in range(num_nodes)]).flatten()
        static_atom_positions = atom_positions.copy()
        # Precompute edge-specific pairings
        # edge_pairings = find_edge_pairings(sorted_edges, atom_positions)

        result = minimize(
            self._objective_function_after,
            initial_rotations.flatten(),
            args=(G, static_atom_positions),
            method=self.opt_method,
            options={
                "maxfun": self.maxfun,
                "maxiter": self.maxiter,
                "disp": self.display,
                "eps": self.eps,
                "iprint": self.iprint,
            },
        )

        optimized_rotations = result.x.reshape(-1, 3, 3)
        optimized_rotations = [
            reorthogonalize_matrix(R) for R in optimized_rotations
        ]
        optimized_rotations = np.array(optimized_rotations)

        ## # Print the optimized pairings after optimization
        ## print("Optimized Pairings (after optimization):")
        ## for (i, j), pairs in edge_pairings.items():
        ##     print(f"Node {i} and Node {j}:")
        ##     for idx_i, idx_j in pairs:
        ##         print(f"  node{i}_{idx_i} -- node{j}_{idx_j}")
        ## print()

        return optimized_rotations, static_atom_positions


    def _apply_rotations_to_atom_positions(self, optimized_rotations, G, atom_positions):
        """
        Apply the optimized rotation matrices to the atom positions.

        Parameters:
            optimized_rotations (list): Optimized rotation matrices for each node.
            G (networkx.Graph): Graph structure.
            atom_positions (dict): Original positions of X atoms for each node.

        Returns:
            dict: Rotated positions for each node.
        """
        rotated_positions = {}

        for i, node in enumerate(self.sorted_nodes):
            # if node type is V
            # if 'DV' in G.nodes[node]['type']:
            # continue
            R = optimized_rotations[i]

            original_positions = atom_positions[i]

            com = G.nodes[node]["ccoords"]

            # Translate, rotate, and translate back to preserve the mass center
            translated_positions = original_positions - com
            rotated_translated_positions = np.dot(translated_positions, R.T)
            rotated_positions[node] = rotated_translated_positions + com

        return rotated_positions

    def _scale_objective_function(self, params, old_cell_params, old_cartesian_coords,
                             new_cartesian_coords):
        a_new, b_new, c_new, _, _, _ = params
        a_old, b_old, c_old, alpha_old, beta_old, gamma_old = old_cell_params

        # Compute transformation matrix for the old unit cell, T is the unit cell matrix
        T_old = unit_cell_to_cartesian_matrix(a_old, b_old, c_old, alpha_old,
                                            beta_old, gamma_old)
        T_old_inv = np.linalg.inv(T_old)
        old_fractional_coords = cartesian_to_fractional(old_cartesian_coords,
                                                        T_old_inv)

        # backup
        # old_fractional_coords = cartesian_to_fractional(old_cartesian_coords,T_old_inv)

        # Compute transformation matrix for the new unit cell
        T_new = unit_cell_to_cartesian_matrix(a_new, b_new, c_new, alpha_old,
                                            beta_old, gamma_old)
        T_new_inv = np.linalg.inv(T_new)

        # Convert the new Cartesian coordinates to fractional coordinate using the old unit cell

        # Recalculate fractional coordinates from updated Cartesian coordinates
        new_fractional_coords = cartesian_to_fractional(new_cartesian_coords,
                                                        T_new_inv)

        # Compute difference from original fractional coordinates
        diff = new_fractional_coords - old_fractional_coords
        return np.sum(diff**2)  # Sum of squared differences


 
    def _optimize_cell_parameters(self, cell_info, original_ccoords, updated_ccoords):

        assert_msg_critical("scipy" in sys.modules,
                            "scipy is required for optimize_cell_parameters.")

        # Old cell parameters (example values)
        old_cell_params = cell_info  # [a, b, c, alpha, beta, gamma]

        # Old Cartesian coordinates of points (example values)
        old_cartesian_coords = np.vstack(list(
            original_ccoords.values()))  # original_ccoords

        # New Cartesian coordinates of the same points (example values)
        new_cartesian_coords = np.vstack(list(
            updated_ccoords.values()))  # updated_ccoords
        # Initial guess for new unit cell parameters (e.g., slightly modified cell)
        initial_params = cell_info

        # Bounds: a, b, c > 3; angles [0, 180]
        bounds = [(3, None), (3, None), (3, None)] + [(20, 180)] * 3

        # Optimize using L-BFGS-B to minimize the objective function
        result = minimize(
            self._scale_objective_function,
            x0=initial_params,
            args=(old_cell_params, old_cartesian_coords, new_cartesian_coords),
            method="L-BFGS-B",
            bounds=bounds,
        )

        # Extract optimized parameters
        optimized_params = np.round(result.x, 5)
        self.ostream.print_info(f"Optimized New Cell Parameters: {optimized_params}\nTemplate Cell Parameters: {cell_info}")

        return optimized_params

    def _get_edge_lengths(self, G):
        edge_lengths = {}
        lengths = []
        for e in G.edges():
            i, j = e
            length = np.linalg.norm(G.nodes[i]["ccoords"] - G.nodes[j]["ccoords"])
            length = np.round(length, 3)
            edge_lengths[(i, j)] = length
            edge_lengths[(j, i)] = length
            lengths.append(length)
        # print('edge lengths:',set(lengths)) #debug
        if len(set(lengths)) != 1:
            print("more than one type of edge length")
            # if the length are close, which can be shown by std
            if np.std(lengths) < 0.1:
                print("the edge lengths are close")
            else:
                print("the edge lengths are not close")
            print(set(lengths))
        return edge_lengths, set(lengths)

    def _apply_rotations_to_Xatoms_positions(
        self,
        optimized_rotations,
        G,
        sorted_nodes,
        sorted_edges_of_sortednodeidx,
        Xatoms_positions_dict,
    ):
        """
        Apply the optimized rotation matrices to the atom positions.

        Parameters:
            optimized_rotations (list): Optimized rotation matrices for each node.
            G (networkx.Graph): Graph structure.
            atom_positions (dict): Original positions of X atoms for each node.

        Returns:
            dict: Rotated positions for each node.
        """
        rotated_positions = Xatoms_positions_dict.copy()

        for i, node in enumerate(sorted_nodes):
            # if node type is V
            # if 'DV' in G.nodes[node]['type']:
            # continue
            R = optimized_rotations[i]

            original_positions = rotated_positions[i][:, 1:]
            com = G.nodes[node]["ccoords"]

            # Translate, rotate, and translate back to preserve the mass center
            translated_positions = original_positions - com
            rotated_translated_positions = np.dot(translated_positions, R.T)
            rotated_positions[i][:, 1:] = rotated_translated_positions + com
        edge_pair = find_edge_pairings(sorted_nodes, sorted_edges_of_sortednodeidx,
                                    rotated_positions)
        # print("Optimized Pairings (after optimization):") #DEBUG

        optimized_pair = {}

        for (i, j), pair in edge_pair.items():
            # print(f"Node {sorted_nodes[i]} and Node {sorted_nodes[j]}:") #DEBUG
            idx_i, idx_j = pair
            # print(f"  node{sorted_nodes[i]}_{int(idx_i)} -- node{sorted_nodes[j]}_{int(idx_j)}") #DEBUG
            optimized_pair[sorted_nodes[i],
                        sorted_nodes[j]] = (int(idx_i), int(idx_j))

        return rotated_positions, optimized_pair


    # use optimized_params to update all of nodes ccoords in G, according to the fccoords
    def _update_ccoords_by_optimized_cell_params(self, G, optimized_params):
        sG = G.copy()
        a, b, c, alpha, beta, gamma = optimized_params
        T_unitcell = unit_cell_to_cartesian_matrix(a, b, c, alpha, beta, gamma)
        updated_ccoords = {}
        for n in sG.nodes():
            updated_ccoords[n] = fractional_to_cartesian(T_unitcell,
                                                        sG.nodes[n]["fcoords"].T).T
            sG.nodes[n]["ccoords"] = updated_ccoords[n]
        return sG, updated_ccoords


    def optimize(self):  # TODO: modified for mil53
        """
        two optimization steps:
        1. optimize the node rotation
        2. optimize the cell parameters to fit the target MOF cell
        """
            # Add row indices as the first column
        def addidx(array):
            row_indices = np.arange(array.shape[0]).reshape(-1, 1).astype(int)
            new_array = np.hstack((row_indices, array))
            return new_array

        if self.ec_x_ccoords is not None:
            ec_x_ccoords = self.ec_x_ccoords
        if self.ec_ccoords is not None:
            ecoords = self.ec_ccoords

        G = self.G
        node_xcoords = self.node_x_ccoords
        node_coords = self.node_ccoords
        linker_length = self.linker_length
        constant_length = self.constant_length

        x_com_length = np.mean([np.linalg.norm(i) for i in node_xcoords])
        sorted_nodes = self.sorted_nodes
        sorted_edges = self.sorted_edges


        nodes_atoms = {}
        for n in sorted_nodes:
            if "CV" in n:
                nodes_atoms[n] = self.ec_atom
            else:
                nodes_atoms[n] = self.node_atom

        Xatoms_positions_dict = {}
        node_positions_dict = {}
        # reindex the nodes in the Xatoms_positions with the index in the sorted_nodes, like G has 16 nodes[2,5,7], but the new dictionary should be [0,1,2]
        for n in sorted_nodes:
            if "CV" in n:
                Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    G.nodes[n]["ccoords"] + ec_x_ccoords)
            else:
                Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    G.nodes[n]["ccoords"] + node_xcoords)

        for n in sorted_nodes:
            if "CV" in n:
                node_positions_dict[sorted_nodes.index(n)] = (
                    G.nodes[n]["ccoords"] + ecoords)
            else:
                node_positions_dict[sorted_nodes.index(n)] = (
                    G.nodes[n]["ccoords"] + node_coords)

        # reindex the edges in the G with the index in the sorted_nodes
        sorted_edges_of_sortednodeidx = [(sorted_nodes.index(e[0]),
                                          sorted_nodes.index(e[1]))
                                         for e in sorted_edges]

        # Optimize rotations
        num_nodes = G.number_of_nodes()
        pname_list = [pname(n) for n in sorted_nodes]
        pname_set = set(pname_list)
        pname_set_dict = {}
        for node_pname in pname_set:
            pname_set_dict[node_pname] = {
                "ind_ofsortednodes": [],
            }
        for i, node in enumerate(sorted_nodes):
            pname_set_dict[pname(node)]["ind_ofsortednodes"].append(i)
            if len(pname_set_dict[pname(node)]
                   ["ind_ofsortednodes"]) == 1:  # first node
                pname_set_dict[pname(node)]["rot_trans"] = get_rot_trans_matrix(
                    node, G, sorted_nodes,
                    Xatoms_positions_dict)  # initial guess
        self.pname_set_dict = pname_set_dict

        for p_name in pname_set_dict:
            rot, trans = pname_set_dict[p_name]["rot_trans"]
            for k in pname_set_dict[p_name]["ind_ofsortednodes"]:
                node = sorted_nodes[k]

                Xatoms_positions_dict[k][:, 1:] = (np.dot(
                    Xatoms_positions_dict[k][:, 1:] - G.nodes[node]["ccoords"],
                    rot,
                ) + trans + G.nodes[node]["ccoords"])
                node_positions_dict[k] = (np.dot(
                    node_positions_dict[k] - G.nodes[node]["ccoords"], rot) +
                                          trans + G.nodes[node]["ccoords"])
        ###3D free rotation
        if not hasattr(self, "saved_optimized_rotations"):
            print("-" * 80)
            print(" " * 20, "start to optimize the rotations", " " * 20)
            print("-" * 80)

            initial_guess_set_rotations = (np.eye(3, 3).reshape(1, 3, 3).repeat(
                len(pname_set), axis=0))

            ####TODO: modified for mil53
            (
                optimized_rotations_pre,
                _,
            ) = self._optimize_rotations_pre(
                num_nodes,
                G,
                Xatoms_positions_dict,
                initial_guess_set_rotations,
            )

            (
                optimized_set_rotations,
                _,
            ) = self._optimize_rotations_after(
                num_nodes,
                G,
                Xatoms_positions_dict,
                # initial_guess_set_rotations,  # TODO: modified for mil53
                optimized_rotations_pre,
            )
            print("-" * 80)
            print(" " * 20, "rotations optimization completed", " " * 20)
            print("-" * 80)
            # to save the optimized rotations as npy
            if hasattr(self, "to_save_optimized_rotations_filename"):
                np.save(
                    self.to_save_optimized_rotations_filename + ".npy",
                    optimized_set_rotations,
                )
                print(
                    "optimized rotations are saved to: ",
                    self.to_save_optimized_rotations_filename + ".npy",
                )

        else:
            if hasattr(self, "use_saved_rotations_as_initial_guess"):
                if self.use_saved_rotations_as_initial_guess:
                    print("use the saved optimized_rotations as initial guess")
                    print("-" * 80)
                    print(" " * 20, "start to optimize the rotations", " " * 20)
                    print("-" * 80)

                    saved_set_rotations = self.saved_optimized_rotations.reshape(
                        -1, 3, 3)

                    (
                        optimized_set_rotations,
                        _,
                    ) = self._optimize_rotations_after(
                        num_nodes,
                        G,
                        Xatoms_positions_dict,
                        saved_set_rotations,
                    )
                    print("-" * 80)
                    print(" " * 20, "rotations optimization completed",
                          " " * 20)
                    print("-" * 80)
                    # to save the optimized rotations as npy
                    if hasattr(self, "to_save_optimized_rotations_filename"):
                        np.save(
                            self.to_save_optimized_rotations_filename + ".npy",
                            optimized_set_rotations,
                        )
                        print(
                            "optimized rotations are saved to: ",
                            self.to_save_optimized_rotations_filename + ".npy",
                        )

                else:
                    optimized_set_rotations = self.saved_optimized_rotations.reshape(
                        -1, 3, 3)

            else:
                print(
                    "use the loaded optimized_rotations from the previous optimization"
                )
                optimized_set_rotations = self.saved_optimized_rotations.reshape(
                    -1, 3, 3)

        optimized_rotations = self._expand_setrots(pname_set_dict,
                                             optimized_set_rotations,
                                             sorted_nodes)
        # Apply rotations
        rotated_node_positions = self._apply_rotations_to_atom_positions(
            optimized_rotations, G, node_positions_dict)

        # Save results to XYZ
        # save_xyz("optimized_nodesstructure.xyz", rotated_node_positions) #DEBUG

        rotated_Xatoms_positions_dict, optimized_pair = (
            self._apply_rotations_to_Xatoms_positions(
                optimized_rotations,
                G,
                Xatoms_positions_dict,
            ))

        start_node = sorted_edges[0][
            0]  # find_nearest_node_to_beginning_point(G)
        # loop all of the edges in G and get the lengths of the edges, length is the distance between the two nodes ccoords
        edge_lengths, lengths = self._get_edge_lengths(G)

        x_com_length = np.mean([np.linalg.norm(i) for i in node_xcoords])
        new_edge_length = linker_length + 2 * constant_length + 2 * x_com_length
        # update the node ccoords in G by loop edge, start from the start_node, and then update the connected node ccoords by the edge length, and update the next node ccords from the updated node

        updated_ccoords, original_ccoords = self._update_node_ccoords(
            G, edge_lengths, start_node, new_edge_length)
        # exclude the start_node in updated_ccoords and original_ccoords
        updated_ccoords = {
            k: v for k, v in updated_ccoords.items() if k != start_node
        }
        original_ccoords = {
            k: v for k, v in original_ccoords.items() if k != start_node
        }

        # use optimized_params to update all of nodes ccoords in G, according to the fccoords
        if not hasattr(self, "optimized_params"):
            print("-" * 80)
            print(" " * 20, "start to optimize the cell parameters", " " * 20)
            print("-" * 80)
            optimized_params = self._optimize_cell_parameters(self.cell_info,
                                                        original_ccoords,
                                                        updated_ccoords)
            print("-" * 80)
            print(" " * 20, "cell parameters optimization completed", " " * 20)
            print("-" * 80)
        else:
            print("use the optimized_params from the previous optimization")
            optimized_params = self.optimized_params

        sc_unit_cell = unit_cell_to_cartesian_matrix(
            optimized_params[0],
            optimized_params[1],
            optimized_params[2],
            optimized_params[3],
            optimized_params[4],
            optimized_params[5],
        )
        sc_unit_cell_inv = np.linalg.inv(sc_unit_cell)
        sG, scaled_ccoords = self._update_ccoords_by_optimized_cell_params(
            self.G, optimized_params)
        scaled_node_positions_dict = {}
        scaled_Xatoms_positions_dict = {}

        for n in sorted_nodes:
            if "CV" in n:
                scaled_Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    sG.nodes[n]["ccoords"] + ec_x_ccoords)
            else:
                scaled_Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    sG.nodes[n]["ccoords"] + node_xcoords)

        for n in sorted_nodes:
            if "CV" in n:
                scaled_node_positions_dict[sorted_nodes.index(n)] = (
                    sG.nodes[n]["ccoords"] + ecoords)
            else:
                scaled_node_positions_dict[sorted_nodes.index(n)] = (
                    sG.nodes[n]["ccoords"] + node_coords)

        # Apply rotations
        for p_name in pname_set_dict:
            rot, trans = pname_set_dict[p_name]["rot_trans"]
            for k in pname_set_dict[p_name]["ind_ofsortednodes"]:
                node = sorted_nodes[k]
                scaled_Xatoms_positions_dict[k][:, 1:] = (np.dot(
                    scaled_Xatoms_positions_dict[k][:, 1:] -
                    sG.nodes[node]["ccoords"],
                    rot,
                ) + trans + sG.nodes[node]["ccoords"])

                scaled_node_positions_dict[k] = (np.dot(
                    scaled_node_positions_dict[k] - sG.nodes[node]["ccoords"],
                    rot) + trans + sG.nodes[node]["ccoords"])

        scaled_rotated_node_positions = self._apply_rotations_to_atom_positions(
            optimized_rotations, sG, scaled_node_positions_dict)
        scaled_rotated_Xatoms_positions, optimized_pair = (
            self._apply_rotations_to_Xatoms_positions(
                optimized_rotations,
                sG,
                scaled_Xatoms_positions_dict,
            ))
        # Save results to XYZ

        self.sorted_nodes = sorted_nodes
        self.sorted_edges = sorted_edges
        self.sorted_edges_of_sortednodeidx = sorted_edges_of_sortednodeidx
        self.optimized_rotations = optimized_rotations
        self.optimized_params = optimized_params
        self.new_edge_length = new_edge_length
        self.optimized_pair = optimized_pair
        self.scaled_rotated_node_positions = scaled_rotated_node_positions
        self.scaled_rotated_Xatoms_positions = scaled_rotated_Xatoms_positions
        self.sc_unit_cell = sc_unit_cell
        self.sc_unit_cell_inv = sc_unit_cell_inv
        self.sG_node = sG
        self.nodes_atom = nodes_atoms
        self.rotated_node_positions = rotated_node_positions
        self.Xatoms_positions_dict = Xatoms_positions_dict
        self.node_positions_dict = node_positions_dict



def fetch_X_atoms_ind_array(array, column, X):
    # array: input array
    # column: column index to check for label
    # X: label to search for

    ind = [
        k for k in range(len(array)) if re.sub(r"\d", "", array[k, column]) == X
    ]
    x_array = array[ind]
    return ind, x_array


def recenter_and_norm_vectors(vectors, extra_mass_center=None):
    vectors = np.array(vectors)
    if extra_mass_center is not None:
        mass_center = extra_mass_center
    else:
        mass_center = np.mean(vectors, axis=0)
    vectors = vectors - mass_center
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return vectors, mass_center


def get_connected_nodes_vectors(node, G):
    # use adjacent nodes to get vectors
    vectors = []
    for i in list(G.neighbors(node)):
        vectors.append(G.nodes[i]["ccoords"])
    return vectors, G.nodes[node]["ccoords"]


def get_rot_trans_matrix(node, G, sorted_nodes, Xatoms_positions_dict):
    node_id = sorted_nodes.index(node)
    node_xvecs = Xatoms_positions_dict[node_id][:, 1:]
    vecsA, _ = recenter_and_norm_vectors(node_xvecs, extra_mass_center=None)
    v2, node_center = get_connected_nodes_vectors(node, G)
    vecsB, _ = recenter_and_norm_vectors(v2, extra_mass_center=node_center)
    _, rot, tran = superimpose_rotation_only(vecsA, vecsB)
    return rot, tran




if __name__ == "__main__":
    #make a simple test
    node_a_positions = np.array([[ "X1", 0.0, 1.0, 0.0],
                                 [ "X2", 1.0, 0.0, 0.0],
                                 [ "X3", 0.0, -1.0, 0.0]])
    node_b_positions = np.array([[ "X1", 1.0, 0.0, 0.0],
                                 [ "X2", 0.0, -1.0, 0.0],
                                 [ "X3", -1.0, 0.0, 0.0]])
    atom_positions = {0: node_a_positions[:,1:].astype(float),
                      1: node_b_positions[:,1:].astype(float)}
    print("Original Positions:")
    print(atom_positions)
    print(node_a_positions[:,1:].astype(float).shape)
    G = nx.Graph()
    G.add_node(0, ccoords=np.array([[0.0, 0.0, 0.0]]))  # make it a 3x1 array
    G.add_node(1, ccoords=np.array([[0.0, 0.0, 0.0]]))  # make it a 3x1 array
    G.add_edge(0,1)
    sorted_nodes = [0,1]
    sorted_edges = [(0,1)]
    pname_set_dict = {'A': {'ind_ofsortednodes': [0]},
                      'B': {'ind_ofsortednodes': [1]}}
    initial_set_rotations = np.array([[[1,0,0],[0,1,0],[0,0,1]],
                                      [[1,0,0],[0,1,0],[0,0,1]]]).reshape(2,3,3)
    optimizer = OptimizationDriver()
    optimizer.sorted_nodes = sorted_nodes
    optimizer.sorted_edges = sorted_edges
    optimizer.pname_set_dict = pname_set_dict
    optimized_set_rotations, static_atom_positions = optimizer.optimize_rotations_pre(
        num_nodes=2,
        G=G,
        atom_positions=atom_positions,
        initial_set_rotations=initial_set_rotations,)
    optimized_rotations = optimizer._expand_set_rots(pname_set_dict, optimized_set_rotations, sorted_nodes)
    print("Optimized Rotations (pre):")
    print(optimized_rotations)
    rotated_positions = optimizer._apply_rotations_to_atom_positions(optimized_rotations, G, atom_positions)
    print("Rotated Positions (pre):")
    print(rotated_positions)        

    #test scale optimization
    cell_info = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    original_ccoords = {0: np.array([[0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, -1.0, 0.0]]),
                        1: np.array([[1.0, 0.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                    [-1.0, 0.0, 0.0]])}         
    updated_ccoords = {0: np.array([[0.0, 2.0, 0.0],
                                   [2.0, 0.0, 0.0],
                                   [0.0, -2.0, 0.0]]),
                       1: np.array([[2.0, 0.0, 0.0],
                                   [0.0, -2.0, 0.0],        
                                   [-2.0, 0.0, 0.0]])}         
    optimized_params = optimizer.optimize_cell_parameters(cell_info, original_ccoords, updated_ccoords)
    print("Optimized Cell Parameters:")
    print(optimized_params)

