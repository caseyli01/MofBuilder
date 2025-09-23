import numpy as np
import networkx as nx
from ..io.cif_reader import CifReader
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys

#creat Net object
#Net.G is the graph of the net
#Net.cell_info is the cell parameters of the net
#Net.unit_cell is the unit cell matrix of the net
#Net.unit_cell_inv is the inverse of the unit cell matrix of the net


class FrameNet:
    '''creates the net graph from cif file'''

    def __init__(self, comm=None, ostream=None, filepath=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        if ostream is None:
            if comm.Get_rank() == mpi_master():
                ostream = OutputStream(sys.stdout)
            else:
                ostream = OutputStream(None)

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        self.G = nx.Graph()
        
        self.cifreader = CifReader()
        self.cif_file = None
        self.edge_length_range = []

        #will be set by create_net
        self.vvnode333 = None
        self.ecnode333 = None
        self.eenode333 = None
        self.edge_length_range = []
        self.cell_info = None
        self.unit_cell = None
        self.unit_cell_inv = None
        self.pair_vertex_edge = None
        self.max_degree = None
        self.sorted_nodes = None
        self.sorted_edges = None
        self.linker_topic = None  

        #debug
        self._debug = False


# Utility functions for node naming and coordinate extraction

# Supercell and unit cell utilities

    def _make_supercell_3x3x3(self, array_xyz):
        """
        Generates a 3x3x3 supercell by shifting the input coordinates.
        array_xyz: (N, 3) or (3,) numpy array
        Returns: (N*27, 3) numpy array
        """
        array_xyz = np.atleast_2d(array_xyz)
        shifts = np.array([-1, 0, 1])
        mesh = np.stack(np.meshgrid(shifts, shifts, shifts), -1).reshape(-1, 3)
        supercell = (array_xyz[:, None, :] + mesh).reshape(-1, 3)
        return supercell

    # Unit cell extraction and coordinate conversion
    def _extract_unit_cell(self, cell_info):
        """
        Extracts the unit cell matrix from cell parameters.
        cell_info: [a, b, c, alpha, beta, gamma]
        Returns: (3, 3) numpy array
        """
        aL, bL, cL, alpha, beta, gamma = map(float, cell_info)
        ax, ay, az = aL, 0.0, 0.0
        bx = bL * np.cos(np.deg2rad(gamma))
        by = bL * np.sin(np.deg2rad(gamma))
        bz = 0.0
        cx = cL * np.cos(np.deg2rad(beta))
        cy = (cL * bL * np.cos(np.deg2rad(alpha)) -
              bx * cx) / by if by != 0 else 0.0
        cz = np.sqrt(max(cL**2.0 - cx**2.0 - cy**2.0, 0.0))
        unit_cell = np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T
        return unit_cell

    def _c2f_coords(self, coords, unit_cell):
        """
        Converts cartesian coordinates to fractional coordinates.
        """
        unit_cell_inv = np.linalg.inv(unit_cell)
        return np.dot(coords, unit_cell_inv.T)

    def _f2c_coords(self, fcoords, unit_cell):
        """
        Converts fractional coordinates to cartesian coordinates.
        """
        return np.dot(fcoords, unit_cell.T)

    def _check_inside_unit_cell(self, point):
        """
        Checks if a point is inside the unit cell [0, 1).
        """
        point = np.asarray(point)
        return np.all((point >= 0.0) & (point < 1.0))

    def _check_moded_fcoords(self, point):
        """
        Checks if the fractional coordinates are already modded to [0, 1).
        """
        point = np.asarray(point)
        return np.all(np.isclose(np.mod(point, 1), point))

    def _check_edge_inunitcell(self, G, e):
        if "DV" in G.nodes[e[0]]["type"] or "DV" in G.nodes[e[1]]["type"]:
            return False
        return True

    def _put_V_ahead_of_CV(self, edge):
        # Ensures 'V' node is first in edge tuple
        if edge[0].startswith('V'):
            return edge
        return (edge[1], edge[0])

    # Graph construction and edge finding

    def _find_pair_v_e(self, distance_range=None):
        """
        Finds pairs of vertex nodes and edge nodes based on distance.
        Returns: list of pairs, count, and the constructed graph.
        """
        distance_range = self.edge_length_range if distance_range is None else distance_range
        unit_cell = self.unit_cell
        vvnode333 = self.vvnode333
        eenode333 = self.eenode333

        G = self.G
        pair_vertex_edge = []
        vvnode333 = np.asarray(vvnode333)
        eenode333 = np.asarray(eenode333)
        for e in eenode333:
            dist = np.linalg.norm(np.dot(unit_cell, (vvnode333 - e).T).T,
                                  axis=1)
            # If no range provided, take two closest
            if not distance_range:
                idx = np.argpartition(dist, 2)[:2]
                v1_idx, v2_idx = idx
                v1, v2 = vvnode333[v1_idx], vvnode333[v2_idx]
            # If range provided, find pair of nodes within range
            else:
                candidates = np.where((dist > distance_range[0])
                                      & (dist < distance_range[1]))[0]
                if len(candidates) != 2:
                    if self._debug:
                        self.ostream.print_warning(
                            f"Warning: found {len(candidates)} candidates for E node at {e}, expected 2."
                        )
                        self.ostream.print_info(
                            f"Distances to all V nodes: {dist}, min: {np.min(dist)}, candidates: {candidates}"
                        )
                    self.ostream.flush()
                    continue
                v1_idx, v2_idx = candidates
                v1, v2 = vvnode333[v1_idx], vvnode333[v2_idx]
            center = (v1 + v2) / 2
            if self._check_inside_unit_cell(
                    v1) or self._check_inside_unit_cell(v2):
                if np.linalg.norm(center - e) < 1e-3:
                    G.add_node(f"V{v1_idx}", fcoords=v1, note="V", type="V")
                    G.add_node(f"V{v2_idx}", fcoords=v2, note="V", type="V")
                    G.add_edge(f"V{v1_idx}",
                               f"V{v2_idx}",
                               fcoords=(v1, v2),
                               fc_center=e)
                    pair_vertex_edge.append((v1, v2, e))
        self.G = G
        self.pair_vertex_edge = pair_vertex_edge

    def _find_pair_v_e_c(self):
        """
        Finds pairs of vertex nodes and center nodes for linkers in MOF.
        Returns: list of pairs, count, and the constructed graph.
        """
        vvnode333 = self.vvnode333
        ecnode333 = self.ecnode333
        eenode333 = self.eenode333
        unit_cell = self.unit_cell

        G = self.G
        pair_vertex_edge = []
        for e in eenode333:
            dist_v_e = np.linalg.norm(np.dot(unit_cell, (vvnode333 - e).T).T,
                                      axis=1)
            v1_idx = np.argmin(dist_v_e)
            v1 = vvnode333[v1_idx]
            dist_c_e = np.linalg.norm(np.dot(unit_cell, (ecnode333 - e).T).T,
                                      axis=1)
            v2_idx = np.argmin(dist_c_e)
            v2 = ecnode333[v2_idx]
            center = (v1 + v2) / 2
            if self._check_inside_unit_cell(
                    v1) or self._check_inside_unit_cell(v2):
                if np.linalg.norm(center - e) < 0.1:
                    G.add_node(f"V{v1_idx}", fcoords=v1, note="V")
                    G.add_node(f"CV{v2_idx}", fcoords=v2, note="CV")
                    G.add_edge(f"V{v1_idx}",
                               f"CV{v2_idx}",
                               fcoords=(v1, v2),
                               fc_center=e)
                    pair_vertex_edge.append((f"V{v1_idx}", f"CV{v2_idx}", e))
        self.G = G
        self.pair_vertex_edge = pair_vertex_edge

    def _add_ccoords(self, G, unit_cell):
        """
        Adds cartesian coordinates to each node in the graph.
        """
        for n in G.nodes():
            G.nodes[n]["ccoords"] = np.dot(unit_cell, G.nodes[n]["fcoords"])
        return G

    def _set_DV_V(self, G):
        """
        Sets node types to 'V' or 'DV' based on their fractional coordinates.
        Renames nodes accordingly.
        Returns: updated graph and maximum node degree.
        """
        G = self.G.copy()
        for n in list(G.nodes()):
            if self._check_moded_fcoords(G.nodes[n]["fcoords"]):
                G.nodes[n]["type"] = "V"
                G = nx.relabel_nodes(
                    G, {n: pname(n) + "_" + str(np.array([0.0, 0.0, 0.0]))})
            else:
                G.nodes[n]["type"] = "DV"
                diff = G.nodes[n]["fcoords"] - np.mod(G.nodes[n]["fcoords"], 1)
                for n1 in G.nodes():
                    if np.all(
                            np.isclose(G.nodes[n1]["fcoords"],
                                       np.mod(G.nodes[n]["fcoords"], 1))):
                        n1_name = pname(n1)
                G = nx.relabel_nodes(G, {n: n1_name + "_" + str(diff)})
        max_degree = max(dict(G.degree()).values())
        self.G = G.copy()
        self.max_degree = max_degree

        #check if the self.max_degree is correct
        if self._debug:
            self.ostream.print_info(
                f"Max degree of the net: {self.max_degree}")
            self.ostream.print_info(f"Nodes in the net: {G.nodes(data=True)}")
            self.ostream.print_info(f"Edges in the net: {G.edges(data=True)}")
            self.ostream.flush()

    def _find_unitcell_e(self, all_e):
        """
        Finds unique edge centers within the unit cell by checking against supercell translations.
        Args:
            all_e (list of np.ndarray): List of edge centers (fractional coordinates).
        Returns:
            list of np.ndarray: Unique edge centers in the unit cell.
        """
        if not all_e:
            return []

        unique_e = [all_e[0]]
        for e_check in all_e[1:]:
            # Generate all supercell translations of current unique edges
            supercell_e = self._make_supercell_3x3x3(np.array(unique_e))
            # If e_check is not close to any supercell translation, it's unique
            if not np.any(
                    np.all(np.isclose(e_check, supercell_e, atol=1e-8),
                           axis=1)):
                unique_e.append(e_check)
        return unique_e

    def _set_DE_E(self):
        """
        Sets edge types to 'E' or 'DE' based on their uniqueness in the unit cell.
        """
        G = self.G.copy()
        all_e = [G.edges[e]["fc_center"].copy() for e in G.edges()]
        unique_e = np.vstack(self._find_unitcell_e(all_e))
        for e in G.edges():
            if np.any(
                    np.all(np.isclose(G.edges[e]["fc_center"], unique_e),
                           axis=1)):
                G.edges[e]["type"] = "E"
            else:
                G.edges[e]["type"] = "DE"
        self.G = G.copy()

    def _sort_nodes_by_type_connectivity(self):
        G = self.G.copy()
        CV_nodes = [n for n in G.nodes() if G.nodes[n]["note"] == "CV"]
        if len(CV_nodes) == 0:  # ditopic linker MOF
            Vnodes = [n for n in G.nodes() if G.nodes[n]["type"] == "V"]
            DVnodes = [n for n in G.nodes() if G.nodes[n]["type"] == "DV"]
            Vnodes = sorted(Vnodes, key=lambda x: G.degree(x), reverse=True)
            DVnodes = sorted(DVnodes, key=lambda x: G.degree(x), reverse=True)
            self.sorted_nodes = Vnodes + DVnodes
        else:
            # CV+V
            # get CV_Vnode
            CV_Vnodes = [
                n for n in G.nodes()
                if G.nodes[n]["type"] == "V" and G.nodes[n]["note"] == "CV"
            ]
            CV_DVnodes = [
                n for n in G.nodes()
                if G.nodes[n]["type"] == "DV" and G.nodes[n]["note"] == "CV"
            ]
            V_Vnodes = [
                n for n in G.nodes()
                if G.nodes[n]["type"] == "V" and G.nodes[n]["note"] == "V"
            ]
            V_DVnodes = [
                n for n in G.nodes()
                if G.nodes[n]["type"] == "DV" and G.nodes[n]["note"] == "V"
            ]
            CV_Vnodes = sorted(CV_Vnodes,
                               key=lambda x: G.degree(x),
                               reverse=True)
            CV_DVnodes = sorted(CV_DVnodes,
                                key=lambda x: G.degree(x),
                                reverse=True)
            V_Vnodes = sorted(V_Vnodes,
                              key=lambda x: G.degree(x),
                              reverse=True)
            V_DVnodes = sorted(V_DVnodes,
                               key=lambda x: G.degree(x),
                               reverse=True)

            self.sorted_nodes = CV_Vnodes + V_Vnodes + CV_DVnodes + V_DVnodes
        self.G = G.copy()
        if self._debug:
            self.ostream.print_info(f"Sorted nodes: {self.sorted_nodes}")

    def _find_and_sort_edges_bynodeconnectivity(self):
        """
        Sorts edges by node connectivity, prioritizing unit cell edges.
        Returns: sorted list of edges.
        """
        G = self.G.copy()
        sorted_nodes = self.sorted_nodes

        all_edges = list(G.edges())
        sorted_edges = []

        # Add unit cell edges first
        ei = 0
        while ei < len(all_edges):
            e = all_edges[ei]
            if self._check_edge_inunitcell(G, e):
                sorted_edges.append(self._put_V_ahead_of_CV(e))
                all_edges.pop(ei)
            else:
                ei += 1

        # Sort remaining edges by sorted_nodes
        for n in sorted_nodes:
            ei = 0
            while ei < len(all_edges):
                e = all_edges[ei]
                if n in e:
                    sorted_edges.append(
                        self._put_V_ahead_of_CV(e if n == e[0] else (e[1],
                                                                     e[0])))
                    all_edges.pop(ei)
                else:
                    ei += 1
        self.G = G.copy()
        self.sorted_edges = sorted_edges

    def create_net(self, cif_file=None):
        """
        Main function to create the net graph from CIF data.
        """
        self.cif_file = cif_file if cif_file is not None else self.cif_file
        self.cifreader.read_cif(self.cif_file)
        self.ostream.print_info(f"fetching Vertex from {self.cif_file}")
        self.ostream.flush()
        _, _, self.vvnode = self.cifreader.get_type_atoms_fcoords_in_primitive_cell(
            target_type="V")
        self.ostream.print_info(f"fetching Edge from {self.cif_file}")
        self.ostream.flush()
        _, _, self.eenode = self.cifreader.get_type_atoms_fcoords_in_primitive_cell(
            target_type="E")
        self.vvnode333 = self._make_supercell_3x3x3(self.vvnode)
        self.eenode333 = self._make_supercell_3x3x3(self.eenode)
        if self.cifreader.EC_con is not None:
            self.ostream.print_info(
                f"fetching Edge Center from {self.cif_file}")
            self.ostream.flush()
            _, _, self.ecnode = self.cifreader.get_type_atoms_fcoords_in_primitive_cell(
                target_type="EC")
            self.ecnode333 = self._make_supercell_3x3x3(self.ecnode)

        self.cell_info = self.cifreader.cell_info
        self.unit_cell = self._extract_unit_cell(self.cell_info)
        self.unit_cell_inv = np.linalg.inv(self.unit_cell)

        if self.cifreader.EC_con is None:  #ditopic linker MOF
            self.linker_topic = 2
            self._find_pair_v_e(distance_range=self.edge_length_range)
            self._add_ccoords(self.G, self.unit_cell)
            self._set_DV_V(self.G)
            self._set_DE_E()
            self._sort_nodes_by_type_connectivity()
            self._find_and_sort_edges_bynodeconnectivity()
        else:  # multitopic linker MOF
            self._find_pair_v_e_c()
            self._add_ccoords(self.G, self.unit_cell)
            self._set_DV_V(self.G)
            self._set_DE_E()
            self._sort_nodes_by_type_connectivity()
            self._find_and_sort_edges_bynodeconnectivity()
            self.linker_topic = self.cifreader.EC_con

        if self._debug:
            self.ostream.print_info(f"Net created from {self.cif_file}")
            self.ostream.print_info(
                f"Nodes in the net: {self.G.nodes(data=True)}")
            self.ostream.print_info(
                f"Edges in the net: {self.G.edges(data=True)}")
            self.ostream.print_info(f"Sorted nodes: {self.sorted_nodes}")
            self.ostream.print_info(f"Sorted edges: {self.sorted_edges}")
            self.ostream.print_info(
                f"Max degree of the net: {self.max_degree}")
            self.ostream.flush()

        self.ostream.print_info(
            f"Net created from {self.cif_file} with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges."
        )
        self.ostream.flush()

        if int(self.max_degree) != int(self.cifreader.V_con):
            self.ostream.print_warning(
                f"Max degree of the net: {self.max_degree}, {self.max_degree==self.cifreader.V_con} to the net connectivity {self.cifreader.V_con}"
            )
        else:
            self.ostream.print_info(
                f"This net fits a {self.max_degree}-connected node with {self.linker_topic}-connected linker Framework."
            )
            self.ostream.flush()


def pname(s):
    """
    Extracts the primitive cell vertex node name from a string.
    Example: 'V1_[-1.  0.  0.]' -> 'V1'
    """
    return s.split("_")[0]


def lname(s):
    """
    Extracts the lattice vector from a node name string.
    Example: 'V1_[-1.  0.  0.]' -> np.array([-1., 0., 0.])
    """
    parts = s.split("_")
    if len(parts) < 2:
        return np.array([0.0, 0.0, 0.0])
    return np.asanyarray(parts[1][1:-1].split(), dtype=float)


def arr_dimension(arr):
    """
    Returns the dimension of a numpy array.
    """
    return 2 if arr.ndim > 1 else 1


def is_list_A_in_B(A, B):
    """
    Checks if all arrays in A are close to corresponding arrays in B.
    """
    return all(np.allclose(a, b, atol=1e-9) for a, b in zip(A, B))


if __name__ == "__main__":
    cif_file = "tests/testdata/test.cif"
    netgraph = FrameNet()
    netgraph.cifreader._debug = netgraph._debug
    #netgraph._debug = True
    netgraph.create_net(cif_file=cif_file)
    #print("cell info:", netgraph.cell_info)
    #print("unit cell:\n", netgraph.unit_cell)
    #print("unit cell inv:\n", netgraph.unit_cell_inv)
