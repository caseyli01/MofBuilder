import sys
from pathlib import Path
import numpy as np
import networkx as nx
import mpi4py.MPI as MPI

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.molecule import Molecule

from ..io.pdb_reader import PdbReader
from ..io.pdb_writer import PdbWriter

class FrameLinker:
    def __init__(self, comm=None, ostream=None, filepath=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)
        self.properties = {}
        self.filename = filepath
        self.target_dir = None
        self.new_xyzfilename = None
        self.linker_connectivity = 2
        self.pdbreader = PdbReader(comm=self.comm, ostream=self.ostream)
        self.pdbwriter = PdbWriter(comm=self.comm, ostream=self.ostream)
        self._debug = False
        self.linker_data = None
        self.lines = None #center fragment lines
        self.rows = None #outer fragment lines
        self.save_files = False
    
    def check_dirs(self, passfilecheck=True):
        if not passfilecheck:
            assert_msg_critical(
                Path(self.filename).exists(),
                f"Linker file {self.filename} not found"
            )
        self.target_dir = Path(self.target_dir or Path.cwd())
        self.target_dir.mkdir(parents=True, exist_ok=True)
        base = Path(self.filename).stem
        if self.save_files:
            self.new_pdbfilename = self.target_dir / f"{base}_processed.pdb"
    
        if not passfilecheck :
            self.ostream.print_info(f"Processing linker file: {self.filename} ...")
        else:
            self.ostream.print_info(f"Processing linker data ...")
        self.ostream.print_info(f"Linker topic: {self.linker_connectivity}")
        self.ostream.flush()

        if self._debug:
            self.ostream.print_info(f"Target directory: {self.target_dir}")

        

    def _create_lG(self, molecule):
        matrix = molecule.get_connectivity_matrix()
        coords = molecule.get_coordinates_in_angstrom()
        labels = molecule.get_labels()
        dist_matrix = molecule.get_distance_matrix_in_angstrom()
        mass_center_bohr = molecule.center_of_mass_in_bohr()
        bohr_to_angstrom = 0.529177
        mass_center_angstrom = np.asarray(mass_center_bohr) * bohr_to_angstrom
        coords -= mass_center_angstrom

        metal_elements_list = {
            "Ag", "Al", "Au", "Ba", "Be", "Bi", "Ca", "Cd", "Ce", "Co", "Cr", "Cs", "Cu", "Fe", "Ga", "Gd", "Hf", "Hg",
            "In", "Ir", "K", "Li", "Mg", "Mn", "Na", "Ni", "Pb", "Pd", "Pt", "Rb", "Rh", "Sc", "Sn", "Sr", "Ti", "V",
            "W", "Y", "Zn", "Zr"
        }

        lG = nx.Graph()
        metals = [i for i, label in enumerate(labels) if label in metal_elements_list]
        for i, label in enumerate(labels):
            lG.add_node(i, label=label, coords=coords[i])
        for i in range(len(labels)):
            for j in np.where(matrix[i] == 1)[0]:
                if i not in metals and j not in metals:
                    lG.add_edge(i, j, weight=dist_matrix[i, j])
        if self._debug:
            self.ostream.print_info(f"Number of atoms: {len(labels)}")
            self.ostream.print_info(f"Number of metal atoms: {len(metals)}")
            self.ostream.print_info(f"Number of bonds in linker graph: {lG.number_of_edges()}")
            self.ostream.flush()
        self.lG = lG
        self.metals = metals
        self.mass_center_angstrom = mass_center_angstrom

    def _find_center_highly_connected_isolated_cycle(self, lG):
        max_frag_num = 0
        min_frag_size_std = float('inf')
        center_cycle = []
        cycles = list(nx.simple_cycles(lG, length_bound=200))
        for cycle in cycles:
            lG_temp = lG.copy()
            lG_temp.remove_nodes_from(cycle)
            frag_num = nx.number_connected_components(lG_temp)
            frag_sizes = [len(f) for f in nx.connected_components(lG_temp)]
            frag_size_std = np.std(frag_sizes)
            if frag_num > max_frag_num or (frag_num == max_frag_num and frag_size_std < min_frag_size_std):
                max_frag_num = frag_num
                min_frag_size_std = frag_size_std
                center_cycle = cycle
        return center_cycle

    def _find_center_cycle_nodes(self, lG):
        return self._find_center_highly_connected_isolated_cycle(lG)

    def _check_two_points_center(self, lG, centers):
        if nx.shortest_path_length(lG, centers[0], centers[1]) != 1:
            return False
        G = lG.copy()
        G.remove_edge(centers[0], centers[1])
        return nx.number_connected_components(G) == 2

    def _in_same_cycle(self, G, nodes):
        for cycle in nx.cycle_basis(G):
            if set(nodes).issubset(cycle):
                return cycle
        return None

    def _find_centers(self, lG):
        barycenter = nx.barycenter(lG)
        normalcenter = nx.center(lG)
        if set(barycenter).intersection(normalcenter):
            if self._in_same_cycle(lG, normalcenter) is not None:
                return normalcenter
            return barycenter
        return barycenter

    def _distinguish_G_centers(self, lG):
        centers = self._find_centers(lG)
        if len(centers) == 1:
            center_class = "onepoint"
            center_nodes = centers
        elif len(centers) == 2:
            if self._check_two_points_center(lG, centers):
                center_class = "twopoints"
                center_nodes = centers
            else:
                lG.remove_edge(centers[0], centers[1])
                center_class = "cycle"
                center_nodes = self._find_center_cycle_nodes(lG)
        else:
            center_class = "cycle"
            center_nodes = self._find_center_cycle_nodes(lG)
        if self._debug:
            self.ostream.print_info(f"center_nodes: {center_nodes}, center_class: {center_class}, n_centers: {len(centers)}")
        self.lG = lG
        self.center_nodes = center_nodes
        self.center_class = center_class

    def _classify_nodes(self):
        lG = self.lG.copy()
        for center_ind, center in enumerate(self.center_nodes):
            lengths = nx.single_source_shortest_path_length(lG, center)
            for k in lengths:
                if center_ind == 0 or lengths[k] < lG.nodes[k].get("cnodes_l", (None, float('inf')))[1]:
                    lG.nodes[k]["cnodes_l"] = (center, lengths[k])
                elif lengths[k] == lG.nodes[k]["cnodes_l"][1]:
                    lG.nodes[k]["cnodes_l"] = (-1, lengths[k])
        self.lG = lG

    # ... (rest of the code remains mostly unchanged, apply similar optimizations for helper functions)

    def _get_pairX_outer_frag(self, connected_pairXs, outer_frag_nodes):
        for x in list(connected_pairXs):
            pairXs = [connected_pairXs[x][1], connected_pairXs[x][3]]
            if set(pairXs) < set(outer_frag_nodes):
                break
        return pairXs


    def _cleave_outer_frag_subgraph(self, lG, pairXs, outer_frag_nodes):
        subgraph_outer_frag = lG.subgraph(outer_frag_nodes)
        kick_nodes = []
        #test to remove nodeX and kick the small frag
        for i in pairXs:
            lG_temp = subgraph_outer_frag.copy()
            lG_temp.remove_node(i)
            frags = list(nx.connected_components(lG_temp))
            #sort frags by size
            frags.sort(key=len)
            small_frag_nodes = list(frags[0])
            if len(frags)==1:
                continue
            if self._debug:
                self.ostream.print_info(f"small_frag_nodes: {small_frag_nodes}")
                self.ostream.flush()
            kick_nodes.extend(small_frag_nodes)

        if self._debug:
            self.ostream.print_info(f"kick_nodes: {kick_nodes}")
            self.ostream.flush()

        subgraph_single_frag = lG.subgraph(outer_frag_nodes - set(kick_nodes))
        return subgraph_single_frag


    def _lines_of_center_frag(self):
        subgraph_center_frag = self.subgraph_center_frag
        Xs_indices = self.Xs_indices
        metals = self.metals
        labels = self.molecule_labels
        coords = self.molecule_coords
        mass_center_angstrom = self.mass_center_angstrom

        count = 1
        lines = []
        Xs = []
        for cn in list(subgraph_center_frag.nodes):
            label = subgraph_center_frag.nodes[cn]["label"]
            coord = subgraph_center_frag.nodes[cn]["coords"]
            if cn not in Xs_indices:
                name = label + str(count)
            else:
                name = "X" + str(count)
                Xs.append(count - 1)
            count += 1
            lines.append([name, label, coord[0], coord[1], coord[2]])
        for cm in metals:
            label = labels[cm]
            coord = coords[cm] - mass_center_angstrom
            name = label + str(count)
            lines.append([name, label, coord[0], coord[1], coord[2]])
        
        self.lines = lines
        self.center_Xs = Xs

    def create_pdb(self, filename, lines):
        header = f"REMARK   Generated by MOFbuilder Linker module\n"
        self.pdbwriter.write(filename, header=header, lines=lines)
        if self._debug:
            self.ostream.print_info(f"Written PDB file: {filename}")
            self.ostream.flush()

    def _lines_of_single_frag(self, subgraph_single_frag, Xs_indices):
        count = 1
        rows = []
        Xs = []
        for sn in list(subgraph_single_frag.nodes):
            label = subgraph_single_frag.nodes[sn]["label"]
            coord = subgraph_single_frag.nodes[sn]["coords"]
            if sn not in Xs_indices:
                name = label + str(count)
            else:
                name = "X" + str(count)
                Xs.append(count - 1)
            count += 1
            rows.append([name, label, coord[0], coord[1], coord[2]])
        return rows, Xs

    def process_linker_molecule(self,molecule, linker_connectivity):
        """
        Processes the linker molecule based on the linker_connectivity and center classification.
        Identifies center nodes, Xs (connection points), fragments, and writes PDB files for each fragment.
        """
        if self.save_files:
            save_nodes_dir = Path(self.target_dir, "nodes")
            save_edges_dir = Path(self.target_dir, "edges")
        self.molecule_coords = molecule.get_coordinates_in_angstrom()
        self.molecule_labels = molecule.get_labels()

        # Remove metal atoms from the graph
        self._create_lG(molecule)
        self.lG.remove_nodes_from(self.metals)
        self._distinguish_G_centers(self.lG)

        # For large cycles, reduce center nodes to a pair
        if linker_connectivity == 2 and len(self.center_nodes) > 6:
            self.center_nodes = self._find_center_nodes_pair(self.lG, self.center_nodes)

        self._classify_nodes()
        if self._debug:
            self.ostream.print_info(f"Linker topic: {linker_connectivity}")
            self.ostream.print_info(f"Center class: {self.center_class}")
            self.ostream.print_info(f"Center nodes: {self.center_nodes}")

        # Multitopic linker: center is a cycle
        if self.center_class == "cycle" and linker_connectivity > 2:
            if self._debug:
                self.ostream.print_info(f"tritopic/tetratopic/multitopic: center is a cycle")
            connected_pairXs = {}
            Xs_indices = []
            innerX_coords = []

            # Find Xs connected to the cycle
            for k in range(len(self.center_nodes)):
                linker_C_l = []
                l_list = []
                for n in self.lG.nodes:
                    if (self.lG.nodes[n]["cnodes_l"][0] == self.center_nodes[k] and
                            self.lG.nodes[n]["label"] == "C"):
                        linker_C_l.append((n, self.lG.nodes[n]["cnodes_l"]))
                        l_list.append(self.lG.nodes[n]["cnodes_l"][1])
                center_connected_C_ind = [
                    ind for ind, value in enumerate(l_list) if value == 1
                ]
                outer_connected_C_ind = [
                    ind for ind, value in enumerate(l_list)
                    if value == (max(l_list) - 1)
                ]
                if len(center_connected_C_ind) == 1 and len(
                        outer_connected_C_ind) == 1:
                    inner_X = linker_C_l[center_connected_C_ind[0]]
                    outer_X = linker_C_l[outer_connected_C_ind[0]]
                    if self.center_nodes[k] not in [inner_X[0], outer_X[0]]:
                        if self._debug:
                            self.ostream.print_info(f"find connected X in edge frag: {inner_X[0]}, {outer_X[0]}, {self.center_nodes[k]}")
                        self.lG.remove_edge(inner_X[0], self.center_nodes[k])
                        connected_pairXs[self.center_nodes[k]] = (
                            "inner_X",
                            inner_X[0],
                            "outer_X",
                            outer_X[0],
                        )
                        Xs_indices += [self.center_nodes[k], inner_X[0], outer_X[0]]
                        innerX_coords.append(self.lG.nodes[inner_X[0]]["coords"])

            if (nx.number_connected_components(self.lG)
                    != linker_connectivity + 1):  # for check linker_connectivitys+1
                self.ostream.print_warning(f"wrong fragments")
                raise ValueError

            self.Xs_indices = Xs_indices
            self.innerX_coords = innerX_coords
            self.connected_pairXs = connected_pairXs

        elif self.linker_connectivity == 2:
            if self.center_class == "twopoints":
                if self._debug:
                    self.ostream.print_info(f"ditopic linker: center are two points")
                
                Xs_indices = []
                for k in range(len(self.center_nodes)):
                    linker_C_l = []
                    l_list = []
                    for n in self.lG.nodes:
                        if (self.lG.nodes[n]["cnodes_l"][0] == self.center_nodes[k] and
                                self.lG.nodes[n]["label"] == "C"):
                            linker_C_l.append((n, self.lG.nodes[n]["cnodes_l"]))
                            l_list.append(self.lG.nodes[n]["cnodes_l"][1])

                    outer_connected_C_ind = [
                        ind for ind, value in enumerate(l_list)
                        if value == (max(l_list) - 1)
                    ]

                    if len(outer_connected_C_ind) == 1:
                        outer_X = linker_C_l[outer_connected_C_ind[0]]
                        if self.center_nodes[k] not in [outer_X[0]]:
                            if self._debug:
                                self.ostream.print_info(f"find connected X in edge: {outer_X[0]}")
                            Xs_indices += [outer_X[0]]

                self.Xs_indices = Xs_indices
                

            if self.center_class == "onepoint":
                if self._debug:
                    self.ostream.print_info(f"ditopic linker: center is a point")
                Xs_indices = []
                linker_C_l = []
                l_list = []
                for n in self.lG.nodes:
                    if (self.lG.nodes[n]["cnodes_l"][0] == self.center_nodes[0] and
                            self.lG.nodes[n]["label"] == "C"):
                        linker_C_l.append((n, self.lG.nodes[n]["cnodes_l"]))
                        l_list.append(self.lG.nodes[n]["cnodes_l"][1])

                outer_connected_C_ind = [
                    ind for ind, value in enumerate(l_list)
                    if value == (max(l_list) - 1)
                ]
                for m in range(len(outer_connected_C_ind)):
                    outer_X = linker_C_l[outer_connected_C_ind[m]]
                    if self._debug:
                        self.ostream.print_info(f"find connected X in edge: {outer_X[0]}")
                    Xs_indices += [outer_X[0]]
                self.Xs_indices = Xs_indices

            if self.center_class == "cycle":
                if self._debug:
                    self.ostream.print_info(f"ditopic linker: center is a cycle")
                connected_pairXs = {}
                Xs_indices = []
                for k in range(len(self.center_nodes)):
                    linker_C_l = []
                    l_list = []
                    for n in self.lG.nodes:
                        if (self.lG.nodes[n]["cnodes_l"][0] == self.center_nodes[k] and
                                self.lG.nodes[n]["label"] == "C"):
                            linker_C_l.append((n, self.lG.nodes[n]["cnodes_l"]))
                            l_list.append(self.lG.nodes[n]["cnodes_l"][1])
                    outer_connected_C_ind = [
                        ind for ind, value in enumerate(l_list)
                        if value == (max(l_list) - 1)
                    ]
                    if len(outer_connected_C_ind) == 1:
                        outer_X = linker_C_l[outer_connected_C_ind[0]]
                        if self.center_nodes[k] not in [outer_X[0]]:
                            if self._debug:
                                self.ostream.print_info(f"find connected X in edge: {outer_X[0]}")
                            Xs_indices += [outer_X[0]]
                self.Xs_indices = Xs_indices

                if len(self.Xs_indices) < 2:
                    if self._debug:
                        self.ostream.print_info(f"Xs in the center cycle")
                    # the linker is a cycle, but no Xs are found by the dist, then the X in in the center cycle:
                    # the node whose adjacents(nonH) more than 2 are the Xs
                    for n in self.center_nodes:
                        adj_nonH_num = 0
                        if self.lG.nodes[n]["label"] == "C":
                            adj_nodes = list(self.lG.adj[n])
                            for adj in adj_nodes:
                                if self.lG.nodes[adj]["label"] != "H":
                                    adj_nonH_num += 1
                            if adj_nonH_num > 2:
                                self.Xs_indices.append(n)
                    #double check Xs_indices
                    if len(self.Xs_indices) > self.linker_connectivity:
                        #cut bond connected to Xs_indices and if there is a fragment include H then should exclude this Xs
                        #use lG fragment
                        for x in self.Xs_indices:
                            lG_temp = self.lG.copy()
                            lG_temp.remove_node(x)
                            if nx.number_connected_components(lG_temp) == self.linker_connectivity:
                                frags = list(nx.connected_components(lG_temp))
                                #sort frags by size
                                frags.sort(key=len)
                                small_frag_nodes = list(frags[0])
                                #labels of small frag
                                small_frag_labels = [
                                    lG_temp.nodes[n]["label"]
                                    for n in small_frag_nodes
                                ]
                                if not any(label == "H"
                                        for label in small_frag_labels):
                                    continue
                                else:
                                    self.Xs_indices.remove(x)
                            else:
                                self.Xs_indices.remove(x)

        else:
            raise ValueError(
                "failed to recognize a multitopic linker whose center is not a cycle"
            )

        # write pdb files
        if self.linker_connectivity > 2:  # multitopic
            frag_nodes = list(
                sorted(nx.connected_components(self.lG), key=len, reverse=True))
            for f in frag_nodes:
                if set(self.center_nodes) < set(f):
                    center_frag_nodes = f
                else:
                    outer_frag_nodes = f
            if self._debug:
                self.ostream.print_info(f"center_frag_nodes: {center_frag_nodes}")
                self.ostream.print_info(f"outer_frag_nodes: {outer_frag_nodes}")
            self.subgraph_center_frag = self.lG.subgraph(center_frag_nodes)
            self._lines_of_center_frag()
            self.pairXs = self._get_pairX_outer_frag(connected_pairXs, outer_frag_nodes)
            self.subgraph_single_frag = self._cleave_outer_frag_subgraph(
                self.lG, self.pairXs, outer_frag_nodes)
            if self._debug:
                self.ostream.print_info(f"subgraph_single_frag nodes: {self.subgraph_single_frag.nodes}")
            self.rows, self.frag_Xs = self._lines_of_single_frag(self.subgraph_single_frag, self.Xs_indices)
            if self._debug:
                self.ostream.print_info(f"subgraph_single_frag nodes: {self.subgraph_single_frag.nodes}")
            if linker_connectivity == 3:
                if self._debug:
                    self.ostream.print_info(f"linker_center_frag: {self.subgraph_center_frag.number_of_nodes()}, {self.center_Xs}")
                    self.ostream.print_info(f"linker_outer_frag: {self.subgraph_single_frag.number_of_nodes()}, {self.frag_Xs}")
                linker_center_node_pdb_name = str(Path(save_nodes_dir, "tricenter"))
                if self.save_files:
                    self.create_pdb(linker_center_node_pdb_name, self.lines)
                    linker_branch_pdb_name = str(Path(save_edges_dir, "triedge"))
                    self.create_pdb(linker_branch_pdb_name, self.rows)

            elif linker_connectivity == 4:
                if self._debug:
                    self.ostream.print_info(f"center_frag: {self.subgraph_center_frag.number_of_nodes()}, {self.center_Xs}")
                    self.ostream.print_info(f"outer_frag: {self.subgraph_single_frag.number_of_nodes()}, {self.frag_Xs}")
                if self.save_files:
                    linker_center_node_pdb_name = str(Path(save_nodes_dir, "tetracenter"))
                    self.create_pdb(linker_center_node_pdb_name, self.lines)
                    linker_branch_pdb_name = str(Path(save_edges_dir, "tetraedge"))
                    self.create_pdb(linker_branch_pdb_name, self.rows)

            else:
                linker_center_node_pdb_name = str(
                    Path(save_nodes_dir, "multicenter"))
                if self.save_files:
                    self.create_pdb(linker_center_node_pdb_name, self.lines)
                    linker_branch_pdb_name = str(Path(save_edges_dir, "multiedge"))
                    self.create_pdb(linker_branch_pdb_name, self.rows)


        elif linker_connectivity == 2:  # ditopic
            pairXs = Xs_indices
            if self._debug:
                self.ostream.print_info(f"pairXs: {pairXs}")
            self.subgraph_center_frag = self._cleave_outer_frag_subgraph(self.lG, pairXs, self.lG.nodes)
            self._lines_of_center_frag()

            if self.save_files:
                edge_pdb_name = str(Path(save_edges_dir, "diedge"))
                self.create_pdb(edge_pdb_name, self.lines)
            if self._debug:
                self.ostream.print_info(f"linker_center_frag: {self.subgraph_center_frag.number_of_nodes()}, {self.center_Xs}")





    def create(self, molecule=None):
        if self.save_files:
            assert_msg_critical(self.target_dir is not None, "Linker: target_dir is not set. Please set the target directory.")
        #assert_msg_critical(self.linker_connectivity in [2, 3, 4] or int(self.linker_connectivity) > 4, "Linker: linker_connectivity should be 2, 3, 4 or >4.")

        if molecule is None:
            assert_msg_critical(self.filename is not None, "Linker: filename is not set. Please set the filename of the linker molecule.")
            self.check_dirs()
            self.molecule = Molecule.read_xyz_file(self.filename)
        else:
            self.molecule = molecule

        self.process_linker_molecule(self.molecule, self.linker_connectivity)
        self.linker_center_data, self.linker_center_X_data = self.pdbreader.expand_arr2data(self.lines)
        self.linker_outer_data, self.linker_outer_X_data = self.pdbreader.expand_arr2data(self.rows)
        self.ostream.print_info("Linker processing completed.")
        if hasattr(self, "new_pdbfilename"):
            self.ostream.print_info(f"Processed linker file is saved as: {self.new_pdbfilename}")
        self.ostream.flush()


if __name__ == "__main__":
    linker_test = FrameLinker()
    linker_test.linker_connectivity = 2
    linker_test.filename = "tests/testdata/testlinker.xyz"
    #linker_test.target_dir = "tests/testoutput"
    #linker_test._debug = True
    linker_test.create()

    linker_test.linker_connectivity = 4
    linker_test.filename = "tests/testdata/testtetralinker.xyz"
    #linker_test.target_dir = "tests/testoutput"
    linker_test.create()

    linker_test.linker_connectivity = 3
    linker_test.filename = "tests/testdata/testtrilinker.xyz"
    #linker_test.target_dir = "tests/testoutput"
    linker_test.create()
