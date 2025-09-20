import sys
from pathlib import Path

import numpy as np
import networkx as nx
import mpi4py.MPI as MPI

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.molecule import Molecule

from ..io.basic import nn, nl
from .superimpose import superimpose
from ..io.pdb_reader import PdbReader
from ..io.pdb_writer import PdbWriter


class Node:

    def __init__(self, comm=None, ostream=None, filepath=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        if ostream is None:
            ostream = OutputStream(sys.stdout if self.rank ==
                                   mpi_master() else None)
        self.ostream = ostream

        self.properties = {}
        self.filename = filepath
        self.target_dir = None
        self.new_pdbfilename = None
        self.new_dummy_dictfilename = None

        self.dummy_node = False
        self.node_metal_type = None

        self.node_data = None
        self.node_xyz_string = None
        self.nodeG = nx.Graph()

        self.pdbreader = PdbReader(comm=self.comm, ostream=self.ostream)
        self.pdbwriter = PdbWriter(comm=self.comm, ostream=self.ostream)

        self.sG = None
        self.sG_subparts = None
        self._debug = False

        # properties for dummy node
        self.metal_valence = None
        self.dummy_pdbfile = None
        self.dummy_node_split_dict = None
        self.dummy_node_split_dict_path = None
        self.node_data = None

    def check_dirs(self):
        assert_msg_critical(
            Path(self.filename).exists(),
            f"Node pdb file {self.filename} not found")
        self.target_dir = Path(self.target_dir or Path.cwd())
        self.target_dir.mkdir(parents=True, exist_ok=True)
        base = Path(self.filename).stem
        self.new_pdbfilename = self.target_dir / (f"{base}_dummy.pdb"
                                                  if self.dummy_node else Path(
                                                      self.filename).name)
        if self.dummy_node:
            self.new_dummy_dictfilename = self.target_dir / f"{base}_dummy_dict.txt"
        if self._debug:
            self.ostream.print_info(f"Node pdb file: {self.filename}")
            self.ostream.print_info(f"Target directory: {self.target_dir}")
            self.ostream.print_info(f"New pdb file: {self.new_pdbfilename}")
            self.ostream.flush()

    def _nodepdb2xyz(self):
        self.pdbreader.filepath = self.filename
        self.pdbreader.read_pdb()
        self.node_data = self.pdbreader.data
        xyz_lines = [f"{len(self.node_data)}\n", "\n"]
        xyz_lines += [f"{n[1]} {n[5]} {n[6]} {n[7]}\n" for n in self.node_data]
        self.node_xyz_string = ''.join(xyz_lines)
        if self._debug:
            self.ostream.print_info(
                f"Node xyz string generated with {len(self.node_data)} atoms.")
            self.ostream.print_info(self.node_xyz_string)
            self.ostream.flush()

    def _nodepdb2G(self):
        node_data = self.node_data
        G = self.nodeG
        metal = nn(self.node_metal_type)
        coords = np.array(node_data)[:, 5:8].astype(np.float64)
        for n, c in zip(node_data, coords):
            G.add_node(n[0], ccoords=c, type=n[1])

        node_mol = Molecule.read_xyz_string(self.node_xyz_string)
        con_matrix = node_mol.get_connectivity_matrix()
        for i in range(len(node_data)):
            for j in range(i, len(node_data)):
                if con_matrix[i, j] == 1:
                    ni, nj = nn(node_data[i][0]), nn(node_data[j][0])
                    if ni == nj == metal or (ni == metal and nj in [
                            "X", "H"
                    ]) or (nj == metal and ni in ["X", "H"]):
                        continue
                    G.add_edge(node_data[i][0], node_data[j][0])
        self.nodeG = G

    def _fetch_template(self, metal):
        templates = {
            "Zr":
            np.array([[0.70710678, 0.70710678, 0.0],
                      [-0.70710678, 0.70710678, 0.0],
                      [-0.70710678, -0.70710678, 0.0],
                      [0.70710678, -0.70710678, 0.0],
                      [0.0, 0.70710678, 0.70710678],
                      [0.0, -0.70710678, 0.70710678],
                      [0.0, -0.70710678, -0.70710678],
                      [0.0, 0.70710678, -0.70710678]]),
            "Hf":
            np.array([[0.70710678, 0.70710678, 0.0],
                      [-0.70710678, 0.70710678, 0.0],
                      [-0.70710678, -0.70710678, 0.0],
                      [0.70710678, -0.70710678, 0.0],
                      [0.0, 0.70710678, 0.70710678],
                      [0.0, -0.70710678, 0.70710678],
                      [0.0, -0.70710678, -0.70710678],
                      [0.0, 0.70710678, -0.70710678]]),
            "Al":
            np.array([[0.74, 0.0, 0.0], [0.0, 0.74, 0.0], [0.0, 0.0, 0.74],
                      [-0.74, 0.0, 0.0], [0.0, -0.74, 0.0], [0.0, 0.0,
                                                             -0.74]]),
            "Fe":
            np.array([[0.86, 0.0, 0.0], [0.0, 0.86, 0.0], [0.0, 0.0, 0.86],
                      [-0.86, 0.0, 0.0], [0.0, -0.86, 0.0], [0.0, 0.0,
                                                             -0.86]]),
            "Cr":
            np.array([[0.82, 0.0, 0.0], [0.0, 0.82, 0.0], [0.0, 0.0, 0.82],
                      [-0.82, 0.0, 0.0], [0.0, -0.82, 0.0], [0.0, 0.0, -0.82]])
        }
        assert_msg_critical(
            metal in templates,
            f"Error: metal type {metal} not supported for dummy atom addition. Supported metals are {list(templates)}."
        )
        return templates[metal]

    def _order_ccoords(self, d_ccoords, template, target_metal_coords):
        d_ccoords -= target_metal_coords
        _, rot, _ = superimpose(template, d_ccoords)
        return np.dot(template, rot) + target_metal_coords

    def _add_dummy_atoms_nodepdb(self):
        metal = self.node_metal_type
        self.metal_valence = 4 if metal in [
            "Zr", "Hf"
        ] else 3 if metal in ["Al", "Fe", "Cr"] else None
        self.dummy_pdbfile = f"{Path(self.filename).stem}_dummy.pdb"
        template = self._fetch_template(metal)
        sG = self.nodeG.copy()

        if self._debug:
            self.ostream.print_info(
                f"Original node graph has {sG.number_of_nodes()} nodes and {sG.number_of_edges()} edges."
            )
            self.ostream.flush()

        nodes = list(sG.nodes())
        ind_max = max(map(lambda n: int(nl(n)), nodes))
        hydrogen_nodes = [n for n in nodes if nn(n) == "H"]
        oxygen_nodes = [n for n in nodes if nn(n) == "O"]
        metal_nodes = [n for n in nodes if nn(n) == metal]

        self.ostream.print_info(
            f"Found {len(metal_nodes)} metal nodes, {len(oxygen_nodes)} oxygen nodes, {len(hydrogen_nodes)} hydrogen nodes."
        )
        self.ostream.flush()

        # Add missing edges between metal and nearest oxygens
        for metal_n in metal_nodes:
            dists = [(oxy_n,
                      np.linalg.norm(sG.nodes[metal_n]['ccoords'] -
                                     sG.nodes[oxy_n]['ccoords']))
                     for oxy_n in oxygen_nodes]
            for oxy_node, _ in sorted(
                    dists, key=lambda x: x[1])[:2 * self.metal_valence]:
                if oxy_node not in sG.adj[metal_n]:
                    sG.add_edge(metal_n, oxy_node)

        count = ind_max + 1
        for mn in metal_nodes:
            neighbor_nodes = list(sG.adj[mn])
            if len(neighbor_nodes) == 2 * self.metal_valence and all(
                    nn(i) == "O" for i in neighbor_nodes):
                beginning_cc = np.round(sG.nodes[mn]["ccoords"], 4)
                d_ccoords = []
                for nO in neighbor_nodes:
                    sO = np.round(sG.nodes[nO]["ccoords"], 4)
                    cnorm_vec = (
                        sO - beginning_cc) / np.linalg.norm(sO - beginning_cc)
                    d_ccoords.append(beginning_cc + cnorm_vec)
                    sG.remove_edge(mn, nO)
                ordered_ccoords = self._order_ccoords(d_ccoords, template,
                                                      beginning_cc)
                for row in range(len(d_ccoords)):
                    d_name = f"D{count}"
                    sG.add_node(d_name, type="D", ccoords=ordered_ccoords[row])
                    sG.add_edge(mn, d_name)
                    count += 1

        # Ensure hydrogens are connected to nearest oxygen
        for hn in hydrogen_nodes:
            if not list(nx.neighbors(sG, hn)):
                nearest_o = min(oxygen_nodes,
                                key=lambda on: np.linalg.norm(sG.nodes[hn][
                                    "ccoords"] - sG.nodes[on]["ccoords"]))
                sG.add_edge(hn, nearest_o)

        self.sG = sG
        self.sG_subparts = sorted(nx.connected_components(sG),
                                  key=len,
                                  reverse=True)

    def _lines_of_atoms(self, subgraph, subgraph_nodes):
        rows = [[
            sn, subgraph.nodes[sn]["type"], *subgraph.nodes[sn]["ccoords"]
        ] for sn in subgraph_nodes]
        has_dummy = any(nn(i) == "D" for i in subgraph_nodes)
        rows.sort(key=lambda x: (x[1], x[0]), reverse=not has_dummy)
        return rows

    def _get_bonds_from_subgraph(self, subgraph):
        bonds = []
        for atom1, atom2 in subgraph.edges():
            bond_type = "A" if nn(atom1) == "X" or nn(atom2) == "X" else "S"
            bonds.append([atom1, atom2, 1, ".", bond_type])
        return bonds

    def _write_dummy_node_pdb(self):
        dummy_pdbfile_full_path = Path(self.target_dir) / self.dummy_pdbfile
        metal = self.node_metal_type
        sG = self.sG

        all_atom_lines = []
        all_atom_bonds = []

        for subnodes in self.subpart_nodes:
            subgraph = sG.subgraph(subnodes)
            sorted_nodes = sorted(subnodes)
            all_atom_lines.extend(self._lines_of_atoms(subgraph, sorted_nodes))
            all_atom_bonds.extend(self._get_bonds_from_subgraph(subgraph))

        header = (
            "Generated by MOFbuilder\n"
            f"REMARK Dummy atoms added to {Path(self.dummy_pdbfile).name}\n"
            f"REMARK Metal type: {metal}, Valence: {self.metal_valence}\n"
            f"REMARK Total atoms: {len(all_atom_lines)}, Total bonds: {len(all_atom_bonds)}\n"
        )
        self.pdbwriter.write(dummy_pdbfile_full_path,
                             header=header,
                             lines=all_atom_lines)

    def _generate_dummy_node_split_dict(self):
        head, tail = [], []
        for sub in self.sG_subparts:
            l = [nn(i) for i in sub]
            if "X" not in l:
                head.append(sorted(sub))
            else:
                tail.append(sorted(sub))

        self.subpart_nodes = head + tail

        sub_headlens = [len(i) for i in head]
        sub_taillens = [len(i) for i in tail]
        dummy_count = sum(1 for i in sub_headlens if i not in (1, 2, 3))
        hho_count = sub_headlens.count(3)
        ho_count = sub_headlens.count(2)
        o_count = sub_headlens.count(1)
        dummy_res_len = next((i for i in sub_headlens if i not in (1, 2, 3)),
                             0)
        ooc_count = sub_taillens.count(3)

        node_split_dict = {
            "HHO_count": hho_count,
            "HO_count": ho_count,
            "O_count": o_count,
            "METAL_count": dummy_count,
            "dummy_res_len": dummy_res_len,
            "OOC_count": ooc_count,
            "inres_count": hho_count + ho_count + o_count + dummy_count
        }
        self.dummy_node_split_dict = node_split_dict
        self.dummy_node_split_dict_path = Path(
            self.target_dir) / (Path(self.filename).stem + "_dummy_dict.txt")

    def _write_dummy_node_split_dict(self):
        if self._debug:
            self.ostream.print_info(
                f"Writing dummy node split dictionary to {self.dummy_node_split_dict_path}"
            )
            self.ostream.flush()
        with open(self.dummy_node_split_dict_path, "w") as f:
            for key, value in self.dummy_node_split_dict.items():
                f.write(f"{key:20} {value:<4}\n")
        self.ostream.print_info(
            f"Dummy {self.node_metal_type} node split dictionary saved to {self.dummy_node_split_dict_path}"
        )

    def _copy_node_pdb2target(self):
        target_path = Path(self.target_dir) / Path(self.filename).name
        if self._debug:
            self.ostream.print_info(
                f"Copying node pdb file to target directory: {target_path}")
            self.ostream.flush()
        with open(self.filename, "r") as src, open(target_path, "w") as dst:
            dst.write(src.read())
        self.ostream.print_info(f"Node pdb file copied to {target_path}")

    def create(self):
        self.check_dirs()
        self._nodepdb2xyz()
        self._nodepdb2G()
        self.pdbwriter._debug = self._debug
        if self.dummy_node:
            self._add_dummy_atoms_nodepdb()
            self._generate_dummy_node_split_dict()
            self.ostream.print_info("Adding dummy atoms to node...")
            self.ostream.flush()
            self._write_dummy_node_pdb()
            self._write_dummy_node_split_dict()
            if self._debug:
                self.ostream.print_info(
                    f"Final node graph has {self.sG.number_of_nodes()} nodes and {self.sG.number_of_edges()} edges."
                )
                self.ostream.flush()
        else:
            self.ostream.print_info(
                "No dummy atoms to add. Copying original node pdb file to target directory..."
            )
            self.ostream.flush()
            self._copy_node_pdb2target()

        #read the new pdb file to get self.node_data
        self.pdbreader.filepath = self.new_pdbfilename
        self.pdbreader.read_pdb()
        self.node_data = self.pdbreader.data
        self.ostream.print_info(
            f"Node processing completed. New pdb file at {self.new_pdbfilename}."
        )
        self.ostream.print_info(
            f"Node created from {self.filename} with metal type {self.node_metal_type}."
        )
        self.ostream.flush()
        # return self.sG, self.dummy_node_split_dict if self.dummy_node else None


if __name__ == "__main__":
    node_test = Node()
    node_test.filename = "tests/testdata/testnode.pdb"
    node_test.target_dir = "tests/testoutput"
    node_test.node_metal_type = "Zr"
    node_test.dummy_node = True
    node_test._debug = False
    node_test.create()
