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

from pathlib import Path

from ..io.basic import pname, is_list_A_in_B, nn
from ..io.pdb_reader import PdbReader
from ..io.pdb_writer import PdbWriter
from ..io.gro_writer import GroWriter
from ..io.xyz_writer import XyzWriter
from ..io.cif_writer import CifWriter
from ..utils.geometry import cartesian_to_fractional, fractional_to_cartesian
from .superimpose import superimpose_rotation_only


class MofWriter:

    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)

        #write the eG to the file
        #need to be set before use
        self.filename = "mofbuilder_output.pdb"
        self.G = None
        self.frame_cell_info = None  #a list of 6 elements, a,b,c,alpha,beta,gamma
        self.supercell_boundary = None  #a list of 6 elements, x_min,x_max,y_min,y_max,z_min,z_max
        self.sc_unit_cell = None  #3x3 matrix of the supercell unit cell
        self.xoo_dict = None  #dict of xoo atom indices in the edge
        self.dummy_atom_node_dict = None  #dict of dummy atom counts in the node

        self.target_directory = None  #target directory to save the output files

        self.merged_data = None  #merged data of nodes, edges, terms
        self.merged_f_data = None  #merged fractional data of nodes, edges
        self._debug = False  #debug mode

    def _remove_xoo_from_node(self, G, xoo_dict):
        """
        remove the XOO atoms from the node after adding the terminations, add ['noxoo_f_points'] to the node in eG
        """
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

    def convert_graph_to_data(self, G, sc_unit_cell):
        #convert the graph to the data array and anlyze the residues
        rG = self._remove_xoo_from_node(G, self.xoo_dict)

        def arr2data(arr, residue_name=None, residue_number=None, note=None):
            #arr type is [atom_type,atom_label,x,y,z]
            if arr is None or len(arr) == 0:
                return None, None
            if isinstance(arr, list):
                arr = np.vstack(arr)

            data = []
            for i in range(len(arr)):
                atom_type = arr[i, 0]
                atom_label = arr[i, 1]
                value_x = float(arr[i, 2])
                value_y = float(arr[i, 3])
                value_z = float(arr[i, 4])
                atom_number = i + 1
                residue_name = "MOL" if residue_name is None else residue_name
                residue_number = 1 if residue_number is None else residue_number
                charge = 0.0
                spin = 0
                note = nn(atom_type) if note is None else note
                data.append([
                    atom_type, atom_label, atom_number, residue_name,
                    residue_number, value_x, value_y, value_z, spin, charge,
                    note
                ])
            data = np.vstack(data)
            return data

        def get_node_data(n, G, sc_unit_cell):
            node_f_points = G.nodes[n]["noxoo_f_points"]
            res_name = G.nodes[n]["name"]
            res_idx = G.nodes[n]["index"]
            node_f_coords = node_f_points[:, 2:5]  # fractional coordinates
            c_coords = fractional_to_cartesian(
                node_f_coords, sc_unit_cell)  # Cartesian coordinates
            node_arr = np.hstack((node_f_points[:, 0:2], c_coords))
            node_data = arr2data(node_arr,
                                 residue_name=res_name,
                                 residue_number=res_idx,
                                 note=n)
            return node_data

        def get_edge_data(n, G, sc_unit_cell):
            edge_f_points = np.vstack(
                (G.nodes[n]["f_points"], G.nodes[n]["xoo_f_points"]))
            res_name = G.nodes[n]["name"]
            res_idx = G.nodes[n]["index"]
            edge_f_coords = edge_f_points[:, 2:5]  # fractional coordinates
            c_coords = fractional_to_cartesian(
                edge_f_coords, sc_unit_cell)  # Cartesian coordinates
            edge_arr = np.hstack((edge_f_points[:, 0:2], c_coords))
            edge_data = arr2data(edge_arr,
                                 residue_name=res_name,
                                 residue_number=res_idx,
                                 note=n)
            return edge_data

        cG = self._remove_xoo_from_node(G, self.xoo_dict)
        count = 0
        term_count = 0
        nodes_data = []
        terms_data = []
        edges_data = []
        for n in cG.nodes():
            if pname(n) != "EDGE":
                node_data = get_node_data(n, rG, sc_unit_cell)
                cG.nodes[n]["data"] = node_data
                nodes_data.append(node_data)
                #check if the node have terminations
                if "term_c_points" in cG.nodes[n]:
                    for term_ind_key, c_positions in cG.nodes[n][
                            "term_c_points"].items():
                        term_name = "T" + cG.nodes[n]["name"]
                        term_count -= 1
                        term_data = arr2data(c_positions,
                                             residue_name=term_name,
                                             residue_number=term_count,
                                             note=term_ind_key)
                        terms_data.append(term_data)
            elif pname(n) == "EDGE":
                edge_data = get_edge_data(n, cG, sc_unit_cell)
                cG.nodes[n]["data"] = edge_data
                edges_data.append(edge_data)

        self.nodes_data = nodes_data
        self.edges_data = edges_data
        self.terms_data = terms_data 
        self.cG = cG

    def get_merged_data(self, dummy_atom_node_dict=None):
        #merge the nodes, edges, terms data to a single array
        #rename the dummy atom names if dummy_atom_node_dict is provided
        nodes_data = self.nodes_data
        edges_data = self.edges_data
        terms_data = self.terms_data
        if dummy_atom_node_dict is not None:
            nodes_data = self._rename_node_name(nodes_data,
                                                dummy_atom_node_dict)
        else:
            nodes_data = np.vstack(nodes_data)
        edges_data = np.vstack(edges_data) if len(
            edges_data) > 0 else np.empty((0, 11))
        terms_data = np.vstack(terms_data) if len(
            terms_data) > 0 else np.empty((0, 11))

        merged_data = np.vstack(
            (nodes_data, edges_data, terms_data
             )) if len(edges_data) > 0 or len(terms_data) > 0 else nodes_data
        self.merged_data = merged_data
        return merged_data


    def convert_graph_to_fcoords_data(self, G, supercell_boundary):
        #convert the graph to the data array and anlyze the residues
        rG = self._remove_xoo_from_node(G, self.xoo_dict)

        def arr2data(arr, residue_name=None, residue_number=None, note=None):
            #arr type is [atom_type,atom_label,x,y,z]
            if arr is None or len(arr) == 0:
                return None, None
            if isinstance(arr, list):
                arr = np.vstack(arr)

            data = []
            for i in range(len(arr)):
                atom_type = arr[i, 0]
                atom_label = arr[i, 1]
                value_x = float(arr[i, 2])
                value_y = float(arr[i, 3])
                value_z = float(arr[i, 4])
                atom_number = i + 1
                residue_name = "MOL" if residue_name is None else residue_name
                residue_number = 1 if residue_number is None else residue_number
                charge = 0.0
                spin = 0
                note = nn(atom_type) if note is None else note
                data.append([
                    atom_type, atom_label, atom_number, residue_name,
                    residue_number, value_x, value_y, value_z, spin, charge,
                    note
                ])
            data = np.vstack(data)
            return data

        def get_node_fcoords_data(n, G):
            node_f_points = G.nodes[n]["noxoo_f_points"] # fractional coordinates
            res_name = G.nodes[n]["name"]
            res_idx = G.nodes[n]["index"]
            node_data = arr2data(node_f_points,
                                 residue_name=res_name,
                                 residue_number=res_idx,
                                 note=n)
            return node_data

        def get_edge_fcoords_data(n, G):
            edge_f_points = np.vstack(
                (G.nodes[n]["f_points"], G.nodes[n]["xoo_f_points"]))
            res_name = G.nodes[n]["name"]
            res_idx = G.nodes[n]["index"]
            edge_data = arr2data(edge_f_points,
                                 residue_name=res_name,
                                 residue_number=res_idx,
                                 note=n)
            return edge_data
        
        def check_supercell_box_range(f_coords, box):
            #box is a list of 6 elements [x_min,x_max,y_min,y_max,z_min,z_max]
            if f_coords.ndim == 2: # ditopic linker saved two connected nodes fcoords 
                f_coords = np.mean(f_coords, axis=0)
            x, y, z = map(float, f_coords)
            box_xmax, box_ymax, box_zmax = map(float, box)
            if x < 0 or x >= box[0]:
                return False
            if y < 0 or y >= box[1]:
                return False
            if z < 0 or z >= box[2]:
                return False
            self.ostream.print_info(f"f_coords: {f_coords}, box: {box}")
            self.ostream.flush()
            return True

        cG = self._remove_xoo_from_node(G, self.xoo_dict)
        count = 0
        term_count = 0
        nodes_data = []
        terms_data = []
        edges_data = []
        for n in cG.nodes():
            if pname(n) != "EDGE":
                if not check_supercell_box_range(cG.nodes[n]["fcoords"], supercell_boundary):
                    continue
                node_f_data = get_node_fcoords_data(n, rG)
                cG.nodes[n]["f_data"] = node_f_data
                self.ostream.print_info(f"node {n} f_data shape: {node_f_data.shape}")
                nodes_data.append(node_f_data)
                #check if the node have terminations

            elif pname(n) == "EDGE":
                if not check_supercell_box_range(cG.nodes[n]["fcoords"], supercell_boundary):
                    continue
                edge_f_data = get_edge_fcoords_data(n, cG)
                self.ostream.print_info(f"edge {n} f_data shape: {edge_f_data.shape}")
                self.ostream.flush()
                cG.nodes[n]["f_data"] = edge_f_data
                edges_data.append(edge_f_data)

        self.nodes_f_data = np.vstack(nodes_data)
        self.edges_f_data = np.vstack(edges_data)
        self.cG = cG

    def get_merged_fcoords_data(self):
        #merge the nodes, edges to a single array
        #rename the dummy atom names if dummy_atom_node_dict is provided
        nodes_f_data = np.vstack(self.nodes_f_data) if len(self.nodes_f_data)>0 else np.empty((0,11))
        edges_f_data = np.vstack(self.edges_f_data) if len(self.edges_f_data)>0 else np.empty((0,11))


        merged_f_data = np.vstack(
            (nodes_f_data, edges_f_data
             )) if len(edges_f_data) > 0 else nodes_f_data
        self.merged_f_data = merged_f_data
        return merged_f_data



    def write_pdb(self, skip_merge=False):
        if self.merged_data is None:
            skip_merge = False

        if skip_merge:
            merged_data = self.merged_data
        else:
            self.convert_graph_to_data(self.G, self.sc_unit_cell)
            merged_data = self.get_merged_data(self.dummy_atom_node_dict)
            self.merged_data = merged_data

        pdb_writer = PdbWriter(comm=self.comm, ostream=self.ostream)
        header = "REMARK   Generated by MOFbuilder\n"
        filename = self.filename
        if self.target_directory is not None:
            filename = str(Path(self.target_directory, filename))
        if self._debug:
            self.ostream.print_info(
                f"targeting directory: {self.target_directory}")
            self.ostream.print_info(f"writing pdb file to: {filename}")
        pdb_writer.write(filepath=filename, header=header, lines=merged_data)

    def write_xyz(self, skip_merge=False):
        if self.merged_data is None:
            skip_merge = False

        if skip_merge:
            merged_data = self.merged_data
        else:
            self.convert_graph_to_data(self.G, self.sc_unit_cell)
            merged_data = self.get_merged_data(self.dummy_atom_node_dict)
            self.merged_data = merged_data

        xyz_writer = XyzWriter(comm=self.comm, ostream=self.ostream)
        header = "Generated by MOFbuilder\n"
        filename = self.filename
        if self.target_directory is not None:
            filename = str(Path(self.target_directory, filename))
        if self._debug:
            self.ostream.print_info(
                f"targeting directory: {self.target_directory}")
            self.ostream.print_info(f"writing xyz file to: {filename}")
        xyz_writer.write(filepath=filename, header=header, lines=merged_data)

    def write_gro(self, skip_merge=False):
        if self.merged_data is None:
            skip_merge = False
        if skip_merge:
            merged_data = self.merged_data
        else:
            self.convert_graph_to_data(self.G, self.sc_unit_cell)
            merged_data = self.get_merged_data(self.dummy_atom_node_dict)
            self.merged_data = merged_data

        gro_writer = GroWriter(comm=self.comm, ostream=self.ostream)
        header = "Generated by MOFbuilder\n"
        filename = self.filename
        if self.target_directory is not None:
            filename = str(Path(self.target_directory, filename))
        if self._debug:
            self.ostream.print_info(
                f"targeting directory: {self.target_directory}")
            self.ostream.print_info(f"writing gro file to: {filename}")
        gro_writer.write(filepath=filename,
                         header=header,
                         lines=merged_data,
                         box=self.frame_cell_info)
    def write_cif(self,skip_merge=False,supercell_boundary=None,frame_cell_info=None):
        #cif file use f_coords so no terminations needed and boundary box to filter
        if frame_cell_info is None:
            frame_cell_info = self.frame_cell_info
        if supercell_boundary is None:
            supercell_boundary = self.supercell_boundary
        if self.merged_f_data is None:
            skip_merge = False
        if skip_merge:
            merged_f_data = self.merged_f_data
        else:
            self.convert_graph_to_fcoords_data(self.G, supercell_boundary)
            merged_f_data = self.get_merged_fcoords_data()
            self.merged_f_data = merged_f_data

        assert_msg_critical(frame_cell_info is not None, "frame_cell_info is not provided for cif writing")
        assert_msg_critical(supercell_boundary is not None, "supercell_boundary is not provided for cif writing")


        cif_writer = CifWriter(comm=self.comm, ostream=self.ostream)
        header = "Generated by MOFbuilder\n"
        filename = self.filename
        if self.target_directory is not None:
            filename = str(Path(self.target_directory, filename))
        if self._debug:
            self.ostream.print_info(
                f"targeting directory: {self.target_directory}")
            self.ostream.print_info(f"writing cif file to: {filename}")
        header = "Generated by MOFbuilder\n"
        cif_writer.write(filepath=filename,
                         header=header,
                         lines=self.merged_f_data,
                         cell_info=frame_cell_info,
                         supercell_boundary=supercell_boundary)


    def _rename_node_name(self, nodes_data, dummy_atom_node_dict):
        if dummy_atom_node_dict is None:
            return np.vstack(nodes_data)
        nodes_num = len(nodes_data)
        metal_count = dummy_atom_node_dict["METAL_count"]
        dummy_res_len = int(dummy_atom_node_dict["dummy_res_len"])
        hho_count = dummy_atom_node_dict["HHO_count"]
        ho_count = dummy_atom_node_dict["HO_count"]
        o_count = dummy_atom_node_dict["O_count"]
        #number for slice
        metal_num = metal_count * dummy_res_len
        hho_num = hho_count * 3
        ho_num = ho_count * 2
        o_num = o_count * 1

        # generate new_name_list for all dummy atoms in order
        # For each residue, repeat the name for the number of atoms in that residue, incrementing the residue index
# Build tuples of (name, count) for each residue type
        residue_specs = (
            [("METAL", dummy_res_len)] * metal_count +
            [("HHO", 3)] * hho_count +
            [("HO", 2)] * ho_count +
            [("O", 1)] * o_count
        )

        # Use list comprehension with enumerate for fast generation
        new_name_list = [
            f"{name}_{i+1}"
            for i, (name, count) in enumerate(residue_specs)
            for _ in range(count)
        ]
        nodes_data = np.vstack(nodes_data)
        if self._debug:
            self.ostream.print_info(
                f"dummy node split dict: {dummy_atom_node_dict}")
            self.ostream.print_info(f"new_name_list: {new_name_list}")

        name_col = np.tile(new_name_list, nodes_num).reshape(-1, 1)
        #hstack the new_name_col to the stacked array, replacing the original name column
        rename_data = np.hstack((nodes_data[:, 0:3], name_col, nodes_data[:,
                                                                          4:]))
        return rename_data



 

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

    