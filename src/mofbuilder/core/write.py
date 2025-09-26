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
        self.sc_unit_cell = None  #3x3 matrix of the supercell unit cell
        self.xoo_dict = None  #dict of xoo atom indices in the edge
        self.dummy_atom_node_dict = None  #dict of dummy atom counts in the node

        self.target_directory = None  #target directory to save the output files

        self.merged_data = None  #merged data of nodes, edges, terms
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


def convert_node_array_to_gro_lines(array, line_num_start, res_num_start):
    formatted_gro_lines = []

    for i in range(len(array)):
        line = array[i]
        ind_inres = i + 1
        name = line[1]
        value_atom_number_in_gro = int(ind_inres +
                                       line_num_start)  # atom_number
        value_label = re.sub(r"\d", "", line[2]) + str(ind_inres)  # atom_label
        value_resname = str(name)[
            0:3]  # +str(eG.nodes[n]['index'])  # residue_name
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
            node_xoo_cvecs = (
                node_xoo_ccoords[:, 2:5].astype(float) - node_oo_center_cvec
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
