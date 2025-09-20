import re
import numpy as np
from pathlib import Path
from .basic import convert_fraction_to_decimal, remove_bracket, remove_quotes, remove_tail_number, extract_quote_lines
from .basic import find_keyword, extract_xyz_lines
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys
from ..core.framework import Net
"""
atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
"""
#this reader will read cif file and create a Topology object


class CifReader:

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

        self.filepath = filepath
        self.data = None

        # debug
        self._debug = False

    def _extract_value_str_slice(self, s):
        if len(s) == 0:
            return 0
        sign = 1
        mul_value = 1
        s_list = list(s[0])

        if "-" in s_list:
            sign = -1
        if "*" in s_list:
            mul_value = s_list[s_list.index("*") - 1]

        return sign * int(mul_value)

    def _extract_value_from_str(self, s):
        s = re.sub(r" ", "", s)  # remove space
        s = re.sub(r"(?<=[+-])", ",", s[::-1])[::-1]
        if s[0] == ",":
            s = s[1:]
        s_list = list(s.split(","))
        # find the slice of x
        x_slice = [s_list[i] for i in range(len(s_list)) if "x" in s_list[i]]
        y_slice = [s_list[i] for i in range(len(s_list)) if "y" in s_list[i]]
        z_slice = [s_list[i] for i in range(len(s_list)) if "z" in s_list[i]]
        # find the only digit in the slice no x,y,z
        const_slice = [
            s_list[i] for i in range(len(s_list)) if "x" not in s_list[i]
            and "y" not in s_list[i] and "z" not in s_list[i]
        ]
        # extract the coefficient and constant from slice
        # if * exist then use the value before *, if - exist then *-1
        x_coeff = self._extract_value_str_slice(x_slice)
        y_coeff = self._extract_value_str_slice(y_slice)
        z_coeff = self._extract_value_str_slice(z_slice)
        if len(const_slice) == 0:
            const = 0
        else:
            const = const_slice[0]
            const = convert_fraction_to_decimal(const)

        return x_coeff, y_coeff, z_coeff, const

    def _extract_transformation_matrix_from_symmetry_operator(self, expr_str):
        expr_str = str(expr_str)
        expr_str = expr_str.strip("\n")
        expr_str = expr_str.replace(" ", "")
        split_str = expr_str.split(",")
        transformation_matrix = np.zeros((4, 4))
        transformation_matrix[3, 3] = 1
        for i in range(len(split_str)):
            x_coeff, y_coeff, z_coeff, const = self._extract_value_from_str(
                split_str[i])
            transformation_matrix[i] = [x_coeff, y_coeff, z_coeff, const]

        return transformation_matrix

    def _extract_symmetry_operation_from_lines(self, symmetry_sector):
        symmetry_operation = []
        for i in range(len(symmetry_sector)):
            # Regular expression to match terms with coefficients and variables
            pattern = r"([+-]?\d*\.?\d*)\s*([xyz])"  # at least find a x/-x/y/-y/z/-z
            match = re.search(pattern, symmetry_sector[i])
            if match:
                string = remove_quotes(symmetry_sector[i].strip("\n"))
                no_space_string = string.replace(" ", "")
                symmetry_operation.append(no_space_string)
        if len(symmetry_operation) < 2:
            if self._debug:
                self.ostream.print_info("P1 cell")
                self.ostream.flush()
            symmetry_operation = ["x,y,z"]
        else:
            if self._debug:
                self.ostream.print_info(f"apply {len(symmetry_operation)}  symmetry operation")
                self.ostream.flush()

        return symmetry_operation

    def read_cif(self, cif_file=None):
        if cif_file is not None:
            self.filepath = cif_file
        assert_msg_critical(
            Path(self.filepath).exists(),
            f"cif file {self.filepath} not found")
        if self._debug:
            self.ostream.print_info(f"Reading cif file {self.filepath}")
            self.ostream.flush()

        with open(self.filepath, "r") as f:
            lines = f.readlines()
            nonempty_lines = [line for line in lines if line.strip()]

        first_line = nonempty_lines[0]
        if re.search(r"^data_", first_line):
            self.net_name = first_line.split()[0].split("_")[1]
            self.ostream.print_info(f"Found net name: {self.net_name}")
            self.ostream.flush()
            self.spacegroup = first_line.split()[0].split("_")[2]
            self.hall_number = re.search(r"hall_number:\s*(\d+)",
                                         first_line)[1]
            self.V_con = re.search(r"V_con:\s*(\d+)", first_line)[1]
            self.ostream.print_info(f"Spacegroup: {self.spacegroup}")
            self.ostream.print_info(f"Hall_number: {self.hall_number}")
            if self._debug:
                self.ostream.print_info(f"Found V_con: {self.V_con}")
            self.ostream.flush()
            if re.search(r"EC_con:\s*(\d+)", first_line):
                self.EC_con = re.search(r"EC_con:\s*(\d+)", first_line)[1]
                if self._debug:
                    self.ostream.print_info(f"Found EC_con: {self.EC_con}")
                    self.ostream.flush()
            else:
                self.EC_con = None
                self.ostream.print_info(f"This net is for ditopic linker")
                self.ostream.flush()
        else:
            self.ostream.print_info(f"Not found net name")
            self.ostream.flush()

        # nonempty_lines=lines
        keyword1 = r"loop_"
        keyword2 = r"x,\s*y,\s*z"
        keyword3 = r"-x"
        # find the symmetry sector begin with x,y,z, beteen can have space or tab and comma,but just x start, not '-x'
        # keyword2 = "x,\s*y,\s*z"

        loop_key = []
        loop_key.append(0)
        linenumber = 0
        for i in range(
                len(nonempty_lines)):  # search for keywords and get linenumber
            # m is find keywor1 or (find keyword2 without keyword3)
            m = find_keyword(keyword1, nonempty_lines[i]) or (
                find_keyword(keyword2, nonempty_lines[i]) and
                (not find_keyword(keyword3, nonempty_lines[i])))

            a = re.search(r"_cell_length_a", nonempty_lines[i])
            b = re.search(r"_cell_length_b", nonempty_lines[i])
            c = re.search(r"_cell_length_c", nonempty_lines[i])
            alpha = re.search(r"_cell_angle_alpha", nonempty_lines[i])
            beta = re.search(r"_cell_angle_beta", nonempty_lines[i])
            gamma = re.search(r"_cell_angle_gamma", nonempty_lines[i])

            if m:
                loop_key.append(linenumber)
            # if not nonempty_lines[i].strip():
            #    loop_key.append(linenumber)

            else:
                if a:
                    cell_length_a = remove_bracket(
                        nonempty_lines[i].split()[1])
                elif b:
                    cell_length_b = remove_bracket(
                        nonempty_lines[i].split()[1])
                elif c:
                    cell_length_c = remove_bracket(
                        nonempty_lines[i].split()[1])
                elif alpha:
                    cell_angle_alpha = remove_bracket(
                        nonempty_lines[i].split()[1])
                elif beta:
                    cell_angle_beta = remove_bracket(
                        nonempty_lines[i].split()[1])
                elif gamma:
                    cell_angle_gamma = remove_bracket(
                        nonempty_lines[i].split()[1])

            linenumber += 1
        loop_key.append(len(nonempty_lines))
        loop_key = list(set(loop_key))
        loop_key.sort()

        cell_info = [
            cell_length_a,
            cell_length_b,
            cell_length_c,
            cell_angle_alpha,
            cell_angle_beta,
            cell_angle_gamma,
        ]

        # find symmetry sectors and atom_site_sectors
        cif_sectors = []
        for i in range(len(loop_key) - 1):
            cif_sectors.append(nonempty_lines[loop_key[i]:loop_key[i + 1]])
        for i in range(
                len(cif_sectors)):  # find '\s*x,\s*y,\s*z' symmetry sector
            if re.search(keyword2, cif_sectors[i][0]):
                symmetry_sector = cif_sectors[i]

            if len(cif_sectors[i]) > 1:
                if re.search(r"_atom_site_label\s+",
                             cif_sectors[i][1]):  # line0 is _loop
                    atom_site_sector = cif_sectors[i]
        self.cell_info = cell_info
        self.symmetry_sector = symmetry_sector
        self.atom_site_sector = atom_site_sector

    def _limit_value_0_1(self, new_array_metal_xyz):
        # use np.mod to limit the value in [0,1]
        new_array_metal_xyz = np.mod(new_array_metal_xyz, 1)
        return new_array_metal_xyz

    def _wrap_fccords_to_0_1(self, fccords):
        #to make all atoms are around com
        fccords = np.unique(np.array(fccords, dtype=float), axis=0)
        fccords = self._limit_value_0_1(fccords)
        fccords += 0.5
        fccords = self._limit_value_0_1(fccords)
        fccords += -0.5
        #remove duplicate fcoords
        fccords = np.unique(np.array(fccords, dtype=float), axis=0)
        return fccords

    def _apply_sym_operator(self, symmetry_operations, array_metal_xyz):
        array_metal_extend_xyz = np.hstack(
            (array_metal_xyz, np.ones((len(array_metal_xyz), 1))))
        cell_array_metal_xyz = np.empty((0, 3))
        for sym_line_idx in range(len(symmetry_operations)):
            transfromation_matrix = self._extract_transformation_matrix_from_symmetry_operator(
                symmetry_operations[sym_line_idx])
            # print(
            #    f"symmetry_operations[sym_line_idx],{symmetry_operations[sym_line_idx]}\n,transfromation_matrix{transfromation_matrix}"
            # )

            new_extend_xyz = np.matmul(transfromation_matrix,
                                       array_metal_extend_xyz.T).T
            new_xyz = new_extend_xyz[:, 0:3]
            cell_array_metal_xyz = np.vstack((cell_array_metal_xyz, new_xyz))

        round_cell_array_metal_xyz = np.round(
            self._wrap_fccords_to_0_1(cell_array_metal_xyz), 4)
        _, unique_indices = np.unique(round_cell_array_metal_xyz,
                                      axis=0,
                                      return_index=True)
        unique_indices.sort()
        unique_metal_array = round_cell_array_metal_xyz[unique_indices]

        return unique_metal_array, unique_indices

    def _extract_atoms_fcoords_from_lines(self, atom_site_sector):
        atom_site_lines = []
        keyword = r"_"
        for line in atom_site_sector:  # search for keywords and get linenumber
            m = re.search(keyword, line)
            if m is None:
                atom_site_lines.append(line)

        #array_atom is atom_type and atom_label
        #array_xyz is fractional coordinates
        array_atom = np.zeros((len(atom_site_lines), 2), dtype=object)
        array_xyz = np.zeros((len(atom_site_lines), 3))

        for i in range(len(atom_site_lines)):
            for j in [0, 1, 2, 3, 4]:  # only need atom type, atom label, x,y,z
                if j < 2:
                    array_atom[i, j] = remove_tail_number(
                        atom_site_lines[i].split()[j])
                else:
                    array_xyz[i, (j - 2)] = remove_bracket(
                        atom_site_lines[i].split()[j])
        if self._debug:
            self.ostream.print_info(
                f"Found {len(array_atom)} atoms in atom_site_sector")
            self.ostream.print_info(f"array_atom {array_atom}")
            self.ostream.print_info(f"array_xyz {array_xyz}")
            self.ostream.flush()
        return array_atom, array_xyz

    def get_type_atoms_fcoords_in_primitive_cell(self, target_type=None):
        """
        need to read cif file first using read_cif
        """
        array_atom, array_xyz = self._extract_atoms_fcoords_from_lines(
            self.atom_site_sector)
        if target_type is None:
            self.ostream.print_info(
                f"target_type not specified, use {target_type} as default")
            self.ostream.flush()
        if len(self.symmetry_sector) > 1:  # need to apply symmetry operations
            array_metal_xyz = array_xyz[array_atom[:, 0] == target_type]
            array_metal_xyz = np.round(array_metal_xyz, 4)
            symmetry_sector_neat = extract_quote_lines(self.symmetry_sector)
            if len(symmetry_sector_neat) < 2:  # if no quote, then find x,y,z
                symmetry_sector_neat = extract_xyz_lines(self.symmetry_sector)
            symmetry_operations = self._extract_symmetry_operation_from_lines(
                symmetry_sector_neat)
            no_sym_array_metal_xyz, no_sym_indices = self._apply_sym_operator(
                symmetry_operations, array_metal_xyz)
            array_metal_xyz_final = no_sym_array_metal_xyz

        else:
            array_metal_xyz = array_xyz[array_atom[:, 0] == target_type]
            array_metal_xyz_final = np.round(array_metal_xyz, 4)

        self.fcoords = self._wrap_fccords_to_0_1(array_metal_xyz_final)
        self.target_fcoords = self._wrap_fccords_to_0_1(array_metal_xyz_final)

        #make data
        self.data = []
        for i in range(len(self.fcoords)):
            atom_number = i + 1
            atom_type = array_atom[i, 0] + str(atom_number)
            atom_label = array_atom[i, 1]
            residue_name = 'MOL'
            residue_number = 1
            x = self.fcoords[i, 0]
            y = self.fcoords[i, 1]
            z = self.fcoords[i, 2]
            spin = 1.00
            charge = 0.0
            note = ''
            self.data.append([
                atom_type, atom_label, atom_number, residue_name,
                residue_number, x, y, z, spin, charge, note
            ])
        self.data = np.vstack(self.data)
        if self._debug:
            self.ostream.print_info(
                f"Found {len(self.fcoords)} {target_type} atoms in primitive cell"
            )
            self.ostream.print_info(f"fcoords {self.fcoords}")
            self.ostream.flush()

        return self.cell_info, self.data, self.target_fcoords


if __name__ == "__main__":
    cif_file = "tests/testdata/test.cif"
    cif_reader = CifReader(filepath=cif_file)
    cif_reader._debug = True
    cif_reader.read_cif()
    cif_reader.get_type_atoms_fcoords_in_primitive_cell(target_type="V")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(cif_reader.target_fcoords)
        print(cif_reader.cell_info)

    cif_reader.get_type_atoms_fcoords_in_primitive_cell(target_type="E")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(cif_reader.target_fcoords)
        print("*" * 50)
        print(cif_reader.cell_info)
