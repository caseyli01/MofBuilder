import numpy as np
from pathlib import Path
from .basic import nn
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys

"""
atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
"""
class PdbReader:
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
        self.com_target_type = "X"
        self.data = None

        # debug
        self._debug = False

    def read_pdb(self):
        assert_msg_critical(
            Path(self.filepath).exists(),
            f"pdb file {self.filepath} not found")
        if self._debug:
            self.ostream.print_info(f"Reading pdb file {self.filepath}")

        inputfile = str(self.filepath)
        with open(inputfile, "r") as fp:
            lines = fp.readlines()

        data = []
        count=1
        for line in lines:
            line = line.strip()
            if len(line) > 0:  # skip blank line
                if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                    atom_type = line[12:16].strip()  # atom_label X1
                    value_x = float(line[30:38])  # x
                    value_y = float(line[38:46])  # y
                    value_z = float(line[46:54])  # z
                    atom_label = nn(line[67:80].strip())  # atom_note
                    atom_number = count  # atom_number
                    residue_name = line[17:20].strip()  # residue_name
                    residue_number = int(line[22:26]) if line[22:26].strip() != "" else 1 # residue_number
                    charge = float(line[54:60]) if line[54:60].strip() != "" else 0.0  # charge
                    spin = float(line[60:66]) if line[60:66].strip() != "" else 0  # spin
                    note = line[80:].strip()  # note
                    count+=1
                    data.append(
                        [atom_type, atom_label, atom_number, residue_name, residue_number, value_x, value_y, value_z, spin, charge, note])
        self.data = np.vstack(data)

    def process_node_pdb(self):
        # pdb only have cartesian coordinates
        self.read_pdb()
        node_atoms = self.data[:, 0:2]
        node_ccoords = self.data[:, 5:8]
        node_ccoords = node_ccoords.astype(float)
        com_type_indices = [
            i for i in range(len(node_atoms))
            if nn(node_atoms[i, 0]) == self.com_target_type
        ]
        x_indices = [
            j for j in range(len(node_atoms)) if nn(node_atoms[j, 0]) == "X"
        ]
        node_x_ccoords = self.data[x_indices, 5:8]
        node_x_ccoords = node_x_ccoords.astype(float)
        com_type_ccoords = node_ccoords[com_type_indices]
        com_type = np.mean(com_type_ccoords, axis=0)
        node_ccoords = node_ccoords - com_type
        node_x_ccoords = node_x_ccoords - com_type

        if self._debug:
            self.ostream.print_info(
                f"center of mass type {self.com_target_type} at {com_type}")

            self.ostream.print_info(f"number of atoms: {len(node_atoms)}")
            self.ostream.print_info(
                f"number of X atoms: {len(node_x_ccoords)}")

        self.node_atoms = node_atoms
        self.node_ccoords = node_ccoords
        self.node_x_ccoords = node_x_ccoords


if __name__ == "__main__":
    pdb = PdbReader(filepath="tests/testdata/testnode.pdb")
    pdb._debug = True
    pdb.read_pdb()
    print(pdb.data)
    pdb.process_node_pdb()

