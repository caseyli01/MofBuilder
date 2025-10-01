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


class GroReader:

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

    def read_gro(self, filepath=None, recenter=False, com_type=None):
        if filepath is not None:
            self.filepath = filepath
        assert_msg_critical(
            Path(self.filepath).exists(),
            f"gro file {self.filepath} not found")
        if self._debug:
            self.ostream.print_info(f"Reading gro file {self.filepath}")

        inputfile = str(self.filepath)
        with open(inputfile, "r") as fp:
            lines = fp.readlines()

        data = []
        count = 1

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines[2:]:
            if len(line.strip()) < 4:
                continue
            #read the line by %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            residue_number = int(line[0:5].strip())
            residue_name = line[5:10].strip()
            atom_label = line[10:15].strip()
            atom_number = int(line[15:20].strip())
            x = float(line[20:28]) * 10
            y = float(line[28:36]) * 10
            z = float(line[36:44]) * 10
            atom_type = line[44:46].strip()
            spin = 1.00
            charge = 0.0
            note = ''
            data.append((atom_type, atom_label, atom_number, residue_name,
                         residue_number, x, y, z, spin, charge, note))

        data = []
        count = 1
        for line in lines:
            line = line.strip()
            if len(line.strip()) < 4:
                continue
            residue_number = int(line[0:5].strip())
            residue_name = line[5:10].strip()
            atom_label = line[10:15].strip()
            atom_number = count
            x = float(line[20:28]) * 10
            y = float(line[28:36]) * 10
            z = float(line[36:44]) * 10
            atom_type = line[44:46].strip()
            charge = 0.0  # default charge
            spin = 1.00  # default spin
            note = nn(atom_type)  # note
            count += 1
            data.append([
                atom_type, atom_label, atom_number, residue_name,
                residue_number, x, y, z, spin, charge, note
            ])
        self.data = np.vstack(data)

        #should set type of array elements
        def type_data(arr):
            arr[:, 2] = arr[:, 2].astype(int)
            arr[:, 4] = arr[:, 4].astype(int)
            arr[:, 5:8] = arr[:, 5:8].astype(float)
            arr[:, 8] = arr[:, 8].astype(float)
            arr[:, 9] = arr[:, 9].astype(float)
            return arr

        self.data = type_data(self.data)

        if recenter:
            #if not define com_type, use all atoms to calculate com
            #else use the specified atom type to calculate com
            if com_type is None:
                com_type_ccoords = self.data[:, 5:8].astype(float)
            else:
                if com_type not in self.data[:, -1]:
                    self.ostream.print_warning(
                        f"com_type {com_type} not in the pdb file, use all atoms to calculate com"
                    )
                    com_type_ccoords = self.data[:, 5:8].astype(float)
                else:
                    com_type_ccoords = self.data[
                        self.data[:, -1] == com_type][:, 5:8].astype(float)
            com = np.mean(com_type_ccoords, axis=0)
            if self._debug:
                self.ostream.print_info(
                    f"Center of mass type {com_type} at {com}")
            self.data[:, 5:8] = self.data[:, 5:8].astype(float) - com
