import numpy as np
from pathlib import Path
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys
import re

"""
atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
"""

class XyzReader:
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

    #all the info to convert is atom_type,atom_label, atom_number, residue_name, residue_number, x, y, z, charge, comment
    def read_xyz(self, filepath=None, recenter=False, com_type=None, residue_name='MOL', residue_number=1):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        comment = lines[1].strip()
        # Extract the atom coordinates from the subsequent lines
        data = []
        for line in lines[2:]:
            if len(line.strip().split()) < 4:
                continue
            parts = line.split()
            atom_type = parts[0]
            atom_type = re.sub(r'\d', '', atom_type)  #remove digits
            atom_number = len(data) + 1
            atom_label = atom_type + str(atom_number)
            res_name = residue_name
            res_number = residue_number
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            if len(parts) > 4:
                charge = float(parts[4])
            else:
                charge = 0.0
            if len(parts) > 5:
                note = parts[5]
            else:
                note = ''
            spin = 1.00
            data.append((atom_type, atom_label, atom_number, res_name,
                          res_number, x, y, z, spin, charge, note))



        
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
