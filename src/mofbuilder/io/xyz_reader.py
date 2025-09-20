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
    def read_xyz(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        comment = lines[1].strip()
        # Extract the atom coordinates from the subsequent lines
        atom_data = []
        for line in lines[2:]:
            if len(line.strip().split()) < 4:
                continue
            parts = line.split()
            atom_type = parts[0]
            atom_type = re.sub(r'\d', '', atom_type)  #remove digits
            atom_number = len(atom_data) + 1
            atom_label = atom_type + str(atom_number)
            residue_name = self.residue_name
            residue_number = self.residue_number
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
            atom_data.append((atom_type, atom_label, atom_number, residue_name,
                              residue_number, x, y, z, spin, charge, note))
        self.comment = self.comment + ' ' + comment
        # Convert the list of tuples to a numpy array
        atom_data = self._settype_atomdata(atom_data)
        self.atom_data = atom_data
        return atom_data