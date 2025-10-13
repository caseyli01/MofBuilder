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

    def read_pdb(self,filepath=None,recenter=False,com_type=None):
        if filepath is not None:
            self.filepath = filepath
        assert_msg_critical(
            Path(self.filepath).exists(),
            f"pdb file {self.filepath} not found")
        if self._debug:
            self.ostream.print_info(f"Reading pdb file {self.filepath}")

        inputfile = str(self.filepath)
        with open(inputfile, "r") as fp:
            lines = fp.readlines()

        data = []
        #count=1
        for line in lines:
            line = line.strip()
            if len(line) > 0:  # skip blank line
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_number    = int(line[6:11]) if line[6:11].strip() else 1  # atom serial
                    atom_type      = line[12:16].strip()  # atom name (e.g. "X1")
                    residue_name   = line[17:20].strip()  # residue name
                    chain_id       = line[21].strip() if line[21].strip() else "A"  # chain
                    residue_number = int(line[22:26]) if line[22:26].strip() else 1  # residue number
                    value_x        = float(line[30:38])
                    value_y        = float(line[38:46])
                    value_z        = float(line[46:54])
                    occupancy      = float(line[54:60]) if line[54:60].strip() else 0.0
                    b_factor       = float(line[60:66]) if line[60:66].strip() else 0.0
                    atom_label     = line[76:78].strip()  # element symbol (e.g. "C")
                    charge         = line[78:80].strip()  # formal charge (string like "2+")

                    # custom mapping
                    note = nn(atom_type)

                    data.append([
                        atom_type,       # assigned atom name, e.g. "CA"
                        atom_label,      # element symbol, e.g. "C"
                        atom_number,     # serial number
                        residue_name,    # residue name
                        residue_number,  # residue index
                        value_x, value_y, value_z,
                        occupancy,       # occupancy
                        b_factor,       # B-factor
                        note             # custom mapping
                    ])
        
        
        self.data = np.vstack(data)
        #should set type of array elements
        def type_data(arr):
            arr[:,2] = arr[:,2].astype(int)
            arr[:,4] = arr[:,4].astype(int)
            arr[:,5:8] = arr[:,5:8].astype(float)
            arr[:,8] = arr[:,8].astype(float)
            arr[:,9] = arr[:,9].astype(float)
            return arr
        self.data = type_data(self.data)

        if recenter:
            #if not define com_type, use all atoms to calculate com
            #else use the specified atom type to calculate com
            if com_type is None:
                com_type_ccoords = self.data[:,5:8].astype(float)
            else:
                if com_type not in self.data[:,-1]:
                    self.ostream.print_warning(f"com_type {com_type} not in the pdb file, use all atoms to calculate com")
                    com_type_ccoords = self.data[:,5:8].astype(float)
                else:
                    com_type_ccoords = self.data[self.data[:,-1]==com_type][:,5:8].astype(float)
            com = np.mean(com_type_ccoords, axis=0)
            if self._debug:
                self.ostream.print_info(f"Center of mass type {com_type} at {com}")
            self.data[:, 5:8] = self.data[:, 5:8].astype(float) - com

        self.X_data = self.data[self.data[:,-1]=='X']

    def expand_arr2data(self, arr):
        #arr type is [atom_type,atom_label,x,y,z]
        if arr is None or len(arr) == 0:
            return None,None
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
            residue_name = "MOL"
            residue_number = 1
            charge = 0.0
            spin = 0
            note = nn(atom_type)
            data.append(
                [atom_type, atom_label, atom_number, residue_name, residue_number, value_x, value_y, value_z, spin,
                 charge, note])
        data = np.vstack(data)
        X_data = data[data[:,-1]=='X']
        return data,X_data

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

