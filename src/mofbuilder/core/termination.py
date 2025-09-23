import sys
from pathlib import Path
import numpy as np
import networkx as nx
import mpi4py.MPI as MPI

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.molecule import Molecule

from ..io.basic import nn
from ..io.pdb_reader import PdbReader
from ..io.pdb_writer import PdbWriter

class FrameTermination:
    """
    Handles the termination of the simulation.
    """
    def __init__(self, comm=None, ostream=None, filepath=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)
        self.properties = {}
        self.filename = filepath
        self.X_atom = "X"
        self.pdbreader = PdbReader(comm=self.comm, ostream=self.ostream)

        self._debug = False
        self.X_data = None
        self.termination_data = None
    

    def read_termination_file(self):
        if self.filename is None:
            return None
        assert_msg_critical(Path(self.filename).is_file(), f"Termination file {self.filename} does not exist.")
        if self._debug:
            self.ostream.print_info(f"Reading termination file {self.filename}")
        self.pdbreader.read_pdb(self.filename)
        self.termination_data = self.pdbreader.data
        if self._debug:
            self.ostream.print_info(f"Got {len(self.termination_data)} atoms from termination file.")
            self.ostream.flush()


    def extract_X_data(self):
        assert_msg_critical(self.termination_data is not None, "Termination data is None. Cannot extract X atoms.")
        indices = [i for i in range(len(self.termination_data)) if nn(self.termination_data[i, 0]) == self.X_atom]
        self.X_data = self.termination_data[indices]
        if self._debug:
            self.ostream.print_info(f"Extracted {len(self.X_data)} atoms of type {self.X_atom} for termination.")
            self.ostream.print_info(f"X_data: {self.X_data}")
            self.ostream.flush()

    def create(self):
        self.read_termination_file()
        if self.termination_data is not None:
            self.extract_X_data()


if __name__ == "__main__":
    term = FrameTermination(filepath="tests/testdata/testterm.pdb")
    term._debug = True
    term.create()
