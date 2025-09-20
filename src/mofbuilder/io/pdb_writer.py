from pathlib import Path
import sys
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
import mpi4py.MPI as MPI
from veloxchem.errorhandler import assert_msg_critical

class PdbWriter:    

    def __init__(self, comm=None, ostream=None, filepath=None, debug=False):
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
        self._debug = debug


    def write(self, filepath=None, header='', lines=[]):
        """
        line format:
        atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
        1         2    3      4            5              6  7  8 9    10 11
        ATOM      1    C       MOL          1            1.000 2.000 3.000 1.00 0.00 C1
        """
        "data format[atom_type, atom_label, atom_number, residue_name, residue_number, value_x, value_y, value_z, spin, charge, note]"
        filepath = Path(filepath) if filepath is not None else Path(self.filepath)
        assert_msg_critical(filepath is not None, "pdb filepath is not specified")
        # check if the file directory exists and create it if it doesn't
        self.file_dir = Path(filepath).parent
        if self._debug:
            self.ostream.print_info(f"targeting directory: {self.file_dir}")
        self.file_dir.mkdir(parents=True, exist_ok=True)

        if filepath.suffix != ".pdb":
            filepath = filepath.with_suffix(".pdb")
        
        newpdb = []
        newpdb.append(header)

        with open(filepath, "w") as fp:
            # Iterate over each line in the input file
            for i in range(len(lines)):
                # Split the line into individual values (assuming they are separated by spaces)
                values = lines[i]
                # Extract values based on their positions in the format string
                value1 = "ATOM"
                value2 = int(i + 1)
                value3 = values[0]  # label
                value4 = "MOL"  # residue
                value5 = 1  # residue number
                value6 = float(values[2])  # x
                value7 = float(values[3])  # y
                value8 = float(values[4])  # z
                value9 = "1.00"
                value10 = "0.00"
                value11 = values[1]  # note
                # Format the values using the specified format string
                formatted_line = "%-6s%5d%5s%4s%10d%8.3f%8.3f%8.3f%6s%6s%4s" % (
                    value1,
                    value2,
                    value3,
                    value4,
                    value5,
                    value6,
                    value7,
                    value8,
                    value9,
                    value10,
                    value11,
                )
                newpdb.append(formatted_line + "\n")
            fp.writelines(newpdb)