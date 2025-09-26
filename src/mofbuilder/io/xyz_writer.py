from pathlib import Path
import sys
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
import mpi4py.MPI as MPI
from veloxchem.errorhandler import assert_msg_critical

class XyzWriter:    

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

        if filepath.suffix != ".xyz":
            filepath = filepath.with_suffix(".xyz")
        
        newxyz = []
        newxyz.append(f"{len(lines)}\n")
        newxyz.append(header)

        with open(filepath, "w") as fp:
            # Iterate over each line in the input file
            for i in range(len(lines)):
                values = lines[i]
                atom_label = values[1]
                atom_number = i + 1
                x = float(values[5])
                y = float(values[6])
                z = float(values[7])
                #xyz format line is 
                formatted_line = "%-5s%8.3f%8.3f%8.3f" % (
                    atom_label,
                    x,
                    y,
                    z,
                )
                newxyz.append(formatted_line + "\n")
            fp.writelines(newxyz)
