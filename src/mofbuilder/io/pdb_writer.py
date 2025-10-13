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
        filepath = Path(filepath) if filepath is not None else Path(
            self.filepath)
        assert_msg_critical(filepath is not None,
                            "pdb filepath is not specified")
        # check if the file directory exists and create it if it doesn't
        self.file_dir = Path(filepath).parent
        if self._debug:
            self.ostream.print_info(f"targeting directory: {self.file_dir}")
        self.file_dir.mkdir(parents=True, exist_ok=True)

        if filepath.suffix != ".pdb":
            filepath = filepath.with_suffix(".pdb")

        newpdb = []
        newpdb.append(header)
        last_name = ""
        last_residue_number = 0
        residue_count = 0

        with open(filepath, "w") as fp:
            # Iterate over each line in the input file
            for i in range(len(lines)):
                values = lines[i]
                if lines[i][3] != last_name or lines[i][
                        4] != last_residue_number:
                    residue_count += 1
                    last_name = lines[i][3]
                    last_residue_number = lines[i][4]
                atom_type = values[0]
                atom_label = values[1]+str(i+1)
                atom_number = i + 1
                residue_name = values[3].split('_')[0]
                residue_number = residue_count
                x = values[5]
                y = values[6]
                z = values[7]
                spin = values[8]
                charge = values[9]
                note = values[10]
                # Format the values using the specified format string

                # Format the values using the specified format string
                formatted_line = (
                    "%-6s%5d %-4s%s%3s %1s%4d%s   "
                    "%8.3f%8.3f%8.3f%6.2f%6.2f          %4s"
                ) % (
                    "ATOM",                 # record
                    int(atom_number),       # atom serial
                    atom_label,             # atom name
                    "",                     # altLoc (blank)
                    residue_name[:3],       # residue name
                    "A",                    # chain ID (default A)
                    int(residue_number),    # resSeq
                    "",                     # iCode (blank)
                    float(x),               # x
                    float(y),               # y
                    float(z),               # z
                    1.00,                   # occupancy (default 1.00, or spin if that's what you mean)
                    0.00,                   # B-factor (default 0.00, or use real)
                    note if isinstance(note, str) else note[1]  # element symbol
                )
                newpdb.append(formatted_line + "\n")
            fp.writelines(newpdb)
