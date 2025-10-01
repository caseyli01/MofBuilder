from pathlib import Path
import sys
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
import mpi4py.MPI as MPI
from veloxchem.errorhandler import assert_msg_critical
import datetime

class CifWriter:    

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

        #set
        self.cell_info = None
        self.supercell_boundary = None
        self.unit_cell = None
        self.filepath = None






    def write(self, filepath=None, header='', lines=[], supercell_boundary=None, cell_info=None):
        """
        line format:
        atom_type, atom_label, atom_number, residue_name, residue_number, x, y, z, spin, charge, note
        1         2    3      4            5              6  7  8 9    10 11
        ATOM      1    C       MOL          1            1.000 2.000 3.000 1.00 0.00 C1
        """
        "data format[atom_type, atom_label, atom_number, residue_name, residue_number, value_x, value_y, value_z, spin, charge, note]"
        filepath = Path(filepath) if filepath is not None else Path(self.filepath)
        assert_msg_critical(filepath is not None, "ciffilepath is not specified")
        # check if the file directory exists and create it if it doesn't
        self.file_dir = Path(filepath).parent
        if self._debug:
            self.ostream.print_info(f"targeting directory: {self.file_dir}")
        self.file_dir.mkdir(parents=True, exist_ok=True)
        a,b,c,alpha,beta,gamma = cell_info
        x_max, y_max, z_max = map(float, supercell_boundary)

        if filepath.suffix != ".cif":
            filepath = filepath.with_suffix(".cif")


        with open(filepath, 'w') as new_cif:
            new_cif.write('data_' + filepath.name[0:-4] + '\n')
            new_cif.write('_audit_creation_date              ' + datetime.datetime.today().strftime('%Y-%m-%d') + '\n')
            new_cif.write("_audit_creation_method           "+header + '\n')
            new_cif.write("_symmetry_space_group_name     'P1'" + '\n')
            new_cif.write('_symmetry_Int_Tables_number       1' + '\n')
            new_cif.write('_symmetry_cell_setting            triclinic' + '\n')
            new_cif.write('loop_' + '\n')
            new_cif.write('_symmetry_equiv_pos_as_xyz' + '\n')
            new_cif.write('  x,y,z' + '\n')
            new_cif.write('_cell_length_a                    ' + str(a) + '\n')
            new_cif.write('_cell_length_b                    ' + str(b) + '\n')
            new_cif.write('_cell_length_c                    ' + str(c) + '\n')
            new_cif.write('_cell_angle_alpha                 ' + str(alpha) + '\n')
            new_cif.write('_cell_angle_beta                  ' + str(beta) + '\n')
            new_cif.write('_cell_angle_gamma                 ' + str(gamma) + '\n')
            new_cif.write('loop_' + '\n')
            new_cif.write('_atom_site_label' + '\n')
            new_cif.write('_atom_site_type_symbol' + '\n')
            new_cif.write('_atom_site_fract_x' + '\n')
            new_cif.write('_atom_site_fract_y' + '\n')
            new_cif.write('_atom_site_fract_z' + '\n')

            # Iterate over each line in the input file
            for i in range(len(lines)):
                values = lines[i]
                atom_type = values[0]
                atom_label = values[1]
                #atom_number = i + 1
                x = float(values[5])/x_max if x_max !=0 else 0
                y = float(values[6])/y_max if y_max !=0 else 0
                z = float(values[7])/z_max if z_max !=0 else 0
                formatted_line = "%7s%4s%15.10f%15.10f%15.10f" % (
                    atom_type, atom_label, x, y, z)
                new_cif.write(formatted_line + '\n')
                
            new_cif.write('loop_' + '\n')