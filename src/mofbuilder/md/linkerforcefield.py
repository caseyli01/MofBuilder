import numpy as np
import sys
import networkx as nx
import re
import linecache
from pathlib import Path
from veloxchem.molecule import Molecule
from veloxchem.mmforcefieldgenerator import MMForceFieldGenerator
from veloxchem.xtbdriver import XtbDriver
from veloxchem.optimizationdriver import OptimizationDriver
from veloxchem.molecularbasis import MolecularBasis
from veloxchem.scfrestdriver import ScfRestrictedDriver
from veloxchem.scfunrestdriver import ScfUnrestrictedDriver
from ..io.basic import nn
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master, hartree_in_kcalpermol, hartree_in_kjpermol
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI


class LinkerForceFieldGenerator:

    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)

        #write the eG to the file
        #need to be set before use
        self.linker_optimization = True
        self.optimize_drv = "xtb"  #xtb or qm
        #self.mmforcefield_generator = MMForceFieldGenerator()
        self.linker_ff_name = "Linker"
        self.linker_residue_name = "EDG"  #default residue name for linker

        self.linker_charge = -2
        self.linker_multiplicity = 1
        self.target_directory = None
        self.save_files = False

        self.src_linker_forcefield_itpfile = None  #set before use
        self.src_linker_molecule = None  #set before use
        self.dest_linker_molecule = None  #will be generated after reconnecting
        self.linker_itp_path = None  #final itp path after mapping

        self._debug = False

    def _reconnect_linker_molecule(self, linker_mol_data=None):
        #use x-x pair to set the connectivity of the linker molecule
        "x22 C 22 EDGE 2 6.13775354767322 23.363695120662403 10.596310043131485 0 0.0 EDGE_2"
        xyz_string = ''
        atom_idx = 0
        X_indices_coords = []
        lower_x_indices_coords = []  #list of (index, [x,y,z])

        atom_coords = []
        assert_msg_critical(
            linker_mol_data is not None,
            "no linker molecule set when trying to generate forcefield for linker"
        )

        lines = linker_mol_data
        xyz_string += f"{len(lines)}\n\n"
        for i in range(len(lines)):
            atom_type = lines[i, 0]
            atom_label = lines[i, 1]
            x = float(lines[i, 5])
            y = float(lines[i, 6])
            z = float(lines[i, 7])
            atom_coords.append((nn(atom_label), [x, y, z]))
            if nn(atom_type) in ['X']:
                X_indices_coords.append((atom_idx, [x, y, z]))
            elif nn(atom_type) in ['x']:
                lower_x_indices_coords.append((atom_idx, [x, y, z]))
            atom_idx += 1
            xyz_string += f"{nn(atom_label)} {x} {y} {z}\n"
        molecule = Molecule.read_xyz_string(xyz_string)

        connectivity_matrix = molecule.get_connectivity_matrix()
        connectivity_matrix, connect_constraints = self._reconnect(
            X_indices_coords, connectivity_matrix)
        connectivity_matrix, connect_constraints = self._reconnect(
            lower_x_indices_coords, connectivity_matrix, connect_constraints)
        if self._debug:
            self.ostream.print_info(f"constaints{connect_constraints}")
            self.ostream.flush()
        self.reconnected_connectivity_matrix = connectivity_matrix
        self.reconnected_constraints = connect_constraints
        return molecule

    def generate_reconnected_molecule_forcefield(self, linker_mol_data=None):
        if linker_mol_data is None:
            return
        molecule = self._reconnect_linker_molecule(linker_mol_data)
        molecule.set_charge(self.linker_charge)
        molecule.set_multiplicity(self.linker_multiplicity)

        if not self.linker_optimization:
            #make x indices to be split from middle
            #connectivity_matrix,connect_constraints = self.reconnect(X_indices_coords, connectivity_matrix)
            #connectivity_matrix,connect_constraints = self.reconnect(lower_x_indices_coords, connectivity_matrix,connect_constraints)
            ff_gen = MMForceFieldGenerator()
            ff_gen.topology_update_flag = True
            ff_gen.ostream.mute()
            ff_gen.connectivity_matrix = self.reconnected_connectivity_matrix
            ff_gen.create_topology(molecule, resp=True)
            self.linker_itp_path = Path(
                self.target_directory, self.linker_ff_name).with_suffix('.itp')
            ffname = str(self.linker_itp_path).removesuffix('.itp')
            ff_gen.write_gromacs_files(filename=ffname,
                                       mol_name=self.linker_residue_name)
        elif (self.linker_optimization and self.optimize_drv == "qm"):
            constrained_opt_linker_mol, scf_result = self._dft_optimize(
                molecule, self.reconnected_constraints)
            constrained_opt_linker_mol.set_charge(self.linker_charge)
            constrained_opt_linker_mol.set_multiplicity(self.linker_multiplicity)
            free_opt_linker_mol, scf_result = self._dft_optimize(
                constrained_opt_linker_mol, None)
            free_opt_linker_mol.set_charge(self.linker_charge)
            free_opt_linker_mol.set_multiplicity(self.linker_multiplicity)
            ff_gen = MMForceFieldGenerator()
            #basis = MolecularBasis.read(free_opt_linker_mol, "def2-svp")
            self.linker_itp_path = Path(
                self.target_directory, self.linker_ff_name).with_suffix('.itp')
            ffname = str(self.linker_itp_path).removesuffix('.itp')
            ff_gen.create_topology(free_opt_linker_mol, resp=True)
            ff_gen.write_gromacs_files(filename=ffname,
                                       mol_name=self.linker_residue_name)
        elif (self.linker_optimization and self.optimize_drv == "xtb"):
            self.ostream.print_info(
                f"xtb driver is using for linker optimization")
            self.ostream.flush()
            opt_linker_mol = self._xtb_optimize(molecule,
                                                self.reconnected_constraints)
            opt_linker_mol.set_charge(self.linker_charge)
            opt_linker_mol.set_multiplicity(self.linker_multiplicity)
            ff_gen = MMForceFieldGenerator()
            self.ostream.print_info(
                f"generating forcefield of linker molecule for Gromacs")
            self.ostream.flush()
            self.linker_itp_path = Path(
                self.target_directory,
                Path(self.linker_ff_name).with_suffix('.itp'))
            ffname = str(self.linker_itp_path).removesuffix('.itp')
            ff_gen.create_topology(opt_linker_mol, resp=True)
            Path(self.linker_itp_path).parent.mkdir(parents=True,
                                                    exist_ok=True)
            ff_gen.write_gromacs_files(filename=ffname,
                                       mol_name=self.linker_residue_name)

    def _reconnect(self,
                   X_indices_coords,
                   connectivity_matrix,
                   connect_constraints=[]):
        if not connect_constraints:
            connect_constraints = []
        half_len_X_num = len(X_indices_coords) // 2
        X1_ind_coords = X_indices_coords[:half_len_X_num]
        X2_ind_coords = X_indices_coords[half_len_X_num:]
        #print(f"X1_indices: {X1_ind_coords}")
        #print(f"X2_indices: {X2_ind_coords}")

        for i, j in zip(X1_ind_coords, X2_ind_coords):
            #check distance
            dist = np.linalg.norm(np.array(i[1]) - np.array(j[1]))
            if dist < 4.0:  #threshold for x-x bond
                #clean all other bonds to X, should n
                #connectivity_matrix[i[0], :] = 0
                #connectivity_matrix[:, i[0]] = 0
                #connectivity_matrix[j[0], :] = 0
                #connectivity_matrix[:, j[0]] = 0
                #rebuild bond between X and C
                connectivity_matrix[i[0], j[0]] = 1
                connectivity_matrix[j[0], i[0]] = 1
                if self._debug:
                    self.ostream.print_info(
                        f"X-X distance {dist} is within threshold, bond created between indices {i[0]} and {j[0]}."
                    )
                connect_constraints.append(
                    f"set distance {i[0]+1} {j[0]+1} 1.54")
            else:
                continue
        return connectivity_matrix, connect_constraints

    def _xtb_optimize(self, molecule, constraints):
        xtb_drv = XtbDriver()
        xtb_drv.ostream.mute()
        xtb_results = xtb_drv.compute(molecule)
        opt_drv = OptimizationDriver(xtb_drv)
        opt_drv.conv_energy = 1e-04
        opt_drv.conv_drms = 1e-02
        opt_drv.conv_dmax = 2e-02
        opt_drv.conv_grms = 4e-03
        opt_drv.conv_gmax = 8e-03
        opt_drv.constraints = constraints
        opt_drv.tmax = 0.02
        opt_drv.max_iter = 100
        opt_drv.conv_maxiter = True
        opt_results = opt_drv.compute(molecule)
        self.ostream.print_info(
            "Optimization of linker molecule completed successfully with xtb driver."
        )
        self.ostream.flush()
        opt_mol = Molecule.read_xyz_string(opt_results["final_geometry"])
        opt_energy = opt_results['opt_energies'][-1] * hartree_in_kcalpermol()
        self.ostream.print_info(
            f"Optimization energy is {opt_energy} kcal/mol")
        self.ostream.flush()
        if self.save_files:
            fname = str(Path(self.target_directory, "linker_opt.xyz"))
            opt_mol.write_xyz_file(fname)
        return opt_mol

    def _dft_optimize(self, molecule, constraints):
        if molecule.get_multiplicity() == 1:
            mol_scf_drv = ScfRestrictedDriver()
        else:
            mol_scf_drv = ScfUnrestrictedDriver()
        basis = MolecularBasis.read(molecule, "def2-svp")
        mol_scf_drv.conv_thresh = 1e-4
        mol_scf_drv.xcfun = "blyp"
        mol_scf_drv.ri_coulomb = True
        mol_scf_drv.grid_level = 4
        mol_scf_results = mol_scf_drv.compute(molecule, basis)
        mol_opt_drv = OptimizationDriver(mol_scf_drv)
        mol_opt_drv.conv_energy = 1e-04
        mol_opt_drv.conv_drms = 1e-02
        mol_opt_drv.conv_dmax = 2e-02
        mol_opt_drv.conv_grms = 4e-03
        mol_opt_drv.conv_gmax = 8e-03
        mol_opt_drv.constraints = constraints
        mol_opt_drv.tmax = 0.02
        mol_opt_drv.max_iter = 200
        opt_results = mol_opt_drv.compute(molecule, basis, mol_scf_results)
        self.ostream.print_info(
            "Optimization of linker molecule completed successfully with DFT.")
        self.ostream.flush()
        opt_mol = Molecule.read_xyz_string(opt_results["final_geometry"])
        opt_energy = opt_results['opt_energies'][-1] * hartree_in_kcalpermol()
        self.ostream.print_info(
            f"Optimization energy is {opt_energy} kcal/mol")
        self.ostream.flush()
        if self.save_files:
            fname = str(Path(self.target_directory, "linker_opt.xyz"))
            opt_mol.write_xyz_file(fname)
        return opt_mol, mol_scf_results

    #mapping molecule indices to process for linker forcefield

    def _create_graph_from_matrix(self, matrix, coords_matrix, labels):
        """ Create a graph from a given connectivity matrix. """
        G = nx.Graph()
        for n in range(len(matrix)):
            G.add_node(n, atom_id=n, coords=coords_matrix[n], label=labels[n])
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                G.add_edge(i, j, weight=matrix[i][j], orientation=0)
        ref_vec = None
        boundary_nodes = [
            n for n, d in G.degree()
            if ((d == 1) and (nn(G.nodes[n]['label']) in ['O', 'Y']))
        ]
        #find 2 step away neighbors of boundary nodes
        for bn in boundary_nodes:
            #find 2 step away neighbors of bn, use path length to find not for loop
            two_step_neighbors = [
                n for n, d in nx.single_source_shortest_path_length(
                    G, source=bn).items() if d == 2
            ]

            for nnbr in two_step_neighbors:
                if (not G.has_edge(bn, nnbr)
                        and (nn(G.nodes[nnbr]['label']) in ['O', 'Y'])):
                    #add edge from bn to nnbr
                    # determine orientation
                    vec_bn_nnbr = np.array(coords_matrix[nnbr]) - np.array(
                        coords_matrix[bn])
                    if ref_vec is None:
                        ref_vec = vec_bn_nnbr
                    dot_ref = np.dot(ref_vec, vec_bn_nnbr)
                    if dot_ref >= 0:
                        orientation = 1
                    else:
                        orientation = -1
                    G.add_edge(bn, nnbr, weight=0.5, orientation=orientation)
        return G

    def _find_isomorphism_and_mapping(self, src_matrix1, dest_matrix2,
                                      src_coords_matrix, dest_coords_matrix,
                                      src_labels, dest_labels):
        """ Check if two matrices are isomorphic and return the mapping. """
        G1 = self._create_graph_from_matrix(src_matrix1, src_coords_matrix,
                                            src_labels)
        G2 = self._create_graph_from_matrix(dest_matrix2, dest_coords_matrix,
                                            dest_labels)

        #flip_G2 = G2.copy()  # Create a copy of G2 to flip edges
        #for u, v, data in flip_G2.edges(data=True):
        #    data['orientation'] *= -1  # Flip the orientation
        #edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        #    ['weight', 'orientation'], [None, None])

        def orientation_match(e1, e2):
            return e1.get('orientation', 0) == e2.get('orientation', 0) or \
                e1.get('orientation', 0) == -e2.get('orientation', 0)

        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2, orientation_match)

        #gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if gm.is_isomorphic():
            #return mapping to node attribute atom_id
            mapping = {
                G1.nodes[k]['atom_id']: G2.nodes[v]['atom_id']
                for k, v in gm.mapping.items()
            }
            return True, mapping
        else:
            #try flipped G2
            ##gm_flip = nx.algorithms.isomorphism.GraphMatcher(
            #    G1, flip_G2, edge_match=edge_match)
            #if gm_flip.is_isomorphic():
            #    mapping = {
            #        G1.nodes[k]['atom_id']: flip_G2.nodes[v]['atom_id']
            #        for k, v in gm_flip.mapping.items()
            #    }
            #    return True, mapping
            return False, None

    def map_existing_forcefield(self, linker_mol_data=None):
        save_itp_path = Path(self.target_directory,
                             self.linker_ff_name).with_suffix('.itp')
        mapper = ForceFieldMapper(comm=self.comm, ostream=self.ostream)
        mapper.dest_molecule = self._reconnect_linker_molecule(linker_mol_data)
        mapper.dest_molecule_connectivity_matrix = self.reconnected_connectivity_matrix
        mapper.src_molecule_forcefield_itpfile = self.src_linker_forcefield_itpfile
        mapper.src_molecule = self.src_linker_molecule
        mapped_sections = mapper._map_forcefield_sections(
            dest_resname=self.linker_residue_name)
        mapper.write_mapped_itp_file(mapped_sections, str(save_itp_path))
        self.linker_itp_path = save_itp_path


class ForceFieldMapper:

    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)
        self.src_molecule_forcefield_itpfile = None
        self.src_molecule = None
        self.dest_molecule_forcefield_itpfile = None
        self.dest_molecule = None
        self.target_directory = None
        self.save_files = False
        self._debug = False

        self.dest_molecule_connectivity_matrix = None

    def _get_mapping_between_two_molecules(self, src_molecule, dest_molecule):
        src_mol_connectivity = src_molecule.get_connectivity_matrix()
        if self.dest_molecule_connectivity_matrix is not None:
            dest_mol_connectivity = self.dest_molecule_connectivity_matrix
        else:
            dest_mol_connectivity = dest_molecule.get_connectivity_matrix()
        # add clockwise connection to edge atoms based on np.cross product
        # to make the connectivity more unique
        # this is to avoid the issue of symmetric molecules
        #for edge atoms whose connectivity is 1, find the connected atom
        ##add clockwise connection to boundary atoms based on np.cross product

        isomorphic, mapping = LinkerForceFieldGenerator(
        )._find_isomorphism_and_mapping(
            src_mol_connectivity, dest_mol_connectivity,
            src_molecule.get_coordinates_in_angstrom(),
            dest_molecule.get_coordinates_in_angstrom(),
            src_molecule.get_labels(), dest_molecule.get_labels())

        #rebuild the connectivity of the linker in MOF
        #reconnect the X-X bonds and seperate the frags
        if not isomorphic:
            raise ValueError(
                "The linker molecule in MOF is not isomorphic to the reference linker molecule."
            )
        else:
            #sort mapping by src indices
            mapping = dict(sorted(mapping.items()))
            #mapping should be like {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
            mapping = {
                k + 1: v + 1
                for k, v in mapping.items()
            }  # Ensure it's a standard dict
            if self._debug:
                self.ostream.print_info(
                    f"Mapping between source and destination linker molecule: {mapping}"
                )
                self.ostream.flush()
            return mapping

    def _parse_itp_file(self, itpfile):
        assert_msg_critical(itpfile is not None,
                            "No source forcefield itp file provided.")

        # Read the input file and identify sections
        sections = {}

        with open(itpfile, 'r') as fp:
            number = []
            lineNumber = 1
            keyword = "]"
            lines = fp.readlines()  #input('Keyword:')
            for eachline in lines:  #search for keywords and get linenumber
                m = re.search(keyword, eachline)
                if m is not None:
                    number.append(lineNumber - 1)  #split by linenumber
                lineNumber += 1
            number.append(len(lines))
            number = list(set(number))  #remove duplicates and sort
            number.sort()
            size = int(len(number))
            for i in range(size - 1):
                start = number[i]
                end = number[i + 1]
                middlelines = lines[start:end]
                #drop empty sections and empty lines
                middlelines = [line for line in middlelines if line.strip()]
                section = re.findall(r'\[(.*?)\]', middlelines[0])
                title = section[0].split()[0]
                if title == 'dihedrals':
                    if 'impropers' in middlelines[1]:
                        title = 'dihedrals_im'
                sections[title] = middlelines  #exclude the keyword line
        #check sections
        if self._debug:
            self.ostream.print_info(
                f"Sections found in itp file: {list(sections.keys())}")
            self.ostream.flush()
        return sections

    def _format_atomtypes(self, lines):
        return lines

    def _format_moleculetype(self, lines, new_resname):
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        values = lines[2].split()
        values[0] = new_resname
        formatted_line = "%-7s%7s" % (values[0], values[1])
        newff.append(formatted_line + "\n")
        return newff

    def _format_atoms(self, lines, src_dest_map, dest_atomlabels,
                      dest_resname):
        #set the index to dest index
        #set the atom name to dest atom name

        new_atoms_section = []
        new_atoms_section.append(lines[0])
        new_atoms_section.append(lines[1])
        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values) == 0:
                continue
            src_index = int(values[0])
            dest_index = src_dest_map[src_index]  #recover to 0-based index
            values[
                4] = f"{dest_atomlabels[dest_index-1]}{dest_index}"  #recover to 0-based index"
            values[5] = dest_index
            values[0] = dest_index
            values[6] = float(values[6])
            values[7] = float(values[7])
            values[3] = dest_resname

            if len(values) > 8:
                formatted_line = "%7d%7s%7s%7s%7s%7d%15.8f%15.6f%7s" % (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                )
                new_atoms_section.append(formatted_line)

            else:
                formatted_line = "%7s%7s%7s%7s%7s%7s%15.8f%15.6f" % (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                )
                new_atoms_section.append(formatted_line)

        #sort lines by line.split()[0]
        new_atoms_section_header = new_atoms_section[:2]
        new_atoms_section_main = sorted(new_atoms_section[2:],
                                        key=lambda i: int(i.split()[0]))
        new_main = []
        #get last line of charge sum
        charge_sum = 0.0
        for line in new_atoms_section_main:
            values = line.split()
            charge_sum += float(values[6])
            charge_sum_line = line + f" ;qtol  {charge_sum:15.6f}\n"
            new_main.append(charge_sum_line)

        new_atoms_section = new_atoms_section_header + new_main
        return new_atoms_section

    def _format_bonds(self, lines, src_dest_map):
        new_bonds_section = []
        new_bonds_section.append(lines[0])
        new_bonds_section.append(lines[1])

        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values) == 0:
                continue
            src_bond_i_index = int(values[0])
            src_bond_j_index = int(values[1])
            values[0] = src_dest_map[src_bond_i_index]
            values[1] = src_dest_map[src_bond_j_index]
            values[3] = float(values[3])
            values[4] = float(values[4])

            formatted_line = "%7s%7s%6s%15.7f%15.6f" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
            )
            new_bonds_section.append(formatted_line + "\n")
        return new_bonds_section

    def _format_pairs(self, lines, src_dest_map):
        new_pairs_section = []
        new_pairs_section.append(lines[0])
        new_pairs_section.append(lines[1])
        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values) == 0:
                continue
            src_pair_i_index = int(values[0])
            values[0] = src_dest_map[src_pair_i_index]
            src_pair_j_index = int(values[1])
            values[1] = src_dest_map[src_pair_j_index]

            formatted_line = "%7s%7s%6s" % (values[0], values[1], values[2])
            new_pairs_section.append(formatted_line + "\n")
        return new_pairs_section

    def _format_angles(self, lines, src_dest_map):
        new_angles_section = []
        new_angles_section.append(lines[0])
        new_angles_section.append(lines[1])
        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values) == 0:
                continue
            src_angle_i_index = int(values[0])
            values[0] = src_dest_map[src_angle_i_index]
            src_angle_j_index = int(values[1])
            values[1] = src_dest_map[src_angle_j_index]
            src_angle_k_index = int(values[2])
            values[2] = src_dest_map[src_angle_k_index]

            values[4] = float(values[4])
            values[5] = float(values[5])

            formatted_line = "%7s%7s%7s%6s%13.7f%12.6f" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
            )
            new_angles_section.append(formatted_line + "\n")
        return new_angles_section

    def _format_dihedrals(self, lines, src_dest_map):
        new_dihedrals_section = []
        new_dihedrals_section.append(lines[0])
        new_dihedrals_section.append(lines[1])
        new_dihedrals_section.append(lines[2])
        for i in range(3, len(lines)):
            values = lines[i].split()
            if len(values) == 0:
                continue
            src_dihedral_i_index = int(values[0])
            values[0] = src_dest_map[src_dihedral_i_index]
            src_dihedral_j_index = int(values[1])
            values[1] = src_dest_map[src_dihedral_j_index]
            src_dihedral_k_index = int(values[2])
            values[2] = src_dest_map[src_dihedral_k_index]
            src_dihedral_l_index = int(values[3])
            values[3] = src_dest_map[src_dihedral_l_index]

            values[5] = float(values[5])
            values[6] = float(values[6])

            formatted_line = "%7s%7s%7s%7s%6s%13.7f%12.7f%3s" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
            )
            new_dihedrals_section.append(formatted_line + "\n")
        return new_dihedrals_section

    def _map_forcefield_sections(self, dest_resname="MOL"):
        src_dest_map = self._get_mapping_between_two_molecules(
            self.src_molecule, self.dest_molecule)
        self.ostream.print_info(f"mapping: {src_dest_map}")
        dest_atomlabels = self.dest_molecule.get_labels()
        sections = self._parse_itp_file(self.src_molecule_forcefield_itpfile)
        if self._debug:
            self.ostream.print_info(
                f"Forcefield sections found in source itp file: {list(sections.keys())}"
            )
            self.ostream.flush()
        dest_sections = {}
        if 'atomtypes' in sections:
            dest_sections['atomtypes'] = self._format_atomtypes(
                sections['atomtypes'])
        if 'moleculetype' in sections:
            dest_sections['moleculetype'] = self._format_moleculetype(
                sections['moleculetype'], dest_resname)
        if 'atoms' in sections:
            dest_sections['atoms'] = self._format_atoms(
                sections['atoms'], src_dest_map, dest_atomlabels, dest_resname)
        if 'bonds' in sections:
            dest_sections['bonds'] = self._format_bonds(
                sections['bonds'], src_dest_map)
        if 'pairs' in sections:
            dest_sections['pairs'] = self._format_pairs(
                sections['pairs'], src_dest_map)
        if 'angles' in sections:
            dest_sections['angles'] = self._format_angles(
                sections['angles'], src_dest_map)
        if 'dihedrals' in sections:
            dest_sections['dihedrals'] = self._format_dihedrals(
                sections['dihedrals'], src_dest_map)
        if 'dihedrals_im' in sections:
            dest_sections['dihedrals_im'] = self._format_dihedrals(
                sections['dihedrals_im'], src_dest_map)
        if self._debug:
            self.ostream.print_info(
                f"Forcefield sections mapped from source to destination molecule."
            )
            self.ostream.print_info(
                f"Sections in mapped itp file: {list(dest_sections.keys())}")
            self.ostream.print_info(f"mapping {src_dest_map}")
            self.ostream.print_info(
                f"destination atom labels {dest_atomlabels}")
            self.ostream.flush()
        self.mapped_sections = dest_sections
        return dest_sections

    def write_mapped_itp_file(self, mapped_sections, output_itpfile):
        if not Path(output_itpfile).suffix == '.itp':
            output_itpfile = Path(output_itpfile).with_suffix('.itp')
        self.ostream.print_info(
            f"Writing mapped forcefield to itp file: {output_itpfile}")
        self.ostream.flush()
        with open(output_itpfile, 'w') as fp:
            for section_name, lines in mapped_sections.items():
                #fp.write(f"[{section_name}]\n")
                for line in lines:
                    fp.write(line)
                fp.write("\n")  # Add a newline between sections


if __name__ == "__main__":
    mapper = ForceFieldMapper()
    mapper.src_molecule_forcefield_itpfile = "hho copy.itp"
    mapper.src_molecule = Molecule.read_xyz_file("src_mol.xyz")

    #mapper.dest_molecule_forcefield_itpfile = "dest.itp"
    mapper.dest_molecule = Molecule.read_smiles("O")
    mapped_sections = mapper._map_forcefield_sections(dest_resname="WAT")
    mapper.write_mapped_itp_file(mapped_sections, "mapped.itp")
