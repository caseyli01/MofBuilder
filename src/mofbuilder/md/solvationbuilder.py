import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
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
import sys


def safe_dict_copy(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = safe_dict_copy(v)
        elif isinstance(v, np.ndarray):
            new_d[k] = v.copy()
        elif isinstance(v, list):
            new_d[k] = list(v)
        else:
            new_d[k] = v
    return new_d


class SolventPacker:

    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)

        self.buffer = 1.8  # Å
        self.box_size = None
        self.trial_rounds = 10
        self.max_fill_rounds = 500  # Maximum number of filling rounds
        #scalar to control the number of candidates generated in each round, 1.0 means generate number of candidates equal to cavity number
        self.scalar = 1.0
        #use solute file
        self.solute_file = None

        #use solvents names
        self.solvents_names = []
        self.solvents_files = []
        self.solvents_proportions = []
        self.solvents_quantities = []

        self.custom_solvent_data = {}

        self._debug = False

    def _read_xyz(self, filename):
        labels, coords = [], []
        with open(filename) as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                labels.append(parts[0])
                coords.append(
                    [float(parts[1]),
                     float(parts[2]),
                     float(parts[3])])
        com = np.mean(coords, axis=0)
        coords = np.array(coords) - com  # center at origin
        return labels, coords

    def _generate_candidates_each_solvent(self,
                                          solvent_coords,
                                          solvent_labels,
                                          solvent_n_atoms,
                                          target_mol_number,
                                          residue_idx_start=0,
                                          points_template=None,
                                          box_size=None,
                                          rot=True):

        #use inner box size to avoid placing solvent on the boundary
        if points_template is None:
            random_points = self._box2randompoints(None, box_size, target_mol_number)
        else:
            if points_template.shape[0] < target_mol_number:
                #add random points to fill the rest
                n_additional = target_mol_number - points_template.shape[0]
                random_points = self._box2randompoints(points_template, box_size, n_additional)
            else:
                random_points = points_template
        #shuffle the random points
        #np.random.shuffle(random_points)
        target_mol_number = random_points.shape[0]
        if target_mol_number == 0:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty(
                (0, 1)), np.empty((0, 1))
        if rot:
            rots = R.random(target_mol_number).as_matrix()
            coords_exp = solvent_coords[np.newaxis, :, :]
            rot_coords = np.matmul(coords_exp, rots.transpose(0, 2, 1))
            candidates = rot_coords.reshape(-1, 3)
        else:
            candidates = np.tile(solvent_coords, (target_mol_number, 1))
        candidates += np.repeat(random_points, solvent_n_atoms, axis=0)

        labels = np.array(list(solvent_labels) * target_mol_number).reshape(
            -1, 1)
        residue_idx = np.repeat(
            np.arange(residue_idx_start,
                      residue_idx_start + target_mol_number),
            solvent_n_atoms).reshape(-1, 1)

        return random_points, candidates, labels, residue_idx

    def _box2randompoints(self, points_template, box_size, n_additional):
        if points_template is None:
            points_template = np.empty((0, 3))
        additional_points = np.random.rand(n_additional, 3)
        additional_points[:, 0] = additional_points[:, 0] * (box_size[0][1] - box_size[0][0]) + box_size[0][0]
        additional_points[:, 1] = additional_points[:, 1] * (box_size[1][1] - box_size[1][0]) + box_size[1][0]
        additional_points[:, 2] = additional_points[:, 2] * (box_size[2][1] - box_size[2][0]) + box_size[2][0]
        random_points = np.vstack((points_template, additional_points))
        return random_points

    def remove_overlaps_kdtree(self, existing_coords, candidate_coords,
                                candidate_residues):
        candidate_residues = candidate_residues.reshape(-1)

        # === Round 1: overlap with existing atoms ===
        tree_existing = cKDTree(existing_coords)
        # faster than query_ball_point for uniform radius search if you just need overlap boolean
        dists, _ = tree_existing.query(candidate_coords,
                                       k=1,
                                       distance_upper_bound=self.buffer)
        mask_overlap_existing = np.isfinite(dists)
        bad_residues_existing = np.unique(
            candidate_residues[mask_overlap_existing])

        # === Round 2: overlap among candidates ===
        tree_candidates = cKDTree(candidate_coords)
        pairs = np.array(list(tree_candidates.query_pairs(
            r=self.buffer)))  # (i,j) pairs within radius

        if pairs.size > 0:
            res_i = candidate_residues[pairs[:, 0]]
            res_j = candidate_residues[pairs[:, 1]]
            # mark both residues if they differ (overlap between different residues)
            mask_diff = res_i != res_j
            bad_residues_candidates = np.unique(
                np.concatenate([res_i[mask_diff], res_j[mask_diff]]))
        else:
            bad_residues_candidates = np.array([],
                                               dtype=candidate_residues.dtype)

        # === Combine ===
        bad_residues = np.union1d(bad_residues_existing,
                                  bad_residues_candidates)
        keep_mask = ~np.isin(candidate_residues, bad_residues)
        drop_mask = ~keep_mask
        return keep_mask, drop_mask

    def _remove_overlaps_kdtree(self, existing_coords, candidate_coords,
                               candidate_residues):
        candidate_residues = candidate_residues.reshape(-1)

        # === Round 1: overlaps with existing atoms ===
        tree_existing = cKDTree(existing_coords)
        dists, _ = tree_existing.query(candidate_coords,
                                       k=1,
                                       distance_upper_bound=self.buffer)
        mask_overlap_existing = np.isfinite(dists)
        bad_residues_existing = np.unique(
            candidate_residues[mask_overlap_existing])

        # === Round 2: residue–residue overlaps (atom-level proximity) ===
        tree_candidates = cKDTree(candidate_coords)
        atom_pairs = np.array(list(
            tree_candidates.query_pairs(r=self.buffer)))  # (i, j)

        if len(atom_pairs) > 0:
            res_i = candidate_residues[atom_pairs[:, 0]]
            res_j = candidate_residues[atom_pairs[:, 1]]
            # only different residues
            diff_mask = res_i != res_j
            residue_pairs = np.unique(np.sort(
                np.stack([res_i[diff_mask], res_j[diff_mask]], axis=1)),
                                      axis=0)
        else:
            residue_pairs = np.empty((0, 2), dtype=candidate_residues.dtype)

        # === Round 3: vectorized decision (keep smaller residue id) ===
        if len(residue_pairs) > 0:
            smaller = np.minimum(residue_pairs[:, 0], residue_pairs[:, 1])
            larger = np.maximum(residue_pairs[:, 0], residue_pairs[:, 1])
            bad_residues_candidates = np.unique(larger)
            keep_residues = np.unique(smaller)
        else:
            bad_residues_candidates = np.array([],
                                               dtype=candidate_residues.dtype)
            keep_residues = np.unique(candidate_residues)

        # === Combine ===
        bad_residues = np.union1d(bad_residues_existing,
                                  bad_residues_candidates)
        keep_mask = np.isin(
            candidate_residues,
            keep_residues) & ~np.isin(candidate_residues, bad_residues)
        drop_mask = ~keep_mask

        return keep_mask, drop_mask

    def _generate_candidates(self,
                             sol_dict,
                             target_number,
                             res_start=0,
                             points_template=None,
                             box_size=None,
                             rot=True):
        '''
        return all_data, sol_dict, res_start
        all_data: dict with keys 'coords', 'labels', 'residue_idx', 'atoms_number'
        sol_dict: updated sol_dict with keys 'extended_residue_idx', 'extended_com_points'
        res_start: updated res_start by adding target_number mols
        '''

        all_data = {}
        all_sol_mols = []
        for solvent_name in sol_dict:
            n_mol = int(target_number * sol_dict[solvent_name]['proportion'])
            if n_mol == 0:
                if sol_dict[solvent_name]['proportion'] > 0:
                    n_mol = 1
            all_sol_mols.append(n_mol)
        all_sol_atoms_num = [
            n_mol * sol_dict[solvent_name]['n_atoms']
            for n_mol, solvent_name in zip(all_sol_mols, sol_dict)
        ]

        all_data['atoms_number'] = sum(all_sol_atoms_num)
        all_data['coords'] = np.empty((0, 3))
        all_data['labels'] = np.empty((0, 1))
        all_data['residue_idx'] = np.empty((0, 1))
        start_idx = 0
        if target_number == 0:
            return all_data, sol_dict, res_start, np.empty((0, 3))
        if points_template is not None:
            if points_template.shape[0] < target_number:
                #add random points to fill the rest
                n_additional = target_number - points_template.shape[0]
                points_template = self._box2randompoints(points_template, box_size, n_additional)
            if points_template.shape[0] > target_number:
                points_template = points_template[:target_number]
        all_res_com_random_points = np.empty((0, 3))
        for i, solvent_name in enumerate(sol_dict):
            #solvents_dict[solvent_name]['extended_residue_idx'] = np.empty((0, all_candidates_data['atoms_number']), dtype=bool)
            _target_mol_number = all_sol_mols[i]

            com_random_points, candidates, labels, residue_idx = self._generate_candidates_each_solvent(
                sol_dict[solvent_name]['coords'],
                sol_dict[solvent_name]['labels'],
                sol_dict[solvent_name]['n_atoms'],
                _target_mol_number,
                residue_idx_start=res_start,
                points_template=points_template[sum(all_sol_mols[:i]
                                                    ):sum(all_sol_mols[:i +
                                                                       1])]
                if points_template is not None else None,
                box_size=box_size,
                rot=rot)
            #each res_com_random_points is the center of each solvent molecule in candidates
            #expand res_com_random_points to match the number of atoms in candidates
            res_com_random_points = np.repeat(
                com_random_points, sol_dict[solvent_name]['n_atoms'], axis=0)
            # Create a mask for the solvent with True values for the current residue indices
            ex_residue_idx = np.zeros((sum(all_sol_atoms_num), 1), dtype=bool)

            end_idx = start_idx + _target_mol_number * sol_dict[solvent_name][
                'n_atoms']
            ex_residue_idx[start_idx:end_idx] = True
            start_idx = end_idx

            res_start += _target_mol_number

            sol_dict[solvent_name]['extended_residue_idx'] = np.vstack(
                (sol_dict[solvent_name]['extended_residue_idx'],
                 ex_residue_idx))
            all_res_com_random_points = np.vstack(
                (all_res_com_random_points, res_com_random_points))

            all_data['coords'] = np.vstack((all_data['coords'], candidates))
            all_data['labels'] = np.vstack((all_data['labels'], labels))
            all_data['residue_idx'] = np.vstack(
                (all_data['residue_idx'], residue_idx))

        return all_data, sol_dict, res_start, all_res_com_random_points

    def add_custom_solvent(self, solvent_file, density, molar_mass):
        if isinstance(solvent_file, str) and Path(solvent_file).is_file():
            self.custom_solvent_data[str(Path(solvent_file).stem)] = {
                "file": solvent_file,
                "density": density,
                "molar_mass": molar_mass
            }
        elif isinstance(solvent_file, list):
            for f, d, m in zip(solvent_file, density, molar_mass):
                self._add_custom_solvent(f, d, m)

    def _get_density_molarmass(self, name):
        if name == 'water':
            return 1.0, 18.015  # g/cm³, g/mol
        elif name == 'dmso':
            return 1.1, 78.13
        elif name == 'methanol':
            return 0.792, 32.04
        elif name == 'ethanol':
            return 0.789, 46.07
        elif name == 'acetone':
            return 0.784, 58.08
        elif name == 'acetonitrile':
            return 0.786, 41.05
        elif name == 'chloroform':
            return 1.48, 119.38
        elif name == 'dichloromethane':
            return 1.33, 84.93
        elif name == 'toluene':
            return 0.866, 92.14
        elif name == 'benzene':
            return 0.876, 78.11
        else:
            self.ostream.print_info(
                f"Unknown solvent {name}, should provide density and molar mass."
            )
            #check if custom solvent data is provided
            if name in self.custom_solvent_data:
                if self.custom_solvent_data[name][
                        'density'] is not None and self.custom_solvent_data[
                            name]['molar_mass'] is not None:
                    return self.custom_solvent_data[name][
                        'density'], self.custom_solvent_data[name][
                            'molar_mass']
                else:
                    raise ValueError(
                        f"Custom solvent {name} must have both density and molar mass provided."
                    )

    def _xyzfiles2mols(self, solvents_xyz_files):
        import veloxchem as vlx
        solvents_mols = []
        for solvent_file in solvents_xyz_files:
            mol = vlx.Molecule.read_xyz_file(solvent_file)
            solvents_mols.append(mol)
        return solvents_mols

    def _initialize_solvents_dict(self,
                                  solvents_files=[],
                                  proportion=[],
                                  quantities=[]):
        #use proportion if provided, otherwise use quantities
        if proportion:
            #normalize proportion
            total_prop = sum(proportion)
            proportion = [p / total_prop for p in proportion]

        elif quantities:
            total_quant = sum(quantities)
            proportion = [q / total_quant for q in quantities]

        else:
            raise ValueError(
                "Either proportion or quantities must be provided.")

        solvents_mols = self._xyzfiles2mols(solvents_files)
        solvents_names = [Path(f).stem
                          for f in solvents_files] if solvents_files else []

        solvents_dict = {}
        for i, solvent_molecule in enumerate(solvents_mols):
            solvent_name = solvents_names[i]
            ds, molm = self._get_density_molarmass(solvent_name)
            if ds is None or molm is None:
                raise ValueError(
                    f"Solvent {solvent_name} must have both density and molar mass provided."
                )
            solvents_dict[solvent_name] = {
                'molecule':
                solvent_molecule,
                'density':
                ds,
                'molar_mass':
                molm,
                'proportion':
                proportion[i] if proportion else 0.0,
                'target_quantity':
                quantities[i] if quantities else self._density2number(
                    ds, molm, self.box_size, proportion[i]),
                'labels':
                solvent_molecule.get_labels(),
                'coords':
                solvent_molecule.get_coordinates_in_angstrom() - np.mean(
                    solvent_molecule.get_coordinates_in_angstrom(), axis=0),
                'n_atoms':
                len(solvent_molecule.get_labels()),
                'extended_residue_idx':
                np.empty((0, 1), dtype=bool),
                'extended_com_points':
                np.empty((0, 3), dtype=float)
            }

        return solvents_dict

    def molecule_radii(self, coords):
        #get a radii of the molecule
        #align the molecule longest axis to x axis
        #use radii
        dist_from_center = np.linalg.norm(coords, axis=1)
        max_dist = np.max(dist_from_center) / 2
        return max_dist

    def mols_radii(self, solvents_dict):
        #get a radii of the molecule
        #align the molecule longest axis to x axis
        #use radii
        radii = []
        for solvent_name in solvents_dict:
            coords = solvents_dict[solvent_name]['coords']
            if solvents_dict[solvent_name]['proportion'] < 0.1:
                continue
            radii.append(self.molecule_radii(
                coords))  #*solvents_dict[solvent_name]['proportion'])
        #avg_radii = sum(radii)
        max_radii = max(radii)
        return max_radii

    def load_solute_info(self, solute_file):
        solute_labels, solute_coords = self._read_xyz(solute_file)
        solute_info = {
            'labels': solute_labels,
            'coords': solute_coords,
            'n_atoms': len(solute_labels),
        }
        return solute_info

    def grid_points_template(self, solvents_dict, box_size, grid_spacing=None):
        #generate a cubic grid points with given spacing and box size
        x_points = np.arange(0 + grid_spacing, box_size[0] - grid_spacing,
                             2 * grid_spacing)
        y_points = np.arange(0 + grid_spacing, box_size[1] - grid_spacing,
                             2 * grid_spacing)
        z_points = np.arange(0 + grid_spacing, box_size[2] - grid_spacing,
                             2 * grid_spacing)
        xx, yy, zz = np.meshgrid(x_points, y_points, z_points)
        points_template = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        if self._debug:
            self.ostream.print_info(
                f"Generated {points_template.shape[0]} template points for solvent placement with grid spacing {grid_spacing:.2f} Å."
            )
            self.ostream.flush()
        return points_template

    def solvate(self, trial_rounds=1):

        #calculate the proportion of each solvent
        #LOAD solute and solvents
        original_solvents_dict = self._initialize_solvents_dict(
            self.solvents_files, self.solvents_proportions,
            self.solvents_quantities)

        solute_dict = self.load_solute_info(self.solute_file)

        self.original_solvents_dict = original_solvents_dict
        self.solute_dict = solute_dict

        total_number = sum([
            original_solvents_dict[solvent_name]['target_quantity']
            for solvent_name in original_solvents_dict
        ])
        self.ostream.print_info(
            f"Total target solvent mols to add: {total_number}")
        self.ostream.flush()
        trial_rounds = max(1, trial_rounds)

        if total_number == 0:
            self.ostream.print_info("No solvents to add.")
            return

        if self.box_size is None:
            solute_radius = self.molecule_radii()
            self.box_size = np.array([solute_radius * 2] * 3)

        best_accepted_coords = None
        best_accepted_labels = None
        best_accepted_residues = None
        max_added = 0
        residue_idx = 0

        grid_spacing = self.mols_radii(original_solvents_dict) + self.buffer
        points_template = self.grid_points_template(original_solvents_dict,
                                                    self.box_size,
                                                    grid_spacing=grid_spacing)
        self.safe_box = [[grid_spacing, self.box_size[0]-grid_spacing],
                         [grid_spacing, self.box_size[1]-grid_spacing],
                         [grid_spacing, self.box_size[2]-grid_spacing]]

        # --- Trial loop for random seeds ---
        for trial in range(trial_rounds):
            if self._debug:
                self.ostream.print_info(
                    f"Starting trial {trial+1}/{trial_rounds}.")
                self.ostream.flush()

            #initial the setting for each trial
            candidates_res_idx = np.empty(0)
            np.random.seed(trial)  # different random seed for each trial

            #delete previous [extended_residue_idx] in solvents_dict by making a safe copy from original_solvents_dict
            solvents_dict = safe_dict_copy(original_solvents_dict)

            #reset extended residue idx for each solvent at the start of each trial
            #points template should be cubic grid points with spacing of buffer, avoid boundary

            all_candidates_data, solvents_dict, res_start_idx, all_res_com_points = self._generate_candidates(
                solvents_dict,
                points_template.shape[0],
                res_start=0,
                points_template=points_template,
                box_size=self.safe_box,
                rot=True)

            all_candidate_coords = all_candidates_data['coords'].astype(float)
            all_candidate_labels = all_candidates_data['labels']
            all_candidate_residues = all_candidates_data['residue_idx'].astype(
                int)
            candidates_res_idx = np.r_[candidates_res_idx,
                                       all_candidate_residues.flatten()]
            residue_idx += total_number

            #create a 1d empty array to store the keep mask for each round
            keep_masks = np.empty((0), dtype=bool)

            # --- Round 1 overlap removal ---
            keep_mask, drop_mask = self._remove_overlaps_kdtree(
                solute_dict['coords'], all_candidate_coords,
                all_candidate_residues)

            accepted_coords = all_candidate_coords[keep_mask]
            accepted_labels = all_candidate_labels[keep_mask]
            accepted_residues = all_candidate_residues[keep_mask]

            if self._debug:
                self.ostream.print_info(
                    f"Trial {trial+1} initial: {len(set(accepted_residues.flatten()))} added, {len(set(all_candidate_residues[drop_mask].flatten()))} left in cavity."
                )
                self.ostream.flush()
            keep_masks = np.r_[keep_masks, keep_mask]

            #cavity_coords = all_candidate_coords[drop_mask]
            #cavity_residues = all_candidate_residues[drop_mask]
            #count accepted solvent seperately on assigned residue idx

            # --- Iterative cavity filling (big round) ---
            max_fill_rounds = self.max_fill_rounds
            round_idx = 0
            round_drop_mask = None

            #the scalar is used to control the times of number of candidates generated in each round
            #scalar = 1 if self.scalar is None else self.scalar
            cavity_number = total_number - len(set(accepted_residues.flatten()))
            while round_idx < max_fill_rounds and cavity_number > 0:
                round_idx += 1
                if self._debug:
                    self.ostream.print_info(
                        f"Starting fill round {round_idx} with {cavity_number} mols to fill."
                    )
                    self.ostream.flush()

                if cavity_number == 0:
                    break
                #if too many overlap, decrease the number of possible centers
                new_points_num = cavity_number if cavity_number > 2000 else max(1000, cavity_number)
                round_all_candidates_data, solvents_dict, _, all_res_com_points = self._generate_candidates(
                    solvents_dict,
                    target_number=new_points_num,
                    res_start=res_start_idx,
                    points_template=None,
                    box_size=self.safe_box)  #debug

                res_start_idx += new_points_num

                if self._debug:
                    self.ostream.print_info(
                        f"Round {round_idx}: start kdtree overlap removal...")
                    self.ostream.flush()
                round_keep_mask, round_drop_mask = self._remove_overlaps_kdtree(
                    accepted_coords, round_all_candidates_data['coords'],
                    round_all_candidates_data['residue_idx'])
                if self._debug:
                    self.ostream.print_info(
                        f"Round {round_idx}: kdtree overlap removal done.")
                    self.ostream.flush()

                candidates_res_idx = np.r_[
                    candidates_res_idx,
                    round_all_candidates_data['residue_idx'].flatten()]

                keep_masks = np.r_[keep_masks, round_keep_mask]

                round_keep_coords = round_all_candidates_data['coords'][
                    round_keep_mask]
                round_keep_labels = round_all_candidates_data['labels'][
                    round_keep_mask]
                round_keep_residues = round_all_candidates_data['residue_idx'][
                    round_keep_mask]

                round_drop_residues = round_all_candidates_data['residue_idx'][
                    round_drop_mask]

                if self._debug:
                    self.ostream.print_info(
                        f"Round {round_idx}: {len(set(round_keep_residues.flatten()))} added, {len(set(round_drop_residues.flatten()))} left in cavity."
                    )
                self.ostream.flush()

                keep_res_num = len(set(round_keep_residues.flatten()))
                #drop_res_num = len(set(round_drop_residues.flatten()))
                cavity_number -= keep_res_num

                if cavity_number == 0:
                    break

                # Update accepted mols
                accepted_coords = np.vstack((accepted_coords, round_keep_coords))
                accepted_labels = np.r_[accepted_labels, round_keep_labels]
                accepted_residues = np.r_[accepted_residues,round_keep_residues]

            # --- Update best trial ---

            if accepted_coords.shape[0] > max_added:
                max_added = accepted_coords.shape[0]
                best_accepted_coords = accepted_coords.copy()
                best_accepted_labels = accepted_labels.copy()
                best_accepted_residues = accepted_residues.copy()
                best_accepted_total_number = len(set(best_accepted_residues.flatten()))
                best_solvents_dict = safe_dict_copy(solvents_dict)
                best_keep_masks = keep_masks.copy()
                best_candidates_res_idx = candidates_res_idx.copy()

        # --- Merge solute and best solvent trial ---
        if best_accepted_coords is not None:
            self.ostream.print_info(
                f"Best trial added {best_accepted_total_number} mols, atoms: {best_accepted_coords.shape[0]}.")
            self.ostream.flush()
            #calculate density
            #use best keeps masks to each solvent as extended residue idx to count the number of each solvent
            accepted_proportions = []
            target_proportions = []
            proportion_diff = []
            total_number_limit = total_number
            overshoot_flag = True

            for solvent_name in best_solvents_dict:
                #incase overshoot, only select beginning[:target_mol*n_atom] values
                accepted_atoms_number = best_solvents_dict[solvent_name][
                    'extended_residue_idx'][best_keep_masks].sum()
                accepted_quantity = accepted_atoms_number // best_solvents_dict[
                    solvent_name]['n_atoms']
                overshoot_flag = (accepted_quantity
                                  > best_solvents_dict[solvent_name]
                                  ['target_quantity']) or overshoot_flag
                best_solvents_dict[solvent_name][
                    'accepted_atoms_number'] = accepted_atoms_number
                best_solvents_dict[solvent_name][
                    'accepted_quantity'] = accepted_quantity
                best_solvents_dict[solvent_name]['accepted_mols_ind'] = (
                    best_solvents_dict[solvent_name]['extended_residue_idx']
                ).ravel() & best_keep_masks.ravel()
                solvent_residues = best_candidates_res_idx[
                    best_solvents_dict[solvent_name]['accepted_mols_ind']]
                #filter the accepted atoms coords and labels for this solvent by index
                resid_mask = np.isin(best_accepted_residues,
                                     np.unique(solvent_residues))
                best_solvents_dict[solvent_name][
                    'accepted_atoms_coords'] = best_accepted_coords[
                        resid_mask.flatten()]
                best_solvents_dict[solvent_name][
                    'accepted_atoms_labels'] = best_accepted_labels[
                        resid_mask.flatten()]
                #make sure the accepted proportion is not more than target proportion
                best_solvents_dict[solvent_name][
                    'accepted_proportion'] = accepted_quantity / len(
                        set(best_accepted_residues.flatten()))
                accepted_proportions.append(
                    best_solvents_dict[solvent_name]['accepted_proportion'])
                target_proportions.append(
                    best_solvents_dict[solvent_name]['proportion'])
                proportion_diff.append(
                    best_solvents_dict[solvent_name]['accepted_proportion'] -
                    best_solvents_dict[solvent_name]['proportion'])
                total_number_limit = min(
                    total_number_limit,
                    int(accepted_quantity /
                        best_solvents_dict[solvent_name]['proportion']
                        if best_solvents_dict[solvent_name]['proportion'] >
                        0 else total_number))
                best_solvents_dict[solvent_name][
                    'accepted_density'] = self._number2density(
                        accepted_quantity,
                        best_solvents_dict[solvent_name]['molar_mass'],
                        self.box_size)
                self.ostream.print_info("*" * 80)
                self.ostream.print_info(
                    f"total number limit after checking each solvent: {total_number_limit}"
                )
                self.ostream.flush()
            #if overshoot
            self.ostream.print_info("*" * 80)
            self.ostream.print_info(
                f"total number limit after checking each solvent: {total_number_limit}"
            )
            self.ostream.flush()
            if overshoot_flag:
                for solvent_name in best_solvents_dict:
                    #check if overshoot
                    limited_quantity = int(
                        total_number_limit *
                        best_solvents_dict[solvent_name]['proportion'])
                    if best_solvents_dict[solvent_name][
                            'accepted_quantity'] > limited_quantity:
                        overshoot_number = (
                            accepted_atoms_number - limited_quantity *
                            best_solvents_dict[solvent_name]['n_atoms'])
                        if self._debug:
                            self.ostream.print_info(
                                f"Overshoot {solvent_name}: will kick {overshoot_number} atoms."
                            )

                        best_solvents_dict[solvent_name][
                            'accepted_atoms_number'] = limited_quantity * best_solvents_dict[
                                solvent_name]['n_atoms']
                        best_solvents_dict[solvent_name][
                            'accepted_quantity'] = limited_quantity
                        best_solvents_dict[solvent_name][
                            'accepted_atoms_labels'] = best_solvents_dict[
                                solvent_name][
                                    'accepted_atoms_labels'][:best_solvents_dict[
                                        solvent_name]['accepted_atoms_number']]
                        best_solvents_dict[solvent_name][
                            'accepted_atoms_coords'] = best_solvents_dict[
                                solvent_name][
                                    'accepted_atoms_coords'][:best_solvents_dict[
                                        solvent_name]['accepted_atoms_number']]
                        best_solvents_dict[solvent_name][
                            'accepted_density'] = self._number2density(
                                limited_quantity,
                                best_solvents_dict[solvent_name]['molar_mass'],
                                self.box_size)
                        if self._debug:
                            self.ostream.print_info(
                                f"Kicked {overshoot_number} atoms for {solvent_name}."
                            )

            self.ostream.print_info("*" * 80)
            for solvent_name in best_solvents_dict:
                self.ostream.print_info(
                    f"Final accepted {solvent_name}: {best_solvents_dict[solvent_name]['accepted_quantity']} mols, {best_solvents_dict[solvent_name]['accepted_atoms_number']} atoms."
                )
                self.ostream.print_info(
                    f"Final accepted density of {solvent_name}: {best_solvents_dict[solvent_name]['accepted_density']:.4f} g/cm³. target density: {best_solvents_dict[solvent_name]['proportion'] * best_solvents_dict[solvent_name]['density']:.4f} g/cm³"
                )
            self.ostream.print_info("*" * 80)
            self.ostream.flush()

        else:
            self.ostream.print_warning(
                "No solvent mols were added in any trial.")
            self.ostream.flush()

        # --- Calculate density ---
        #return final_coords, final_labels
        self.best_solvents_dict = best_solvents_dict

        return best_solvents_dict

    def _density2number(self, density, molar_mass, box_size, proportion=1.0):
        """
        Calculate the number of mols that can fit in the box given the density and molar mass.
        """
        volume_A3 = np.prod(box_size)  # Å³
        volume_cm3 = volume_A3 * 1e-24  # cm³
        # Calculate number of mols
        N_A = 6.022e23  # Avogadro's number
        n_mols = int(density * volume_cm3 * N_A * proportion / molar_mass)
        return n_mols

    def _number2density(self, n_mols, molar_mass, box_size):
        """
        Calculate the density of the system given the number of mols and molar mass.
        """
        volume_A3 = np.prod(box_size)  # Å³
        volume_cm3 = volume_A3 * 1e-24  # cm³
        # Approximate mass calculation
        mass_g = n_mols * molar_mass / 6.022e23  # g
        density = mass_g / volume_cm3  # g/cm³
        return density


if __name__ == "__main__":

    def solvent_number_from_density(box_size, density, molar_mass):
        V_A3 = np.prod(box_size)  # Å³
        V_cm3 = V_A3 * 1e-24  # cm³
        N_A = 6.022e23  # Avogadro's number
        n_mols = int(density * V_cm3 * N_A / molar_mass)
        return n_mols


if __name__ == "__main__":
    import time
    packer = SolventPacker()
    packer.box_size = np.array([100, 100, 100])  # Å
    packer.buffer = 1.8  # Å
    packer.max_fill_rounds = 400
    start_time = time.time()
    best_solvents_dict, best_keep_masks = packer.solvate(
        #solute_file="output/UiO-66_mofbuilder_output.xyz",
        solute_file="water.xyz",
        solvents_files=["water.xyz", "dmso.xyz"],
        target_solvents_numbers=[33000, 0000],
        output_file="solvated_structure.xyz",
        trial_rounds=1)
    print("Total time (s):", time.time() - start_time)

    import veloxchem as vlx
    water = vlx.Molecule.read_xyz_file("water.xyz")
    dmso = vlx.Molecule.read_xyz_file("dmso.xyz")
    solb = vlx.SolvationBuilder()
    solb.write_gromacs_files = True
    restart_time = time.time()
    #solb.solvate(solute=water,solvent='spce',box=[100,100,100])
    #solb.custom_solvate(solute=dmso,solvents=[dmso],quantities=[5288],box=[100,100,100])
    print("Restart time (s):", time.time() - restart_time)

    #packer.buffer = 6
    #best_solvents_dict, best_keep_masks = packer.solvate(
    #    solute_file="solvated_structure.xyz",
    #    #solute_file="water.xyz",
    #    solvents_files=["water.xyz", "dmso.xyz"],
    #    target_solvents_numbers=[200, 0],
    #    output_file="1solvated_structure.xyz",
    #    trial_rounds=10)
    ##print(best_solvents_dict)
