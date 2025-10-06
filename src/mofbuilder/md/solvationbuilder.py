import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


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

    def __init__(self):
        self.buffer = 1.5  # Å
        self.box_size = None
        self.max_fill_rounds = 1000  # Maximum number of filling rounds

    def read_xyz(self, filename):
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
        return labels, np.array(coords)

    def _generate_candidates_each_solvent(self,
                                         solvent_coords,
                                         solvent_labels,
                                         solvent_n_atoms,
                                         target_mol_number,
                                         residue_idx_start=0,
                                         points_template=None):

        if points_template is None:
            random_points = np.random.rand(target_mol_number,
                                           3) * self.box_size
        else:
            random_points = points_template
        target_mol_number = random_points.shape[0]
        if target_mol_number == 0:
            return np.empty((0, 3)), np.empty((0, 1)), np.empty((0, 1))

        rots = R.random(target_mol_number).as_matrix()
        coords_exp = solvent_coords[np.newaxis, :, :]
        rot_coords = np.matmul(coords_exp, rots.transpose(0, 2, 1))
        candidates = rot_coords.reshape(-1, 3)
        candidates += np.repeat(random_points, solvent_n_atoms, axis=0)

        labels = np.array(list(solvent_labels) * target_mol_number).reshape(
            -1, 1)
        residue_idx = np.repeat(
            np.arange(residue_idx_start,
                      residue_idx_start + target_mol_number),
            solvent_n_atoms).reshape(-1, 1)

        return candidates, labels, residue_idx

    def _remove_overlaps_kdtree(self, existing_coords, candidate_coords,
                               candidate_residues):
        candidate_residues = candidate_residues.reshape(-1)

        # Round 1: overlap with existing atoms
        tree_existing = cKDTree(existing_coords)
        neighbors_existing = tree_existing.query_ball_point(candidate_coords,
                                                            r=self.buffer)
        mask_overlap_existing = np.array(
            [len(neigh) > 0 for neigh in neighbors_existing], dtype=bool)
        bad_residues_existing = np.unique(
            candidate_residues[mask_overlap_existing])

        # Round 2: overlap among candidates
        tree_candidates = cKDTree(candidate_coords)
        neighbors_candidates = tree_candidates.query_ball_point(
            candidate_coords, r=self.buffer)
        mask_overlap_candidates = np.array([
            any(candidate_residues[neigh] != candidate_residues[i]
                for neigh in neighbors_candidates[i] if neigh != i)
            for i in range(len(candidate_coords))
        ]).astype(bool)
        bad_residues_candidates = np.unique(
            candidate_residues[mask_overlap_candidates])

        bad_residues = np.union1d(bad_residues_existing,
                                  bad_residues_candidates)
        keep_mask = ~np.isin(candidate_residues, bad_residues)
        drop_mask = ~keep_mask
        return keep_mask, drop_mask

    def _cluster_candidates(self, candidate_coords, candidate_residues,
                           cluster_radius):
        candidate_residues = candidate_residues.reshape(-1)
        unique_residues = np.unique(candidate_residues)
        centers = np.array([
            candidate_coords[candidate_residues == res].astype(
                np.float32).mean(axis=0) for res in unique_residues
        ])

        diff = centers[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        neighbors_mask = dist2 < cluster_radius**2
        np.fill_diagonal(neighbors_mask, False)

        isolated_mask = ~np.any(neighbors_mask, axis=1)
        accepted_centers = centers[isolated_mask]
        accepted_residues = unique_residues[isolated_mask]

        return accepted_centers, accepted_residues

    def _generate_candidates(self, sol_dict, target_number, res_start=0):

        all_data = {}
        all_sol_mols = []
        for solvent_name in sol_dict:
            n_mol = int(target_number * sol_dict[solvent_name]['ratio'])
            if n_mol == 0:
                if sol_dict[solvent_name]['ratio'] > 0:
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

        for i, solvent_name in enumerate(sol_dict):
            #solvents_dict[solvent_name]['extended_residue_idx'] = np.empty((0, all_candidates_data['atoms_number']), dtype=bool)
            _target_mol_number = all_sol_mols[i]

            candidates, labels, residue_idx = self._generate_candidates_each_solvent(
                sol_dict[solvent_name]['coords'],
                sol_dict[solvent_name]['labels'],
                sol_dict[solvent_name]['n_atoms'],
                _target_mol_number,
                residue_idx_start=res_start)
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

            all_data['coords'] = np.vstack((all_data['coords'], candidates))
            all_data['labels'] = np.vstack((all_data['labels'], labels))
            all_data['residue_idx'] = np.vstack(
                (all_data['residue_idx'], residue_idx))

        return all_data, sol_dict, res_start

    def solvate(
            self,
            solute_file,
            solvents_files,
            target_solvents_numbers=[0],  #number of solvent molecules
            output_file="solvated_structure.xyz",
            trial_rounds=10):

        # --- read solute and solvents ---
        original_solvents_dict = {}
        #calculate the ratio of each solvent
        total_number = sum(target_solvents_numbers)
        if total_number == 0:
            print("No solvents to add.")
            return
        residue_idx = 0

        for i, solvent_name in enumerate(solvents_files):
            solvent_file = solvents_files[i]
            solvent_labels, solvent_coords = self.read_xyz(solvent_file)
            original_solvents_dict[solvent_name] = {
                'file': solvent_file,
                'labels': solvent_labels,
                'coords': solvent_coords,
                'n_atoms': len(solvent_labels),
                'ratio': target_solvents_numbers[i] / total_number,
                'target_molecules_number': target_solvents_numbers[i],
                'extended_residue_idx': np.empty((0, 1), dtype=bool)
            }
        #print("Solvent ratios:", {
        #    k: original_solvents_dict[k]['ratio']
        #    for k in original_solvents_dict
        #})

        solute_labels, solute_coords = self.read_xyz(solute_file)

        best_accepted_coords = None
        best_accepted_labels = None
        max_added = 0

        # --- Trial loop for random seeds ---
        for trial in range(trial_rounds):
            print("-" * 40)
            print(f"Trial {trial+1}/{trial_rounds}")

            candidates_res_idx = np.empty(0)
            np.random.seed(trial)  # different random seed for each trial
            #delete previous extended residue idx
            solvents_dict = safe_dict_copy(original_solvents_dict)

            #reset extended residue idx for each solvent at the start of each trial
            # --- Generate initial solvent candidates ---
            all_candidates_data, solvents_dict, res_start_idx = self._generate_candidates(
                solvents_dict, total_number, res_start=0)

            all_candidate_coords = all_candidates_data['coords'].astype(float)
            all_candidate_labels = all_candidates_data['labels']
            all_candidate_residues = all_candidates_data['residue_idx'].astype(
                int)

            candidates_res_idx = np.r_[candidates_res_idx,
                                       all_candidate_residues.flatten()]
            residue_idx += total_number

            #creat a 1d empty array to store the keep mask for each round

            keep_masks = np.empty((0), dtype=bool)
            # --- Round 1 overlap removal ---
            keep_mask, drop_mask = self._remove_overlaps_kdtree(
                solute_coords, all_candidate_coords, all_candidate_residues)

            accepted_coords = all_candidate_coords[keep_mask]
            accepted_labels = all_candidate_labels[keep_mask]
            accepted_residues = all_candidate_residues[keep_mask]

            keep_masks = np.r_[keep_masks, keep_mask]

            cavity_coords = all_candidate_coords[drop_mask]
            cavity_residues = all_candidate_residues[drop_mask]
            #count accepted water and dmso based on residue idx

            print(
                f"After Round 1 overlap removal: {len(set(accepted_residues.flatten()))} accepted, {len(set(cavity_residues.flatten()))} left in cavity."
            )

            # --- Iterative cavity filling (big round) ---
            max_fill_rounds = self.max_fill_rounds
            round_idx = 0
            while round_idx < max_fill_rounds and cavity_coords.shape[0] > 0:
                round_idx += 1
                possible_centers, _ = self._cluster_candidates(
                    cavity_coords, cavity_residues, cluster_radius=self.buffer)

                if possible_centers.shape[0] == 0:
                    break

                round_all_candidates_data, solvents_dict, _ = self._generate_candidates(
                    solvents_dict,
                    target_number=possible_centers.shape[0],
                    res_start=res_start_idx)

                residue_idx += possible_centers.shape[0]

                round_keep_mask, round_drop_mask = self._remove_overlaps_kdtree(
                    accepted_coords, round_all_candidates_data['coords'],
                    round_all_candidates_data['residue_idx'])

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

                print(
                    f"Round {round_idx}: {len(set(round_keep_residues.flatten()))} added, {len(set(round_drop_residues.flatten()))} left in cavity."
                )

                if round_keep_coords.shape[0] == 0:
                    break

                # Update accepted molecules
                accepted_coords = np.vstack(
                    (accepted_coords, round_keep_coords))
                accepted_labels = np.r_[accepted_labels, round_keep_labels]
                accepted_residues = np.r_[accepted_residues,
                                          round_keep_residues]

                # Update cavity for next iteration
                cavity_coords = round_all_candidates_data['coords'][
                    round_drop_mask]
                cavity_residues = round_all_candidates_data['residue_idx'][
                    round_drop_mask]

            # --- Update best trial ---

            n_added = accepted_coords.shape[0]

            if n_added > max_added:
                max_added = n_added
                best_accepted_coords = accepted_coords.copy()
                best_accepted_labels = accepted_labels.copy()
                best_accepted_residues = accepted_residues.copy()

                best_solvents_dict = safe_dict_copy(solvents_dict)
                best_keep_masks = keep_masks.copy()
                best_candidates_res_idx = candidates_res_idx.copy()

        # --- Merge solute and best solvent trial ---
        if best_accepted_coords is not None:
            print("*" * 80)
            final_coords = np.vstack(
                (solute_coords, best_accepted_coords.astype(float)))
            final_labels = np.r_[np.array(solute_labels),
                                 best_accepted_labels.flatten()]
            solute_residues = np.array([-1] * len(solute_labels))
            final_residues = np.r_[solute_residues,
                                   best_accepted_residues.flatten()]

            #calculate density
            #use best keeps masks to each solvent as extended residue idx to count the number of each solvent

            kick_res_idx = []
            for solvent_name in best_solvents_dict:
                #incase overshoot, only select beginning[:target_mol*n_atom] values
                accepted_atoms_number = best_solvents_dict[solvent_name][
                    'extended_residue_idx'][best_keep_masks].sum()
                accepted_molecules_number = accepted_atoms_number // best_solvents_dict[
                    solvent_name]['n_atoms']

                best_solvents_dict[solvent_name][
                    'accepted_atoms_number'] = accepted_atoms_number
                best_solvents_dict[solvent_name][
                    'accepted_molecules_number'] = accepted_molecules_number
                best_solvents_dict[solvent_name]['accepted_molecules_ind'] = (
                    best_solvents_dict[solvent_name]['extended_residue_idx']
                ).ravel() & best_keep_masks.ravel()

                def flip_true_in_mask(mask, flip_number):
                    true_idx = np.where(mask)[0]
                    mask[true_idx[-flip_number:]] = False
                    return mask, true_idx[-flip_number:]

                if accepted_molecules_number > best_solvents_dict[
                        solvent_name]['target_molecules_number']:
                    overshoot_number = (
                        accepted_atoms_number -
                        (best_solvents_dict[solvent_name]
                         ['target_molecules_number'] *
                         best_solvents_dict[solvent_name]['n_atoms']))
                    print(
                        f"Overshoot {solvent_name}: will kick {overshoot_number} atoms."
                    )
                    best_solvents_dict[solvent_name][
                        'accepted_atoms_number'] = best_solvents_dict[
                            solvent_name][
                                'target_molecules_number'] * best_solvents_dict[
                                    solvent_name]['n_atoms']
                    best_solvents_dict[solvent_name][
                        'accepted_molecules_number'] = best_solvents_dict[
                            solvent_name]['target_molecules_number']

                    best_solvents_dict[solvent_name][
                        'accepted_molecules_ind'], flipped_idx = flip_true_in_mask(
                            best_solvents_dict[solvent_name]
                            ['accepted_molecules_ind'], overshoot_number)

                    kick_res_idx.extend(
                        list(set(best_candidates_res_idx[flipped_idx])))
                    print(
                        f"Accepted {best_solvents_dict[solvent_name]['accepted_molecules_number']} {solvent_name} molecules ({best_solvents_dict[solvent_name]['accepted_atoms_number']} atoms)."
                    )

            if kick_res_idx:
                print(f"will kick residue {kick_res_idx}")
                mask = ~np.isin(final_residues, np.array(kick_res_idx))
                print(f"before kick{final_coords.shape}")
                final_coords = final_coords[mask]
                final_labels = final_labels[mask]
                final_residues = final_residues[mask]
                print(f"after kick{final_coords.shape}")

            print("*" * 80)
            print(
                "Final solvent composition:", {
                    k: best_solvents_dict[k]['accepted_molecules_number']
                    for k in best_solvents_dict
                })

            #water_molar_mass = 18.015  # g/mol
            #dmso_molar_mass = 78.13  # g/mol
            #added_water_density = self.calculated_density(added_water_num, water_molar_mass)
            #added_dmso_density = self.calculated_density(added_dmso_num, dmso_molar_mass)


#
#final_water_density = added_water_density
#final_dmso_density = added_dmso_density
#if self.verbose:
#    print("Final water density (g/cm³):", final_water_density)
#    print("Final dmso density (g/cm³):", final_dmso_density)
        else:
            raise ValueError(
                "No valid solvent molecules were added in any trial.")

        self.write_solvated_structure(final_coords, final_labels,
                                      final_residues, output_file)

        # --- Calculate density ---
        #return final_coords, final_labels
        return best_solvents_dict, best_keep_masks

    def write_solvated_structure(self, final_coords, final_labels,
                                 final_residues, output_file):
        with open(output_file, "w") as fp:
            fp.write(f"{final_coords.shape[0]}\n")
            fp.write("Solvated structure\n")
            for label, (x, y, z), note in zip(final_labels, final_coords,
                                              final_residues):
                fp.write(f"{label:5s} {x:.4f} {y:.4f} {z:.4f} {note}\n")
        print(f"Wrote solvated structure to {output_file}")
        print(f" Total atoms: {final_coords.shape[0]}")

    def calculated_density(self, n_molecules, molar_mass):
        """
            Calculate the density of the solvated system in g/cm³.

            Returns
            -------
            density : float
                Density in g/cm³.
            """
        if self.box_size is None:
            raise ValueError("Box size is not set.")

        volume_A3 = np.prod(self.box_size)  # Å³
        volume_cm3 = volume_A3 * 1e-24  # cm³

        # Approximate mass calculation
        mass_g = n_molecules * molar_mass / 6.022e23  # g
        density = mass_g / volume_cm3  # g/cm³
        return density

if __name__ == "__main__":

    def solvent_number_from_density(box_size, density, molar_mass):
        V_A3 = np.prod(box_size)  # Å³
        V_cm3 = V_A3 * 1e-24  # cm³
        N_A = 6.022e23  # Avogadro's number
        n_molecules = int(density * V_cm3 * N_A / molar_mass)
        return n_molecules

if __name__ == "__main__":
    packer = SolventPacker()
    packer.box_size = np.array([10, 10, 10])
    packer.buffer = 2  # Å
    packer.max_fill_rounds = 5000
    best_solvents_dict, best_keep_masks = packer.solvate(
        #solute_file="output/UiO-66_mofbuilder_output.xyz",
        solute_file="water.xyz",
        solvents_files=["water.xyz", "dmso.xyz"],
        target_solvents_numbers=[50, 1],
        output_file="solvated_structure.xyz",
        trial_rounds=300)
    packer.buffer=6
    best_solvents_dict, best_keep_masks = packer.solvate(
        solute_file="solvated_structure.xyz",
        #solute_file="water.xyz",
        solvents_files=["water.xyz", "dmso.xyz"],
        target_solvents_numbers=[20, 0],
        output_file="1solvated_structure.xyz",
        trial_rounds=300)
    #print(best_solvents_dict)