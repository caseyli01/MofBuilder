import itertools
import numpy as np


def sort_by_distance(arr):
    # Calculate distances from the first element to all other elements
    distances = [(np.linalg.norm(arr[0] - arr[i]), i) for i in range(len(arr))]
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])
    return distances


def match_vectors(arr1, arr2, num):
    # Get sorted distances
    sorted_distances_arr1 = sort_by_distance(arr1)
    sorted_distances_arr2 = sort_by_distance(arr2)

    # Select the indices by distance matching in limited number

    indices_arr1 = [sorted_distances_arr1[j][1] for j in range(num)]
    indices_arr2 = [sorted_distances_arr2[j][1] for j in range(num)]

    # reorder the matching vectors# which can induce the smallest RMSD
    closest_vectors_arr1 = np.array([arr1[i] for i in indices_arr1])
    closest_vectors_arr2 = np.array([arr2[i] for i in indices_arr2])

    return closest_vectors_arr1, closest_vectors_arr2

def superimpose(src_arr, target_arr, min_rmsd=1e6):
    """
    Find the best rotation and translation that aligns src_arr to target_arr.

    Procedure:
    - Convert inputs to numpy arrays.
    - Select up to 6 matching vectors from each set based on distance patterns
      (using match_vectors). This reduces the search space for correspondences.
    - Try all permutations of the selected vectors from arr1 and compute the
      SVD-based superposition against the selected vectors from arr2.
    - Keep the rotation/translation that yields the smallest RMSD.

    Returns:
    - min_rmsd: best RMSD found
    - best_rot: 3x3 rotation matrix
    - best_tran: translation vector (length 3)
    """
    # Ensure inputs are numpy arrays
    src_arr = np.asarray(src_arr)
    target_arr = np.asarray(target_arr)

    # Select up to 6 representative vectors from each array to match by distance
    m_src, m_target = match_vectors(src_arr, target_arr, min(6, len(src_arr), len(target_arr)))

    # Initialize best transformation to identity/no-translation
    best_rot, best_tran = np.eye(3), np.zeros(3)

    # Try every possible correspondence (permutation) of the selected vectors
    for perm in itertools.permutations(m_src):
        # Compute RMSD, rotation and translation for this correspondence
        rmsd, rot, tran = svd_superimpose(np.asarray(perm), m_target)

        # Keep the transform that gives the smallest RMSD
        if rmsd < min_rmsd:
            min_rmsd, best_rot, best_tran = rmsd, rot, tran

    return min_rmsd, best_rot, best_tran


def svd_superimpose(src_arr, target_arr):
    """
    Calculates RMSD and rotation matrix for superimposing two sets of points,
    using SVD. Ref.: "Least-Squares Fitting of Two 3-D Point Sets", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, 1987, PAMI-9(5),
    698-700. DOI: 10.1109/TPAMI.1987.4767965
    """

    src_arr = np.array(src_arr)
    target_arr = np.array(target_arr)

    com1 = np.sum(src_arr, axis=0) / src_arr.shape[0]
    com2 = np.sum(target_arr, axis=0) / target_arr.shape[0]

    src_arr -= com1
    target_arr -= com2

    cov_mat = np.matmul(src_arr.T, target_arr)
    U, s, Vt = np.linalg.svd(cov_mat)

    rot_mat = np.matmul(U, Vt)
    if np.linalg.det(rot_mat) < 0:
        Vt[-1, :] *= -1.0
        rot_mat = np.matmul(U, Vt)

    diff = target_arr - np.matmul(src_arr, rot_mat)
    rmsd = np.sqrt(np.sum(diff**2) / diff.shape[0])
    trans = com2 - np.dot(com1, rot_mat)

    return rmsd, rot_mat, trans


def superimpose_rotation_only(arr1, arr2, min_rmsd=1e6):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    m_arr1, m_arr2 = match_vectors(arr1, arr2, min(6, len(arr1), len(arr2)))
    best_rot, best_tran = np.eye(3), np.zeros(3)
    for perm in itertools.permutations(m_arr1):
        rmsd, rot, tran = svd_superimpose(np.asarray(perm), m_arr2)
        if rmsd < min_rmsd:
            min_rmsd, best_rot, best_tran = rmsd, rot, tran
            if np.allclose(np.dot(best_tran, np.zeros(3)), 1e-1):
                break

    return min_rmsd, best_rot, best_tran