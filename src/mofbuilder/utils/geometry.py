import numpy as np
# util functions for Cartesian and fractional coordinates conversion


def unit_cell_to_cartesian_matrix(aL, bL, cL, alpha, beta, gamma):
    pi = np.pi
    """Convert unit cell parameters to a Cartesian transformation matrix."""
    aL, bL, cL, alpha, beta, gamma = list(
        map(float, (aL, bL, cL, alpha, beta, gamma)))
    ax = aL
    ay = 0.0
    az = 0.0
    bx = bL * np.cos(gamma * pi / 180.0)
    by = bL * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = cL * np.cos(beta * pi / 180.0)
    cy = (cL * bL * np.cos(alpha * pi / 180.0) - bx * cx) / by
    cz = (cL**2.0 - cx**2.0 - cy**2.0)**0.5
    unit_cell = np.asarray([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T
    return unit_cell


def fractional_to_cartesian(fractional_coords, T):
    T = T.astype(float)
    fractional_coords = fractional_coords.astype(float)
    """Convert fractional coordinates to Cartesian using the transformation matrix."""
    return np.dot(T, fractional_coords.T).T


def cartesian_to_fractional(cartesian_coords, unit_cell_inv):
    cartesian_coords = cartesian_coords.astype(float)
    unit_cell_inv = unit_cell_inv.astype(float)
    """Convert Cartesian coordinates to fractional coordinates using the inverse transformation matrix."""
    return np.dot(unit_cell_inv, cartesian_coords.T).T


def locate_min_idx(a_array):
    # print(a_array,np.min(a_array))
    idx = np.argmin(a_array)
    row_idx = idx // a_array.shape[1]
    col_idx = idx % a_array.shape[1]
    return row_idx, col_idx


def reorthogonalize_matrix(matrix):
    """
    Ensure the matrix is a valid rotation matrix with determinant = 1.
    """
    U, _, Vt = np.linalg.svd(matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    return R


def find_optimal_pairings(node_i_positions, node_j_positions):
    """
    Find the optimal one-to-one pairing between atoms in two nodes using the Hungarian algorithm.
    """
    num_i, num_j = len(node_i_positions), len(node_j_positions)
    cost_matrix = np.zeros((num_i, num_j))
    for i in range(num_i):
        for j in range(num_j):
            cost_matrix[i, j] = np.linalg.norm(node_i_positions[i, 1:] -
                                               node_j_positions[j, 1:])

    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # print(cost_matrix.shape) #DEBUG
    row_ind, col_ind = locate_min_idx(cost_matrix)
    # print(row_ind,col_ind,cost_matrix) #DEBUG

    return [row_ind, col_ind]


def find_edge_pairings(sorted_nodes, sorted_edges, atom_positions):
    """
    Identify optimal pairings for each edge in the graph.

    Parameters:
        G (networkx.Graph): Graph structure with edges between nodes.
        atom_positions (dict): Positions of X atoms for each node.

    Returns:
        dict: Mapping of edges to optimal atom pairs.
            Example: {(0, 1): [(0, 3), (1, 2)], ...}
    """

    edge_pairings = {}

    for i, j in sorted_edges:
        node_i_positions = atom_positions[i]  # [index,x,y,z]
        node_j_positions = atom_positions[j]  # [index,x,y,z]

        # Find optimal pairings for this edge

        pairs = find_optimal_pairings(node_i_positions, node_j_positions)
        # print(sorted_nodes[i],sorted_nodes[j],pairs) #DEBUG
        edge_pairings[(i, j)] = pairs  # update_pairs(pairs,atom_positions,i,j)
        # idx_0,idx_1 = pairs[0]
        # x_idx_0 = atom_positions[i][idx_0][0]
        # x_idx_1 = atom_positions[j][idx_1][0]
    #
    # edge_pairings[(i, j)] = update_pairs(pairs,atom_positions,i,j) #but only first pair match
    # atom_positions[i] = np.delete(atom_positions[i], idx_0, axis=0)
    # atom_positions[j] = np.delete(atom_positions[j], idx_1, axis=0)

    return edge_pairings


def Carte_points_generator(xyz_num):
    """Generate a 3D grid of points with integer coordinates.
    
    Parameters:
        xyz_num (tuple): Number of divisions in x, y, and z directions.
        
    Returns:
        ndarray: Array of points with shape (n, 3).
    """
    x_num, y_num, z_num = xyz_num
    # Use meshgrid for efficient point generation
    x = np.arange(x_num + 1)
    y = np.arange(y_num + 1)
    z = np.arange(z_num + 1)
    
    # Create meshgrid and reshape to get all points
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    
    return points



