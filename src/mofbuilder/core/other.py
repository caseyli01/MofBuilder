import re
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from ..utils.geometry import fractional_to_cartesian
# Function to fetch indices and coordinates of atoms with a specific label
def fetch_X_atoms_ind_array(array, column, X):
    # array: input array
    # column: column index to check for label
    # X: label to search for

    ind = [
        k for k in range(len(array)) if re.sub(r"\d", "", array[k, column]) == X
    ]
    x_array = array[ind]
    return ind, x_array


def find_pair_x_edge_fc(x_matrix, edge_matrix, sc_unit_cell):
        dist_matrix = np.zeros((len(x_matrix), len(edge_matrix)))
        x_matrix = fractional_to_cartesian(x_matrix, sc_unit_cell)
        edge_matrix = fractional_to_cartesian(edge_matrix, sc_unit_cell)
        for i in range(len(x_matrix)):
            for j in range(len(edge_matrix)):
                dist_matrix[i, j] = np.linalg.norm(x_matrix[i] - edge_matrix[j])
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return row_ind, col_ind

def order_edge_array(row_ind, col_ind, edges_array):
    # according col_ind order to reorder the connected edge points
    old_split = np.vsplit(edges_array, len(col_ind))
    old_order = []
    for i in range(len(col_ind)):
        old_order.append((row_ind[i], col_ind[i],
                        old_split[sorted(col_ind).index(col_ind[i])]))
    new_order = sorted(old_order, key=lambda x: x[0])
    ordered_arr = np.vstack([new_order[j][2] for j in range(len(new_order))])
    return ordered_arr