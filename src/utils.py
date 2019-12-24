import numpy as np


def Union(lists):
    # Python program to illustrate union
    # Without repetition
    final_list = list(set().union(*lists))
    return sorted(final_list)


def sort_by_l2norm(matrix, order='ascending'):
    '''
    Input:
        matrix: an 2D array of shape (# of features, feature dimension)
    Output:
        sorted_matrix: an matrix sorted along with axis 0
        original_idx: indices that indicate which original
                      positions the ranked rows originally belonged to
    '''
    assert order in ['ascending', 'descending']
    sorted_idxs = np.argsort(np.linalg.norm(matrix, axis=-1))
    sorted_matrix = matrix[sorted_idxs]
    original_idx = np.zeros(sorted_idxs.shape[0], dtype='uint8')
    for i, rank in enumerate(sorted_idxs):
        original_idx[rank] = i
    if order == 'descending':
        sorted_matrix = sorted_matrix[::-1]
        original_idx = original_idx[::-1]
    return sorted_matrix, original_idx
