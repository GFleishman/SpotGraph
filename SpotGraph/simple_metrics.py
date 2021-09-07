import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def cells_per_neighborhood(neighborhood_matrix):
    """
    Count the number of unique cells in every neighborhood

    Parameters
    ----------
    neighborhood_matrix : NxM array
        The array returned by ``extract_neighborhoods``

    Returns
    -------
    counts : 1D array
        Length N
    """

    # initialize container
    counts = np.empty(neighborhood_matrix.shape[0], dtype=int)

    # loop over all neighborhoods
    for iii, neighborhood in enumerate(neighborhood_matrix):
        unique = np.unique(neighborhood[3:])
        counts[iii]= len(unique)-1 if 0. in unique else len(unique)

    # return
    return counts


def distances(neighborhood_matrix):
    """
    Compute distance to all cells within the neighborhood

    Parameters
    ----------
    neighborhood_matrix : NxM array
        The array returned by ``extract_neighborhoods``

    Returns
    -------
    distances : list
        list of list of tuples; For each spot there is a list
        of tuples. Each tuple is a segment ID and the distance
        of the spot to that segment ID.
    """

    # initialize container
    distances = []

    # get edge size and center coordinate array
    edge = int(round(np.cbrt(len(neighborhood_matrix[0, 3:]))))
    center = np.array([ (edge-1)/2, ]*3)[None, :]

    # loop over all neighborhoods
    for iii, neighborhood in enumerate(neighborhood_matrix):

        # reformat neighborhood and get unique labels
        neighborhood = neighborhood[3:].reshape((edge,)*3)
        unique = np.unique(neighborhood)
        unique = [x for x in unique if x != 0]
        
        # initialize container
        dists = np.empty(len(unique), dtype=float)

        # loop over unique labels
        for jjj, label in enumerate(unique):
            coords = np.column_stack(np.nonzero(neighborhood == label))
            dists[jjj] = cdist(center, coords).squeeze().min()

        # sort
        inds = np.argsort(dists)
        unique, dists = np.array(unique)[inds], np.array(dists)[inds]

        # package and append
        distances.append(zip(unique, dists))

    # final return
    return distances


def nearest_neighbors(spots, max_distance):
    """
    Get all spot pairs that are less than a given distance apart

    Parameters
    ----------
    spots : Nx3 array
        array of spot coordinates
    max_distance : float
        distance threshold for point pairs

    Returns
    -------
    pairs : dict
        key (i, j) gives distances between points i and j
    """

    # put spots into KDTree, compute distances within threshold
    tree = cKDTree(spots)
    return tree.sparse_distance_matrix(tree, max_distance, output_type='dict')

