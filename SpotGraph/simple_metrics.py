import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix


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


def distances(neighborhood_matrix, kernel=None, kernel_kwargs={}, normalize=False):
    """
    Compute distance to all cells within the neighborhood

    Parameters
    ----------
    neighborhood_matrix : Nx3 array
        The array returned by ``extract_neighborhoods``
    kernel : callable
        Function to apply to distance, default None
    kernel_kwargs : dict
        Any arguments to kernel function
    normalize : bool
        Normalize the distribution of distances for each node

    Returns
    -------
    distances : scipy.sparse.dok_matrix
        NxM sparse matrix; key (i, j) is the distance from
        spot i to cell j; if cell j is not in the neighborhood
        of spot i then the distance is 0.
    """

    # initialize container
    nspots = neighborhood_matrix.shape[0]
    ncells = neighborhood_matrix[:, 3:].max() + 1
    distances = dok_matrix((nspots, ncells), dtype=float)

    # get edge size and center coordinate
    edge = int(round(np.cbrt(len(neighborhood_matrix[0, 3:]))))
    center = np.array([ (edge-1)/2, ]*3)[None, :]

    # loop over all neighborhoods
    for iii, neighborhood in enumerate(neighborhood_matrix):

        # reformat neighborhood and get unique labels
        neighborhood = neighborhood[3:].reshape((edge,)*3)
        unique = np.unique(neighborhood)
        unique = [x for x in unique if x != 0]  # 0 is background

        # initialize local container
        dists = np.empty(len(unique), dtype=float)

        # loop over unique labels
        for jjj, label in enumerate(unique):
            coords = np.column_stack(np.nonzero(neighborhood == label))
            d = cdist(center, coords).squeeze().min()
            if d == 0: d = 1e-20  # distinguish from dok_matrix fill value
            if kernel: d = kernel(d, **kernel_kwargs)
            dists[jjj] = d

        # normalize
        if normalize: dists = dists / np.sum(dists)

        # put in sprase matrix
        for label, d in zip(unique, dists):
            distances[iii, label] = d

    # final return
    return distances


def nearest_neighbors(spots, max_distance, distance_to_self=1e-20):
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
    pairs : scipy.sparse.dok_matrix
        key (i, j) gives distances between points i and j
    """

    # put spots into KDTree, compute distances within threshold
    tree = cKDTree(spots)
    dok = tree.sparse_distance_matrix(tree, max_distance)
    dok.setdiag(distance_to_self)  # to distinguish from fill value
    return dok


def exponential_kernel(distance, sigma):
    """
    Evaluate e^(-``distance`` / ``sigma``)

    Parameters
    ----------
    distance : float
        Positive scalar value
    sigma : float
        Standard deviation of kernel

    Returns
    -------
    exponentiated distance value e^(-``distance`` / ``sigma``)
    """

    return np.exp(-distance / sigma)

