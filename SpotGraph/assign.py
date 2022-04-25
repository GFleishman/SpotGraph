import numpy as np


def hard_assign(
    metric_dok,
    spot_counts=None,
    min_or_max='min',
    threshold=None,
    return_assignments=True,
):
    """
    Assign every spot to a single segment; count number of spots assigned
    to each segment. Values in `metric_dok` determine assignment. Minimum
    or maximum values can be used. An absolute threshold can be given.

    Parameters
    ----------
    metric_dok : NxM scipy.sparse.dok_matrix
        N is the number of spots, M is the number of segments. Entry (i, j)
        scores the affinity of spot i for segment j. For example, the output
        of SpotGraph.simple_metrics.spot_to_cell_distances can be used
        directly.
    min_or_max : string (Default: 'min')
        Only two options: 'min' or 'max'. Determines whether the minimum or
        maximum score along each row of `metric_dok` is used for assignment.
        E.g. use 'min' for distances and use 'max' for probabilities.
    threshold : float (Default: None)
        The optimal metric value must still surpass this threshold to assign
        the spot. E.g. 0.5 microns for distances or 0.75 for probabilities.
    return_assignments : bool (Default: True)
        In addition to the number of spots assigned each segment, return
        the assigned segment id for each spot.

    Returns
    -------
    counts : 1-d numpy array. Length (M,)
        The number of spots assigned to each of the M segments
    assignments : 1-d numpy array. Length (N,)    Optional
        The segment id for each of the N spots. 0 means the spot was not
        assigned a segment.
    """

    # using min or max functions
    if min_or_max == 'min':
        fff = lambda x: np.argmin(x)
        ggg = (lambda x: True) if threshold is None else (lambda x: x < threshold)
    elif min_or_max == 'max':
        fff = lambda x: np.argmax(x)
        ggg = (lambda x: True) if threshold is None else (lambda x: x > threshold)
    else:
        raise ValueError("min_or_max must be 'min' or 'max'")
    
    # convert format
    metric_csr = metric_dok.tocsr()

    # number of spots and segments
    nspots = metric_csr.shape[0]
    nsegments = metric_csr.shape[1]

    # container for results
    assignments = np.zeros(nspots, dtype=np.uint32)
    counts = np.zeros(nsegments, dtype=np.uint32)

    # create a spot_counts array
    if spot_counts is None:
        spot_counts = np.ones(nspots, dtype=np.uint8)

    # loop over spots, assign
    for iii in range(nspots):

        # get distances and segment ids
        row = metric_csr.getrow(iii)
        dists = row.data
        segment_ids = row.indices

        # make sure there are possible assignments
        if len(dists) > 0:

            # get best distance and segment assignment
            indx = fff(dists)
            dist = dists[indx]
            segment_id = segment_ids[indx]

            # check threshold and assign
            if ggg(dist):
                assignments[iii] = segment_id
                counts[segment_id] += spot_counts[iii]

    # return counts
    if return_assignments:
        return counts, assignments
    else:
        return counts


def soft_assign(metric_dok):
    """
    Assign every spot to multiple segments in a weighted fashion.

    Parameters
    ----------
    metric_dok : NxM scipy.sparse.dok_matrix
        N is the number of spots, M is the number of segments. Entry (i, j)
        scores the affinity of spot i for segment j. For example, the output
        of SpotGraph.simple_metrics.spot_to_cell_distances can be used
        directly. This function really only makes sense if rows of `metric_dok`
        are normalized probability distributions.

    Returns
    -------
    counts : 1d numpy array of length (M,)
        The (soft) number of spots assigned to each segment. These are
        potentially non-integer floating point values, since spots are
        assigned as distributions.
    """

    # convert format
    metric_csr = metric_dok.tocsr()

    # number of spots and segments
    nspots = metric_csr.shape[0]
    nsegments = metric_csr.shape[1]

    # container for results
    counts = np.zeros(nsegments, dtype=np.float32)

    # loop over spots, soft assign
    for iii in range(nspots):
        row = metric_csr.getrow(iii)
        counts[row.indices] += row.data

    # return
    return counts


def random_assign(metric_dok, nsamples, return_assignments=True):
    """
    """

    # convert format
    metric_csr = metric_dok.tocsr()

    # number of spots and segments
    nspots = metric_csr.shape[0]
    nsegments = metric_csr.shape[1]

    # container for results
    assignments = np.zeros((nspots, nsamples), dtype=np.uint32)
    counts = np.zeros((nsegments, nsamples), dtype=np.uint32)

    # loop over spots, random assign
    for iii in range(nspots):

        # get indices
        options = metric_csr.getrow(iii).indices
        if len(options) > 0:
            selected = options[np.random.randint(len(options), size=nsamples)]
            assignments[iii] = selected
            counts[selected, range(nsamples)] += 1

    # return counts
    if return_assignments:
        return counts, assignments
    else:
        return counts

