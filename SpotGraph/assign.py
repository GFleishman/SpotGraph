import numpy as np


def assign(
    metric_dok,
    min_or_max='min',
    return_counts=True,
):
    """
    """

    # convert format
    metric_csr = metric_dok.tocsr()

    # number of spots and segments
    nspots = metric_csr.shape[0]
    nsegments = metric_csr.shape[1]

    # container for results
    assignments = np.zeros(nspots, dtype=np.uint32)
    counts = np.zeros(nsegments, dtype=np.uint32)

    # loop over spots, assign
    for spot in range(nspots):

        # get row as key/value parallel lists
        row = metric_csr.getrow(spot).todok()
        keys = list(row.keys())
        vals = list(row.values())

        if len(keys) > 0:
            # get assignment id
            if min_or_max == 'min':
                segment_id = keys[np.argmin(vals)][1]
            elif min_or_max == 'max':
                segment_id = keys[np.argmax(vals)][1]
            else:
                raise ValueError("min_or_max must be 'min' or 'max'")
    
            # increment correct segment
            assignments[spot] = segment_id
            counts[segment_id] += 1

    # return counts
    if return_counts:
        return assignments, counts
    else:
        return assignments


def soft_assign(metric_dok):
    """
    """

    # convert format
    metric_csr = metric_dok.tocsr()

    # number of spots and segments
    nspots = metric_csr.shape[0]
    nsegments = metric_csr.shape[1]

    # container for results
    counts = np.zeros(nsegments, dtype=np.float32)

    # loop over spots, soft assign
    for spot in range(nspots):

        # get row as key/value parallel lists
        row = metric_csr.getrow(spot).todok()
        keys = list(row.keys())
        vals = list(row.values())

        # iterate over all neighbors, accumulate assignment
        for k, v in zip(keys, vals):
           counts[k[1]] += v

    # return
    return counts


