import numpy as np
import dask.array as da
from ClusterWrap.decorator import cluster


def extract_neighborhoods(
    spots,
    segments,
    spacing,
    radius,
):
    """
    Extract cubical neighborhoods around spot coordinates

    Parameters
    ----------
    spots : ndarray
        Nx3 array of N spot coordinates in physical units
    segments : ndarray (e.g. zarr.Array)
        3D image of cell or nuclei segments
    spacing : ndarray
        the voxel spacing of segments
    radius : float
        Neighborhood of ``radius`` physical units in each direction
        centered on every spot is extracted

    Returns
    -------
    neighborhoods : ndarray
        NxM array;
        N is the length of spots
        M is the size of the flattened neighborhood
    """

    # convert spots to voxel units
    spots = np.round(spots / spacing).astype(int)

    # get radius in voxel units
    radius = np.round(radius / spacing).astype(int)

    # initialize container
    nrows, ncols = len(spots), np.prod( 2*radius + 1 )
    neighborhoods = np.empty((nrows, ncols), dtype=segments.dtype)

    # loop through spots
    for iii, spot in enumerate(spots):

        # get crop of data
        neighborhood = tuple(slice(s-r, s+r+1) for s, r in zip(spot, radius))
        data = segments[neighborhood]

        # check if crop was on the edge, if so ignore the spot
        # TODO: don't ignore, determine which sides to pad with zeros
        if np.prod(data.shape) != ncols:
            continue

        # save the crop
        neighborhoods[iii] = data.ravel()

    # return
    return neighborhoods


@cluster
def extract_neighborhoods_distributed(
    spots,
    segments,
    spacing,
    radius,
    nblocks=10,
    cluster=None,
    cluster_kwargs={},
):
    """
    Distribute ``extract_neighborhoods`` with dask

    Parameters
    ----------
    spots : ndarray
        Nx3 array of N spot coordinates in physical units
    segments : ndarray (e.g. zarr.Array)
        3D image of cell or nuclei segments
    spacing : ndarray
        the voxel spacing of segments
    radius : float
        Neighborhood of ``radius`` physical units in each direction
        centered on every spot is extracted
    nblocks : int
        The number of parallel blocks to process
    cluster_kwargs : dict
        Arguments to ``ClusterWrap.cluster.janelia_lsf_cluster``

    Returns
    -------
    neighborhoods : ndarray
        NxM array;
        N is the length of spots
        M is the size of the flattened neighborhood
    """

    # load a local copy for shape and dtype reference
    sh, dt = spots.shape, spots.dtype

    # determine chunksize
    chunksize = (int(round(sh[0] / nblocks)), 3)

    # wrap spots as dask array, let worker load chunks
    spots = da.from_array(spots, chunks=chunksize)

    # determine output chunksize
    r = np.round(radius / spacing).astype(int)
    chunksize = (chunksize[0], np.prod( 2*r + 1 ))

    # map function over blocks
    neighborhoods = da.map_blocks(
        extract_neighborhoods, spots,
        segments=segments, spacing=spacing, radius=radius,
        dtype=segments.dtype,
        chunks=chunksize,
    )

    # start cluster, execute, and return
    return neighborhoods.compute()

