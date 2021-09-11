import numpy as np
import dask
import dask.array as da
import ClusterWrap


def extract_neighborhoods(
    spots,
    segments,
    radius,
    ratio=None,
):
    """
    Extract cubical neighborhoods around spot coordinates

    Parameters
    ----------
    spots : ndarray
        Nx3 array of N spot coordinates
    segments : ndarray (e.g. zarr.Array)
        3D image of cell or nuclei segments
    radius : int
        Neighborhood of ``radius`` voxels in each direction
        centered on every spot is extracted
    ratio : tuple length 3
        Each spot coordinate is divided by ``ratio`` to determine
        the neighborhood center in ``segments``

    Returns
    -------
    neighborhoods : ndarray
        NxM array; M is: 3 + (2*``radius``+1)**3
        That is the spot coordinate, and the flattened neighborhood
    """

    # initialize container
    nrows, ncols = len(spots), (2*radius + 1)**3 + 3
    neighborhoods = np.empty((nrows, ncols), dtype=segments.dtype)

    # loop through spots
    for iii, spot in enumerate(spots):
        center = spot / ratio if ratio else None
        center = center.round().astype(int)
        neighborhoods[iii, :3] = spot
        neighborhoods[iii, 3:] = segments[center[0]-radius:center[0]+radius+1,
                                          center[1]-radius:center[1]+radius+1,
                                          center[2]-radius:center[2]+radius+1,].ravel()

    # return
    return neighborhoods


def extract_neighborhoods_distributed(
    spots,
    segments,
    radius,
    ratio=None,
    nblocks=10,
    cluster_kwargs={},
):
    """
    Distribute ``extract_neighborhoods`` with dask

    Parameters
    ----------
    spots : string
        The filepath to the spots data on disk
    segments : ndarray (e.g. zarr.Array)
        3D image of cell or nuclei segments
    radius : int
        Neighborhood of ``radius`` voxels in each direction
        centered on every spot is extracted
    ratio : tuple length 3
        Each spot coordinate is divided by ``ratio`` to determine
        the neighborhood center in ``segments``
    nblocks : int
        The number of parallel blocks to process
    cluster_kwargs : dict
        Arguments to ``ClusterWrap.cluster.janelia_lsf_cluster``

    Returns
    -------
    neighborhoods : ndarray
        NxM array; M is: 3 + (2*``radius``+1)**3
        That is the spot coordinate, and the flattened neighborhood
    """

    # load a local copy for shape and dtype reference
    spots_local = np.loadtxt(spots)
    sh, dt = spots_local.shape, spots_local.dtype

    # determine chunksize
    chunksize = (int(round(sh[0] / nblocks)), 3)

    # wrap spots as dask array, let worker load chunks
    spots = dask.delayed(np.loadtxt)(spots)
    spots = da.from_delayed(spots, shape=sh, dtype=dt)
    spots = spots[:, :3]
    spots = spots.rechunk(chunksize)

    # determine output chunksize
    chunksize = (chunksize[0], (2*radius + 1)**3 + 3)

    # map function over blocks
    neighborhoods = da.map_blocks(
        extract_neighborhoods, spots,
        segments=segments, radius=radius, ratio=ratio,
        dtype=segments.dtype,
        chunks=chunksize,
    )

    # start cluster, execute, and return
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:
        return neighborhoods.compute()

