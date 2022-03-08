import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from SpotGraph.simple_metrics import nearest_neighbors
from SpotGraph.simple_metrics import exponential_kernel


def spot_to_spot_neighbors_graph(
    spots,
    max_distance,
    kernel=None,
    kernel_kwargs={},
):
    """
    Weighted directed graph representation of spot neighbors

    Parameters
    ----------
    spots : Nx3 array
        N three dimensional coordinates
    max_distance : float
        distance threshold below which two spots are considered neighbors
    kernel : callable
        Function to apply to distance, default None
    kernel_kwargs : dict
        Any arguments to kernel function

    Returns
    -------
    DG : networkx.DiGraph
        A directed edge goes from spot I to spot J if the distance
        between I and J is less than ``max_distance``. Edge weights
        are the normalized kernel of the distance; i.e. the out degree
        of all nodes sums to 1.
    """

    # get distances
    dists = nearest_neighbors(spots, max_distance).tocsr()

    # apply kernel
    if kernel is not None:
        a, b, c = dists.indices, dists.indptr, dists.shape
        data = kernel(dists.data, **kernel_kwargs)
        dists = csr_matrix((data, a, b), shape=c)

    # initialize graph with spot indices as nodes
    DG = nx.DiGraph()
    DG.add_nodes_from(range(len(spots)))

    # add weighted edges to graph
    for iii in range(len(spots)):
        weights = dists.getrow(iii)
        weights = weights / np.sum(weights)
        indices = weights.indices
        weights = weights.data
        edges = zip((iii,)*len(indices), indices, weights)
        DG.add_weighted_edges_from(edges)

    # return graph
    return DG


def smooth_distances_with_graph(
    distances, graph,
):
    """
    Spot to cell distances are replaced with weighted average of
    spot to cell distances of all neighbors (including the center
    spot)

    Parameters
    ----------
    distances : NxM dok_matrix
        The N spots by M cells matrix of distances or probabilies of
        assignment. (i, j) is the distance/probability that spot i
        is assigned to cell j.
    graph : networkx.DiGraph
        Directed graph with N nodes. Edge (i, j) is weighted by the
        normalized exponential kernel of the distance between nodes
        i and j.

    Returns
    -------
    smooth_distances : NxM dok_matrix
        Sparse matrix like ``distances`` but smoothed by weighted
        averaging along the out-going edges of ever node.
    """

    # convert to better format
    distances_csr = distances.tocsr()

    # initialize container
    smooth_distances = lil_matrix(distances.shape, dtype=distances.dtype)

    # loop over all spots, weight-sum rows
    for iii in range(len(graph)):
        smooth_row = None
        for jjj, edge in graph.adj[iii].items():
            if smooth_row is None:
                smooth_row = edge['weight'] * distances_csr.getrow(jjj)
            else:
                smooth_row += edge['weight'] * distances_csr.getrow(jjj)
        smooth_distances[iii] = smooth_row.tolil()

    # return
    return smooth_distances.todok()


def spot_to_cell_bipartite_graph(distances):
    """
    Creates a bipartite graph between spots and cells. An edge exists
    between a spot and a cell if the cell is in the neighborhood of
    the spot. Edges are weighted by assignment probability.

    Parameters
    ----------
    distances : NxM dok_matrix
        The ith row is the cell assignment probability distribution
        of spot i.

    Returns
    -------
    graph : networkx.Graph
        A bipartite graph with spots labeled 0 and cells labeled 1
        Edges only go between spots and cells
        Edge weights are assignment probabilities
        Out degree of every spot node sums to 1.
    """

    # specify some paraters
    nspots, ncells = distances.shape

    # initialize container
    g = nx.Graph()
    g.add_nodes_from(range(nspots), bipartite=0)
    g.add_nodes_from(range(nspots, nspots+ncells), bipartite=1)

    # add edges
    for edge, weight in dict(distances).items():
        g.add_edge(edge[0], nspots+edge[1], weight=weight)

    # return
    return g

