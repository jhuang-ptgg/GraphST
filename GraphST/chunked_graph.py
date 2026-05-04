import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

try:
    import dask
    from dask import delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


def construct_interaction_chunked(adata, n_neighbors=3, chunk_size=50000, n_jobs=-1, use_pynndescent=None):
    """Build sparse KNN graph without dense NxN allocation.

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['spatial'] with spatial coordinates.
    n_neighbors : int
        Number of nearest neighbors (excluding self).
    chunk_size : int
        Number of rows to query at a time.
    n_jobs : int
        Number of parallel jobs for KNN queries.
    use_pynndescent : bool or None
        If None, auto-detect (use pynndescent if available and n_obs > 100K).
    """
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    # Decide which KNN backend to use
    if use_pynndescent is None:
        use_pynndescent = HAS_PYNNDESCENT and n_spot > 100000

    if use_pynndescent:
        indices = _knn_pynndescent(position, n_neighbors, n_jobs)
    else:
        indices = _knn_sklearn_chunked(position, n_neighbors, chunk_size, n_jobs)

    # Build sparse adjacency from KNN indices
    row_idx = np.repeat(np.arange(n_spot), n_neighbors)
    col_idx = indices[:, 1:].flatten()  # exclude self (column 0)
    data = np.ones(len(row_idx), dtype=np.float32)

    interaction = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_spot, n_spot))

    # Store directed graph_neigh (before symmetrization)
    adata.obsm['graph_neigh'] = interaction.copy()

    # Symmetrize: adj = interaction + interaction.T, clip to [0,1]
    adj = interaction + interaction.T
    adj.data = np.clip(adj.data, 0, 1)
    adj.eliminate_zeros()

    adata.obsm['adj'] = adj
    print('Chunked sparse graph constructed!')


def _knn_sklearn_chunked(position, n_neighbors, chunk_size, n_jobs):
    """Query KNN in chunks using sklearn NearestNeighbors."""
    n_spot = position.shape[0]

    # Fit index on all coordinates (only n_spot x 2 = small)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(position)

    if HAS_DASK and n_spot > chunk_size:
        # Parallel chunked queries via dask.delayed
        chunk_starts = list(range(0, n_spot, chunk_size))

        @delayed
        def _query_chunk(start):
            end = min(start + chunk_size, n_spot)
            _, idx = nbrs.kneighbors(position[start:end])
            return idx

        results = dask.compute(*[_query_chunk(s) for s in chunk_starts])
        indices = np.vstack(results)
    else:
        # Single-shot query
        _, indices = nbrs.kneighbors(position)

    return indices


def _knn_pynndescent(position, n_neighbors, n_jobs):
    """Fast approximate KNN via pynndescent."""
    index = NNDescent(position, n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    indices, _ = index.neighbor_graph
    return indices
