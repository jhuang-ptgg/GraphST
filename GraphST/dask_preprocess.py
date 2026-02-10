import numpy as np
import scipy.sparse as sp

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def get_feature_chunked(adata, chunk_size=50000, deconvolution=False):
    """Extract features as chunked arrays without densifying the full matrix.

    Parameters
    ----------
    adata : AnnData
        AnnData object (adata.X should be sparse after preprocessing).
    chunk_size : int
        Number of rows per chunk.
    deconvolution : bool
        If True, use all genes; otherwise use highly_variable genes.
    """
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata.var['highly_variable']]

    X = adata_Vars.X
    n_obs, n_genes = X.shape

    if sp.issparse(X):
        # Convert sparse to dense in chunks to avoid full densification
        if HAS_DASK:
            feat = _sparse_to_dask(X, chunk_size)
        else:
            # Fallback: chunked densification into pre-allocated array
            feat = np.empty((n_obs, n_genes), dtype=np.float32)
            for start in range(0, n_obs, chunk_size):
                end = min(start + chunk_size, n_obs)
                feat[start:end] = X[start:end].toarray()
    else:
        feat = np.asarray(X, dtype=np.float32)

    # Data augmentation via row permutation
    feat_a = permutation_chunked(feat)

    # Store back — if dask, compute to numpy for downstream compatibility
    if HAS_DASK and isinstance(feat, da.Array):
        adata.obsm['feat'] = feat.compute()
        adata.obsm['feat_a'] = feat_a.compute()
    else:
        adata.obsm['feat'] = feat
        adata.obsm['feat_a'] = feat_a


def permutation_chunked(feature):
    """Permute rows of a feature array (numpy or dask).

    Parameters
    ----------
    feature : numpy array or dask array
        Feature matrix of shape (n_obs, n_genes).

    Returns
    -------
    Permuted feature array.
    """
    if HAS_DASK and isinstance(feature, da.Array):
        # Compute to numpy, permute, convert back to dask
        feat_np = feature.compute()
        ids = np.random.permutation(feat_np.shape[0])
        feat_permuted = feat_np[ids]
        return da.from_array(feat_permuted, chunks=feature.chunks)
    else:
        ids = np.random.permutation(feature.shape[0])
        return feature[ids]


def _sparse_to_dask(sparse_matrix, chunk_size):
    """Convert scipy sparse matrix to dask array with row-wise chunking.

    Densifies one chunk at a time to keep memory bounded.
    """
    n_obs, n_genes = sparse_matrix.shape

    # Ensure CSR for efficient row slicing
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    chunks = []
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        chunk = da.from_delayed(
            _delayed_toarray(sparse_matrix, start, end),
            shape=(end - start, n_genes),
            dtype=np.float32,
        )
        chunks.append(chunk)

    return da.concatenate(chunks, axis=0)


try:
    from dask import delayed

    @delayed
    def _delayed_toarray(sparse_matrix, start, end):
        return sparse_matrix[start:end].toarray().astype(np.float32)
except ImportError:
    pass
