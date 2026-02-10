import numpy as np
import scipy.sparse as sp
import torch
from .preprocess import sparse_mx_to_torch_sparse_tensor


class SpatialClusterSampler:
    """ClusterGCN-style sampler for spatial graphs.

    Partitions spots into spatial grid clusters. Each mini-batch
    includes a cluster plus its 1-hop boundary neighbors.

    Parameters
    ----------
    adj_sparse : scipy.sparse matrix
        Sparse adjacency matrix (n_spot x n_spot).
    positions : np.ndarray
        Spatial coordinates (n_spot x 2).
    features : np.ndarray
        Feature matrix (n_spot x n_genes).
    features_a : np.ndarray
        Augmented feature matrix (n_spot x n_genes).
    label_CSL : np.ndarray
        Contrastive labels (n_spot x 2).
    graph_neigh : scipy.sparse matrix
        Directed KNN graph (before symmetrization).
    cluster_size : int
        Target number of spots per spatial cluster.
    shuffle : bool
        Whether to shuffle cluster order each iteration.
    """

    def __init__(self, adj_sparse, positions, features, features_a,
                 label_CSL, graph_neigh, cluster_size=10000, shuffle=True):
        self.adj = adj_sparse.tocsr()
        self.positions = positions
        self.features = features
        self.features_a = features_a
        self.label_CSL = label_CSL
        self.graph_neigh = graph_neigh.tocsr() if sp.issparse(graph_neigh) else sp.csr_matrix(graph_neigh)
        self.cluster_size = cluster_size
        self.shuffle = shuffle
        self.n_spot = positions.shape[0]

        # Build spatial grid partitions
        self.clusters = self._build_grid_clusters()

    def _build_grid_clusters(self):
        """Partition spots into rectangular grid cells."""
        pos = self.positions
        x_min, y_min = pos.min(axis=0)
        x_max, y_max = pos.max(axis=0)

        # Estimate grid dimensions to achieve target cluster_size
        n_clusters_approx = max(1, self.n_spot // self.cluster_size)
        aspect_ratio = (x_max - x_min + 1e-8) / (y_max - y_min + 1e-8)
        n_cols = max(1, int(np.sqrt(n_clusters_approx * aspect_ratio)))
        n_rows = max(1, int(np.ceil(n_clusters_approx / n_cols)))

        x_edges = np.linspace(x_min, x_max + 1e-8, n_cols + 1)
        y_edges = np.linspace(y_min, y_max + 1e-8, n_rows + 1)

        clusters = []
        for i in range(n_rows):
            for j in range(n_cols):
                mask = (
                    (pos[:, 0] >= x_edges[j]) & (pos[:, 0] < x_edges[j + 1]) &
                    (pos[:, 1] >= y_edges[i]) & (pos[:, 1] < y_edges[i + 1])
                )
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    clusters.append(indices)

        return clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        """Yields (sub_features, sub_features_a, sub_adj, sub_graph_neigh, sub_label_CSL, node_indices)."""
        order = list(range(len(self.clusters)))
        if self.shuffle:
            np.random.shuffle(order)

        for idx in order:
            core_nodes = self.clusters[idx]

            # Expand by 1-hop neighbors
            neighbor_rows = self.adj[core_nodes]
            if sp.issparse(neighbor_rows):
                all_neighbors = np.unique(neighbor_rows.indices)
            else:
                all_neighbors = np.unique(np.where(neighbor_rows > 0)[1])

            # Union of core + boundary
            batch_nodes = np.union1d(core_nodes, all_neighbors)
            batch_nodes.sort()

            # Extract sub-matrices
            sub_adj = self.adj[batch_nodes][:, batch_nodes]
            sub_graph_neigh = self.graph_neigh[batch_nodes][:, batch_nodes]

            # Add self-loops to graph_neigh for readout
            n_batch = len(batch_nodes)
            sub_graph_neigh = sub_graph_neigh + sp.eye(n_batch, dtype=np.float32)

            # Extract features
            sub_features = self.features[batch_nodes]
            sub_features_a = self.features_a[batch_nodes]
            sub_label_CSL = self.label_CSL[batch_nodes]

            # Convert to torch tensors
            sub_features = torch.FloatTensor(sub_features)
            sub_features_a = torch.FloatTensor(sub_features_a)
            sub_label_CSL = torch.FloatTensor(sub_label_CSL)

            # Normalize adjacency: D^{-1/2} (A+I) D^{-1/2}
            sub_adj_normalized = _normalize_sparse_adj(sub_adj)
            sub_adj_torch = sparse_mx_to_torch_sparse_tensor(sub_adj_normalized)

            sub_graph_neigh_torch = sparse_mx_to_torch_sparse_tensor(
                sp.coo_matrix(sub_graph_neigh, dtype=np.float32)
            )

            yield (sub_features, sub_features_a, sub_adj_torch,
                   sub_graph_neigh_torch, sub_label_CSL, batch_nodes)


def _normalize_sparse_adj(adj):
    """Symmetric normalization: D^{-1/2} (A+I) D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj_hat).dot(d_mat).tocoo()
