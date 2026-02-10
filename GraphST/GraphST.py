import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed, sparse_mx_to_torch_sparse_tensor
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse, Encoder_map, Encoder_map_lowrank, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sp
import pandas as pd

class GraphST():
    def __init__(self,
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        large_scale = None,
        chunk_size = 50000,
        batch_size = 10000,
        map_rank = 128,
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning.
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning.
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning.
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning.
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data.
        large_scale : bool or None, optional
            Enable large-scale mode with mini-batch training. None = auto-detect (threshold: 100K spots).
            False = original full-batch code path. True = chunked/batched path.
        chunk_size : int, optional
            Rows per chunk for Dask/KNN preprocessing. The default is 50000.
        batch_size : int, optional
            Spots per training mini-batch (ClusterGCN cluster size). The default is 10000.
        map_rank : int, optional
            Rank for low-rank deconvolution mapping matrix. The default is 128.

        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.map_rank = map_rank

        # Auto-detect large_scale mode
        if large_scale is None:
            self.large_scale = adata.n_obs > 100000
        else:
            self.large_scale = large_scale

        if self.large_scale:
            print(f'Large-scale mode enabled for {adata.n_obs} spots (chunk_size={chunk_size}, batch_size={batch_size})')

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata, large_scale_threshold=0 if self.large_scale else 100000, chunk_size=self.chunk_size)
           else:
              construct_interaction(self.adata, large_scale_threshold=0 if self.large_scale else 100000, chunk_size=self.chunk_size)

        if 'label_CSL' not in adata.obsm.keys():
           add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
           if self.large_scale:
              from .dask_preprocess import get_feature_chunked
              get_feature_chunked(self.adata, chunk_size=self.chunk_size, deconvolution=self.deconvolution)
           else:
              get_feature(self.adata)

        if self.large_scale:
            self._init_large_scale(dim_output)
        else:
            self._init_standard(dim_output)

    def _init_standard(self, dim_output):
        """Original full-batch initialization — loads all tensors to device."""
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)

        if self.deconvolution:
           self._init_deconvolution_data()

    def _init_large_scale(self, dim_output):
        """Large-scale initialization — keep data on CPU, use sampler for batching."""
        self.features_np = np.asarray(self.adata.obsm['feat'], dtype=np.float32)
        self.features_a_np = np.asarray(self.adata.obsm['feat_a'], dtype=np.float32)
        self.label_CSL_np = np.asarray(self.adata.obsm['label_CSL'], dtype=np.float32)

        self.adj_sparse = self.adata.obsm['adj']
        if not sp.issparse(self.adj_sparse):
            self.adj_sparse = sp.csr_matrix(self.adj_sparse)

        self.graph_neigh_sparse = self.adata.obsm['graph_neigh']
        if not sp.issparse(self.graph_neigh_sparse):
            self.graph_neigh_sparse = sp.csr_matrix(self.graph_neigh_sparse)

        self.dim_input = self.features_np.shape[1]
        self.dim_output = dim_output

        # Build the sampler
        from .graph_sampler import SpatialClusterSampler
        self.sampler = SpatialClusterSampler(
            adj_sparse=self.adj_sparse,
            positions=self.adata.obsm['spatial'],
            features=self.features_np,
            features_a=self.features_a_np,
            label_CSL=self.label_CSL_np,
            graph_neigh=self.graph_neigh_sparse,
            cluster_size=self.batch_size,
        )
        print(f'Spatial sampler created with {len(self.sampler)} clusters')

        if self.deconvolution:
           self._init_deconvolution_data()

    def _init_deconvolution_data(self):
        """Shared deconvolution data loading for both standard and large-scale modes."""
        adata = self.adata
        adata_sc = self.adata_sc if hasattr(self, 'adata_sc') else None

        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
           self.feat_sp = adata.X.toarray()[:, ]
        else:
           self.feat_sp = adata.X[:, ]
        if adata_sc is not None:
           self.adata_sc = adata_sc
           if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
              self.feat_sc = adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = adata_sc.X[:, ]

           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values

           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)

           self.dim_input = self.feat_sc.shape[1]
           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs

    def train(self):
        if self.large_scale:
            return self._train_large_scale()
        else:
            return self._train_standard()

    def _train_standard(self):
        """Original full-batch training loop."""
        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
           self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin to train ST data...')
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)

            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished for ST data!")

        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]

                return self.emb_rec
             else:
                if self.datatype in ['Stereo', 'Slide']:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                   self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                else:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec

                return self.adata

    def _train_large_scale(self):
        """Mini-batch training loop using ClusterGCN spatial sampling."""
        # Always use sparse encoder for large-scale
        # Use a dummy graph_neigh — actual per-batch graph_neigh is passed in forward()
        dummy_graph_neigh = torch.zeros(1, 1).to(self.device)
        self.model = Encoder_sparse(self.dim_input, self.dim_output, dummy_graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin to train ST data (large-scale mini-batch)...')

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Re-permute features each epoch
            perm_ids = np.random.permutation(self.features_np.shape[0])
            self.features_a_np = self.features_np[perm_ids]
            # Update sampler's features_a
            self.sampler.features_a = self.features_a_np

            for (sub_feat, sub_feat_a, sub_adj, sub_graph_neigh,
                 sub_label_CSL, batch_nodes) in self.sampler:

                sub_feat = sub_feat.to(self.device)
                sub_feat_a = sub_feat_a.to(self.device)
                sub_adj = sub_adj.to(self.device)
                sub_graph_neigh = sub_graph_neigh.to(self.device)
                sub_label_CSL = sub_label_CSL.to(self.device)

                hiden_feat, emb, ret, ret_a = self.model(sub_feat, sub_feat_a, sub_adj, sub_graph_neigh)

                loss_sl_1 = self.loss_CSL(ret, sub_label_CSL)
                loss_sl_2 = self.loss_CSL(ret_a, sub_label_CSL)
                loss_feat = F.mse_loss(sub_feat, emb)

                loss = self.alpha * loss_feat + self.beta * (loss_sl_1 + loss_sl_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        print("Optimization finished for ST data (large-scale)!")

        # Compute embeddings in chunks (no-grad forward pass)
        self.model.eval()
        emb_list = []
        node_order = []

        with torch.no_grad():
            for (sub_feat, sub_feat_a, sub_adj, sub_graph_neigh,
                 sub_label_CSL, batch_nodes) in self.sampler:

                sub_feat = sub_feat.to(self.device)
                sub_feat_a = sub_feat_a.to(self.device)
                sub_adj = sub_adj.to(self.device)
                sub_graph_neigh = sub_graph_neigh.to(self.device)

                _, emb, _, _ = self.model(sub_feat, sub_feat_a, sub_adj, sub_graph_neigh)
                emb = F.normalize(emb, p=2, dim=1).detach().cpu().numpy()

                emb_list.append(emb)
                node_order.append(batch_nodes)

        # Assemble full embedding matrix, handling overlapping boundary nodes
        n_spots = self.features_np.shape[0]
        n_dim = emb_list[0].shape[1]
        full_emb = np.zeros((n_spots, n_dim), dtype=np.float32)
        count = np.zeros(n_spots, dtype=np.float32)

        for emb_chunk, nodes in zip(emb_list, node_order):
            full_emb[nodes] += emb_chunk
            count[nodes] += 1

        # Average overlapping embeddings
        mask = count > 0
        full_emb[mask] /= count[mask, np.newaxis]

        if self.deconvolution:
            self.emb_rec = torch.FloatTensor(full_emb).to(self.device)
            return self.emb_rec
        else:
            self.emb_rec = full_emb
            self.adata.obsm['emb'] = self.emb_rec
            return self.adata

    def train_sc(self):
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)

        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()

            emb = self.model_sc(self.feat_sc)
            loss = F.mse_loss(emb, self.feat_sc)

            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()

        print("Optimization finished for cell representation learning!")

        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)

            return emb_sc

    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()

        # Handle large-scale: emb_sp may be adata (not tensor) when not deconvolving
        if self.deconvolution and not isinstance(emb_sp, torch.Tensor):
            # Large-scale mode returned adata, extract embedding
            emb_sp_np = self.adata.obsm['emb']
            emb_sp = torch.FloatTensor(emb_sp_np).to(self.device)

        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()

        # Normalize features for consistence between ST and scRNA-seq
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)

        # Use low-rank mapping for large-scale
        if self.large_scale:
            self.model_map = Encoder_map_lowrank(self.n_cell, self.n_spot, rank=self.map_rank).to(self.device)
        else:
            self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)

        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()

            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)

            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()

        print("Mapping matrix learning finished!")

        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell

            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell

            return self.adata, self.adata_sc

    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)

        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)

        return loss_recon, loss_NCE

    def Noise_Cross_Entropy(self, pred_sp, emb_sp, graph_neigh=None):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot

        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.
        graph_neigh : torch tensor, optional
            Batch-local graph neighbor matrix. If None, uses self.graph_neigh.

        Returns
        -------
        loss : float
            Loss value.

        '''
        if graph_neigh is None:
            graph_neigh = self.graph_neigh

        mat = self.cosine_similarity(pred_sp, emb_sp)
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))

        # positive pairs
        p = torch.exp(mat)
        if graph_neigh.is_sparse:
            p = torch.spmm(graph_neigh, p.t()).t()
            p = p.sum(axis=1)
        else:
            p = torch.mul(p, graph_neigh).sum(axis=1)

        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()

        return loss

    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.
        '''

        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)

        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M
