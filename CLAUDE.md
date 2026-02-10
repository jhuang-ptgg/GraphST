# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraphST is a graph self-supervised contrastive learning model for spatial transcriptomics (ST) data. It performs three tasks: spatial clustering, ST data integration/batch correction, and scRNA-seq cell type deconvolution onto ST. Published in Nature Communications (2023).

## Setup & Installation

```bash
pip install -e .
```

Key dependencies: torch (>=1.8), scanpy, anndata, scipy, scikit-learn, POT (optimal transport), rpy2 (for mclust clustering only), tqdm.

No test suite exists. No linter or formatter is configured.

## Architecture

All source lives in `GraphST/` (single flat package). The pipeline flows:

1. **`preprocess.py`** — Data preparation pipeline:
   - `preprocess()`: HVG selection → normalize → log1p → scale (operates on AnnData)
   - `construct_interaction()`: Builds NxN dense adjacency from spatial coordinates via pairwise distance (`ot.dist`). Stores in `adata.obsm['adj']` and `adata.obsm['graph_neigh']`
   - `construct_interaction_KNN()`: Same but uses sklearn `NearestNeighbors` (for Stereo-seq/Slide-seq)
   - `get_feature()`: Extracts dense feature matrix from `adata.X`, stores `adata.obsm['feat']` and augmented `adata.obsm['feat_a']`
   - `preprocess_adj()` / `preprocess_adj_sparse()`: Symmetric normalization (D^{-1/2} A D^{-1/2} + I). Sparse version returns torch sparse tensor
   - `sparse_mx_to_torch_sparse_tensor()`: scipy sparse → torch sparse conversion utility

2. **`model.py`** — PyTorch neural network modules:
   - `Encoder`: Dense GCN encoder (1-layer encode via `adj @ (feat @ W)`, 1-layer decode). Used for 10X Visium data
   - `Encoder_sparse`: Same architecture but uses `torch.spmm` for adjacency multiplication. Used for Stereo-seq/Slide-seq
   - `AvgReadout`: Graph readout via `torch.mm(mask, emb)` where mask is `graph_neigh`
   - `Discriminator`: Bilinear discriminator for contrastive learning (DGI-style)
   - `Encoder_sc`: Autoencoder for scRNA-seq representation learning (3-layer encode/decode)
   - `Encoder_map`: Learnable (n_cell × n_spot) mapping matrix for deconvolution

3. **`GraphST.py`** — Main `GraphST` class orchestrating the full pipeline:
   - `__init__()`: Runs preprocessing if needed, loads all tensors to device
   - `train()`: Full-batch GCN training loop with MSE reconstruction + BCE contrastive loss
   - `train_map()`: Deconvolution pipeline — trains ST encoder, scRNA encoder, then mapping matrix
   - `Noise_Cross_Entropy()`: Contrastive loss using spatial neighbors as positive pairs

4. **`utils.py`** — Clustering and projection utilities:
   - `clustering()`: PCA → mclust/leiden/louvain clustering with optional `refine_label()`
   - `refine_label()`: Majority-vote label smoothing using spatial KNN (uses dense `ot.dist`)
   - `project_cell_to_spot()`: Projects cell types onto spots via learned mapping matrix

## Key Data Flow

The pipeline communicates through `adata.obsm` slots:
- `'spatial'` → input coordinates
- `'adj'`, `'graph_neigh'` → adjacency matrices (currently dense NxN)
- `'feat'`, `'feat_a'` → feature and augmented feature matrices
- `'label_CSL'` → contrastive labels (Nx2)
- `'emb'` → learned embeddings (output)
- `'map_matrix'` → cell-to-spot mapping (deconvolution output)

## Data Type Branching

The `datatype` parameter controls two code paths:
- `'10X'` (default): Dense adjacency, `Encoder`, `preprocess_adj()`
- `'Stereo'` or `'Slide'`: KNN graph construction, `Encoder_sparse`, `preprocess_adj_sparse()`

## Scalability Constraints

The current implementation allocates dense NxN matrices in `construct_interaction()`, `construct_interaction_KNN()`, `refine_label()`, and `Noise_Cross_Entropy()`. This limits practical use to ~50K spots. The `Encoder_map` also allocates a full (n_cell × n_spot) parameter matrix.
