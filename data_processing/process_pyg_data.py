"""
Functions for the scalar and vector tracks of ESCGNN, 
using pytorch-geometric Data classes encapsulation.
"""
import sys
sys.path.insert(0, '../')
import os
import warnings
from typing import Tuple, List, Dict, Optional, Callable, Literal, Any
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.utils import to_undirected as _to_undirected
import gc
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_undirected
import math  # for pi

from models.escgnn_data_classes import ESCGNNData


"""
PyG data utilities
"""
def get_square_matrix_coo_indices(d: int) -> torch.Tensor:
    """
    Generates coo indices for a d x d matrix "at the origin".

    Args:
        d: dimension (size of one side of the square matrix).
    Returns:
        Tensor of shape (2, d) holding the coo indices.
    """
    root_indices = torch.stack((
        torch.repeat_interleave(torch.arange(d), d), 
        torch.tile(torch.arange(d), (d,))
    ))
    return root_indices


def torch_sparse_identity(size: int) -> torch.Tensor:
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = torch.ones(size)
    return torch.sparse_coo_tensor(indices, values, (size, size))


def edge_index_to_dict(
    edge_index: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Converts a PyTorch Geometric edge_index tensor into a dictionary where
    each node index maps to a tensor of its connected neighbors.
    
    Args:
        edge_index: shape (2, num_edges), where each column 
        represents an edge (source, target).
    Returns:
        dict: A dictionary where keys are node indices and 
            values are tensors of connected node indices.
    """
    edge_dict = {}
    for src, dst in edge_index.t():  # Transpose to iterate over edges
        src, dst = src.item(), dst.item()
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(dst)

    # Convert lists to tensors for efficiency
    for key in edge_dict:
        edge_dict[key] = torch.tensor(edge_dict[key], dtype=torch.long)
    
    return edge_dict


"""
Distance-reweighting kernel functions
"""
def compute_node_median_edge_weight(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the median edge weight for each node in an (undirected) graph.

    These can be used to, e.g., scale a Gaussian distance kernel function: if the scale parameter is median_i * median_j, then the kernel weight between nodes i and j is w_ij = exp(-||x_i - x_j||² / (median_i * median_j)). This more fairly scales the kernel across nodes from different levels of density versus sparsity in their neighborhoods.

    The input `edge_index` for undirected graphs typically stores both edge
    directions (i→j and j→i). This function first removes symmetric duplicates
    by keeping a single canonical direction per undirected edge (i < j), while
    keeping self-loops (i == i) only once. It then computes, for each node, the
    median of the weights of all incident edges.

    Args:
        edge_index: Long tensor of shape (2, E) containing edge endpoints.
        edge_weight: Tensor of shape (E,) with edge weights corresponding to
            `edge_index` columns.
        num_nodes: Optional total number of nodes. If None, inferred as
            max(edge_index) + 1. If `edge_index` is empty and `num_nodes` is not
            provided, returns an empty tensor.

    Returns:
        Tensor of shape (num_nodes,) giving the median edge weight per node.
        Nodes with no incident edges receive 0.
    """
    # Handle degenerate case early
    if edge_index.numel() == 0:
        if num_nodes is None:
            return torch.empty((0,), dtype=edge_weight.dtype, device=edge_weight.device)
        return torch.zeros((num_nodes,), dtype=edge_weight.dtype, device=edge_weight.device)

    # Infer number of nodes if not provided
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)

    # Keep only one entry per undirected edge (canonicalize by src < dst)
    # and keep self-loops once (src == dst)
    keep_mask = (src < dst) | (src == dst)
    src_kept = src[keep_mask]
    dst_kept = dst[keep_mask]
    w_kept = edge_weight[keep_mask]

    # No edges remain after filtering
    if w_kept.numel() == 0:
        return torch.zeros((num_nodes,), dtype=edge_weight.dtype, device=edge_weight.device)

    # Build node-incidence lists by duplicating weights to both endpoints.
    # For self-loops, only count once.
    num_kept = w_kept.numel()
    node_ids = torch.cat([src_kept, dst_kept], dim=0)
    weights_dup = torch.cat([w_kept, w_kept], dim=0)

    self_mask_kept = (src_kept == dst_kept)
    if self_mask_kept.any():
        self_idx = torch.nonzero(self_mask_kept, as_tuple=False).squeeze(1)
        dup_positions = num_kept + self_idx
        mask_keep = torch.ones(2 * num_kept, dtype=torch.bool, device=edge_index.device)
        mask_keep[dup_positions] = False
        node_ids = node_ids[mask_keep]
        weights_dup = weights_dup[mask_keep]

    # Still no incidents after removing duplicates
    if node_ids.numel() == 0:
        return torch.zeros((num_nodes,), dtype=edge_weight.dtype, device=edge_weight.device)

    # Ensure weights live on the same device as indices for advanced indexing
    weights_dup = weights_dup.to(node_ids.device)

    # Group by node id
    sorted_ids, perm = torch.sort(node_ids)
    sorted_weights = weights_dup[perm]
    counts = torch.bincount(sorted_ids, minlength=num_nodes)
    offsets = torch.cumsum(counts, dim=0)
    starts = offsets - counts

    # Compute per-node medians; nodes with zero degree get 0
    medians = torch.zeros((num_nodes,), dtype=sorted_weights.dtype, device=sorted_weights.device)
    for node in range(num_nodes):
        cnt = int(counts[node].item())
        if cnt == 0:
            continue
        s = int(starts[node].item())
        e = int(offsets[node].item())
        vals = sorted_weights[s:e]
        if cnt % 2 == 1:
            k = cnt // 2 + 1  # 1-based index for kthvalue
            med = torch.kthvalue(vals, k).values
        else:
            k1 = cnt // 2
            k2 = k1 + 1
            v1 = torch.kthvalue(vals, k1).values
            v2 = torch.kthvalue(vals, k2).values
            med = 0.5 * (v1 + v2)
        medians[node] = med

    return medians.to(edge_weight.dtype)
    

def get_local_pca_kernel_weights(
    r: torch.Tensor,
    kernel: Literal["gaussian", "cosine_cutoff", "epanechnikov"],
    r_cut: Optional[float] = None,
    gaussian_eps: Optional[float] = None,
    take_sqrt: bool = True,
) -> torch.Tensor:
    """
    Computes weights for V_i (a matrix of neighbor difference vectors)
    using one of several supported radial kernels.

    Supported kernels:
        - "gaussian":
            w(r) = exp(-r^2 / eps)    if eps is not None
            w(r) = exp(-r^2)          if eps is None
        - "cosine_cutoff":
            w(r) = 0.5 * [cos(pi * r / r_cut) + 1]    if r < r_cut
                    = 0                               otherwise
        - "epanechnikov":
            w(r) = 1 - (r / r_cut)^2    if r < r_cut
                    = 0                 otherwise

    Args:
        r: tensor of shape (num_neighbors, ) of distances between a point and its neighbors.
        kernel: Which kernel to use: "gaussian", "cosine_cutoff", or "epanechnikov".
        r_cut: Required for cutoff kernels; ignored for "gaussian" unless used for manual cutoff.
        gaussian_eps: Optional scaling factor for "gaussian", equal to the variance (sigma^2) of 
            the Gaussian kernel. If None, defaults to 1 (no scaling).
        take_sqrt: Whether to return sqrt(weights), useful before weighted PCA/SVD.

    Returns:
        Tensor of shape (num_neighbors,) containing kernel weights.
    """
    if kernel == "gaussian":
        w = torch.exp(-r**2 / gaussian_eps) \
            if gaussian_eps is not None else torch.exp(-r**2)
        if r_cut is not None:
            w = torch.where(r < r_cut, w, torch.zeros_like(w))

    elif kernel == "cosine_cutoff":
        if r_cut is None:
            raise ValueError("r_cut must be provided for cosine_cutoff kernel.")
        w = 0.5 * (torch.cos(math.pi * r / r_cut) + 1.0)
        w = torch.where(r < r_cut, w, torch.zeros_like(w))
        w = torch.clamp(w, min=0.0)

    elif kernel == "epanechnikov":
        if r_cut is None:
            raise ValueError("r_cut must be provided for epanechnikov kernel.")
        w = 1.0 - (r / r_cut) ** 2
        w = torch.where(r < r_cut, w, torch.zeros_like(w))
        w = torch.clamp(w, min=0.0)

    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

    if take_sqrt:
        w = torch.sqrt(w)

    return w


"""
Distance-based graph construction
"""
# def build_distance_cutoff_edges(
#     pos: torch.Tensor,
#     r_cutoff: float = 5.0,
#     max_num_neighbors: int = 32,
# ) -> torch.Tensor:
#     """
#     Construct an undirected radius graph.

#     Args:
#         pos: Node coordinates.
#         r_cutoff: Connection threshold.
#         max_num_neighbors: Hard cap per node to keep 
#             large molecules manageable.
#     Returns:
#         radius_graph: Tensor of shape (2, num_edges)
#             containing the edge indices.
#     """
#     # radius_graph automatically adds both (i→j, j→i) edges
#     return radius_graph(
#         pos,
#         r_cutoff,
#         loop=False,
#         max_num_neighbors=max_num_neighbors,
#     )


# ------------------------------------------------------------------
# Weighted adjacency (Gaussian RBF × cosine cutoff)
# ------------------------------------------------------------------
def build_weighted_radius_adjacency(
    data: Data,
    geometry_key: str,
    graph_construction: Literal['k-nn', 'radius', 'reweight_existing_edges', 'unweighted_knn'] = 'k-nn',
    # k: int = 4, # max num bonds carbon can have
    r_cutoff: float = 5.0,
    gaussian_eps: float = 4.0,
    max_num_neighbors: int = 32,
    use_weighted_adjacency: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Return (edge_index, A_sparse, dists, cosine_cut) using a smooth 
    distance kernel.

    Weight formula (commonly used in molecular graphs): a combination 
    of a Gaussian RBF and a cosine cutoff (to enforce a cutoff radius).

        w(r) = exp(-(r/eps)^2) * 0.5 * (cos(π r / r_c) + 1)   for r < r_c
               0                                              otherwise

    Args:
        data : Data object.
        geometry_key : Key for node coordinates used to build the graph.
        graph_type : 'k-nn', 'radius', 'reweight_existing_edges', or 'unweighted_knn'.
        k : number of nearest neighbors for k-nn graph.
        r_cutoff : radius cutoff in angstroms.
            Distance cutoff r_c.
        gaussian_eps : Gaussian width eps (=sigma^2).
        max_num_neighbors : Cap neighbors per node.
    """
    pos = data[geometry_key]
    if graph_construction == 'k-nn' \
    or graph_construction == 'unweighted_knn':
        edge_index = knn_graph(
            pos, 
            k=max_num_neighbors, 
            loop=False
        )
        # 'reduce' controls the edge feature merging method
        edge_index = to_undirected(edge_index, reduce='min')
    elif graph_construction == 'radius':
        edge_index = radius_graph(
            pos,
            r_cutoff,
            loop=False,
            max_num_neighbors=max_num_neighbors,
        )
        edge_index = to_undirected(edge_index)
    elif graph_construction == 'reweight_existing_edges':
        edge_index = data.edge_index
    

    # Compute edge weights using Gaussian RBF × cosine cutoff kernel
    # These weights will dictate neighbor weights during diffusion
    dists, cosine_cut = None, None
    weights = None
    if (graph_construction != 'unweighted_knn') and use_weighted_adjacency:
        row, col = edge_index
        dists = (pos[row] - pos[col]).norm(dim=1)
        gauss = torch.exp(-((dists / gaussian_eps) ** 2))
        cosine_cut = 0.5 * (torch.cos(math.pi * dists / r_cutoff) + 1.0)
        weights = gauss * cosine_cut

    # Only construct sparse adjacency when weights are present
    if weights is not None:
        N = pos.shape[0]
        A_sparse = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            weights,
            (N, N),
        ).coalesce()
    else:
        A_sparse = None

    # For downstream usage and lightweight saving, return edge weights directly
    edge_weight = A_sparse._values() if A_sparse is not None else None
    return edge_index, edge_weight, dists, cosine_cut


"""
Change-of-basis functions
"""
def get_l_singular_vecs(
    A: torch.Tensor,
    num_vecs: int,
    rank_deficiency_strategy: Optional[Literal['Tikhonov']] = None,
    tikhonov_eps: float = 1e-3,
) -> torch.Tensor:
    """
    Using `torch.linalg.svd`, retrieves the left
    singular vectors of a matrix A. Note that if
    `full_matrices=False`, the desired number of
    vectors may not be returned.

    Args:
        A: the matrix tensor to be decomposed, of shape (d, n).
        num_vecs: the number of left singular
            vectors to compute.
        rank_deficiency_strategy: optional strategy for handling rank   
            deficiency.
        tikhonov_eps: Tikhonov isotropic regularisation parameter
    Returns:
        A matrix tensor of shape (vector_dimension,
        num_left_singular_vectors).
    """
    # Check if rank-deficiency handling is requested
    if (rank_deficiency_strategy is not None) \
    and (rank_deficiency_strategy.lower() == 'tikhonov'):
        # If the neighborhood size (number of columns of A) is < num_vecs,
        # we treat it as rank-deficient
        if A.shape[1] < num_vecs:
            # Form isotropically regularized covariance matrix and take 
            # its eigenvectors. This yields a deterministic, full-rank
            # orthonormal frame whose first r directions coincide with 
            # the data-driven ones up to O(eps).
            d = A.shape[0]
            cov = torch.mm(A, A.T) \
                + tikhonov_eps * torch.eye(d, device=A.device, dtype=A.dtype)
            # Eigen-decomposition (symmetric, so use eigh)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # torch.linalg.eigh returns ascending order; reverse
            rev_idx = torch.flip(torch.arange(eigvecs.shape[1]), dims=[0])
            eigvecs = eigvecs[:, rev_idx]
            return eigvecs[:, :num_vecs]

    # Fall back to default behavior (standard SVD)
    us = torch.linalg.svd(
        A,
        full_matrices=True
    )[0][:, :num_vecs]  # shape d x num_vecs
    return us



def match_B_col_directions_to_A(
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    """
    Re-aligns the vector columns in a matrix B
    to agree with the vector columns of a matrix
    A by comparing the column-wise dot products.
    If the dot product is < 0, the sign of elements
    in the B column is flipped.

    Args:
        A: matrix tensor.
        B: matrix tensor to be 'aligned' column-wise
            with A's column vectors.
    Returns:
        Matrix tensor of the processed B.
    """
    B = torch.stack([
        b if torch.dot(a, b) >= 0.0 else -b \
        for a, b in zip(A.T, B.T) # transposes -> iterates over cols
    ], dim=1) # re-stack columns
    return B


def align_Oj_to_Oi_procrustes(
    O_i: torch.Tensor,
    O_j: torch.Tensor,
) -> torch.Tensor:
    """
    Align O_j to O_i via orthogonal Procrustes on the span of columns.
    Returns O_j_aligned = O_j @ R_opt where R_opt = argmin_R ||O_i - O_j R||_F, R ∈ O(d).
    """
    # Compute S = O_i^T O_j, then SVD to get optimal rotation in column space
    S = O_i.T @ O_j
    U, _, Vt = torch.linalg.svd(S, full_matrices=False)
    R_opt = U @ Vt
    return O_j @ R_opt


def get_C_i_dict(
    data: Data,
    geom_feat_key: str,
    use_mean_recentering: bool = False,
    kernel_fn: Callable = get_local_pca_kernel_weights,
    kernel_fn_kwargs: Dict[str, Any] = {},
    *,
    use_precomputed_edge_weights: bool = False,
    edge_index_override: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """
    Generates a dictionary of C_i (tensor)  
    matrices for each point x_i. As a local PCA, the
    process is (1) do neighbor vector subtractions, as
    the 'recentering' step; (2) use a kernel to re-
    weight the distances in the neighbor vector 
    subtractions.

    Args:
        data: a pytorch-geometric Data object.
        geom_feat_key: the string key of the Data 
            object used in computing local PCA, of shape: (num_nodes, D).
        use_mean_recentering: on p. 4 of Singer and Wu, 
            they note it is more common to subtract off mean
            (centroid) point from neighbors of x_i when constructing
            C_i matrices, though their approach uses the 
            coordinates of x_i itself. If true, this function
            employs this more common approach.
        kernel_fn: a function to calculate new distance weights
            between a point x_i and its neighborgs.
        kernel_fn_kwargs: kwargs to pass to the 'kernel_fn'.
        edge_index_override: optional edge_index to use for neighbor sets
            when constructing local PCA inputs. If None, data.edge_index
            is used.
    Returns:
        Dictionary keyed by point/node index (i.e., the
        'i' of each x_i), where values are the tensor
        matrices of shape (num_neighbors, vector_dimension),
        where columns are the centered and kernel-scaled
        difference vectors between x_i and each of its 
        neighbors.
    """
    # grab vector node feature matrix
    # Use vector features for local PCA / bases
    v = data[geom_feat_key]
    
    # for each node x_i, find its neighbors; contain all 
    # in a dictionary
    edge_index = edge_index_override if edge_index_override is not None else data.edge_index
    nbrs_dict = edge_index_to_dict(edge_index)
    # print('nbrs_dict[0]:', nbrs_dict[0])
    
    # get V_i (the re-centered neighbor vector features matrix)
    # for each node, each of shape (num_neighbors, d)
    if use_mean_recentering:
        V_i_dict = {
            i: torch.stack([
                v[j, :] \
                - torch.mean(
                    v[torch.cat((torch.tensor([i]), nbr_idx)), :], 
                    dim=0
                ) \
                for j in nbr_idx
            ], dim=0) \
            for i, nbr_idx in nbrs_dict.items()
        }
    else:
        V_i_dict = {
            i: torch.stack([
                v[j, :] - v[i, :] for j in nbr_idx
            ], dim=0) \
            for i, nbr_idx in nbrs_dict.items()
        }
    # print('V_i_dict[0]:', V_i_dict[0])

    ''' DEPRECATED
    use_chemical_distances = False
    if ('chemical' in kernel_fn.__name__):
        # Prepare chemistry-aware arguments if available
        # Precompute bond type map: (src, dst) -> bond_type_idx (0-3)
        bond_type_map = None
        if hasattr(data, 'edge_attr') and (data.edge_attr is not None):
            edge_types = torch.argmax(data.edge_attr, dim=1).tolist()
            bond_type_map = {
                (int(src.item()), int(dst.item())): edge_types[k]
                for k, (src, dst) in enumerate(data.edge_index.t())
            }

        # Grab atomic numbers
        z = getattr(data, 'z', None)

        # If both bond_type_map and z are not None, we can use chemical weights
        if (bond_type_map is not None) and (z is not None):
            use_chemical_distances = True
    '''

    w_ij_dict = {}
    if use_precomputed_edge_weights:
        if edge_index_override is not None:
            raise ValueError(
                "use_precomputed_edge_weights=True is not supported with edge_index_override."
            )
        # Build a fast lookup from (src, dst) to weight
        ei = data.edge_index.to(torch.long)
        ew = getattr(data, 'edge_weight', None)
        if ew is None:
            raise ValueError("use_precomputed_edge_weights=True but data.edge_weight is None")
        # Ensure on CPU for Python dict ops
        ei_cpu = ei.cpu()
        ew_cpu = ew.detach().cpu()
        pair_to_w = {}
        for col in range(ei_cpu.shape[1]):
            src = int(ei_cpu[0, col].item())
            dst = int(ei_cpu[1, col].item())
            pair_to_w[(src, dst)] = ew_cpu[col]
        for i, V_i in V_i_dict.items():
            nbr_idx = nbrs_dict[i]
            w_list = []
            for j in nbr_idx.tolist():
                w = pair_to_w.get((int(i), int(j)))
                if w is None:
                    w = pair_to_w.get((int(j), int(i)))
                if w is None:
                    # fallback to 1.0 if no weight found (should not happen)
                    w_val = 1.0
                else:
                    w_val = float(w.item())
                w_list.append(w_val)
            w_ij_dict[i] = torch.tensor(w_list, dtype=V_i.dtype, device=V_i.device)
    else:
        for i, V_i in V_i_dict.items():
            ''' DEPRECATED
        if use_chemical_distances:
            nbr_idx = nbrs_dict[i]
            bond_types_i = torch.tensor([
                bond_type_map.get(
                    (int(i), int(j)), 
                    bond_type_map.get((int(j), int(i)), 0)
                )
                for j in nbr_idx
            ], device=V_i.device)
            kernel_fn_kwargs = kernel_fn_kwargs | {
                'bond_types_i': bond_types_i,
                'z_i': int(z[i].item()),
                'z_neighbors': z[nbr_idx],
            }
        '''
            # get the weights for each neighbor
            r = torch.norm(V_i, p=2, dim=1)
            w_ij = kernel_fn(r, **kernel_fn_kwargs)
            w_ij_dict[i] = w_ij
    
    # calculate each w_ij*(x_j - x_i) by multiplying neighbor
    # weights (of length num_neighbors) on each row of V_i of 
    # shape, then transposing -> shape (d, num_neighbors)
    C_i_dict = {
        i: torch.einsum('n,nd->dn', w_ij_dict[i], V_i) \
        for i, V_i in V_i_dict.items()
    }
    # print('C_i_dict[0]:', C_i_dict[0])

    return C_i_dict


def calc_O_ij(
    O_i: torch.Tensor,
    O_j: torch.Tensor,
    enforce_sign: bool = True,
    sing_vect_align_method: Literal['column_dot', 'procrustes'] = 'column_dot',
) -> torch.Tensor:
    """
    Computes O_ij, the change-of-basis matrix that
    combines the local PCA from points i and j to
    make an equivariant representation for operators
    acting on points i and j (with vector features
    of dimension d).
    
    Args:
        O_i: the local PCA loadings (from left singular 
            vectors of an SVD of C_i) for point x_i,
            of shape (n, d).
        O_j: same as O_i but for point x_j.
        enforce_sign: since SVD can return singular
            vectors nondeterministically 'off' by a
            factor of -1, if this argument is true,
            we enforce the left singular vectors of 
            O_i and O_j to be in the same direction 
            by checking their inner (dot) product; 
            if not, we flip the vector from O_j ('v'). 
        sing_vect_align_method: method to align the singular
            vectors of O_i and O_j.
    Returns:
        O_ij tensor of shape (d, d), where d is the 
        dimension of the vector node feature.
    """
    # print(f"O_i.shape: {O_i.shape}")
    # print(f"O_j.shape: {O_j.shape}")
    if enforce_sign:
        if sing_vect_align_method == 'procrustes':
            O_j = align_Oj_to_Oi_procrustes(O_i, O_j)
        else:
            O_j = match_B_col_directions_to_A(O_i, O_j)
    # Note: O_i @ O_j.T is equivariant to rotations; O_i.T @ O_j is invariant
    O_ij = torch.mm(O_i, O_j.T)
    return O_ij


"""
Main processing classes and functions
"""
def get_P(
    data: Data,
    row_normalize: bool = True,
    device: str = 'cpu',
    filter_warnings: bool = True,
) -> torch.Tensor:
    r"""
    Computes P, the lazy random walk diffusion 
    operator on a graph's scalar node features,
    defined as $P = 0.5 (I - D^{-1}A)$ [row-normalized,
    for Px operations],where the graph is the 
    disconnected batch graph of a torch_geometric Batch 
    object.

    Args:
        data: a pytorch-geometric Data container object.
        row_normalize: whether to row-normalize the 
            operator matrix. For message passing using Px 
            (left multiplication by P), row-normalization is 
            needed to average messages from neighbors, into
            node i, in the intuitive message-passing sense.
            NOTE: If P is column-normalized, then in Px, each 
            node i still aggregates from neighbors j, but weights 
            the message from node j based on how connected j is. 
            As a result, the higher the degree of node j, the less 
            weight node j's message carries. This is a different, 
            'reverse-weighted' mix of node values:
                $(Px)_i = \sum_j (A_{ij}/d_j) x_j$,
            but can be useful if we want diffusion that adjusts
            for the connectedness of neighbors (e.g., under the
            assumption that high-degree neighbors are from 
            'oversampled' regions of an underlying manifold).
        device: string device key (e.g., 'cpu', 'cuda', 
            'mps') for placing the output tensor; if
            None, will check for cuda, else assign to cpu.
        filter_warnings: whether to filter warnings from the 
            torch.sparse library.
    Returns:
        Sparse P matrix tensor, of shape 
        (N, N), where N is the number of nodes
        (determined from data.num_nodes, data.x.shape[0], 
        or inferred from edge_index), the total number 
        of nodes across all batched graphs. Note Pparse is 'doubly
        sparse': sparse off of block diagonals,
        and each block is itself a sparse operator
        P_i for each graph x_i.
    """
    if filter_warnings:
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        
    # Get number of nodes, with fallback options
    if hasattr(data, 'num_nodes') and data.num_nodes is not None:
        num_nodes = data.num_nodes
    elif hasattr(data, 'x') and data.x is not None:
        num_nodes = data.x.shape[0]
    else:
        # Fallback: infer from edge_index
        num_nodes = int(data.edge_index.max()) + 1
    
    # Handle edge weights - use ones if not provided
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weight = data.edge_weight
    else:
        edge_weight = torch.ones(data.edge_index.shape[1])

    # Ensure edge_index and edge_weight are on the same device for coalesce()
    # PyG's coalesce expects both tensors to live on the same device as the
    # indexing it performs. We canonicalize to CPU here, then move the sparse
    # tensor to the requested device afterwards.
    edge_index_cpu = data.edge_index.cpu()
    edge_weight_cpu = edge_weight.to(edge_index_cpu.device)

    A_sparse = to_torch_coo_tensor(
        edge_index_cpu,
        edge_weight_cpu,
        (num_nodes, num_nodes)
    ).to(device)
    D = A_sparse.sum(dim=1).to_dense()
    # last I checked, 'torch.sparse.spdiags' doesn't work on cuda 12.4,
    # -> make sure to use function with cpu tensors, then move resulting 
    # sparse tensor to cuda if available
    D = D.squeeze().to('cpu')
    # for nodes with degree 0, prevent division by 0 error
    D_inv = torch.where(D > 0, (1. / D), 0.)
    D_inv = torch.sparse.spdiags(
        diagonals=D_inv, 
        offsets=torch.zeros(1).long().to('cpu'),
        shape=(len(D), len(D))
    ).to(device)

    I = torch_sparse_identity(len(D)).to(device)
    D_inv_A_sparse = torch.sparse.mm(D_inv, A_sparse)
    if not row_normalize:
        P_sparse = 0.5 * (I + D_inv_A_sparse.T)
    else:
        P_sparse = 0.5 * (I + D_inv_A_sparse)
    
    return P_sparse


def select_molecule_dirac_nodes(
    data: Data,
    coords_key: str, 
    atom_types_key: str, 
    k: int = 3
) -> torch.Tensor:
    """
    Args:
        data: a pytorch-geometric Data container object.
        coords_key: the string key of the Data 
            object for its vector-valued node features
            matrix, of shape: (num_nodes, vector_dimension).
        atom_types_key: the string key of the Data 
            object for its atomic numbers, of shape: 
            (num_nodes,).
        k: number of Dirac nodes to select
    Returns:
        selected_indices: indices of atoms to use for Dirac features
    """
    coords = data[coords_key]
    atom_types = data[atom_types_key]
    heavy_mask = atom_types != 1  # exclude hydrogens (atomic number 1)
    heavy_coords = coords[heavy_mask]
    
    if heavy_coords.shape[0] < k:
        # fallback: allow Hs if not enough heavy atoms
        heavy_coords = coords
        heavy_mask = torch.ones(coords.shape[0], dtype=torch.bool)

    centroid = heavy_coords.mean(dim=0, keepdim=True)
    dists = torch.norm(heavy_coords - centroid, dim=1)
    topk = torch.topk(dists, k=k, largest=True).indices
    selected = heavy_mask.nonzero(as_tuple=True)[0][topk]
    
    return selected  # indices in the original atom list


def process_pyg_data(
    data: Data,
    data_i: Optional[int] = None,
    row_normalize: bool = True,
    use_mean_recentering: bool = False,
    vector_feat_key: Optional[str] = None,
    geom_feat_key: Optional[str] = None,
    # Local (node) PCA kernel function parameters
    local_pca_kernel_fn: Callable = get_local_pca_kernel_weights,
    local_pca_kernel_fn_kwargs: Dict[str, Any] = {
        # keep mild, since diffusion operators also use kernel distance weights;
        # (we don't want to over-localize)
        'kernel': 'epanechnikov',  
        'r_cut': 5.0,
    },
    local_pca_k: Optional[int] = None,
    enforce_sign: bool = True,
    device: Optional[str] = None,
    *,
    graph_construction: Optional[Literal['k-nn', 'radius', 'unweighted_knn']] = None,
    graph_construction_kwargs: Dict[str, Any] = {
        # 'k': 4, # max num bonds carbon can have
        'r_cutoff': 5.0,
        'gaussian_eps': 4.0,
        'max_num_neighbors': 4, # max num bonds carbon can have is 4
    },
    return_data_object: bool = True,
    rank_deficiency_strategy: Optional[Literal['Tikhonov']] = None,
    tikhonov_eps: float = 1e-3,
    adjacency_rbf_eps: Optional[float] = None,
    num_edge_features: int = 0,
    include_diracs: bool = False,
    dirac_fn: Callable = select_molecule_dirac_nodes,
    dirac_kwargs: Dict[str, Any] = {
        'coords_key': 'pos',
        'atom_types_key': 'z',
        'k': 3
    },
    hdf5_tensor_dtype: str = 'float16',
    edge_feature_type: str = 'bessel',
    sing_vect_align_method: Literal['column_dot', 'procrustes'] = 'column_dot',
    # New: whether to compute line-graph diffusion operator P_line
    compute_line_operator: bool = False,
    # New: when saving to HDF5 (return_data_object=False), optionally skip saving P
    save_scalar_operator: bool = True,
    equivar_or_invar: Literal['equivariant', 'invariant'] = 'equivariant',
    debug: bool = False,
) -> Data:
    """
    Processes a pytorch-geometric Data object to 
    add lazy random walk diffusion operator matrices 
    for the scalar and vector node features as attributes 
    of the Data, as well as the reshaped vector feature 
    that the vector feature diffusion operator can act on.

    Args:
        data: a pytorch-geometric Data object.
        data_i: an index associated with the Data object;
            useful if returning only 'P' and 'Q' tensors
            for later re-assignment.
        use_mean_recentering: on p. 4 of Singer and Wu, 
            they note it is more common to subtract off mean
            (centroid) point from neighbors of x_i when constructing
            C_i matrices, though their approach uses the 
            coordinates of x_i itself. If true, this function
            employs this more common approach.
        vector_feat_key: the string key of the Data 
            object for its vector-valued node features
            matrix, of shape: (num_nodes, vector_dimension).
            This function reshapes into a shape suitable for
            diffusion and reattaches to the Data object.
        geom_feat_key: the string key of the Data object
            for geometry used to build O-frames and local PCA.
        local_pca_kernel_fn: a function to calculate new distance weights
            between a point x_i and its neighbors.
        local_pca_kernel_fn_kwargs: kwargs to pass through to 
            'local_pca_kernel_fn'.
        local_pca_k: optional neighbor count to use for local PCA/O-frame
            construction. If provided, a k-NN graph is built from geom_feat_key
            and used only for local PCA neighbor sets (diffusion operators still
            use data.edge_index).
        enforce_sign: see description of this parameter in
            the called 'calc_O_ij' function.
        device: string device key (e.g., 'cpu', 'cuda', 
            'mps') for placing the output tensor; if
            None, will check for cuda, else assign to cpu.
        graph_construction: if not None, type of new graph construction to use,
            which will override any edge_index and edge_weight already
            present in the Data object.
        adjacency_distance_cutoff: Distance cutoff for the adjacency 
            matrix construction.
        max_num_neighbors: Maximum number of neighbors per node in 
            the adjacency matrix construction.
        return_data_object: if True, output is a ESCGNNData
            object with 'P', 'Q', and other attributes. 
            If false, a nested dictionary of the 'P' and 'Q' 
            sparse tensors coo components are returned instead.
        rank_deficiency_strategy: Strategy for handling the case k < d.
            Currently supports 'Tikhonov' (adds isotropic regularisation).
        tikhonov_eps: Regularisation magnitude used when 
            rank_deficiency_strategy=='Tikhonov'.
        adjacency_rbf_eps: Gaussian width eps (=sigma^2) for the 
            Gaussian RBF used in the adjacency matrix construction.
        num_edge_features: Number of edge features.
        hdf5_tensor_dtype: dtype for the HDF5 tensor files.
        edge_feature_type: str, type of edge features to compute (default: 'bessel').
        compute_line_operator: bool, compute and return line-graph operator 'P_line' if True.
        save_scalar_operator: bool, when not returning a Data object, include 'P' in outputs if True.
        equivar_or_invar: whether to use equivariant (O_i O_j^T) or invariant
            (O_i^T O_j) transport blocks.
        debug: if True, print singular value ratio stats from local PCA.
    Returns:
        A modified pytorch-geometric Data object, with 'P' 
        (diffusion operator matrix for scalar node features),
        'Q' (operator for vector features), and 'v', the
        reshaped vector node features to cohere with Q.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if vector_feat_key is None:
        if geom_feat_key is None:
            raise ValueError("vector_feat_key must be set when geom_feat_key is None.")
        vector_feat_key = geom_feat_key
    if geom_feat_key is None:
        geom_feat_key = vector_feat_key

    # --------------------------------------------------------------
    # (Optional) replace bond graph with distance-cutoff graph
    # This constructs the neighbor/edge set for each node, used to
    # calculate the diffusion operator matrices. Note, however, that
    # the kernel function used in the local PCA step does not have to
    # be the same as the kernel function used in the diffusion operators
    # (though the neighbor set should be the same).
    # --------------------------------------------------------------
    local_pca_edge_index = None
    if local_pca_k is not None:
        if local_pca_k <= 0:
            raise ValueError("local_pca_k must be a positive integer.")
        if not hasattr(data, geom_feat_key):
            raise ValueError(
                f"local_pca_k requires geom_feat_key '{geom_feat_key}' to exist on data."
            )
        geom_for_knn = data[geom_feat_key]
        local_pca_edge_index = knn_graph(
            geom_for_knn,
            k=int(local_pca_k),
            loop=False,
        )
        # Ensure undirected for stable incidence with data.edge_index
        local_pca_edge_index = to_undirected(local_pca_edge_index)

    if graph_construction is not None \
    and hasattr(data, geom_feat_key):
        edge_index, edge_weight, dists, cosine_cut = build_weighted_radius_adjacency(
            data,
            geom_feat_key,
            graph_construction=graph_construction,
            **graph_construction_kwargs,
        )
        data.edge_index = edge_index
        # Ensure weighted diffusion operators use these weights when present
        if edge_weight is not None:
            data.edge_weight = edge_weight

        # -----------------------------
        # Edge features (optional, extensible)
        # -----------------------------
        if (num_edge_features is not None) and (num_edge_features > 0):
            if (dists is not None) and (cosine_cut is not None) and (edge_feature_type == 'bessel'):
                # Accept either explicit adjacency_distance_cutoff or fall back to r_cutoff
                adjacency_distance_cutoff = graph_construction_kwargs.get('adjacency_distance_cutoff', graph_construction_kwargs.get('r_cutoff'))
                if adjacency_distance_cutoff is not None:
                    data.edge_features = compute_bessel_edge_features(
                        dists, cosine_cut, adjacency_distance_cutoff, num_edge_features, hdf5_tensor_dtype
                    )
                else:
                    print("No adjacency distance cutoff provided; skipping edge feature computation.")
            elif edge_feature_type != 'bessel':
                print(f"Edge feature type '{edge_feature_type}' not yet supported...no edge features will be used.")
            else:
                # Unweighted graphs or missing distances: skip edge features gracefully
                pass

    # calculate P_scalars matrix and coalesce, so indices
    # and values are nice and ordered
    P = get_P(data, row_normalize, device).coalesce()
    # print('P.indices()[:, :10]:', P.indices()[:, :10])
    P_indices = P.indices()
    P_values = P.values()
    
    # grab vector feature tensor (for return) and PCA feature tensor (for bases)
    v = data[vector_feat_key]
    v_pca = data[vector_feat_key]
    d = v_pca.shape[1]
    if equivar_or_invar not in {"equivariant", "invariant"}:
        raise ValueError("equivar_or_invar must be 'equivariant' or 'invariant'.")

    # get the centered, kernel-reweighted distance matrices
    C_i_dict = get_C_i_dict(
        data,
        geom_feat_key,
        use_mean_recentering,
        local_pca_kernel_fn,
        local_pca_kernel_fn_kwargs,
        edge_index_override=local_pca_edge_index,
    )

    # get first d cols of the left singular matrix U_i from
    # the SVD of each C_i, of shape (d, num_neighbors)
    # -> cols of each O_i are a basis for R^d centered at x_i
    O_i_dict = {
        i: get_l_singular_vecs(
            C_i_dict[i],
            d,
            rank_deficiency_strategy=rank_deficiency_strategy,
            tikhonov_eps=tikhonov_eps
        )
        for i in C_i_dict.keys()
    }
    if debug:
        sv12 = []
        sv23 = []
        eps = 1e-12
        for C_i in C_i_dict.values():
            sv = torch.linalg.svdvals(C_i)
            if sv.numel() < 3:
                continue
            s1, s2, s3 = sv[0], sv[1], sv[2]
            if float(s2) > eps:
                sv12.append(float((s1 / s2).item()))
            if float(s3) > eps:
                sv23.append(float((s2 / s3).item()))
        if sv12:
            print(
                "PCA SV ratios (sv1/sv2): "
                f"min={np.min(sv12):.4g}, "
                f"median={np.median(sv12):.4g}, "
                f"max={np.max(sv12):.4g}"
            )
        else:
            print("PCA SV ratios (sv1/sv2): no valid values")
        if sv23:
            print(
                "PCA SV ratios (sv2/sv3): "
                f"min={np.min(sv23):.4g}, "
                f"median={np.median(sv23):.4g}, "
                f"max={np.max(sv23):.4g}"
            )
        else:
            print("PCA SV ratios (sv2/sv3): no valid values")
    # print('O_i_dict[0]:', O_i_dict[0])
    # for i, O_i in O_i_dict.items():
    #     print(O_i.shape)

    # calc Q sparse coo values and indices
    # Optimize: process unique edges (i < j) once, generate both O_ij and O_ji blocks
    root_indices = get_square_matrix_coo_indices(d)
    
    # Extract unique edges (i < j) to avoid duplicate alignment work
    mask = P_indices[0] < P_indices[1]
    unique_edge_indices = P_indices[:, mask]  # (2, num_unique_edges)
    num_unique_edges = unique_edge_indices.shape[1]
    
    # Build mapping from (i,j) to p_ij value for quick lookup
    edge_to_p_val = {}
    for edge_idx in range(P_indices.shape[1]):
        i = int(P_indices[0, edge_idx].item())
        j = int(P_indices[1, edge_idx].item())
        p_val = float(P_values[edge_idx].item())
        edge_to_p_val[(i, j)] = p_val
    
    Q_ijs = []
    Q_indices = []
    
    # Process each unique edge once, generating both directions
    # NOTE: since SVD can return a decomposition that is non-unique
    # Loop through unique edges (i < j) once, generate both O_ij and O_ji blocks
    for edge_idx in range(num_unique_edges):
        i = int(unique_edge_indices[0, edge_idx].item())
        j = int(unique_edge_indices[1, edge_idx].item())
        # Feature-space transport using O-frames from vector_feat_key
        O_i = O_i_dict[i]
        O_j = O_i_dict[j]

        # Align O_j to O_i once
        if enforce_sign:
            if sing_vect_align_method == 'procrustes':
                O_j_aligned = align_Oj_to_Oi_procrustes(O_i, O_j)
            else:
                O_j_aligned = match_B_col_directions_to_A(O_i, O_j)
        else:
            O_j_aligned = O_j

        if equivar_or_invar == "equivariant":
            O_ij_unweighted = torch.mm(O_i, O_j_aligned.T)
            O_ji_unweighted = torch.mm(O_j_aligned, O_i.T)
        else:  # invariant
            O_ij_unweighted = torch.mm(O_i.T, O_j_aligned)
            O_ji_unweighted = torch.mm(O_j_aligned.T, O_i)
        
        # Apply P weights to both blocks
        p_ij = edge_to_p_val.get((i, j), 0.0)
        p_ji = edge_to_p_val.get((j, i), 0.0)
        
        Q_ij = p_ij * O_ij_unweighted
        Q_ji = p_ji * O_ji_unweighted
        
        # Add i->j block
        Q_ijs.append(Q_ij.ravel())
        offset_i, offset_j = (d * i), (d * j)
        Q_indices.append(torch.stack([
            root_indices[0] + offset_i,
            root_indices[1] + offset_j
        ]))
        
        # Add j->i block
        Q_ijs.append(Q_ji.ravel())
        Q_indices.append(torch.stack([
            root_indices[0] + offset_j,
            root_indices[1] + offset_i
        ]))
    
    # Add lazy self-loops (0.5 * I_d for each node)
    # (We add them manually since O_i @ O_i.T = I_d when O_i is orthonormal and square)
    num_nodes = v.shape[0]
    I_d = torch.eye(d, dtype=v.dtype, device=v.device) * 0.5
    for i in range(num_nodes):
        Q_ijs.append(I_d.ravel())
        offset_i = d * i
        Q_indices.append(torch.stack([
            root_indices[0] + offset_i,
            root_indices[1] + offset_i
        ]))
    
    # concatenate all indices and values tensors
    Q_ijs = torch.cat(Q_ijs)
    Q_indices = torch.cat(Q_indices, dim=1)
    # print(Q_indices.shape)
    
    # Q has shape (dn x dn), e.g. (3n x 3n) in R^3
    # where n = v.shape[0], the number elements in the 
    # (length-d) vector attributes
    side_size = d * v.shape[0]
    Q = torch.sparse_coo_tensor(
        indices=Q_indices.to(device), 
        values=Q_ijs.to(device), 
        size=(side_size, side_size)
    )
        
    # Optional: build line-graph operator P_line
    def _build_line_graph_operator(_data: Data, _device: str) -> torch.Tensor:
        """
        Construct the diffusion operator on the line graph L(G): nodes are original edges.
        Two line-graph nodes are connected if their corresponding edges in G share an endpoint.
        """
        # Ensure undirected pairing for stable incidence
        ei = _to_undirected(_data.edge_index) if hasattr(_data, 'edge_index') else None
        if ei is None:
            raise ValueError("Data must have edge_index to construct line graph")
        # Number of original edges (treat each undirected pair once)
        # Build canonical edge list (i<j)
        src = ei[0].to(torch.long)
        dst = ei[1].to(torch.long)
        mask = src < dst
        src = src[mask]
        dst = dst[mask]
        E = src.numel()
        # Map (i,j) -> edge id
        edge_ids = {}
        for k in range(E):
            edge_ids[(int(src[k].item()), int(dst[k].item()))] = k
        # Incidence: node -> list of incident edge ids
        num_nodes = int(_data.num_nodes) if hasattr(_data, 'num_nodes') else int(ei.max().item() + 1)
        incident = [[] for _ in range(num_nodes)]
        for k in range(E):
            i = int(src[k].item()); j = int(dst[k].item())
            incident[i].append(k)
            incident[j].append(k)
        # Build line-graph edges: connect incident edge pairs per original node
        lg_src = []
        lg_dst = []
        for nbr_edges in incident:
            # connect all unordered pairs among incident edges (both directions)
            n = len(nbr_edges)
            for a in range(n):
                ea = nbr_edges[a]
                for b in range(a + 1, n):
                    eb = nbr_edges[b]
                    # add both directions
                    lg_src.append(ea); lg_dst.append(eb)
                    lg_src.append(eb); lg_dst.append(ea)
        if len(lg_src) == 0:
            lg_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            lg_edge_index = torch.tensor([lg_src, lg_dst], dtype=torch.long)
        lg_data = Data(edge_index=lg_edge_index)
        # Number of nodes in line graph is E
        lg_data.num_nodes = E
        return get_P(lg_data, row_normalize=True, device=_device).coalesce()

    P_line = None
    if compute_line_operator:
        P_line = _build_line_graph_operator(data, device)

    if return_data_object:
        # assign new attributes to the PyG data object and return
        # use 'v' for new vector feature vector, to avoid collision
        # with the scalar feature matrix 'x'
        data['P'] = P
        data['Q'] = Q
        if P_line is not None:
            data['P_line'] = P_line
        # data[new_vector_feat_key] = v.view(-1, 1)  # now done in ESCGNNDatasetHDF5
    
        # use ESCGNNData (Data subclass) so attributes like Q
        # collate how we want in batching
        # Ensure batching treats all operators (including P_line) as block-diagonal
        _op_keys = ('P', 'Q') if (P_line is None) else ('P', 'Q', 'P_line')
        data = ESCGNNData(**dict(data), operator_keys=_op_keys)
        return data
    else:
        # Cast dtype for saving
        target_dtype = getattr(torch, hdf5_tensor_dtype)

        tensor_data_dict = {
            'original_idx': data_i,
            'Q': {
                'indices': Q._indices(),
                'values': Q._values().to(target_dtype),
                'size': Q.size()
            }
        }
        if save_scalar_operator:
            tensor_data_dict['P'] = {
                'indices': P._indices(),
                'values': P._values().to(target_dtype),
                'size': P.size()
            }
        if P_line is not None:
            tensor_data_dict['P_line'] = {
                'indices': P_line._indices(),
                'values': P_line._values().to(target_dtype),
                'size': P_line.size()
            }
        # If new graph connectivity was computed, add it to the dict
        if graph_construction is not None:
            tensor_data_dict['edge_index'] = edge_index.cpu().to(torch.int64)
            if edge_weight is not None:
                tensor_data_dict['edge_weight'] = edge_weight.to(target_dtype).cpu()

        # If new edge features were computed, add them to the dict
        if (num_edge_features > 0):
            tensor_data_dict['edge_features'] = {
                'values': data.edge_features.flatten(),
                'shape': torch.tensor(data.edge_features.shape, dtype=torch.int64)
            }

        # If Dirac nodes are requested, add them to the dict
        if include_diracs:
            dirac_nodes = dirac_fn(
                data,
                **dirac_kwargs
            )
            tensor_data_dict['dirac_nodes'] = dirac_nodes.cpu().to(torch.int64)

        return tensor_data_dict


# This function is useful after Qv multiplication, which is
# (nd, nd) x (nd, 1) -> (nd, 1). This function can reshape 
# Qv 'back' into (n, d) [the vector feature matrix shape 
# where rows are nodes' vector features].
def reshape_nd_vector_to_n_by_d_matrix(
    v: torch.Tensor,
    n: int
) -> torch.Tensor:
    """
    Reshapes a raveled vector (where v1 is stacked
    on v2 and so on) of shape (nd, 1), where n is 
    the number of vectors and d is their constant
    dimension, into a matrix where the vectors are
    rows, of shape (n, d).
    
    Args:
        v: tensor of shape (nd, 1), where
            n is the number of nodes and d
            is the vector feature dimension.
        n: the number of nodes in the graph.
    Returns:
        Matrix tensor of shape (n, d).
    """
    return v.reshape(n, -1)
    # return torch.stack(
    #     torch.chunk(v.squeeze(), n), 
    #     dim=0
    # )


def process_edge_features_to_node_features(data: Data) -> Data:
    """
    Process edge features into node features by aggregating bond types for each node.
    For bond-typed datasets, edge_attr can be a tensor of shape (num_edges, K)
    with one-hot encoded bond types.
    
    For each node, we compute the proportion of each bond type it has,
    resulting in a new node feature of shape (num_nodes, 4).
    
    Args:
        data: PyTorch Geometric Data object with edge_attr tensor
        
    Returns:
        Modified Data object with new node features added
    """
    # Get edge index and attributes
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    
    # Initialize tensor to store bond type counts for each node
    num_nodes = data.num_nodes
    num_bond_types = edge_attr.size(1)
    bond_counts = torch.zeros((num_nodes, num_bond_types), dtype=torch.float)
    
    # Count bond types for each node
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        bond_counts[src] += edge_attr[i]
        bond_counts[dst] += edge_attr[i]  # Count for both source and target nodes
    
    # L1 normalize to get proportions
    row_sums = bond_counts.sum(dim=1, keepdim=True)
    # Avoid division by zero for isolated nodes
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    bond_proportions = bond_counts / row_sums
    
    # Add the new features to the data object
    data.bond_node_features = bond_proportions
    
    return data


"""
Methods for re-attaching sparse COO tensors to Data objects
"""
def load_coo_tensors(
    filepath: str,
    tensor_keys: Tuple[str] = ('P', 'Q')
) -> Dict[str, torch.Tensor]:
    """
    Loads sparse COO (Coordinate) format tensors from a saved file and reconstructs them.
    
    Args:
        filepath: Path to the saved tensor file.
        tensor_keys: Tuple of string keys for the tensors to load. Defaults to ('P', 'Q').
    
    Returns:
        Dictionary mapping tensor keys to reconstructed sparse COO tensors.
    """
    try:
        with open(filepath, 'rb') as f:
            packed = torch.load(f, weights_only=True)
        return {
            key: torch.sparse_coo_tensor(
                packed[key]['indices'], 
                packed[key]['values'], 
                packed[key]['size']
            ) \
            for key in tensor_keys
        }
    finally:
        # Ensure file is closed even if an error occurs
        if 'f' in locals():
            f.close()


def process_single_file(
    tensor_dict: Dict[str, Dict[str, torch.Tensor]],
    data_idx: int,
    dataset: List[Data],
    tensor_keys: Tuple[str, ...],
) -> Tuple[int, ESCGNNData]:
    """
    Process a single tensor dictionary and attach it to the corresponding Data object.
    
    Args:
        tensor_dict: Dictionary containing tensor data with keys for indices, values, and size
        data_idx: Original index of the Data object in the full dataset
        dataset: List of Data objects to process
        tensor_keys: Tuple of string keys for the tensors to attach
        
    Returns:
        Tuple containing:
        - data_idx: The index of the processed Data object
        - data: The processed ESCGNNData object with attached tensors
        
    Raises:
        ValueError: If required tensor data is missing for any key
        IndexError: If the original index is not found in the dataset
    """
    # Find the data object with matching original_idx
    data = None
    for i, d in enumerate(dataset):
        if hasattr(d, 'original_idx') and d.original_idx == data_idx:
            data = d
            break
    
    if data is None:
        raise IndexError(f"Could not find data object with original_idx {data_idx} in dataset")
    
    # Attach tensors / attributes according to key type
    for key in tensor_keys:
        if key not in tensor_dict:
            # Treat edge_weight and edge_features as optional when absent
            if key in ('edge_weight', 'edge_features'):
                continue
            raise ValueError(f"Missing tensor data for key '{key}' in tensor dict")

        # ---------------- Sparse operators ----------------
        if key in ('P', 'Q'):
            tensor_data = tensor_dict[key]
            indices = tensor_data['indices']
            values = tensor_data['values']
            size = tensor_data['size']

            sparse_tensor = torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=size
            )
            setattr(data, key, sparse_tensor)

        # ---------------- Dense graph structure ----------------
        elif key == 'edge_index':
            data.edge_index = tensor_dict['edge_index'].long()
            # Remove bond-based attributes if present
            if hasattr(data, 'edge_attr'):
                delattr(data, 'edge_attr')

        elif key == 'edge_weight':
            data.edge_weight = tensor_dict['edge_weight']

        # ---------------- Dense edge features (e.g. Bessels) ----------------
        elif key == 'edge_features':
            b_data = tensor_dict['edge_features']
            vals = b_data['values']
            shape = b_data['shape']
            data.edge_features = vals.view(*shape)

        else:
            raise ValueError(f"Unsupported tensor key '{key}' encountered in tensor dict")
    
    # Convert to ESCGNNData object
    if not isinstance(data, ESCGNNData):
        data = ESCGNNData(**dict(data))
    
    return data_idx, data


def process_batch_file(
    file: str,
    dir_path: str,
    dataset: List[Data],
    tensor_keys: Tuple[str, ...],
    clear_every_n_graphs: int = 2048
) -> List[Tuple[int, ESCGNNData]]:
    """
    Process a batch file containing multiple tensor dictionaries.
    
    Args:
        file: Name of the batch file to process
        dir_path: Directory containing the batch file
        dataset: List of Data objects to process
        tensor_keys: Tuple of string keys for the tensors to attach
        clear_every_n_graphs: Number of graphs to process before clearing CUDA cache.
            Default is 2048. Set to 0 to disable cache clearing.
        
    Returns:
        List of tuples containing:
        - data_idx: The index of each processed Data object
        - data: The processed ESCGNNData object with attached tensors
        
    Raises:
        ValueError: If no 'original_idx' is found in any tensor dict
    """
    results = []
    f = None
    try:
        f = open(os.path.join(dir_path, file), 'rb')
        batch_data = torch.load(f, weights_only=True)
        if isinstance(batch_data, list):
            for i, tensor_dict in enumerate(batch_data):
                original_idx = tensor_dict.get('original_idx')
                if original_idx is None:
                    raise ValueError(f"No 'original_idx' found in tensor dict at index {i} in {file}")
                results.append(process_single_file(tensor_dict, original_idx, dataset, tensor_keys))
                
                # Clear cache after every nth graph within this batch
                if (clear_every_n_graphs > 0) and ((i + 1) % clear_every_n_graphs == 0):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        # Ensure file is closed even if an error occurs
        if f is not None:
            f.close()
        # Clear any remaining memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
    return results


def process_individual_file(
    file: str,
    dir_path: str,
    dataset: List[Data],
    tensor_keys: Tuple[str, ...],
) -> Tuple[int, ESCGNNData]:
    """
    Process an individual tensor file.
    
    Args:
        file: Name of the individual file to process
        dir_path: Directory containing the file
        dataset: List of Data objects to process
        tensor_keys: Tuple of string keys for the tensors to attach
        
    Returns:
        Tuple containing:
        - data_idx: The index of the processed Data object
        - data: The processed ESCGNNData object with attached tensors
        
    Raises:
        ValueError: If no 'original_idx' is found in the tensor dict or if the filename format is invalid
    """
    f = None
    try:
        f = open(os.path.join(dir_path, file), 'rb')
        tensor_dict = torch.load(f, weights_only=True)
        original_idx = tensor_dict.get('original_idx')
        if original_idx is None:
            raise ValueError(f"No 'original_idx' found in tensor dict in {file}")
        result = process_single_file(tensor_dict, original_idx, dataset, tensor_keys)
        
        # Clear cache after processing individual file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
            
        return result
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid filename format: {file}") from e
    finally:
        # Ensure file is closed even if an error occurs
        if f is not None:
            f.close()
        # Clear any remaining memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection


def attach_coo_tensor_attr_to_data(
    dataset: Data,
    dir_path: str,
    tensor_keys: Tuple[str, ...] = ('P', 'Q'),
    warn_if_mismatch: bool = True,
    num_workers: int = 1,
    sort_by_original_idx: bool = False,
    clear_every_n_chunks: int = 1,
    chunk_size: int = 1
) -> List[ESCGNNData]:
    """
    Attach sparse COO tensors to Data objects in the dataset.
    
    Args:
        dataset: Dataset containing Data objects
        dir_path: Directory containing saved tensor files
        tensor_keys: Keys for tensor data (default: ('P', 'Q'))
        warn_if_mismatch: Whether to warn if tensor file count doesn't match dataset size
        num_workers: Number of worker processes to use for parallel processing
        sort_by_original_idx: Whether to sort the results by the original index
        clear_every_n_chunks: Number of chunks to process before clearing CUDA cache.
            Default is 1. Set to 0 to disable cache clearing.
        chunk_size: Number of files to process in each chunk. Default is 1. Helps with memory 
            usage and making sure too many files are not opened at once.
    Returns:
        List of ESCGNNData objects with sparse tensors attached
        
    Raises:
        ValueError: If no tensor data is found for a graph
        RuntimeError: If tensor file count doesn't match dataset size
    """
    import multiprocessing as mp
    from functools import partial
    import gc
    
    # Only process data in the main process
    if torch.distributed.is_initialized() and (torch.distributed.get_rank() != 0):
        # If not in main process, wait for main process to finish
        torch.distributed.barrier()
        return dataset
    
    # Count total number of graphs across all tensor files
    total_graphs = 0
    batch_files = []
    individual_files = []
    
    # Use a context manager to ensure the file is closed after reading
    for file in os.listdir(dir_path):
        if file.endswith('.pt'):
            if file.startswith('batch_'):
                # Load batch file to count graphs
                with open(os.path.join(dir_path, file), 'rb') as f:
                    batch_data = torch.load(f, weights_only=True)
                    if isinstance(batch_data, list):
                        total_graphs += len(batch_data)
                        batch_files.append(file)
            else:
                # Individual file
                total_graphs += 1
                individual_files.append(file)
    
    if warn_if_mismatch and (total_graphs != len(dataset)):
        warnings.warn(
            f"Number of graphs in tensor files ({total_graphs}) "
            f"does not match dataset size ({len(dataset)})"
        )

    # Initialize list to store results
    all_results = []
    
    # Process files in parallel
    if num_workers > 1:
        ctx = mp.get_context('fork')  # 'fork' is safe on Linux/Mac
        with ctx.Pool(processes=num_workers) as pool:
            # Process batch files if any exist
            if batch_files:
                # Process batch files in chunks
                for i in range(0, len(batch_files), chunk_size):
                    chunk = batch_files[i:i + chunk_size]
                    process_batch = partial(
                        process_batch_file, 
                        dir_path=dir_path, 
                        dataset=dataset, 
                        tensor_keys=tensor_keys,
                        clear_every_n_graphs=0  # Disable per-graph cache clearing
                    )
                    batch_results = pool.map(process_batch, chunk)
                    all_results.extend([result for batch_result in batch_results for result in batch_result])
                    
                    # Clear cache every n chunks
                    if clear_every_n_chunks > 0 and (i // chunk_size + 1) % clear_every_n_chunks == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()  # Force garbage collection
            
            # Process individual files if any exist
            if individual_files:
                # Process individual files in chunks
                for i in range(0, len(individual_files), chunk_size):
                    chunk = individual_files[i:i + chunk_size]
                    process_individual = partial(
                        process_individual_file,
                        dir_path=dir_path,
                        dataset=dataset,
                        tensor_keys=tensor_keys,
                    )
                    individual_results = pool.map(process_individual, chunk)
                    all_results.extend([result for result in individual_results])
                    
                    # Clear cache every n chunks
                    if clear_every_n_chunks > 0 and (i // chunk_size + 1) % clear_every_n_chunks == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()  # Force garbage collection
    else:
        # Process files sequentially, batch then individual
        if batch_files:
            batch_results = [
                process_batch_file(
                    file, dir_path, dataset, tensor_keys, 
                    clear_every_n_graphs=0  # Disable per-graph cache clearing
                ) for file in batch_files
            ]
            all_results.extend([result for batch_result in batch_results for result in batch_result])
            
            # Clear cache after processing all batch files
            if clear_every_n_chunks > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection
        
        if individual_files:
            individual_results = [
                process_individual_file(file, dir_path, dataset, tensor_keys) \
                for file in individual_files
            ]
            all_results.extend([result for result in individual_results])
            
            # Clear cache after processing all individual files
            if clear_every_n_chunks > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection
    
    if sort_by_original_idx:
        # Sort results by data_idx and extract just the data objects
        all_results.sort(key=lambda x: x[0])  # Sort by data_idx

    all_results = [data for _, data in all_results]  # Return just the data objects (omit the data_idx)
    
    # If using distributed training, ensure all processes wait for main process to finish
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    return all_results


def compute_bessel_edge_features(
    dists: torch.Tensor,
    cosine_cut: torch.Tensor,
    adjacency_distance_cutoff: float,
    num_edge_features: int,
    hdf5_tensor_dtype: str,
) -> torch.Tensor:
    """
    Compute Bessel function-based edge features for a set of edge distances.
    Args:
        dists: Tensor of edge distances (E,)
        cosine_cut: Tensor of cosine cutoff values (E,)
        adjacency_distance_cutoff: float, cutoff radius
        num_edge_features: int, number of Bessel functions (K)
        hdf5_tensor_dtype: str, dtype for output tensor
    Returns:
        Tensor of shape (E, K) with Bessel features
    """
    k_idx = torch.arange(1, num_edge_features + 1, device=dists.device).float()  # (K,)
    d_col = dists.unsqueeze(1)  # (E,1)
    bessel = torch.sin(k_idx * math.pi * d_col / adjacency_distance_cutoff) / (d_col + 1e-8)
    bessel = bessel * cosine_cut.unsqueeze(1)  # (E,K)
    return bessel.to(getattr(torch, hdf5_tensor_dtype))




''' DEPRECATED
def K_chemical_weights(
    V_i: torch.Tensor,
    bond_types_i: torch.Tensor,
    z_i: int,
    z_neighbors: torch.Tensor,
    eps: float = 0.05,
    bond_multipliers: Optional[torch.Tensor] = None,
    use_pair_eneg_factor: bool = False,
    eneg_table: Optional[Dict[int, float]] = None,
    covalent_radii: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    r"""
    Chemically aware distance-weighting kernel for molecular graphs.

    Weight for neighbor $j$ of center $i$ is constructed as
        $$w_{ij} = b_{ij} \cdot f(Z_i, Z_j) \cdot \exp(-((||r_i - r_j|| - L_0)^2) / \epsilon)$$
    where
        - $b_{ij}$: multiplier depending on bond type (single, double, triple, aromatic)
        - $f$: optional electronegativity / element-pair factor (default 1)
        - radial: Gaussian on the difference between observed bond length
            and a reference covalent bond length $L_0$ that depends on the element pair.

    All factors are vectorized; if optional tables are not supplied,
    sensible defaults for common elements are embedded.

    Args:
        V_i: tensor of the shape (num_neighbors, d)
            matrix for the ith node/point's recentered 
            neighborhood points.
        bond_types_i: (k,) bond-type indices 0-3 (single, 
            double, triple, aromatic).
        z_i: atomic number of central atom (int).
        z_neighbors: (k,) atomic numbers of neighbors.
        eps: radial Gaussian width (in Å²). Smaller -> sharper 
            length filter.
        bond_multipliers: length-4 tensor giving multiplier per bond
            class. If None, uses tensor([1.0,1.3,1.5,1.1]).
        use_pair_eneg_factor: whether to multiply by an element-pair    
            electronegativity factor.
        eneg_table: dict mapping atomic number -> Pauling electronegativity.
            Only needed if use_pair_eneg_factor is True.
        covalent_radii: dict mapping atomic number -> covalent radius (Å).
            If None, built-in values (e.g., common elements) are used.
    Returns:
        Tensor of shape (k,) containing the weights w_{ij}.

    References:
        Pauling, L. The Nature of the Chemical Bond, 3rd ed. (1960)
        Cordero, S. et al. Dalton Trans. 2008, 2832-2838  (covalent radii)
        Behler, J.; Parrinello, M. Phys. Rev. Lett. 2007, 98, 146401 (radial Gaussian)
    """
    device = V_i.device

    if bond_multipliers is None:
        bond_multipliers = torch.tensor([1.0, 1.3, 1.5, 1.1], device=device)
    # Clip indices in case exotic bond types appear
    b = bond_multipliers[bond_types_i.clamp(0, len(bond_multipliers) - 1)]

    # Electronegativity factor f(Z_i, Z_j) = 1 + |Chi_i - Chi_j|
    if use_pair_eneg_factor:
        if eneg_table is None:
            eneg_table = {
                1: 2.20,  # H
                6: 2.55,  # C
                7: 3.04,  # N
                8: 3.44,  # O
                9: 3.98,  # F
            }
        chi_i = eneg_table.get(int(z_i), 2.5)
        chi_j = torch.tensor(
            [eneg_table.get(int(z), 2.5) for z in z_neighbors], device=device
        )
        f = 1.0 + torch.abs(chi_j - chi_i)
    else:
        f = 1.0

    # Reference covalent bond length L0 = r_covalent(Z_i) + r_covalent(Z_j)
    if covalent_radii is None:
        covalent_radii = {
            1: 0.31,  # H
            6: 0.76,  # C
            7: 0.71,  # N
            8: 0.66,  # O
            9: 0.57,  # F
        }
    r_i = covalent_radii.get(int(z_i), 0.75)
    r_j = torch.tensor(
        [covalent_radii.get(int(z), 0.75) for z in z_neighbors], device=device
    )
    L0 = r_i + r_j

    # Radial Gaussian term
    bond_len = torch.norm(V_i, dim=1)
    radial = torch.exp(-((bond_len - L0) ** 2) / eps)

    return b * f * radial


def K_Gaussian_weights(
    V_i: torch.Tensor,
    eps: Optional[float] = None,
    take_sqrt: bool = True,
) -> torch.Tensor:
    """
    Computes weights for V_i, the matrix where
    columns are (x_i - neighbor of x_i) vectors,
    using a Gaussian kernel.
    
    Args:
        V_i: tensor of the shape (num_neighbors, d)
            matrix for the ith node/point's recentered 
            neighborhood points.
        eps: optional rescaling factor (leaving
            'None' is equivalent to setting to
            1.0, but skips the division by eps).
        take_sqrt: whether to take the square root of the 
            weights. Default is True, since PCA/SVD is based
            on the singular values of the covariance matrix,
            and this function is primarily used before the local 
            SVD step.
    Returns:
       Tensor of shape (num_neighbors, ) of weights
       for rescaling the neighbor difference vectors.
    """
    numerator = -(torch.norm(V_i, p=2, dim=1) ** 2)
    if eps is not None:
        numerator /= eps
    w = torch.exp(numerator)
    if take_sqrt:
        w = torch.sqrt(w)
    return w


def K_CosineCutoff_weights(
    V_i: torch.Tensor,
    r_cut: float,
    take_sqrt: bool = True,
) -> torch.Tensor:
    """
    Computes weights for V_i, the matrix where
    columns are (x_i - neighbor of x_i) vectors,
    using a cosine cutoff kernel.

    Args:
        V_i: tensor of shape (num_neighbors, d)
            Matrix for the ith node's recentered 
            neighborhood vectors.
        r_cut: radial cutoff (same units as V_i). Values
            beyond this distance get weight 0.
        take_sqrt: whether to take the square root of the 
            weights (recommended for PCA/SVD usage).
    
    Returns:
        Tensor of shape (num_neighbors,) with weights for
        rescaling the neighbor difference vectors.
    """
    r = torch.norm(V_i, p=2, dim=1)  # shape: (num_neighbors,)
    w = 0.5 * (torch.cos(math.pi * r / r_cut) + 1.0)
    w = torch.where(r < r_cut, w, torch.zeros_like(w))
    w = torch.clamp(w, min=0.0)  # ensure no negative weights
    if take_sqrt:
        w = torch.sqrt(w)
    return w
'''