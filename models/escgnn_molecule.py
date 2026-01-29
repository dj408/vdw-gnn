"""

Notes:
1. We use a k-NN graph structure. Molecular properties are influenced by nonbonded interactions, and using only the chemical graph can miss those. Using k-NN captures local nonbonded geometry in addition to bonded edges.
- We can append Euclidean distances to the one-hot bond type edge features where this introduces non-bonded edges (see escgnn_data_classes.py). This way, non-bonded edges scattering coefficients (from the line graph) are richer than a monolithic 'no bond', when they could very greatly in distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
# from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from models.base_module import BaseModule
from typing import Dict, Any, Optional, List
from models.class_maps import MLP_NONLIN_MODULE_MAP

# ------------------------------------------------------------
# Model class
# ------------------------------------------------------------
class ESCGNN_MOLECULE(BaseModule):

    atomic_num_embed_mapping = {
        1: 0, 
        6: 1, 
        7: 2, 
        8: 3, 
        9: 4,
    }
    
    def __init__(
        self,
        *,
        # Edge feature width (scattered edge feature count)
        edge_context_dim: int,
        # Atom embedding inputs
        num_atom_types: int,
        atomic_number_key: str = "z",
        # Split wavelet counts to reflect vector/scalar diffusion configs
        num_wavelets_vector: int = 8,
        num_wavelets_scalar: int = 8,
        num_msg_pass_layers: int = 4,
        hidden_dim: int = 128,
        cross_condition_type: Optional[str] = None, # "cross_attention", "film"
        cross_condition_kwargs: Optional[Dict[str, Any]] = None, # {"num_heads": 4},
        vector_scatter_feature_key: str = "vector_scatter",
        line_scatter_feature_key: str = "edge_scatter",
        msg_kwargs: Optional[Dict[str, Any]] = None,  # e.g., "dropout"
        readout_kwargs: Optional[Dict[str, Any]] = {
            "readout_type": "deepsets",
        },
        base_module_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(**(base_module_kwargs or {}))

        # --- Set attributes ---
        self.msg_kwargs = msg_kwargs or {}
        self.readout_kwargs = readout_kwargs or {}
        self.readout_type = self.readout_kwargs['readout_type']
        self.readout_kwargs.pop('readout_type', None)
        self.vector_scatter_feature_key = vector_scatter_feature_key
        self.line_scatter_feature_key = line_scatter_feature_key
        self.num_atom_types = int(num_atom_types)
        self.atomic_number_key = str(atomic_number_key)
        self.num_wavelets_vector = int(num_wavelets_vector)
        self.num_wavelets_scalar = int(num_wavelets_scalar)

        # --- Initialize modules ---
        # Atom type embedding
        self.atom_emb = nn.Embedding(self.num_atom_types, hidden_dim)

        # Conditioner (cross-attention vs FiLM)
        # Attends over edges; less useful if input edge features are already globally informative (e.g., scattering coefficients)
        self.cross_condition_type = cross_condition_type
        if self.cross_condition_type is not None:

            # Learnable edge-message prototype (shared across edges), 
            # initialized to small random values
            self.proto_edge_msg = nn.Parameter(
                torch.randn(hidden_dim) * 1e-3
            )
            if cross_condition_type == "cross_attention":
                # Consume raw edge features (edge_dim) directly; conditioner projects to hidden_dim
                self.condition_fn = CrossAttentionConditioner(
                    hidden_dim, 
                    edge_context_dim, 
                    **cross_condition_kwargs,
                )
            elif cross_condition_type == "film":
                self.condition_fn = FiLMConditioner(
                    hidden_dim, 
                    hidden_dim, 
                    **cross_condition_kwargs,
                )
            else:
                raise ValueError(f"Unknown cross_condition_type: {cross_condition_type}")
        else:
            self.condition_fn = None

        # Message passing layers
        self.layers = nn.ModuleList([
            MessagePassingFlow(
                input_dim=edge_context_dim if (i == 0) else hidden_dim, 
                hidden_dim=hidden_dim,
                num_wavelets=self.num_wavelets_vector,
                **self.msg_kwargs
            )
            for i, _ in enumerate(range(num_msg_pass_layers))
        ])

        # Readout module
        self.readout = build_readout(
            readout_type=self.readout_type,
            input_dim=hidden_dim,
            num_vec_wavelets=self.num_wavelets_vector,
            readout_kwargs=self.readout_kwargs,
        )

    def forward(self, batch):
        """
        Args:
            batch: torch_geometric.data.Batch with attributes:
              - x: (N, node_dim) node features
              - edge_index: (2, E) edge connectivity
              - <line_scatter_feature_key>: (E, num_rbf_scatter * W_line) edge scatter features
              - <vector_scatter_feature_key>: (N, W_vec, d) vector scatter features
              - batch: (N,) batch indices
        """

        # Initialize outputs dict
        outputs: Dict[str, torch.Tensor] = {}

        # Include additional attributes with predictions/targets in the model output dict
        # in case the loss function needs them
        if self.attributes_to_include_with_preds is not None:
            for attr in self.attributes_to_include_with_preds:
                if attr in batch:
                    outputs[attr] = getattr(batch, attr)

        # Grab features from batch
        # x: scalar node state from atom embedding
        x = self._embed_atoms(batch)
        # E: scattered edge features
        scat_edge_feat = getattr(batch, self.line_scatter_feature_key)  # (E, num_rbf_scatter * W_line)
        # V: scattered vector features
        scat_vec_feat = getattr(batch, self.vector_scatter_feature_key)  # (N, W_vec, d)
        # Reorder to (N, d, W_vec) for downstream invariants/gating code
        scat_vec_feat = scat_vec_feat.permute(0, 2, 1).contiguous()
        edge_index = batch.edge_index
        batch_index = batch.batch

        # Load edge messages M from learnable shared prototype proto_edge_msg
        if self.cross_condition_type is not None:
            msgs_edge = self.proto_edge_msg \
                .unsqueeze(0) \
                .expand(scat_edge_feat.size(0), -1)  # (E, hidden_dim)

            # Condition msgs_edge by scat_edge_feat: cross-attend msgs_edge with per-edge fingerprints (scat_edge_feat) to turn the shared msgs_edge into contextualized, edge-specific messages
            if self.cross_condition_type == "cross_attention":
                # Compute per-edge graph indices to avoid attending across the entire batch
                # Map each edge to its source node's graph id
                edge_graph_index = batch_index[edge_index[0]]
                msgs_edge = self.condition_fn(
                    query=msgs_edge, 
                    key=scat_edge_feat, 
                    value=scat_edge_feat,
                    edge_graph_index=edge_graph_index,
                )
            elif self.cross_condition_type == "film":
                msgs_edge = self.condition_fn(
                    msgs_edge, 
                    scat_edge_feat,
                )
        else:
            # No cross-attention or FiLM conditioning:
            # Edge messages start as scat_edge_feat
            msgs_edge = scat_edge_feat

        # Run message passing layers
        for layer in self.layers:
            x, msgs_edge, scat_vec_feat = layer(
                x, 
                msgs_edge, 
                scat_vec_feat, 
                edge_index,
            )

        # Pool into graph-level representation
        preds = self.readout(x, scat_vec_feat, batch_index)

        # Return output dict with predictions
        outputs['preds'] = preds
        return outputs


    def _map_atomic_numbers(self, batch: Batch) -> torch.Tensor:
        """
        Map atomic numbers {1,6,7,8,9} to contiguous indices {0..4} to 
        avoid device-side embedding index errors.
        """
        z = getattr(batch, self.atomic_number_key).long()
        z_flat = z.view(-1).tolist()
        mapped = [self.atomic_num_embed_mapping.get(int(zz), 0) for zz in z_flat]
        idx = torch.tensor(mapped, device=z.device, dtype=torch.long)
        return idx


    def _embed_atoms(self, batch: Batch) -> torch.Tensor:
        """
        Produce per-node scalar embeddings from categorical atomic numbers in `batch`.

        If `batch[atomic_number_key]` is not already in 0..num_atom_types-1, it is
        remapped to contiguous indices based on sort order of unique encountered values.
        """
        if hasattr(batch, self.atomic_number_key):
            idx = self._map_atomic_numbers(batch)
            return self.atom_emb(idx)
        else:
            raise ValueError(f"Atomic number key {self.atomic_number_key} not found in batch")


# ------------------------------------------------------------
# Message Passing Layer
# ------------------------------------------------------------
class MessagePassingFlow(nn.Module):
    norm_modules = {
            "layer": nn.LayerNorm, 
            "batch": nn.BatchNorm1d,
            # "graph": nn.GraphNorm,
        }

    def __init__(
        self,
        input_dim,
        hidden_dim, 
        num_wavelets, 
        dropout=0.0,
        norm="layer",
    ):
        super().__init__()
        
        self.norm_module = self.norm_modules[norm]
        self.norm_layer = self.norm_module(input_dim)

        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # self.norm_module(hidden_dim), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.invariant_mlp = nn.Sequential(
            nn.Linear(3 * num_wavelets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, num_wavelets),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_mlp[0].bias, -2.0)

    def forward(self, x, M, V, edge_index):

        # Normalize messages before message passing
        M = self.norm_layer(M)

        # Edge update
        if self.input_proj is not None:
            M = self.input_proj(M)
        M = M + self.msg_mlp(M)

        # Edge invariants → α_ij aggregation weights, learned from invariants of scattered vector features V
        edge_invariants = compute_edge_invariants(V, edge_index)  # (M, 3*num_wavelets)
        alpha = torch.sigmoid(self.invariant_mlp(edge_invariants))  # (M, 1)

        # Node update: weighted sum of edge messages
        weighted_M = M * alpha
        m_i = scatter(weighted_M, edge_index[0], dim=0, reduce="sum")  # (N, hidden_dim)

        # Node update: residual-style
        x = x + self.node_mlp(m_i)

        # Vector gating update
        gates = self.gate_mlp(x)  # (N, num_wavelets)
        V = V * gates.unsqueeze(1)
        return x, M, V


# ------------------------------------------------------------
# Helper: Compute vector norms from scattered features
# ------------------------------------------------------------
def compute_vector_norms(V):
    """
    V: (N, 3, num_wavelets) tensor of vector features
    Returns:
        norms per wavelet: (N, num_wavelets)
    """
    norms = torch.linalg.norm(V, dim=1)  # (N, num_wavelets)
    return norms


def compute_edge_invariants(V, edge_index):
    """
    Compute edge-wise invariants from vector features.

    Args:
        V: (N, 3, num_wavelets) vector features
        edge_index: (2, M) edges (row=source, col=target)

    Returns:
        edge_invariants: (M, 3*num_wavelets)
            concat[ ||V_i||, ||V_j||, cos(V_i, V_j) ] for each edge
    """
    row, col = edge_index
    V_norms = compute_vector_norms(V)  # (N, num_wavelets)

    norms_src = V_norms[row]  # (M, num_wavelets)
    norms_dst = V_norms[col]  # (M, num_wavelets)

    dot_prods = (V[row] * V[col]).sum(dim=1)  # (M, num_wavelets)
    denom = norms_src * norms_dst + 1e-8
    cos_sims = dot_prods / denom  # (M, num_wavelets)

    edge_invariants = torch.cat([norms_src, norms_dst, cos_sims], dim=-1)  # (M, 3*num_wavelets)
    return edge_invariants


# ------------------------------------------------------------
# Cross-Attention Helper
# ------------------------------------------------------------
class CrossAttentionConditioner(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        edge_context_dim, 
        num_heads=4,
    ):
        """
        Implements a simple multi-head cross-attention:
        Query = messages M
        Key, Value = edge features E
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
        )
        # edge_proj handles K/V dimension alignment to the hidden_dim of queries Q
        self.edge_proj = nn.Linear(edge_context_dim, hidden_dim)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        edge_graph_index: Optional[torch.Tensor] = None,
    ):
        """
        query: M (num_edges, hidden_dim)
        key: E (num_edges, edge_dim)
        value: E (num_edges, edge_dim)
        """
        # Fast path: if no graph segmentation provided, attend across entire batch (O(E^2))
        if edge_graph_index is None:
            K = self.edge_proj(key).unsqueeze(0)   # (1, E, hidden_dim)
            V = self.edge_proj(value).unsqueeze(0) # (1, E, hidden_dim)
            Q = query.unsqueeze(0)                 # (1, E, hidden_dim)
            attn_out, _ = self.attn(Q, K, V)
            attn_out = attn_out.squeeze(0)
            return query + attn_out

        # Memory-safe path: run attention per-graph and stitch results back
        # edge_graph_index: (E,) with values in [0, B-1]
        device = query.device
        proj_K = self.edge_proj(key)
        proj_V = self.edge_proj(value)

        # Sort edges by graph id to create contiguous blocks per graph
        sorted_g, perm = torch.sort(edge_graph_index)
        Q_sorted = query[perm]
        K_sorted = proj_K[perm]
        V_sorted = proj_V[perm]

        # Find segment boundaries where graph id changes
        if Q_sorted.shape[0] == 0:
            return query  # nothing to do
        change_locs = torch.nonzero(sorted_g[1:] != sorted_g[:-1], as_tuple=False).flatten() + 1
        starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), change_locs])
        ends = torch.cat([change_locs, torch.tensor([Q_sorted.shape[0]], device=device, dtype=torch.long)])

        outputs = torch.empty_like(Q_sorted)
        for s, e in zip(starts.tolist(), ends.tolist()):
            Qb = Q_sorted[s:e].unsqueeze(0)
            Kb = K_sorted[s:e].unsqueeze(0)
            Vb = V_sorted[s:e].unsqueeze(0)
            out_b, _ = self.attn(Qb, Kb, Vb)
            outputs[s:e] = out_b.squeeze(0)

        # Invert permutation to restore original edge order
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=device)
        attn_out = outputs[inv_perm]
        return query + attn_out


# ------------------------------------------------------------
# FiLM Helper
# ------------------------------------------------------------
class FiLMConditioner(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        """
        FiLM layer: generates per-feature scale and shift from edge features
        and applies to messages M.
        """
        super().__init__()
        self.gamma = nn.Linear(edge_dim, hidden_dim)
        self.beta = nn.Linear(edge_dim, hidden_dim)

        # Initialize beta bias small negative to start near zero shift
        nn.init.constant_(self.beta.bias, -1.0)

    def forward(self, M, E):
        """
        M: messages (num_edges, hidden_dim)
        E: edge features (num_edges, edge_dim)
        """
        gamma = self.gamma(E)  # scale
        beta = self.beta(E)    # shift
        return M * (1 + gamma) + beta


# ------------------------------------------------------------
# Readout functions
# ------------------------------------------------------------
class DeepSetsReadout(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_vec_wavelets: int, 
        hidden_dim: int = 128,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim + num_vec_wavelets, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),  # regression target
        )

    def forward(self, x, V, batch):
        invariants = compute_vector_norms(V)  # (N, num_wavelets)
        h = torch.cat([x, invariants], dim=-1)  # (N, node_dim + num_vec_wavelets)

        h = self.pre_mlp(h)
        h = scatter(h, batch, dim=0, reduce="sum")  # (B, hidden_dim)
        out = self.post_mlp(h)  # (B, 1)
        return out


class MLPReadout(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_vec_wavelets: int, 
        output_dim: int = 1,
        hidden_dim: List[int] = [128, 64, 32, 16],
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        input_and_hidden_dims = [input_dim + num_vec_wavelets] + hidden_dim
        layers = []
        for i in range(len(input_and_hidden_dims) - 1):
            layers.append(nn.Linear(input_and_hidden_dims[i], input_and_hidden_dims[i + 1]))
            layers.append(activation())
            layers.append(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, V, batch):
        invariants = compute_vector_norms(V)  # (N, num_wavelets)
        h = torch.cat([x, invariants], dim=-1)

        h = scatter(h, batch, dim=0, reduce="sum")
        out = self.mlp(h)
        return out


def build_readout(
    readout_type: str, 
    input_dim: int, 
    num_vec_wavelets: int, 
    readout_kwargs: Optional[Dict[str, Any]] = {},
):
    # Map activation key to torch.nn.Module class
    activation_key = readout_kwargs.get("activation", 'silu')
    readout_kwargs["activation"] = MLP_NONLIN_MODULE_MAP[activation_key]

    if readout_type == "deepsets":
        return DeepSetsReadout(
            input_dim, 
            num_vec_wavelets, 
            **readout_kwargs,
        )
    elif readout_type == "mlp":
        return MLPReadout(
            input_dim, 
            num_vec_wavelets, 
            **readout_kwargs,
        )
    else:
        raise ValueError(f"Unknown readout_type {readout_type}")

