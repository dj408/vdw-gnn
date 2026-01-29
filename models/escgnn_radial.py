from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    global_add_pool, 
    global_mean_pool, 
    global_max_pool,
)
from torch_geometric.utils import to_undirected
# Local imports
import models.base_module as bm
from geo_scat import batch_scatter
from pyg_utilities import check_if_undirected

try:
    from torch_scatter import scatter_add
except ImportError:  # Fallback – slower pyg util, diff params order
    from torch_geometric.utils import scatter_
    scatter_add = lambda src, index, dim=0, dim_size=None: scatter_("add", src, index, dim, dim_size)
    

class ESCGNNRadial(bm.BaseModule):
    """
    Bessel-RBF ESCGNN with joint scalar/vector tracks and low-rank 
    gated message passing.
    """
    def __init__(
        self,
        # --- Feature keys -------------------------------------------------
        scalar_feature_key: str = "z",
        vector_feature_key: str = "pos",
        vec_feat_norms_stats: Optional[Dict[str, torch.Tensor]] = None,
        edge_rbf_key: str = "edge_features",
        bond_type_key: str = "edge_type",
        # --- Basic dims ---------------------------------------------------
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        bond_emb_dim: int = 8,
        num_rbf: int = 16,
        scalar_emb_dim: int = 64,
        d_vector: int = 3,
        use_temporal_residuals: bool = True,
        eps: float = 1e-10,  # small constant to avoid division by zero
        # --- Wavelet settings --------------------------------------------
        scalar_diffusion_op_key: str = "P",
        vector_diffusion_op_key: str = "Q",
        ablate_scalar_embedding: bool = False,
        ablate_scalar_wavelet_batch_norm: bool = False,
        ablate_vector_wavelet_batch_norm: bool = True,
        ablate_second_order_wavelets: bool = False,
        wavelet_J: int = 4,
        wavelet_J_prime: Optional[int] = None,
        include_lowpass_wavelet: bool = True,
        use_dirac_nodes: bool = True,
        # --- Scalar condensation MLP -------------------------------------
        scalar_condense_hidden_dims: Sequence[int] = [256, 128, 64],
        d_scalar_hidden: int = 64,
        # --- Custom diffusion scales (optional) --------------------------
        custom_scalar_scales: Optional[torch.Tensor] = None,
        custom_vector_scales: Optional[torch.Tensor] = None,
        # --- Vector gate MLP / Low-rank params ---------------------------
        vector_gate_mlp_hidden_dims: Sequence[int] = [128, 128],
        vector_gate_mlp_nonlin_fn: nn.Module = nn.SiLU,
        vector_gate_rank: int = 8,
        # --- Scalar gate MLP / Low-rank params ---------------------------
        scalar_gate_mlp_hidden_dims: Sequence[int] = [128, 128],
        scalar_gate_nonlin_fn: nn.Module = nn.SiLU,
        scalar_gate_rank: int = 8,
        # --- Pooling / read-out ------------------------------------------
        pool_stats: Sequence[str] = ('sum', 'mean'),
        readout_hidden_dims: Sequence[int] = [256, 128, 64],
        mlp_nonlin_fn: nn.Module = nn.SiLU,
        pred_output_dim: int = 1,
        # ---- BaseModule kwargs -----------------------------------------
        base_module_kwargs: Optional[Dict] = None,
        num_msg_pass_layers: int = 1,
        use_residual_connections: bool = False,
    ) -> None:
        bm.BaseModule.__init__(self, **(base_module_kwargs or {}))
        # override the BaseModule flag
        self.has_lazy_parameter_initialization = True

        # Store keys / hparams
        self.scalar_feature_key = scalar_feature_key
        self.vector_feature_key = vector_feature_key
        self._vec_stats = vec_feat_norms_stats
        self.edge_rbf_key = edge_rbf_key
        self.bond_type_key = bond_type_key
        self.scalar_diffusion_op_key = scalar_diffusion_op_key
        self.vector_diffusion_op_key = vector_diffusion_op_key
        self.eps = eps
        self.d_scalar = scalar_emb_dim
        self.d_vector = d_vector
        self.ablate_scalar_embedding = ablate_scalar_embedding
        self.ablate_scalar_wavelet_batch_norm = ablate_scalar_wavelet_batch_norm
        self.ablate_vector_wavelet_batch_norm = ablate_vector_wavelet_batch_norm
        self.scat_orders = (1, 2) if not ablate_second_order_wavelets else (1,)
        self.wavelet_J = wavelet_J
        self.wavelet_J_prime = wavelet_J_prime
        self.include_lowpass_wavelet = include_lowpass_wavelet
        self.use_dirac_nodes = use_dirac_nodes
        self.use_temporal_residuals = use_temporal_residuals
        self.pool_stats = tuple(pool_stats)
        self.num_rbf = num_rbf
        self.num_msg_pass_layers = num_msg_pass_layers
        self.use_residual_connections = use_residual_connections
        # Per-layer MLP containers
        self.scalar_condense_mlps = nn.ModuleList()
        self.vector_gate_mlps = nn.ModuleList()
        self.scalar_gate_mlps = nn.ModuleList()

        # Embeddings
        if not ablate_scalar_embedding:
            self.scalar_embedding = nn.Embedding(
                num_atom_types, scalar_emb_dim
            )
        self.bond_embedding = nn.Embedding(num_bond_types + 1, bond_emb_dim)  # +1 dummy class

        # Store custom diffusion scales
        self.custom_scalar_scales = custom_scalar_scales
        self.custom_vector_scales = custom_vector_scales

        # Learnable linear layer to collapse scalar channels (d_scalar) → 1
        # self.scalar_channel_mixer = nn.Linear(scalar_emb_dim, 1, bias=False)

        self.scalar_condense_hidden_dims = scalar_condense_hidden_dims
        self.d_scalar_hidden = d_scalar_hidden
        self.vector_gate_mlp_hidden_dims = vector_gate_mlp_hidden_dims
        self.vector_gate_mlp_nonlin_fn = vector_gate_mlp_nonlin_fn \
            if isinstance(vector_gate_mlp_nonlin_fn, type) else vector_gate_mlp_nonlin_fn.__class__
        self.vector_gate_rank = vector_gate_rank
        self.scalar_gate_mlp_hidden_dims = scalar_gate_mlp_hidden_dims
        self.scalar_gate_nonlin_fn = scalar_gate_nonlin_fn \
            if isinstance(scalar_gate_nonlin_fn, type) else scalar_gate_nonlin_fn.__class__
        self.scalar_gate_rank = scalar_gate_rank

        # Lazy modules and containers
        self.scalar_condense_mlp: Optional[nn.Sequential] = None
        self.readout_mlp: Optional[nn.Sequential] = None
        self._scalar_bn_layers = nn.ModuleList()
        self._vector_bn_layers = nn.ModuleList() 
        self.vector_gate_basis: Optional[nn.Parameter] = None
        self.scalar_gate_basis: Optional[nn.Parameter] = None

        # Readout MLP
        self._readout_hidden_dims = list(readout_hidden_dims)
        self._readout_act_cls = mlp_nonlin_fn \
            if isinstance(mlp_nonlin_fn, type) else mlp_nonlin_fn.__class__
        self._pred_output_dim = pred_output_dim

        # ------------------------------------------------------------------
        # DEPRECATED: Vector-track global statistics (Welford) and per-graph cache
        # ------------------------------------------------------------------
        # Statistics dict keys:
        #   sum   : running sum over node-wise norms (Tensor[W])
        #   sumsq : running sum of squares (Tensor[W])
        #   count : total number of node samples accumulated (int)
        #   mean  : final mean per wavelet band (Tensor[W]) – set in finalize
        #   std   : final std per band  (Tensor[W]) – set in finalize
        #   ready : bool flag indicating mean/std are frozen and ready for use
        # self._vec_stats: Dict[str, Any] = {
        #     "sum":   None,
        #     "sumsq": None,
        #     "count": 0,
        #     "mean":  None,
        #     "std":   None,
        #     "ready": False,
        # }

        # Per-graph cache:  original_idx -> standardized vector norms (fp32 CPU)
        # self._vec_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # General MLP builder
    # ------------------------------------------------------------------
    def _build_or_get_mlp(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        nonlin_fn: type[nn.Module],
        output_dim: int,
        weight_init_fn = nn.init.kaiming_uniform_,
        attr_name: Optional[str] = None,  # Only for single-use MLPs
    ) -> nn.Sequential:
        """
        Build or retrieve an MLP. If attr_name is provided, register as self.{attr_name} (for single-use MLPs, e.g., the readout MLP).
        For per-layer MLPs, just return the MLP.
        """
        if attr_name is not None:
            mlp = getattr(self, attr_name, None)
            dims = [input_dim] + list(hidden_dims) + [output_dim]
            if (
                mlp is None
                or not isinstance(mlp, nn.Sequential)
                or getattr(mlp[0], 'in_features', None) != input_dim
                or getattr(mlp[-1], 'out_features', None) != output_dim
            ):
                layers: List[nn.Module] = []
                for i in range(len(dims) - 2):
                    lin = nn.Linear(dims[i], dims[i+1])
                    weight_init_fn(lin.weight)
                    layers.append(lin)
                    layers.append(nonlin_fn())
                lin = nn.Linear(dims[-2], dims[-1])
                weight_init_fn(lin.weight)
                layers.append(lin)
                mlp = nn.Sequential(*layers).to(self.device)
                setattr(self, attr_name, mlp)
            return getattr(self, attr_name)
        else:
            dims = [input_dim] + list(hidden_dims) + [output_dim]
            layers: List[nn.Module] = []
            for i in range(len(dims) - 2):
                lin = nn.Linear(dims[i], dims[i+1])
                weight_init_fn(lin.weight)
                layers.append(lin)
                layers.append(nonlin_fn())
            lin = nn.Linear(dims[-2], dims[-1])
            weight_init_fn(lin.weight)
            layers.append(lin)
            mlp = nn.Sequential(*layers).to(self.device)
            return mlp

    def _lazy_build_gate_basis(
        self,
        basis_attr: str,
        rank: int,
        out_dim: int,
    ) -> nn.Parameter:
        existing = getattr(self, basis_attr)
        if (existing is None) or (existing.shape != (rank, out_dim)):
            new_basis = nn.Parameter(torch.empty(rank, out_dim)).to(self.device)
            nn.init.kaiming_uniform_(new_basis)
            setattr(self, basis_attr, new_basis)  # this should register the parameter
            # self.register_parameter(basis_attr, new_basis)
            return new_basis
        return existing

    # ------------------------------------------------------------------
    # BN helpers (scalar per wavelet, vector norm per wavelet)
    # ------------------------------------------------------------------
    def _apply_scalar_bn(
        self,
        h: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Apply independent batch normalization to each (channel, wavelet) 
        combination.
        
        Flattens (N, C, W) -> (N, C*W), applies BatchNorm1d(C*W), then 
        reshapes back. Each scalar channel within each wavelet band is normalized independently.
        """
        N, C, W = h.shape
        while len(self._scalar_bn_layers) <= layer_idx:
            self._scalar_bn_layers.append(None)
        
        # Need single BatchNorm1d layer with C*W features
        expected_features = C * W
        if (self._scalar_bn_layers[layer_idx] is None) \
        or (self._scalar_bn_layers[layer_idx].num_features != expected_features):
            self._scalar_bn_layers[layer_idx] = nn.BatchNorm1d(expected_features).to(h.device)
        
        # Flatten (N, C, W) -> (N, C*W), apply BN, reshape back
        # This treats each channel-wavelet combination independently
        h_flat = h.view(N, C * W)
        h_bn_flat = self._scalar_bn_layers[layer_idx](h_flat)
        return h_bn_flat.view(N, C, W)

    # ------------------------------------------------------------------
    # Vector rescaling helper – keep direction, update magnitude
    # ------------------------------------------------------------------
    def _rescale_vectors_to_standardized_norm(
        self,
        v: torch.Tensor,   # (N, d, W)
        v_norms: torch.Tensor,  # (N, W)
        v_norms_scaled: torch.Tensor,  # (N, W) already standardized to N(0,1)
    ) -> torch.Tensor:
        """
        Return vectors whose magnitudes match *v_norms_scaled* per wavelet band.

        Each vector v is transformed to
            v' = v * (v_norms_scaled / ||v||)
        where division by zero is avoided with self.eps.
        """
        scale = (v_norms_scaled / (v_norms + self.eps)).unsqueeze(1)  # (N,1,W)
        return v * scale
     
    # ------------------------------------------------------------------
    # Global pooling helper
    # ------------------------------------------------------------------
    @staticmethod
    def _global_pool(
        node_feat: torch.Tensor,
        batch: torch.Tensor,
        stats: Tuple[str, ...],
    ) -> torch.Tensor:
        """Efficiently compute and concatenate requested pooling statistics.

        Supported statistics:
          1  'sum'  - per-graph sum
          2  'mean' - per-graph mean (computed via sum / count)
          3  'max'  - per-graph max (falls back to torch_geometric impl.)
          4  'var'  - per-graph *population* variance  E[x^2] - (E[x])^2

        Only the minimum number of scatter passes are performed: one for
        the sum (always needed for mean / var), one for node counts, and
        optionally one for the element-wise sum of squares when variance
        is requested.
        """

        need_sum = any(s in stats for s in ("sum", "mean", "var"))
        need_mean_or_var = any(s in stats for s in ("mean", "var"))

        pieces: List[torch.Tensor] = []

        # Sum pooling ---------------------------------------------------
        if need_sum:
            sum_pool = global_add_pool(node_feat, batch)  # (B,F)
        if "sum" in stats:
            pieces.append(sum_pool)

        # Node counts ----------------------------------------------------
        if need_mean_or_var:
            ones = node_feat.new_ones((node_feat.size(0), 1))
            n_nodes = global_add_pool(ones, batch)  # (B,1)

        # Mean pooling ---------------------------------------------------
        if "mean" in stats:
            mean_pool = sum_pool / n_nodes.clamp_min(1.0)
            pieces.append(mean_pool)

        # Variance pooling ----------------------------------------------
        if "var" in stats:
            sum_sq = global_add_pool(node_feat ** 2, batch)  # (B,F)
            mean_pool = sum_pool / n_nodes.clamp_min(1.0)
            var_pool = sum_sq / n_nodes.clamp_min(1.0) - mean_pool ** 2
            pieces.append(var_pool)

        # Max pooling (no cheap alternative) ----------------------------
        if "max" in stats:
            pieces.append(global_max_pool(node_feat, batch))

        if not pieces:
            raise ValueError("`pool_stats` must contain at least one supported statistic.")

        return torch.cat(pieces, dim=-1)

    # ------------------------------------------------------------------
    # Scattering helper
    # ------------------------------------------------------------------
    def _scatter_track(
        self,
        x_init: torch.Tensor,  # (N, d_scalar|d_vector)
        diffusion_op: torch.Tensor,
        *,
        is_vector: bool = False,
        vector_dim: Optional[int] = None,
        custom_scales: Optional[torch.Tensor] = None,
        verbosity: int = 0,
    ) -> torch.Tensor:
        if verbosity > 0:
            print(f"\n_scatter_track:is_vector: {is_vector}")
        N = x_init.size(0)
        outs: List[torch.Tensor] = [x_init.unsqueeze(-1)]

        if is_vector:
            # Flatten vector features for multiplication by Q_sparse
            x_prev = x_init.view(N * self.d_vector, 1)  # (N*d_vector, 1)
        else:
            x_prev = x_init  # (N, d_scalar)

        for order in self.scat_orders:
            if verbosity > 0:
                print(f"  order {order}, x_prev.shape: {x_prev.shape}")

            W = get_Batch_Wjxs(
                x=x_prev,
                P_sparse=diffusion_op,
                J=self.wavelet_J,
                include_lowpass=self.include_lowpass_wavelet,
                filter_stack_dim=-1,
                vector_dim=vector_dim if is_vector else None,
                diffusion_scales=custom_scales,
            )
            if verbosity > 0:
                print(f"    W.shape (before subsetting): {W.shape}")
            
            # 2nd-order subsetting (only keep lower-pass wavelets applied to higher-pass 1st-order bands)
            if order == 2:
                num_w = W.shape[-1]
                if self.wavelet_J_prime is None:
                    mask = torch.triu(torch.ones(num_w, num_w, dtype=torch.bool, device=W.device), 1)
                    if is_vector:
                        W = W.view(N, vector_dim, num_w, num_w)[:, :, mask]
                    else:
                        orig_ch = W.shape[1] // num_w
                        W = W.view(N, orig_ch, num_w, num_w)[:, :, mask]
                else:
                    W = W[..., -self.wavelet_J_prime :]
            elif (order == 1) and is_vector:
                # Unflatten filtered vector feature before adding to outs
                W = W.view(N, vector_dim, -1)  # (N, d_vector, W)

            if verbosity > 0:
                print(f"    W.shape (after subsetting): {W.shape}")
            outs.append(W)
            x_prev = W.reshape(N * vector_dim, -1) if is_vector else W.reshape(N, -1)
            out = torch.cat(outs, dim=-1)  # concat over wavelet dim -> (N, d_scalar|d_vector, W)
            if verbosity > 0:
                print(f"    out.shape: {out.shape}")
        return out
    

    # DEPRECATED: use _message_passing_layer instead
    '''
    def _gate_vector_track(
        self,
        batch: Batch,
        v_src: torch.Tensor,
        vector_gate_in: torch.Tensor,
    ) -> torch.Tensor:
        
        # Extract batch indices
        edge_index = batch.edge_index
        # src_nodes_idx = edge_index[0]  # (E,)
        tgt_nodes_idx = edge_index[1]  # (E,)
        
        # Build vector gate MLP and basis transform matrix
        vector_mlp = self._build_or_get_mlp(
            input_dim=vector_gate_in.size(-1), 
            hidden_dims=self.vector_gate_mlp_hidden_dims,
            nonlin_fn=self._readout_act_cls,
            output_dim=self.vector_gate_rank,
        )

        B_vector = self._lazy_build_gate_basis(
            "vector_gate_basis", 
            self.vector_gate_rank, 
            v_src.shape[-1]
        )

        # Gate (edge-feature) MLPs forward passes and activations
        # Softplus in vector track ensures non-negative gating weights (smooth
        # approximation to ReLU). Sigmoid in scalar track ensures gating weights
        # are between 0 and 1, similar to attention weights.
        gate_vector = F.softplus(
            torch.matmul(
                vector_mlp(vector_gate_in), 
                B_vector,
            ),
        )  # (E,W)

        # Gated message passing: apply gating weights and aggregate to 
        # 'target' nodes
        v_msg = gate_vector.unsqueeze(1) * v_src  # (E, 1, W) * (E, d_vector, W) -> (E, d_vector, W)
        v_agg = scatter_add(
            src=v_msg, 
            index=tgt_nodes_idx,  # (E,)
            dim=0, 
            dim_size=batch.num_nodes,  # (N,)
        )  # (N, d_vector, W)

        return v_agg
    '''

    # ------------------------------------------------------------------
    # Optional methods to run at start of first epoch
    # ------------------------------------------------------------------
    def run_epoch_zero_methods(self, batch: Batch) -> None:
        """
        Run any methods that need to be run at the start of the first epoch
        (overrides BaseModule.run_epoch_zero_methods, always called in first 
        epoch of train.train_model).
        """
        return None

    # ------------------------------------------------------------------
    # Gating functions
    # ------------------------------------------------------------------
    def _apply_scalar_gating(
        self,
        h_src_hidden: torch.Tensor,  # (E, d_scalar_hidden)
        gate_context: torch.Tensor,  # (E, 2*d_scalar_hidden + edge_feat_dim)
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Apply low-rank scalar gating to condensed scalar features.
        The gating weights are produced from *gate_context* (which can
        incorporate information from source and target nodes as well as
        edge features) and then applied to *h_src_hidden*.
        
        Args:
            h_src_hidden: Condensed scalar features from the *source* node.
            gate_context: Feature context used to generate gating weights.
            layer_idx: Layer index for per-layer MLPs.
        Returns:
            Gated scalar features with the same dimensionality as
            *h_src_hidden*.
        """
        # Build or get per-layer scalar gate MLP (input is gate_context)
        while len(self.scalar_gate_mlps) <= layer_idx:
            self.scalar_gate_mlps.append(None)
        if self.scalar_gate_mlps[layer_idx] is None \
        or getattr(self.scalar_gate_mlps[layer_idx][0], 'in_features', None) != gate_context.size(-1) \
        or getattr(self.scalar_gate_mlps[layer_idx][-1], 'out_features', None) != self.scalar_gate_rank:
            scalar_gate_mlp = self._build_or_get_mlp(
                input_dim=gate_context.size(-1),
                hidden_dims=self.scalar_gate_mlp_hidden_dims,
                nonlin_fn=self.scalar_gate_nonlin_fn,
                output_dim=self.scalar_gate_rank,
            )
            self.scalar_gate_mlps[layer_idx] = scalar_gate_mlp

        # Build gate basis matrix to project low-rank outputs to feature dim
        B_scalar = self._lazy_build_gate_basis(
            "scalar_gate_basis",
            self.scalar_gate_rank,
            h_src_hidden.shape[-1],
        )

        # Sigmoid gating weights in [0,1]
        gate_scalar = torch.sigmoid(
            torch.matmul(
                self.scalar_gate_mlps[layer_idx](gate_context),
                B_scalar,
            )
        )  # (E, d_scalar_hidden)

        return h_src_hidden * gate_scalar  # Element-wise gating

    def _apply_vector_gating(
        self,
        v_src: torch.Tensor,  # (E, d_vector, W)
        v_norm_src: torch.Tensor,  # (E, W)
        v_norm_tgt: torch.Tensor,  # (E, W)
        cos_src: torch.Tensor,  # (E, W)
        cos_tgt: torch.Tensor,  # (E, W)
        shared_edge_feat: torch.Tensor,  # (E, bond_emb_dim + num_rbf)
        h_msg: torch.Tensor,  # (E, d_scalar_hidden)
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Apply low-rank vector gating to vector features.
        
        Args:
            v_src: Source vector features
            v_norm_src: Vector norms from source nodes
            v_norm_tgt: Vector norms from target nodes
            cos_src: Cosine similarities from source nodes
            cos_tgt: Cosine similarities from target nodes
            shared_edge_feat: Additional edge features
            h_msg: Gated scalar message features
            layer_idx: Layer index for per-layer MLPs
            
        Returns:
            Gated vector features
        """
        # Build gate input by concatenating all relevant features (source & target)
        vector_gate_in = torch.cat(
            [v_norm_src, v_norm_tgt, cos_src, cos_tgt, shared_edge_feat, h_msg],
            dim=1,
        )  # (E, 4*W + bond_emb_dim + num_rbf + d_scalar_hidden)
        
        # Build or get per-layer vector gate MLP
        while len(self.vector_gate_mlps) <= layer_idx:
            self.vector_gate_mlps.append(None)
        if self.vector_gate_mlps[layer_idx] is None \
        or getattr(self.vector_gate_mlps[layer_idx][0], 'in_features', None) != vector_gate_in.size(-1) \
        or getattr(self.vector_gate_mlps[layer_idx][-1], 'out_features', None) != self.vector_gate_rank:
            vector_mlp = self._build_or_get_mlp(
                input_dim=vector_gate_in.size(-1),
                hidden_dims=self.vector_gate_mlp_hidden_dims,
                nonlin_fn=self.vector_gate_mlp_nonlin_fn,
                output_dim=self.vector_gate_rank,
            )
            self.vector_gate_mlps[layer_idx] = vector_mlp
        
        # Build gate basis matrix
        B_vector = self._lazy_build_gate_basis(
            "vector_gate_basis",
            self.vector_gate_rank,
            v_src.shape[-1],
        )
        
        # Apply softplus gating (ensures non-negative gating weights)
        gate_vector = F.softplus(
            torch.matmul(
                self.vector_gate_mlps[layer_idx](vector_gate_in),
                B_vector,
            ),
        )  # (E, W)
        
        # Apply gating and return
        return gate_vector.unsqueeze(1) * v_src  # (E, 1, W) * (E, d_vector, W) -> (E, d_vector, W)

    # ------------------------------------------------------------------
    # Message passing method
    # ------------------------------------------------------------------
    def _message_passing_layer(
        self,
        h_in: torch.Tensor,  # (N, d_scalar, W) or (N, d_scalar_hidden)
        v_in: torch.Tensor,  # (N, d_vector, W)
        shared_edge_feat: torch.Tensor,  # (E, bond_emb_dim + num_rbf)
        batch: Batch,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Extract batch indices
        edge_index = batch.edge_index
        src_nodes_idx = edge_index[0]  # (E,)
        tgt_nodes_idx = edge_index[1]  # (E,)

        # --- Scalar track -------------------------------------------------
        # Gather scalar features from BOTH source and target nodes
        h_src = h_in[src_nodes_idx]  # (E, d_scalar, W) or (E, d_scalar_hidden)
        h_tgt = h_in[tgt_nodes_idx]  # (E, d_scalar, W) or (E, d_scalar_hidden)

        if h_src.dim() == 3:
            h_flat_src = h_src.reshape(edge_index.size(1), -1)  # (E, d_scalar*W)
            h_flat_tgt = h_tgt.reshape(edge_index.size(1), -1)  # (E, d_scalar*W)
        else:
            h_flat_src = h_src  # (E, d_scalar_hidden)
            h_flat_tgt = h_tgt  # (E, d_scalar_hidden)

        # Condense scalar features for source and target separately (shared MLP)
        scalar_mlp_src_in = torch.cat([h_flat_src, shared_edge_feat], dim=1)
        scalar_mlp_tgt_in = torch.cat([h_flat_tgt, shared_edge_feat], dim=1)
        while len(self.scalar_condense_mlps) <= layer_idx:
            self.scalar_condense_mlps.append(None)
        if self.scalar_condense_mlps[layer_idx] is None \
        or getattr(self.scalar_condense_mlps[layer_idx][0], 'in_features', None) != scalar_mlp_src_in.size(-1) \
        or getattr(self.scalar_condense_mlps[layer_idx][-1], 'out_features', None) != self.d_scalar_hidden:
            scalar_mlp = self._build_or_get_mlp(
                input_dim=scalar_mlp_src_in.size(-1),
                hidden_dims=self.scalar_condense_hidden_dims,
                nonlin_fn=self._readout_act_cls,
                output_dim=self.d_scalar_hidden,
            )
            self.scalar_condense_mlps[layer_idx] = scalar_mlp
        h_hidden_src = self.scalar_condense_mlps[layer_idx](scalar_mlp_src_in)  # (E, d_scalar_hidden)
        h_hidden_tgt = self.scalar_condense_mlps[layer_idx](scalar_mlp_tgt_in)  # (E, d_scalar_hidden)

        # Scalar gating uses both source and target condensed features
        scalar_gate_in = torch.cat([h_hidden_src, h_hidden_tgt, shared_edge_feat], dim=1)
        h_msg = self._apply_scalar_gating(h_hidden_src, scalar_gate_in, layer_idx)  # (E, d_scalar_hidden)

        # Aggregate scalar features to 'target' nodes
        h_agg = scatter_add(
            src=h_msg,
            index=tgt_nodes_idx,
            dim=0,
            dim_size=batch.num_nodes,
        )  # (N, d_scalar_hidden)

        # --- Vector track ---
        # Get vector features from 'source'/neighbor nodes ('v_src') as edge 
        # features to modulate via gating
        v_src = v_in[src_nodes_idx]  # (E, d_vector, W)

        # NOTE: for the first layer, v_norms_scaled and cos_theta are computed from v_scatt; 
        # for subsequent layers, recompute
        v_norms = torch.norm(v_in, dim=1)  # (N, W)
        if not self.ablate_vector_wavelet_batch_norm \
        and self._vec_stats is not None:
            if 'mean' in self._vec_stats and 'std' in self._vec_stats:
                mean = self._vec_stats['mean'].to(v_norms.device)
                std  = self._vec_stats['std'].to(v_norms.device)
                v_norms_scaled = (v_norms - mean) / std
                v_in = self._rescale_vectors_to_standardized_norm(
                    v_in,
                    v_norms,
                    v_norms_scaled,
                )
                # Recompute norms after rescaling
                v_norms = torch.norm(v_in, dim=1)  # (N, W)
        v_norm_src = v_norms[src_nodes_idx]  # (E, W)
        v_norm_tgt = v_norms[tgt_nodes_idx]  # (E, W)

        # Compute cosine similarities between original and scattered vectors
        # (no standardization here; already on [-1, 1])
        v_in_unit = v_in / (v_norms.unsqueeze(1) + self.eps)  # (N, d_vector, W)
        v0 = getattr(batch, self.vector_feature_key)
        v0_unit = v0 / (v0.norm(dim=1, keepdim=True) + self.eps)  # (N, d_vector)
        cos_theta = (v0_unit.unsqueeze(-1) * v_in_unit).sum(dim=1)  # (N, W)
        cos_src = cos_theta[src_nodes_idx]  # (E, W)
        cos_tgt = cos_theta[tgt_nodes_idx]  # (E, W)

        # Apply vector gating
        v_msg = self._apply_vector_gating(
            v_src,
            v_norm_src,
            v_norm_tgt,
            cos_src,
            cos_tgt,
            shared_edge_feat,
            h_msg,
            layer_idx,
        )  # (E, d_vector, W)

        # Aggregate to 'target' nodes
        v_agg = scatter_add(
            src=v_msg,
            index=tgt_nodes_idx,
            dim=0,
            dim_size=batch.num_nodes,
        )  # (N, d_vector, W)
        return h_agg, v_agg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Batch | Data,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(batch, Data):
            batch = Batch.from_data_list([batch])
        
        # ------------------------------------------------------------------
        # Scattering (on node embeddings + Dirac nodes)
        # - We embed then scatter to give the wavelets rich, continuous vector 
        #   data instead of one-hot, discrete data.
        # - The learned embedding will be optimized for scattering and other
        #   model transforms since it is end-to-end trained.
        # - The embedding is important for tasks where node features are
        #   predictive; the Dirac nodes are important for tasks where overall
        #   geometry is important.
        # ------------------------------------------------------------------
        
        # --- Scalar track ---
        h_raw = getattr(batch, self.scalar_feature_key)
        P_sparse = batch[self.scalar_diffusion_op_key]

        # If scalar embedding is ablated, we use the raw scalar features
        h0 = self.scalar_embedding(h_raw) \
            if not self.ablate_scalar_embedding \
            else h_raw
        
        if self.use_dirac_nodes and hasattr(batch, "dirac_nodes"):
            h0 = torch.cat([h0, batch.dirac_nodes], dim=1)
        h_scatt = self._scatter_track(
            x_init=h0,
            diffusion_op=P_sparse,
            custom_scales=self.custom_scalar_scales,
        )  # (N, d_scalar, W)
        if not self.ablate_scalar_wavelet_batch_norm:
            h_scatt = self._apply_scalar_bn(h_scatt, layer_idx=0)  # (N, d_scalar, W)

        # --- Vector track ---
        v0 = getattr(batch, self.vector_feature_key)
        Q_sparse = batch[self.vector_diffusion_op_key]
        v_scatt = self._scatter_track(
            x_init=v0,
            diffusion_op=Q_sparse,
            is_vector=True,
            vector_dim=self.d_vector,
            custom_scales=self.custom_vector_scales,
        )  # (N, d_vector, W)
        if not self.ablate_vector_wavelet_batch_norm \
        and self._vec_stats is not None:
            if 'mean' in self._vec_stats and 'std' in self._vec_stats:
                v_norms = torch.norm(v_scatt, dim=1)
                mean = self._vec_stats['mean'].to(v_norms.device)
                std  = self._vec_stats['std'].to(v_norms.device)
                v_norms_scaled = (v_norms - mean) / std
                v_scatt = self._rescale_vectors_to_standardized_norm(
                    v_scatt,
                    v_norms,
                    v_norms_scaled,
                )

        # ------------------------------------------------------------------
        # Additional edge features
        # ------------------------------------------------------------------
        # Additional (scalar) edge features to include in msg passing (on top
        # of the neighbor node features):
        # - radial Bessel functions
        # - bond type embedding
        edge_rbf = getattr(batch, self.edge_rbf_key, None)
        if edge_rbf is None:
            edge_rbf = torch.empty(batch.edge_index.size(1), 0, device=self.device)
        edge_type = getattr(batch, self.bond_type_key, None)
        bond_emb = self.bond_embedding(edge_type) \
            if edge_type is not None \
            else torch.empty(batch.edge_index.size(1), 0, device=self.device)
        shared_edge_feat = torch.cat([edge_rbf, bond_emb], dim=1)  # (E, bond_emb_dim + num_rbf)

        # Check if graph is undirected (avoid collision with PyG's BaseData `is_undirected`)
        batch.has_two_way_edges = check_if_undirected(batch.edge_index)
        # print(f"[DEBUG] Batch graph has two-way edges:\n\tBefore: {batch.has_two_way_edges}")

        # Convert to undirected graph (duplicate edges and associated features)
        if not batch.has_two_way_edges:
            # Store edge features temporarily for to_undirected
            batch.edge_attr = shared_edge_feat  # (E, bond_emb_dim + num_rbf)
            edge_index_ud, edge_attr_ud = to_undirected(
                batch.edge_index,
                batch.edge_attr,
                num_nodes=batch.num_nodes,
            )

            # Update batch and local variables with undirected data
            batch.edge_index = edge_index_ud  # (2E, 2)
            # print(f"\tAfter:  {check_if_undirected(batch.edge_index)}")
            shared_edge_feat = edge_attr_ud  # (2E, bond_emb_dim + num_rbf)
            batch.edge_attr = edge_attr_ud
            batch.has_two_way_edges = True
            # edge_index = edge_index_ud

        # ------------------------------------------------------------------
        # Message passing layers
        # ------------------------------------------------------------------
        h, v = h_scatt, v_scatt
        for i in range(self.num_msg_pass_layers):
            h_new, v_new = self._message_passing_layer(
                h, v, shared_edge_feat, batch, i,
            )
            if self.use_residual_connections and i > 0:
                # Only add residual if shapes match
                if h.shape == h_new.shape:
                    h = h + h_new
                else:
                    h = h_new
                if v.shape == v_new.shape:
                    v = v + v_new
                else:
                    v = v_new
            else:
                h, v = h_new, v_new

        # ------------------------------------------------------------------
        # Pooling
        # ------------------------------------------------------------------
        # --- Scalar features ---
        h_pooled = self._global_pool(
            h,
            batch.batch,
            self.pool_stats,
        )

        # --- Vector features ---
        # Sum gated vectors per graph so we preserve the batch dimension.
        # `batch.batch` maps each node to its graph index (0..B-1).
        # Summing gated vectors, by wavelet band, (a) keeps both magnitude and direction 
        # of the aggregate vector, and (b) allows constructive and destructive 
        # interference of directional signals (get 'net' alignment/orientation)
        # The index tensor must match ``v_agg`` shape for ``torch_scatter.scatter_add``.
        idx = batch.batch.to(v.device).view(-1, 1, 1).expand_as(v)
        gated_v_sums = scatter_add(
            src=v,
            index=idx,
            dim=0,
        )  # (B, d_vector, W)
        gated_v_sums_norms = torch.norm(gated_v_sums, dim=1)  # (B, W)

        # Compute cosine similarities between summed vectors of wavelet bands, per graph 
        # (only keep unique pairs / upper-triangular part)
        gated_v_sums_units = gated_v_sums / (gated_v_sums_norms.unsqueeze(1) + self.eps)
        cos_matrix = torch.matmul(
            gated_v_sums_units.transpose(1, 2),
            gated_v_sums_units,
        )
        W_tot = cos_matrix.size(-1)
        triu_i, triu_j = torch.triu_indices(
            W_tot, W_tot, offset=1, device=self.device,
        )
        gated_v_sums_cosines = cos_matrix[:, triu_i, triu_j]

        graph_feat = torch.cat(
            [h_pooled, gated_v_sums_norms, gated_v_sums_cosines],
            dim=-1,
        )
        # ------------------------------------------------------------------
        # Readout
        # ------------------------------------------------------------------
        mlp = self._build_or_get_mlp(
            input_dim=graph_feat.size(-1),
            hidden_dims=self._readout_hidden_dims,
            nonlin_fn=self._readout_act_cls,
            output_dim=self._pred_output_dim,
        )
        preds = mlp(graph_feat).squeeze(-1)
        return {"preds": preds} 
    
