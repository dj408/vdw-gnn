"""
ESCGNN variant for macaque reaching with supervised contrastive learning.

Model class: ESCGNN_macaque_supcon

This model learns trajectory embeddings using supervised contrastive loss,
where trajectories with the same condition label are pulled together in
embedding space while different conditions are pushed apart.

Architecture:
1. Vector scattering: multi-scale wavelet features per time point
2. VectorBatchNorm: normalize wavelet channels to mean norm=1
3. TemporalEncoder: 1D convolutions to detect temporal motifs
4. ProjectionMLP: project to embedding space

Two modes:
- 'pretrain': Learn embeddings with SupConLoss
- 'finetune': Freeze encoder, add classification head
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal, Tuple
import os

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from models.escgnn_modular import ESCGNNModular
from models.custom_loss_fns import SupConLoss, SupConCrossEntropyLoss
from models.nn_utilities import ProjectionMLP
from geo_scat import VectorBatchNorm
from pyg_utilities import infer_device_from_batch


class TemporalEncoder(nn.Module):
    """
    1D convolutional encoder to detect temporal motifs across trajectory time points.
    
    Different trajectories may exhibit similar neural patterns but phase-shifted
    in time. This encoder uses 1D convolutions to learn translation-invariant
    temporal features.
    
    Architecture:
        Conv1d → Activation → Conv1d → Activation → AdaptiveAvgPool1d → Flatten
    
    Input shape: (batch, in_channels, time_steps)
    Output shape: (batch, out_dim)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = [128, 128],
        kernel_sizes: List[int] = [3, 5],
        paddings: Optional[List[int]] = None,
        activation: type[nn.Module] = nn.ReLU,
        out_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize TemporalEncoder.
        
        Args:
            in_channels: Number of input channels (d*W_total)
            hidden_channels: List of hidden channel dimensions for Conv1d layers
            kernel_sizes: List of kernel sizes for each Conv1d layer
            paddings: List of padding values. If None, auto-computed as kernel_size//2
            activation: Activation function class (e.g., nn.ReLU, nn.SiLU)
            out_dim: Output dimension after pooling. If None, uses last hidden_channels
        """
        super().__init__()
        
        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError(
                f"Length of hidden_channels ({len(hidden_channels)}) must match "
                f"kernel_sizes ({len(kernel_sizes)})"
            )
        
        if paddings is None:
            # Auto-compute padding to preserve time dimension
            paddings = [k // 2 for k in kernel_sizes]
        
        if len(paddings) != len(kernel_sizes):
            raise ValueError(
                f"Length of paddings ({len(paddings)}) must match "
                f"kernel_sizes ({len(kernel_sizes)})"
            )
        
        self.in_channels = in_channels
        self.out_dim = out_dim if out_dim is not None else hidden_channels[-1]
        
        # Build convolutional layers
        layers: List[nn.Module] = []
        prev_channels = in_channels
        
        for i, (hidden_ch, kernel, pad) in enumerate(
            zip(hidden_channels, kernel_sizes, paddings)
        ):
            layers.append(
                nn.Conv1d(
                    prev_channels,
                    hidden_ch,
                    kernel_size=kernel,
                    padding=pad,
                )
            )
            layers.append(activation())
            prev_channels = hidden_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to handle variable-length sequences
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize convolutional layer weights using Kaiming initialization.
        
        Uses Kaiming/He initialization which is well-suited for ReLU-like
        activations. This helps prevent vanishing/exploding gradients by
        maintaining variance across layers.
        """
        for module in self.conv_layers.modules():
            if isinstance(module, nn.Conv1d):
                # Kaiming normal initialization for conv layers
                # mode='fan_in' preserves magnitude of variance of weights in forward pass
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    # Initialize biases to small constant
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode temporal sequences.
        
        Args:
            x: Input tensor of shape (batch, in_channels, time_steps)
            
        Returns:
            Encoded features of shape (batch, out_dim)
        """
        # Apply convolutions: (batch, in_channels, time) → (batch, hidden_ch, time)
        x = self.conv_layers(x)
        
        # Pool over time: (batch, hidden_ch, time) → (batch, hidden_ch, 1)
        x = self.pool(x)
        
        # Flatten: (batch, hidden_ch, 1) → (batch, hidden_ch)
        x = x.squeeze(-1)
        
        return x
    
    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_dim={self.out_dim}'


class ESCGNN_macaque_supcon(ESCGNNModular):
    """
    ESCGNN variant for supervised contrastive learning on macaque trajectories.
    
    This model learns trajectory embeddings where trajectories with the same
    condition label cluster together in embedding space. Can be used in two modes:
    
    1. 'pretrain': Learn embeddings with supervised contrastive loss
    2. 'finetune': Freeze encoder weights and train classification head
    
    Architecture flow:
        Vector scattering → VectorBatchNorm → TemporalEncoder → ProjectionMLP
        
    For 'finetune' mode, adds a classification head on top of frozen embeddings.
    """
    
    # Default configurations
    VECTOR_BN_DEFAULTS = {
        'ablate': False,
        'momentum': 0.1,
        'eps': 1e-6,
        'track_running_stats': True,
    }
    
    TEMPORAL_ENCODER_DEFAULTS = {
        'hidden_channels': [128, 128],
        'kernel_sizes': [3, 5],
        'paddings': None,  # Auto-computed
        'activation': nn.ReLU,
        'out_dim': 128,
    }
    
    PROJECTION_MLP_DEFAULTS = {
        'hidden_dim': 256,
        'embedding_dim': 128,
        'activation': nn.ReLU,
        'use_batch_norm': True,
    }
    
    CLASSIFICATION_HEAD_DEFAULTS = {
        'hidden_dims': [64],
        'activation': nn.ReLU,
        'dropout_p': 0.1,
    }
    
    def __init__(
        self,
        *,
        base_module_kwargs: Dict[str, Any],
        vector_track_kwargs: Dict[str, Any],
        vector_bn_kwargs: Optional[Dict[str, Any]] = None,
        temporal_encoder_kwargs: Optional[Dict[str, Any]] = None,
        projection_mlp_kwargs: Optional[Dict[str, Any]] = None,
        classification_head_kwargs: Optional[Dict[str, Any]] = None,
        mode: Literal['pretrain', 'finetune', 'train'] = 'pretrain',
        freeze_encoder: bool = False,
        num_classes: int = 7,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize ESCGNN_macaque_supcon model.
        
        Args:
            base_module_kwargs: Passed to BaseModule (task, loss, metrics, device, etc.)
            vector_track_kwargs: Keys for vector scattering track:
                - 'feature_key': str (e.g., 'spike_data')
                - 'diffusion_op_key': str (e.g., 'Q')
                - 'vector_dim': int (K channels in spike_data)
                - 'diffusion_kwargs': dict containing wavelet scales settings
                - 'num_layers': int (1 or 2 for 0th/1st/2nd order)
            vector_bn_kwargs: VectorBatchNorm configuration
            temporal_encoder_kwargs: TemporalEncoder configuration
            projection_mlp_kwargs: ProjectionMLP configuration
            classification_head_kwargs: Classification head configuration (for finetune/train modes)
            mode: 'pretrain' (SupCon), 'finetune' (CE on frozen encoder), or 'train' (joint SupCon+CE)
            freeze_encoder: If True, freeze all encoder weights (for finetune/train modes)
            num_classes: Number of condition classes (default: 7)
            temperature: Temperature for SupConLoss (default: 0.1)
        """
        # Initialize parent with ablated scalar track
        super().__init__(
            base_module_kwargs=base_module_kwargs,
            ablate_scalar_track=True,  # Only use vector track
            ablate_vector_track=False,
            scalar_track_kwargs={},
            vector_track_kwargs=vector_track_kwargs,
            mixing_kwargs=None,  # No within-track mixing
            neighbor_kwargs=None,
            head_kwargs=None,
            readout_kwargs=None,
        )
        
        self.mode = mode
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.target_attr = base_module_kwargs.get('target_name', None)
        
        # Merge kwargs with defaults
        self.vector_bn_kwargs = {
            **self.VECTOR_BN_DEFAULTS,
            **(vector_bn_kwargs or {})
        }
        self.temporal_encoder_kwargs = {
            **self.TEMPORAL_ENCODER_DEFAULTS,
            **(temporal_encoder_kwargs or {})
        }
        self.projection_mlp_kwargs = {
            **self.PROJECTION_MLP_DEFAULTS,
            **(projection_mlp_kwargs or {})
        }
        self.classification_head_kwargs = {
            **self.CLASSIFICATION_HEAD_DEFAULTS,
            **(classification_head_kwargs or {})
        }

        # Allow ablating the TemporalEncoder when configured.
        # If the YAML/model_config sets temporal_hidden_channels to null or [],
        # we skip creating/using the TemporalEncoder and instead perform a simple
        # temporal average pooling over the scattering features.
        hidden_channels = self.temporal_encoder_kwargs.get('hidden_channels')
        self.ablate_temporal_encoder = (
            hidden_channels is None or (isinstance(hidden_channels, (list, tuple)) and len(hidden_channels) == 0)
        )
        
        # Set up loss function based on mode
        if mode == 'pretrain':
            self.loss_fn = SupConLoss(temperature=temperature)
        elif mode == 'finetune':
            # Use cross-entropy for (multiclass) classification
            self.loss_fn = nn.CrossEntropyLoss()
        elif mode == 'train':
            self.loss_fn = SupConCrossEntropyLoss(temperature=temperature)
        else:
            raise ValueError(f"Unsupported ESCGNN_macaque_supcon mode: {mode}")

        if (self.mode == 'train') and not isinstance(self.target_attr, str):
            raise ValueError(
                "ESCGNN_macaque_supcon (train mode) requires 'target_name' "
                "within base_module_kwargs to locate batch labels."
            )
        
        # Layers (lazy initialization)
        self.ablate_vector_bn = self.vector_bn_kwargs.get('ablate', False)
        self.vector_bn: Optional[VectorBatchNorm] = None
        self.temporal_encoder: Optional[TemporalEncoder] = None
        self.projection_mlp: Optional[ProjectionMLP] = None
        self.classification_head: Optional[nn.Module] = None
        
        # Signal lazy parameter initialization
        self.has_lazy_parameter_initialization = True
    
    def _lazy_init_encoder_layers(
        self,
        *,
        num_wavelets: int,
        d: int,
        nodes_per_graph: int,
        device: torch.device,
    ) -> None:
        """
        Lazily initialize encoder layers based on scattering output dimensions.
        
        Args:
            num_wavelets: Total number of wavelet channels (W_total)
            d: Vector dimensionality (spike data channels)
            device: Device to place layers on
        """
        # VectorBatchNorm: normalizes each wavelet channel
        if (not self.ablate_vector_bn) and (self.vector_bn is None):
            self.vector_bn = VectorBatchNorm(
                num_wavelets=num_wavelets,
                **self.vector_bn_kwargs
            ).to(device)

        # TemporalEncoder: Conv1d over time (optional, can be ablated)
        # Input channels = d * W_total (flattened wavelet channels per vector dim)
        if (not self.ablate_temporal_encoder) and (self.temporal_encoder is None):
            in_channels = d * num_wavelets
            self.temporal_encoder = TemporalEncoder(
                in_channels=in_channels,
                **self.temporal_encoder_kwargs
            ).to(device)

        # ProjectionMLP: maps temporal encoding (or pooled features) to embedding space
        if self.projection_mlp is None:
            if self.ablate_temporal_encoder:
                # Without TemporalEncoder, we flatten all scattering coefficients
                # across time and channels, resulting in a feature dim of
                # (nodes_per_graph * d * num_wavelets) per trajectory.
                temporal_out_dim = nodes_per_graph * d * num_wavelets
            else:
                temporal_out_dim = self.temporal_encoder_kwargs.get(
                    'out_dim',
                    self.temporal_encoder_kwargs.get('hidden_channels', [128])[-1]
                )
            self.projection_mlp = ProjectionMLP(
                in_dim=temporal_out_dim,
                **self.projection_mlp_kwargs
            ).to(device)
        
        # Classification head (finetune/train modes)
        if (self.mode in ('finetune', 'train')) and (self.classification_head is None):
            embedding_dim = self.projection_mlp_kwargs['embedding_dim']
            self._lazy_init_classification_head(
                in_dim=embedding_dim,
                device=device,
            )
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            self._freeze_encoder_weights()
    
    def _lazy_init_classification_head(
        self,
        *,
        in_dim: int,
        device: torch.device,
    ) -> None:
        """
        Initialize classification head for finetune mode.
        
        Args:
            in_dim: Input dimension (embedding_dim)
            device: Device to place head on
        """
        hidden_dims: List[int] = self.classification_head_kwargs['hidden_dims']
        activation_cls: type[nn.Module] = self.classification_head_kwargs['activation']
        dropout_p: float = self.classification_head_kwargs['dropout_p']
        
        layers: List[nn.Module] = []
        dims = [in_dim] + hidden_dims + [self.num_classes]
        
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_cls())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        
        # Final layer to logits
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.classification_head = nn.Sequential(*layers).to(device)
        
        # Initialize classification head weights
        self._init_classification_head_weights()
    
    def _init_classification_head_weights(self) -> None:
        """
        Initialize classification head weights.
        
        Uses Kaiming initialization for hidden layers and Xavier for the
        final output layer (logits).
        """
        if self.classification_head is None:
            return
        
        linear_idx = 0
        total_linear_layers = sum(
            1 for m in self.classification_head.modules() if isinstance(m, nn.Linear)
        )
        
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                linear_idx += 1
                
                if linear_idx == total_linear_layers:
                    # Final output layer: use Xavier initialization
                    # Better for classification logits
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    # Hidden layers: use Kaiming initialization
                    nn.init.kaiming_uniform_(
                        module.weight,
                        mode='fan_in',
                        nonlinearity='relu'
                    )
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
    
    def _freeze_encoder_weights(self) -> None:
        """Freeze all encoder weights for finetune mode."""
        if self.vector_bn is not None:
            for param in self.vector_bn.parameters():
                param.requires_grad = False
        
        if self.temporal_encoder is not None:
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False
        
        if self.projection_mlp is not None:
            for param in self.projection_mlp.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for supervised contrastive learning or classification.
        
        Args:
            batch: PyG Batch containing trajectory graphs
            
        Returns:
            Dictionary with:
                - 'preds': Embeddings (pretrain) or logits (finetune)
                - 'embeddings': Normalized embeddings (for visualization/analysis)
        """
        outputs: Dict[str, torch.Tensor] = {}
        device = infer_device_from_batch(
            batch,
            feature_keys=[self.vector_track_kwargs.get('feature_key')],
            operator_keys=[self.vector_track_kwargs.get('diffusion_op_key')],
        )
        
        # Extract vector features and diffusion operator
        vec_key = self.vector_track_kwargs.get('feature_key')
        op_key = self.vector_track_kwargs.get('diffusion_op_key')
        
        if (vec_key is None) or (op_key is None):
            raise ValueError(
                "vector_track_kwargs must define 'feature_key' and 'diffusion_op_key'"
            )
        
        x_v = getattr(batch, vec_key).to(device)  # (N, d) per trajectory
        Q = getattr(batch, op_key).to(device)     # Diffusion operator
        
        # Determine number of trajectories and time points
        # Assuming batch contains B trajectory graphs, each with N=35 nodes
        if hasattr(batch, 'batch'):
            batch_index = batch.batch  # (B*N,) node-to-graph assignment
            num_graphs = int(batch_index.max().item()) + 1
            nodes_per_graph = x_v.shape[0] // num_graphs
        else:
            # Single trajectory case
            num_graphs = 1
            nodes_per_graph = x_v.shape[0]
            batch_index = None
        
        # 1. Vector scattering: (N, d) → (N, 1, d, W_total)
        W_vector = self._scatter(
            track='vector',
            x0=x_v,
            P_or_Q=Q,
            kwargs=self.vector_track_kwargs,
            batch_index=batch_index,
        )  # Shape: (B*N, 1, d, W_total)
        
        # Get dimensions for lazy initialization
        _, _, d, W_total = W_vector.shape
        
        # Lazy initialize encoder layers
        if self.vector_bn is None:
            self._lazy_init_encoder_layers(
                num_wavelets=W_total,
                d=d,
                nodes_per_graph=nodes_per_graph,
                device=device,
            )
        
        # 2. VectorBatchNorm: normalize each wavelet channel
        # Input: (B*N, 1, d, W_total), Output: same shape
        if not self.ablate_vector_bn:
            W_vector_normalized = self.vector_bn(W_vector)
        else:
            W_vector_normalized = W_vector
        
        # 3. Reshape for temporal processing
        # (B*N, 1, d, W_total) → (B, N, 1, d, W_total)
        # First reshape to separate trajectories
        W_reshaped = W_vector_normalized.view(
            num_graphs, nodes_per_graph, 1, d, W_total
        )  # (B, N, 1, d, W_total)
        
        # Flatten wavelet and vector dimensions: (B, N, d*W_total)
        W_flat = W_reshaped.squeeze(2).reshape(
            num_graphs, nodes_per_graph, d * W_total
        )  # (B, N, d*W_total)
        
        # 4. TemporalEncoder (optional): detect temporal motifs
        if self.ablate_temporal_encoder:
            # Ablated case: flatten all time steps and channels into a single
            # feature vector per trajectory: (B, N, d*W_total) -> (B, N*d*W_total)
            h = W_flat.reshape(num_graphs, nodes_per_graph * d * W_total)
        else:
            # Non-ablated: Conv1d over time dimension
            # Transpose for Conv1d: (B, d*W_total, N)
            W_for_conv = W_flat.transpose(1, 2)  # (B, d*W_total, N)
            # Input: (B, d*W_total, N), Output: (B, temporal_out_dim)
            h = self.temporal_encoder(W_for_conv)
        
        # 5. ProjectionMLP: project to embedding space
        # Input: (B, temporal_out_dim), Output: (B, embedding_dim)
        embeddings = self.projection_mlp(h)
        
        # Store embeddings for visualization/analysis
        outputs['embeddings'] = embeddings
        
        # 6. Mode-dependent output
        if self.mode == 'pretrain':
            # For SupConLoss, return embeddings as predictions
            outputs['preds'] = embeddings
        else:  # finetune or train modes
            # Classification head: embeddings → logits
            logits = self.classification_head(embeddings)
            outputs['preds'] = logits
            if self.mode == 'train':
                labels = getattr(batch, self.target_attr, None)
                if labels is None:
                    raise ValueError(
                        "ESCGNN_macaque_supcon (train mode) requires "
                        f"batch.{self.target_attr} to compute SupConCrossEntropyLoss."
                    )
                if not isinstance(labels, torch.Tensor):
                    labels = torch.as_tensor(labels)
                labels = labels.to(device)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                labels = labels.view(-1).long()
                outputs['preds_tasks'] = {
                    'supcon': embeddings,
                    'cross_entropy': logits,
                }
                outputs['targets_tasks'] = {
                    'supcon': labels,
                    'cross_entropy': labels,
                }
        
        return outputs
    
    def run_epoch_zero_methods(
        self,
        batch: Batch,
    ) -> None:
        """
        Materialize lazily-created layers without requiring a full training step.
        
        This is invoked by the trainer at epoch 0 and before DDP wrapping.
        """
        device = infer_device_from_batch(
            batch,
            feature_keys=[self.vector_track_kwargs.get('feature_key')],
            operator_keys=[self.vector_track_kwargs.get('diffusion_op_key')],
        )
        
        vec_key = self.vector_track_kwargs.get('feature_key')
        op_key = self.vector_track_kwargs.get('diffusion_op_key')
        
        x_v = getattr(batch, vec_key).to(device)
        Q = getattr(batch, op_key).to(device)
        
        # Determine batch structure
        if hasattr(batch, 'batch'):
            batch_index = batch.batch
        else:
            batch_index = None
        
        # Run minimal scattering to determine dimensions
        W_vector = self._scatter(
            track='vector',
            x0=x_v,
            P_or_Q=Q,
            kwargs=self.vector_track_kwargs,
            batch_index=batch_index,
        )
        
        # Extract dimensions
        # Shape: (B*N, 1, d, W_total)
        BN, _, d, W_total = W_vector.shape

        # Infer num_graphs and nodes_per_graph for lazy initialization
        if hasattr(batch, 'batch') and batch.batch is not None:
            num_graphs = int(batch.batch.max().item()) + 1
            nodes_per_graph = BN // num_graphs
        else:
            num_graphs = 1
            nodes_per_graph = BN
        
        # Initialize all encoder layers
        self._lazy_init_encoder_layers(
            num_wavelets=W_total,
            d=d,
            nodes_per_graph=nodes_per_graph,
            device=device,
        )


    def visualize_embeddings(
        self,
        dataloader_dict: Dict[str, Any],
        save_dir: str,
        labels_attr: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        s: int = 12,
    ) -> None:
        """
        Visualize trajectory embeddings for each set in the dataloader_dict as a 
        3D scatter plot and save to disk.

        Assumes the projection MLP outputs embeddings of dimension 3. This method
        iterates over the provided dataloader, collects embeddings (and optional
        labels when available), and saves a 3D scatter plot in the specified
        directory. If labels are present on the batches (e.g., in the 'y' field),
        points are colored by label.

        Args:
            dataloader_dict: Dictionary containing 'train' and 'valid' and possibly
                'test' keys, each with a DataLoader yielding PyG Batch objects for 
                visualization.
            save_dir: Directory path where the plot image will be saved.
            labels_attr: Optional attribute name on each batch providing labels
                (e.g., the dataset target key). If None or not present on the
                batch objects, points are plotted with a single color.
            figsize: Tuple of width and height in inches for the figure.
            s: Size of the points in the scatter plot.
        """
        if dataloader_dict is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARNING] Could not import matplotlib for embedding visualization: {e}")
            return None

        device = self.get_device() if hasattr(self, "get_device") else torch.device("cpu")

        # Ensure save directory exists
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Could not create save_dir '{save_dir}' for embeddings plot: {e}")
            return None

        self.eval()

        for set_key in dataloader_dict.keys():
            dataloader = dataloader_dict[set_key]
            all_set_embeddings: List[torch.Tensor] = []
            all_set_labels: List[torch.Tensor] = []

            with torch.no_grad():
                for batch in dataloader:
                    if hasattr(batch, "to"):
                        batch = batch.to(device)

                    outputs = self(batch)
                    if "embeddings" in outputs:
                        embeddings = outputs["embeddings"]
                    else:
                        embeddings = outputs["preds"]

                    all_set_embeddings.append(embeddings.detach().cpu())

                    # Try to extract labels for coloring, if available
                    labels_tensor: Optional[torch.Tensor] = None
                    if isinstance(labels_attr, str) and hasattr(batch, labels_attr):
                        labels_tensor = getattr(batch, labels_attr)

                    if labels_tensor is not None:
                        all_set_labels.append(labels_tensor.detach().cpu())

            if len(all_set_embeddings) == 0:
                print("[WARNING] visualize_embeddings: no embeddings collected; skipping plot.")
                return None

            embeddings_cat = torch.cat(all_set_embeddings, dim=0)
            if embeddings_cat.dim() != 2 or embeddings_cat.shape[1] < 3:
                print(
                    f"[WARNING] visualize_embeddings: expected embeddings of shape (N, 3+), "
                    f"got {tuple(embeddings_cat.shape)}; skipping plot."
                )
                return None

            # Use first three dimensions for 3D scatter
            emb_np = embeddings_cat[:, 0:3].numpy()

            labels_np = None
            if len(all_set_labels) > 0:
                try:
                    labels_cat = torch.cat(all_set_labels, dim=0)
                    labels_np = labels_cat.squeeze().numpy()
                except Exception as e:
                    print(
                        "[WARNING] visualize_embeddings: could not convert labels to numpy "
                        f"array for color mapping: {e}"
                    )
                    labels_np = None

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            # ax.clear()

            if labels_np is not None:
                # Use a discrete colormap with exactly self.num_classes entries and
                # integer-spaced ticks so the colorbar shows one color per label.
                num_classes = getattr(self, "num_classes", int(labels_np.max()) + 1)
                cmap = plt.get_cmap("tab10", num_classes)

                scatter = ax.scatter(
                    emb_np[:, 0],
                    emb_np[:, 1],
                    emb_np[:, 2],
                    c=labels_np,
                    cmap=cmap,
                    s=s,
                    alpha=0.8,
                    vmin=-0.5,
                    vmax=num_classes - 0.5,
                )

                cbar = fig.colorbar(
                    scatter,
                    ax=ax,
                    label="Label",
                    ticks=list(range(num_classes)),
                    shrink=2.0 / 3.0,
                )
                cbar.ax.set_yticklabels([str(tick) for tick in range(num_classes)])
            else:
                ax.scatter(
                    emb_np[:, 0],
                    emb_np[:, 1],
                    emb_np[:, 2],
                    c="tab:blue",
                    s=s,
                    alpha=0.8,
                )

            ax.set_xlabel("Embed dim 1")
            ax.set_ylabel("Embed dim 2")
            ax.set_zlabel("Embed dim 3")
            ax.set_title(f"Trajectory embeddings ({set_key} set)")

            save_path = os.path.join(save_dir, f"embeddings_{set_key}_3d.png")
            try:
                fig.savefig(save_path, bbox_inches="tight", dpi=150)
                print(f"[INFO] Saved {set_key} set embeddings (n={len(labels_np)}) visualization to: {save_path}")
            except Exception as e:
                print(f"[WARNING] Could not save embedding visualization to '{save_path}': {e}")
            finally:
            #     plt.clf()
                plt.close(fig)

        return None

