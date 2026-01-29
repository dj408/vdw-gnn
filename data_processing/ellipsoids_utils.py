from typing import Any, Optional, List, Dict
import torch
from torch.utils.data import Dataset

from .ellipsoid_data_classes import (
    EllipsoidDatasetLoader,
    get_ellipsoid_dataset_info,
)
from models.comparisons.tangent_bundle_nn import (
    get_laplacians,
    EPSILON,
    EPSILON_PCA,
    GAMMA,
)


def load_ellipsoid_dataset(
    data_config: Any,
    *,
    print_info: bool = False,
    main_print=print,
):
    """
    Load ellipsoid dataset and optionally print dataset info.
    """
    data_dir = getattr(data_config, 'data_dir', None)
    if data_dir is None:
        raise ValueError("data_dir must be specified in config for ellipsoid datasets")

    if print_info:
        info = get_ellipsoid_dataset_info(
            data_dir,
            data_config.dataset_filename,
            scalar_feat_key=data_config.scalar_feat_key,
            vector_feat_key=data_config.vector_feat_key,
            target_key=data_config.target_key,
        )
        main_print(f"Ellipsoid dataset info:")
        main_print(f"  Dataset size: {info.get('dataset_size', 'N/A')}")
        main_print(f"  Nodes per sample: {info.get('num_nodes_per_sample', 'N/A')}")
        main_print(f"  Vector dimension: {info.get('vector_dim', 'N/A')}")
        if 'target_dim' in info:
            main_print(f"  Graph target dimension: {info['target_dim']}")
        main_print(
            f"  Configured task/target: {data_config.task} / {data_config.target_key}"
            f" (dim={data_config.target_dim})",
        )

    dataset = EllipsoidDatasetLoader(
        data_dir=data_dir,
        dataset_filename=data_config.dataset_filename,
    )
    return dataset


def attach_tnn_operators(
    dataset: Dataset,
    *,
    vector_feat_key: str,
) -> Dataset:
    """
    Compute and cache sheaf Laplacians and local frames for each graph for TNN.
    """
    for idx in range(len(dataset)):
        data = dataset[idx]
        if not hasattr(data, vector_feat_key):
            raise RuntimeError(f"Data sample missing '{vector_feat_key}' for TNN preprocessing.")
        coords = getattr(data, vector_feat_key)
        coords_np = coords.detach().cpu().numpy()
        Delta_n_numpy, _, _, O_i_collection, d_hat, _ = get_laplacians(
            coords_np,
            epsilon=EPSILON,
            epsilon_pca=EPSILON_PCA,
            gamma_svd=GAMMA,
            tnn_or_gnn="tnn",
        )
        operator = torch.from_numpy(Delta_n_numpy).to(torch.float32)
        O_tensor = torch.stack([torch.from_numpy(o).to(torch.float32) for o in O_i_collection], dim=0)
        data.tnn_operator = operator
        data.tnn_O = O_tensor
        data.tnn_d_hat = int(d_hat)
        data.tnn_node_count = int(coords_np.shape[0])
        dataset[idx] = data
    return dataset


class RotatedDataset(Dataset):
    """
    Wrap an existing Dataset/Subset and rotate `pos` and specified vector attributes.

    - Rotation is applied on-the-fly in __getitem__ to keep memory usage low.
    - Optionally recomputes diffusion operators P and Q for the rotated sample.
    
    Args:
        base_ds: The underlying dataset to wrap
        vector_attribs_to_rotate: List of attribute names to rotate (default common ellipsoid attrs)
        R: 3x3 rotation matrix; defaults to 90-deg rotation about z-axis
        recompute_pq: Whether to recompute diffusion operators after rotation
        recompute_kwargs: Kwargs forwarded to data_processing.process_pyg_data.process_pyg_data
                          Should include at least 'vector_feat_key'
    """

    R_90deg_z = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=torch.float32)

    def __init__(
        self,
        base_ds: Dataset,
        *,
        vector_attribs_to_rotate: Optional[List[str]] = None,
        R: Optional[torch.Tensor] = None,
        recompute_pq: bool = True,
        recompute_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.base_ds = base_ds
        if vector_attribs_to_rotate is None:
            self.vector_attribs_to_rotate = [
                'pos',
                'y_base_normals',
                'y_global_harmonic_normals', 
                'y_multiscale_harmonic_normals',
                'y_random_harmonic_normals',
                'y_spectral_vector_field'
            ]
        else:
            self.vector_attribs_to_rotate = vector_attribs_to_rotate
        self.R = R if (R is not None) else self.R_90deg_z
        self.recompute_pq = recompute_pq
        self.recompute_kwargs = recompute_kwargs or {}

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        data = self.base_ds[idx]

        # Rotate specified vector attributes
        for attr_name in self.vector_attribs_to_rotate:
            if hasattr(data, attr_name):
                attr_tensor = getattr(data, attr_name)
                if attr_tensor is None:
                    continue
                if attr_tensor.dim() == 2:
                    device = attr_tensor.device
                    R = self.R.to(device)
                    d_rot = R.shape[0]
                    if attr_tensor.shape[1] >= d_rot:
                        if attr_tensor.shape[1] == d_rot:
                            rotated = attr_tensor @ R.T
                        else:
                            head = attr_tensor[:, :d_rot]
                            tail = attr_tensor[:, d_rot:]
                            rotated = torch.cat([head @ R.T, tail], dim=1)
                        setattr(data, attr_name, rotated)

        # Recompute P/Q once after rotation if requested
        if self.recompute_pq and (not getattr(data, '_pq_recomputed', False)):
            try:
                from data_processing.process_pyg_data import process_pyg_data
                vec_key = self.recompute_kwargs.get('vector_feat_key', 'pos')
                device = getattr(data, vec_key).device if hasattr(data, vec_key) else 'cpu'
                data = process_pyg_data(
                    data,
                    geom_feat_key=vec_key,
                    device=device,
                    return_data_object=self.recompute_kwargs.get('return_data_object', True),
                    num_edge_features=self.recompute_kwargs.get('num_edge_features'),
                    hdf5_tensor_dtype=self.recompute_kwargs.get('hdf5_tensor_dtype', 'float16'),
                    graph_construction=self.recompute_kwargs.get('graph_construction', None),
                    use_mean_recentering=self.recompute_kwargs.get('use_mean_recentering', False),
                    sing_vect_align_method=self.recompute_kwargs.get('sing_vect_align_method', 'column_dot'),
                    local_pca_kernel_fn_kwargs=self.recompute_kwargs.get('local_pca_kernel_fn_kwargs', {}),
                )
                data._pq_recomputed = True
            except Exception:
                # Best-effort; leave as-is on failure
                return data

        return data



