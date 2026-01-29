import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from torch_geometric.data import Dataset, Data
from torch import Tensor
from sklearn.metrics.pairwise import pairwise_distances
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import torch
import pickle
import os


class EllipsoidPointCloudGenerator:
    """
    Generates point clouds sampled from 3D ellipsoids.
    """
    def __init__(self, random_state_seed: int = 573823):
        """
        Initialize the generator with a random state.
        
        Args:
            random_state_seed: Seed for numpy RandomState
        """
        self.rng = np.random.RandomState(random_state_seed)
    
    def generate_ellipsoid_points(
        self,
        num_ellipsoids: int,
        num_points: int,
        shape_dict: Dict[str, Tuple[float, float]],
        random_state_seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate point clouds from ellipsoids.
        
        Args:
            num_ellipsoids: Number of ellipsoids to generate
            num_points: Number of points to sample on each ellipsoid
            shape_dict: Dictionary with keys 'a', 'b', 'c' mapping to 
                (mean, std) tuples for normal distribution sampling
            random_state_seed: Optional seed override
            
        Returns:
            List of point clouds, each as (num_points, 3) array
        """
        if random_state_seed is not None:
            rng = np.random.RandomState(random_state_seed)
        else:
            rng = self.rng
            
        point_clouds = []
        
        for _ in range(num_ellipsoids):
            # Sample ellipsoid parameters from normal distributions
            a = rng.normal(shape_dict['a'][0], shape_dict['a'][1])
            b = rng.normal(shape_dict['b'][0], shape_dict['b'][1])
            c = rng.normal(shape_dict['c'][0], shape_dict['c'][1])
            
            # Ensure positive values
            a, b, c = abs(a), abs(b), abs(c)
            
            # Generate points on unit sphere
            phi = rng.uniform(0, 2 * np.pi, num_points)
            theta = np.arccos(2 * rng.uniform(0, 1, num_points) - 1)
            
            # Convert to Cartesian coordinates on unit sphere
            x_unit = np.sin(theta) * np.cos(phi)
            y_unit = np.sin(theta) * np.sin(phi)
            z_unit = np.cos(theta)
            
            # Scale by ellipsoid parameters
            x = a * x_unit
            y = b * y_unit
            z = c * z_unit
            
            # Stack into point cloud
            points = np.column_stack([x, y, z])
            point_clouds.append(points)
            
        return point_clouds

    def generate_ellipsoid_points_with_params(
        self,
        num_ellipsoids: int,
        num_points: int,
        shape_dict: Dict[str, Tuple[float, float]],
        random_state_seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float, float]]]:
        """
        Generate point clouds and return the underlying ellipsoid parameters.

        Args:
            num_ellipsoids: Number of ellipsoids to generate
            num_points: Number of points to sample on each ellipsoid
            shape_dict: Dictionary with keys 'a', 'b', 'c' mapping to
                (mean, std) tuples for normal distribution sampling
            random_state_seed: Optional seed override

        Returns:
            Tuple of:
              - List of point clouds, each as (num_points, 3) array
              - List of (a, b, c) tuples used to generate each ellipsoid
        """
        if random_state_seed is not None:
            rng = np.random.RandomState(random_state_seed)
        else:
            rng = self.rng

        point_clouds: List[np.ndarray] = []
        ellipsoid_params: List[Tuple[float, float, float]] = []

        for _ in range(num_ellipsoids):
            a = rng.normal(shape_dict['a'][0], shape_dict['a'][1])
            b = rng.normal(shape_dict['b'][0], shape_dict['b'][1])
            c = rng.normal(shape_dict['c'][0], shape_dict['c'][1])

            a, b, c = abs(a), abs(b), abs(c)

            phi = rng.uniform(0, 2 * np.pi, num_points)
            theta = np.arccos(2 * rng.uniform(0, 1, num_points) - 1)

            x_unit = np.sin(theta) * np.cos(phi)
            y_unit = np.sin(theta) * np.sin(phi)
            z_unit = np.cos(theta)

            x = a * x_unit
            y = b * y_unit
            z = c * z_unit

            points = np.column_stack([x, y, z])
            point_clouds.append(points)
            ellipsoid_params.append((a, b, c))

        return point_clouds, ellipsoid_params
    
    def compute_ellipsoid_diameter(self, points: np.ndarray) -> float:
        """
        Compute the diameter of an ellipsoid from its point cloud.
        
        Args:
            points: Point cloud as (num_points, 3) array
            
        Returns:
            Maximum Euclidean distance between any two points
        """
        # Compute pairwise distances more efficiently than a double loop
        # Leverage scikit-learn's pairwise_distances (backed by C/NumPy) and
        # simply take the maximum value. For the default 64 points this is
        # trivial to fit in memory; if you later sample thousands of points
        # consider using scipy.spatial.distance.pdist instead, which returns a
        # condensed vector.

        dists = pairwise_distances(points)
        return float(dists.max())


class EllipsoidDataset(Dataset):
    """
    PyTorch Geometric dataset of ellipsoid point clouds with k-NN graphs.
    """
    
    def __init__(
        self,
        point_clouds: List[np.ndarray],
        graph_level_targets: List[Dict[str, Tensor]],
        k: int = 5,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        random_weights_range: Optional[Tuple[int, int]] = (-5, 6),
        random_seed: Optional[int] = None,
        node_vector_targets: Optional[List[Dict[str, Tensor]]] = None,
        node_target_prefix: str = 'y_node_',
        graph_target_prefix: str = 'y_graph_',
        dirac_types: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            point_clouds: List of point clouds as (num_points, 3) arrays
            targets: List of target values (diameters)
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
            random_seed: Random seed for weight generation
        """
        self.point_clouds = point_clouds
        self.graph_level_targets = graph_level_targets
        self.k = k
        self.random_seed = random_seed
        self.random_weights_range = random_weights_range
        self.precomputed_node_vector_targets = node_vector_targets
        self.node_target_prefix = node_target_prefix
        self.graph_target_prefix = graph_target_prefix
        # Which Dirac node types to compute indices for (e.g., ['max','min'])
        self.dirac_types = dirac_types
        super().__init__(transform, pre_transform)
    
    def len(self):
        return len(self.point_clouds)
    
    def get(self, idx):
        points = self.point_clouds[idx]
        graph_targets = self.graph_level_targets[idx]
        
        # Convert to torch tensor
        pos = torch.tensor(points, dtype=torch.float)
        
        # Create k-NN graph (make it undirected so every node has outgoing edges)
        edge_index = knn_graph(pos, k=self.k, batch=None)
        try:
            edge_index = to_undirected(edge_index)
        except ImportError:
            # Fallback: manually add reversed edges
            edge_index = torch.cat([edge_index, edge_index[[1,0], :]], dim=1)
        
        # Node features: use coordinates as features as well
        # x = pos.clone()  # Ensure x and pos are identical coordinate tensors
        
        # Create PyTorch Geometric Data object
        data = Data(
            # x=x,  # [REMOVED] Node features (xyz coordinates, same as pos)
            pos=pos,  # Node positions (xyz coordinates, same as x)
            edge_index=edge_index,
            y=None  # Will set from graph targets if present
        )
        
        # Add graph-level targets with prefix, and set y from diameter if available
        if isinstance(graph_targets, dict):
            for key, value in graph_targets.items():
                val_tensor = value if isinstance(value, Tensor) else torch.tensor([float(value)], dtype=torch.float32)
                data[f"{self.graph_target_prefix}{key}"] = val_tensor
            if 'diameter' in graph_targets:
                diam_val = graph_targets['diameter']
                data.y = diam_val if isinstance(diam_val, Tensor) else torch.tensor([float(diam_val)], dtype=torch.float32)
        else:
            # Backward compatibility: if a single scalar was passed
            data.y = torch.tensor([graph_targets], dtype=torch.float32)

        # Add precomputed vector targets if available
        if self.precomputed_node_vector_targets is not None:
            vec_dict = self.precomputed_node_vector_targets[idx]
            for key, value in vec_dict.items():
                data[f"{self.node_target_prefix}{key}"] = value

        # --------------------------------------------------
        # Compute and attach Dirac node indices per requested type
        # 'diracs' is stored as a dict mapping type -> index (int)
        # Types supported: 'max' (max ||pos||), 'min' (min ||pos||)
        # --------------------------------------------------
        if self.dirac_types is not None:
            try:
                # Use distances from the centroid rather than the origin so
                # that 'min' and 'max' refer to closest/farthest from the
                # ellipsoid's center of mass (mean of points), not from (0,0,0).
                centroid = torch.mean(pos, dim=0, keepdim=True)
                centered_pos = pos - centroid
                norms = torch.norm(centered_pos, dim=1)
                diracs_dict: Dict[str, int] = {}
                if 'max' in self.dirac_types:
                    diracs_dict['max'] = int(torch.argmax(norms).item())
                if 'min' in self.dirac_types:
                    diracs_dict['min'] = int(torch.argmin(norms).item())
                if len(diracs_dict) > 0:
                    data.diracs = diracs_dict
            except Exception:
                # If anything goes wrong, skip attaching diracs
                pass
        
        return data


class _EllipsoidDatasetUnpickler(pickle.Unpickler):
    """Custom unpickler that resolves `__main__.EllipsoidDataset`."""

    def find_class(self, module: str, name: str):  # noqa: D401  (simple method)
        if module == "__main__" and name == "EllipsoidDataset":
            # Redirect to the actual class implementation
            return EllipsoidDataset
        return super().find_class(module, name)


class EllipsoidDatasetLoader(Dataset):
    """
    Dataset loader for pre-generated ellipsoid datasets.
    
    This class loads ellipsoid datasets that were saved using the 
    ellipsoid_dataset.py script and provides them in a format compatible
    with the ESCGNN training pipeline.
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_filename: str = 'ellipsoid_dataset.pkl',
        transform=None,
        pre_transform=None
    ):
        """
        Initialize the ellipsoid dataset loader.
        
        Args:
            data_dir: Directory containing the ellipsoid dataset pickle file
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
        """
        self.data_dir = data_dir
        self.dataset_filename = dataset_filename
        self.data_list = None
        
        # Load the dataset
        self._load_dataset()
        
        super().__init__(transform, pre_transform)
    
    def _load_dataset(self):
        """Load the ellipsoid dataset from pickle file."""
        # Look for the dataset file
        dataset_path = os.path.join(self.data_dir, self.dataset_filename)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Ellipsoid dataset not found at {dataset_path}. "
                f"Please run the ellipsoid_dataset.py script first to generate the dataset."
            )
        
        # Load the dataset using the compatibility unpickler so that datasets
        # produced by the generator script (saved under `__main__`) can be
        # loaded without errors.
        with open(dataset_path, 'rb') as f:
            self.data_list = _EllipsoidDatasetUnpickler(f).load()
        
        print(f"Loaded ellipsoid dataset with {len(self.data_list)} samples")
    
    def len(self):
        """Return the number of samples in the dataset."""
        return len(self.data_list)
    
    def get(self, idx):
        """Get a single sample from the dataset."""
        return self.data_list[idx]


def get_ellipsoid_dataset_info(
    data_dir: str, 
    dataset_filename: str = 'ellipsoid_dataset.pkl',
    scalar_feat_key: str = 'x',
    vector_feat_key: str = 'pos',
    target_key: str = 'y',
) -> Dict[str, any]:
    """
    Get information about an ellipsoid dataset.
    
    Args:
        data_dir: Directory containing the ellipsoid dataset
        
    Returns:
        Dictionary containing dataset information
    """
    dataset_path = os.path.join(data_dir, dataset_filename)
    
    if not os.path.exists(dataset_path):
        return {
            'error': f"Dataset not found at {dataset_path}"
        }
    
    # Load the dataset to get information
    with open(dataset_path, 'rb') as f:
        dataset = _EllipsoidDatasetUnpickler(f).load()
    
    # Get basic information
    info = {
        'dataset_size': len(dataset),
        'available_splits': ['combined'],  # Single combined dataset
    }
    
    # Get sample information from first few samples
    if len(dataset) > 0:
        sample = dataset[0]
        info['num_nodes 1st sample'] = sample[scalar_feat_key].shape[0] if hasattr(sample, scalar_feat_key) else 'N/A'
        info['vector_dim'] = sample[vector_feat_key].shape[1] if hasattr(sample, vector_feat_key) else 'N/A'
        info['target_dim'] = sample[target_key].shape[0] if hasattr(sample, target_key) else 'N/A'
        
        # Get sample target statistics
        targets = [data[target_key] \
            if hasattr(data, target_key) \
            else 0.0 for data in dataset[:100]]  # Sample first 100
        targets = torch.stack(targets)
        dim = 0 if targets.ndim == 1 else 1
        info['target_mean'] = torch.mean(targets, dim=dim)
        info['target_std'] = torch.std(targets, dim=dim)
        info['target_min'] = torch.min(targets, dim=dim)
        info['target_max'] = torch.max(targets, dim=dim)
    
    return info 