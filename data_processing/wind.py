import os
import time
import xarray as xr  # requires netCDF4 (not auto-installed)
import numpy as np
import pandas as pd
from typing import List, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from models import nn_utilities as nnu
from scipy.spatial.transform import Rotation


U_WIND_FILENAME = "u-wind_1Jan2016_mean_10m.nc"
V_WIND_FILENAME = "v-wind_1Jan2016_mean_10m.nc"
EARTH_RADIUS = 6356.8  # from Battiloro et al. (2024), section VI.b
WIND_VECTOR_KEY = "wind_local_3d"
POS_KEY = "pos"
WIND_VECTOR_COLS = (
    f"{WIND_VECTOR_KEY}_x",
    f"{WIND_VECTOR_KEY}_y",
    f"{WIND_VECTOR_KEY}_z",
)

# Wind rotation dataset fixed params
# NOTE: bounds are in (south, north, east, west) format,
# and lat ranges from -90 to 90; lon ranges from 0 to 360.
WIND_ROT_SAMPLE_N = 400
WIND_ROT_TRAIN_BOUNDS = (24.0, 49.0, 235.0, 293.0)  # USA lower 48
# WIND_ROT_TEST_BOUNDS = (35.0, 60.0, 350.0, 50.0)  # Europe
WIND_ROT_TEST_BOUNDS = (-44.0, -10.0, 154.0, 113.0)  # Australia
# WIND_ROT_TEST_BOUNDS = (71, 90.0, 360.0, 0.0)  # North pole cap; use 77.5 south for equal lines of latitude instead of roughly equal area
# WIND_ROT_TRAIN_BOUNDS = (24.0, 49.0, -67.0, -125.0)
# WIND_ROT_TEST_BOUNDS = (35.0, 60.0, -10.0, 50.0)
WIND_ROT_MASK_PROP = 0.1


def open_wind_data(root: str) -> pd.DataFrame:
    zonal = xr.open_dataset(
        os.path.join(root, U_WIND_FILENAME)
    )
    meridional = xr.open_dataset(
        os.path.join(root, V_WIND_FILENAME)
    )
    df = pd.merge(
        zonal.to_dataframe(),
        meridional.to_dataframe(),
        on=["lat", "lon"]
    ).reset_index()
    nnu.raise_if_nonfinite_array(
        df[["uwnd", "vwnd"]].to_numpy(dtype=float, copy=False),
        name="open_wind_data: uwnd/vwnd",
    )
    return df


def convert_to_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert latitude and longitude to Cartesian coordinates.
    Note that lat and lon ranges from -90 to 90 and 0 to 360, respectively, and must be converted to radians first.
    """
    lat_rad = np.deg2rad(df["lat"].to_numpy(dtype=float, copy=False))
    lon_rad = np.deg2rad(df["lon"].to_numpy(dtype=float, copy=False))
    df["x"] = EARTH_RADIUS * np.cos(lat_rad) * np.cos(lon_rad)
    df["y"] = EARTH_RADIUS * np.cos(lat_rad) * np.sin(lon_rad)
    df["z"] = EARTH_RADIUS * np.sin(lat_rad)
    return df


def compute_wind_local_3d(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert zonal/meridional wind components to 3D tangent vectors in m/s.
    Assumes uwind is positive eastward, vwind is positive northward.
    The resulting wind vectors live in the Earth-centered, Earth-fixed (ECEF) Cartesian frame.
    """
    lat_rad = np.deg2rad(df["lat"].to_numpy(dtype=float, copy=False))
    lon_rad = np.deg2rad(df["lon"].to_numpy(dtype=float, copy=False))
    uwnd = df["uwnd"].to_numpy(dtype=float, copy=False)
    vwnd = df["vwnd"].to_numpy(dtype=float, copy=False)

    east = np.column_stack([
        -np.sin(lon_rad),
        np.cos(lon_rad),
        np.zeros_like(lon_rad),
    ])
    north = np.column_stack([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad),
    ])
    wind_local_3d = (uwnd[:, None] * east) + (vwnd[:, None] * north)
    nnu.raise_if_nonfinite_array(
        wind_local_3d, 
        name="compute_wind_local_3d: wind_local_3d"
    )

    df[WIND_VECTOR_COLS[0]] = wind_local_3d[:, 0]
    df[WIND_VECTOR_COLS[1]] = wind_local_3d[:, 1]
    df[WIND_VECTOR_COLS[2]] = wind_local_3d[:, 2]
    df[WIND_VECTOR_KEY] = list(wind_local_3d)
    return df


def add_unit_pos(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add unit-sphere position vectors to the dataframe.
    """
    if not all(col in df.columns for col in ["x", "y", "z"]):
        raise ValueError("add_unit_pos requires x/y/z columns on the dataframe.")
    pos = np.column_stack([
        df["x"].to_numpy(dtype=float, copy=False),
        df["y"].to_numpy(dtype=float, copy=False),
        df["z"].to_numpy(dtype=float, copy=False),
    ])
    pos_unit = pos / float(EARTH_RADIUS)
    nnu.raise_if_nonfinite_array(pos_unit, name="add_unit_pos: pos_unit")
    df[POS_KEY] = list(pos_unit)
    return df


def create_masked_dataset(
    df: pd.DataFrame,
    rng: np.random.RandomState,
    coord_cols: List[str] = ['x', 'y', 'z'],
    target_cols: List[str] = ['uwnd', 'vwnd'],
    vector_feat_key: str = "v",
    target_col: str = "target_vec",
    sample_n: int = 400,
    mask_prop: float = 0.1,
    pre_sampled: bool = False,
) -> pd.DataFrame:
    """
    Create a masked dataset, where held-out points are replaced with the mean of
    the train-observed set (reconstruction task).
    If pre_sampled=True, the provided df is used as-is (no additional sampling).
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe for masking is empty.")

    if mask_prop < 0.0 or mask_prop >= 1.0:
        raise ValueError("mask_prop must be in [0.0, 1.0).")

    sample_n = min(sample_n, len(df)) \
        if sample_n is not None else len(df)
    df_source = df if pre_sampled \
        else df.sample(n=sample_n, random_state=rng)
    df_n = df_source.copy()

    # Identify held-out rows across train/valid/test splits
    n_total = len(df_n)
    desired_total = int(round(mask_prop * n_total))
    split_base = desired_total // 3
    remainder = desired_total - 3 * split_base
    split_counts = [
        split_base + (1 if i < remainder else 0) for i in range(3)
    ]
    if sum(split_counts) > n_total:
        raise ValueError("Holdout split sizes exceed total number of nodes.")
    perm = rng.permutation(n_total)
    train_idx = perm[:split_counts[0]]
    valid_idx = perm[split_counts[0]:split_counts[0] + split_counts[1]]
    test_idx = perm[split_counts[0] + split_counts[1]:sum(split_counts)]
    is_train_holdout = np.zeros(n_total, dtype=bool)
    is_valid_holdout = np.zeros(n_total, dtype=bool)
    is_test_holdout = np.zeros(n_total, dtype=bool)
    is_train_holdout[train_idx] = True
    is_valid_holdout[valid_idx] = True
    is_test_holdout[test_idx] = True
    is_holdout = is_train_holdout | is_valid_holdout | is_test_holdout
    is_observed = ~is_holdout

    # Compute mean of unmasked targets
    target_arr = df_n[target_cols].to_numpy()
    nnu.raise_if_nonfinite_array(target_arr, name="create_masked_dataset: target_arr")
    if not np.any(is_observed):
        raise ValueError("No observed nodes available to compute mean replacement values.")
    mean_replace_vals = df_n.loc[is_observed, target_cols].mean().to_numpy()
    nnu.raise_if_nonfinite_array(
        mean_replace_vals,
        name="create_masked_dataset: mean_replace_vals",
    )
    print(f"Mean replace values:\n{mean_replace_vals}")

    # Build masked vector features (mean where hold-out, else original)
    vec_arr = target_arr.copy()
    vec_arr[is_holdout] = mean_replace_vals
    nnu.raise_if_nonfinite_array(vec_arr, name="create_masked_dataset: vec_arr")

    # Expected baseline MSE (masked nodes vs. mean replacement)
    if np.any(is_holdout):
        mse = np.mean((target_arr[is_holdout] - vec_arr[is_holdout]) ** 2)
        print(f"Mean model MSE: {mse:.4f}")
    else:
        print("Mean model MSE: nan (no held-out nodes)")

    # Persist columns
    df_n[vector_feat_key] = list(vec_arr)
    df_n[target_col] = list(target_arr)
    df_n['is_train_holdout'] = is_train_holdout
    df_n['is_valid_holdout'] = is_valid_holdout
    df_n['is_test_holdout'] = is_test_holdout
    df_n['is_observed'] = is_observed

    # Optional: remove unneeded columns
    # cols_to_drop = target_cols + ['lat', 'lon']
    # df_n.drop(columns=cols_to_drop, inplace=True)

    return df_n


def create_knn_graph(
    df: pd.DataFrame,
    knn_k: int = 10,
    vector_feat_key: str = "v",
    target_col: str = "target_vec",
    edge_weight_eps: float = 1e-6,
) -> Data:
    """
    Create a KNN graph as a PyTorch Geometric Data object with:
    - undirected observed-observed edges
    - directed observed -> masked edges
    - inverse-distance edge weights
    - row-normalized incoming weights (per target node)
    """
    coords = np.stack(df[POS_KEY].to_numpy()).astype(np.float32, copy=False)
    nnu.raise_if_nonfinite_array(coords, name="create_knn_graph: coords")
    pos_arr = torch.tensor(coords, dtype=torch.float32)
    vec_arr = torch.tensor(np.stack(df[vector_feat_key].to_numpy()), dtype=torch.float32)
    target_arr = torch.tensor(np.stack(df[target_col].to_numpy()), dtype=torch.float32)
    nnu.raise_if_nonfinite_tensor(vec_arr, name=f"create_knn_graph: {vector_feat_key}")
    nnu.raise_if_nonfinite_tensor(target_arr, name=f"create_knn_graph: {target_col}")

    is_train_holdout = df["is_train_holdout"].to_numpy(dtype=bool)
    is_valid_holdout = df["is_valid_holdout"].to_numpy(dtype=bool)
    is_test_holdout = df["is_test_holdout"].to_numpy(dtype=bool)
    is_holdout = is_train_holdout | is_valid_holdout | is_test_holdout
    is_observed = ~is_holdout

    observed_idx = np.where(is_observed)[0]
    masked_idx = np.where(is_holdout)[0]
    if observed_idx.size == 0:
        raise ValueError("No observed nodes available to build a graph.")

    edge_index_parts = []
    edge_weight_parts = []

    # Observed-observed undirected edges
    k_obs = min(knn_k, max(int(observed_idx.size) - 1, 0))
    if k_obs > 0:
        obs_coords = torch.tensor(coords[observed_idx], dtype=torch.float32)
        obs_edge_index = knn_graph(
            obs_coords,
            k=int(k_obs),
            loop=False,
        )
        obs_idx_t = torch.tensor(observed_idx, dtype=torch.long)
        obs_edge_index = obs_idx_t[obs_edge_index]
        obs_src = obs_edge_index[0].cpu().numpy()
        obs_dst = obs_edge_index[1].cpu().numpy()
        obs_dist = np.linalg.norm(coords[obs_src] - coords[obs_dst], axis=1)
        obs_weight = 1.0 / (obs_dist + edge_weight_eps)
        obs_edge_index, obs_weight = to_undirected(
            obs_edge_index,
            torch.tensor(obs_weight, dtype=torch.float32),
            reduce="min",
        )
        edge_index_parts.append(obs_edge_index)
        edge_weight_parts.append(obs_weight)

    # Observed -> masked directed edges
    k_mask = min(knn_k, int(observed_idx.size))
    if masked_idx.size > 0 and k_mask > 0:
        obs_coords_np = coords[observed_idx]
        mask_coords_np = coords[masked_idx]
        dists = np.linalg.norm(
            mask_coords_np[:, None, :] - obs_coords_np[None, :, :],
            axis=2,
        )
        nn_idx = np.argpartition(dists, kth=k_mask - 1, axis=1)[:, :k_mask]
        src_list = []
        dst_list = []
        weight_list = []
        for row_idx, mask_node in enumerate(masked_idx):
            obs_neighbors = observed_idx[nn_idx[row_idx]]
            nn_dists = dists[row_idx, nn_idx[row_idx]]
            src_list.append(obs_neighbors)
            dst_list.append(np.full_like(obs_neighbors, mask_node))
            weight_list.append(1.0 / (nn_dists + edge_weight_eps))
        src = np.concatenate(src_list) if src_list else np.array([], dtype=int)
        dst = np.concatenate(dst_list) if dst_list else np.array([], dtype=int)
        weights = np.concatenate(weight_list) if weight_list else np.array([], dtype=float)
        if src.size > 0:
            mask_edge_index = torch.tensor(
                np.stack([src, dst]),
                dtype=torch.long,
            )
            mask_edge_weight = torch.tensor(weights, dtype=torch.float32)
            edge_index_parts.append(mask_edge_index)
            edge_weight_parts.append(mask_edge_weight)

    if edge_index_parts:
        edge_index = torch.cat(edge_index_parts, dim=1)
        edge_weight = torch.cat(edge_weight_parts, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    if edge_weight.numel() > 0:
        # Normalize by target (incoming) so Q aggregates incoming messages directly.
        row = edge_index[1]
        row_sums = torch.zeros(
            (pos_arr.shape[0],),
            dtype=edge_weight.dtype,
            device=edge_weight.device,
        )
        row_sums.index_add_(0, row, edge_weight)
        scale = torch.zeros_like(row_sums)
        nonzero = row_sums > 0
        scale[nonzero] = 0.5 / row_sums[nonzero]
        edge_weight = edge_weight * scale[row]

    train_mask = torch.tensor(is_train_holdout, dtype=torch.bool)
    valid_mask = torch.tensor(is_valid_holdout, dtype=torch.bool)
    test_mask = torch.tensor(is_test_holdout, dtype=torch.bool)

    data = Data(
        y=target_arr,
        pos=pos_arr,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )
    # Store wind vectors under configurable key
    setattr(data, vector_feat_key, vec_arr)

    return data


def create_dataset(
    root: str,
    rng: np.random.RandomState,
    knn_k: int = 4,
    sample_n: int = 400,
    mask_prop: float = 0.1,
    vector_feat_key: str = WIND_VECTOR_KEY,
    target_cols: Optional[List[str]] = None,
) -> Data:
    """
    Create a dataset of wind data as a PyTorch Geometric Data object (k-NN graph).
    """
    # Process wind data
    t0 = time.time()
    df = open_wind_data(root)
    df = convert_to_cartesian(df)
    df = compute_wind_local_3d(df)
    df = add_unit_pos(df)
    t1 = time.time()
    print(f"Wind data processed in: {t1 - t0:.4f} seconds")

    t2 = time.time()
    # Create masked dataset
    if target_cols is None:
        target_cols = list(WIND_VECTOR_COLS)

    df = create_masked_dataset(
        df, 
        rng=rng,
        target_cols=target_cols,
        sample_n=sample_n,
        mask_prop=mask_prop,
        vector_feat_key=vector_feat_key,
        target_col="target_vec",
    )

    # Create KNN graph (PyG Data object)
    data = create_knn_graph(
        df, 
        knn_k=knn_k, 
        vector_feat_key=vector_feat_key,
        target_col="target_vec",
    )
    t3 = time.time()
    print(f"Graph dataset created in: {t3 - t2:.4f} seconds")

    return data


def _normalize_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """
    Normalize bounds to (lat_min, lat_max, lon_east, lon_west) keeping longitude
    orientation intact to allow wrap-around handling (e.g., 350 -> 50).
    """
    south, north, east, west = bounds
    lat_min, lat_max = min(south, north), max(south, north)
    lon_east, lon_west = east, west
    return lat_min, lat_max, lon_east, lon_west


def _filter_region(
    df: pd.DataFrame, 
    bounds: tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Return rows within the provided lat/lon bounds.
    """
    lat_min, lat_max, lon_east, lon_west = _normalize_bounds(bounds)
    lon_series = df["lon"]
    if lon_east <= lon_west:
        lon_mask = (lon_series >= lon_east) & (lon_series <= lon_west)
    else:
        # Wrap-around interval: keep [lon_east, 360) U [0, lon_west]
        lon_mask = (lon_series >= lon_east) | (lon_series <= lon_west)

    region_mask = (
        (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
        & lon_mask
    )
    region_df = df.loc[region_mask]
    if region_df.empty:
        raise ValueError(
            f"No rows found for bounds (lat: {lat_min}-{lat_max}, lon: {lon_east}-{lon_west})."
        )
    return region_df


def wind_rot_seed_rng(
    rotation_seed: int,
    combo_idx: int = 0,
    rep_idx: int = 0,
) -> np.random.RandomState:
    """
    Seed helper to keep regional sampling consistent across models.
    """
    base_seed = int(rotation_seed)
    adj_seed = base_seed + combo_idx * 1000 + rep_idx
    return np.random.RandomState(adj_seed)


def create_wind_rot_datasets(
    root: str,
    rng: np.random.RandomState,
    knn_k: int = 4,
    sample_n: int = WIND_ROT_SAMPLE_N,
    mask_prop: float = WIND_ROT_MASK_PROP,
    vector_feat_key: str = WIND_VECTOR_KEY,
    target_cols: Optional[List[str]] = None,
    train_bounds: tuple[float, float, float, float] = WIND_ROT_TRAIN_BOUNDS,
    test_bounds: tuple[float, float, float, float] = WIND_ROT_TEST_BOUNDS,
) -> tuple[Data, Data]:
    """
    Build train (region A) and test (region B) wind datasets for rotation
    equivariance evaluation.
    """
    t0 = time.time()
    df = open_wind_data(root)
    df = convert_to_cartesian(df)
    df = compute_wind_local_3d(df)
    df = add_unit_pos(df)
    t1 = time.time()
    print(f"[wind_rot] Wind data processed in: {t1 - t0:.4f} seconds")

    if target_cols is None:
        target_cols = list(WIND_VECTOR_COLS)

    df_train_region = _filter_region(df, train_bounds)
    df_test_region = _filter_region(df, test_bounds)

    train_masked = create_masked_dataset(
        df=df_train_region,
        rng=rng,
        target_cols=target_cols,
        vector_feat_key=vector_feat_key,
        target_col="target_vec",
        sample_n=sample_n,
        mask_prop=mask_prop,
        pre_sampled=False,
    )
    test_masked = create_masked_dataset(
        df=df_test_region,
        rng=rng,
        target_cols=target_cols,
        vector_feat_key=vector_feat_key,
        target_col="target_vec",
        sample_n=sample_n,
        mask_prop=mask_prop,
        pre_sampled=False,
    )

    train_data = create_knn_graph(
        train_masked,
        knn_k=knn_k,
        vector_feat_key=vector_feat_key,
        target_col="target_vec",
    )
    test_data = create_knn_graph(
        test_masked,
        knn_k=knn_k,
        vector_feat_key=vector_feat_key,
        target_col="target_vec",
    )

    t2 = time.time()
    print(f"[wind_rot] Train/Test graphs created in: {t2 - t1:.4f} seconds")
    return train_data, test_data


def apply_random_2d_rotation(
    data: Data,
    rng: np.random.RandomState,
    vector_feat_key: str = "v",
    target_key: str = "y",
    degrees_min: float = -180.0,
    degrees_max: float = 180.0,
    *,
    return_matrix: bool = False,
) -> tuple[Data, float] | tuple[Data, float, torch.Tensor]:
    """
    Apply a single random 2D rotation (shared across all nodes) to vector
    features and targets. For 3D vectors, rotation is applied in the x-y plane.
    Returns a cloned Data object and the sampled degree.
    """
    if not hasattr(data, vector_feat_key):
        raise ValueError(f"Data is missing vector feature key '{vector_feat_key}'.")
    if not hasattr(data, target_key):
        raise ValueError(f"Data is missing target key '{target_key}'.")

    angle_deg = float(rng.uniform(degrees_min, degrees_max))
    angle_rad = np.deg2rad(angle_deg)
    cos_val = float(np.cos(angle_rad))
    sin_val = float(np.sin(angle_rad))

    rotated = data.clone()
    vec = getattr(rotated, vector_feat_key)
    tgt = getattr(rotated, target_key)

    if vec.shape[-1] != tgt.shape[-1]:
        raise ValueError("Vector feature and target dimensions must match.")
    if vec.shape[-1] == 2:
        rot_mat = vec.new_tensor([[cos_val, -sin_val], [sin_val, cos_val]])
        rotated_vec = vec @ rot_mat.T
        rotated_tgt = tgt @ rot_mat.T
    elif vec.shape[-1] == 3:
        rot_mat = vec.new_tensor([
            [cos_val, -sin_val, 0.0],
            [sin_val, cos_val, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated_vec = vec @ rot_mat.T
        rotated_tgt = tgt @ rot_mat.T
    else:
        raise ValueError("apply_random_2d_rotation expects 2D or 3D vectors.")

    setattr(rotated, vector_feat_key, rotated_vec)
    setattr(rotated, target_key, rotated_tgt)

    if return_matrix:
        return rotated, angle_deg, rot_mat
    return rotated, angle_deg


def apply_random_3d_rotation(
    data: Data,
    rng: np.random.RandomState,
    vector_feat_key: str = "v",
    target_key: str = "y",
    degrees_min: float = 90.0,
    degrees_max: float = 160.0,
    max_attempts: int = 500,
    *,
    return_matrix: bool = False,
) -> tuple[Data, float] | tuple[Data, float, torch.Tensor]:
    """
    Apply a random 3D rotation (shared across all nodes) to vector
    features and targets (and positions if present). Rotation is sampled uniformly from SO(3), then
    accepted only if its overall rotation angle (i.e., Rotation.magnitude()) falls within
    [degrees_min, degrees_max] (in degrees).

    Returns a cloned Data object and the accepted rotation's overall angle in degrees.
    """
    if not hasattr(data, vector_feat_key):
        raise ValueError(f"Data is missing vector feature key '{vector_feat_key}'.")
    if not hasattr(data, target_key):
        raise ValueError(f"Data is missing target key '{target_key}'.")
    if degrees_min < 0.0 or degrees_max > 180.0 or degrees_min >= degrees_max:
        raise ValueError("degrees_min/max must satisfy 0 <= min < max <= 180.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

    rot = None
    for _ in range(max_attempts):
        candidate = Rotation.random(random_state=rng)
        candidate_deg = float(np.rad2deg(candidate.magnitude()))
        if (candidate_deg >= degrees_min) and (candidate_deg <= degrees_max):
            rot = candidate
            break
    if rot is None:
        raise Exception("Failed to sample a rotation satisfying angle constraints.")
    rot_mat = rot.as_matrix()

    rotated = data.clone()
    vec = getattr(rotated, vector_feat_key)
    tgt = getattr(rotated, target_key)
    if vec.shape[-1] != 3 or tgt.shape[-1] != 3:
        raise ValueError("apply_random_3d_rotation expects 3D vectors and targets.")

    rot_tensor = vec.new_tensor(rot_mat)
    rotated_vec = vec @ rot_tensor.T
    rotated_tgt = tgt @ rot_tensor.T

    setattr(rotated, vector_feat_key, rotated_vec)
    setattr(rotated, target_key, rotated_tgt)
    if hasattr(rotated, "pos"):
        pos = getattr(rotated, "pos")
        if pos is not None:
            if pos.shape[-1] != 3:
                raise ValueError("apply_random_3d_rotation expects 3D positions when present.")
            rotated_pos = pos @ rot_tensor.T
            setattr(rotated, "pos", rotated_pos)

    angle_deg = float(np.rad2deg(rot.magnitude()))
    if return_matrix:
        return rotated, angle_deg, rot_tensor
    return rotated, angle_deg


def plot_wind_local_3d_vectors(
    data: Data,
    vector_key: str = WIND_VECTOR_KEY,
    sample_n: Optional[int] = 500,
    rng: Optional[np.random.RandomState] = None,
    arrow_length: float = 0.2,
    figsize: tuple[float, float] = (7, 7),
    show: bool = True,
) -> None:
    """
    Plot 3D wind vectors on the unit sphere from a PyG Data object.
    """
    import matplotlib.pyplot as plt

    if not hasattr(data, vector_key):
        raise ValueError(f"Data is missing vector feature key '{vector_key}'.")
    if not hasattr(data, POS_KEY):
        raise ValueError(f"Data is missing '{POS_KEY}' coordinates for plotting.")

    wind_vecs = getattr(data, vector_key).detach().cpu().numpy()
    pos = getattr(data, POS_KEY).detach().cpu().numpy()

    if wind_vecs.shape[-1] != 3 or pos.shape[-1] != 3:
        raise ValueError("plot_wind_local_3d_vectors expects 3D vectors and positions.")

    if sample_n is not None and sample_n < wind_vecs.shape[0]:
        rng = rng if rng is not None else np.random.RandomState(0)
        idx = rng.choice(wind_vecs.shape[0], size=sample_n, replace=False)
        wind_vecs = wind_vecs[idx]
        pos = pos[idx]

    # rescale vectors to arrow_length, to appear better on the unit sphere
    norms = np.linalg.norm(wind_vecs, axis=1, keepdims=True)
    scale = np.where(norms > 0.0, arrow_length / norms, 0.0)
    wind_vecs_plot = wind_vecs * scale

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0.0, 2.0 * np.pi, 64)
    v = np.linspace(0.0, np.pi, 32)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="lightgray",
        alpha=0.15,
        linewidth=0.0,
        zorder=0,
    )

    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        wind_vecs_plot[:, 0],
        wind_vecs_plot[:, 1],
        wind_vecs_plot[:, 2],
        normalize=False,
        arrow_length_ratio=0.2,
        color="tab:blue",
    )
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        color="red",
        s=8,
        alpha=0.8,
        depthshade=False,
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("wind_local_3d vectors")
    if show:
        plt.show()
