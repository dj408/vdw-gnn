r"""
Data processing methods for the macaque reaching data.

Originally processed by Gosztolai et al. 2025 ("MARBLE"): "recorded at 20ms sampling and smoothed at 100ms Gaussian filter"
 - URL: https://dataverse.harvard.edu/file.xhtml?fileId=6969883&version=12.0
 - Note that the spike data ('rate_data_20ms_100ms.pkl') has 60 time points, but the kinematics data ('kinematics.pkl') has 35 time points (60 - GO_CUE of 25 = 700ms/20ms).

IMPORTANT!
 - Days 32-43 have 'None' ("Down" condition) in the `kinematics` data
 - However, the processed `spike_data` excludes this condition, so there are only spike trains for 7 conditions, not 8

The spike_data ("rate_data_20ms_100ms.pkl") contains arrays (separated by day, but stacked by condition within days) of trial spike trains. Each trial is a row, of shape (n_neurons, n_timepoints) = (24, 60). This represents the following preprocessing steps already applied:
 - Smoothing with a 100ms Gaussian filter
 - Subsampling to 20ms intervals

The methods in this file follow this pipeline (per day):
(1) Global data denoising
    (a) Slice the spike data from GO_CUE (or GO_CUE + CUT_T_AFTER_GO_CUE if CUT_T_AFTER_GO_CUE > 0) to end. Kinematics data (already pre-sliced to GO_CUE=25, giving 35 timepoints) are further sliced by CUT_T_AFTER_GO_CUE if CUT_T_AFTER_GO_CUE > 0.
    (b) Nonlinear: Apply a Savitzky-Golay filter to each trial's neural trajectory if SAVGOL_FILTER_KWARGS is not None ().
    (b) Linear: Optional PCA: Skip PCA if GLOBAL_PCA_REDUCED_DIM is None or >= 24, or GLOBAL_PCA_VARIANCE_RETAINED is None or == 1.0. Otherwise, PCA is enabled. When enabled, fit PCA per day on all trials for that day in 'transductive' mode; in 'inductive' mode, fit only on the train trajectories. If GLOBAL_PCA_VARIANCE_RETAINED is in (0,1), set the number of components to retain that variance; otherwise use GLOBAL_PCA_REDUCED_DIM.
(2) Transform and construct graphs: 
    (a) Use the (optionally PCA-reduced) neural state x[t] after filtering and slicing (GO_CUE + CUT_T_AFTER_GO_CUE) to build daily spatial CkNN graphs and compute O-frames. 
    (b) For trajectory path graphs, use neural velocity vectors v[t] = x[t+1] - x[t] assigned to time t as vector node features. Node targets use the kinematics (positions and velocities) sliced to match the spike data and truncated to the node count (since the last time point will not have a velocity vector associated with it).

Conceptually, this pipeline 
(1) Slices the neural state data from GO_CUE (or GO_CUE + CUT_T_AFTER_GO_CUE if CUT_T_AFTER_GO_CUE > 0), globally denoises the data to a new ambient dimension D, then collects all time points from all trials (regardless of condition) onto an assumed 'neural state' manifold (that is, a CkNN graph scaffold of a manifold), with an assumed intrinsic dimension d < D.
(2) Numerically differentiates the neural state data (of dimension D) at each time point to get 'neural velocity' vectors, also of dimension D. 
    - As a convention, even though this is the instantaneous velocity for time t + 0.5, the velocity vectors are assigned to time the prior time point t. That is, v[t] = x[t+1] - x[t]. As a result, we drop the last time point from the neural state data.
(3) Performs a local SVD at each node to learn the local tangent space (i.e., get a basis for it from singular vectors) based on local curvature, and construct the O-frames (with a sign-flipping trick for aligning neighboring nodes' singular vectors) that can project node features from D to d. 
    - Specifically, O_i is comprised of singular vectors from the local SVD at node i. Each O_i \in R^{D \times d}. 
    - The manifold intrinsic dimension d can be hardcoded, or determined by the rounded mean participation ratio / spectral entropy of the singular values from each local SVD.
(4) Each v[t] \in R^D is associated with a point/node i, and is next projected into the point's d-dimensional local tangent space using the O-frames from (3), to get the 'tangent neural velocity' vector as the node feature. 
    - Formally, v^{\text{tangent}}[t] = O_i[:, :d]^T v[t], where v^{\text{tangent}}[t] is the tangent neural velocity vector at node i for time t.
    - Note that this means we 'differentiate then project'. If instead we projected each x[t] into its local tangent space before we differentiated/subtracted, this would incorrectly mix different tangent spaces, that is, fail to align in tangent space first, since x[t+1] has a different tangent space than x[t]. In other words, we need to measure global displacement first, before expressing it in local frames of reference, which can vary by rotations and translations. Think of trying to measure the displacement of a plane flying from New York to Melbourne in local 2-d coordinates.
(5) Each block of the Q diffusion operator matrix (used to diffuse the tangent vector node features across local frames) is constructed based on GEOMETRIC_MODE:
    - 'equivariant': Q[i,j] = O_i @ O_j.T (requires D=d, i.e., O_i \in R^{d x d})
        Preserves SO(d) equivariance of tangent space vectors.
        Use when the manifold dimension d equals the ambient dimension D globally.
    - 'invariant': Q[i,j] = O_i[:, :d].T @ O_j[:, :d] (allows D>d, i.e., O_i \in R^{D x d})
        Projects to intrinsic space, gaining SO(D) invariance but losing ambient equivariance.
        Use when the manifold has intrinsic dimension d < ambient dimension D.
    Hence Q \in R^{Nd x Nd}. Using Q[i,j], we can diffuse v_i with v_j (both in R^d), since Q[i,j] transports v_j into node i's local frame.
"""

import os 
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Literal, Sequence, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree, remove_self_loops
from .process_pyg_data import compute_node_median_edge_weight
from scipy.signal import savgol_filter
from numpy.random import default_rng
from typing import Callable
import torch
import torch.nn as nn
from .cknn import cknneighbors_graph
from sklearn.neighbors import KDTree, BallTree, NearestNeighbors
import h5py
from .process_pyg_data import (
    get_C_i_dict, 
    get_square_matrix_coo_indices,
    match_B_col_directions_to_A,
)
from models.escgnn_data_classes import ESCGNNData
from .data_utilities import multi_strat_multi_fold_idx_sample

# Default parallel backend for data processing
PARALLEL_BACKEND: Literal["processes", "threads"] = "threads"
NUM_WORKERS_DEFAULT: int = 8

# Hardcoded list of 7 conditions, in the same order as the MARBLE code
CONDITIONS: List[str] = [
    "DownLeft", "Left", "UpLeft", "Up", 
    "UpRight", "Right", "DownRight"
]

# Dimensionality of the original neural state data
ORIGINAL_NEURAL_STATE_DIM: int = 24

# Index of first timepoint AFTER 'go' to include (slice from here to end)
# Note kinematics data is already sliced down to 35 time points
# Default: 25 -> leaves last 35 of 60 timepoints -> 34 after finite difference -> 33 for OLE decoder in MARBLE, since they remove the last ()haps didn't realize np.diff alreadys does this
GO_CUE: int = 25

# Additional time points to cut after GO_CUE (there's a variable delay before the lever moves)
CUT_T_AFTER_GO_CUE: int = 0  

# Attribute name for the spike data in the data dicts and pytorch-geometric Data graph objects
NEURAL_STATE_ATTRIBUTE: str = "v_state"
NEURAL_VELOCITY_ATTRIBUTE: str = "v_vel"

# Graph construction mode: use CkNN or standard k-NN
# - True: Use continuous k-NN (CkNN) with delta parameter for adaptive neighborhoods
# - False: Use standard k-NN with fixed neighborhood size
KNN_N_NEIGHBORS: int = 30
USE_CKNN_GRAPH: bool = True
CKNN_DELTA: float = 1.5  # 1.4 in MARBLE Sup. Material; 1.5 in code

# Geometric mode: 'equivariant' (SO(d)-equivariant wavelets) or 'invariant'
# Controls how Q[i,j] is computed:
# - 'equivariant': O_i @ O_j.T (equivariant to rotations)
#     Requires D = d (GLOBAL_PCA_REDUCED_DIM must equal MANIFOLD_INTRINSIC_DIM)
#     O-frames are (d x d) square matrices
#     Q[i,j] preserves SO(d) equivariance of tangent space vectors
# - 'invariant': O_i.T @ O_j (invariant to rotations)
#     Allows D > d (ambient dimension can exceed intrinsic dimension)
#     O-frames are (D x d) rectangular matrices
#     Q[i,j] projects to intrinsic space, losing ambient equivariance
GEOMETRIC_MODE: Literal['equivariant', 'invariant'] = 'equivariant'

# Dimensionality of the manifold intrinsic dimension / tangent space / neural velocity vectors
# (overridden if GLOBAL_PCA_VARIANCE_RETAINED is not None, or < 1.0)
GLOBAL_PCA_REDUCED_DIM: Optional[int] = None # 0.95

# Fraction of variance to retain in global PCA (if in (0,1)); set to 1.0 or None to disable 
# variance-based selection; otherwise, dimension found will override GLOBAL_PCA_REDUCED_DIM
GLOBAL_PCA_VARIANCE_RETAINED: Optional[float] = None # 0.9

# Dimensionality of the manifold intrinsic dimension / tangent space / locally-projected 
# neural velocity vectors
# For 'equivariant' mode: MUST be an integer (not 'spectral_entropy')
# For 'invariant' mode: Can be integer or 'spectral_entropy' for automatic estimation
# Special case: use 'match_pca' to set d equal to the realized PCA dimension (enables
# equivariant mode when PCA uses variance retention).
MANIFOLD_INTRINSIC_DIM: int | str = 24 # 'match_pca' | 'spectral_entropy'

# Seed for the train/val/test splits
DEFAULT_SPLIT_SEED: int = 123456

# Default Savitzky-Golay filter kwargs
APPLY_SAVGOL_FILTER_BEFORE_PCA: bool = True
SAVGOL_FILTER_KWARGS: Dict[str, Any] = {
    "window_length": 9,  # MARBLE uses 9
    "polyorder": 2,  # MARBLE uses 2
}

# Default to apply standard scaler before PCA
# NOTE: this will reduce the dominance of high-variance features (neuron channels) in the PCA,
# and can lead to lower overall variance explained by the PCA. We default to False because it 
# seems to discard useful signal, perhaps from between-channel variability.
# And, skipping the per-channel standardization leaves the PCA scores with huge variance anisotropy, 
# which causes the CkNN ratio test to accept vastly more neighbors. The resulting dense graphs 
# make every downstream step—median-kernel reweighting, O-frame SVD prep, and Q block generation 
# slower. 
APPLY_STANDARD_SCALER_BEFORE_PCA: bool = False

# Target preprocessing: 'standard' (z-score) or 'minmax' (scale to [-1, 1])
# - 'minmax' is recommended for position (bounded workspace, no outliers)
# - 'standard' works for velocity (zero-centered, unbounded distribution)
TARGET_SCALER_TYPE: Literal['standard', 'minmax'] = 'minmax'


def load_data_dicts(data_root) -> Tuple[Dict]:
    """
    Loads the data dictionaries from the data root directory.

    `kinematics` structure:
    {
        day: {
            trial_id: {
                "condition": condition,
                "kinematics": kinematics, shape (5, n_timepoints=35=60 - GO_CUE of 25=700ms/20ms), first two are xy positions, next two are xy velocities, last seems to be an artifact of time subsampling (arrays of all 1s)
                "time": time, shape (n_timepoints, 1)
            }
        }
    }

    `spike_data` structure:
    {
        day: {
            condition: spike_data, shape (n_trials, n_neurons=24, n_timepoints=60)
        }
    }

    `trial_ids` structure:
    {
        day: {
            condition: trial_ids, shape (n_trials, 1)
        }
    }
    """
    with open(os.path.join(data_root, "kinematics.pkl"), 'rb') as f:
        kinematics = pickle.load(f)
    with open(os.path.join(data_root, "trial_ids.pkl"), 'rb') as f:
        trial_ids = pickle.load(f)
    with open(os.path.join(data_root, "rate_data_20ms_100ms.pkl"), 'rb') as f:
        spike_data = pickle.load(f)
    return kinematics, trial_ids, spike_data


def _merge_one_day(
    day: Any,
    conditions: Dict[str, np.ndarray],
    trial_ids_day: Dict[str, np.ndarray],
    kinematics_day: Dict[Any, Dict[str, Any]],
    include_lever_velocity: bool,
) -> List[Dict[str, Any]]:
    """
    Helper to merge one day's worth of data. Returns a list of sample dicts.
    """
    day_list: List[Dict[str, Any]] = []
    for condition, arr in conditions.items():
        # condition arrays are (n_trials, n_neurons, n_timepoints)
        for i, trial_arr in enumerate(arr):
            trial_id = trial_ids_day[condition][i]
            # Kinematics are already sliced to GO_CUE (35 timepoints); apply CUT_T_AFTER_GO_CUE if needed
            lever_pos_raw = kinematics_day[trial_id]["kinematics"][:2]
            lever_pos = _slice_kinematics(lever_pos_raw)
            data_dict = {
                "day": day,
                "trial_id": trial_id,
                NEURAL_VELOCITY_ATTRIBUTE: trial_arr,
                "lever_pos": lever_pos,
                "condition": condition,
            }
            if include_lever_velocity:
                lever_vel_raw = kinematics_day[trial_id]["kinematics"][2:4]
                lever_vel = _slice_kinematics(lever_vel_raw)
                data_dict["lever_vel"] = lever_vel
            day_list.append(data_dict)
    return day_list


def merge_data_dicts(
    kinematics: Dict,
    trial_ids: Dict,
    spike_data: Dict,
    include_lever_velocity: bool = False,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    days_included_idx: Optional[List[int]] = None,
    return_grouped_by_day: bool = False,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Combines the data dictionaries into a single list of per-trial dictionaries.

    NOTE 1:
    Since days 0-31 do not have 'None' ("Down" condition) it appears that Gosztolai et al. 2025 excluded this condition from the spike data. This method loops through the spike data, and so also excludes this condition (though these trials do exist in the kinematics data).

    NOTE 2:
    - `kinematics` has original trial order, with random conditions
    - `trial_ids` has arrays assigning trials to conditions (unique only within days)
    - `spike_data` arrays are condition-stacked according to id lists in `trial_ids`

    Thus, kinematic targets for each row can be indexed as `spike_data[day][condition][i]` goes with `day_trial_id = trial_ids[day][condition][i]` and target `kinematics[day][day_trial_id]`.

    Parallelization:
    - Set `num_workers > 1` to parallelize per-day merging using either processes (default) or threads.

    If `return_grouped_by_day` is True, returns a list of lists of dictionaries, grouped by day. Otherwise, returns a flat list of dictionaries.
    """
    # Determine day ordering and selection
    all_days_sorted = sorted(spike_data.keys())
    if days_included_idx is not None:
        selected_days = [all_days_sorted[i] for i in days_included_idx if 0 <= i < len(all_days_sorted)]
    else:
        selected_days = all_days_sorted

    print(f"Merging macaque data for {len(selected_days)} days: {selected_days}")

    # Prepare container(s)
    if return_grouped_by_day:
        # Always return aligned to all_days_sorted so merged[day_index] works
        grouped: List[List[Dict[str, Any]]] = [[] for _ in range(len(all_days_sorted))]
    else:
        flat: List[Dict[str, Any]] = []

    # Sequential path or effectively single worker
    if not num_workers or num_workers <= 1:
        for day in selected_days:
            conditions = spike_data[day]
            day_list = _merge_one_day(
                day,
                conditions,
                trial_ids[day],
                kinematics[day],
                include_lever_velocity,
            )
            if return_grouped_by_day:
                day_idx = all_days_sorted.index(day)
                grouped[day_idx] = day_list
            else:
                flat.extend(day_list)
        return grouped if return_grouped_by_day else flat

    # Parallel path per day
    executor_cls = ProcessPoolExecutor \
        if backend == "processes" else ThreadPoolExecutor
    with executor_cls(max_workers=num_workers) as ex:
        futures = {
            ex.submit(
                _merge_one_day,
                day,
                spike_data[day],
                trial_ids[day],
                kinematics[day],
                include_lever_velocity,
            ): day
            for day in selected_days
        }
        for fut, day in futures.items():
            try:
                day_list = fut.result()
                if return_grouped_by_day:
                    day_idx = all_days_sorted.index(day)
                    grouped[day_idx] = day_list
                else:
                    flat.extend(day_list)
            except Exception as e:
                print(f"merge_data_dicts worker failed (day {day}): {e}")
                pass

    return grouped if return_grouped_by_day else flat


def _slice_spike_data(trial_spike: np.ndarray) -> np.ndarray:
    """
    Slice a single trial's spike matrix (n_neurons, n_timepoints)
    to keep only timepoints from GO_CUE (or GO_CUE + CUT_T_AFTER_GO_CUE if CUT_T_AFTER_GO_CUE > 0) to end.
    """
    if trial_spike.ndim != 2:
        raise ValueError(
            "Expected trial_spike to be 2D (n_neurons, n_timepoints)"
        )
    start_idx = GO_CUE + (CUT_T_AFTER_GO_CUE if CUT_T_AFTER_GO_CUE > 0 else 0)
    return trial_spike[:, start_idx:]


def _slice_kinematics(kinematics_array: np.ndarray) -> np.ndarray:
    """
    Slice kinematics array (already pre-sliced to GO_CUE) by CUT_T_AFTER_GO_CUE.
    If CUT_T_AFTER_GO_CUE <= 0, returns the array unchanged.
    
    Args:
        kinematics_array: Array of shape (n_features, n_timepoints) where n_timepoints
            is already sliced to 35 (60 - GO_CUE of 25).
    
    Returns:
        Sliced array of shape (n_features, n_timepoints - CUT_T_AFTER_GO_CUE) if
        CUT_T_AFTER_GO_CUE > 0, otherwise returns the original array.
    """
    if CUT_T_AFTER_GO_CUE <= 0:
        return kinematics_array
    if kinematics_array.ndim != 2:
        raise ValueError(
            "Expected kinematics_array to be 2D (n_features, n_timepoints)"
        )
    return kinematics_array[:, CUT_T_AFTER_GO_CUE:]


def _split_indices_per_day(
    samples: List[Dict[str, Any]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministically split per-day sample indices into train/val/test.
    """
    n = len(samples)
    rng = default_rng(seed=seed)
    indices = np.arange(n)

    # Shuffle indices
    rng.shuffle(indices)

    # Calculate number of samples for each split
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # n_test = n - n_train - n_val

    # Split indices
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def _fit_target_scalers_for_day(
    samples: List[Dict[str, Any]],
    fit_indices: Optional[List[int]],
    scaler_type: Literal['standard', 'minmax'] = TARGET_SCALER_TYPE,
) -> Tuple[Union[StandardScaler, MinMaxScaler], Union[StandardScaler, MinMaxScaler]]:
    """
    Fit scalers for lever position and velocity targets using provided indices.
    - In 'inductive' mode, pass only TRAIN indices.
    - In 'transductive' mode, pass None to fit on ALL indices.
    
    Args:
        samples: List of sample dictionaries
        fit_indices: Indices to fit on (None = all)
        scaler_type: 'standard' (z-score) or 'minmax' (scale to [-1, 1])
    
    Returns:
        pos_scaler: Scaler fit on lever_pos (x, y)
        vel_scaler: Scaler fit on lever_vel (vx, vy)
    
    Note: Each dimension (x/y or vx/vy) is scaled independently.
    """
    fit_iter = range(len(samples)) if fit_indices is None else fit_indices
    
    pos_list: List[np.ndarray] = []
    vel_list: List[np.ndarray] = []
    
    for idx in fit_iter:
        pos = samples[idx].get("lever_pos")  # (2, T)
        vel = samples[idx].get("lever_vel")  # (2, T)
        if pos is not None:
            pos_list.append(pos.T)  # (T, 2)
        if vel is not None:
            vel_list.append(vel.T)  # (T, 2)
    
    # Stack across all timepoints
    pos_stack = np.concatenate(pos_list, axis=0) if pos_list else np.zeros((0, 2))
    vel_stack = np.concatenate(vel_list, axis=0) if vel_list else np.zeros((0, 2))
    
    # Create scalers based on type
    if scaler_type == 'minmax':
        # Scale to [-1, 1] instead of default [0, 1]
        pos_scaler = MinMaxScaler(feature_range=(-1, 1))
        vel_scaler = MinMaxScaler(feature_range=(-1, 1))
    else:  # 'standard'
        pos_scaler = StandardScaler(with_mean=True, with_std=True)
        vel_scaler = StandardScaler(with_mean=True, with_std=True)
    
    # Fit scalers
    pos_fitted = pos_stack.shape[0] > 0
    vel_fitted = vel_stack.shape[0] > 0
    if pos_fitted:
        pos_scaler.fit(pos_stack)
    if vel_fitted:
        vel_scaler.fit(vel_stack)
    
    # Print statistics
    if scaler_type == 'minmax':
        pos_stats = (
            f"pos: min=[{pos_scaler.data_min_[0]:.2f}, {pos_scaler.data_min_[1]:.2f}], "
            f"max=[{pos_scaler.data_max_[0]:.2f}, {pos_scaler.data_max_[1]:.2f}]"
            if pos_fitted
            else "pos: no samples available to fit scaler"
        )
        vel_stats = (
            f"vel: min=[{vel_scaler.data_min_[0]:.2f}, {vel_scaler.data_min_[1]:.2f}], "
            f"max=[{vel_scaler.data_max_[0]:.2f}, {vel_scaler.data_max_[1]:.2f}]"
            if vel_fitted
            else "vel: no samples available to fit scaler"
        )
        print(f"[INFO] Fitted MinMax scalers to [-1, 1] - {pos_stats}; {vel_stats}")
    else:
        pos_stats = (
            f"pos: mean=[{pos_scaler.mean_[0]:.2f}, {pos_scaler.mean_[1]:.2f}], "
            f"std=[{np.sqrt(pos_scaler.var_[0]):.2f}, {np.sqrt(pos_scaler.var_[1]):.2f}]"
            if pos_fitted
            else "pos: no samples available to fit scaler"
        )
        vel_stats = (
            f"vel: mean=[{vel_scaler.mean_[0]:.2f}, {vel_scaler.mean_[1]:.2f}], "
            f"std=[{np.sqrt(vel_scaler.var_[0]):.2f}, {np.sqrt(vel_scaler.var_[1]):.2f}]"
            if vel_fitted
            else "vel: no samples available to fit scaler"
        )
        print(f"[INFO] Fitted StandardScalers - {pos_stats}; {vel_stats}")
    
    return pos_scaler, vel_scaler
    

def _fit_scalar_and_pca_for_day(
    samples: List[Dict[str, Any]],
    fit_indices: Optional[List[int]],
    pca_dim_or_var_retained: Union[int, float],
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
    apply_standard_scaler_before_pca: bool = APPLY_STANDARD_SCALER_BEFORE_PCA,
    savgol_filter_kwargs: Optional[Dict[str, Any]] = SAVGOL_FILTER_KWARGS,
) -> Tuple[StandardScaler, PCA]:
    """
    Optionally apply Savitzky-Golay filter, then fit StandardScaler and PCA for a given day using the provided indices.
    - In 'inductive' mode, pass only TRAIN indices. 
    - In 'transductive' mode, pass None to fit on ALL indices (train + val + test) for the full day.
    """
    # Assemble training matrix by stacking timepoints across trials
    X_list: List[np.ndarray] = []
    fit_iter = range(len(samples)) if fit_indices is None else fit_indices
    window_length = savgol_filter_kwargs["window_length"]
    polyorder = savgol_filter_kwargs["polyorder"]
    
    for idx in fit_iter:
        # Extract spike data for this trial
        trial = samples[idx][NEURAL_VELOCITY_ATTRIBUTE]

        # Slice after GO_CUE
        trial = _slice_spike_data(trial)

        # Apply filter if requested
        if apply_savgol_filter_before_pca:
            trial = savgol_filter(
                trial,
                window_length=window_length,
                polyorder=polyorder,
                axis=1,
            )

        # Append to list
        X_list.append(trial.T)  # (t, n_neurons)
    
    # Stack timepoints across trials
    X_stack = np.concatenate(X_list, axis=0)  # (sum_t, n_neurons)

    # Fit standard scaler (use independent scaling parameters for each neuron channel)
    if apply_standard_scaler_before_pca:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_stack)
    else:
        scaler = None
        X_scaled = X_stack

    # Fit PCA per spec (integer components or variance-retention target)
    # Note scikit-learn's PCA implementation automatically handles variance-retention target by selecting the minimal k such that cumulative explained variance >= target
    pca = PCA(
        n_components=pca_dim_or_var_retained, 
        svd_solver='full' if isinstance(pca_dim_or_var_retained, float) else 'auto'
    )
    pca.fit(X_scaled)
    print(
        f"[INFO] Fitted PCA with {pca.n_components_} components and explained variance of "
        f"{pca.explained_variance_ratio_.sum():.2f}"
    )

    # Return scaler and PCA objects for transforming other data splits
    return scaler, pca


# ---------------------------------------------------------------------------
# Helper utilities for day selection and stratified folds
# ---------------------------------------------------------------------------

def _normalize_day_indices(
    day_index_field: Optional[Union[int, Sequence[int]]],
) -> List[int]:
    """
    Normalize the macaque day selection into a deduplicated list of indices.
    """
    if day_index_field is None:
        return []
    if isinstance(day_index_field, (int, np.integer)):
        return [int(day_index_field)]
    if isinstance(day_index_field, str):
        return [int(day_index_field)]
    if isinstance(day_index_field, (list, tuple)):
        normalized: List[int] = []
        seen: set[int] = set()
        for raw_value in day_index_field:
            if raw_value is None:
                continue
            if isinstance(raw_value, (int, np.integer)):
                idx = int(raw_value)
            else:
                idx = int(raw_value)
            if idx in seen:
                continue
            seen.add(idx)
            normalized.append(idx)
        return normalized
    raise ValueError(
        "macaque_day_index must be an int or a sequence of ints."
    )


def _format_day_label(
    day_indices: List[int],
) -> Union[int, str]:
    """
    Create a stable key that represents the selected day(s).
    """
    if not day_indices:
        raise ValueError("At least one day index must be provided.")
    if len(day_indices) == 1:
        return day_indices[0]
    joined = "_".join(str(idx) for idx in sorted(day_indices))
    return f"days_{joined}"


def _collect_samples_for_days(
    grouped_samples: List[List[Dict[str, Any]]],
    day_indices: List[int],
) -> List[Dict[str, Any]]:
    """
    Gather and concatenate per-day samples according to the requested day indices.
    """
    if not day_indices:
        raise ValueError("At least one day index must be provided to collect samples.")
    if not grouped_samples:
        raise ValueError("Grouped samples list is empty; nothing to collect.")
    max_idx = len(grouped_samples)
    combined: List[Dict[str, Any]] = []
    for idx in day_indices:
        if idx < 0 or idx >= max_idx:
            raise ValueError(f"Day index {idx} is out of bounds for available days (0-{max_idx - 1}).")
        day_samples = grouped_samples[idx]
        if not day_samples:
            raise ValueError(f"No samples found for requested day index {idx}.")
        combined.extend(day_samples)
    return combined


def _condition_to_index(
    condition: Optional[str],
) -> int:
    """
    Convert a condition string into its canonical integer index for stratification.
    """
    if isinstance(condition, str) and (condition in CONDITIONS):
        return CONDITIONS.index(condition)
    return -1


def _build_stratified_folds(
    samples: List[Dict[str, Any]],
    k_folds: int,
    seed: int,
) -> List[List[int]]:
    """
    Build stratified folds over the combined day+condition strata.
    """
    if k_folds < 3:
        raise ValueError("k_folds must be at least 3 to allocate train/valid/test splits.")
    if len(samples) < k_folds:
        raise ValueError(
            f"Cannot create {k_folds} folds from only {len(samples)} samples."
        )
    day_mapping: Dict[Any, int] = {}
    day_labels: List[int] = []
    condition_labels: List[int] = []
    for sample in samples:
        day_value = sample.get("day")
        if day_value not in day_mapping:
            day_mapping[day_value] = len(day_mapping)
        day_labels.append(day_mapping[day_value])
        condition_labels.append(_condition_to_index(sample.get("condition")))
    folds_raw = multi_strat_multi_fold_idx_sample(
        strata_label_ll=[day_labels, condition_labels],
        n_folds=k_folds,
        seed=seed,
        return_np_arrays_l=False,
    )
    folds: List[List[int]] = [[int(idx) for idx in fold] for fold in folds_raw]
    return folds


def _resolve_fold_split_indices(
    folds: List[List[int]],
    fold_i: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Convert stratified folds into train/valid/test index lists.
    """
    num_folds = len(folds)
    if fold_i < 0 or fold_i >= num_folds:
        raise ValueError(f"fold_i must be in [0, {num_folds - 1}].")
    valid_idx_pos = (fold_i + 1) % num_folds
    test_idx = list(folds[fold_i])
    valid_idx = list(folds[valid_idx_pos])
    train_idx: List[int] = []
    for idx, fold in enumerate(folds):
        if idx in (fold_i, valid_idx_pos):
            continue
        train_idx.extend(fold)
    if not train_idx:
        raise ValueError("Train split is empty; please check the stratified folds.")
    return train_idx, valid_idx, test_idx


def _prepare_fold_indices(
    samples: List[Dict[str, Any]],
    k_folds: int,
    fold_i: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Convenience helper that builds stratified folds and returns split indices.
    """
    folds = _build_stratified_folds(samples, k_folds, seed)
    return _resolve_fold_split_indices(folds, fold_i)


def process_by_day_with_splits(
    merged_data: List[Dict[str, Any]],
    seed: int = DEFAULT_SPLIT_SEED,
    mode: Literal["transductive", "inductive"] = "transductive",
    transform_num_workers: Optional[int] = None,
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
    savgol_filter_kwargs: Optional[Dict[str, Any]] = SAVGOL_FILTER_KWARGS,
) -> Dict[Any, Dict[str, List[Dict[str, Any]]]]:
    """
    End-to-end processing by day with 80/10/10 splits and train-only PCA fitting (in 'inductive' mode).

    Steps:
    - Filter each trial (if savgol_filter_kwargs provided), slice after GO_CUE.
    - Optionally fit per-day PCA on TRAIN-only (inductive) or ALL (transductive), with components chosen by GLOBAL_PCA_VARIANCE_RETAINED in (0,1) or GLOBAL_PCA_REDUCED_DIM.
    - Transform to neural state x[t] (or keep original if PCA disabled).
    - Compute neural velocity v[t] = x[t+1] - x[t] and store it under SPIKE_DATA_ATTRIBUTE; store x[t] under NEURAL_STATE_ATTRIBUTE.

    Returns a dict mapping day -> { 'train'|'valid'|'test': list of samples },
    where each sample contains NEURAL_STATE_ATTRIBUTE (k_state, T_state_after_go) and
    SPIKE_DATA_ATTRIBUTE holding velocities (k_state, T_state_after_go-1).
    """
    # Group samples by day
    day_to_samples: Dict[Any, List[Dict[str, Any]]] = {}
    for sample in merged_data:
        day = sample["day"]
        day_to_samples.setdefault(day, []).append(sample)

    rng = default_rng(seed=seed)
    outputs: Dict[Any, Dict[str, List[Dict[str, Any]]]] = {}

    for day, samples in day_to_samples.items():
        # Per-day shuffle and split
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        samples_shuffled = [samples[i] for i in idx]

        train_idx, val_idx, test_idx = _split_indices_per_day(
            samples_shuffled,
            seed=seed,
        )

        # Decide PCA usage and n_components
        pca_enabled = True
        pca_dim_or_var_retained = float(GLOBAL_PCA_VARIANCE_RETAINED)
        if (GLOBAL_PCA_REDUCED_DIM is not None):
            pca_dim_or_var_retained = int(GLOBAL_PCA_REDUCED_DIM)
        else:
            pca_enabled = False

        # Choose PCA fitting scope per mode; fit only if enabled
        scaler: Optional[StandardScaler] = None
        pca: Optional[PCA] = None
        if pca_enabled:
            if mode == "transductive":
                pca_train_idx = None
            else:
                pca_train_idx = train_idx

            # Fit scaler and PCA
            scaler, pca = _fit_scalar_and_pca_for_day(
                samples_shuffled,
                pca_train_idx,
                pca_dim_or_var_retained=pca_dim_or_var_retained,
                apply_savgol_filter_before_pca=apply_savgol_filter_before_pca,
                savgol_filter_kwargs=savgol_filter_kwargs
            )
            # Update global reduced dim to realized component count when variance specified
            try:
                realized = getattr(pca, 'n_components_', None)
                if isinstance(realized, int) and realized > 0:
                    globals()['GLOBAL_PCA_REDUCED_DIM'] = realized
            except Exception as e:
                print(f"Warning: could not update GLOBAL_PCA_REDUCED_DIM from fitted PCA: {e}")

        # Define inner helper to transform data)
        def _transform_one(i: int) -> Dict[str, Any]:
            samp = samples_shuffled[i]
            trial = samp[NEURAL_VELOCITY_ATTRIBUTE]

            # 1. Slice after GO_CUE
            trial_proc = _slice_spike_data(trial)

            # 2. Apply filtering if configured
            if apply_savgol_filter_before_pca:
                trial_proc = savgol_filter(
                    trial_proc,
                    window_length=savgol_filter_kwargs["window_length"],
                    polyorder=savgol_filter_kwargs["polyorder"],
                    axis=1,
                )

            # 3. Optional standard scaling and PCA transforms
            if pca_enabled and (pca is not None):
                X = trial_proc.T  # (t, n_neurons)
                X_scaled = scaler.transform(X) if scaler is not None else X
                X_scores = pca.transform(X_scaled)
                state_k_t = X_scores.T  # (k, t)
            else:
                state_k_t = trial_proc

            # 4. Neural velocity vectors assigned to time t
            vel_k_t = np.diff(state_k_t, n=1, axis=1)
            out = dict(samp)
            out[NEURAL_STATE_ATTRIBUTE] = state_k_t
            out[NEURAL_VELOCITY_ATTRIBUTE] = vel_k_t

            # 5. Return transformed sample
            return out

        # Define helper to apply _transform_one to data by split
        def transform_idx_list(indices: List[int]) -> List[Dict[str, Any]]:
            if (transform_num_workers is None) \
            or (transform_num_workers <= 1):
                return [_transform_one(i) for i in indices]
            with ThreadPoolExecutor(max_workers=transform_num_workers) as ex:
                return list(ex.map(_transform_one, indices))

        # Apply inner helper to create each split
        outputs[day] = {
            "train": transform_idx_list(train_idx),
            "valid": transform_idx_list(val_idx),
            "test": transform_idx_list(test_idx),
        }
    return outputs


def _build_path_graph_for_sample(
    sample: Dict[str, Any],
    spike_data_attribute: str = NEURAL_VELOCITY_ATTRIBUTE,
    task_level: str = 'node',
) -> Data:
    """
    Construct an undirected path graph for a single trajectory with:
    - Vector node features under SPIKE_DATA_ATTRIBUTE (t_nodes, k)
    - Scalar node features x as normalized time index t/(T-1)
    - Targets: pos_xy and vel_xy
      - If task_level='node': (t_nodes, 2) for each target
      - If task_level='graph': (1, 2) for each target (final node only)
    - Metadata: day, trial_id, condition_idx
      - condition_idx (torch.long): Integer class label (0-6) for one of 7 trial conditions.
        Used as target for multi-class classification tasks.
    """
    spike_k_t = sample[spike_data_attribute]  # (k, t_nodes)
    if not isinstance(spike_k_t, np.ndarray):
        raise ValueError("Expected spike_data to be numpy array")
    # k = spike_k_t.shape[0]
    t_nodes = spike_k_t.shape[1]
    X_vec = spike_k_t.T.astype(np.float32)  # (t_nodes, k)

    # Time scalar feature normalized to [0,1]
    if t_nodes <= 1:
        t_norm = np.zeros((t_nodes, 1), dtype=np.float32)
    else:
        t_idx = np.arange(t_nodes, dtype=np.float32)
        t_norm = (t_idx / float(t_nodes - 1)).reshape(-1, 1)

    # Targets: use kinematics (already sliced to GO_CUE + CUT_T_AFTER_GO_CUE)
    pos = sample.get("lever_pos")  # (2, T)
    vel = sample.get("lever_vel")  # (2, T) or None
    if pos is None:
        raise ValueError("lever_pos missing in sample for path graph")

    # Set targets based on task_level
    if task_level == 'graph':
        # Graph-level: use only the final timepoint's target
        # No need to align to all nodes, just extract last value
        pos_t = pos[:, -1:].T.astype(np.float32)  # (1, 2)
        if vel is not None:
            vel_t = vel[:, -1:].T.astype(np.float32)  # (1, 2)
    else:
        # Node-level: align targets to the number of feature nodes
        # NOTE: Kinematics are already sliced to match spike data (GO_CUE + CUT_T_AFTER_GO_CUE)
        pos_t = pos[:, :t_nodes].T.astype(np.float32)  # (t_nodes, 2)
        if vel is not None:
            vel_t = vel[:, :t_nodes].T.astype(np.float32)  # (t_nodes, 2)
    
    t_nodes = X_vec.shape[0]

    # Path edges (undirected)
    if t_nodes > 1:
        src = np.arange(t_nodes - 1, dtype=np.int64)
        dst = src + 1
        edge_index = np.vstack(
            [np.concatenate([src, dst]), np.concatenate([dst, src])]
        )
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    cond = sample.get("condition")
    cond_idx = CONDITIONS.index(cond) \
        if (isinstance(cond, str) and (cond in CONDITIONS)) else -1
    
    data = Data(
        edge_index=torch.from_numpy(edge_index),
        x=torch.from_numpy(t_norm),
        num_nodes=t_nodes,
        pos_xy=torch.from_numpy(pos_t),
        vel_xy=torch.from_numpy(vel_t),
        day=sample.get("day"),
        trial_id=sample.get("trial_id"),
        condition_idx=torch.tensor(cond_idx, dtype=torch.long),
    )
    data[NEURAL_VELOCITY_ATTRIBUTE] = torch.from_numpy(X_vec)

    # Attach per-node neural state positions for NN O-frame assignment in inductive mode
    state_k_t = sample.get(NEURAL_STATE_ATTRIBUTE)
    if isinstance(state_k_t, np.ndarray) and state_k_t.ndim == 2:
        state_nodes = state_k_t[:, :t_nodes].T.astype(np.float32)
        data["state_pos"] = torch.from_numpy(state_nodes)
    return data


def _worker_build_day_paths(
    day: Any, 
    splits: Dict[str, List[Dict[str, Any]]],
    task_level: str = 'node',
) -> Tuple[Any, Dict[str, List[Data]]]:
    out = {
        sk: [
            _build_path_graph_for_sample(s, task_level=task_level) \
            for s in splits.get(sk, [])
        ]
        for sk in ("train", "valid", "test")
    }
    return day, out


def _worker_build_one_day_graph(
    day: Any,
    splits_dict_full: Dict[str, List[Dict[str, Any]]],
    mode: Literal["transductive", "inductive"],
    spatial_graph_dir: Optional[str],
    force_recompute: bool,
    n_neighbors: int,
    delta: float,
    metric: str,
    include_self: bool,
    reweight_with_median_kernel: bool,
    feature_key_for_day_graph: str,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = "threads",
    min_neighbors: Optional[int] = None,
) -> Tuple[Any, Data]:
    # Load existing if available
    if spatial_graph_dir is not None:
        pt_path = os.path.join(spatial_graph_dir, f"day_{day}.pt")
        if (not force_recompute) and os.path.exists(pt_path):
            return day, load_day_graph_pt(pt_path)
    # Choose splits dict for graph building
    splits_dict = (
        splits_dict_full 
        if (mode == "transductive") 
        else {"train": splits_dict_full.get("train", [])}
    )
    print(
        f"Building spatial day graph for day {day} with {len(splits_dict['train'])} "
        f"train samples"
    )
    dg = _build_spatial_day_graph(
        day,
        splits_dict,
        n_neighbors=n_neighbors,
        cknn_delta=delta,
        metric=metric,
        include_self=include_self,
        reweight_with_median_kernel=reweight_with_median_kernel,
        neural_state_attrib=feature_key_for_day_graph,
        min_neighbors=min_neighbors,
    )
    O_frames = _compute_day_O_frames(
        dg,
        vector_feat_key=NEURAL_STATE_ATTRIBUTE,
        target_dim=MANIFOLD_INTRINSIC_DIM,
        num_workers=num_workers,
        backend=backend,
    )
    if mode == "transductive":
        dg.O_frames = O_frames
        
    else:  # inductive mode
        train_mask = dg.train_mask.cpu().numpy().astype(bool)
        # Cache TRAIN neural-state positions (PCA-reduced spikes)
        train_pos = dg[NEURAL_STATE_ATTRIBUTE].cpu().numpy()[train_mask]
        O_train = O_frames[train_mask]

        # Cache train positions and O-frames for nearest-neighbor assignment in path graphs
        dg._train_pos_cache = torch.from_numpy(train_pos)
        dg._O_train_cache = O_train

        # O_frames on the day graph are for TRAIN nodes only; VAL/TEST nodes receive
        # nearest TRAIN O-frames later in _resolve_O_nodes_for_path_graph.
        dg.O_frames = O_frames

    # Save graph if requested
    if spatial_graph_dir is not None:
        save_day_graph_pt(dg, spatial_graph_dir, day)

    return day, dg


def _build_spatial_day_graph(
    day: Any,
    splits_dict: Dict[str, List[Dict[str, Any]]],
    n_neighbors: int = KNN_N_NEIGHBORS,
    cknn_delta: float = CKNN_DELTA,
    metric: str = 'euclidean',
    include_self: bool = False,
    reweight_with_median_kernel: bool = True,
    neural_state_attrib: str = NEURAL_STATE_ATTRIBUTE,
    min_neighbors: Optional[int] = None,
) -> Data:
    """
    Builds a single PyG Data graph for a day using spatial graph construction
    across all trials and timepoints (aligned to state nodes after GO_CUE).
    Node features are the (optionally PCA-reduced) neural states per timepoint.
    That is, neighborship in this graph is defined in the neural state feature space,
    NOT by velocity or time series data (as in the downstream path graphs).
    
    Graph construction mode is controlled by USE_CKNN_GRAPH global constant:
    - If True: Use continuous k-NN (CkNN) with delta parameter for adaptive neighborhoods
    - If False: Use standard k-NN with fixed neighborhood size (unweighted, undirected)
    """
    # Concatenate node features across splits and trials
    x_list: List[np.ndarray] = []  # (t_nodes_i, k)
    lever_pos_list: List[np.ndarray] = []  # (t_nodes_i, 2)
    lever_vel_list: List[np.ndarray] = []  # (t_nodes_i, 2)
    train_mask_list: List[np.ndarray] = []
    valid_mask_list: List[np.ndarray] = []
    test_mask_list: List[np.ndarray] = []
    node_trial_id_list: List[np.ndarray] = []  # (t_nodes_i,)
    node_trial_time_idx_list: List[np.ndarray] = []  # (t_nodes_i,)
    node_condition_idx_list: List[np.ndarray] = []  # (t_nodes_i,)
    trial_ids_run: List[int] = []  # one per trajectory
    trial_ptr: List[int] = [0]  # CSR-style pointer into concatenated nodes
    trial_split_labels: List[int] = []  # 0=train,1=val,2=test per trajectory

    for split_key in ("train", "valid", "test"):
        samples = splits_dict.get(split_key, [])
        for samp in samples:
            spike_k_t = samp[neural_state_attrib]  # (k, t_nodes)
            if not isinstance(spike_k_t, np.ndarray):
                raise ValueError("Expected feature array to be numpy array")
            lever_pos = samp.get("lever_pos")
            lever_vel = samp.get("lever_vel")
            
            # Align to neural state timeline after GO_CUE + CUT_T_AFTER_GO_CUE
            t_nodes = int(spike_k_t.shape[1])
            if t_nodes <= 0:
                continue
            x_list.append(spike_k_t[:, :t_nodes].T)

            # Kinematics already sliced to match spike data; align without additional slicing
            lever_pos_list.append(lever_pos[:, :t_nodes].T)
            lever_vel_list.append(lever_vel[:, :t_nodes].T)

            # Train/valid/test masks
            mask = np.ones((t_nodes,), dtype=bool)
            train_mask_list.append(
                mask if split_key == "train" else np.zeros_like(mask)
            )
            valid_mask_list.append(
                mask if split_key == "valid" else np.zeros_like(mask)
            )
            test_mask_list.append(
                mask if split_key == "test" else np.zeros_like(mask)
            )

            # node-level trial id and time index
            trial_id_val = samp.get("trial_id")
            try:
                trial_id_int = int(trial_id_val)
            except Exception:
                # fallback: enumerate as increasing ids if non-numeric
                trial_id_int = int(len(trial_ids_run))
            node_trial_id_list.append(np.full((t_nodes,), trial_id_int, dtype=np.int64))
            node_trial_time_idx_list.append(np.arange(t_nodes, dtype=np.int64))
            # condition index
            cond = samp.get("condition")
            cond_idx = CONDITIONS.index(cond) \
                if (isinstance(cond, str) and (cond in CONDITIONS)) else -1
            node_condition_idx_list.append(
                np.full((t_nodes,), cond_idx, dtype=np.int64)
            )

            # Trajectory-level bookkeeping
            trial_ids_run.append(trial_id_int)
            trial_ptr.append(trial_ptr[-1] + t_nodes)
            split_lbl = 0 if split_key == "train" \
                else (1 if split_key == "valid" else 2)
            trial_split_labels.append(split_lbl)

    if not x_list:
        raise ValueError(f"No nodes assembled for day {day}")

    X_feat = np.concatenate(x_list, axis=0)  # (N, k) PCA-reduced neural state
    lever_pos = np.concatenate(lever_pos_list, axis=0)  # (N, 2) lever positions (targets)
    lever_vel = np.concatenate(lever_vel_list, axis=0)  # (N, 2) lever velocities (targets)
    train_mask = np.concatenate(train_mask_list, axis=0)
    valid_mask = np.concatenate(valid_mask_list, axis=0)
    test_mask = np.concatenate(test_mask_list, axis=0)
    node_trial_id = np.concatenate(node_trial_id_list, axis=0)
    node_trial_time_idx = np.concatenate(node_trial_time_idx_list, axis=0)
    node_condition_idx = np.concatenate(node_condition_idx_list, axis=0)

    # Spatial graph construction: CkNN or standard k-NN
    num_nodes = X_feat.shape[0]
    
    def _ensure_min_degree(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        X_feat: np.ndarray,
        *,
        min_deg: int,
        include_self_flag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Guarantee each node has at least min_deg neighbors by adding k-NN edges if needed."""
        if min_deg <= 0:
            return edge_index, edge_weight
        num_nodes_local = X_feat.shape[0]
        deg = torch.bincount(edge_index[0], minlength=num_nodes_local)
        deficient = deg < min_deg
        if not bool(deficient.any()):
            return edge_index, edge_weight

        from sklearn.neighbors import NearestNeighbors

        k_knn = min_deg + (1 if include_self_flag else 0)
        knn = NearestNeighbors(
            n_neighbors=min(num_nodes_local, k_knn),
            metric=metric,
            algorithm='auto',
        )
        knn.fit(X_feat)
        distances, indices = knn.kneighbors(X_feat)

        edge_list = []
        weight_list = []
        for src in range(num_nodes_local):
            for j_idx, dst in enumerate(indices[src]):
                if (not include_self_flag) and (src == dst):
                    continue
                edge_list.append((src, int(dst), float(distances[src, j_idx])))
        if edge_list:
            edge_array = np.array(edge_list)
            add_ei = torch.from_numpy(edge_array[:, :2].T.astype(np.int64))
            add_ew = torch.from_numpy(edge_array[:, 2].astype(np.float32))
            edge_index = torch.cat([edge_index, add_ei], dim=1)
            edge_weight = torch.cat([edge_weight, add_ew], dim=0)
            edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='min')
        return edge_index, edge_weight

    if USE_CKNN_GRAPH:
        # Continuous k-NN with delta parameter
        edge_index_np, edge_weight_np = cknneighbors_graph(
            X_feat,
            n_neighbors=n_neighbors,
            delta=cknn_delta,
            metric=metric,
            t='inf',  # overridden by use_raw_distances=True
            include_self=include_self,
            is_sparse=True,
            use_raw_distances=True,
        )
        edge_index = torch.from_numpy(edge_index_np).long()
        edge_weight = torch.from_numpy(edge_weight_np).float()
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='min')
        if min_neighbors is not None and min_neighbors > 0:
            edge_index, edge_weight = _ensure_min_degree(
                edge_index,
                edge_weight,
                X_feat,
                min_deg=int(min_neighbors),
                include_self_flag=include_self,
            )
    else:
        # Standard k-NN graph (unweighted, undirected, symmetric)
        from sklearn.neighbors import NearestNeighbors
        
        # Build k-NN graph using sklearn
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + (1 if include_self else 0),
            metric=metric,
            algorithm='auto',
        )
        nbrs.fit(X_feat)
        distances, indices = nbrs.kneighbors(X_feat)
        
        # Convert to edge list format
        edge_list = []
        for i in range(num_nodes):
            for j_idx, j in enumerate(indices[i]):
                if not include_self and i == j:
                    continue
                edge_list.append((i, j, distances[i, j_idx]))
        
        if len(edge_list) > 0:
            edge_array = np.array(edge_list)
            edge_index_np = edge_array[:, :2].T.astype(np.int64)
            edge_weight_np = edge_array[:, 2].astype(np.float32)
            
            edge_index = torch.from_numpy(edge_index_np).long()
            edge_weight = torch.from_numpy(edge_weight_np).float()
            
            # Make undirected with symmetric edges
            edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='min')
            if min_neighbors is not None and min_neighbors > 0:
                edge_index, edge_weight = _ensure_min_degree(
                    edge_index,
                    edge_weight,
                    X_feat,
                    min_deg=int(min_neighbors),
                    include_self_flag=include_self,
                )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)

    # Create Data object for spatial graph
    data = Data()
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    data.num_nodes = num_nodes
    data.day = day

    # Node features
    # data.x = torch.from_numpy(X_feat).float()  # avoid redundant x field
    data[NEURAL_STATE_ATTRIBUTE] = torch.from_numpy(X_feat).float()
    data.pos_xy = torch.from_numpy(lever_pos).float()
    data.vel_xy = torch.from_numpy(lever_vel).float()

    # Node-to-trial linkages
    data.node_trial_id = torch.from_numpy(node_trial_id).long()
    data.node_trial_time_idx = torch.from_numpy(node_trial_time_idx).long()
    data.node_condition_idx = torch.from_numpy(node_condition_idx).long()

    # Masks
    data.train_mask = torch.from_numpy(train_mask)
    data.valid_mask = torch.from_numpy(valid_mask)
    data.test_mask = torch.from_numpy(test_mask)

    # Trajectory-level CSR-style mapping and split labels
    data.trial_ids = torch.as_tensor(trial_ids_run, dtype=torch.long)
    data.trial_ptr = torch.as_tensor(trial_ptr, dtype=torch.long)
    data.trial_split = torch.as_tensor(trial_split_labels, dtype=torch.long)

    # Median neighbor distance and optional kernel reweighting
    median_nbr_dist = compute_node_median_edge_weight(
        data.edge_index,
        data.edge_weight,
        num_nodes=num_nodes,
    )
    data.median_nbr_dist = median_nbr_dist
    if reweight_with_median_kernel:
        src, dst = data.edge_index
        sigma_src = median_nbr_dist[src]
        sigma_dst = median_nbr_dist[dst]
        denom = sigma_src * sigma_dst
        eps = 1e-12
        d2 = data.edge_weight * data.edge_weight
        kernel_w = torch.exp(-d2 / (denom + eps))
        data.edge_weight = kernel_w

    return data


def build_spatial_graphs_by_day(
    outputs_by_day: Dict[Any, Dict[str, List[Dict[str, Any]]]],
    days_included_idx: Optional[List[int]] = None,
    n_neighbors: int = KNN_N_NEIGHBORS,
    cknn_delta: float = CKNN_DELTA,
    metric: str = 'euclidean',
    include_self: bool = False,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    reweight_with_median_kernel: bool = True,
    min_neighbors: Optional[int] = None,
) -> Dict[Any, Data]:
    """
    Build a single spatial CkNN graph per day from processed splits.
    Returns a dict mapping day -> Data.
    """
    # Filter days
    items = list(outputs_by_day.items())
    if days_included_idx is not None:
        items = [items[i] for i in days_included_idx if 0 <= i < len(items)]

    graphs_by_day: Dict[Any, Data] = {}

    if not num_workers or num_workers <= 1:
        for day, splits_dict in items:
            g = _build_spatial_day_graph(
                day,
                splits_dict,
                n_neighbors=n_neighbors,
                cknn_delta=cknn_delta,
                metric=metric,
                include_self=include_self,
                reweight_with_median_kernel=reweight_with_median_kernel,
                min_neighbors=min_neighbors,
            )
            graphs_by_day[day] = g
        return graphs_by_day

    executor_cls = ProcessPoolExecutor \
        if backend == "processes" else ThreadPoolExecutor
    with executor_cls(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                _build_spatial_day_graph,
                day,
                splits_dict,
                n_neighbors,
                cknn_delta,
                metric,
                include_self,
                reweight_with_median_kernel,
                min_neighbors,
            )
            for day, splits_dict in items
        ]
        keys = [day for day, _ in items]
        for key, fut in zip(keys, futures):
            try:
                graphs_by_day[key] = fut.result()
            except Exception as e:
                print(f"build_graphs_by_day worker failed for day {key}: {e}")
                pass

    return graphs_by_day


def _worker_compute_node_svd(
    args: Tuple[int, Optional[torch.Tensor], int, torch.dtype, torch.device],
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Worker function to compute SVD for a single node. Compatible with multiprocessing.
    
    Args:
        args: Tuple of (node_idx, C_i, D, dtype, device)
            - node_idx: index of the node
            - C_i: covariance matrix for node i, shape (D, n_i) or None
            - D: ambient dimension
            - dtype: data type for tensors
            - device: device for tensors
    
    Returns:
        Tuple of (node_idx, U_i, S_i_padded)
            - node_idx: index of the node (for correct ordering)
            - U_i: left singular vectors, shape (D, D)
            - S_i_padded: singular values padded to length D, shape (D,)
    """
    node_idx, C_i, D, dtype, device = args
    
    # if C_i is None or C_i.numel() == 0 or int(C_i.shape[1]) == 0:
    #     # Fallback for isolated nodes: identity frame, zero singular values
    #     I_D = torch.eye(D, dtype=dtype, device=device)
    #     return node_idx, I_D, torch.zeros((D,), dtype=dtype, device=device)
    
    C_i = C_i.to(dtype=dtype)
    # SVD on (D x n_i); request full matrices to always get U of shape (D x D)
    U_i, S_i, Vt_i = torch.linalg.svd(C_i, full_matrices=True)
    
    # Pad S to length D for intrinsic-dim estimation consistency
    s_pad = torch.zeros((D,), dtype=S_i.dtype, device=S_i.device)
    s_pad[: min(D, S_i.shape[0])] = S_i[: min(D, S_i.shape[0])]
    
    return node_idx, U_i, s_pad


def _compute_day_O_frames(
    day_graph: Data,
    vector_feat_key: str = NEURAL_STATE_ATTRIBUTE,
    use_mean_recentering: bool = False,
    use_precomputed_edge_weights: bool = True,
    target_dim: Optional[Union[int, str]] = None,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = "threads",
) -> torch.Tensor:
    """
    Compute O_i local frames (left singular vectors) for each node in a day's spatial graph.
    Returns tensor of shape (N, D, d), where D = ambient vector feature dimension, d = manifold intrinsic dimension.
    
    Args:
        day_graph: PyG Data object for the day
        vector_feat_key: key for vector features in day_graph
        use_mean_recentering: whether to use mean recentering in C_i computation
        use_precomputed_edge_weights: whether to use precomputed edge weights
        target_dim: target intrinsic dimension (int, "participation_ratio", or "spectral_entropy")
        num_workers: number of workers for parallel SVD computation. If None or <=1, runs serially.
        backend: "processes" or "threads" for parallel execution
    
    Returns:
        O_stack: tensor of shape (N, D, d) containing O_i frames for each node
    """
    v = day_graph[vector_feat_key]
    D = int(v.shape[1])
    num_nodes = int(day_graph.num_nodes)
    
    C_i_dict = get_C_i_dict(
        day_graph,
        geom_feat_key=vector_feat_key,
        use_mean_recentering=use_mean_recentering,
        kernel_fn_kwargs={},
        use_precomputed_edge_weights=use_precomputed_edge_weights,
    )

    # Compute per-node SVDs directly on C_i (D x n_i) to construct O_i from left singular vectors
    U_list: List[torch.Tensor] = [None] * num_nodes  # pre-allocate list
    S_list: List[torch.Tensor] = [None] * num_nodes

    if (num_workers is None) or (num_workers <= 1):
        # Serial execution
        for i in range(num_nodes):
            _, U_i, S_i_padded = _worker_compute_node_svd(
                (i, C_i_dict.get(i), D, v.dtype, v.device)
            )
            U_list[i] = U_i
            S_list[i] = S_i_padded
    else:
        # Parallel execution
        executor_cls = ProcessPoolExecutor if backend == "processes" \
            else ThreadPoolExecutor
        
        # Prepare arguments for all nodes
        args_list = [
            (i, C_i_dict.get(i), D, v.dtype, v.device)
            for i in range(num_nodes)
        ]
        
        with executor_cls(max_workers=num_workers) as ex:
            # Submit all SVD tasks
            futures = [ex.submit(_worker_compute_node_svd, args) for args in args_list]
            
            # Collect results
            for fut in futures:
                try:
                    node_idx, U_i, S_i_padded = fut.result()
                    U_list[node_idx] = U_i
                    S_list[node_idx] = S_i_padded
                except Exception as e:
                    print(f"SVD computation failed for a node: {e}")
                    # Fallback will be handled by checking for None values below

        # Handle any failed computations
        # I_D = torch.eye(D, dtype=v.dtype, device=v.device)
        # for i in range(num_nodes):
        #     if U_list[i] is None:
        #         U_list[i] = I_D
        #         S_list[i] = torch.zeros((D,), dtype=v.dtype, device=v.device)

    # Stack singular values to (N, D) for intrinsic dimension estimation if requested
    S_stack = torch.stack(S_list, dim=0)  # (N, D)

    # Decide effective intrinsic dimension
    if isinstance(target_dim, str) and (target_dim in ("participation_ratio", "spectral_entropy")):
        d = _estimate_intrinsic_dim(S_stack, method=target_dim)
    else:
        d = D if target_dim is None else int(max(1, min(D, int(target_dim))))

    # Stack O_i = U_i[:, :d] across nodes -> (N, D, d)
    O_list = [U_i[:, :d].contiguous() for U_i in U_list]
    O_stack = torch.stack(O_list, dim=0)
    return O_stack


def _estimate_intrinsic_dim(
    S: torch.Tensor,
    method: Literal[
        "participation_ratio", 
        "spectral_entropy"
    ] = "spectral_entropy",
):
    """
    Args:
        S: (N,D) tensor of singular values from the SVD of the covariance matrix C_i for each node i.
        method: method to estimate the intrinsic dimension.
    Returns:
        d_eff: integer intrinsic dimension.
    
    Method notes:
    - Participation ratio / spectral entropy: The formula is essentially comparing the sum of squares (denominator) to the square of sums (numerator). If the values are highly variable (concentrated), the sum of squares grows faster than the square of sums (when normalized by a constant). By taking their ratio, the formula penalizes uneven distributions. The denominator's sensitivity to large values causes the overall ratio to decrease when the variance is concentrated in a few components, and to increase when it is spread out.
    """
    if method in ("participation_ratio", "spectral_entropy"):
        var = S ** 2  # (N,D)
        d_eff_local = (var.sum(dim=1) ** 2) / (var ** 2).sum(dim=1)  # unrounded float per point, shape (N,)
        d_eff = d_eff_local.mean().round().int()  # global avg integer dimension
    print(
        f"[INFO] macaque_reaching._estimate_intrinsic_dim: {method} "
        f"-> d={d_eff}"
    )
    return d_eff


def _assign_O_frames_by_nearest_nbr(
    train_pos: np.ndarray,
    train_O: torch.Tensor,
    query_pos: np.ndarray,
) -> torch.Tensor:
    """
    For each query position, copy the O-frame of its nearest train position.
        train_pos: (N_train, 2)
        train_O: (N_train, d, d)
        query_pos: (N_query, 2)
    Returns (N_query, d, d)
    """
    if train_pos.shape[0] == 0:
        raise ValueError(
            "No train positions provided for nearest-neighbor assignment"
        )
    tree = KDTree(train_pos)
    dists, idx = tree.query(query_pos, k=1, return_distance=True)
    idx = idx.reshape(-1)
    assigned = train_O[idx]
    return assigned


def _resolve_O_nodes_for_path_graph(
    path_graph: Data,
    day_graph: Data,
    mode: Literal["transductive", "inductive"] = "transductive",
) -> torch.Tensor:
    """
    Resolve O_i per node for a given path graph using the day's O_frames.
    Returns a tensor of shape (N, D, D), where D is the day feature dimension.
    """
    N = int(path_graph.num_nodes)
    if mode == "transductive":
        trial_id = int(path_graph.trial_id)
        j = int(
            (day_graph.trial_ids == trial_id) \
                .nonzero(as_tuple=True)[0][0] \
                .item()
        )
        start = int(day_graph.trial_ptr[j].item())
        O_nodes = day_graph.O_frames[start:start + N]
    elif mode == "inductive":
        # Use neural state positions per node as query points
        X_query = path_graph["state_pos"].numpy()

        # Retrieve cached TRAIN neural states and corresponding O-frames
        train_pos_cache = getattr(day_graph, '_train_pos_cache', None)
        if train_pos_cache is None:
            train_mask = day_graph.train_mask.cpu().numpy().astype(bool)
            train_pos_cache = day_graph[NEURAL_STATE_ATTRIBUTE][train_mask]
        train_pos = train_pos_cache.cpu().numpy()

        # Retrieve cached TRAIN O-frames; reconstruct if absent
        O_train = getattr(day_graph, '_O_train_cache', None)
        if O_train is None:
            train_mask = day_graph.train_mask.cpu().numpy().astype(bool)
            O_frames = getattr(day_graph, 'O_frames', None)
            if O_frames is None:
                raise ValueError("day_graph.O_frames missing; cannot assign O-frames in inductive mode")
            O_train = O_frames[train_mask]
        assigned = _assign_O_frames_by_nearest_nbr(
            train_pos, 
            O_train, 
            X_query
        )
        O_nodes = assigned if isinstance(assigned, torch.Tensor) \
            else torch.from_numpy(assigned)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return O_nodes


def _project_path_graph_feat_to_d(
    path_graph: Data,
    day_graph: Data,
    mode: Literal["transductive", "inductive"] = "transductive",
    vec_feat_to_project: str = NEURAL_VELOCITY_ATTRIBUTE,
) -> Data:
    """
    Projects D-dimensional vector features of all nodes in one path graph to d-dimensional intrinsic space via O-frames. Replaces the original features with the projected features.

    Args:
        path_graph: Data object for one path graph
        day_graph: Data object for the day graph
        mode: mode of graph building
        vec_feat_to_project: attribute name for the vector features
    Returns:
        path_graph: Data object for one path graph with projected features
    """
    Xn = path_graph[vec_feat_to_project]
    if not isinstance(Xn, torch.Tensor):
        Xn = torch.as_tensor(Xn)

    # In 'inductive' mode, we assign O-frames for valid and test nodes by nearest-neighbor train nodes
    O_nodes = _resolve_O_nodes_for_path_graph(path_graph, day_graph, mode)  # (N,D,d_eff)
    # d = int(O_nodes.shape[-1])
    # if d >= int(Xn.shape[1]):
    #     return path_graph
    if Xn.device != O_nodes.device:
        Xn = Xn.to(O_nodes.device)

    # Use batch matrix mult. to project D-dim. node features to d-dim.
    # Y = torch.bmm(O_nodes.transpose(1, 2), Xn.unsqueeze(2)).squeeze(2)  # (N,d)
    Xn_proj = torch.einsum('NDd,ND->Nd', O_nodes, Xn)  # (N,d)
    path_graph[vec_feat_to_project] = Xn_proj
    return path_graph


def save_day_graph_pt(
    day_graph: Data, 
    save_dir: str, 
    day: Any, 
    use_float16: bool = False
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"day_{day}.pt")
    pack = {
        "edge_index": day_graph.edge_index.cpu().to(torch.int64),
        "edge_weight": day_graph.edge_weight.cpu().to(torch.float16 if use_float16 else torch.float32),
        NEURAL_STATE_ATTRIBUTE: day_graph[NEURAL_STATE_ATTRIBUTE].cpu().to(torch.float16 if use_float16 else torch.float32),
        "train_mask": day_graph.train_mask.cpu(),
        "valid_mask": day_graph.valid_mask.cpu(),
        "test_mask": day_graph.test_mask.cpu(),
        "trial_ids": day_graph.trial_ids.cpu(),
        "trial_ptr": day_graph.trial_ptr.cpu(),
        "O_frames": getattr(day_graph, "O_frames", None).cpu() if hasattr(day_graph, "O_frames") else None,
        "num_nodes": int(day_graph.num_nodes),
        "day": day,
    }
    # Optional fields
    if hasattr(day_graph, 'pos_xy'):
        pack["pos_xy"] = day_graph.pos_xy.cpu().to(torch.float32)
    if hasattr(day_graph, 'vel_xy'):
        pack["vel_xy"] = day_graph.vel_xy.cpu().to(torch.float32)
    torch.save(pack, path)
    return path


def load_day_graph_pt(path: str) -> Data:
    pack = torch.load(path, weights_only=False)
    data = Data()
    data.edge_index = pack["edge_index"]
    data.edge_weight = pack["edge_weight"]
    data[NEURAL_STATE_ATTRIBUTE] = pack[NEURAL_STATE_ATTRIBUTE]
    data.train_mask = pack["train_mask"]
    data.valid_mask = pack["valid_mask"]
    data.test_mask = pack["test_mask"]
    data.trial_ids = pack["trial_ids"]
    data.trial_ptr = pack["trial_ptr"]
    if pack.get("O_frames") is not None:
        data.O_frames = pack["O_frames"]
    if "pos_xy" in pack:
        data.pos_xy = pack["pos_xy"]
    if "vel_xy" in pack:
        data.vel_xy = pack["vel_xy"]
    data.num_nodes = pack.get("num_nodes")
    data.day = pack.get("day")
    return data


def macaque_build_spatial_graphs_with_O(
    outputs_by_day: Dict[int, Dict[str, List[Dict[str, Any]]]],
    days_included_idx: Optional[List[int]] = None,
    mode: Literal["transductive", "inductive"] = "transductive",
    spatial_graph_dir: Optional[str] = None,
    force_recompute: bool = False,
    n_neighbors: int = KNN_N_NEIGHBORS,
    cknn_delta: float = CKNN_DELTA,
    metric: str = 'euclidean',
    include_self: bool = False,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    reweight_with_median_kernel: bool = True,
    feature_key_for_day_graph: str = NEURAL_STATE_ATTRIBUTE,
    min_neighbors: Optional[int] = None,
) -> Dict[Any, Data]:
    """
    Build or load per-day spatial graphs and compute O_i frames.
    - Transductive: graph over all nodes; compute O for all nodes.
    - Inductive: graph over train nodes only; compute O for train; assign val/test by NN.
    If spatial_graph_dir is provided, save/load .pt per day.

    Args:
        outputs_by_day: output of process_by_day_with_splits
        days_included_idx: list of day indices (dictionary keys) to include from outputs_by_day.
            If None, all days in outputs_by_day are included. Example: [0, 5, 31] will include
            only days 0, 5, and 31 from outputs_by_day.
        mode: mode of graph building
        spatial_graph_dir: directory to save/load spatial graphs
        force_recompute: whether to force recompute spatial graphs
        n_neighbors: number of neighbors for CkNN
        cknn_delta: delta for CkNN
        metric: metric for CkNN
        include_self: whether to include self-loops in the graph
        num_workers: number of workers for parallel SVD computation within each day.
            If None or <=1, SVD is computed serially.
        backend: backend for parallel SVD execution ("processes" or "threads")
        reweight_with_median_kernel: whether to reweight the graph with the median kernel
        feature_key_for_day_graph: key for the feature to use for the day graph
    Returns:
        graphs_by_day: Dict[Any, Data]
        - Keys are day indices
        - Values are Data objects representing the spatial graphs for the day
        
    Note:
        Days are processed serially, but within each day, per-node SVD computation
        can be parallelized using num_workers and backend.
    """
    # Optionally subset days by dictionary keys
    if days_included_idx is not None:
        # Filter by day indices (dictionary keys), not positional indices
        items = [(day, splits) for day, splits in outputs_by_day.items() 
                 if day in days_included_idx]
    else:
        items = list(outputs_by_day.items())

    graphs_by_day: Dict[Any, Data] = {}
    print(f"Building spatial graphs with O_i for {len(items)} days")
    
    # Loop through days serially; parallelization happens within each day for SVD computation
    for day, splits_dict_full in items:
        k, v = _worker_build_one_day_graph(
            day,
            splits_dict_full,
            mode,
            spatial_graph_dir,
            force_recompute,
            n_neighbors,
            cknn_delta,
            metric,
            include_self,
            reweight_with_median_kernel,
            feature_key_for_day_graph,
            num_workers=num_workers,
            backend=backend,
            min_neighbors=min_neighbors,
        )
        graphs_by_day[k] = v

    return graphs_by_day


def _worker_compute_Q_blocks_for_edge(
    args: Tuple[int, int, torch.Tensor, torch.Tensor, int, str, torch.Tensor, bool]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """
    Compute Q blocks for one unique edge (i < j).
    
    Args:
        args: (edge_idx, i, j, p_i, p_j, O_frames, d, geometric_mode, root_indices)
            - i, j: node indices (i < j for unique edges)
            - O_frames: (N, D, d) tensor
            - d: intrinsic dimension
            - geometric_mode: 'equivariant' or 'invariant'
            - root_indices: (2, d²) pre-computed block pattern
    
    Returns:
        Tuple of (blocks_idx, blocks_vals, error_msg)
            - blocks_idx: (2, 2*d²) indices for both i->j and j->i blocks (None if error)
            - blocks_vals: (2*d²,) values for both blocks (None if error)
            - error_msg: None if success, error string if failed
    """
    i, j, p_i, p_j, O_frames, d, geometric_mode, root, apply_p_weights = args
    
    try:
        # Extract O-frames for both nodes
        O_i = O_frames[i][:, :d]  # (D, d)
        O_j = O_frames[j][:, :d]  # (D, d)
        
        # Align O_j to O_i once
        O_j_aligned = match_B_col_directions_to_A(O_i, O_j)
        
        # Compute both blocks using the single alignment
        if geometric_mode == 'equivariant':
            O_ij = torch.mm(O_i, O_j_aligned.T)  # Q[i,j] block (d, d)
            O_ji = torch.mm(O_j_aligned, O_i.T)  # Q[j,i] block (d, d)
        else:  # 'invariant'
            O_ij = torch.mm(O_i.T, O_j_aligned)  # Q[i,j] block (d, d)
            O_ji = torch.mm(O_j_aligned.T, O_i)  # Q[j,i] block (d, d)

        # Apply P row-normalization weights to both blocks (if requested)
        if apply_p_weights:
            O_ij = p_i * O_ij
            O_ji = p_j * O_ji
        
        # Generate indices for both blocks
        offset_i = d * i
        offset_j = d * j
        
        # Block i->j at [d*i : d*(i+1), d*j : d*(j+1)]
        idx_ij = torch.stack([root[0] + offset_i, root[1] + offset_j])
        
        # Block j->i at [d*j : d*(j+1), d*i : d*(i+1)]
        idx_ji = torch.stack([root[0] + offset_j, root[1] + offset_i])
        
        # Concatenate both blocks
        blocks_idx = torch.cat([idx_ij, idx_ji], dim=1)  # (2, 2*d²)
        blocks_vals = torch.cat([O_ij.reshape(-1), O_ji.reshape(-1)])  # (2*d²,)
        
        return blocks_idx, blocks_vals, None
        
    except Exception as e:
        return None, None, str(e)


def _build_Q_for_path_graph(
    path_graph: Data,
    day_graph: Data,
    mode: Literal["transductive", "inductive"] = "transductive",
    spike_data_attribute: str = NEURAL_VELOCITY_ATTRIBUTE,
    geometric_mode: Literal['equivariant', 'invariant'] = GEOMETRIC_MODE,
    apply_p_weights: bool = True,
) -> Dict[str, Any]:
    """
    Build Q sparse tensor components for a trajectory path graph using day O-frames.
    Returns dict with keys: indices (2,E*d^2), values (E*d^2,), size (2,).
    
    Args:
        path_graph: Path graph for one trajectory
        day_graph: Day graph with O_frames
        mode: 'transductive' or 'inductive'
        spike_data_attribute: Key for vector features
        geometric_mode: 'equivariant' (requires D=d) or 'invariant' (allows D>d)
    
    Raises:
        ValueError: If geometric_mode='equivariant' but O-frames are not square (D≠d)
    """
    d = int(path_graph[spike_data_attribute].shape[1])
    N = int(path_graph.num_nodes)
    Q_size = (d * N, d * N)
    # Resolve O_i per node in path graph
    O_nodes = _resolve_O_nodes_for_path_graph(path_graph, day_graph, mode)
    
    # Validate O-frame shapes for equivariant mode
    if geometric_mode == 'equivariant':
        D = int(O_nodes.shape[1])  # ambient dimension
        if D != d:
            raise ValueError(
                f"Equivariant mode requires O-frames to be square (D=d), but got "
                f"D={D}, d={d}. Set GLOBAL_PCA_REDUCED_DIM and MANIFOLD_INTRINSIC_DIM to the same value "
                f"to use equivariant mode."
            )

    # Build Q block-sparse in the current feature dim d
    ei = path_graph.edge_index
    # if ei.numel() == 0:
    #     return {
    #         "indices": torch.empty((2, 0), dtype=torch.int64).numpy(),
    #         "values": torch.empty((0,), dtype=torch.float32).numpy(),
    #         "size": np.array(Q_size, dtype=np.int64),
    #     }

    root = get_square_matrix_coo_indices(d)  # (2, d²)
    blocks_idx = []
    blocks_vals = []
    
    # Compute node degrees for row normalization
    ei, _ = remove_self_loops(ei)
    node_degrees = degree(ei[0], num_nodes=N, dtype=torch.float32)
    
    # Extract unique edges (i < j) to avoid duplicate work
    mask = ei[0] < ei[1]  # no self-loops (i == j)
    unique_edges = ei[:, mask]  # (2, E/2)
    
    for e in range(unique_edges.shape[1]):
        i = int(unique_edges[0, e].item())
        j = int(unique_edges[1, e].item())
        
        # Use only the first d columns (intrinsic/tangent dim) and align once
        O_i = O_nodes[i][:, :d]
        O_j = O_nodes[j][:, :d]
        O_j_aligned = match_B_col_directions_to_A(O_i, O_j)
        
        # Compute both blocks using single alignment
        if geometric_mode == 'equivariant':
            O_ij = torch.mm(O_i, O_j_aligned.T)  # (d,d)
            O_ji = torch.mm(O_j_aligned, O_i.T)  # (d,d)
        else:
            O_ij = torch.mm(O_i.T, O_j_aligned)  # (d,d)
            O_ji = torch.mm(O_j_aligned.T, O_i)  # (d,d)
        
        # Row-normalize off-diagonal blocks: 
        # Q[i,j] = 0.5 * (1/d_i) * O_ij, where d_i is the degree of node i
        d_i = node_degrees[i].item()
        d_j = node_degrees[j].item()

        # Compute off-diagonal row-normalization weights, avoiding division by zero
        if apply_p_weights:
            norm_i = 0.5 / d_i if d_i > 0 else 0.0
            norm_j = 0.5 / d_j if d_j > 0 else 0.0
        else:
            norm_i = 1.0
            norm_j = 1.0
        
        O_ij_normalized = O_ij * norm_i
        O_ji_normalized = O_ji * norm_j
        
        # Generate indices for both blocks
        offset_i = d * i
        offset_j = d * j
        
        # Block i->j
        idx_ij = torch.stack([root[0] + offset_i, root[1] + offset_j])
        blocks_idx.append(idx_ij)
        blocks_vals.append(O_ij_normalized.reshape(-1))
        
        # Block j->i
        idx_ji = torch.stack([root[0] + offset_j, root[1] + offset_i])
        blocks_idx.append(idx_ji)
        blocks_vals.append(O_ji_normalized.reshape(-1))
    
    # Add self-loops (0.5 * I_d for lazy random walk)
    # (We add them manually since O_i @ O_i.T = I_d when O_i is orthonormal and square [equivar.],
    # and O_i.T @ O_i = I_d when O_i is D x d, cols are orthonormal, and D > d [invariant])
    diag_scale = 0.5 if apply_p_weights else 1.0
    I_d = torch.eye(d, dtype=O_nodes.dtype, device=O_nodes.device) * diag_scale
    for i in range(N):
        offset_i = d * i
        idx_ii = torch.stack([root[0] + offset_i, root[1] + offset_i])
        blocks_idx.append(idx_ii)
        blocks_vals.append(I_d.reshape(-1))

    Q_vals = torch.cat(blocks_vals)
    Q_idx = torch.cat(blocks_idx, dim=1)

    return {
        "indices": Q_idx.cpu().to(torch.int64).numpy(),
        "values": Q_vals.cpu().to(torch.float32).numpy(),
        "size": np.array(Q_size, dtype=np.int64),
    }


def _build_Q_for_spatial_graph_parallel(
    spatial_graph: Data,
    geometric_mode: Literal['equivariant', 'invariant'] = GEOMETRIC_MODE,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = "threads",
    apply_p_weights: bool = True,
) -> torch.sparse_coo_tensor:
    """
    Build Q operator for spatial graph with parallelization and validation.
    
    Args:
        spatial_graph: Spatial graph with O_frames attribute
        geometric_mode: 'equivariant' (requires D=d) or 'invariant' (allows D>d)
        num_workers: Number of workers for parallel processing (None or <=1 for serial)
        backend: 'processes' or 'threads' for parallel execution
    
    Returns:
        Q: Sparse COO tensor of shape (d*N, d*N).
        When ``apply_p_weights`` is False, off-diagonal blocks contain only the
        unweighted O-frame products and diagonal blocks are identity matrices.
    
    Validates:
        - O-frame shape compatibility with geometric_mode
        - Symmetric block structure (block at (i,j) implies block at (j,i))
        - Diagonal blocks present for all nodes
        - Reports any failed edge computations
    """
    # Extract graph properties
    O_frames = spatial_graph.O_frames  # (N, D, d)
    edge_index = spatial_graph.edge_index
    num_nodes = int(spatial_graph.num_nodes)
    d = int(O_frames.shape[-1])
    D = int(O_frames.shape[1])
    
    # Validate O-frame shapes for equivariant mode
    if geometric_mode == 'equivariant':
        if D != d:
            raise ValueError(
                f"Equivariant mode requires O-frames to be square (D=d), but got "
                f"D={D}, d={d}. Set GLOBAL_PCA_REDUCED_DIM and MANIFOLD_INTRINSIC_DIM "
                f"to the same value."
            )
    
    Q_size = (d * num_nodes, d * num_nodes)
    
    # Pre-compute root indices for block pattern
    root = get_square_matrix_coo_indices(d)  # (2, d²)
    
    # Compute node degrees for row normalization
    ei, _ = remove_self_loops(edge_index)
    node_degrees = degree(ei[0], num_nodes=num_nodes, dtype=torch.float32)
    
    # Extract unique edges (i < j) to avoid duplicate work
    mask = edge_index[0] < edge_index[1]
    unique_edges = edge_index[:, mask]  # (2, E/2)
    
    print(
        f"  Building Q with {unique_edges.shape[1]} unique edges, {num_nodes} nodes "
        f"(vec feat dim d={d}), using {num_workers} CPU cores."
    )

    def get_Q_off_diag_block_row_norm_weight(
        node_degree: torch.Tensor
    ) -> torch.Tensor:
        if not apply_p_weights:
            return torch.tensor(1.0, dtype=torch.float32)
        return 0.5 / node_degree if node_degree > 0 else 0.0
    
    # Prepare arguments for workers
    args_list = [
        (
            int(edge[0].item()),  # i
            int(edge[1].item()),  # j
            get_Q_off_diag_block_row_norm_weight(node_degrees[edge[0].item()]),  # p_i
            get_Q_off_diag_block_row_norm_weight(node_degrees[edge[1].item()]),  # p_j
            O_frames, 
            d, 
            geometric_mode, 
            root,
            apply_p_weights
        ) \
        for edge in unique_edges.T
    ]
    
    # Process edges (parallel or serial)
    results = []
    failed_count = 0
    
    if num_workers and num_workers > 1:
        executor_cls = ProcessPoolExecutor if backend == "processes" else ThreadPoolExecutor
        with executor_cls(max_workers=num_workers) as ex:
            futures = [
                ex.submit(_worker_compute_Q_blocks_for_edge, args) for args in args_list
            ]
            for fut in futures:
                try:
                    blocks_idx, blocks_vals, error_msg = fut.result()
                    if error_msg is None:
                        results.append((blocks_idx, blocks_vals))
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"  Worker failed with exception: {e}")
                    failed_count += 1
    else:
        # Serial processing
        for args in args_list:
            blocks_idx, blocks_vals, error_msg = _worker_compute_Q_blocks_for_edge(args)
            if error_msg is None:
                results.append((blocks_idx, blocks_vals))
            else:
                failed_count += 1
    
    # Add self-loops (0.5 * I_d for lazy random walk)
    # (We add them manually since O_i @ O_i.T = I_d when O_i is orthonormal and square [equivariant mode],
    # and O_i.T @ O_i = I_d when O_i is D x d, cols are orthonormal, and D > d [invariant mode])
    diag_scale = 0.5 if apply_p_weights else 1.0
    I_d = torch.eye(d, dtype=O_frames.dtype, device=O_frames.device) * diag_scale
    for i in range(num_nodes):
        offset_i = d * i
        idx_ii = torch.stack([root[0] + offset_i, root[1] + offset_i])
        results.append((idx_ii, I_d.reshape(-1)))
    
    # Concatenate all blocks
    if len(results) == 0:
        raise ValueError("No edges found in spatial graph; Q cannot be built")
    all_idx = torch.cat([r[0] for r in results], dim=1)
    all_vals = torch.cat([r[1] for r in results])
    Q = torch.sparse_coo_tensor(
        indices=all_idx.to(torch.long),
        values=all_vals.to(torch.float32),
        size=Q_size
    )
    
    # Validation: check for symmetric block structure
    Q_coalesced = Q.coalesce()
    indices = Q_coalesced.indices()
    
    # Count blocks (not individual elements)
    # Each block has d² entries, so divide indices by d to get block coordinates
    block_i = indices[0] // d
    block_j = indices[1] // d
    block_pairs = torch.stack([block_i, block_j], dim=0)
    unique_blocks = torch.unique(block_pairs, dim=1)
    
    # Check diagonal blocks present
    diag_blocks = unique_blocks[:, unique_blocks[0] == unique_blocks[1]]
    num_diag_blocks = diag_blocks.shape[1]
    
    print(f"  Q construction complete: {unique_blocks.shape[1]} unique blocks, "
          f"{num_diag_blocks} diagonal blocks")
    
    if num_diag_blocks != num_nodes:
        print(f"  Warning: Expected {num_nodes} diagonal blocks, found {num_diag_blocks}")
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} edges failed during Q computation")
    
    return Q


def _compute_q_block_metadata(
    Q: torch.sparse_coo_tensor,
    edge_index: torch.Tensor,
    vector_dim: int,
    num_nodes: int,
) -> Dict[str, torch.Tensor]:
    """
    Derive block-level metadata for a block-sparse diffusion operator.

    Args:
        Q: Sparse COO tensor whose blocks follow the (i,j) node ordering.
        edge_index: Directed edge index tensor of shape (2, E).
        vector_dim: Dimension ``d`` of each vector feature (block height/width).
        num_nodes: Number of nodes in the graph.

    Returns:
        Dict with keys:
            - 'block_pairs': LongTensor of shape (2, n_blocks) with source/target
              node indices for each block.
            - 'block_edge_ids': LongTensor of shape (n_blocks,) mapping each block
              to the corresponding directed edge index (or -1 for diagonal blocks).
    """
    Q = Q.coalesce()
    if Q.layout != torch.sparse_coo:
        raise ValueError("Q must be a sparse COO tensor to compute block metadata.")
    block_size = int(vector_dim * vector_dim)
    values = Q.values()
    if values.numel() % block_size != 0:
        raise ValueError(
            f"Q values (len={values.numel()}) not divisible by block size {block_size}."
        )
    num_blocks = values.numel() // block_size
    indices = Q.indices()
    rows = indices[0].view(num_blocks, block_size)[:, 0]
    cols = indices[1].view(num_blocks, block_size)[:, 0]
    block_src = torch.div(rows, vector_dim, rounding_mode='floor')
    block_dst = torch.div(cols, vector_dim, rounding_mode='floor')
    block_pairs = torch.stack([block_src, block_dst], dim=0)

    edge_index = edge_index.to(torch.long)
    edge_hash = edge_index[0] * num_nodes + edge_index[1]
    if edge_hash.numel() == 0:
        block_edge_ids = torch.full(
            (num_blocks,), -1, dtype=torch.long, device=block_src.device
        )
    else:
        edge_hash_sorted, sort_idx = torch.sort(edge_hash)
        sorted_edge_ids = torch.arange(
            edge_hash.numel(), dtype=torch.long, device=edge_hash.device
        )[sort_idx]
        block_hash = block_src * num_nodes + block_dst
        insertion_idx = torch.searchsorted(edge_hash_sorted, block_hash)
        insertion_idx = torch.clamp(
            insertion_idx, max=max(int(edge_hash_sorted.numel()) - 1, 0)
        )
        block_edge_ids = torch.full(
            (num_blocks,), -1, dtype=torch.long, device=edge_hash.device
        )
        matched = edge_hash_sorted[insertion_idx] == block_hash
        if matched.any():
            block_edge_ids[matched] = sorted_edge_ids[insertion_idx[matched]]

    return {
        'block_pairs': block_pairs.cpu(),
        'block_edge_ids': block_edge_ids.cpu(),
    }


def macaque_compute_Qs_to_hdf5(
    outputs_by_day: Dict[Any, Dict[str, List[Dict[str, Any]]]],
    graphs_by_day: Dict[Any, Data],
    h5_path: str,
    days_included_idx: Optional[List[int]] = None,
    mode: Literal["transductive", "inductive"] = "transductive",
    path_graph_dir: Optional[str] = None,
    q_num_workers: Optional[int] = None,
) -> None:
    """
    Build path graphs per trajectory and save their Q sparse tensors to HDF5.
    HDF5 layout: Q/day_<day>/trial_<trial_id>/{indices, values, size}
    """
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    items = list(outputs_by_day.items())
    if days_included_idx is not None:
        items = [items[i] for i in days_included_idx if 0 <= i < len(items)]
    with h5py.File(h5_path, 'w') as h5f:
        for day, splits in items:
            day_group = h5f.require_group(f"Q/day_{day}")
            day_graph = graphs_by_day[day]
            # Prepare path graphs: either load from disk or build on the fly
            if path_graph_dir is not None:
                day_pg_path = os.path.join(path_graph_dir, f"day_{day}_paths.pt")
                path_splits = torch.load(day_pg_path, weights_only=False)
            else:
                path_splits = {
                    sk: [_build_path_graph_for_sample(s) for s in splits.get(sk, [])]
                    for sk in ("train", "valid", "test")
                }

            # Parallel Q build per day to tensors in memory; write serially
            def _build_for_pg(
                pg: Data, 
                trial_id: Any
            ) -> Tuple[Any, Dict[str, Any]]:
                # Project features to intrinsic dim if needed
                _project_path_graph_feat_to_d(pg, day_graph, mode=mode)
                Q = _build_Q_for_path_graph(
                    pg, 
                    day_graph, 
                    mode=mode
                )
                return trial_id, Q

            for sk in ("train", "valid", "test"):
                pgs: List[Data] = path_splits.get(sk, [])
                if len(pgs) == 0:
                    continue
                if (q_num_workers is None) or (q_num_workers <= 1):
                    results = [_build_for_pg(pg, getattr(pg, 'trial_id')) for pg in pgs]
                else:
                    with ThreadPoolExecutor(max_workers=q_num_workers) as ex:
                        futures = [ex.submit(_build_for_pg, pg, getattr(pg, 'trial_id')) for pg in pgs]
                        results = [f.result() for f in futures]
                for trial_id, Qdict in results:
                    tgrp = day_group.require_group(f"trial_{trial_id}")
                    tgrp.create_dataset('indices', data=Qdict['indices'], compression="gzip")
                    tgrp.create_dataset('values', data=Qdict['values'], compression="gzip")
                    tgrp.create_dataset('size', data=Qdict['size'], compression="gzip")


def save_path_graphs_by_day(
    outputs_by_day: Dict[Any, Dict[str, List[Dict[str, Any]]]],
    save_dir: str,
    days_included_idx: Optional[List[int]] = None,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    task_level: str = 'node',
) -> None:
    """
    Build and save per-day path graphs (without Q) as .pt files.
    Saves dict per day: {'train': List[Data], 'valid': List[Data], 'test': List[Data]}.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter by dictionary keys, not positional indices
    if days_included_idx is not None:
        items = [(day, splits) for day, splits in outputs_by_day.items() 
                 if day in days_included_idx]
    else:
        items = list(outputs_by_day.items())

    if (num_workers is None) or (num_workers <= 1):
        for day, splits in items:
            k, v = _worker_build_day_paths(day, splits, task_level=task_level)
            torch.save(v, os.path.join(save_dir, f"day_{k}_paths.pt"))
    else:
        executor_cls = ProcessPoolExecutor if backend == "processes" else ThreadPoolExecutor
        with executor_cls(max_workers=num_workers) as ex:
            futures = [
                ex.submit(_worker_build_day_paths, day, splits, task_level) \
                for day, splits in items
            ]
            for fut in futures:
                try:
                    k, v = fut.result()
                    torch.save(v, os.path.join(save_dir, f"day_{k}_paths.pt"))
                except Exception as e:
                    print(f"save_path_graphs_by_day worker failed: {e}")


def process_and_build_day_spatial_graphs(
    data_root: str,
    include_lever_velocity: bool = False,
    reduce_components: int = GLOBAL_PCA_REDUCED_DIM,
    fit_pca_on_filtered: bool = False,
    apply_filter_before_transform: bool = True,
    n_neighbors: int = KNN_N_NEIGHBORS,
    cknn_delta: float = CKNN_DELTA,
    metric: str = 'euclidean',
    include_self: bool = False,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    reweight_with_median_kernel: bool = True,
    days_included_idx: Optional[List[int]] = None,
    mode: Literal["transductive", "inductive"] = "transductive",
    spatial_graph_dir: Optional[str] = None,
    force_recompute_spatial: bool = False,
    min_neighbors: Optional[int] = None,
) -> Dict[Any, Data]:
    """
    End-to-end pipeline: load -> merge -> split+filter+optional PCA -> per-day graphs.

    - Builds one spatial CkNN graph per included day from lever positions across all trials/timepoints (aligned with diffed spike nodes).
    - Splits remain at trajectory level (train/val/test) and are stored as node masks.
    - Days are processed serially, but within each day per-node SVD computation is parallelized via `num_workers` and `backend`.

    Node-trial mapping (per returned day graph):
    - node_trial_id: Long tensor (N,) mapping each node to its trial id.
    - node_trial_time_idx: Long tensor (N,) time index within its trial after GO_CUE/diff alignment.
    - node_condition_idx: Long tensor (N,) condition index per `CONDITIONS` (or -1 if unknown).
    - trial_ids: Long tensor (T,) unique trial ids for the day.
    - trial_ptr: Long tensor (T+1,) CSR-style pointers; nodes for trial j are in [trial_ptr[j], trial_ptr[j+1]).
    - trial_split: Long tensor (T,) values {0=train, 1=val, 2=test} per trajectory.

    Example usage:
        >>> g = graphs_by_day[day]
        >>> # Nodes for trial j
        >>> j = 0
        >>> start, end = int(g.trial_ptr[j].item()), int(g.trial_ptr[j+1].item())
        >>> node_idx_for_trial_j = torch.arange(start, end)
        >>> # Trial id for node i
        >>> i = 10
        >>> tid = int(g.node_trial_id[i].item())
        >>> # Unique train trial ids present in this graph
        >>> train_trial_ids = torch.unique(g.node_trial_id[g.train_mask])
        >>> # Map a trial id to its split label
        >>> tid_query = train_trial_ids[0]
        >>> j_idx = (g.trial_ids == tid_query).nonzero(as_tuple=True)[0].item()
        >>> split_label = int(g.trial_split[j_idx].item())  # 0,1,2 for train/val/test
    """
    kinematics, trial_ids, spike_data = load_data_dicts(data_root)
    merged = merge_data_dicts(
        kinematics,
        trial_ids,
        spike_data,
        include_lever_velocity=include_lever_velocity,
        num_workers=None,
    )
    outputs_by_day = process_by_day_with_splits(
        merged,
        n_components=reduce_components,
        fit_pca_on_filtered=fit_pca_on_filtered,
        apply_filter_before_transform=apply_filter_before_transform,
        mode=mode,
    )
    graphs_by_day = macaque_build_spatial_graphs_with_O(
        outputs_by_day,
        days_included_idx=days_included_idx,
        mode=mode,
        spatial_graph_dir=spatial_graph_dir,
        force_recompute=force_recompute_spatial,
        n_neighbors=n_neighbors,
        cknn_delta=cknn_delta,
        metric=metric,
        include_self=include_self,
        num_workers=num_workers,
        backend=backend,
        reweight_with_median_kernel=reweight_with_median_kernel,
        min_neighbors=min_neighbors,
    )
    return graphs_by_day


def _validate_and_determine_pca_settings() -> Tuple[bool, Union[int, float], bool]:
    """
    Validate geometric mode requirements and determine PCA settings.
    
    Returns:
        Tuple of (pca_enabled, pca_dim_or_var_retained, match_pca_dim)
    """
    # Validate geometric mode requirements
    match_pca_dim = False
    if GEOMETRIC_MODE == 'equivariant':
        if isinstance(MANIFOLD_INTRINSIC_DIM, str):
            if MANIFOLD_INTRINSIC_DIM != "match_pca":
                raise ValueError(
                    f"Equivariant mode requires explicit integer MANIFOLD_INTRINSIC_DIM, "
                    f"but got '{MANIFOLD_INTRINSIC_DIM}'. Set MANIFOLD_INTRINSIC_DIM to an "
                    f"integer value (or 'match_pca' when using variance-based PCA) to use "
                    f"equivariant mode."
                )
            match_pca_dim = True
        if not match_pca_dim and isinstance(MANIFOLD_INTRINSIC_DIM, str):
            raise ValueError(
                f"Equivariant mode requires explicit integer MANIFOLD_INTRINSIC_DIM, "
                f"but got '{MANIFOLD_INTRINSIC_DIM}'. Set MANIFOLD_INTRINSIC_DIM to an "
                f"integer value to use equivariant mode."
            )
        if not match_pca_dim:
            d = int(MANIFOLD_INTRINSIC_DIM)
            if GLOBAL_PCA_REDUCED_DIM is None or GLOBAL_PCA_REDUCED_DIM != d:
                print(
                    f"[INFO] Equivariant mode: Setting GLOBAL_PCA_REDUCED_DIM = "
                    f"MANIFOLD_INTRINSIC_DIM = {d} (required for D = d)"
                )
                globals()['GLOBAL_PCA_REDUCED_DIM'] = d

    # Determine PCA settings
    pca_enabled = True
    variance_target: float | None = None
    if (
        GLOBAL_PCA_VARIANCE_RETAINED is not None
        and isinstance(GLOBAL_PCA_VARIANCE_RETAINED, float)
        and (0.0 < GLOBAL_PCA_VARIANCE_RETAINED < 1.0)
    ):
        variance_target = float(GLOBAL_PCA_VARIANCE_RETAINED)
    elif (
        GLOBAL_PCA_REDUCED_DIM is not None
        and isinstance(GLOBAL_PCA_REDUCED_DIM, float)
        and (0.0 < GLOBAL_PCA_REDUCED_DIM < 1.0)
    ):
        variance_target = float(GLOBAL_PCA_REDUCED_DIM)

    if variance_target is not None:
        pca_dim_or_var_retained = variance_target
    elif GLOBAL_PCA_REDUCED_DIM is not None:
        if GLOBAL_PCA_REDUCED_DIM >= ORIGINAL_NEURAL_STATE_DIM:
            print(
                f"Warning: GLOBAL_PCA_REDUCED_DIM ({GLOBAL_PCA_REDUCED_DIM}) >= "
                f"original dim ({ORIGINAL_NEURAL_STATE_DIM}). Disabling PCA."
            )
            pca_enabled = False
        else:
            pca_dim_or_var_retained = int(GLOBAL_PCA_REDUCED_DIM)
    else:
        pca_enabled = False

    if not pca_enabled:
        return False, None, match_pca_dim
    if match_pca_dim and variance_target is None:
        raise ValueError(
            "MANIFOLD_INTRINSIC_DIM='match_pca' requires variance-retention PCA "
            "(set GLOBAL_PCA_VARIANCE_RETAINED or GLOBAL_PCA_REDUCED_DIM to a value in (0,1))."
        )
    return True, pca_dim_or_var_retained, match_pca_dim


def _fit_day_preprocessing(
    samples: List[Dict[str, Any]],
    fit_indices: Optional[List[int]],
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
) -> Tuple[Optional[StandardScaler], Optional[PCA], StandardScaler, StandardScaler]:
    """
    Fit PCA and target scalers for a day's samples.
    
    Args:
        samples: List of sample dictionaries
        fit_indices: Indices to fit on (None = all samples)
        apply_savgol_filter_before_pca: Whether to apply Savitzky-Golay filter
        
    Returns:
        Tuple of (scaler, pca, pos_scaler, vel_scaler)
    """
    # Validate and determine PCA settings
    pca_enabled, pca_dim_or_var_retained, match_pca_dim = _validate_and_determine_pca_settings()
    
    # Fit PCA if enabled
    scaler, pca = None, None
    if pca_enabled:
        scaler, pca = _fit_scalar_and_pca_for_day(
            samples,
            fit_indices,
            pca_dim_or_var_retained=pca_dim_or_var_retained,
            apply_savgol_filter_before_pca=apply_savgol_filter_before_pca,
        )
        # Update global reduced dim to realized component count when variance specified
        try:
            realized = getattr(pca, 'n_components_', None)
            if isinstance(realized, int) and realized > 0:
                globals()['GLOBAL_PCA_REDUCED_DIM'] = realized
                if match_pca_dim:
                    globals()['MANIFOLD_INTRINSIC_DIM'] = realized
                    if GEOMETRIC_MODE == 'equivariant':
                        print(
                            f"[INFO] match_pca: setting MANIFOLD_INTRINSIC_DIM = "
                            f"GLOBAL_PCA_REDUCED_DIM = {realized}"
                        )
        except Exception as e:
            print(f"Warning: could not update GLOBAL_PCA_REDUCED_DIM from fitted PCA: {e}")
    
    # Fit target scalers
    pos_scaler, vel_scaler = _fit_target_scalers_for_day(
        samples,
        fit_indices,
    )
    
    return scaler, pca, pos_scaler, vel_scaler


def _create_sample_transform_fn(
    scaler: Optional[StandardScaler],
    pca: Optional[PCA],
    pos_scaler: StandardScaler,
    vel_scaler: StandardScaler,
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
) -> Callable[[List[Dict[str, Any]], int], Dict[str, Any]]:
    """
    Create a transformer function that processes samples with fitted preprocessing.
    
    Args:
        scaler: Fitted StandardScaler (or None)
        pca: Fitted PCA (or None)
        pos_scaler: Fitted position target scaler
        vel_scaler: Fitted velocity target scaler
        apply_savgol_filter_before_pca: Whether to apply Savitzky-Golay filter
        
    Returns:
        Transform function: (samples, index) -> transformed_sample_dict
    """
    # Allow PCA even when no scaler is used
    pca_enabled = pca is not None
    
    def _transform_one(samples: List[Dict[str, Any]], i: int) -> Dict[str, Any]:
        s = samples[i]
        trial = s[NEURAL_VELOCITY_ATTRIBUTE]
        trial_proc = _slice_spike_data(trial)

        if apply_savgol_filter_before_pca:
            trial_proc = savgol_filter(
                trial_proc,
                window_length=SAVGOL_FILTER_KWARGS["window_length"],
                polyorder=SAVGOL_FILTER_KWARGS["polyorder"],
                axis=1,
            )

        if pca_enabled:
            X = trial_proc.T
            X_scaled = scaler.transform(X) if scaler is not None else X
            X_scores = pca.transform(X_scaled)
            state_k_t = X_scores.T
        else:
            state_k_t = trial_proc
        
        vel_k_t = np.diff(state_k_t, n=1, axis=1)
        
        # Apply target standardization
        lever_pos = s.get("lever_pos")
        lever_vel = s.get("lever_vel")
        if lever_pos is not None:
            lever_pos_scaled = pos_scaler.transform(lever_pos.T).T
        else:
            lever_pos_scaled = lever_pos
        if lever_vel is not None:
            lever_vel_scaled = vel_scaler.transform(lever_vel.T).T
        else:
            lever_vel_scaled = lever_vel
        
        out_dict = dict(s)
        out_dict[NEURAL_STATE_ATTRIBUTE] = state_k_t
        out_dict[NEURAL_VELOCITY_ATTRIBUTE] = vel_k_t
        out_dict["lever_pos"] = lever_pos_scaled
        out_dict["lever_vel"] = lever_vel_scaled
        return out_dict
    
    return _transform_one


def cyclically_choose_valid_and_train_folds(
    num_samples: int,
    k_folds: int,
    fold_i: int,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[int], List[int]]:
    """
    Choose valid and train folds deterministically (cyclically) from non-test folds.
    Args:
        num_samples: number of samples
        k_folds: number of folds
        fold_i: index of the fold to choose valid and train folds from
        seed: seed for the random number generator
    Returns:
        train_idx: list of indices for the train fold
        valid_idx: list of indices for the valid fold
    """
    rng = default_rng(seed=seed)
    idx_all = np.arange(num_samples)
    rng.shuffle(idx_all)

    base = num_samples // int(k_folds)
    rem = num_samples % int(k_folds)
    folds: List[List[int]] = []
    start = 0
    for i in range(int(k_folds)):
        sz = base + (1 if i < rem else 0)
        folds.append(idx_all[start:start + sz].tolist())
        start += sz

    # Cyclical validation and train fold indices
    val_fold_i = (fold_i + 1) % k_folds
    train_fold_is = [
        i for i in range(k_folds) \
        if i not in (fold_i, val_fold_i)
    ]
    test_idx = folds[fold_i]
    valid_idx = folds[val_fold_i]
    train_idx = [idx for fold_i in train_fold_is for idx in folds[fold_i]]

    return train_idx, valid_idx, test_idx


def macaque_prepare_kth_fold(
    data_root: str,
    day_index: Union[int, Sequence[int]],
    k_folds: int,
    fold_i: int,
    seed: int = DEFAULT_SPLIT_SEED,
    mode: Literal["transductive", "inductive"] = "inductive",
    include_lever_velocity: bool = True,
    n_neighbors: int = KNN_N_NEIGHBORS,
    cknn_delta: float = CKNN_DELTA,
    metric: str = 'euclidean',
    include_self: bool = False,
    reweight_with_median_kernel: bool = True,
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
    parallel_backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    num_workers: Optional[int] = NUM_WORKERS_DEFAULT,
    target_key: Optional[str] = None,
    task_level: str = 'node',
) -> Dict[str, List[Data]]:
    """
    Prepare one k-fold split for the macaque dataset (single or multiple days), returning
    a dict { 'train'|'valid'|'test': List[Data] } of path graphs with a sparse
    COO attribute 'Q' attached to each.

    - Equal-sized folds by trial with deterministic shuffle by seed
    - Validation fold chosen cyclically from test fold index + 1
    - Spatial graph/O-frames:
      - inductive: build on TRAIN nodes only; assign VAL/TEST O by nearest TRAIN
      - transductive: build on ALL nodes; O exists for all nodes
    - Q is computed and attached directly; no HDF5 is written
    - Target extraction: if task_level='graph', only final timepoint targets are kept
      per trajectory (set at graph construction time in _build_path_graph_for_sample)
    - Geometric mode (global constant GEOMETRIC_MODE):
      - 'equivariant': Requires D=d (GLOBAL_PCA_REDUCED_DIM = MANIFOLD_INTRINSIC_DIM)
        and explicit integer MANIFOLD_INTRINSIC_DIM. Produces SO(d)-equivariant wavelets.
      - 'invariant': Allows D>d and automatic d estimation. Produces SO(D)-invariant wavelets.

    Note that this function uses list comprehensions to transform the samples in each split, instead of the ProcessPoolExecutor/ThreadPoolExecutor parallelization strategy (as in macaque_build_spatial_graphs_with_O). This seems adequate for the small data set size for each day.
    """
    # ------------------------------------------------------------
    # Step 1: Load base dicts and merge
    # ------------------------------------------------------------
    day_indices = _normalize_day_indices(day_index)
    if not day_indices:
        raise ValueError("macaque_day_index must specify at least one day.")
    day_label = _format_day_label(day_indices)
    kinematics, trial_ids, spike_data = load_data_dicts(data_root)
    merged = merge_data_dicts(
        kinematics,
        trial_ids,
        spike_data,
        include_lever_velocity=include_lever_velocity,
        num_workers=None,
        days_included_idx=day_indices,
        return_grouped_by_day=True,
    )
    samples = _collect_samples_for_days(merged, day_indices)
    print(f"Num. samples for selection {day_label}: {len(samples)}")

    # ------------------------------------------------------------
    # Step 2: Set train/valid/test folds
    # ------------------------------------------------------------
    train_idx, valid_idx, test_idx = _prepare_fold_indices(
        samples,
        k_folds,
        fold_i,
        seed,
    )

    # ------------------------------------------------------------
    # Step 3: Fit PCA and target scalers
    # ------------------------------------------------------------
    scaler, pca, pos_scaler, vel_scaler = _fit_day_preprocessing(
        samples,
        train_idx if mode == "inductive" else None,
        apply_savgol_filter_before_pca=apply_savgol_filter_before_pca,
    )

    # ------------------------------------------------------------
    # Step 4: Apply processing steps to all samples in the day
    # ------------------------------------------------------------
    # Create transformer with fitted preprocessing
    transform_sample = _create_sample_transform_fn(
        scaler, pca, pos_scaler, vel_scaler, apply_savgol_filter_before_pca
    )

    split_samples: Dict[str, List[Dict[str, Any]]] = {
        'train': [transform_sample(samples, i) for i in train_idx],
        'valid': [transform_sample(samples, i) for i in valid_idx],
        'test':  [transform_sample(samples, i) for i in test_idx],
    }

    # ------------------------------------------------------------
    # Step 6: Build spatial graph for this day
    # ------------------------------------------------------------
    # Use TRAIN nodes only (inductive mode) or all nodes (transductive mode)
    splits_by_day = {day_label: split_samples}
    spatial_graphs = macaque_build_spatial_graphs_with_O(
        outputs_by_day=splits_by_day,
        days_included_idx=[day_label],  # Filter by day index (dictionary key)
        mode=mode,
        spatial_graph_dir=None,
        force_recompute=False,
        n_neighbors=int(n_neighbors),
        cknn_delta=float(cknn_delta),
        metric=str(metric),
        include_self=bool(include_self),
        num_workers=num_workers,  # For parallel SVD computation within the day
        backend=parallel_backend,
        reweight_with_median_kernel=bool(reweight_with_median_kernel),
    )
    spatial_graph = spatial_graphs[day_label]

    # ------------------------------------------------------------
    # Step 7: Build path graphs per split
    # ------------------------------------------------------------
    _, path_split = _worker_build_day_paths(
        day_label, 
        splits_by_day[day_label],
        task_level=task_level,
    )

    for set_key in ('train', 'valid', 'test'):
        path_graphs = path_split.get(set_key, [])

        # Project 'neural velocity' vector features to intrinsic manifold dim.
        for path_graph in path_graphs:
            _project_path_graph_feat_to_d(
                path_graph, 
                spatial_graph, 
                mode=mode
            )

    # Define helper to attach Q to one path graph
    def _attach_Q(pg: Data) -> Data:
        Qdict = _build_Q_for_path_graph(pg, spatial_graph, mode=mode)
        idx_t = torch.from_numpy(Qdict['indices']).to(torch.long)
        vals_t = torch.from_numpy(Qdict['values']).to(torch.float32)
        size = tuple(int(x) for x in Qdict['size'].tolist())
        try:
            Q = torch.sparse_coo_tensor(
                indices=idx_t, 
                values=vals_t, 
                size=size
            )
            setattr(pg, 'Q', Q)
        except Exception as e:
            raise ValueError(
                f"Error constructing Q for trial {getattr(pg, 'trial_id', None)}: {e}"
            )
        # Ensure correct block-diagonal batching for sparse operators by using ESCGNNData
        if not isinstance(pg, ESCGNNData):
            pg = ESCGNNData(**dict(pg), operator_keys=('Q',))
        return pg

    # Attach Q to all path graphs by split
    print(f"Attaching Q to path graphs for {day_label}...")
    out_graphs = {
        'train': [_attach_Q(pg) for pg in path_split.get('train', [])],
        'valid': [_attach_Q(pg) for pg in path_split.get('valid', [])],
        'test':  [_attach_Q(pg) for pg in path_split.get('test', [])],
    }

    print(f"\tDone.")
    return out_graphs


def macaque_prepare_spatial_graph_only_for_fold(
    data_root: str,
    day_index: Union[int, Sequence[int]],
    k_folds: int,
    fold_i: int,
    seed: int = DEFAULT_SPLIT_SEED,
    mode: Literal["transductive", "inductive"] = "transductive",
    include_lever_velocity: bool = True,
    n_neighbors: int = KNN_N_NEIGHBORS,
    cknn_delta: float = CKNN_DELTA,
    metric: str = 'euclidean',
    include_self: bool = False,
    reweight_with_median_kernel: bool = True,
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
    parallel_backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
    num_workers: Optional[int] = NUM_WORKERS_DEFAULT,
    target_key: Optional[str] = None,
    min_neighbors: Optional[int] = None,
) -> Data:
    """
    Prepare a single spatial graph with Q operator for one k-fold split (no path graphs).
    
    This function creates a unified spatial graph containing all timepoints from all trials
    for the selected day(s), with train/valid/test masks attached at the node level.
    
    Pipeline:
    1. Load and merge data for the specified day
    2. Split trials into k folds (deterministic)
    3. Fit PCA and target scalers:
       - Inductive mode: fit on train trials only (NOT IMPLEMENTED)
       - Transductive mode: fit on all trials
    4. Build spatial CkNN graph over all neural state nodes
    5. Compute O-frames for all nodes (transductive) or train nodes (inductive)
    6. Build Q operator matrix for entire spatial graph
    7. Attach node-level targets, masks, and trial mapping metadata
    
    Args:
        data_root: Path to macaque data directory
        day_index: Index or indices of the day(s) to process
        k_folds: Number of folds for cross-validation
        fold_i: Index of current fold (0 to k_folds-1)
        seed: Random seed for deterministic splitting
        mode: 'transductive' (fit on all data) or 'inductive' (fit on train only)
        include_lever_velocity: Whether to include velocity targets
        n_neighbors: Number of neighbors for CkNN graph
        cknn_delta: Delta parameter for CkNN
        metric: Distance metric for CkNN
        include_self: Whether to include self-loops
        reweight_with_median_kernel: Whether to apply kernel reweighting
        apply_savgol_filter_before_pca: Whether to apply Savitzky-Golay filter
        parallel_backend: Backend for parallel processing ('processes' or 'threads')
        num_workers: Number of workers for parallel processing
        target_key: Target attribute name (used to detect trajectory-level tasks)
        
    Returns:
        Data object with:
        - v_vel: Neural velocity node features (N, d) after PCA/projection to tangent space
        - Q: Sparse COO diffusion operator (Nd x Nd)
        - pos_xy: Position targets (N, 2) - node-level
        - vel_xy: Velocity targets (N, 2) - node-level
        - train_mask, valid_mask, test_mask: Boolean masks (N,) for node-level splits
        - trial_ids: Unique trial IDs (num_trajectories,)
        - trial_ptr: CSR-style pointers to map nodes to trials (num_trajectories + 1,)
        - node_trial_id: Trial ID per node (N,)
        - trial_split: Split label per trial (num_trajectories,) - 0=train, 1=valid, 2=test
        - num_nodes: Total number of nodes
        - day: Day index
        
    Note:
        - In transductive mode, PCA and scalers are fit on all data
        - In inductive mode: NOT IMPLEMENTED (raises NotImplementedError)
        - For trajectory-level tasks (target_key contains 'final_'), targets remain
          node-level but should be pooled by trial_id during forward pass
    """
    # Enforce transductive mode only for now
    if mode == "inductive":
        raise NotImplementedError(
            "Inductive mode is not yet implemented for spatial graph pipeline "
            "(scatter_path_graphs=False). Please use mode='transductive'."
        )
    
    # ------------------------------------------------------------
    # Step 1: Load base dicts and merge
    # ------------------------------------------------------------
    day_indices = _normalize_day_indices(day_index)
    if not day_indices:
        raise ValueError("macaque_day_index must specify at least one day.")
    day_label = _format_day_label(day_indices)
    kinematics, trial_ids, spike_data = load_data_dicts(data_root)
    merged = merge_data_dicts(
        kinematics,
        trial_ids,
        spike_data,
        include_lever_velocity=include_lever_velocity,
        num_workers=None,
        days_included_idx=day_indices,
        return_grouped_by_day=True,
    )
    samples = _collect_samples_for_days(merged, day_indices)
    print(f"Num. samples for selection {day_label}: {len(samples)}")

    # ------------------------------------------------------------
    # Step 2: Set train/valid/test folds
    # ------------------------------------------------------------
    train_idx, valid_idx, test_idx = _prepare_fold_indices(
        samples,
        k_folds,
        fold_i,
        seed,
    )

    # ------------------------------------------------------------
    # Step 3: Fit PCA and target scalers (transductive mode only)
    # ------------------------------------------------------------
    scaler, pca, pos_scaler, vel_scaler = _fit_day_preprocessing(
        samples,
        None,  # Fit on all data (transductive)
        apply_savgol_filter_before_pca=apply_savgol_filter_before_pca,
    )

    # ------------------------------------------------------------
    # Step 4: Apply processing to all samples
    # ------------------------------------------------------------
    # Create transformer with fitted preprocessing
    transform_sample = _create_sample_transform_fn(
        scaler, pca, pos_scaler, vel_scaler, apply_savgol_filter_before_pca
    )

    # Transform all samples
    all_samples_transformed = [transform_sample(samples, i) for i in range(len(samples))]
    
    # ------------------------------------------------------------
    # Step 5: Precompute velocities and align states with velocity nodes
    # ------------------------------------------------------------
    # For T states, we can compute T-1 velocities: v[i] = x[i+1] - x[i]
    # Each velocity v[i] is assigned to node i
    # We compute velocities first, then truncate states to match (T-1 nodes)
    for sample in all_samples_transformed:
        state = sample[NEURAL_STATE_ATTRIBUTE]  # (k, T)
        if state.shape[1] > 1:
            # Compute velocity using full state sequence: T states -> T-1 velocities
            velocity = np.diff(state, n=1, axis=1)  # (k, T-1)
            # Truncate state to match velocity length (drop last state)
            sample[NEURAL_STATE_ATTRIBUTE] = state[:, :-1]  # (k, T-1)
            # Store velocity in ambient space for later projection
            sample['_velocity_ambient'] = velocity  # (k, T-1)
        
        # Truncate targets to match (keep first T-1 timepoints)
        pos = sample.get("lever_pos")
        if pos is not None and pos.shape[1] > 0:
            sample["lever_pos"] = pos[:, :-1]
        vel = sample.get("lever_vel")
        if vel is not None and vel.shape[1] > 0:
            sample["lever_vel"] = vel[:, :-1]

    # ------------------------------------------------------------
    # Step 6: Build spatial graph (with train/valid/test masks)
    # ------------------------------------------------------------
    # Organize into splits dict for compatibility with existing function
    split_samples = {
        'train': [all_samples_transformed[i] for i in train_idx],
        'valid': [all_samples_transformed[i] for i in valid_idx],
        'test':  [all_samples_transformed[i] for i in test_idx],
    }
    
    splits_by_day = {day_label: split_samples}
    
    # Build spatial graph with O-frames
    spatial_graphs = macaque_build_spatial_graphs_with_O(
        outputs_by_day=splits_by_day,
        days_included_idx=[day_label],
        mode=mode,
        spatial_graph_dir=None,
        force_recompute=False,
        n_neighbors=int(n_neighbors),
        cknn_delta=float(cknn_delta),
        metric=str(metric),
        include_self=bool(include_self),
        num_workers=num_workers,
        backend=parallel_backend,
        reweight_with_median_kernel=bool(reweight_with_median_kernel),
        min_neighbors=min_neighbors,
    )
    spatial_graph = spatial_graphs[day_label]

    # ------------------------------------------------------------
    # Step 7: Build Q operator for entire spatial graph
    # ------------------------------------------------------------
    print(f"Building Q operator for spatial graph ({day_label})...")
    
    Q = _build_Q_for_spatial_graph_parallel(
        spatial_graph,
        geometric_mode=GEOMETRIC_MODE,
        num_workers=num_workers,
        backend=parallel_backend,
    )
    
    # Attach Q to spatial graph
    spatial_graph.Q = Q
    
    # ------------------------------------------------------------
    # Step 8: Project precomputed neural velocity features to tangent space
    # ------------------------------------------------------------
    # Retrieve precomputed velocities (in ambient space) from samples
    # These were computed before building the graph, ensuring proper alignment
    O_frames = spatial_graph.O_frames  # (N, D, d)
    num_nodes = int(spatial_graph.num_nodes)
    trial_ptr = spatial_graph.trial_ptr
    num_trials = len(spatial_graph.trial_ids)
    trial_ids_list = spatial_graph.trial_ids.tolist()
    
    # Build mapping from trial_id to sample
    trial_id_to_sample = {}
    for split_key in ('train', 'valid', 'test'):
        for sample in split_samples[split_key]:
            tid = sample.get('trial_id')
            if tid is not None:
                trial_id_to_sample[tid] = sample
    
    # Collect velocities in graph node order
    v_vel_list = []
    for trial_idx in range(num_trials):
        trial_id = trial_ids_list[trial_idx]
        sample = trial_id_to_sample.get(trial_id)
        
        # if sample is None or '_velocity_ambient' not in sample:
        #     # Fallback: zero velocity (shouldn't happen)
        #     start = int(trial_ptr[trial_idx].item())
        #     end = int(trial_ptr[trial_idx + 1].item())
        #     n_nodes = end - start
        #     v_vel_list.append(np.zeros((n_nodes, GLOBAL_PCA_REDUCED_DIM), dtype=np.float32))
        #     continue
        
        # Get precomputed velocity in ambient space (k, t-1)
        vel_ambient = sample['_velocity_ambient']  # (D, T-1) numpy array
        vel_ambient_nodes = vel_ambient.T  # (T-1, D) to match node order
        v_vel_list.append(vel_ambient_nodes)
    
    # Concatenate velocities from all trials and convert to torch
    v_vel = np.concatenate(v_vel_list, axis=0)  # (N, D)
    v_vel = torch.from_numpy(v_vel).to(O_frames.dtype).to(O_frames.device)
    
    # Project velocities to tangent space using O-frames
    v_vel_projected = torch.einsum('NDd,ND->Nd', O_frames, v_vel)  # (N, d)
    
    # Attach projected velocities as node features
    spatial_graph[NEURAL_VELOCITY_ATTRIBUTE] = v_vel_projected
    
    # ------------------------------------------------------------
    # Step 9: Prepare targets for trajectory-level or node-level tasks
    # ------------------------------------------------------------
    # Targets remain at node level; pooling happens in forward pass if needed
    # Spatial graph already has pos_xy and vel_xy as node-level targets
    
    # Print summary statistics
    print(f"Spatial graph prepared: {num_nodes} nodes, {num_trials} trials")
    
    # Compute and print neighbor count statistics
    nbr_stats = get_graph_nbr_ct_stats({day_label: spatial_graph})
    stats = nbr_stats[day_label]
    print(f"Neighbor count stats: min={stats['min']:.1f}, q1={stats['q1']:.1f}, "
          f"median={stats['median']:.1f}, q3={stats['q3']:.1f}, max={stats['max']:.1f}")
    
    return spatial_graph


def get_graph_nbr_ct_stats(
    graphs_by_day: Dict[Any, Data],
) -> Dict[Any, Dict[str, float]]:
    """
    Compute node neighbor count statistics for each day's single graph and
    report the undirected edge count.

    Returns: dict mapping day -> { 'min', 'q1', 'median', 'q3', 'max', 'undirected_edge_count' }.
    """
    stats_out: Dict[Any, Dict[str, float]] = {}

    for day, g in graphs_by_day.items():
        num_nodes = int(getattr(g, 'num_nodes', (int(g.edge_index.max().item()) + 1)))
        vals = torch.ones(g.edge_index.shape[1], dtype=torch.float32, device=g.edge_index.device)
        A = torch.sparse_coo_tensor(g.edge_index, vals, (num_nodes, num_nodes)).coalesce()
        nnz = int(A._nnz())
        diag_mask = A.indices()[0] == A.indices()[1]
        diag_ct = int(diag_mask.sum().item())
        undirected_edge_count = max(0, (nnz - diag_ct) // 2)
        counts = torch.sparse.sum(A, dim=1).to_dense().to(torch.float32)
        if counts.numel() == 0:
            stats_out[day] = {
                "min": 0.0,
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0,
                "max": 0.0,
                "undirected_edge_count": 0,
            }
            continue
        qs = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32, device=counts.device)
        q_vals = torch.quantile(counts, qs).cpu().tolist()
        stats_out[day] = {
            "min": float(torch.min(counts).item()),
            "q1": float(q_vals[0]),
            "median": float(q_vals[1]),
            "q3": float(q_vals[2]),
            "max": float(torch.max(counts).item()),
            "undirected_edge_count": float(undirected_edge_count),
        }

    return stats_out


def load_path_graphs_by_day(
    load_dir: str,
    days_included_idx: Optional[List[int]] = None,
    h5_path: Optional[str] = None,
    vector_operator_key: str = 'Q',
) -> Dict[Any, Dict[str, List[Data]]]:
    """
    Load per-day path graph shards saved by `save_path_graphs_by_day`.

    Args:
        load_dir: Directory containing files named like `day_<day>_paths.pt`.
        days_included_idx: Optional list of integer indices into the sorted available day files. If None, all days are loaded.

    Returns:
        Dict mapping day -> dict with keys 'train'|'valid'|'test' and values
        List[Data] of path graphs (without Q).
    """
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Path graphs directory not found: {load_dir}")

    files = [
        f for f in os.listdir(load_dir) \
        if f.startswith("day_") and f.endswith("_paths.pt")
    ]
    if not files:
        raise FileNotFoundError(f"No path graphs found in {load_dir}")

    # Extract day keys and sort
    def _parse_day(fname: str) -> Any:
        core = fname[len("day_"):-len("_paths.pt")]
        try:
            return int(core)
        except Exception:
            return core

    days_and_files = sorted(
        [( _parse_day(f), f ) for f in files], key=lambda x: x[0]
    )
    if days_included_idx is not None:
        chosen = [days_and_files[i] for i in days_included_idx if 0 <= i < len(days_and_files)]
    else:
        chosen = days_and_files

    out: Dict[Any, Dict[str, List[Data]]] = {}

    # Open HDF5 once if provided
    h5f = None
    if h5_path is not None:
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        h5f = h5py.File(h5_path, 'r')

    try:
        for day, fname in chosen:
            path = os.path.join(load_dir, fname)
            split_dict = torch.load(path, weights_only=False)

            # Optionally attach Q from HDF5
            if h5f is not None:
                day_group_name = f"{vector_operator_key}/day_{day}"
                if day_group_name not in h5f:
                    # If day missing, leave graphs unattached
                    out[day] = split_dict
                    continue
                day_group = h5f[day_group_name]

                for sk in ("train", "valid", "test"):
                    graphs: List[Data] = split_dict.get(sk, [])
                    split_dict[sk] = [
                        _attach_Q_from_hdf5_to_graph(
                            g, day_group, day, vector_operator_key
                        ) or g
                        for g in graphs
                    ]

            out[day] = split_dict
    finally:
        if h5f is not None:
            h5f.close()
    return out


def _attach_Q_from_hdf5_to_graph(
    g: Data,
    day_group: Any,
    day: Any,
    vector_operator_key: str,
) -> Optional[Data]:
    """
    Attach Q sparse COO tensor from HDF5 for a single path graph if available.
    Returns the graph on success or when data is absent; returns None on errors.
    """
    # Check that trial id is present (needed to retrieve the correct trial group from the HDF5 file)
    trial_id = getattr(g, 'trial_id', None)
    if trial_id is None:
        raise ValueError(f"Trial id not found for graph: {g}")
    trial_group_name = f"trial_{trial_id}"
    if trial_group_name not in day_group:
        raise ValueError(f"Trial group not found for day {day}, trial {trial_id}")
    tgrp = day_group[trial_group_name]
    if ("indices" not in tgrp) \
    or ("values" not in tgrp) \
    or ("size" not in tgrp):
        raise ValueError(f"Indices, values, or size not found for day {day}, trial {trial_id}")

    idx_np = tgrp["indices"][...]
    vals_np = tgrp["values"][...]
    size_np = tgrp["size"][...]
    idx_t = torch.from_numpy(idx_np).to(torch.long)
    vals_t = torch.from_numpy(vals_np).to(torch.float32)
    size_t = tuple(int(x) for x in size_np.tolist())

    try:
        Q = torch.sparse_coo_tensor(indices=idx_t, values=vals_t, size=size_t)
        setattr(g, vector_operator_key, Q)
        return g
    except Exception as e:
        raise ValueError(f"Error attaching Q to path graph for day {day}, trial {trial_id}: {e}")



# --------------------------------------------------------
# MARBLE-related data classes and methods
# --------------------------------------------------------
@dataclass
class MarbleTrialSequence:
    """
    Container describing one trial's aligned neural state trajectory for MARBLE.
    """
    trial_id: int
    day: Any
    condition_idx: int
    positions: np.ndarray  # shape (T_nodes, D)
    velocities: np.ndarray  # shape (T_nodes, D)
    node_indices: Optional[np.ndarray] = None  # indices into concatenated train nodes
    neighbor_indices: Optional[np.ndarray] = None  # (T_nodes, k) for held-out trials
    neighbor_weights: Optional[np.ndarray] = None  # (T_nodes, k) normalized weights


@dataclass
class MarbleFoldData:
    """
    Aggregated MARBLE-ready data for one k-fold split.
    """
    train_trials: List[MarbleTrialSequence]
    valid_trials: List[MarbleTrialSequence]
    test_trials: List[MarbleTrialSequence]
    train_positions: np.ndarray
    train_velocities: np.ndarray
    train_trial_ids: np.ndarray
    train_time_ids: np.ndarray
    train_condition_ids: np.ndarray
    train_trial_ptr: np.ndarray
    train_trial_conditions: np.ndarray
    nodes_per_trial: Optional[int]
    state_dim: int
    k_neighbors_used: int


def _build_marble_trial_sequence(
    sample: Dict[str, Any],
) -> MarbleTrialSequence:
    """
    Convert a processed sample dict into a MarbleTrialSequence with aligned positions/velocities.
    """
    state = sample.get(NEURAL_STATE_ATTRIBUTE)
    vel = sample.get(NEURAL_VELOCITY_ATTRIBUTE)
    if state is None or vel is None:
        raise ValueError("Sample missing neural state or velocity attributes required for MARBLE.")
    if state.ndim != 2 or vel.ndim != 2:
        raise ValueError("State and velocity arrays must be 2-D.")
    n_nodes = min(state.shape[1], vel.shape[1])
    if n_nodes <= 0:
        raise ValueError("Encountered empty trial after preprocessing.")
    positions = state[:, :n_nodes].T.astype(np.float32, copy=True)
    velocities = vel[:, :n_nodes].T.astype(np.float32, copy=True)
    cond_idx = _condition_to_index(sample.get("condition"))
    trial_id = int(sample.get("trial_id"))
    return MarbleTrialSequence(
        trial_id=trial_id,
        day=sample.get("day"),
        condition_idx=cond_idx,
        positions=positions,
        velocities=velocities,
    )


def _concat_train_sequences(
    train_trials: List[MarbleTrialSequence],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate positions/velocities across train trials and build per-node metadata.
    """
    if not train_trials:
        raise ValueError("No training trials provided for MARBLE fold data.")
    positions_list: List[np.ndarray] = []
    velocities_list: List[np.ndarray] = []
    trial_ids_list: List[np.ndarray] = []
    time_ids_list: List[np.ndarray] = []
    condition_ids_list: List[np.ndarray] = []
    trial_ptr = [0]
    trial_conditions: List[int] = []
    offset = 0
    for seq in train_trials:
        n = seq.positions.shape[0]
        seq.node_indices = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        positions_list.append(seq.positions)
        velocities_list.append(seq.velocities)
        trial_ids_list.append(np.full((n,), seq.trial_id, dtype=np.int64))
        time_ids_list.append(np.arange(n, dtype=np.int64))
        condition_ids_list.append(np.full((n,), seq.condition_idx, dtype=np.int64))
        trial_ptr.append(trial_ptr[-1] + n)
        trial_conditions.append(seq.condition_idx)
    train_positions = np.concatenate(positions_list, axis=0)
    train_velocities = np.concatenate(velocities_list, axis=0)
    train_trial_ids = np.concatenate(trial_ids_list, axis=0)
    train_time_ids = np.concatenate(time_ids_list, axis=0)
    train_condition_ids = np.concatenate(condition_ids_list, axis=0)
    train_trial_ptr = np.asarray(trial_ptr, dtype=np.int64)
    train_trial_conditions = np.asarray(trial_conditions, dtype=np.int64)
    return (
        train_positions,
        train_velocities,
        train_trial_ids,
        train_time_ids,
        train_condition_ids,
        train_trial_ptr,
        train_trial_conditions,
    )


def compute_adapt_gauss_kernel_kth_neighbor_wts(
    distances: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute normalized Gaussian kernel weights row-wise using adaptive bandwidths 
    scaled by the distance to the kth neighbor.
    """
    if distances.ndim != 2:
        raise ValueError("Distances must be a 2-D array.")
    # Scale parameter is distance to kth neighbor
    sigma = distances[:, -1].copy()
    needs_fallback = sigma <= eps
    if np.any(needs_fallback):
        fallback = np.min(distances[distances > eps]) if np.any(distances > eps) else 1.0
        sigma[needs_fallback] = fallback
    denom = (sigma[:, None] ** 2) + eps
    weights = np.exp(-(distances ** 2) / denom)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    weights /= row_sums
    return weights.astype(np.float32, copy=False)


def macaque_prepare_marble_fold_data(
    data_root: str,
    day_index: Union[int, Sequence[int]],
    k_folds: int,
    fold_i: int,
    seed: int = DEFAULT_SPLIT_SEED,
    include_lever_velocity: bool = False,
    k_neighbors: int = 30,
    apply_savgol_filter_before_pca: bool = APPLY_SAVGOL_FILTER_BEFORE_PCA,
    cknn_dist_metric: str = "euclidean",
) -> MarbleFoldData:
    """
    Prepare MARBLE-ready trial sequences and train-node metadata for a specific fold.

    Returns:
        MarbleFoldData with concatenated train arrays and neighbor caches for held-out trials.
    """
    day_indices = _normalize_day_indices(day_index)
    if not day_indices:
        raise ValueError("At least one macaque day index must be provided.")
    kinematics, trial_ids, spike_data = load_data_dicts(data_root)
    merged = merge_data_dicts(
        kinematics,
        trial_ids,
        spike_data,
        include_lever_velocity=include_lever_velocity,
        num_workers=None,
        days_included_idx=day_indices,
        return_grouped_by_day=True,
    )
    samples = _collect_samples_for_days(merged, day_indices)
    train_idx, valid_idx, test_idx = _prepare_fold_indices(
        samples,
        k_folds=int(k_folds),
        fold_i=int(fold_i),
        seed=int(seed),
    )
    scaler, pca, pos_scaler, vel_scaler = _fit_day_preprocessing(
        samples,
        train_idx,
        apply_savgol_filter_before_pca=apply_savgol_filter_before_pca,
    )
    transform_sample = _create_sample_transform_fn(
        scaler,
        pca,
        pos_scaler,
        vel_scaler,
        apply_savgol_filter_before_pca,
    )
    split_indices = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }
    split_trials: Dict[str, List[MarbleTrialSequence]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for split_name, indices in split_indices.items():
        for idx in indices:
            transformed = transform_sample(samples, idx)
            seq = _build_marble_trial_sequence(transformed)
            split_trials[split_name].append(seq)
    (
        train_positions,
        train_velocities,
        train_trial_ids,
        train_time_ids,
        train_condition_ids,
        train_trial_ptr,
        train_trial_conditions,
    ) = _concat_train_sequences(split_trials["train"])

    if train_positions.shape[0] == 0:
        raise ValueError("No train nodes available after preprocessing for MARBLE fold.")
    node_lengths = {
        seq.positions.shape[0] for seq in split_trials["train"] + split_trials["valid"] + split_trials["test"]
    }
    nodes_per_trial = node_lengths.pop() if len(node_lengths) == 1 else None
    k_eff = int(min(max(1, k_neighbors), train_positions.shape[0]))
    metric = str(cknn_dist_metric or "euclidean").lower()
    if metric == "cosine":
        try:
            tree = BallTree(train_positions, metric="angular")
        except Exception:
            tree = NearestNeighbors(metric="cosine", algorithm="brute")
            tree.fit(train_positions)
            tree_is_ball = False
        else:
            tree_is_ball = True
    else:
        tree = KDTree(train_positions, metric=metric)
        tree_is_ball = False

    # For held-out trials, compute the nearest neighbors and weights in the train set, for later mapping to embeddings
    for seq in split_trials["valid"] + split_trials["test"]:
        if seq.positions.shape[0] == 0:
            continue
        if metric == "cosine" and not tree_is_ball:
            dists, nbr_idx = tree.kneighbors(seq.positions, n_neighbors=k_eff, return_distance=True)
        else:
            dists, nbr_idx = tree.query(seq.positions, k=k_eff, return_distance=True)
        weights = compute_adapt_gauss_kernel_kth_neighbor_wts(dists)
        seq.neighbor_indices = nbr_idx.astype(np.int64, copy=False)
        seq.neighbor_weights = weights

    fold_data = MarbleFoldData(
        train_trials=split_trials["train"],
        valid_trials=split_trials["valid"],
        test_trials=split_trials["test"],
        train_positions=train_positions,
        train_velocities=train_velocities,
        train_trial_ids=train_trial_ids,
        train_time_ids=train_time_ids,
        train_condition_ids=train_condition_ids,
        train_trial_ptr=train_trial_ptr,
        train_trial_conditions=train_trial_conditions,
        nodes_per_trial=nodes_per_trial,
        state_dim=train_positions.shape[1],
        k_neighbors_used=k_eff,
    )
    return fold_data


'''
DEPRECATED METHODS

def _pca_reduce_sample(
    spike: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Reduce dimensionality along the neuron axis using standard scaling + PCA.

    Parameters
    ----------
    spike: np.ndarray
        Array of shape (n_neurons, n_timepoints).
    n_components: int
        Number of principal components to keep (<= n_neurons).

    Returns
    -------
    np.ndarray
        Reduced array of shape (n_components, n_timepoints).
    """
    if spike.ndim != 2:
        raise ValueError("Expected spike data of shape (n_neurons, n_timepoints)")

    n_neurons, _ = spike.shape
    k = min(n_components, n_neurons)

    # Treat timepoints as samples, neurons as features
    X = spike.T  # (n_timepoints, n_neurons)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=k, whiten=False)
    X_scores = pca.fit_transform(X_scaled)  # (n_timepoints, k)
    return X_scores.T  # (k, n_timepoints)


def reduce_spike_data_dimension(
    merged_data: List[Dict[str, Any]],
    method: str = "PCA",
    n_components: int = GLOBAL_PCA_REDUCED_DIM,
    num_workers: Optional[int] = None,
    backend: Literal["processes", "threads"] = PARALLEL_BACKEND,
) -> List[Dict[str, Any]]:
    """
    Apply dimensionality reduction to each sample's SPIKE_DATA_ATTRIBUTE entry.

    Parameters
    ----------
    merged_data: list of dicts
        Output from `merge_data_dicts`, each with key SPIKE_DATA_ATTRIBUTE shaped (n_neurons, n_timepoints).
    method: str
        Currently only "PCA" is supported.
    n_components: int
        Target reduced dimension along the selected axis.
    num_workers: Optional[int]
        If >1, parallelize across samples using the selected backend.
    backend: {'processes', 'threads'}
        Parallel backend to use when `num_workers > 1`.

    Returns
    -------
    List[Dict[str, Any]]: New list with SPIKE_DATA_ATTRIBUTE replaced by reduced arrays.
    """
    if method.upper() != "PCA":
        raise NotImplementedError(
            f"Method {method} not supported. Use 'PCA'.",
        )

    # Sequential path
    if not num_workers or num_workers <= 1:
        reduced: List[Dict[str, Any]] = []
        for sample in merged_data:
            sd = sample[VECTOR_FEAT_DATA_ATTRIBUTE]
            reduced_spike = _pca_reduce_sample(sd, n_components)
            new_sample = dict(sample)
            new_sample[VECTOR_FEAT_DATA_ATTRIBUTE] = reduced_spike
            reduced.append(new_sample)
        return reduced

    # Parallel path
    executor_cls = ProcessPoolExecutor if backend == "processes" else ThreadPoolExecutor
    reduced_parallel: List[Dict[str, Any]] = [None] * len(merged_data)  # type: ignore
    with executor_cls(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_pca_reduce_sample, sample[VECTOR_FEAT_DATA_ATTRIBUTE], n_components): idx
            for idx, sample in enumerate(merged_data)
        }
        for fut, idx in futures.items():
            try:
                reduced_spike = fut.result()
            except Exception as e:
                print(f"reduce_spike_data_dimension worker failed (index {idx}): {e}")
                reduced_spike = merged_data[idx][VECTOR_FEAT_DATA_ATTRIBUTE]
            new_sample = dict(merged_data[idx])
            new_sample[VECTOR_FEAT_DATA_ATTRIBUTE] = reduced_spike
            reduced_parallel[idx] = new_sample

    # Type ignore above; all filled
    return reduced_parallel  # type: ignore


def _transform_trial_with_pca(
    trial: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    apply_filter: bool,
    savgol_filter_kwargs: Optional[Dict[str, Any]] = SAVGOL_FILTER_KWARGS,
) -> np.ndarray:
    """
    Apply optional filter, then transform with pre-fit scaler and PCA (fit on train data, across all trials for that day).
    Returns (k, t) array.
    """
    trial_proc = _slice_after_go_cue(trial)
    if apply_filter:
        trial_proc = savgol_filter(
            trial_proc, 
            window_length=savgol_filter_kwargs["window_length"], 
            polyorder=savgol_filter_kwargs["polyorder"], 
            axis=1
        )
    X = trial_proc.T  # (t, n_neurons)
    X_scaled = scaler.transform(X)
    X_scores = pca.transform(X_scaled)
    return X_scores.T


def _finite_difference_time(
    trial_reduced: np.ndarray,
) -> np.ndarray:
    """
    Given (k, t), apply finite difference along time. Note that np.diff already drops the last element of the array, so we return (k, t-1).
    """
    if trial_reduced.ndim != 2:
        raise ValueError("Expected 2D array (k, t)")
    diffed = np.diff(trial_reduced, n=1, axis=1)
    return diffed


def _positions_for_diff_nodes(
    lever_pos: np.ndarray,
    go_cue: int = GO_CUE,
) -> np.ndarray:
    """
    Produce per-timepoint positions aligned with diffed spike data nodes.
    Steps: slice after GO_CUE (-> 10), then average consecutive columns to yield
    9 midpoint positions aligning with np.diff outputs.
    Returns array of shape (2, t_nodes).
    """
    if lever_pos.ndim != 2 or lever_pos.shape[0] != 2:
        raise ValueError("lever_pos must have shape (2, T)")
    pos_go = lever_pos[:, go_cue:]
    if pos_go.shape[1] < 2:
        return np.empty((2, 0), dtype=pos_go.dtype)
    # Midpoints align with finite differences over 1-step intervals
    pos_mid = 0.5 * (pos_go[:, 1:] + pos_go[:, :-1])
    return pos_mid


def _velocities_for_diff_nodes(
    lever_vel: np.ndarray,
    go_cue: int = GO_CUE,
) -> np.ndarray:
    """
    Align kinematic velocity (2, T) to diffed spike nodes.
    Same alignment as positions: slice after GO_CUE, then average neighbors to midpoints.
    Returns (2, t_nodes).
    """
    if lever_vel is None:
        return np.empty((2, 0), dtype=np.float32)
    if lever_vel.ndim != 2 or lever_vel.shape[0] != 2:
        raise ValueError("lever_vel must have shape (2, T)")
    v_go = lever_vel[:, go_cue:]
    if v_go.shape[1] < 2:
        return np.empty((2, 0), dtype=v_go.dtype)
    v_mid = 0.5 * (v_go[:, 1:] + v_go[:, :-1])
    return v_mid
'''
