"""
LFADS-specific data helpers for macaque reaching experiments.

These utilities reuse the existing trial split logic so LFADS sees the
same train/valid/test trials as other inductive baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

from data_processing.macaque_reaching import MarbleTrialSequence, macaque_prepare_marble_fold_data
from os_utilities import ensure_dir_exists


@dataclass
class LfadsSplitData:
    """
    Container for a single LFADS split and its metadata.
    """

    encod_data: np.ndarray
    recon_data: np.ndarray
    trial_ids: np.ndarray
    condition_ids: np.ndarray


@dataclass
class LfadsFoldData:
    """
    LFADS-ready arrays for a single fold.
    """

    train: LfadsSplitData
    valid: LfadsSplitData
    test: LfadsSplitData
    nodes_per_trial: Optional[int]
    state_dim: int
    n_timepoints: int


def _sequences_to_split(
    sequences: List[MarbleTrialSequence],
    expected_nodes: Optional[int],
) -> Tuple[LfadsSplitData, int]:
    """
    Convert MARBLE trial sequences into LFADS arrays.
    """

    if not sequences:
        empty = np.empty((0, 0, 0), dtype=np.float32)
        split = LfadsSplitData(
            encod_data=empty,
            recon_data=empty,
            trial_ids=np.empty((0,), dtype=np.int64),
            condition_ids=np.empty((0,), dtype=np.int64),
        )
        return split, 0

    arrays: List[np.ndarray] = []
    trial_ids: List[int] = []
    condition_ids: List[int] = []
    timepoints: Optional[int] = None

    for seq in sequences:
        trial_arr = np.asarray(seq.positions, dtype=np.float32)
        if expected_nodes is not None and trial_arr.shape[0] != expected_nodes:
            raise ValueError(
                f"Inconsistent node count ({trial_arr.shape[0]}) for trial {seq.trial_id}; "
                f"expected {expected_nodes}."
            )
        if timepoints is None:
            timepoints = int(trial_arr.shape[0])
        elif int(trial_arr.shape[0]) != timepoints:
            raise ValueError(
                f"Inconsistent timepoints ({trial_arr.shape[0]}) for trial {seq.trial_id}; "
                f"expected {timepoints}."
            )
        arrays.append(trial_arr)
        trial_ids.append(int(seq.trial_id))
        condition_ids.append(int(seq.condition_idx))

    stacked = np.stack(arrays, axis=0)
    split = LfadsSplitData(
        encod_data=stacked,
        recon_data=stacked.copy(),
        trial_ids=np.asarray(trial_ids, dtype=np.int64),
        condition_ids=np.asarray(condition_ids, dtype=np.int64),
    )
    return split, int(timepoints or 0)


def prepare_lfads_fold_data(
    *,
    data_root: str,
    day_index: int | Sequence[int],
    k_folds: int,
    fold_i: int,
    seed: int,
    k_neighbors: int = 30,
    apply_savgol_filter_before_pca: bool = True,
) -> LfadsFoldData:
    """
    Build LFADS arrays for a specific day and fold using existing split logic.
    """

    fold_data = macaque_prepare_marble_fold_data(
        data_root=data_root,
        day_index=day_index,
        k_folds=k_folds,
        fold_i=fold_i,
        seed=seed,
        include_lever_velocity=False,
        k_neighbors=k_neighbors,
        apply_savgol_filter_before_pca=apply_savgol_filter_before_pca,
    )

    expected_nodes = fold_data.nodes_per_trial
    train_split, train_T = _sequences_to_split(fold_data.train_trials, expected_nodes)
    valid_split, valid_T = _sequences_to_split(fold_data.valid_trials, expected_nodes)
    test_split, test_T = _sequences_to_split(fold_data.test_trials, expected_nodes)

    timepoints = max(train_T, valid_T, test_T)
    if timepoints == 0:
        raise ValueError("No trials found for LFADS fold data.")

    return LfadsFoldData(
        train=train_split,
        valid=valid_split,
        test=test_split,
        nodes_per_trial=expected_nodes,
        state_dim=int(train_split.encod_data.shape[-1]) if train_split.encod_data.ndim == 3 else 0,
        n_timepoints=timepoints,
    )


def write_lfads_hdf5(
    *,
    fold_data: LfadsFoldData,
    output_path: Path,
    include_test: bool = True,
) -> None:
    """
    Write LFADS-ready arrays to an HDF5 file.
    """

    output_path = output_path.expanduser().resolve()
    ensure_dir_exists(str(output_path.parent), raise_exception=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("train_encod_data", data=fold_data.train.encod_data)
        h5f.create_dataset("train_recon_data", data=fold_data.train.recon_data)
        h5f.create_dataset("valid_encod_data", data=fold_data.valid.encod_data)
        h5f.create_dataset("valid_recon_data", data=fold_data.valid.recon_data)
        h5f.create_dataset("train_trial_ids", data=fold_data.train.trial_ids)
        h5f.create_dataset("valid_trial_ids", data=fold_data.valid.trial_ids)
        h5f.create_dataset("train_condition_ids", data=fold_data.train.condition_ids)
        h5f.create_dataset("valid_condition_ids", data=fold_data.valid.condition_ids)

        if include_test:
            h5f.create_dataset("test_encod_data", data=fold_data.test.encod_data)
            h5f.create_dataset("test_recon_data", data=fold_data.test.recon_data)
            h5f.create_dataset("test_trial_ids", data=fold_data.test.trial_ids)
            h5f.create_dataset("test_condition_ids", data=fold_data.test.condition_ids)

