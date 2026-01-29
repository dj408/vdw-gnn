"""
Parallel processing script for computing and saving ESCGNN 
Data objects, or just their P and Q sparse tensor data 
(if `proc_kwargs['return_data_object'] == False`).

To add a dataset, do:
(1) import its module at top
(2) add logic for its `dataset_key` in `init_dataset` 
    and the "__main__" call

Example call:
python3 "./scripts/process_vec_feat_data.py" \
--root "/path/to/file" \
--dataset_key "DATASET_KEY" \
--use_mean_recentering True \
--num_workers 4 \
--sample_n 100
"""

# basic modules
import os
import sys
sys.path.insert(0, './')
# sys.path.insert(0, '../')
import time
import pickle
import argparse
import multiprocessing as mp
from typing import Tuple, List, Dict, Any, Optional

# numpy, pytorch modules
from numpy.random import RandomState
from torch_geometric.data import Data

# our modules
import process_pyg_data as process


# define functions used in parallel processing
def init_dataset(
    root_dir: str, 
    dataset_key: str
) -> None:
    global dataset
    global proc_kwargs

    raise NotImplementedError(
        f"Dataset '{dataset_key}' not yet supported. Add a loader in init_dataset()."
    )


def process_one(i) -> Data:
    import process_pyg_data as process
    return process.process_pyg_data(
        dataset[i], 
        data_i=i,
        **proc_kwargs
    )


def parallel_process_dataset(
    root: str,
    dataset_key: str,
    idx: List[int],
    num_workers: int = 4
) -> List[Data]:
    # 'fork' is safe on Linux/Mac; not Windows
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=num_workers,
        initializer=init_dataset,
        initargs=(root, dataset_key)
    ) as pool:         
        data_list = pool.map(process_one, idx)
    return data_list


# run processing in __main__
if __name__ == "__main__":
    
    # command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--root', default=None, type=str, 
        help="root directory where dataset is saved"
    )
    parser.add_argument(
        '--tensor_data_save_dir', default='pq_tensor_data', type=str, 
        help="subdirectory of 'root' for saving tensor data files"
    )
    parser.add_argument(
        '-d', '--dataset_key', default=None, type=str, 
        help="dataset key for the loader in init_dataset()"
    )
    parser.add_argument(
        '--use_mean_recentering', default=False, type=bool, 
        help="use mean centering for vector features"
    )
    parser.add_argument(
        '-w', '--num_workers', default=4, type=int, 
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        '-s', '--sample_n', default=None, type=int, 
        help="number of samples to process; if None, entire dataset will be processed"
    )
    parser.add_argument(
        '--seed', default=542920, type=int, 
        help="random seed for dataset subsampling"
    )
    parser.add_argument(
        '--local_pca_distance_kernel', type=str, default='chemical',
        choices=['chemical', 'gaussian'],
        help='Distance kernel function to use (default: chemical)'
    )
    clargs = parser.parse_args()

    
    # output save directory
    save_dir = os.path.join(clargs.root, clargs.tensor_data_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    
    # init vars made global in parallel processing (in `init_dataset`)
    dataset = None
    kernel_fn = 'K_chemical_weights' \
        if clargs.local_pca_distance_kernel == 'chemical' \
        else 'K_Gaussian_weights'
    proc_kwargs = {
        'use_mean_recentering': clargs.use_mean_recentering,
        'enforce_sign': True,
        'device': 'cpu',
        'return_data_object': False, # returns P and Q sparse coo components in nested dict
        'tensor_data_save_dir': save_dir,
        'kernel_fn': kernel_fn,
    }
    
    # generate index list of graphs to process in parallel
    # first, need to lazy-load dataset metadata to get its length
    raise NotImplementedError(
        f"Dataset '{clargs.dataset_key}' not yet supported. Add a loader in init_dataset()."
    )

    # generate index
    if clargs.sample_n is None:
        idx = list(range(dataset_len))
    else:
        rand_state = RandomState(seed=clargs.seed)
        idx = rand_state.randint(0, dataset_len, size=clargs.sample_n).tolist()

    # execute parallel processing
    time_start = time.time()
    # data_list = parallel_process_dataset(
    _ = parallel_process_dataset(
        root=clargs.root, 
        dataset_key=clargs.dataset_key,
        idx=idx,
        num_workers=clargs.num_workers,
    )

    # option 1: save P and Q sparse tensor data (list of nested dicts)
    # if not proc_kwargs['return_data_object']:
    #     from torch import save
    #     save_dir = os.path.join(clargs.root, "sparse_attr_tensors")
    #     os.makedirs(save_dir, exist_ok=True)
    #     save(data_list, os.path.join(save_dir, f"Ps_and_Qs_dictlist.pt"))

    time_elapsed = time.time() - time_start
    time_min, time_sec = time_elapsed // 60, time_elapsed % 60
    print(
        f"\nProcessing time ({len(idx)} graphs, {clargs.num_workers} workers): "
        f"{time_min} min, {time_sec:.2f} sec"
    )
