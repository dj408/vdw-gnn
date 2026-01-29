# VDW-GNNs: Vector diffusion wavelets for geometric graph neural networks

A PyTorch/PyTorch-Geometric framework for building (optionally) rotationally-equivariant graph neural networks that utilize diffusion wavelets for both scalar and vector node features.

---
## 1&nbsp;&nbsp;Required packages
Core dependencies:
- python>=3.11
- pytorch
- torch-geometric
- torchmetrics
- accelerate
- numpy
- scikit-learn
- scipy
- h5py
- pyyaml

Optional dependencies:
- torch-scatter[1]
- torch-cluster[1]
- torch-sparse[1]
- e3nn (for Tensor Field Networks models, etc.)
- wandb
- pandas
- matplotlib
- pot [used by MARBLE]
- cebra
- statannotations

[1] Can be installed as a dependency of pytorch-geometric. See the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

An example installation inside a conda environment:
```
mamba install -c pytorch -c nvidia -c conda-forge python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.4 numpy scikit-learn pandas matplotlib h5py pyyaml -y && \
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric --extra-index-url https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
pip install torchmetrics accelerate
```

For wind data experiments:
```mamba install -c conda-forge xarray netCDF4``` [for reading '.nc' data files]

For macaque reaching experiments (baselines):
```pip install cebra```
```mamba install pot``` [used by MARBLE]

For results plots:
```mamba install -c conda-forge statannotations``` [will also install seaborn, statsmodels, etc.]

---
## 2&nbsp;&nbsp;Data sets

### 2.1 Ellipsoids
These are synthetic datasets that can be easily reproduced with relevant code in the `data_processing` folder of this repo.

Example: generate an ellipsoids dataset and compute P/Q operators:
```bash
python scripts/python/generate_ellipsoid_dataset.py \
  --save_dir /path/to/data/ellipsoids \
  --config /path/to/code/config/yaml_files/ellipsoids/experiment.yaml \
  --pq_h5_name pq_tensor_data_512.h5 \
  --random_seed 457892 \
  --num_samples 512 \
  --num_nodes_per_graph 128 \
  --knn_graph_k 5 \
  --abc_means 3.0 1.0 1.0 \
  --abc_stdevs 0.5 0.2 0.2 \
  --local_pca_kernel_fn gaussian \
  --laplacian_type sym_norm \
  --num_oversample_points 1024 \
  --k_laplacian 10 \
  --dirac_types max min \
  --modulation_scale 0.9 \
  --harmonic_bands 1-3,6-8,10-12 \
  --harmonic_band_weights 1.0 1.5 2.0 \
  --global_band_index 1 \
  --random_harmonic_k 16 \
  --random_harmonic_coeff_bounds 1.0 2.0 \
  --random_harmonic_smoothing_window 0 \
  --sing_vect_align_method column_dot
```

Example (alternate path layout):
```bash
python scripts/python/generate_ellipsoid_dataset.py \
  --save_dir /path/to/project/data/ellipsoids \
  --config /path/to/project/code/config/yaml_files/ellipsoids/experiment.yaml \
  --pq_h5_name pq_tensor_data_512.h5 \
  --random_seed 457892 \
  --num_samples 512 \
  --num_nodes_per_graph 128 \
  --knn_graph_k 5 \
  --abc_means 3.0 1.0 1.0 \
  --abc_stdevs 0.5 0.2 0.2 \
  --local_pca_kernel_fn gaussian \
  --laplacian_type sym_norm \
  --num_oversample_points 1024 \
  --k_laplacian 10 \
  --dirac_types max min \
  --modulation_scale 0.9 \
  --harmonic_bands 1-3,6-8,10-12 \
  --harmonic_band_weights 1.0 1.5 2.0 \
  --global_band_index 1 \
  --random_harmonic_k 16 \
  --random_harmonic_coeff_bounds 1.0 2.0 \
  --random_harmonic_smoothing_window 0 \
  --sing_vect_align_method column_dot
```

### 2.2 Earth surface wind velocity, 1 January 2016
Downloaded from the NOAA Physical Science Lab data repo: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html (accessed December 2025).

You will need to download the u- and v- wind measurements separately from this site. First, identify the data set with these attributes:

| Variable         | Statistic | Level | TimeScale |
|------------------|-----------|-------|-----------|
| u-wind \| v-wind | Mean      | 10 m  | Daily     |

Then, for each, click the data set's 'Plot/subset all files' icon (which looks like a data plot) in the Options column of the data set's row. This opens a data set explorer page, where you can subset to the single date, and then click 'Download subset of data defined by options' to download an `.nc` file. For a single day, these are not large files (~100 kilobytes).


### 2.3 Macaque reaching data, accessed from the MARBLE (Gosztolai et al. 2025) data repo:
- Trial data: https://dataverse.harvard.edu/api/access/datafile/6969883
- Kinematics (target) data: https://dataverse.harvard.edu/api/access/datafile/6969885
- Trial ids: https://dataverse.harvard.edu/api/access/datafile/6963200

---
## 3&nbsp;&nbsp;Experiment configuration precedence
Configuration values are layered in this order (highest priority first):
1. Command-line arguments (only if the flag is provided)
2. Model YAML (the file passed to `--config`)
3. Experiment YAML in the same directory (`experiment.yaml` or `<dir>.yaml`)
4. Defaults in `config/*_config.py`

In the layered YAML case, the experiment YAML is loaded first and then the
model YAML overlays it. CLI flags override the merged YAML after loading.
If `--config` points directly to an experiment YAML, no model YAML is auto-loaded.
Any keys not specified in YAML or CLI fall back to defaults in the *_config class files.

---
## 4&nbsp;&nbsp;Dataset-specific training examples (VDW/ESCGNN)
Ellipsoids:
```bash
python scripts/python/main_training.py \
  --config config/yaml_files/ellipsoids/escgnn.yaml
```

Wind:
```bash
python scripts/python/run_wind_experiments.py \
  --config config/yaml_files/wind/escgnn.yaml \
  --root_dir . \
  --replications 5 \
  --knn_k 3 \
  --local_pca_k 10 \
  --sample_n 400 \
  --mask_prop 0.1 \
  --do_rotation_eval
```

Macaque:
```bash
python scripts/python/run_macaque_multiday_cv.py \
  --model escgnn \
  --days 0-43 \
  --root_dir /path/to/project
```

---
## 5&nbsp;&nbsp;SLURM scripts
Example SLURM launchers are available in `scripts/slurm/` (e.g., `train.sh`, `train_cv_multi.sh`, `run_wind.sh`, `macaque_kfold_multiday.sh`, `test.sh`). Customize job resources before submitting.

---
## 6&nbsp;&nbsp;Results aggregation helpers
- For wind experiments, use `scripts/python/summarize_wind_tables.py` to aggregate and format results tables.  
- For macaque experiments, use `scripts/python/summarize_day_model_results.py` to aggregate per-day model results and generate tables; then, `scripts/python/plot_macaque_wilcoxon.py` generates the Wilcoxon paired tests plot from the summary_records_mean_std_.pkl` file created.

---