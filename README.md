# SMAHD

SMAHD is a scalable graph autoencoder framework for high-resolution spatial multi-omics integration. It uses view-specific graph encoders, graph-attention layers by default, weighted multi-view reconstruction, and micro-cluster mini-batch training to integrate transcriptomic, proteomic, and epigenomic spatial modalities.

This repository contains the source code, tutorials, and reproducibility scripts used for the SMAHD manuscript.

## Installation

```bash
git clone https://github.com/Little-Eel/SMAHD.git
cd SMAHD
conda create -n env_SMAHD python=3.8
conda activate env_SMAHD
pip install -r requirements.txt
```

PyTorch Geometric packages can be CUDA-version specific. If installation from `requirements.txt` fails, install `torch`, `torch-geometric`, `torch-cluster`, `torch-scatter`, and `torch-sparse` following the official PyTorch Geometric instructions for your CUDA/PyTorch version.

## Repository Contents

- `model/`: SMAHD model, training, preprocessing, and utility code.
- `Tutorial*.ipynb`: example notebooks for simulated and real spatial multi-omics datasets.
- `experiments_smahd_ablation_simulated.py`: simulated module-wise ablation.
- `smahd_realdata_ablation.py`: real-data ablation across RNA+ADT and RNA+ATAC datasets.
- `smahd_scaling_simulated.py`: simulated scaling benchmark.
- `smahd_biological_validation.py`: marker-level biological validation scripts.
- `summarize_*.py`: scripts that summarize experiment outputs into CSV, Markdown, LaTeX tables, and figures.
- `experiments/`: lightweight manuscript result summaries and plots. Large embeddings and generated datasets are intentionally excluded.

## Reproducibility Scripts

Simulated ablation:

```bash
python experiments_smahd_ablation_simulated.py --output-dir experiments/ablation_simulated_5seed --seeds 11,22,33,44,55 --epochs 80
python summarize_smahd_ablation.py --output-dir experiments/ablation_simulated_5seed
```

Simulated scaling benchmark:

```bash
python smahd_scaling_simulated.py --output-dir experiments/scaling_simulated --sizes 2000,5000,10000,20000
```

Real-data analyses require local copies of the corresponding public datasets. By default, scripts look under `data/`. You can instead set:

```bash
export SMAHD_DATA_ROOT=/path/to/smahd_data
```

or edit the dataset paths in `smahd_realdata_ablation.py` to match your local layout.

Example:

```bash
python smahd_realdata_ablation.py --datasets tonsil_s1,tonsil_s2,hln_A1,hln_D1 --output-dir experiments/realdata_ablation
python summarize_smahd_realdata_ablation.py --output-dir experiments/realdata_ablation
```

## Data

The repository does not include raw spatial multi-omics data or large generated embeddings. Public datasets should be downloaded from the sources cited in the manuscript and placed in a local data directory before running the real-data scripts.

## Citation

If you use SMAHD, please cite the accompanying manuscript.
