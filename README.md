# MIGO

Mapping information flow along the central dogma across genome topology, epigenetic states, transcription, and protein abundance has revealed cellular heterogeneity and biological function. However, existing multi-omics sequencing approaches capture only limited modalities in each experiment, failing to profile the full spectrum simultaneously. A further challenge lies in learning universal representations that integrate these disparate data. Here, we present MIGO, a unified generative framework that models five modalities including scRNA-seq, scATAC-seq, scHi-C, ribosome profiling, and protein abundance within a single architecture. MIGO leverages mutual information to learn biological states and maps them into a discrete, shared space using vector quantization. Evaluated on 20 datasets, MIGO demonstrates superior performance in cross-modality generation, batch effect correction, higher-order regulatory units identification, and marker protein discovery. Notably, MIGO is advanced in generating high-resolution three-dimension (3D) chromatin contact maps from scRNA-seq alone. In addition, we dissect cell type-specific translation in human peripheral blood mononuclear cells. Together, MIGO is flexible for cross-modality generation and constructs a virtual space for further AI virtual cells (AIVCs) development.

## Project layout
- `migo/` core package
  - `migo/model/` shared layers, datasets, metrics, and utilities
  - `migo/utils_/` shared preprocessing and evaluation utilities
  - `migo/tasks/<task>/` task-specific `train.py`, `infer.py`, `pretrain.py`
  - `migo/registry.py` task registry and default config resolution
- `configs/` JSON configs for each task and resolution
- `train.py`, `infer.py`, `run.py` unified CLI entrypoints

## Quick start
Install dependencies, then run:
```bash
python train.py --task scRNA_scATAC --config configs/scRNA_scATAC.json
python infer.py --task scRNA_scATAC --config configs/scRNA_scATAC.json
```

## Unified training and inference
Training:
```bash
python train.py --task scRNA_scATAC --config configs/scRNA_scATAC.json
python train.py --task scRNA_ADT --config configs/scRNA_ADT.json
python train.py --task RNA_Ribo_seq --config configs/RNA_Ribo_seq.json
python train.py --task scRNA_scHiC --resolution 1M
```

Inference:
```bash
python infer.py --task scRNA_scATAC --config configs/scRNA_scATAC.json
python infer.py --task scRNA_ADT --config configs/scRNA_ADT.json
python infer.py --task RNA_Ribo_seq --config configs/RNA_Ribo_seq.json
python infer.py --task scRNA_scHiC --resolution 1M
```

## Common CLI overrides
You can override common config fields directly from the CLI (these override the selected `train` or `infer` section):
```bash
python train.py --task scRNA_scATAC --config configs/scRNA_scATAC.json \
  --device cuda:0 --seed 123 --rna-h5ad /path/to/rna.h5ad --atac-h5ad /path/to/atac.h5ad
```

Available overrides:
- `--device` GPU/CPU device, e.g. `cuda:0` or `cpu`
- `--seed` random seed
- `--rna-h5ad`, `--atac-h5ad`, `--adt-h5ad`, `--ribo-h5ad`, `--hic-h5ad`
- `--batch-size`, `--num-workers`
- `--train-batch-size`, `--val-batch-size` (scRNA_scHiC)
- `--model-save-path`, `--log-dir`

## Training and inference flow
1) Prepare data and set paths in the config JSONs (or override via CLI).
2) Run training with `train.py` (or `run.py --mode train`).
3) Checkpoints and logs are written under `model_save_path` and `log_dir`.
4) Run inference with `infer.py` (or `run.py --mode infer`) using the same config.
5) Evaluation outputs are written by each task script in its configured output path.

## Data preprocessing
Preprocessing helpers live in `migo/utils_/data_processed.py`. Use them to build the `h5ad` inputs referenced by configs:
- RNA (ADT/scATAC/scHiC/Ribo): size-factor normalization, optional log1p, HVG selection, PCA, and optional outlier clipping (`RNA_data_preprocessing_*`).
- ATAC: TF-IDF + LSI with optional peak filtering (`ATAC_data_preprocessing`, `TFIDF`, `lsi`).
- Hi-C: matrix-specific preprocessing for scHiC inputs (`RNA_data_preprocessing_schic`).
- Ribo: CLR/standardized Ribo preprocessing plus paired RNA-Ribo utilities (`preprocess_ribo_data`, `RNA_Ribo_unified_preprocessing`).

## Configs
Config files are under `configs/`:
- `configs/scRNA_scATAC.json`
- `configs/scRNA_ADT.json`
- `configs/RNA_Ribo_seq.json`
- `configs/scRNA_scHiC_1M.json`
- `configs/scRNA_scHiC_50K.json`
