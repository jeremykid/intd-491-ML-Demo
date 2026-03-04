# Spam Email Classification Demo

Teaching-first PyTorch Lightning demo for Kaggle spam email datasets using reusable preprocessing, a shared `LightningDataModule`, and notebook-first workflows.

## Project Overview

This repo is designed for classroom use. The main path is `notebooks/local_demo.ipynb`, which walks through the `EmbeddingBag` baseline. The repo also supports side-by-side comparison across:

- `embeddingbag`
- `lstm`
- `transformer`

Notebook paths:

- `notebooks/local_demo.ipynb`: baseline-first local walkthrough
- `notebooks/colab_demo.ipynb`: baseline-first Colab walkthrough
- `notebooks/model_comparison_demo.ipynb`: local comparison notebook for all three models

The baseline notebook covers:

- Kaggle token setup
- Dataset download
- robust preprocessing
- `LightningDataModule` construction
- `LightningModule` training
- validation and test metrics
- single-text prediction
- artifact export

The repo does not commit raw Kaggle data, processed data, logs, checkpoints, or Kaggle credentials.

## Repo Structure

```text
.
├── notebooks/
│   ├── local_demo.ipynb
│   ├── colab_demo.ipynb
│   └── model_comparison_demo.ipynb
├── scripts/
│   ├── download_kaggle.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/spam_lightning/
│   ├── config.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── datamodule.py
│   │   └── text_utils.py
│   ├── models/
│   │   ├── backbones.py
│   │   └── lit_model.py
│   └── utils/
│       ├── logging.py
│       ├── paths.py
│       └── seed.py
└── tests/
```

## Local Jupyter Quickstart

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Lab or Notebook from the repo root:

```bash
jupyter lab
```

4. Open `notebooks/local_demo.ipynb`.

## Kaggle Token Setup

Two supported token flows are built into the notebook and scripts.

### Option 1: Manual setup

Place `kaggle.json` at:

```text
~/.kaggle/kaggle.json
```

Then set secure permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Option 2: Upload from notebook

Both notebooks provide a helper to upload `kaggle.json`, copy it into `~/.kaggle/`, and apply `chmod 600`.

## Local Notebook Walkthrough

`notebooks/local_demo.ipynb` is the primary classroom demo. It stays focused on the `embeddingbag` baseline and covers:

1. Installing required packages
2. Setting `KAGGLE_DATASET_SLUG`
3. Uploading or checking Kaggle credentials
4. Downloading the Kaggle dataset into `data/raw/`
5. Inspecting discovered CSV/TSV files and their columns
6. Preprocessing into `data/processed/train.csv`, `val.csv`, and `test.csv`
7. Inspecting `stats.json`
8. Building a `SpamDataModule`
9. Inspecting one EmbeddingBag batch: `tokens`, `offsets`, `labels`
10. Training a Lightning model with callbacks and logging
11. Testing the best checkpoint
12. Predicting on one custom email text
13. Listing artifacts

For side-by-side architecture comparison, use `notebooks/model_comparison_demo.ipynb`.

Before running the notebook, replace:

```python
KAGGLE_DATASET_SLUG = "harshsinha1234/email-spam-classification"
```

with a real Kaggle dataset slug such as `username/spam-email-classification`.

## Colab Walkthrough

`notebooks/colab_demo.ipynb` mirrors the local baseline notebook, with only a few differences:

- Colab-style package installation
- `google.colab.files.upload()` for `kaggle.json`
- default `num_workers=2`
- optional mixed precision when CUDA is available
- repo path assumptions under `/content/intd-491-ML-Demo`

## CLI Commands

Use the same code path outside the notebook. The default model is `embeddingbag`, but you can switch to `lstm` or `transformer` with `--model_name`.

```bash
python scripts/download_kaggle.py --dataset $KAGGLE_DATASET_SLUG
python scripts/preprocess.py --raw_dir data/raw --out_dir data/processed
python scripts/train.py --data_dir data/processed --model_name embeddingbag --max_epochs 5
python scripts/evaluate.py --ckpt artifacts/embeddingbag-seed42/best.ckpt --data_dir data/processed
python scripts/predict.py --ckpt artifacts/embeddingbag-seed42/best.ckpt --text "free money!!!"
python scripts/train.py --data_dir data/processed --model_name lstm --max_epochs 5
python scripts/train.py --data_dir data/processed --model_name transformer --max_epochs 5
```

If auto-detection cannot identify the right raw file or columns, pass explicit overrides:

```bash
python scripts/preprocess.py \
  --raw_dir data/raw \
  --input_csv data/raw/username__dataset/messages.csv \
  --text_col message \
  --label_col label \
  --label_map spam=1 ham=0
```

## Artifacts Produced

Training writes run-scoped artifacts:

- `artifacts/<run_name>/best.ckpt`
- `artifacts/<run_name>/vocab.json`
- `artifacts/<run_name>/config.json`
- `logs/spam_lightning/<run_name>/`

By default, `run_name` is `<model_name>-seed<seed>`. `config.json` captures column choices, label mapping, model architecture, hyperparameters, seed, and artifact paths.

## Tests

Run unit tests with:

```bash
pytest -q
```

The test suite covers:

- regex tokenization and vocabulary behavior
- column detection and label normalization
- EmbeddingBag and sequence batch collation
- backbone output shape validation
- multi-model config restoration

## Troubleshooting

- `Kaggle credentials not found`
  Ensure `~/.kaggle/kaggle.json` exists, or upload it through the notebook helper and retry.

- `Multiple candidate tabular files were found`
  Re-run preprocessing with `--input_csv` and, if needed, `--text_col` / `--label_col`.

- `Could not map labels to binary {0,1}`
  Provide explicit label mapping such as `--label_map spam=1 ham=0`.

- `Missing processed data file`
  Run preprocessing before training or evaluation.

- `embed_dim must be divisible by num_heads`
  Lower `--transformer_num_heads` or choose an `--embed_dim` divisible by that head count.
