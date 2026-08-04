# Colab Neural Training Runbook

Use this when running the full neural 1X2 model outside the local MacBook.

## Colab Setup

```python
!pip install pandas pyarrow s3fs scikit-learn tensorflow optuna joblib
```

Set DigitalOcean credentials in Colab secrets or environment variables:

```python
import os
os.environ["DO_SPACES_KEY"] = "..."
os.environ["DO_SPACES_SECRET"] = "..."
os.environ["DO_SPACES_ENDPOINT"] = "https://fra1.digitaloceanspaces.com"
os.environ["DO_SPACES_REGION"] = "fra1"
```

Upload or copy this script into Colab:

```text
scripts/colab/train_soccer_three_way_nn_optuna.py
```

If using the full repo in Colab, run it from the repo root.

## Smoke Run

```bash
python scripts/colab/train_soccer_three_way_nn_optuna.py \
  --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
  --output-dir outputs/smoke_soccer_nn \
  --smoke
```

By default, `--smoke` uses `--league eng1` and seasons `2021-2025`.

## League Runs

Use `--league` for normal Colab runs. Supported aliases include `eng1`, `epl`, `eng_1`, `esp1`, `ger1`, `ita1`, `fra1`, and `all`.

```bash
python scripts/colab/train_soccer_three_way_nn_optuna.py \
  --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
  --league eng1 \
  --train-through-season 2024 \
  --test-season 2025 \
  --epochs 50 \
  --n-trials 25 \
  --output-dir outputs/eng1_soccer_nn
```

Use `all` to train on every available league:

```bash
python scripts/colab/train_soccer_three_way_nn_optuna.py \
  --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
  --league all \
  --train-through-season 2024 \
  --test-season 2025 \
  --epochs 50 \
  --n-trials 25 \
  --output-dir outputs/all_soccer_nn
```

## Full Run

```bash
python scripts/colab/train_soccer_three_way_nn_optuna.py \
  --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
  --league all \
  --seasons 2021,2022,2023,2024,2025 \
  --train-through-season 2024 \
  --test-season 2025 \
  --max-features 800 \
  --n-folds 4 \
  --val-size 250 \
  --min-train-size 1200 \
  --epochs 150 \
  --n-trials 50 \
  --output-dir outputs/soccer_three_way_nn_full
```

## Artifacts

The script writes:

```text
soccer_three_way_nn.keras
preprocessing.joblib
test_predictions.parquet
optuna_trials.csv
feature_columns.json
summary.json
```

Upload these artifacts back to DigitalOcean after the run under:

```text
s3://betboard-ml-artifacts/soccer-prediction-data/models/neural_three_way/run_id=<run_id>/
```
