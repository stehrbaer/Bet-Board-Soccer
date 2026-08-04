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

The script writes these files early, before the model is complete:

```text
run_config.json
training.log
load_status.json
load_complete.json
split_profile.json
feature_selection_status.json
dataset_profile.json
feature_columns.json
optuna_trials.csv
best_trial_so_far.json
```

The final successful run also writes:

```text
soccer_three_way_nn.keras
preprocessing.joblib
test_predictions.parquet
summary.json
```

Upload these artifacts back to DigitalOcean after the run under:

```text
s3://betboard-ml-artifacts/soccer-prediction-data/models/neural_three_way/run_id=<run_id>/
```

## First Weeks Export

After a league run finishes, export the first five weeks of predictions:

```bash
python scripts/colab/export_first_weeks_predictions.py \
  --predictions outputs/eng1_soccer_nn/test_predictions.parquet \
  --output-dir outputs/eng1_soccer_nn \
  --weeks 5
```

If `test_predictions.parquet` was created before team names were included, enrich it from the gold input:

```bash
python scripts/colab/export_first_weeks_predictions.py \
  --predictions outputs/eng1_soccer_nn/test_predictions.parquet \
  --gold-input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input/competition=eng.1/season=2025/part-000.parquet \
  --output-dir outputs/eng1_soccer_nn \
  --weeks 5
```

This writes:

```text
first_5_weeks_predictions.csv
first_5_weeks_summary.json
```

The CSV includes `home_team_name`, `away_team_name`, and `matchup_key`.

## Future Fixtures

Fetch the first five weeks of the 2026-27 EPL schedule from ESPN:

```bash
python scripts/colab/fetch_espn_fixtures.py \
  --league eng1 \
  --start-date 2026-08-21 \
  --end-date 2026-09-30 \
  --weeks 5 \
  --output-dir outputs/fixtures/espn_eng1_2026
```

This writes raw ESPN JSON plus normalized fixture CSVs. These fixtures still need prematch features before the neural model can score them.
