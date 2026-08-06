# Colab Neural Training Runbook

Use this when running the full neural 1X2 model outside the local MacBook.

## Colab Setup

```python
!pip install pandas pyarrow duckdb s3fs scikit-learn tensorflow optuna joblib
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

Use `--league` for normal Colab runs. Supported aliases include `eng1`, `epl`, `eng_1`, `eng2`, `eng3`, `esp1`, `esp2`, `ger1`, `ger2`, `ita1`, `ita2`, `fra1`, `fra2`, `ned1`, `por1`, `sco1`, and `all`.

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

That creates one combined multi-league model. To train one separate model per league and keep outputs organized by league, use `--train-each-league`:

```bash
python scripts/colab/train_soccer_three_way_nn_optuna.py \
  --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
  --league all \
  --train-each-league \
  --full-scope-start-season 2021 \
  --train-through-season 2024 \
  --test-season 2025 \
  --epochs 50 \
  --n-trials 25 \
  --next-games 5 \
  --output-dir outputs/soccer_nn_by_league
```

This writes:

```text
outputs/soccer_nn_by_league/batch_plan.json
outputs/soccer_nn_by_league/batch_summary.json
outputs/soccer_nn_by_league/eng1_soccer_nn/
outputs/soccer_nn_by_league/ger1_soccer_nn/
outputs/soccer_nn_by_league/ita1_soccer_nn/
...
```

Each league folder includes the full held-out test predictions plus a compact next-games CSV:

```text
test_predictions.parquet
next_5_games_predictions.csv
next_5_games_summary.json
```

`next_5_games_predictions.csv` is ordered by kickoff time within that league's held-out test season. Use `--next-games 10` for ten games or `--next-games 0` to skip the compact export.

`--league all --train-each-league` discovers regular domestic league IDs only, such as `eng.1` and `ger.1`. Cup and UEFA competitions are skipped by default; pass them explicitly with `--competitions` if you want separate models for them.

Current full `2021-2025` regular league scope in DigitalOcean:

```text
aut.1, den.1, eng.1, eng.2, eng.3, esp.1, esp.2, fra.1,
fra.2, ger.1, ger.2, ita.1, ita.2, ned.1, por.1, sco.1
```

Partial-scope regular leagues found but not full `2021-2025`:

```text
jpn.1, mex.1, nor.1, swe.1, usa.1
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

For a global domestic-league model, use the plain global output folder:

```bash
python scripts/colab/train_soccer_three_way_nn_optuna.py \
  --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
  --league all \
  --full-scope-start-season 2021 \
  --train-through-season 2024 \
  --test-season 2025 \
  --max-features 800 \
  --n-folds 4 \
  --val-size 250 \
  --min-train-size 1200 \
  --epochs 50 \
  --n-trials 25 \
  --next-games 5 \
  --duckdb-preprocess \
  --output-dir outputs/soccer_nn_global
```

With `--league all` and no explicit `--competitions`, the global run now targets full-scope regular domestic leagues only and defaults seasons to `2021` through `--test-season`. This avoids accidentally loading every lake partition, including cups, friendlies, UEFA competitions, partial 2026 partitions, and older one-off seasons.

`--duckdb-preprocess` is recommended for global runs. DuckDB scans the parquet files, filters competition/season partitions, casts unstable columns, and returns only required plus numeric model columns to pandas. This reduces Colab memory pressure before feature selection and TensorFlow training.

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

The exporter auto-enriches team names from the matching gold partition when DigitalOcean credentials are set. If `test_predictions.parquet` was created before team names were included, you can also pass the gold input explicitly:

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

## Draw Threshold Tuning

Before applying draw-risk logic to future fixtures, tune thresholds on held-out test predictions:

```bash
python scripts/colab/tune_draw_thresholds.py \
  --predictions outputs/eng1_soccer_nn/test_predictions.parquet \
  --output-dir outputs/eng1_soccer_nn/draw_threshold_tuning \
  --policy-version draw_policy_eng1_2025_backtest_v1
```

This writes:

```text
draw_threshold_grid.csv
balanced_draw_threshold_candidates.csv
best_draw_thresholds.json
active_draw_policy.json
```

Use `balanced_draw_threshold_candidates.csv` to choose thresholds that improve draw capture without materially reducing total pick accuracy.

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

To pull future fixtures for every full-scope league into league-specific folders:

```bash
python scripts/colab/fetch_espn_fixtures.py \
  --league all \
  --fetch-each-league \
  --start-date 2026-08-01 \
  --end-date 2026-09-30 \
  --weeks 5 \
  --output-dir outputs/fixtures/espn_by_league_2026
```

This writes:

```text
outputs/fixtures/espn_by_league_2026/batch_summary.json
outputs/fixtures/espn_by_league_2026/eng1_fixtures/eng.1_first_5_weeks_fixtures.csv
outputs/fixtures/espn_by_league_2026/ger1_fixtures/ger.1_first_5_weeks_fixtures.csv
...
```

ESPN may not expose every lower league consistently. Batch mode records unavailable league endpoints under `failures` in `batch_summary.json` and keeps the successful fixture pulls.

## Future Fixture Predictions

After `eng1` training finishes and future fixtures are fetched, score the first five EPL weeks:

```bash
python scripts/colab/predict_future_fixtures.py \
  --fixtures outputs/fixtures/espn_eng1_2026/eng.1_first_5_weeks_fixtures.csv \
  --model outputs/eng1_soccer_nn/soccer_three_way_nn.keras \
  --preprocessing outputs/eng1_soccer_nn/preprocessing.joblib \
  --draw-policy configs/draw_policy_eng1.json \
  --history-partitions eng.1:2025,eng.2:2025,eng.3:2025 \
  --output-dir outputs/eng1_soccer_nn/future_2026
```

This writes:

```text
future_predictions.csv
future_model_input.parquet
future_feature_diagnostics.csv
summary.json
```

This is a first-pass future inference bridge. It uses each team's latest historical snapshot from the listed history partitions and imputes model features that are not available for future fixtures yet.

Future prediction outputs include both the raw neural pick and draw-policy recommendation:

```text
raw_model_pick
recommended_pick
draw_risk
draw_gap
home_away_gap
draw_policy_version
```

## Explanation Graph

Build a matchup-filterable graph for future predictions:

```bash
python scripts/colab/build_prediction_explanation_graph.py \
  --predictions outputs/eng1_soccer_nn/future_2026/future_predictions.csv \
  --model-input outputs/eng1_soccer_nn/future_2026/future_model_input.parquet \
  --model outputs/eng1_soccer_nn/soccer_three_way_nn.keras \
  --preprocessing outputs/eng1_soccer_nn/preprocessing.joblib \
  --draw-policy configs/draw_policy_eng1.json \
  --output-dir outputs/eng1_soccer_nn/future_2026/explanations
```

This writes:

```text
prediction_explanation_graph.html
prediction_explanation_graph.json
feature_contributions.csv
```

Open `prediction_explanation_graph.html` in Colab or download it. The page has a matchup dropdown so each graph can be filtered by fixture.
