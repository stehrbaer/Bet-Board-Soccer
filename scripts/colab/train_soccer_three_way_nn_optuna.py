#!/usr/bin/env python3
"""Colab-ready neural three-way soccer training script.

Target contract:
    0 = home win
    1 = draw
    2 = away win

Example Colab run:
    !pip install pandas pyarrow s3fs scikit-learn tensorflow optuna joblib
    !python train_soccer_three_way_nn_optuna.py \
      --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
      --competitions eng.1,esp.1,ger.1,ita.1,fra.1 \
      --train-through-season 2024 \
      --test-season 2025 \
      --n-trials 50 \
      --epochs 150
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


LOGGER = logging.getLogger("soccer_nn")

LEAGUE_ALIASES = {
    "eng1": "eng_1",
    "epl": "eng_1",
    "premierleague": "eng_1",
    "eng2": "eng_2",
    "championship": "eng_2",
    "esp1": "esp_1",
    "laliga": "esp_1",
    "ger1": "ger_1",
    "bundesliga": "ger_1",
    "ita1": "ita_1",
    "seriea": "ita_1",
    "fra1": "fra_1",
    "ligue1": "fra_1",
}


try:
    from betboard_soccer_extension.modeling.nn_preprocessing import (
        TARGET_LABELS,
        choose_numeric_feature_columns,
        make_walkforward_folds,
        prepare_fold,
    )
except ModuleNotFoundError:
    # Allows copying this file into Colab without installing the package, as long
    # as the repo root or src directory is present beside the script.
    if "__file__" in globals():
        candidates = [Path(__file__).resolve().parents[2] / "src"]
    else:
        candidates = [Path.cwd() / "src", Path.cwd()]
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    from betboard_soccer_extension.modeling.nn_preprocessing import (  # type: ignore[no-redef]
        TARGET_LABELS,
        choose_numeric_feature_columns,
        make_walkforward_folds,
        prepare_fold,
    )


@dataclass
class TrainingConfig:
    input: str
    output_dir: str
    competitions: list[str]
    seasons: list[str]
    train_through_season: str
    test_season: str
    max_features: int
    n_folds: int
    val_size: int
    min_train_size: int
    epochs: int
    n_trials: int
    batch_size_default: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Optuna-tuned neural soccer 1X2 model.")
    parser.add_argument("--input", required=True, help="Gold dataset root or local parquet file.")
    parser.add_argument("--output-dir", default="outputs/soccer_three_way_nn")
    parser.add_argument(
        "--league",
        default="all",
        help="League selector such as eng1, eng_1, epl, esp1, ger1, ita1, fra1, or all.",
    )
    parser.add_argument("--competitions", default="", help="Comma-separated competition ids, e.g. eng.1,esp.1")
    parser.add_argument("--seasons", default="", help="Comma-separated seasons to load. Empty loads all under input.")
    parser.add_argument("--train-through-season", default="2024")
    parser.add_argument("--test-season", default="2025")
    parser.add_argument("--max-features", type=int, default=800)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--min-train-size", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Override settings for a fast local/Colab smoke run.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_league(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(".", "_").replace(" ", "")
    if normalized == "all":
        return "all"
    return LEAGUE_ALIASES.get(normalized, normalized)


def resolve_competitions(args: argparse.Namespace) -> list[str]:
    competitions = csv_list(args.competitions)
    if competitions:
        return [normalize_league(comp) for comp in competitions]
    leagues = [normalize_league(league) for league in csv_list(args.league)]
    if not leagues or leagues == ["all"]:
        return []
    if "all" in leagues:
        raise SystemExit("--league all cannot be combined with specific leagues.")
    return leagues


def default_partition_seasons(train_through_season: str, test_season: str) -> str:
    start = 2021
    try:
        end = max(int(train_through_season), int(test_season))
    except ValueError:
        return ""
    return ",".join(str(season) for season in range(start, end + 1))


def s3_storage_options() -> dict[str, Any] | None:
    key = os.getenv("DO_SPACES_KEY")
    secret = os.getenv("DO_SPACES_SECRET")
    if not key or not secret:
        return None
    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {
            "endpoint_url": os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com"),
            "region_name": os.getenv("DO_SPACES_REGION", "fra1"),
        },
    }


def s3_filesystem():
    storage_options = s3_storage_options()
    if storage_options is None:
        raise SystemExit(
            "DigitalOcean Spaces credentials are missing. In Colab set:\n"
            '  os.environ["DO_SPACES_KEY"] = "your_access_key"\n'
            '  os.environ["DO_SPACES_SECRET"] = "your_secret_key"\n'
            'Optional endpoint defaults to https://fra1.digitaloceanspaces.com.'
        )
    try:
        import s3fs
    except ModuleNotFoundError as exc:
        raise SystemExit("s3fs is missing. In Colab run: !pip install -r requirements-colab.txt") from exc
    return s3fs.S3FileSystem(**storage_options)


def s3_key(path: str) -> str:
    return path.removeprefix("s3://")


def partition_paths(root: str, competitions: list[str], seasons: list[str]) -> list[str]:
    if not competitions or not seasons:
        return [root]
    return [f"{root.rstrip('/')}/competition={comp}/season={season}/part-000.parquet" for comp in competitions for season in seasons]


def read_parquet_path(path: str, filesystem=None) -> pd.DataFrame:
    if filesystem is None:
        return pd.read_parquet(path)
    with filesystem.open(s3_key(path), "rb") as handle:
        return pd.read_parquet(handle)


def load_gold_dataset(root: str, competitions: list[str], seasons: list[str]) -> pd.DataFrame:
    filesystem = s3_filesystem() if root.startswith("s3://") else None
    paths = partition_paths(root, competitions, seasons)
    if filesystem is not None and not competitions and not root.endswith(".parquet"):
        root_key = s3_key(root).rstrip("/")
        if seasons:
            parquet_keys = []
            for season in seasons:
                parquet_keys.extend(sorted(filesystem.glob(f"{root_key}/competition=*/season={season}/*.parquet")))
        else:
            parquet_keys = sorted(filesystem.glob(f"{root_key}/**/*.parquet"))
        paths = [f"s3://{path}" for path in parquet_keys]
    LOGGER.info(
        "loading parquet files count=%s input=%s competitions=%s seasons=%s",
        len(paths),
        root,
        competitions or ["all"],
        seasons or ["all"],
    )
    frames: list[pd.DataFrame] = []
    started = time.monotonic()
    for idx, path in enumerate(paths, start=1):
        try:
            LOGGER.info("reading parquet %s/%s %s", idx, len(paths), path)
            frames.append(read_parquet_path(path, filesystem))
        except FileNotFoundError:
            LOGGER.warning("missing partition: %s", path)
    if not frames:
        raise RuntimeError("No input rows loaded.")
    df = pd.concat(frames, ignore_index=True)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df["season"] = df["season"].astype(str)
    df = df.dropna(subset=["kickoff_utc", "result_target"]).copy()
    df["result_target"] = pd.to_numeric(df["result_target"], errors="coerce").astype("Int64")
    df = df[df["result_target"].isin([0, 1, 2])].copy()
    LOGGER.info("loaded rows=%s elapsed_seconds=%.1f", len(df), time.monotonic() - started)
    return df.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is missing. In Colab run:\n"
            "  !pip install tensorflow optuna joblib s3fs pyarrow scikit-learn\n"
        ) from exc
    return tf, Sequential, Input, Dense, Dropout, LeakyReLU, l2, Adam, EarlyStopping


def build_model(tf_pack, input_dim: int, params: dict[str, Any]):
    _, Sequential, Input, Dense, Dropout, LeakyReLU, l2, Adam, _ = tf_pack
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(params["n_units_1"], kernel_regularizer=l2(params["l2_reg"])),
            LeakyReLU(negative_slope=0.01),
            Dropout(params["dropout_1"]),
            Dense(params["n_units_2"], kernel_regularizer=l2(params["l2_reg"])),
            LeakyReLU(negative_slope=0.01),
            Dropout(params["dropout_2"]),
            Dense(params["n_units_3"], kernel_regularizer=l2(params["l2_reg"])),
            LeakyReLU(negative_slope=0.01),
            Dropout(params["dropout_3"]),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="categorical_crossentropy")
    return model


def multiclass_brier(y_true_onehot: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean(np.sum((p_pred - y_true_onehot) ** 2, axis=1)))


def class_brier_scores(y_true_cls: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for cls, label in TARGET_LABELS.items():
        out[f"brier_{label}"] = float(brier_score_loss((y_true_cls == cls).astype(int), p_pred[:, cls]))
    return out


def evaluate_probs(y_true_onehot: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)
    p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)
    y_true_cls = np.argmax(y_true_onehot, axis=1)
    y_pred_cls = np.argmax(p_pred, axis=1)
    metrics = {
        "log_loss": float(log_loss(y_true_cls, p_pred, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "multiclass_brier": multiclass_brier(y_true_onehot, p_pred),
        "mean_prob_home": float(p_pred[:, 0].mean()),
        "mean_prob_draw": float(p_pred[:, 1].mean()),
        "mean_prob_away": float(p_pred[:, 2].mean()),
        "actual_home_rate": float((y_true_cls == 0).mean()),
        "actual_draw_rate": float((y_true_cls == 1).mean()),
        "actual_away_rate": float((y_true_cls == 2).mean()),
    }
    metrics.update(class_brier_scores(y_true_cls, p_pred))
    return metrics


def trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_units_1": trial.suggest_categorical("n_units_1", [64, 128, 256, 384]),
        "n_units_2": trial.suggest_categorical("n_units_2", [32, 64, 128, 192]),
        "n_units_3": trial.suggest_categorical("n_units_3", [16, 32, 64, 96]),
        "dropout_1": trial.suggest_float("dropout_1", 0.05, 0.35),
        "dropout_2": trial.suggest_float("dropout_2", 0.05, 0.35),
        "dropout_3": trial.suggest_float("dropout_3", 0.00, 0.25),
        "l2_reg": trial.suggest_float("l2_reg", 1e-7, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def objective_factory(df: pd.DataFrame, folds, feature_cols: list[str], tf_pack, args: argparse.Namespace):
    tf, *_, EarlyStopping = tf_pack

    def objective(trial: optuna.Trial) -> float:
        LOGGER.info("trial %s started", trial.number)
        tf.keras.backend.clear_session()
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        params = trial_params(trial)
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            fold = prepare_fold(train_df, val_df, feature_cols)
            model = build_model(tf_pack, fold.x_train.shape[1], params)
            early = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
            LOGGER.info(
                "trial %s fold %s/%s train_rows=%s val_rows=%s batch_size=%s",
                trial.number,
                fold_idx + 1,
                len(folds),
                len(train_df),
                len(val_df),
                params["batch_size"],
            )
            model.fit(
                fold.x_train,
                fold.y_train,
                validation_data=(fold.x_val, fold.y_val),
                epochs=args.epochs,
                batch_size=params["batch_size"],
                callbacks=[early],
                verbose=0,
            )
            probs = model.predict(fold.x_val, verbose=0)
            metrics = evaluate_probs(fold.y_val, probs)
            LOGGER.info(
                "trial %s fold %s metrics log_loss=%.5f accuracy=%.4f brier=%.5f",
                trial.number,
                fold_idx + 1,
                metrics["log_loss"],
                metrics["accuracy"],
                metrics["multiclass_brier"],
            )
            trial.set_user_attr(f"fold_{fold_idx}", metrics)
            scores.append(-metrics["log_loss"])
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def train_final_model(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], params, tf_pack, args):
    _, *_, EarlyStopping = tf_pack
    LOGGER.info("final training started train_rows=%s test_rows=%s features=%s", len(train_df), len(test_df), len(feature_cols))
    fold = prepare_fold(train_df, test_df, feature_cols)
    model = build_model(tf_pack, fold.x_train.shape[1], params)
    early = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    model.fit(
        fold.x_train,
        fold.y_train,
        validation_data=(fold.x_val, fold.y_val),
        epochs=args.epochs,
        batch_size=params["batch_size"],
        callbacks=[early],
        verbose=1,
    )
    probs = model.predict(fold.x_val, verbose=0)
    metrics = evaluate_probs(fold.y_val, probs)
    LOGGER.info(
        "final test metrics log_loss=%.5f accuracy=%.4f brier=%.5f",
        metrics["log_loss"],
        metrics["accuracy"],
        metrics["multiclass_brier"],
    )
    prediction_frame = test_df[
        ["match_id", "competition_id", "season", "kickoff_utc", "home_team_id", "away_team_id", "result_target"]
    ].copy()
    prediction_frame["prob_home"] = probs[:, 0]
    prediction_frame["prob_draw"] = probs[:, 1]
    prediction_frame["prob_away"] = probs[:, 2]
    prediction_frame["predicted_target"] = np.argmax(probs, axis=1)
    prediction_frame["predicted_label"] = prediction_frame["predicted_target"].map(TARGET_LABELS)
    return model, fold, metrics, prediction_frame


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.smoke:
        args.n_trials = min(args.n_trials, 3)
        args.epochs = min(args.epochs, 10)
        args.max_features = min(args.max_features, 100)
        args.n_folds = min(args.n_folds, 2)
        args.val_size = min(args.val_size, 80)
        args.min_train_size = min(args.min_train_size, 160)
        if args.league == "all" and not args.competitions and not args.seasons:
            args.league = "eng1"
            args.seasons = "2021,2022,2023,2024,2025"

    competitions = resolve_competitions(args)
    seasons = csv_list(args.seasons)
    if competitions and not seasons:
        args.seasons = default_partition_seasons(args.train_through_season, args.test_season)
        seasons = csv_list(args.seasons)
        LOGGER.info("defaulting seasons for league run to %s", seasons)
    if competitions and not seasons:
        raise SystemExit("--competitions requires --seasons so the script can target partition files.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "run config league=%s competitions=%s seasons=%s train_through=%s test=%s trials=%s epochs=%s smoke=%s",
        args.league,
        competitions or ["all"],
        seasons or ["all"],
        args.train_through_season,
        args.test_season,
        args.n_trials,
        args.epochs,
        args.smoke,
    )
    tf_pack = require_tensorflow()
    tf_pack[0].random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = load_gold_dataset(args.input, competitions, seasons)
    train_all = df[df["season"].astype(str) <= str(args.train_through_season)].copy()
    test = df[df["season"].astype(str) == str(args.test_season)].copy()
    if train_all.empty or test.empty:
        raise RuntimeError(f"Bad train/test split: train_rows={len(train_all)} test_rows={len(test)}")

    feature_cols = choose_numeric_feature_columns(train_all, args.max_features)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns selected.")
    folds = make_walkforward_folds(train_all, args.n_folds, args.val_size, args.min_train_size)
    LOGGER.info("selected features=%s folds=%s", len(feature_cols), len(folds))
    print(json.dumps({"rows": len(df), "train_rows": len(train_all), "test_rows": len(test), "features": len(feature_cols)}, indent=2))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(2, min(5, args.n_trials // 4)), n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    LOGGER.info("optuna optimization started trials=%s folds=%s", args.n_trials, len(folds))
    study.optimize(objective_factory(train_all, folds, feature_cols, tf_pack, args), n_trials=args.n_trials)
    LOGGER.info("optuna optimization finished best_value=%.5f best_params=%s", study.best_value, study.best_params)

    model, fold, test_metrics, predictions = train_final_model(
        df=df,
        train_df=train_all,
        test_df=test,
        feature_cols=feature_cols,
        params=study.best_params,
        tf_pack=tf_pack,
        args=args,
    )

    model_path = output_dir / "soccer_three_way_nn.keras"
    preprocessing_path = output_dir / "preprocessing.joblib"
    predictions_path = output_dir / "test_predictions.parquet"
    trials_path = output_dir / "optuna_trials.csv"
    summary_path = output_dir / "summary.json"
    features_path = output_dir / "feature_columns.json"

    model.save(model_path)
    joblib.dump({"imputer": fold.imputer, "scaler": fold.scaler, "feature_names": feature_cols}, preprocessing_path)
    predictions.to_parquet(predictions_path, index=False)
    study.trials_dataframe().to_csv(trials_path, index=False)
    features_path.write_text(json.dumps(feature_cols, indent=2) + "\n")
    config = TrainingConfig(
        input=args.input,
        output_dir=str(output_dir),
        competitions=competitions,
        seasons=seasons,
        train_through_season=str(args.train_through_season),
        test_season=str(args.test_season),
        max_features=args.max_features,
        n_folds=args.n_folds,
        val_size=args.val_size,
        min_train_size=args.min_train_size,
        epochs=args.epochs,
        n_trials=args.n_trials,
        batch_size_default=0,
        seed=args.seed,
    )
    summary = {
        "config": asdict(config),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_attrs": study.best_trial.user_attrs,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "preprocessing": str(preprocessing_path),
            "predictions": str(predictions_path),
            "trials": str(trials_path),
            "features": str(features_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
