#!/usr/bin/env python3
"""Score future fixtures with a trained soccer 1X2 neural model.

This is a first-pass future inference bridge. It builds prematch rows from each
team's latest historical snapshot, then lets the saved imputer fill features
that are unavailable for future fixtures, such as bookmaker odds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


try:
    from betboard_soccer_extension.modeling.draw_policy import apply_draw_policy, load_draw_policy
except ModuleNotFoundError:
    if "__file__" in globals():
        candidate = Path(__file__).resolve().parents[2] / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    from betboard_soccer_extension.modeling.draw_policy import apply_draw_policy, load_draw_policy  # type: ignore[no-redef]


DEFAULT_GOLD_ROOT = "s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input"
DEFAULT_HISTORY_PARTITIONS = "eng.1:2025,eng.2:2025,eng.3:2025"
COUNTRY_HISTORY_COMPETITIONS = {
    "eng": ["eng.1", "eng.2", "eng.3"],
    "esp": ["esp.1", "esp.2"],
    "fra": ["fra.1", "fra.2"],
    "ger": ["ger.1", "ger.2"],
    "ita": ["ita.1", "ita.2"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future soccer fixtures from a trained neural model.")
    parser.add_argument("--fixtures", default="", help="Normalized fixture CSV from fetch_espn_fixtures.py.")
    parser.add_argument("--model", default="outputs/eng1_soccer_nn/soccer_three_way_nn.keras")
    parser.add_argument("--preprocessing", default="outputs/eng1_soccer_nn/preprocessing.joblib")
    parser.add_argument("--gold-root", default=DEFAULT_GOLD_ROOT)
    parser.add_argument(
        "--history-partitions",
        default=DEFAULT_HISTORY_PARTITIONS,
        help="Comma-separated competition:season partitions used for latest team snapshots.",
    )
    parser.add_argument("--output-dir", default="outputs/eng1_soccer_nn/future_2026")
    parser.add_argument("--draw-policy", default="configs/draw_policy_eng1.json", help="Optional draw policy JSON.")
    parser.add_argument(
        "--predict-each-league",
        action="store_true",
        help="Score every first-weeks fixture file under --fixtures-root.",
    )
    parser.add_argument("--fixtures-root", default="outputs/fixtures/espn_by_league_2026")
    parser.add_argument("--models-root", default="outputs/soccer_nn_by_league")
    parser.add_argument(
        "--global-model-dir",
        default="",
        help="Optional global model dir. If set, batch mode uses this model for every league.",
    )
    parser.add_argument("--history-season", default="2025")
    parser.add_argument(
        "--use-league-draw-policy",
        action="store_true",
        help="In batch mode, prefer each model folder's draw_threshold_tuning/active_draw_policy.json.",
    )
    return parser.parse_args()


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def league_slug(competition_id: str) -> str:
    return competition_id.lower().replace(".", "").replace("_", "").replace("-", "")


def competition_from_fixture_path(path: Path) -> str:
    name = path.name
    if "_first_" in name:
        return name.split("_first_", maxsplit=1)[0]
    if "_fixtures" in name:
        return name.split("_fixtures", maxsplit=1)[0]
    raise ValueError(f"Cannot infer competition id from fixture filename: {path}")


def default_history_partitions_for_competition(competition_id: str, season: str) -> str:
    country = competition_id.split(".", maxsplit=1)[0]
    competitions = COUNTRY_HISTORY_COMPETITIONS.get(country, [competition_id])
    if competition_id not in competitions:
        competitions = [competition_id, *competitions]
    return ",".join(f"{competition}:{season}" for competition in competitions)


def batch_fixture_paths(fixtures_root: Path) -> list[Path]:
    return sorted(fixtures_root.glob("*_fixtures/*_first_*_weeks_fixtures.csv"))


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


def read_parquet(path: str) -> pd.DataFrame:
    if not path.startswith("s3://"):
        return pd.read_parquet(path)
    filesystem = s3_filesystem()
    with filesystem.open(s3_key(path), "rb") as handle:
        return pd.read_parquet(handle)


def partition_path(root: str, partition: str) -> str:
    competition, season = partition.split(":", maxsplit=1)
    return f"{root.rstrip('/')}/competition={competition}/season={season}/part-000.parquet"


def load_history(root: str, partitions: str) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    loaded = []
    for raw in [part.strip() for part in partitions.split(",") if part.strip()]:
        path = partition_path(root, raw)
        try:
            frame = read_parquet(path)
        except FileNotFoundError:
            continue
        frames.append(frame)
        loaded.append(path)
    if not frames:
        raise RuntimeError("No historical partitions loaded for future feature snapshots.")
    history = pd.concat(frames, ignore_index=True)
    history["kickoff_utc"] = pd.to_datetime(history["kickoff_utc"], errors="coerce", utc=True)
    return history.dropna(subset=["kickoff_utc"]).sort_values("kickoff_utc").reset_index(drop=True), loaded


def make_team_snapshots(history: pd.DataFrame) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for side in ["home", "away"]:
        team_id_col = f"{side}_team_id"
        team_name_col = f"{side}_team_name"
        if team_id_col not in history.columns:
            continue
        side_cols = [col for col in history.columns if col.startswith(f"{side}_")]
        for row in history.itertuples(index=False):
            row_dict = row._asdict()
            team_id = row_dict.get(team_id_col)
            if pd.isna(team_id):
                continue
            team_key = str(team_id)
            current = snapshots.get(team_key)
            kickoff = row_dict.get("kickoff_utc")
            if current and current["source_kickoff_utc"] >= kickoff:
                continue
            neutral = {
                col.removeprefix(f"{side}_"): row_dict.get(col)
                for col in side_cols
                if col not in {team_id_col, team_name_col}
            }
            snapshots[team_key] = {
                "team_id": team_key,
                "team_name": row_dict.get(team_name_col),
                "source_kickoff_utc": kickoff,
                "features": neutral,
            }
    return snapshots


def build_future_feature_frame(fixtures: pd.DataFrame, feature_names: list[str], snapshots: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    fixtures = fixtures.copy()
    fixtures["kickoff_utc"] = pd.to_datetime(fixtures["kickoff_utc"], errors="coerce", utc=True)
    fixtures = fixtures.dropna(subset=["kickoff_utc"]).sort_values("kickoff_utc").reset_index(drop=True)
    if "week_start" not in fixtures.columns:
        fixtures["week_start"] = (fixtures["kickoff_utc"] - pd.to_timedelta(fixtures["kickoff_utc"].dt.weekday, unit="D")).dt.date.astype(str)
    if "matchweek" not in fixtures.columns:
        week_numbers = {week: idx for idx, week in enumerate(fixtures["week_start"].drop_duplicates(), start=1)}
        fixtures["matchweek"] = fixtures["week_start"].map(week_numbers)
    if "matchup_number" not in fixtures.columns:
        fixtures["matchup_number"] = fixtures.groupby("week_start").cumcount() + 1

    for idx, fixture in fixtures.iterrows():
        row: dict[str, Any] = {feature: np.nan for feature in feature_names}
        home_id = str(fixture.get("home_team_id"))
        away_id = str(fixture.get("away_team_id"))
        home_snapshot = snapshots.get(home_id)
        away_snapshot = snapshots.get(away_id)

        for feature in feature_names:
            if feature.startswith("home_") and home_snapshot:
                row[feature] = home_snapshot["features"].get(feature.removeprefix("home_"), np.nan)
            elif feature.startswith("away_") and away_snapshot:
                row[feature] = away_snapshot["features"].get(feature.removeprefix("away_"), np.nan)

        row.update(
            {
                "match_id": fixture.get("espn_event_id") or fixture.get("match_id"),
                "week_start": fixture.get("week_start"),
                "matchweek": fixture.get("matchweek"),
                "matchup_number": fixture.get("matchup_number"),
                "competition_id": fixture.get("competition_id"),
                "season": fixture.get("season"),
                "kickoff_utc": fixture.get("kickoff_utc"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_name": fixture.get("home_team_name"),
                "away_team_name": fixture.get("away_team_name"),
                "matchup_key": f"{fixture.get('matchweek')}_{slug(fixture.get('home_team_name'))}_{slug(fixture.get('away_team_name'))}",
            }
        )
        rows.append(row)
        available = pd.Series({feature: row.get(feature) for feature in feature_names}).notna().sum()
        diagnostics.append(
            {
                "matchup_key": row["matchup_key"],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_snapshot_found": home_snapshot is not None,
                "away_snapshot_found": away_snapshot is not None,
                "home_snapshot_date": None if home_snapshot is None else home_snapshot["source_kickoff_utc"],
                "away_snapshot_date": None if away_snapshot is None else away_snapshot["source_kickoff_utc"],
                "available_model_features": int(available),
                "missing_model_features": int(len(feature_names) - available),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def score_future_rows(model_path: str, preprocessing_path: str, future_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is missing. In Colab run: !pip install -r requirements-colab.txt") from exc

    preprocessing = joblib.load(preprocessing_path)
    feature_names = list(preprocessing["feature_names"])
    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]
    model = tf.keras.models.load_model(model_path)

    x_raw = future_df[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = scaler.transform(imputer.transform(x_raw))
    probs = model.predict(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), verbose=0)
    out = future_df[
        [
            "matchup_key",
            "week_start",
            "matchweek",
            "matchup_number",
            "kickoff_utc",
            "competition_id",
            "season",
            "match_id",
            "home_team_id",
            "home_team_name",
            "away_team_id",
            "away_team_name",
        ]
    ].copy()
    out["prob_home"] = probs[:, 0]
    out["prob_draw"] = probs[:, 1]
    out["prob_away"] = probs[:, 2]
    out["prediction"] = np.array(["home", "draw", "away"])[np.argmax(probs, axis=1)]
    return out, feature_names


def predict_fixture_file(
    fixtures_path: Path,
    model_path: Path,
    preprocessing_path: Path,
    output_dir: Path,
    gold_root: str,
    history_partitions: str,
    draw_policy_path: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(joblib.load(preprocessing_path)["feature_names"])
    fixtures = pd.read_csv(fixtures_path)
    history, loaded_history = load_history(gold_root, history_partitions)
    snapshots = make_team_snapshots(history)
    future_df, diagnostics = build_future_feature_frame(fixtures, feature_names, snapshots)
    predictions, trained_features = score_future_rows(str(model_path), str(preprocessing_path), future_df)
    draw_policy = load_draw_policy(draw_policy_path) if draw_policy_path else None
    if draw_policy is not None:
        predictions = apply_draw_policy(predictions, draw_policy)

    feature_path = output_dir / "future_model_input.parquet"
    predictions_path = output_dir / "future_predictions.csv"
    diagnostics_path = output_dir / "future_feature_diagnostics.csv"
    summary_path = output_dir / "summary.json"
    future_df.to_parquet(feature_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    summary = {
        "fixtures": str(fixtures_path),
        "model": str(model_path),
        "preprocessing": str(preprocessing_path),
        "history_partitions": history_partitions,
        "loaded_history": loaded_history,
        "fixtures_scored": len(predictions),
        "trained_features": len(trained_features),
        "draw_policy": None if draw_policy is None else draw_policy.to_dict(),
        "outputs": {
            "future_model_input": str(feature_path),
            "future_predictions": str(predictions_path),
            "feature_diagnostics": str(diagnostics_path),
        },
        "warning": "Future rows use latest historical team snapshots; unavailable future-only features are imputed.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def model_dir_for_competition(args: argparse.Namespace, competition_id: str) -> Path:
    if args.global_model_dir:
        return Path(args.global_model_dir)
    return Path(args.models_root) / f"{league_slug(competition_id)}_soccer_nn"


def draw_policy_for_model_dir(model_dir: Path, fallback: str, use_league_draw_policy: bool) -> str:
    if use_league_draw_policy:
        candidate = model_dir / "draw_threshold_tuning" / "active_draw_policy.json"
        if candidate.exists():
            return str(candidate)
    return fallback


def predict_each_league(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_root = Path(args.fixtures_root)
    fixtures_paths = batch_fixture_paths(fixtures_root)
    if not fixtures_paths:
        raise RuntimeError(f"No first-weeks fixture CSV files found under {fixtures_root}.")

    completed = []
    failures = []
    for fixtures_path in fixtures_paths:
        try:
            competition_id = competition_from_fixture_path(fixtures_path)
            model_dir = model_dir_for_competition(args, competition_id)
            model_path = model_dir / "soccer_three_way_nn.keras"
            preprocessing_path = model_dir / "preprocessing.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model: {model_path}")
            if not preprocessing_path.exists():
                raise FileNotFoundError(f"Missing preprocessing: {preprocessing_path}")
            output_dir = model_dir / "future_2026" / league_slug(competition_id) if args.global_model_dir else model_dir / "future_2026"
            history_partitions = default_history_partitions_for_competition(competition_id, args.history_season)
            draw_policy_path = draw_policy_for_model_dir(model_dir, args.draw_policy, args.use_league_draw_policy)
            summary = predict_fixture_file(
                fixtures_path=fixtures_path,
                model_path=model_path,
                preprocessing_path=preprocessing_path,
                output_dir=output_dir,
                gold_root=args.gold_root,
                history_partitions=history_partitions,
                draw_policy_path=draw_policy_path,
            )
            completed.append(
                {
                    "competition_id": competition_id,
                    "fixtures": str(fixtures_path),
                    "model_dir": str(model_dir),
                    "output_dir": str(output_dir),
                    "fixtures_scored": summary["fixtures_scored"],
                    "future_predictions": summary["outputs"]["future_predictions"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch prediction going across missing leagues.
            failures.append({"fixtures": str(fixtures_path), "error": str(exc)})

    batch_summary = {
        "fixtures_root": str(fixtures_root),
        "models_root": args.models_root,
        "global_model_dir": args.global_model_dir,
        "history_season": args.history_season,
        "completed": completed,
        "failures": failures,
    }
    summary_root = Path(args.global_model_dir) if args.global_model_dir else Path(args.models_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "future_prediction_batch_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2, default=str) + "\n")
    print(json.dumps(batch_summary, indent=2, default=str))
    return batch_summary


def main() -> int:
    args = parse_args()
    if args.predict_each_league:
        predict_each_league(args)
        return 0
    if not args.fixtures:
        raise SystemExit("--fixtures is required unless --predict-each-league is set.")
    summary = predict_fixture_file(
        fixtures_path=Path(args.fixtures),
        model_path=Path(args.model),
        preprocessing_path=Path(args.preprocessing),
        output_dir=Path(args.output_dir),
        gold_root=args.gold_root,
        history_partitions=args.history_partitions,
        draw_policy_path=args.draw_policy,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
