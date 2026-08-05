#!/usr/bin/env python3
"""Export the first N calendar weeks from a prediction parquet file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export first-week soccer model predictions.")
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and JSON outputs.")
    parser.add_argument("--weeks", type=int, default=5, help="Number of calendar weeks to export.")
    parser.add_argument("--gold-input", default="", help="Optional gold parquet file used to add team names by match_id.")
    parser.add_argument(
        "--auto-gold-root",
        default="s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input",
        help="Gold dataset root used to auto-add team names when --gold-input is omitted.",
    )
    parser.add_argument("--allow-id-matchup-key", action="store_true", help="Allow matchup_key to fall back to team IDs.")
    return parser.parse_args()


def ranked_label(row: pd.Series) -> str:
    scores = {
        "home": row["prob_home"],
        "draw": row["prob_draw"],
        "away": row["prob_away"],
    }
    return max(scores, key=scores.get)


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


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


def read_parquet(path: str) -> pd.DataFrame:
    storage_options = s3_storage_options() if path.startswith("s3://") else None
    return pd.read_parquet(path, storage_options=storage_options)


def enrich_team_names(df: pd.DataFrame, gold_input: str) -> pd.DataFrame:
    if {"home_team_name", "away_team_name"}.issubset(df.columns) or not gold_input:
        return df
    gold = read_parquet(gold_input)
    name_cols = ["match_id", "home_team_name", "away_team_name"]
    missing = [col for col in name_cols if col not in gold.columns]
    if missing:
        raise RuntimeError(f"Gold input is missing team-name columns: {missing}")
    names = gold[name_cols].drop_duplicates("match_id")
    return df.merge(names, on="match_id", how="left")


def gold_partition_path(root: str, competition_id: object, season: object) -> str:
    return f"{root.rstrip('/')}/competition={competition_id}/season={season}/part-000.parquet"


def enrich_team_names_auto(df: pd.DataFrame, gold_root: str) -> tuple[pd.DataFrame, list[str]]:
    if {"home_team_name", "away_team_name"}.issubset(df.columns) or not gold_root:
        return df, []
    required = {"competition_id", "season", "match_id"}
    if not required.issubset(df.columns):
        return df, []

    frames = []
    loaded_paths = []
    partitions = df[["competition_id", "season"]].drop_duplicates()
    for row in partitions.itertuples(index=False):
        path = gold_partition_path(gold_root, row.competition_id, row.season)
        try:
            gold = read_parquet(path)
        except FileNotFoundError:
            continue
        name_cols = ["match_id", "home_team_name", "away_team_name"]
        if set(name_cols).issubset(gold.columns):
            frames.append(gold[name_cols].drop_duplicates("match_id"))
            loaded_paths.append(path)

    if not frames:
        return df, loaded_paths
    names = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    return df.merge(names, on="match_id", how="left"), loaded_paths


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet(args.predictions)
    loaded_gold_paths: list[str] = []
    if args.gold_input:
        df = enrich_team_names(df, args.gold_input)
        loaded_gold_paths = [args.gold_input]
    else:
        df, loaded_gold_paths = enrich_team_names_auto(df, args.auto_gold_root)
    if df.empty:
        raise RuntimeError("Prediction file has no rows.")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["kickoff_utc"]).sort_values(["kickoff_utc", "match_id"]).copy()
    df["prediction"] = df.apply(ranked_label, axis=1)
    df["week_start"] = (df["kickoff_utc"] - pd.to_timedelta(df["kickoff_utc"].dt.weekday, unit="D")).dt.date.astype(str)
    has_team_names = {"home_team_name", "away_team_name"}.issubset(df.columns)
    if not has_team_names and not args.allow_id_matchup_key:
        raise RuntimeError(
            "Team names are missing, so matchup_key would use IDs. "
            "Pass --gold-input for the matching season, keep DO_SPACES env vars set for auto-enrichment, "
            "or pass --allow-id-matchup-key to permit ID fallback."
        )
    home_names = df["home_team_name"] if "home_team_name" in df.columns else df["home_team_id"]
    away_names = df["away_team_name"] if "away_team_name" in df.columns else df["away_team_id"]
    df["matchup_key"] = [
        f"{idx}_{slug(home)}_{slug(away)}"
        for idx, (home, away) in enumerate(zip(home_names, away_names), start=1)
    ]

    first_weeks = df["week_start"].drop_duplicates().head(args.weeks).tolist()
    out = df[df["week_start"].isin(first_weeks)].copy()
    display_cols = [
        "week_start",
        "kickoff_utc",
        "competition_id",
        "season",
        "match_id",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "matchup_key",
        "prob_home",
        "prob_draw",
        "prob_away",
        "prediction",
        "predicted_label",
    ]
    display_cols = [col for col in display_cols if col in out.columns]
    out = out[display_cols]

    csv_path = output_dir / f"first_{args.weeks}_weeks_predictions.csv"
    json_path = output_dir / f"first_{args.weeks}_weeks_summary.json"
    out.to_csv(csv_path, index=False)
    summary = {
        "source_predictions": args.predictions,
        "team_name_sources": loaded_gold_paths,
        "has_team_names": {"home_team_name", "away_team_name"}.issubset(out.columns),
        "weeks": first_weeks,
        "rows": len(out),
        "output_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
