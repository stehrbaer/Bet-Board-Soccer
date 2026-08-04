#!/usr/bin/env python3
"""Export the first N calendar weeks from a prediction parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export first-week soccer model predictions.")
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and JSON outputs.")
    parser.add_argument("--weeks", type=int, default=5, help="Number of calendar weeks to export.")
    return parser.parse_args()


def ranked_label(row: pd.Series) -> str:
    scores = {
        "home": row["prob_home"],
        "draw": row["prob_draw"],
        "away": row["prob_away"],
    }
    return max(scores, key=scores.get)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.predictions)
    if df.empty:
        raise RuntimeError("Prediction file has no rows.")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["kickoff_utc"]).sort_values(["kickoff_utc", "match_id"]).copy()
    df["prediction"] = df.apply(ranked_label, axis=1)
    df["week_start"] = df["kickoff_utc"].dt.to_period("W-MON").dt.start_time.astype(str)

    first_weeks = df["week_start"].drop_duplicates().head(args.weeks).tolist()
    out = df[df["week_start"].isin(first_weeks)].copy()
    display_cols = [
        "week_start",
        "kickoff_utc",
        "competition_id",
        "season",
        "match_id",
        "home_team_id",
        "away_team_id",
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
        "weeks": first_weeks,
        "rows": len(out),
        "output_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
