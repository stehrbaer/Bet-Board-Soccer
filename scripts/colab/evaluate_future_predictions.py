#!/usr/bin/env python3
"""Evaluate already-played future soccer predictions against ESPN final scores."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import requests


ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_COMPETITION_ALIASES = {
    "uefa.champions.qual": "uefa.champions_qual",
    "uefa.europa.qual": "uefa.europa_qual",
    "uefa.europa.conf.qual": "uefa.europa_conf_qual",
}
LABELS = ["home", "draw", "away"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade completed future fixture predictions.")
    parser.add_argument(
        "--predictions",
        default="",
        help="Prediction CSV to grade. Can be local or s3://. Use instead of --batch-summary.",
    )
    parser.add_argument(
        "--batch-summary",
        default="",
        help="future_prediction_batch_summary.json from predict_future_fixtures.py.",
    )
    parser.add_argument("--repo-root", default=".", help="Base path for relative batch-summary entries.")
    parser.add_argument("--output-dir", default="outputs/soccer_nn_by_league/future_evaluation")
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


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


def read_csv(path: str | Path) -> pd.DataFrame:
    path_str = str(path)
    if path_str.startswith("s3://"):
        storage_options = s3_storage_options()
        if storage_options is None:
            raise SystemExit(
                "DigitalOcean Spaces credentials are missing. Set DO_SPACES_KEY and DO_SPACES_SECRET first."
            )
        return pd.read_csv(path_str, storage_options=storage_options)
    return pd.read_csv(path_str)


def resolve_path(path: str, repo_root: Path) -> str:
    if path.startswith("s3://"):
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(repo_root / candidate)


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.predictions) == bool(args.batch_summary):
        raise SystemExit("Provide exactly one of --predictions or --batch-summary.")

    repo_root = Path(args.repo_root)
    if args.predictions:
        df = read_csv(resolve_path(args.predictions, repo_root))
        if "league" not in df.columns and "competition_id" in df.columns:
            df["league"] = df["competition_id"]
        return df

    summary_path = Path(resolve_path(args.batch_summary, repo_root))
    summary = json.loads(summary_path.read_text())
    frames = []
    for item in summary.get("completed", []):
        prediction_path = item.get("future_predictions")
        if not prediction_path:
            continue
        frame = read_csv(resolve_path(prediction_path, repo_root))
        frame["source_prediction_file"] = prediction_path
        if "league" not in frame.columns:
            frame["league"] = item.get("competition_id") or frame.get("competition_id")
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No prediction files found in batch summary: {summary_path}")
    return pd.concat(frames, ignore_index=True)


def espn_date(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def espn_competition_id(competition_id: str) -> str:
    return ESPN_COMPETITION_ALIASES.get(competition_id, competition_id)


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def fetch_scoreboard(competition_id: str, start: pd.Timestamp, end: pd.Timestamp, limit: int) -> dict[str, Any]:
    league = espn_competition_id(competition_id)
    url = f"{ESPN_BASE_URL}/{league}/scoreboard"
    response = requests.get(
        url,
        params={
            "dates": f"{espn_date(start)}-{espn_date(end)}",
            "limit": limit,
            "region": "us",
            "lang": "en",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def parse_score(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_scoreboard_results(payload: dict[str, Any], competition_id: str) -> pd.DataFrame:
    rows = []
    for event in payload.get("events", []):
        competitions = event.get("competitions", [])
        competition = competitions[0] if competitions else {}
        status_type = competition.get("status", {}).get("type", {})
        home_score = None
        away_score = None
        home_team_name = None
        away_team_name = None
        for competitor in competition.get("competitors", []):
            team = competitor.get("team", {})
            if competitor.get("homeAway") == "home":
                home_score = parse_score(competitor.get("score"))
                home_team_name = team.get("displayName") or team.get("name")
            elif competitor.get("homeAway") == "away":
                away_score = parse_score(competitor.get("score"))
                away_team_name = team.get("displayName") or team.get("name")

        actual_label = None
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                actual_label = "home"
            elif away_score > home_score:
                actual_label = "away"
            else:
                actual_label = "draw"

        rows.append(
            {
                "match_id": str(event.get("id")),
                "competition_id": competition_id,
                "actual_kickoff_utc": event.get("date"),
                "actual_completed": bool(status_type.get("completed")),
                "actual_status": status_type.get("description"),
                "actual_status_state": status_type.get("state"),
                "actual_home_score": home_score,
                "actual_away_score": away_score,
                "actual_label": actual_label,
                "actual_home_team_name": home_team_name,
                "actual_away_team_name": away_team_name,
            }
        )
    actuals = pd.DataFrame(rows)
    if not actuals.empty:
        actuals["actual_kickoff_utc"] = pd.to_datetime(actuals["actual_kickoff_utc"], errors="coerce", utc=True)
        actuals["home_team_match_key"] = actuals["actual_home_team_name"].map(slug)
        actuals["away_team_match_key"] = actuals["actual_away_team_name"].map(slug)
    return actuals


def fetch_actuals(predictions: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = predictions.copy()
    if "competition_id" not in work.columns:
        if "league" not in work.columns:
            raise ValueError("Predictions need either competition_id or league column.")
        work["competition_id"] = work["league"]
    work["competition_id"] = work["competition_id"].astype(str)
    work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], errors="coerce", utc=True)
    work = work.dropna(subset=["competition_id", "kickoff_utc"])

    frames = []
    failures = []
    for competition_id, group in work.groupby("competition_id"):
        start = group["kickoff_utc"].min().floor("D") - timedelta(days=1)
        end = group["kickoff_utc"].max().ceil("D") + timedelta(days=1)
        try:
            payload = fetch_scoreboard(competition_id, start, end, limit)
            frame = parse_scoreboard_results(payload, competition_id)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - preserve partial grading across leagues.
            failures.append({"competition_id": competition_id, "error": str(exc)})

    actuals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not actuals.empty:
        actuals = actuals.drop_duplicates(subset=["competition_id", "match_id"], keep="last")
    return actuals, failures


def multiclass_log_loss(scored: pd.DataFrame) -> float | None:
    if scored.empty:
        return None
    probs = scored[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    actual = scored["actual_label"].map(LABEL_TO_INDEX).to_numpy(dtype=int)
    return float(-np.log(probs[np.arange(len(actual)), actual]).mean())


def multiclass_brier(scored: pd.DataFrame) -> float | None:
    if scored.empty:
        return None
    probs = scored[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    actual = scored["actual_label"].map(LABEL_TO_INDEX).to_numpy(dtype=int)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actual)), actual] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def metric_summary(scored: pd.DataFrame, total_rows: int, pending_rows: int) -> dict[str, Any]:
    completed = scored[scored["actual_completed"] & scored["actual_label"].notna()].copy()
    raw_pick = completed["prediction"] if "prediction" in completed.columns else pd.Series(dtype=object)
    recommended_pick = (
        completed["recommended_pick"]
        if "recommended_pick" in completed.columns
        else raw_pick
    )
    raw_correct = raw_pick.eq(completed["actual_label"]) if len(completed) else pd.Series(dtype=bool)
    recommended_correct = recommended_pick.eq(completed["actual_label"]) if len(completed) else pd.Series(dtype=bool)
    draw_picks = recommended_pick.eq("draw") if len(completed) else pd.Series(dtype=bool)
    actual_draws = completed["actual_label"].eq("draw") if len(completed) else pd.Series(dtype=bool)
    draw_hits = draw_picks & actual_draws if len(completed) else pd.Series(dtype=bool)
    actual_home_wins = completed["actual_label"].eq("home") if len(completed) else pd.Series(dtype=bool)
    actual_away_wins = completed["actual_label"].eq("away") if len(completed) else pd.Series(dtype=bool)

    return {
        "rows": int(total_rows),
        "completed_rows": int(len(completed)),
        "pending_or_unmatched_rows": int(pending_rows),
        "actual_home_win_count": int(actual_home_wins.sum()),
        "actual_draw_count": int(actual_draws.sum()),
        "actual_away_win_home_loss_count": int(actual_away_wins.sum()),
        "actual_draw_rate": safe_rate(int(actual_draws.sum()), len(completed)),
        "raw_accuracy": safe_rate(int(raw_correct.sum()), len(completed)),
        "recommended_accuracy": safe_rate(int(recommended_correct.sum()), len(completed)),
        "raw_home_win_accuracy": safe_rate(int((raw_correct & actual_home_wins).sum()), int(actual_home_wins.sum())),
        "raw_draw_accuracy": safe_rate(int((raw_correct & actual_draws).sum()), int(actual_draws.sum())),
        "raw_away_win_home_loss_accuracy": safe_rate(int((raw_correct & actual_away_wins).sum()), int(actual_away_wins.sum())),
        "recommended_home_win_accuracy": safe_rate(
            int((recommended_correct & actual_home_wins).sum()), int(actual_home_wins.sum())
        ),
        "recommended_draw_accuracy": safe_rate(
            int((recommended_correct & actual_draws).sum()), int(actual_draws.sum())
        ),
        "recommended_away_win_home_loss_accuracy": safe_rate(
            int((recommended_correct & actual_away_wins).sum()), int(actual_away_wins.sum())
        ),
        "recommended_accuracy_delta": (
            None
            if len(completed) == 0
            else float(safe_rate(int(recommended_correct.sum()), len(completed)) - safe_rate(int(raw_correct.sum()), len(completed)))
        ),
        "log_loss": multiclass_log_loss(completed),
        "multiclass_brier": multiclass_brier(completed),
        "raw_home_pick_rate": safe_rate(int(raw_pick.eq("home").sum()), len(completed)),
        "raw_draw_pick_rate": safe_rate(int(raw_pick.eq("draw").sum()), len(completed)),
        "raw_away_pick_rate": safe_rate(int(raw_pick.eq("away").sum()), len(completed)),
        "recommended_home_pick_rate": safe_rate(int(recommended_pick.eq("home").sum()), len(completed)),
        "recommended_draw_pick_rate": safe_rate(int(draw_picks.sum()), len(completed)),
        "recommended_away_pick_rate": safe_rate(int(recommended_pick.eq("away").sum()), len(completed)),
        "draw_pick_precision": safe_rate(int(draw_hits.sum()), int(draw_picks.sum())),
        "draw_pick_recall": safe_rate(int(draw_hits.sum()), int(actual_draws.sum())),
        "draw_pick_count": int(draw_picks.sum()),
    }


def league_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for competition_id, group in scored.groupby("competition_id"):
        pending = int((~group["actual_completed"] | group["actual_label"].isna()).sum())
        summary = metric_summary(group, total_rows=len(group), pending_rows=pending)
        summary["competition_id"] = competition_id
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    columns = ["competition_id", *[column for column in rows[0] if column != "competition_id"]]
    return pd.DataFrame(rows)[columns].sort_values(["completed_rows", "competition_id"], ascending=[False, True])


def result_type_breakdown(scored: pd.DataFrame) -> pd.DataFrame:
    completed = scored[scored["actual_completed"] & scored["actual_label"].notna()].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "competition_id",
                "actual_result_type",
                "completed_rows",
                "raw_correct",
                "raw_accuracy",
                "recommended_correct",
                "recommended_accuracy",
            ]
        )

    rows = []
    result_names = {
        "home": "home_win",
        "draw": "draw",
        "away": "away_win_home_loss",
    }
    for competition_id, league_group in completed.groupby("competition_id"):
        for label in LABELS:
            group = league_group[league_group["actual_label"].eq(label)]
            raw_correct = int(group["raw_correct"].fillna(False).sum())
            recommended_correct = int(group["recommended_correct"].fillna(False).sum())
            rows.append(
                {
                    "competition_id": competition_id,
                    "actual_result_type": result_names[label],
                    "completed_rows": int(len(group)),
                    "raw_correct": raw_correct,
                    "raw_accuracy": safe_rate(raw_correct, len(group)),
                    "recommended_correct": recommended_correct,
                    "recommended_accuracy": safe_rate(recommended_correct, len(group)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scored = predictions.copy()
    if "competition_id" not in scored.columns and "league" in scored.columns:
        scored["competition_id"] = scored["league"]
    scored["competition_id"] = scored["competition_id"].astype(str)
    has_match_id = "match_id" in scored.columns
    if has_match_id:
        scored["match_id"] = scored["match_id"].astype(str)
    if actuals.empty:
        for column in [
            "actual_completed",
            "actual_kickoff_utc",
            "actual_status",
            "actual_status_state",
            "actual_home_score",
            "actual_away_score",
            "actual_label",
        ]:
            scored[column] = None
    else:
        actuals = actuals.copy()
        actuals["competition_id"] = actuals["competition_id"].astype(str)
        actuals["match_id"] = actuals["match_id"].astype(str)
        if has_match_id:
            scored = scored.merge(actuals, on=["competition_id", "match_id"], how="left")
        else:
            required = {"kickoff_utc", "home_team_name", "away_team_name"}
            missing = required - set(scored.columns)
            if missing:
                raise ValueError(f"Predictions without match_id need columns: {sorted(missing)}")
            scored["kickoff_utc"] = pd.to_datetime(scored["kickoff_utc"], errors="coerce", utc=True)
            scored["home_team_match_key"] = scored["home_team_name"].map(slug)
            scored["away_team_match_key"] = scored["away_team_name"].map(slug)
            scored = scored.merge(
                actuals,
                left_on=["competition_id", "kickoff_utc", "home_team_match_key", "away_team_match_key"],
                right_on=["competition_id", "actual_kickoff_utc", "home_team_match_key", "away_team_match_key"],
                how="left",
                suffixes=("", "_actual"),
            )

    scored["actual_completed"] = scored["actual_completed"].fillna(False).astype(bool)
    completed_mask = scored["actual_completed"] & scored["actual_label"].notna()
    scored["raw_correct"] = np.where(completed_mask, scored["prediction"].eq(scored["actual_label"]), np.nan)
    if "recommended_pick" in scored.columns:
        scored["recommended_correct"] = np.where(completed_mask, scored["recommended_pick"].eq(scored["actual_label"]), np.nan)
    else:
        scored["recommended_correct"] = scored["raw_correct"]

    pending = int((~completed_mask).sum())
    summary = metric_summary(scored, total_rows=len(scored), pending_rows=pending)
    by_league = league_summary(scored)
    by_result_type = result_type_breakdown(scored)
    return scored, summary, by_league, by_result_type


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args)
    actuals, fetch_failures = fetch_actuals(predictions, args.limit)
    scored, summary, by_league, by_result_type = evaluate_predictions(predictions, actuals)
    summary["fetch_failures"] = fetch_failures
    summary["outputs"] = {
        "graded_predictions": str(output_dir / "graded_predictions.csv"),
        "league_summary": str(output_dir / "league_summary.csv"),
        "result_type_breakdown": str(output_dir / "result_type_breakdown.csv"),
        "summary": str(output_dir / "summary.json"),
    }

    scored.to_csv(output_dir / "graded_predictions.csv", index=False)
    by_league.to_csv(output_dir / "league_summary.csv", index=False)
    by_result_type.to_csv(output_dir / "result_type_breakdown.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
