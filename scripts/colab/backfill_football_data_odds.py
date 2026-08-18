#!/usr/bin/env python3
"""Backfill historical soccer results and 1X2 odds from Football-Data.co.uk."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_TO_FOOTBALL_DATA_CODE = {
    "eng.1": "E0",
    "eng.2": "E1",
    "eng.3": "E2",
    "esp.1": "SP1",
    "esp.2": "SP2",
    "fra.1": "F1",
    "fra.2": "F2",
    "ger.1": "D1",
    "ger.2": "D2",
    "ita.1": "I1",
    "ita.2": "I2",
    "ned.1": "N1",
    "por.1": "P1",
    "sco.1": "SC0",
}
FULL_SCOPE_LEAGUES = [
    "eng.1",
    "eng.2",
    "eng.3",
    "esp.1",
    "esp.2",
    "fra.1",
    "fra.2",
    "ger.1",
    "ger.2",
    "ita.1",
    "ita.2",
    "ned.1",
    "por.1",
    "sco.1",
]
ODDS_PROVIDER_PREFIXES = ["B365", "PS", "Max", "Avg", "BW", "IW", "WH", "VC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Football-Data.co.uk historical 1X2 odds.")
    parser.add_argument("--leagues", default="all", help="Comma list of league ids or all.")
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025", help="Start-year seasons, comma list.")
    parser.add_argument("--output-dir", default="outputs/football_data_odds")
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def season_code(season: int | str) -> str:
    start = int(season)
    return f"{str(start)[-2:]}{str(start + 1)[-2:]}"


def league_slug(league: str) -> str:
    return league.lower().replace(".", "").replace("_", "").replace("-", "")


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def decimal_to_american(value: object) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(odds) or odds <= 1.0:
        return None
    if odds >= 2.0:
        return float((odds - 1.0) * 100.0)
    return float(-100.0 / (odds - 1.0))


def resolve_leagues(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return FULL_SCOPE_LEAGUES.copy()
    leagues = [part.strip() for part in value.split(",") if part.strip()]
    return [league for league in leagues if league in LEAGUE_TO_FOOTBALL_DATA_CODE]


def resolve_seasons(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def source_url(league: str, season: int) -> str:
    return f"{BASE_URL}/{season_code(season)}/{LEAGUE_TO_FOOTBALL_DATA_CODE[league]}.csv"


def download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def first_existing(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return np.nan


def normalize_frame(raw: pd.DataFrame, league: str, season: int, url: str) -> pd.DataFrame:
    out = raw.copy()
    out = out.dropna(how="all")
    if "Date" not in out.columns or "HomeTeam" not in out.columns or "AwayTeam" not in out.columns:
        return pd.DataFrame()
    out["match_date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["match_date", "HomeTeam", "AwayTeam"]).copy()
    rows = []
    for _, row in out.iterrows():
        home_odds_decimal = first_existing(row, [f"{prefix}H" for prefix in ODDS_PROVIDER_PREFIXES])
        draw_odds_decimal = first_existing(row, [f"{prefix}D" for prefix in ODDS_PROVIDER_PREFIXES])
        away_odds_decimal = first_existing(row, [f"{prefix}A" for prefix in ODDS_PROVIDER_PREFIXES])
        home_goals = pd.to_numeric(row.get("FTHG"), errors="coerce")
        away_goals = pd.to_numeric(row.get("FTAG"), errors="coerce")
        if pd.notna(home_goals) and pd.notna(away_goals):
            actual_label = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
        else:
            actual_label = None
        rows.append(
            {
                "competition_id": league,
                "season": season,
                "source": "football-data.co.uk",
                "source_url": url,
                "division": row.get("Div"),
                "match_date": row["match_date"].date().isoformat(),
                "home_team_name": row.get("HomeTeam"),
                "away_team_name": row.get("AwayTeam"),
                "home_team_key": slug(row.get("HomeTeam")),
                "away_team_key": slug(row.get("AwayTeam")),
                "home_goals": None if pd.isna(home_goals) else int(home_goals),
                "away_goals": None if pd.isna(away_goals) else int(away_goals),
                "actual_label": actual_label,
                "home_odds_decimal": pd.to_numeric(home_odds_decimal, errors="coerce"),
                "draw_odds_decimal": pd.to_numeric(draw_odds_decimal, errors="coerce"),
                "away_odds_decimal": pd.to_numeric(away_odds_decimal, errors="coerce"),
                "home_odds": decimal_to_american(home_odds_decimal),
                "draw_odds": decimal_to_american(draw_odds_decimal),
                "away_odds": decimal_to_american(away_odds_decimal),
                "b365_home_decimal": pd.to_numeric(row.get("B365H"), errors="coerce"),
                "b365_draw_decimal": pd.to_numeric(row.get("B365D"), errors="coerce"),
                "b365_away_decimal": pd.to_numeric(row.get("B365A"), errors="coerce"),
                "pinnacle_home_decimal": pd.to_numeric(row.get("PSH"), errors="coerce"),
                "pinnacle_draw_decimal": pd.to_numeric(row.get("PSD"), errors="coerce"),
                "pinnacle_away_decimal": pd.to_numeric(row.get("PSA"), errors="coerce"),
                "avg_home_decimal": pd.to_numeric(row.get("AvgH"), errors="coerce"),
                "avg_draw_decimal": pd.to_numeric(row.get("AvgD"), errors="coerce"),
                "avg_away_decimal": pd.to_numeric(row.get("AvgA"), errors="coerce"),
                "max_home_decimal": pd.to_numeric(row.get("MaxH"), errors="coerce"),
                "max_draw_decimal": pd.to_numeric(row.get("MaxD"), errors="coerce"),
                "max_away_decimal": pd.to_numeric(row.get("MaxA"), errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    frames = []
    for league in resolve_leagues(args.leagues):
        for season in resolve_seasons(args.seasons):
            url = source_url(league, season)
            try:
                raw = download_csv(url)
                raw_path = raw_dir / f"{league_slug(league)}_{season}.csv"
                raw.to_csv(raw_path, index=False)
                normalized = normalize_frame(raw, league, season, url)
                norm_dir = normalized_dir / f"competition={league}" / f"season={season}"
                norm_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = norm_dir / "part-000.parquet"
                normalized.to_parquet(parquet_path, index=False)
                if args.write_csv:
                    normalized.to_csv(norm_dir / "part-000.csv", index=False)
                frames.append(normalized)
                summaries.append(
                    {
                        "competition_id": league,
                        "season": season,
                        "source_url": url,
                        "rows": int(len(normalized)),
                        "rows_with_1x2_odds": int(
                            normalized[["home_odds_decimal", "draw_odds_decimal", "away_odds_decimal"]]
                            .notna()
                            .all(axis=1)
                            .sum()
                        ),
                        "raw_csv": str(raw_path),
                        "normalized_parquet": str(parquet_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep batch backfill going across missing leagues.
                failures.append({"competition_id": league, "season": season, "source_url": url, "error": str(exc)})
    all_odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_path = normalized_dir / "football_data_1x2_odds.parquet"
    all_odds.to_parquet(all_path, index=False)
    if args.write_csv:
        all_odds.to_csv(normalized_dir / "football_data_1x2_odds.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "football-data.co.uk",
        "rows": int(len(all_odds)),
        "completed": summaries,
        "failures": failures,
        "outputs": {"combined_parquet": str(all_path)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 1 if failures and not summaries else 0


if __name__ == "__main__":
    raise SystemExit(main())
