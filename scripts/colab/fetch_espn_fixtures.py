#!/usr/bin/env python3
"""Fetch future soccer fixtures from ESPN's public site API."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

LEAGUE_ALIASES = {
    "eng1": "eng.1",
    "eng_1": "eng.1",
    "epl": "eng.1",
    "premierleague": "eng.1",
    "esp1": "esp.1",
    "esp_1": "esp.1",
    "laliga": "esp.1",
    "ger1": "ger.1",
    "ger_1": "ger.1",
    "bundesliga": "ger.1",
    "ita1": "ita.1",
    "ita_1": "ita.1",
    "seriea": "ita.1",
    "fra1": "fra.1",
    "fra_1": "fra.1",
    "ligue1": "fra.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch future fixtures from ESPN.")
    parser.add_argument("--league", default="eng1", help="League alias or ESPN slug, e.g. eng1, epl, eng.1.")
    parser.add_argument("--start-date", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="End date, YYYY-MM-DD. Defaults to start + 60 days.")
    parser.add_argument("--weeks", type=int, default=5, help="Keep the first N fixture weeks in normalized output.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/fixtures/espn_eng1_2026")
    return parser.parse_args()


def normalize_league(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(".", "_").replace(" ", "")
    return LEAGUE_ALIASES.get(normalized, value.strip().lower())


def espn_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")


def fetch_scoreboard(league: str, start_date: str, end_date: str, limit: int) -> dict[str, Any]:
    url = f"{ESPN_BASE_URL}/{league}/scoreboard"
    response = requests.get(
        url,
        params={
            "dates": f"{espn_date(start_date)}-{espn_date(end_date)}",
            "limit": limit,
            "region": "us",
            "lang": "en",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def competitor_fields(competition: dict[str, Any], home_away: str) -> dict[str, Any]:
    for competitor in competition.get("competitors", []):
        if competitor.get("homeAway") == home_away:
            team = competitor.get("team", {})
            return {
                f"{home_away}_team_id": team.get("id"),
                f"{home_away}_team_uid": team.get("uid"),
                f"{home_away}_team_name": team.get("displayName") or team.get("name"),
                f"{home_away}_team_short_name": team.get("shortDisplayName") or team.get("abbreviation"),
                f"{home_away}_team_abbrev": team.get("abbreviation"),
            }
    return {
        f"{home_away}_team_id": None,
        f"{home_away}_team_uid": None,
        f"{home_away}_team_name": None,
        f"{home_away}_team_short_name": None,
        f"{home_away}_team_abbrev": None,
    }


def normalize_events(payload: dict[str, Any], league: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions", [])
        competition = competitions[0] if competitions else {}
        status = competition.get("status", {}).get("type", {})
        venue = competition.get("venue", {})
        row = {
            "espn_event_id": event.get("id"),
            "competition_id": league,
            "season": event.get("season", {}).get("year"),
            "kickoff_utc": event.get("date"),
            "name": event.get("name"),
            "short_name": event.get("shortName"),
            "status": status.get("description"),
            "status_state": status.get("state"),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("fullName"),
            "venue_city": venue.get("address", {}).get("city"),
            "venue_country": venue.get("address", {}).get("country"),
        }
        row.update(competitor_fields(competition, "home"))
        row.update(competitor_fields(competition, "away"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    return df.dropna(subset=["kickoff_utc"]).sort_values(["kickoff_utc", "espn_event_id"]).reset_index(drop=True)


def first_fixture_weeks(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["week_start"] = (out["kickoff_utc"] - pd.to_timedelta(out["kickoff_utc"].dt.weekday, unit="D")).dt.date.astype(str)
    keep_weeks = out["week_start"].drop_duplicates().head(weeks).tolist()
    return out[out["week_start"].isin(keep_weeks)].copy()


def main() -> int:
    args = parse_args()
    league = normalize_league(args.league)
    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = args.end_date or (start + timedelta(days=60)).strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = fetch_scoreboard(league, args.start_date, end_date, args.limit)
    raw_path = output_dir / f"espn_{league}_{espn_date(args.start_date)}_{espn_date(end_date)}_raw.json"
    raw_path.write_text(json.dumps(payload, indent=2) + "\n")

    fixtures = normalize_events(payload, league)
    first_weeks = first_fixture_weeks(fixtures, args.weeks)
    fixtures_path = output_dir / f"{league}_fixtures.csv"
    first_weeks_path = output_dir / f"{league}_first_{args.weeks}_weeks_fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)
    first_weeks.to_csv(first_weeks_path, index=False)

    summary = {
        "league": league,
        "start_date": args.start_date,
        "end_date": end_date,
        "events": len(fixtures),
        "first_weeks_events": len(first_weeks),
        "raw_json": str(raw_path),
        "fixtures_csv": str(fixtures_path),
        "first_weeks_csv": str(first_weeks_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
