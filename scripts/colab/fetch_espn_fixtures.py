#!/usr/bin/env python3
"""Fetch future soccer fixtures from ESPN's public site API."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests


ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

LEAGUE_ALIASES = {
    "aut1": "aut.1",
    "aut_1": "aut.1",
    "austria": "aut.1",
    "den1": "den.1",
    "den_1": "den.1",
    "denmark": "den.1",
    "eng1": "eng.1",
    "eng_1": "eng.1",
    "epl": "eng.1",
    "premierleague": "eng.1",
    "eng2": "eng.2",
    "eng_2": "eng.2",
    "championship": "eng.2",
    "eng3": "eng.3",
    "eng_3": "eng.3",
    "leagueone": "eng.3",
    "esp1": "esp.1",
    "esp_1": "esp.1",
    "laliga": "esp.1",
    "esp2": "esp.2",
    "esp_2": "esp.2",
    "laliga2": "esp.2",
    "ger1": "ger.1",
    "ger_1": "ger.1",
    "bundesliga": "ger.1",
    "ger2": "ger.2",
    "ger_2": "ger.2",
    "bundesliga2": "ger.2",
    "ita1": "ita.1",
    "ita_1": "ita.1",
    "seriea": "ita.1",
    "ita2": "ita.2",
    "ita_2": "ita.2",
    "serieb": "ita.2",
    "fra1": "fra.1",
    "fra_1": "fra.1",
    "ligue1": "fra.1",
    "fra2": "fra.2",
    "fra_2": "fra.2",
    "ligue2": "fra.2",
    "jpn1": "jpn.1",
    "jpn_1": "jpn.1",
    "mex1": "mex.1",
    "mex_1": "mex.1",
    "ned1": "ned.1",
    "ned_1": "ned.1",
    "eredivisie": "ned.1",
    "nor1": "nor.1",
    "nor_1": "nor.1",
    "por1": "por.1",
    "por_1": "por.1",
    "portugal": "por.1",
    "sco1": "sco.1",
    "sco_1": "sco.1",
    "scotland": "sco.1",
    "swe1": "swe.1",
    "swe_1": "swe.1",
    "usa1": "usa.1",
    "usa_1": "usa.1",
    "mls": "usa.1",
}

FULL_SCOPE_LEAGUES = [
    "aut.1",
    "den.1",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch future fixtures from ESPN.")
    parser.add_argument("--league", default="eng1", help="League alias, ESPN slug, comma list, or all.")
    parser.add_argument("--start-date", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="End date, YYYY-MM-DD. Defaults to start + 60 days.")
    parser.add_argument("--weeks", type=int, default=5, help="Keep the first N fixture weeks in normalized output.")
    parser.add_argument(
        "--game-week-lookback-days",
        type=int,
        default=60,
        help="Days before start-date to scan for deriving actual season game_week values.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/fixtures/espn_eng1_2026")
    parser.add_argument(
        "--fetch-each-league",
        action="store_true",
        help="Fetch each selected league into output-dir/<league_slug>_fixtures.",
    )
    return parser.parse_args()


def normalize_league(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(".", "_").replace(" ", "")
    return LEAGUE_ALIASES.get(normalized, value.strip().lower())


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def league_slug(league: str) -> str:
    return league.lower().replace(".", "").replace("_", "").replace("-", "")


def resolve_leagues(value: str) -> list[str]:
    leagues = [normalize_league(part) for part in csv_list(value)]
    if not leagues or leagues == ["all"]:
        return FULL_SCOPE_LEAGUES.copy()
    if "all" in leagues:
        raise SystemExit("--league all cannot be combined with specific leagues.")
    return leagues


def espn_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


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


def merge_payload_events(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {"events": []}
    seen = set()
    for payload in payloads:
        for event in payload.get("events", []):
            event_id = event.get("id")
            if event_id in seen:
                continue
            seen.add(event_id)
            merged["events"].append(event)
    return merged


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
            "season_slug": event.get("season", {}).get("slug"),
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
    if "game_week" not in out.columns:
        week_numbers = {week: idx for idx, week in enumerate(out["week_start"].drop_duplicates(), start=1)}
        out["game_week"] = out["week_start"].map(week_numbers)
    else:
        out["game_week"] = out["game_week"].fillna(out["week_start"].map({week: idx for idx, week in enumerate(out["week_start"].drop_duplicates(), start=1)}))
    out["matchweek"] = out["game_week"]
    relative_week_numbers = {week: idx for idx, week in enumerate(out["week_start"].drop_duplicates(), start=1)}
    out["relative_week"] = out["week_start"].map(relative_week_numbers)
    out["matchup_number"] = out.groupby("week_start").cumcount() + 1
    out["matchup_key"] = [
        f"{game_week}_{slug(home)}_{slug(away)}"
        for game_week, home, away in zip(out["game_week"], out["home_team_name"], out["away_team_name"])
    ]
    keep_weeks = out["week_start"].drop_duplicates().head(weeks).tolist()
    return out[out["week_start"].isin(keep_weeks)].copy()


def add_derived_game_week(fixtures: pd.DataFrame, requested_start_date: str) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    out = fixtures.copy()
    out["week_start"] = (out["kickoff_utc"] - pd.to_timedelta(out["kickoff_utc"].dt.weekday, unit="D")).dt.date.astype(str)
    week_numbers = {week: idx for idx, week in enumerate(out["week_start"].drop_duplicates(), start=1)}
    out["game_week"] = out["week_start"].map(week_numbers)
    out["relative_week"] = out["week_start"].map(week_numbers)
    requested_start = pd.Timestamp(requested_start_date, tz="UTC")
    return out[out["kickoff_utc"] >= requested_start].copy()


def display_fixture_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "week_start",
        "game_week",
        "matchweek",
        "relative_week",
        "matchup_number",
        "matchup_key",
        "kickoff_utc",
        "competition_id",
        "season",
        "espn_event_id",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "venue_name",
        "venue_city",
        "status",
    ]
    return df[[column for column in columns if column in df.columns]].copy()


def fetch_league(args: argparse.Namespace, league: str, output_dir: Path, end_date: str) -> dict[str, Any]:
    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)

    lookback_start = (start - timedelta(days=args.game_week_lookback_days)).strftime("%Y-%m-%d")
    requested_payload = fetch_scoreboard(league, args.start_date, end_date, args.limit)
    week_index_payload = fetch_scoreboard(league, lookback_start, end_date, args.limit)
    payload = merge_payload_events(requested_payload, week_index_payload)
    raw_path = output_dir / f"espn_{league}_{espn_date(args.start_date)}_{espn_date(end_date)}_raw.json"
    raw_path.write_text(json.dumps(payload, indent=2) + "\n")

    fixtures = add_derived_game_week(normalize_events(payload, league), args.start_date)
    first_weeks = first_fixture_weeks(fixtures, args.weeks)
    fixtures_path = output_dir / f"{league}_fixtures.csv"
    first_weeks_path = output_dir / f"{league}_first_{args.weeks}_weeks_fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)
    display_fixture_columns(first_weeks).to_csv(first_weeks_path, index=False)

    summary = {
        "league": league,
        "start_date": args.start_date,
        "end_date": end_date,
        "game_week_lookback_start": lookback_start,
        "game_week_lookback_days": args.game_week_lookback_days,
        "events": len(fixtures),
        "first_weeks_events": len(first_weeks),
        "raw_json": str(raw_path),
        "fixtures_csv": str(fixtures_path),
        "first_weeks_csv": str(first_weeks_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = args.end_date or (start + timedelta(days=60)).strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    leagues = resolve_leagues(args.league)
    if not args.fetch_each_league and len(leagues) != 1:
        raise SystemExit("Multiple leagues require --fetch-each-league.")

    if args.fetch_each_league:
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        failures = []
        for league in leagues:
            league_output_dir = output_dir / f"{league_slug(league)}_fixtures"
            try:
                summaries.append(fetch_league(args, league, league_output_dir, end_date))
            except Exception as exc:  # noqa: BLE001 - keep batch fetching resilient across ESPN league gaps.
                failures.append({"league": league, "error": str(exc)})
        batch_summary = {
            "start_date": args.start_date,
            "end_date": end_date,
            "weeks": args.weeks,
            "leagues_requested": leagues,
            "completed": summaries,
            "failures": failures,
        }
        (output_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2, default=str) + "\n")
        print(json.dumps(batch_summary, indent=2, default=str))
        return 1 if failures and not summaries else 0

    fetch_league(args, leagues[0], output_dir, end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
