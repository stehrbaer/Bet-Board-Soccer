#!/usr/bin/env python3
"""Collect 1X2 soccer odds and join them to future prediction CSVs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_MARKETS = "h2h"
DEFAULT_REGIONS = "us,uk,eu"
DEFAULT_BOOKMAKER_PREFERENCE = [
    "pinnacle",
    "betfair",
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill",
    "bet365",
    "bwin",
]

LEAGUE_TO_ODDS_KEY = {
    "aut.1": "soccer_austria_bundesliga",
    "den.1": "soccer_denmark_superliga",
    "eng.1": "soccer_epl",
    "eng.2": "soccer_efl_champ",
    "eng.3": "soccer_england_league1",
    "esp.1": "soccer_spain_la_liga",
    "esp.2": "soccer_spain_segunda_division",
    "fra.1": "soccer_france_ligue_one",
    "fra.2": "soccer_france_ligue_two",
    "ger.1": "soccer_germany_bundesliga",
    "ger.2": "soccer_germany_bundesliga2",
    "ita.1": "soccer_italy_serie_a",
    "ita.2": "soccer_italy_serie_b",
    "ned.1": "soccer_netherlands_eredivisie",
    "por.1": "soccer_portugal_primeira_liga",
    "sco.1": "soccer_spl",
    "uefa.champions": "soccer_uefa_champs_league",
    "uefa.europa": "soccer_uefa_europa_league",
    "uefa.europa.conf": "soccer_uefa_europa_conf_league",
}
ODDS_KEY_TO_LEAGUE = {value: key for key, value in LEAGUE_TO_ODDS_KEY.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich soccer future predictions with 1X2 odds.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV to enrich.")
    parser.add_argument("--output-dir", default="outputs/soccer_odds_enriched")
    parser.add_argument("--leagues", default="auto", help="Comma list of league ids, or auto from predictions.")
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--bookmakers", default="", help="Optional comma-separated Odds API bookmaker keys.")
    parser.add_argument("--odds-format", default="american", choices=["american", "decimal"])
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--match-time-tolerance-minutes", type=int, default=240)
    return parser.parse_args()


def api_keys() -> list[str]:
    raw = os.getenv("ODDS_API_KEYS") or os.getenv("ODDS_API_KEY") or ""
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if not keys:
        raise SystemExit("Set ODDS_API_KEY or ODDS_API_KEYS before running this script.")
    return keys


def slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bfc\b|\bafc\b|\bcf\b|\bsc\b|\bac\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_forms(value: object) -> set[str]:
    base = slug(value)
    forms = {base} if base else set()
    replacements = {
        "man utd": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham hotspur",
        "inter milan": "internazionale",
        "psg": "paris saint germain",
    }
    if base in replacements:
        forms.add(replacements[base])
    tokens = base.split()
    if len(tokens) > 1:
        forms.add(tokens[-1])
    return {form for form in forms if form}


def names_match(left: object, right: object) -> bool:
    left_forms = name_forms(left)
    right_forms = name_forms(right)
    if not left_forms or not right_forms:
        return False
    if left_forms & right_forms:
        return True
    return any(a in b or b in a for a in left_forms for b in right_forms if len(a) >= 5 and len(b) >= 5)


def american_to_decimal(value: object) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(odds) or odds == 0:
        return None
    if odds > 0:
        return float(1.0 + odds / 100.0)
    return float(1.0 + 100.0 / abs(odds))


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


def as_american(value: object, odds_format: str) -> float | None:
    if odds_format == "american":
        try:
            odds = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(odds) or odds == 0 else odds
    return decimal_to_american(value)


def implied_from_american(value: object) -> float | None:
    dec = american_to_decimal(value)
    if dec is None:
        return None
    return float(1.0 / dec)


def normalize_leagues(value: str, predictions: pd.DataFrame) -> list[str]:
    if value == "auto":
        source = predictions["competition_id"] if "competition_id" in predictions.columns else predictions["league"]
        leagues = sorted(str(item) for item in source.dropna().unique())
    else:
        leagues = [part.strip() for part in value.split(",") if part.strip()]
    return [league for league in leagues if league in LEAGUE_TO_ODDS_KEY]


def extract_bookmaker_prices(event: dict[str, Any], odds_format: str) -> list[dict[str, Any]]:
    rows = []
    for book in event.get("bookmakers", []) or []:
        if not isinstance(book, dict):
            continue
        markets = book.get("markets", []) or []
        h2h = next((market for market in markets if market.get("key") == "h2h"), None)
        if not h2h:
            continue
        prices = {"home": None, "draw": None, "away": None}
        for outcome in h2h.get("outcomes", []) or []:
            name = slug(outcome.get("name"))
            price = as_american(outcome.get("price"), odds_format)
            if price is None:
                continue
            if names_match(name, event.get("home_team")):
                prices["home"] = price
            elif names_match(name, event.get("away_team")):
                prices["away"] = price
            elif name in {"draw", "tie"}:
                prices["draw"] = price
        if prices["home"] is None and prices["away"] is None and prices["draw"] is None:
            continue
        rows.append(
            {
                "bookmaker_key": book.get("key"),
                "bookmaker_title": book.get("title"),
                "bookmaker_last_update": book.get("last_update"),
                "home_odds": prices["home"],
                "draw_odds": prices["draw"],
                "away_odds": prices["away"],
            }
        )
    return rows


def fetch_odds_for_key(
    session: requests.Session,
    keys: list[str],
    odds_key: str,
    regions: str,
    bookmakers: str,
    odds_format: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {
        "regions": regions,
        "markets": DEFAULT_MARKETS,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    attempts = []
    for key in keys:
        request_params = dict(params)
        request_params["apiKey"] = key
        response = session.get(f"{BASE_URL}/sports/{odds_key}/odds", params=request_params, timeout=30)
        attempts.append(
            {
                "key_suffix": key[-6:],
                "status": response.status_code,
                "requests_remaining": response.headers.get("x-requests-remaining"),
                "requests_used": response.headers.get("x-requests-used"),
            }
        )
        if response.status_code == 200:
            payload = response.json()
            return (payload if isinstance(payload, list) else []), {"attempts": attempts}
        if response.status_code in {401, 403, 429}:
            continue
        if response.status_code in {404, 422}:
            return [], {"attempts": attempts, "unavailable": True}
        raise RuntimeError(f"Odds API request failed for {odds_key}: {response.status_code} {response.text[:300]}")
    return [], {"attempts": attempts, "exhausted": True}


def collect_odds(args: argparse.Namespace, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = api_keys()
    leagues = normalize_leagues(args.leagues, predictions)
    session = requests.Session()
    rows = []
    per_key = {}
    fetched_at = datetime.now(timezone.utc).isoformat()
    for league in leagues:
        odds_key = LEAGUE_TO_ODDS_KEY[league]
        events, meta = fetch_odds_for_key(
            session=session,
            keys=keys,
            odds_key=odds_key,
            regions=args.regions,
            bookmakers=args.bookmakers,
            odds_format=args.odds_format,
        )
        per_key[odds_key] = {"league": league, "raw_events": len(events), **meta}
        for event in events:
            book_rows = extract_bookmaker_prices(event, args.odds_format)
            if not book_rows:
                continue
            consensus = {
                side: float(np.nanmean([book[side] for book in book_rows if book.get(side) is not None]))
                if any(book.get(side) is not None for book in book_rows)
                else None
                for side in ["home_odds", "draw_odds", "away_odds"]
            }
            preferred = next(
                (
                    book
                    for preferred_key in DEFAULT_BOOKMAKER_PREFERENCE
                    for book in book_rows
                    if str(book.get("bookmaker_key", "")).lower() == preferred_key
                ),
                book_rows[0],
            )
            rows.append(
                {
                    "competition_id": league,
                    "odds_sport_key": odds_key,
                    "odds_event_id": event.get("id"),
                    "commence_time_utc": event.get("commence_time"),
                    "home_team_odds": event.get("home_team"),
                    "away_team_odds": event.get("away_team"),
                    "bookmaker_key": preferred.get("bookmaker_key"),
                    "bookmaker_title": preferred.get("bookmaker_title"),
                    "bookmaker_last_update": preferred.get("bookmaker_last_update"),
                    "home_odds": preferred.get("home_odds"),
                    "draw_odds": preferred.get("draw_odds"),
                    "away_odds": preferred.get("away_odds"),
                    "home_implied": implied_from_american(preferred.get("home_odds")),
                    "draw_implied": implied_from_american(preferred.get("draw_odds")),
                    "away_implied": implied_from_american(preferred.get("away_odds")),
                    "consensus_home_odds": consensus["home_odds"],
                    "consensus_draw_odds": consensus["draw_odds"],
                    "consensus_away_odds": consensus["away_odds"],
                    "fetched_at_utc": fetched_at,
                    "bookmaker_count": len(book_rows),
                }
            )
        time.sleep(args.request_delay)
    return pd.DataFrame(rows), {"leagues": leagues, "per_odds_key": per_key, "fetched_at_utc": fetched_at}


def find_odds_match(prediction: pd.Series, odds: pd.DataFrame, tolerance_minutes: int) -> pd.Series | None:
    kickoff = prediction.get("kickoff_utc")
    league = str(prediction.get("competition_id"))
    candidates = odds[odds["competition_id"].astype(str).eq(league)].copy()
    if candidates.empty:
        return None
    candidates["commence_time_utc"] = pd.to_datetime(candidates["commence_time_utc"], errors="coerce", utc=True)
    candidates = candidates.dropna(subset=["commence_time_utc"])
    if pd.notna(kickoff):
        delta = (candidates["commence_time_utc"] - kickoff).abs()
        candidates = candidates[delta <= pd.Timedelta(minutes=tolerance_minutes)].copy()
        candidates["_time_delta_seconds"] = delta.loc[candidates.index].dt.total_seconds()
    if candidates.empty:
        return None
    home = prediction.get("home_team_name")
    away = prediction.get("away_team_name")
    candidates["_team_match"] = [
        names_match(home, row.home_team_odds) and names_match(away, row.away_team_odds)
        for row in candidates.itertuples(index=False)
    ]
    candidates = candidates[candidates["_team_match"]].copy()
    if candidates.empty:
        return None
    return candidates.sort_values(["_time_delta_seconds", "bookmaker_count"], ascending=[True, False]).iloc[0]


def add_roi_columns(predictions: pd.DataFrame, odds: pd.DataFrame, tolerance_minutes: int) -> pd.DataFrame:
    out = predictions.copy()
    if "competition_id" not in out.columns and "league" in out.columns:
        out["competition_id"] = out["league"]
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], errors="coerce", utc=True)
    joined_rows = []
    for _, row in out.iterrows():
        match = find_odds_match(row, odds, tolerance_minutes)
        joined_rows.append({} if match is None else match.to_dict())
    joined = pd.DataFrame(joined_rows)
    for column in [
        "odds_event_id",
        "odds_sport_key",
        "home_team_odds",
        "away_team_odds",
        "bookmaker_key",
        "bookmaker_title",
        "bookmaker_last_update",
        "home_odds",
        "draw_odds",
        "away_odds",
        "home_implied",
        "draw_implied",
        "away_implied",
        "consensus_home_odds",
        "consensus_draw_odds",
        "consensus_away_odds",
        "fetched_at_utc",
        "bookmaker_count",
    ]:
        out[column] = joined[column] if column in joined.columns else np.nan
    out["odds_matched"] = out["odds_event_id"].notna()

    for pick_col in ["prediction", "recommended_pick"]:
        if pick_col not in out.columns:
            continue
        odds_col = f"{pick_col}_odds"
        out[odds_col] = np.select(
            [
                out[pick_col].eq("home"),
                out[pick_col].eq("draw"),
                out[pick_col].eq("away"),
            ],
            [out["home_odds"], out["draw_odds"], out["away_odds"]],
            default=np.nan,
        )
        out[f"{pick_col}_decimal_odds"] = out[odds_col].map(american_to_decimal)
        out[f"{pick_col}_model_prob"] = np.select(
            [
                out[pick_col].eq("home"),
                out[pick_col].eq("draw"),
                out[pick_col].eq("away"),
            ],
            [
                pd.to_numeric(out.get("prob_home"), errors="coerce"),
                pd.to_numeric(out.get("prob_draw"), errors="coerce"),
                pd.to_numeric(out.get("prob_away"), errors="coerce"),
            ],
            default=np.nan,
        )
        out[f"{pick_col}_edge"] = out[f"{pick_col}_model_prob"] - (1.0 / out[f"{pick_col}_decimal_odds"])
        if "actual_label" in out.columns:
            completed = out.get("actual_completed", True)
            completed_mask = completed.astype(bool) if hasattr(completed, "astype") else True
            wins = completed_mask & out["actual_label"].notna() & out[pick_col].eq(out["actual_label"])
            graded = completed_mask & out["actual_label"].notna() & out[f"{pick_col}_decimal_odds"].notna()
            out[f"{pick_col}_profit_1u"] = np.where(
                graded,
                np.where(wins, out[f"{pick_col}_decimal_odds"] - 1.0, -1.0),
                np.nan,
            )
    return out


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    odds, meta = collect_odds(args, predictions)
    enriched = add_roi_columns(predictions, odds, args.match_time_tolerance_minutes)

    odds_path = output_dir / "future_1x2_odds.csv"
    enriched_path = output_dir / "future_predictions_with_odds.csv"
    summary_path = output_dir / "odds_enrichment_summary.json"
    odds.to_csv(odds_path, index=False)
    enriched.to_csv(enriched_path, index=False)
    summary = {
        **meta,
        "predictions": args.predictions,
        "prediction_rows": int(len(predictions)),
        "odds_rows": int(len(odds)),
        "odds_matched_rows": int(enriched["odds_matched"].sum()),
        "rows_with_home_odds": int(enriched["home_odds"].notna().sum()),
        "rows_with_draw_odds": int(enriched["draw_odds"].notna().sum()),
        "rows_with_away_odds": int(enriched["away_odds"].notna().sum()),
        "outputs": {
            "odds": str(odds_path),
            "enriched_predictions": str(enriched_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
