from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


FUTURE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "enrich_future_predictions_with_odds.py"
HISTORICAL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "backfill_football_data_odds.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_bookmaker_prices_includes_draw_odds() -> None:
    module = load_module("enrich_future_predictions_with_odds", FUTURE_SCRIPT)
    event = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-08-18T10:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": -120},
                            {"name": "Draw", "price": 260},
                            {"name": "Chelsea", "price": 310},
                        ],
                    }
                ],
            }
        ],
    }

    rows = module.extract_bookmaker_prices(event, "american")

    assert rows == [
        {
            "bookmaker_key": "draftkings",
            "bookmaker_title": "DraftKings",
            "bookmaker_last_update": "2026-08-18T10:00:00Z",
            "home_odds": -120.0,
            "draw_odds": 260.0,
            "away_odds": 310.0,
        }
    ]


def test_add_roi_columns_matches_odds_and_scores_completed_rows() -> None:
    module = load_module("enrich_future_predictions_with_odds", FUTURE_SCRIPT)
    predictions = pd.DataFrame(
        {
            "competition_id": ["eng.1"],
            "kickoff_utc": ["2026-08-18T19:00:00Z"],
            "home_team_name": ["Arsenal"],
            "away_team_name": ["Chelsea"],
            "prob_home": [0.55],
            "prob_draw": [0.25],
            "prob_away": [0.20],
            "prediction": ["home"],
            "recommended_pick": ["draw"],
            "actual_completed": [True],
            "actual_label": ["draw"],
        }
    )
    odds = pd.DataFrame(
        {
            "competition_id": ["eng.1"],
            "odds_event_id": ["odds-1"],
            "odds_sport_key": ["soccer_epl"],
            "commence_time_utc": ["2026-08-18T19:00:00Z"],
            "home_team_odds": ["Arsenal"],
            "away_team_odds": ["Chelsea"],
            "bookmaker_key": ["draftkings"],
            "bookmaker_title": ["DraftKings"],
            "bookmaker_last_update": ["2026-08-18T10:00:00Z"],
            "home_odds": [-120.0],
            "draw_odds": [260.0],
            "away_odds": [310.0],
            "home_implied": [0.545],
            "draw_implied": [0.278],
            "away_implied": [0.244],
            "fetched_at_utc": ["2026-08-18T10:00:00Z"],
            "bookmaker_count": [1],
        }
    )

    out = module.add_roi_columns(predictions, odds, tolerance_minutes=60)

    assert bool(out.loc[0, "odds_matched"]) is True
    assert out.loc[0, "prediction_odds"] == -120.0
    assert out.loc[0, "recommended_pick_odds"] == 260.0
    assert out.loc[0, "prediction_profit_1u"] == -1.0
    assert round(out.loc[0, "recommended_pick_profit_1u"], 2) == 2.6


def test_football_data_normalize_frame_outputs_1x2_odds() -> None:
    module = load_module("backfill_football_data_odds", HISTORICAL_SCRIPT)
    raw = pd.DataFrame(
        {
            "Div": ["E0"],
            "Date": ["15/08/2025"],
            "HomeTeam": ["Liverpool"],
            "AwayTeam": ["Bournemouth"],
            "FTHG": [4],
            "FTAG": [2],
            "B365H": [1.40],
            "B365D": [4.75],
            "B365A": [8.00],
            "AvgH": [1.42],
            "AvgD": [4.60],
            "AvgA": [7.50],
        }
    )

    out = module.normalize_frame(raw, "eng.1", 2025, "https://example.test/E0.csv")

    assert out.loc[0, "actual_label"] == "home"
    assert out.loc[0, "home_odds_decimal"] == 1.40
    assert out.loc[0, "draw_odds_decimal"] == 4.75
    assert out.loc[0, "away_odds_decimal"] == 8.00
    assert round(out.loc[0, "home_odds"], 2) == -250.0
    assert out.loc[0, "draw_odds"] == 375.0
    assert out.loc[0, "away_odds"] == 700.0
