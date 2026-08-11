from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "fetch_espn_fixtures.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("fetch_espn_fixtures", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_leagues_all_uses_full_scope_leagues() -> None:
    module = load_script_module()

    leagues = module.resolve_leagues("all")

    assert "eng.1" in leagues
    assert "ger.1" in leagues
    assert "ita.1" in leagues


def test_resolve_leagues_accepts_comma_list_aliases() -> None:
    module = load_script_module()

    assert module.resolve_leagues("epl,bundesliga2,eredivisie") == ["eng.1", "ger.2", "ned.1"]


def test_league_slug_matches_training_folder_style() -> None:
    module = load_script_module()

    assert module.league_slug("eng.1") == "eng1"
    assert module.league_slug("fra.2") == "fra2"


def test_add_derived_game_week_uses_lookback_weeks_then_filters_start() -> None:
    module = load_script_module()
    fixtures = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(
                ["2026-07-31T18:00:00Z", "2026-08-07T18:00:00Z", "2026-08-14T18:00:00Z"],
                utc=True,
            ),
            "espn_event_id": ["1", "2", "3"],
        }
    )

    out = module.add_derived_game_week(fixtures, "2026-08-10")

    assert out["espn_event_id"].tolist() == ["3"]
    assert out["game_week"].tolist() == [3]
