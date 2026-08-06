from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "train_soccer_three_way_nn_optuna.py"


def load_script_module():
    sys.modules.setdefault("optuna", types.SimpleNamespace(Trial=object))
    spec = importlib.util.spec_from_file_location("train_soccer_three_way_nn_optuna", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_league_aliases_resolve_to_gold_partition_ids() -> None:
    module = load_script_module()

    assert module.normalize_league("eng1") == "eng.1"
    assert module.normalize_league("eng_1") == "eng.1"
    assert module.normalize_league("EPL") == "eng.1"
    assert module.normalize_league("ger2") == "ger.2"
    assert module.normalize_league("eredivisie") == "ned.1"


def test_league_all_resolves_to_no_competition_filter() -> None:
    module = load_script_module()
    args = argparse.Namespace(league="all", competitions="")

    assert module.resolve_competitions(args) == []


def test_competitions_all_resolves_to_no_competition_filter() -> None:
    module = load_script_module()
    args = argparse.Namespace(league="eng1", competitions="all")

    assert module.resolve_competitions(args) == []


def test_league_slug_for_output_folder_names() -> None:
    module = load_script_module()

    assert module.league_slug("eng.1") == "eng1"
    assert module.league_slug("uefa.champions") == "uefachampions"


def test_required_scope_seasons_are_inclusive() -> None:
    module = load_script_module()

    assert module.required_scope_seasons("2021", "2025") == ["2021", "2022", "2023", "2024", "2025"]


def test_export_next_games_predictions_writes_compact_csv(tmp_path: Path) -> None:
    module = load_script_module()
    predictions = pd.DataFrame(
        {
            "match_id": [2, 1, 3],
            "competition_id": ["eng.1", "eng.1", "eng.1"],
            "season": ["2025", "2025", "2025"],
            "kickoff_utc": ["2025-08-16T14:00:00Z", "2025-08-15T19:00:00Z", "2025-08-17T13:00:00Z"],
            "home_team_id": [10, 11, 12],
            "away_team_id": [20, 21, 22],
            "home_team_name": ["Hull City", "Arsenal", "Everton"],
            "away_team_name": ["Manchester United", "Coventry City", "Crystal Palace"],
            "result_target": [2, 0, 1],
            "prob_home": [0.2, 0.7, 0.3],
            "prob_draw": [0.2, 0.2, 0.4],
            "prob_away": [0.6, 0.1, 0.3],
            "predicted_target": [2, 0, 1],
            "predicted_label": ["away", "home", "draw"],
        }
    )

    artifact = module.export_next_games_predictions(predictions, tmp_path, 2)

    assert artifact["rows"] == 2
    out = pd.read_csv(tmp_path / "next_2_games_predictions.csv")
    assert out["matchup_key"].tolist() == [
        "1_arsenal_coventry_city",
        "2_hull_city_manchester_united",
    ]
