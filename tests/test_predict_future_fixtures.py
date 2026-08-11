from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "predict_future_fixtures.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("predict_future_fixtures", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_competition_from_fixture_path() -> None:
    module = load_script_module()

    path = Path("outputs/fixtures/espn_by_league_2026/eng1_fixtures/eng.1_first_5_weeks_fixtures.csv")

    assert module.competition_from_fixture_path(path) == "eng.1"


def test_default_history_partitions_include_country_depth() -> None:
    module = load_script_module()

    assert module.default_history_partitions_for_competition("eng.1", "2025") == "eng.1:2025,eng.2:2025,eng.3:2025"
    assert module.default_history_partitions_for_competition("aut.1", "2025") == "aut.1:2025"
    assert module.default_history_partitions_for_competition("uefa.champions_qual", "2026") == "uefa.champions.qual:2026"


def test_build_future_feature_frame_preserves_game_week() -> None:
    module = load_script_module()
    fixtures = pd.DataFrame(
        {
            "kickoff_utc": ["2026-08-14T18:00:00Z"],
            "week_start": ["2026-08-10"],
            "game_week": [2],
            "matchweek": [2],
            "relative_week": [1],
            "matchup_number": [1],
            "competition_id": ["ger.2"],
            "season": [2026],
            "espn_event_id": ["401"],
            "home_team_id": [100],
            "away_team_id": [200],
            "home_team_name": ["Home Club"],
            "away_team_name": ["Away Club"],
        }
    )

    future, _ = module.build_future_feature_frame(fixtures, [], {})

    assert future.loc[0, "game_week"] == 2
    assert future.loc[0, "relative_week"] == 1
    assert future.loc[0, "matchup_key"] == "2_home_club_away_club"


def test_predict_each_league_writes_batch_summary(tmp_path: Path, monkeypatch) -> None:
    module = load_script_module()
    fixtures_dir = tmp_path / "fixtures" / "eng1_fixtures"
    fixtures_dir.mkdir(parents=True)
    fixtures_path = fixtures_dir / "eng.1_first_5_weeks_fixtures.csv"
    pd.DataFrame({"kickoff_utc": ["2026-08-15T12:00:00Z"]}).to_csv(fixtures_path, index=False)

    model_dir = tmp_path / "models" / "eng1_soccer_nn"
    model_dir.mkdir(parents=True)
    (model_dir / "soccer_three_way_nn.keras").write_text("model")
    (model_dir / "preprocessing.joblib").write_text("preprocessing")

    def fake_predict_fixture_file(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "future_predictions.csv"
        predictions_path.write_text("x\n")
        return {
            "fixtures_scored": 1,
            "outputs": {"future_predictions": str(predictions_path)},
        }

    monkeypatch.setattr(module, "predict_fixture_file", fake_predict_fixture_file)
    args = argparse.Namespace(
        fixtures_root=str(tmp_path / "fixtures"),
        models_root=str(tmp_path / "models"),
        global_model_dir="",
        history_season="2025",
        gold_root="gold",
        draw_policy="",
        use_league_draw_policy=True,
    )

    summary = module.predict_each_league(args)

    assert len(summary["completed"]) == 1
    assert summary["completed"][0]["competition_id"] == "eng.1"
    assert (tmp_path / "models" / "eng1_soccer_nn" / "future_2026" / "future_predictions.csv").exists()
    assert (tmp_path / "models" / "future_prediction_batch_summary.json").exists()
