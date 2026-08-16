from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "evaluate_future_predictions.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_future_predictions", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_scoreboard_results_extracts_completed_result() -> None:
    module = load_script_module()
    payload = {
        "events": [
            {
                "id": "740596",
                "competitions": [
                    {
                        "status": {"type": {"completed": True, "description": "Final", "state": "post"}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": "2",
                                "team": {"displayName": "Arsenal"},
                            },
                            {
                                "homeAway": "away",
                                "score": "1",
                                "team": {"displayName": "Coventry City"},
                            },
                        ],
                    }
                ],
            }
        ]
    }

    actuals = module.parse_scoreboard_results(payload, "eng.1")

    assert actuals.loc[0, "match_id"] == "740596"
    assert bool(actuals.loc[0, "actual_completed"]) is True
    assert actuals.loc[0, "actual_label"] == "home"
    assert actuals.loc[0, "actual_home_score"] == 2.0
    assert actuals.loc[0, "actual_away_score"] == 1.0


def test_evaluate_predictions_scores_raw_and_recommended_picks() -> None:
    module = load_script_module()
    predictions = pd.DataFrame(
        {
            "competition_id": ["eng.1", "eng.1", "eng.1"],
            "match_id": ["1", "2", "3"],
            "kickoff_utc": ["2026-08-01T12:00:00Z"] * 3,
            "prob_home": [0.7, 0.3, 0.2],
            "prob_draw": [0.2, 0.4, 0.3],
            "prob_away": [0.1, 0.3, 0.5],
            "prediction": ["home", "draw", "away"],
            "recommended_pick": ["home", "home", "away"],
        }
    )
    actuals = pd.DataFrame(
        {
            "competition_id": ["eng.1", "eng.1"],
            "match_id": ["1", "2"],
            "actual_completed": [True, True],
            "actual_label": ["home", "draw"],
            "actual_status": ["Final", "Final"],
            "actual_status_state": ["post", "post"],
            "actual_home_score": [1, 0],
            "actual_away_score": [0, 0],
        }
    )

    scored, summary, by_league, by_result_type = module.evaluate_predictions(predictions, actuals)

    assert len(scored) == 3
    assert summary["completed_rows"] == 2
    assert summary["pending_or_unmatched_rows"] == 1
    assert summary["raw_accuracy"] == 1.0
    assert summary["recommended_accuracy"] == 0.5
    assert summary["recommended_accuracy_delta"] == -0.5
    assert summary["actual_draw_count"] == 1
    assert summary["actual_home_win_count"] == 1
    assert summary["actual_away_win_home_loss_count"] == 0
    assert summary["raw_home_win_accuracy"] == 1.0
    assert summary["raw_draw_accuracy"] == 1.0
    assert summary["recommended_home_win_accuracy"] == 1.0
    assert summary["recommended_draw_accuracy"] == 0.0
    assert by_league.loc[0, "competition_id"] == "eng.1"
    assert by_league.loc[0, "recommended_draw_accuracy"] == 0.0
    draw_row = by_result_type[by_result_type["actual_result_type"].eq("draw")].iloc[0]
    assert draw_row["completed_rows"] == 1
    assert draw_row["raw_accuracy"] == 1.0
    assert draw_row["recommended_accuracy"] == 0.0


def test_evaluate_predictions_falls_back_to_team_and_kickoff_matching() -> None:
    module = load_script_module()
    predictions = pd.DataFrame(
        {
            "league": ["eng.1"],
            "kickoff_utc": ["2026-08-01T12:00:00Z"],
            "home_team_name": ["Arsenal"],
            "away_team_name": ["Coventry City"],
            "prob_home": [0.7],
            "prob_draw": [0.2],
            "prob_away": [0.1],
            "prediction": ["home"],
            "recommended_pick": ["home"],
        }
    )
    actuals = pd.DataFrame(
        {
            "competition_id": ["eng.1"],
            "match_id": ["740596"],
            "actual_kickoff_utc": pd.to_datetime(["2026-08-01T12:00:00Z"], utc=True),
            "actual_completed": [True],
            "actual_label": ["home"],
            "actual_status": ["Final"],
            "actual_status_state": ["post"],
            "actual_home_score": [2],
            "actual_away_score": [1],
            "actual_home_team_name": ["Arsenal"],
            "actual_away_team_name": ["Coventry City"],
            "home_team_match_key": ["arsenal"],
            "away_team_match_key": ["coventry_city"],
        }
    )

    scored, summary, _, _ = module.evaluate_predictions(predictions, actuals)

    assert scored.loc[0, "match_id"] == "740596"
    assert summary["completed_rows"] == 1
    assert summary["raw_accuracy"] == 1.0


def test_load_predictions_from_batch_summary(tmp_path: Path) -> None:
    module = load_script_module()
    predictions_path = tmp_path / "models" / "eng1_soccer_nn" / "future_2026" / "future_predictions.csv"
    predictions_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "competition_id": ["eng.1"],
            "match_id": ["1"],
            "kickoff_utc": ["2026-08-01T12:00:00Z"],
            "prob_home": [0.6],
            "prob_draw": [0.2],
            "prob_away": [0.2],
            "prediction": ["home"],
        }
    ).to_csv(predictions_path, index=False)
    summary_path = tmp_path / "models" / "future_prediction_batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "completed": [
                    {
                        "competition_id": "eng.1",
                        "future_predictions": "models/eng1_soccer_nn/future_2026/future_predictions.csv",
                    }
                ]
            }
        )
    )
    args = type(
        "Args",
        (),
        {"predictions": "", "batch_summary": str(summary_path), "repo_root": str(tmp_path)},
    )()

    predictions = module.load_predictions(args)

    assert len(predictions) == 1
    assert predictions.loc[0, "league"] == "eng.1"
    assert predictions.loc[0, "source_prediction_file"].endswith("future_predictions.csv")
