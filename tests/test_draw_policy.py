from __future__ import annotations

import pandas as pd

from betboard_soccer_extension.modeling.draw_policy import DrawPolicy, apply_draw_policy, evaluate_draw_policy


def test_apply_draw_policy_recommends_draw_when_thresholds_match() -> None:
    policy = DrawPolicy(
        version="test",
        min_draw_p=0.30,
        max_draw_gap=0.10,
        max_side_gap=0.06,
        min_balanced_draw_p=0.28,
        min_draw_pick_p=0.22,
    )
    df = pd.DataFrame({"prob_home": [0.36], "prob_draw": [0.34], "prob_away": [0.30]})

    out = apply_draw_policy(df, policy)

    assert out.loc[0, "raw_model_pick"] == "home"
    assert out.loc[0, "draw_risk"]
    assert out.loc[0, "recommended_pick"] == "draw"
    assert out.loc[0, "draw_policy_version"] == "test"


def test_evaluate_draw_policy_reports_draw_capture() -> None:
    policy = DrawPolicy(
        version="test",
        min_draw_p=0.30,
        max_draw_gap=0.10,
        max_side_gap=0.06,
        min_balanced_draw_p=0.28,
        min_draw_pick_p=0.22,
    )
    df = pd.DataFrame(
        {
            "prob_home": [0.36, 0.70],
            "prob_draw": [0.34, 0.20],
            "prob_away": [0.30, 0.10],
            "result_target": [1, 0],
        }
    )
    out = apply_draw_policy(df, policy)

    metrics = evaluate_draw_policy(out)

    assert metrics["policy_accuracy"] == 1.0
    assert metrics["draw_pick_precision"] == 1.0
    assert metrics["draw_pick_recall"] == 1.0
