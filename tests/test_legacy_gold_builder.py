from __future__ import annotations

import pandas as pd

from betboard_soccer_extension.features.legacy_gold_builder import _with_contract_columns


def test_with_contract_columns_maps_targets() -> None:
    df = pd.DataFrame(
        {
            "event_id": ["m1", "m2", "m3"],
            "event_date": ["2025-01-01T12:00:00Z", "2025-01-02T12:00:00Z", "2025-01-03T12:00:00Z"],
            "season_year": [2025, 2025, 2025],
            "league_id": ["eng.1", "eng.1", "eng.1"],
            "home_team_id": ["h1", "h2", "h3"],
            "away_team_id": ["a1", "a2", "a3"],
            "home_score": [2, 1, 0],
            "away_score": [0, 1, 2],
            "result_3way": [2, 1, 0],
            "elo_difference": [10.0, 0.0, -10.0],
        }
    )
    gold = _with_contract_columns(df, "eng_1", "2025", "test_schema")
    assert gold["result_target"].tolist() == [0, 1, 2]
    assert gold["match_id"].tolist() == ["m1", "m2", "m3"]
    assert gold["competition_id"].tolist() == ["eng.1", "eng.1", "eng.1"]
    assert list(gold.columns[:3]) == ["match_id", "competition_id", "season"]
    assert "elo_difference" in gold.columns
