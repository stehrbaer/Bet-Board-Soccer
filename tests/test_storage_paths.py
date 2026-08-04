from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betboard_soccer_extension.storage.object_paths import ObjectPathBuilder


def test_gold_prematch_model_input_key() -> None:
    builder = ObjectPathBuilder("soccer-prediction-data")
    assert (
        builder.gold_prematch_model_input_key("eng.1", "2025")
        == "soccer-prediction-data/gold/prematch_model_input/competition=eng.1/season=2025/part-000.parquet"
    )


def test_raw_key_is_timestamped_and_partitioned() -> None:
    builder = ObjectPathBuilder("soccer-prediction-data")
    key = builder.raw_key(
        "api_football",
        "fixtures",
        datetime(2026, 8, 4, 7, 0, tzinfo=timezone.utc),
        "fixtures.json",
        league="eng.1",
        season="2025",
    )
    assert key == (
        "soccer-prediction-data/raw/api_football/endpoint=fixtures/"
        "league=eng.1/season=2025/extracted_date=2026-08-04/20260804T070000Z_fixtures.json"
    )


def test_rejects_unknown_zone() -> None:
    builder = ObjectPathBuilder()
    with pytest.raises(ValueError):
        builder.key("bad", "x")

