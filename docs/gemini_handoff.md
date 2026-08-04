# Gemini Handoff: Soccer Gold Dataset

## Dataset

Primary Gemini input:

```text
s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input/
```

Partitioning:

```text
competition=<competition_id>/season=<season>/part-000.parquet
```

Current build:

```text
run_id=legacy-gold-2026-08-04
partitions=193
rows_processed_this_run=28649
manifest=s3://betboard-ml-artifacts/soccer-prediction-data/manifests/feature_builds/run_id=legacy-gold-2026-08-04/manifest.json
```

## Target Contract

`result_target` uses the Gemini contract:

```text
0 = home win
1 = draw
2 = away win
```

The legacy BetBoard source used:

```text
2 = home win
1 = draw
0 = away win
```

The conversion is handled by `betboard_soccer_extension.features.legacy_gold_builder`.

## Required Columns

The first columns in each partition are:

```text
match_id
competition_id
season
kickoff_utc
home_team_id
away_team_id
home_goals
away_goals
result_target
feature_cutoff_utc
feature_build_timestamp
feature_schema_version
team_data_completeness
player_data_completeness
lineup_data_status
weather_data_status
```

The remaining columns preserve the wide legacy training feature set.

## Current Data-Quality Status

This first gold build is a compatibility layer over the existing historical soccer lake.

```text
team_data_completeness=legacy_training
player_data_completeness=not_available
lineup_data_status=not_available
weather_data_status=not_available
```

Player, lineup, injury, and weather features should be added in later pipeline phases with strict `information_timestamp` and cutoff rules.

## Colab Loading Sketch

```python
import os
import pandas as pd

storage_options = {
    "key": os.environ["DO_SPACES_KEY"],
    "secret": os.environ["DO_SPACES_SECRET"],
    "client_kwargs": {
        "endpoint_url": "https://fra1.digitaloceanspaces.com",
        "region_name": "fra1",
    },
}

base = "s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input"
df = pd.read_parquet(
    f"{base}/competition=eng.1/season=2025/part-000.parquet",
    storage_options=storage_options,
)
```

For a multi-partition load, use `pyarrow.dataset`, DuckDB, or iterate over manifest objects.

