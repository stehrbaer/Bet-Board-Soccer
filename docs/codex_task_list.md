# Codex Task List

## Phase 1: Repo And Infrastructure

- Scaffold isolated Python package and configs.
- Configure DigitalOcean Spaces through environment variables only.
- Implement deterministic object paths for raw, bronze, silver, gold, manifests, and knowledge graph snapshots.
- Implement checksum and manifest helpers.
- Add CLI commands for environment checks and storage path previews.

## Phase 2: Existing Lake Bootstrap

- [x] Read existing Spaces data under `soccer/training_data/historical`.
- [x] Build an inventory of leagues, seasons, row counts, schemas, and object checksums.
- [x] Convert current historical data into the new `soccer-prediction-data/gold/prematch_model_input` contract.
- [x] Upload a versioned gold dataset and build manifest.
- [ ] Add row-count reconciliation report for all gold partitions.

## Phase 3: Validation

- Add Pydantic/Pandera-style schemas for canonical tables.
- Add leakage checks for rolling features and sequence tables.
- Add duplicate match detection.
- Add row-count and schema reconciliation against source manifests.

## Phase 4: Source Collection

- Add Football-Data.co.uk collector for historical results and odds.
- Add football-data.org collector for fixtures and current metadata.
- Add Open-Meteo collector for venue weather.
- Add API-Football adapters later for lineups, injuries, player stats, and events.

## Phase 5: Gemini Handoff

- Publish schema docs for `gold/prematch_model_input`.
- Publish a Colab loading snippet.
- Publish data-quality known gaps.
- Publish feature ontology and dataset version metadata for the Gemini knowledge graph.
