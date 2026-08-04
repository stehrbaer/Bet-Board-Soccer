# BetBoard Soccer Extension

Data engineering extension for BetBoard soccer prediction work.

This repository owns the Codex side of the project:

- source ingestion
- raw/bronze/silver/gold dataset structure
- DigitalOcean Spaces upload
- manifests, checksums, and validation
- Gemini/Colab-ready Parquet contracts

It does not own model training, attribution, or notebook-specific explanation logic.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Credentials must come from environment variables or your shell profile. Do not commit `.env`.

If you already have the existing local profile from `ML-Model`, this repo can reuse it:

```bash
export DO_SPACES_PROFILE=do-spaces
```

Direct `DO_SPACES_KEY` and `DO_SPACES_SECRET` values take priority when present.

## DigitalOcean Defaults

The initial configuration reuses the existing BetBoard Spaces setup:

- bucket: `betboard-ml-artifacts`
- endpoint: `https://fra1.digitaloceanspaces.com`
- root prefix: `soccer-prediction-data`

The existing historical soccer lake remains under:

```text
soccer/training_data/historical/
```

This extension will publish new project outputs under:

```text
soccer-prediction-data/
  raw/
  bronze/
  silver/
  gold/
  manifests/
  knowledge_graph/
```

## CLI

```bash
python -m betboard_soccer_extension.cli storage-plan
python -m betboard_soccer_extension.cli env-check
python -m betboard_soccer_extension.cli legacy-inventory --limit 10
```

## First Milestone

1. Reuse the existing historical soccer lake as a source.
2. Build `gold/prematch_model_input`.
3. Upload versioned Parquet datasets and manifests to Spaces.
4. Produce a Gemini handoff document with exact paths and schema.
