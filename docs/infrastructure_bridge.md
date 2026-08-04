# Infrastructure Bridge

This repo is intentionally separate from `ML-Model`, but it can reuse the working local infrastructure.

## DigitalOcean Spaces

Existing setup:

- bucket: `betboard-ml-artifacts`
- endpoint: `https://fra1.digitaloceanspaces.com`
- profile: `do-spaces`

Use the existing profile without copying secrets:

```bash
export DO_SPACES_PROFILE=do-spaces
.venv/bin/python -m betboard_soccer_extension.cli legacy-inventory --limit 10
```

The legacy historical soccer source currently lives at:

```text
s3://betboard-ml-artifacts/soccer/training_data/historical/
```

New extension outputs should be written under:

```text
s3://betboard-ml-artifacts/soccer-prediction-data/
```

## Local Source Repo

Use `/Users/davidstehr/Betboard/ML-Model` as a read-only reference for:

- current soccer league IDs
- existing historical training exports
- source coverage audits
- known data gaps

Do not import from `ML-Model` directly in production pipeline code. Copy stable contracts into configs and keep this repo reproducible.
