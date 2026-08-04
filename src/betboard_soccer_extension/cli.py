from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os

from betboard_soccer_extension.features.legacy_gold_builder import build_gold_from_legacy_historical
from betboard_soccer_extension.storage.config import SpacesConfig
from betboard_soccer_extension.storage.object_paths import ObjectPathBuilder
from betboard_soccer_extension.storage.spaces_client import SpacesClient


def cmd_env_check(_: argparse.Namespace) -> int:
    config = SpacesConfig.from_env()
    print(f"bucket={config.bucket}")
    print(f"endpoint_url={config.endpoint_url}")
    print(f"region_name={config.region_name}")
    print(f"root_prefix={config.root_prefix}")
    print(f"profile_name={config.profile_name or ''}")
    print(f"has_do_spaces_key={bool(os.getenv('DO_SPACES_KEY'))}")
    print(f"has_do_spaces_secret={bool(os.getenv('DO_SPACES_SECRET'))}")
    return 0


def cmd_storage_plan(_: argparse.Namespace) -> int:
    config = SpacesConfig.from_env()
    paths = ObjectPathBuilder(config.root_prefix)
    now = datetime.now(timezone.utc)
    print(paths.key("raw", "football_data_uk"))
    print(paths.raw_key("api_football", "fixtures", now, "fixtures.json", league="eng.1", season="2025"))
    print(paths.gold_prematch_model_input_key("eng.1", "2025"))
    print(paths.manifest_key("feature_builds", "example-run"))
    print(paths.legacy_historical_training_prefix())
    return 0


def cmd_legacy_inventory(args: argparse.Namespace) -> int:
    config = SpacesConfig.from_env()
    paths = ObjectPathBuilder(config.root_prefix)
    client = SpacesClient(config)
    prefix = paths.legacy_historical_training_prefix()
    count = 0
    bytes_total = 0
    for item in client.list_objects(prefix):
        count += 1
        bytes_total += int(item.get("Size", 0))
        if count <= args.limit:
            print(f"{item.get('Size', 0)}\t{item['Key']}")
    print(f"objects={count}")
    print(f"bytes={bytes_total}")
    return 0


def _csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {part.strip() for part in value.split(",") if part.strip()}
    return parsed or None


def cmd_build_gold_from_legacy(args: argparse.Namespace) -> int:
    config = SpacesConfig.from_env()
    client = SpacesClient(config)
    paths = ObjectPathBuilder(config.root_prefix)
    result = build_gold_from_legacy_historical(
        client=client,
        path_builder=paths,
        run_id=args.run_id,
        leagues=_csv_set(args.leagues),
        seasons=_csv_set(args.seasons),
        limit=args.limit,
        workers=args.workers,
        skip_existing=not args.no_skip_existing,
    )
    print(f"run_id={result.run_id}")
    print(f"partitions={result.partitions}")
    print(f"rows={result.rows}")
    print(f"manifest_key={result.manifest_key}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betboard-soccer-extension")
    sub = parser.add_subparsers(dest="command", required=True)

    env_check = sub.add_parser("env-check")
    env_check.set_defaults(func=cmd_env_check)

    storage_plan = sub.add_parser("storage-plan")
    storage_plan.set_defaults(func=cmd_storage_plan)

    legacy_inventory = sub.add_parser("legacy-inventory")
    legacy_inventory.add_argument("--limit", type=int, default=25)
    legacy_inventory.set_defaults(func=cmd_legacy_inventory)

    build_gold = sub.add_parser("build-gold-from-legacy")
    build_gold.add_argument("--run-id", required=True)
    build_gold.add_argument("--leagues", default="")
    build_gold.add_argument("--seasons", default="")
    build_gold.add_argument("--limit", type=int)
    build_gold.add_argument("--workers", type=int, default=1)
    build_gold.add_argument("--no-skip-existing", action="store_true")
    build_gold.set_defaults(func=cmd_build_gold_from_legacy)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
