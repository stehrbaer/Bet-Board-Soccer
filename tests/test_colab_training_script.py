from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "train_soccer_three_way_nn_optuna.py"


def load_script_module():
    sys.modules.setdefault("optuna", types.SimpleNamespace(Trial=object))
    spec = importlib.util.spec_from_file_location("train_soccer_three_way_nn_optuna", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_league_aliases_resolve_to_gold_partition_ids() -> None:
    module = load_script_module()

    assert module.normalize_league("eng1") == "eng.1"
    assert module.normalize_league("eng_1") == "eng.1"
    assert module.normalize_league("EPL") == "eng.1"


def test_league_all_resolves_to_no_competition_filter() -> None:
    module = load_script_module()
    args = argparse.Namespace(league="all", competitions="")

    assert module.resolve_competitions(args) == []


def test_competitions_all_resolves_to_no_competition_filter() -> None:
    module = load_script_module()
    args = argparse.Namespace(league="eng1", competitions="all")

    assert module.resolve_competitions(args) == []
