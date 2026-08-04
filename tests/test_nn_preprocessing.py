from __future__ import annotations

import pandas as pd

from betboard_soccer_extension.modeling.nn_preprocessing import (
    choose_numeric_feature_columns,
    make_walkforward_folds,
    one_hot_target,
    prepare_fold,
)


def test_one_hot_target_home_draw_away_order() -> None:
    encoded = one_hot_target([0, 1, 2])
    assert encoded.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_choose_numeric_feature_columns_excludes_leakage() -> None:
    df = pd.DataFrame(
        {
            "match_id": ["a", "b", "c"],
            "safe_feature": [1.0, 2.0, 4.0],
            "home_score": [1, 0, 2],
            "result_target": [0, 1, 2],
            "another_safe_feature": [10.0, 9.0, 7.0],
        }
    )
    cols = choose_numeric_feature_columns(df, max_features=10)
    assert "safe_feature" in cols
    assert "another_safe_feature" in cols
    assert "home_score" not in cols
    assert "result_target" not in cols


def test_make_walkforward_folds() -> None:
    df = pd.DataFrame({"x": range(20)})
    folds = make_walkforward_folds(df, n_folds=2, val_size=4, min_train_size=8)
    assert len(folds) == 2
    assert len(folds[0][0]) == 8
    assert len(folds[0][1]) == 4


def test_prepare_fold_drops_all_missing_training_features() -> None:
    train_df = pd.DataFrame(
        {
            "safe_feature": [1.0, 2.0, 3.0],
            "missing_in_train": [None, None, None],
            "result_target": [0, 1, 2],
        }
    )
    val_df = pd.DataFrame(
        {
            "safe_feature": [4.0, 5.0],
            "missing_in_train": [10.0, 11.0],
            "result_target": [0, 2],
        }
    )

    fold = prepare_fold(train_df, val_df, ["safe_feature", "missing_in_train"])

    assert fold.feature_names == ["safe_feature"]
    assert fold.x_train.shape == (3, 1)
    assert fold.x_val.shape == (2, 1)
