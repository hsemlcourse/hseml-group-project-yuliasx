"""
Юнит-тесты для пайплайна предсказания банкротства.
Запуск: pytest tests/ -v
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    encode_target,
    drop_duplicates,
    handle_missing,
    clip_outliers,
    split_data,
    RAW_FEATURES,
    TARGET_BIN,
    SEED,
)
from src.modeling import clf_metrics, find_best_threshold, train_logistic_baseline

# Тестовые данные
@pytest.fixture
def sample_df():
    
    np.random.seed(SEED)
    n = 500
    data = {col: np.random.randn(n) for col in RAW_FEATURES}
    data["company_name"] = [f"C_{i // 5}" for i in range(n)]
    data["year"] = np.random.randint(1999, 2019, n)
    data["status_label"] = np.random.choice(["alive", "failed"], n, p=[0.93, 0.07])
    return pd.DataFrame(data)


@pytest.fixture
def encoded_df(sample_df):
    return encode_target(sample_df)


@pytest.fixture
def clean_df(encoded_df):
    df = drop_duplicates(encoded_df)
    df = handle_missing(df)
    df = clip_outliers(df, RAW_FEATURES)
    return df


# Тесты предобработки (Preprocessing)
class TestEncodeTarget:
    def test_binary_column_created(self, encoded_df):
        assert TARGET_BIN in encoded_df.columns

    def test_values_are_binary(self, encoded_df):
        assert set(encoded_df[TARGET_BIN].unique()).issubset({0, 1})

    def test_failed_maps_to_one(self, sample_df):
        df = encode_target(sample_df)
        assert df.loc[sample_df["status_label"] == "failed", TARGET_BIN].all()

    def test_alive_maps_to_zero(self, sample_df):
        df = encode_target(sample_df)
        assert not df.loc[sample_df["status_label"] == "alive", TARGET_BIN].any()


class TestDropDuplicates:
    def test_no_duplicates_after_drop(self, encoded_df):
        df = drop_duplicates(encoded_df)
        assert df.duplicated().sum() == 0

    def test_returns_df(self, encoded_df):
        result = drop_duplicates(encoded_df)
        assert isinstance(result, pd.DataFrame)


class TestHandleMissing:
    def test_no_nans_after_handling(self, encoded_df):
        df = encoded_df.copy()
        # Искусственно добавляем пропуски
        df.loc[df.index[:10], "X1"] = np.nan
        result = handle_missing(df)
        assert result[RAW_FEATURES].isnull().sum().sum() == 0


class TestClipOutliers:
    def test_no_values_outside_bounds(self, encoded_df):
        df = clip_outliers(encoded_df, RAW_FEATURES, lower_q=0.05, upper_q=0.95)
        for col in RAW_FEATURES:
            lo = encoded_df[col].quantile(0.05)
            hi = encoded_df[col].quantile(0.95)
            assert df[col].min() >= lo - 1e-9
            assert df[col].max() <= hi + 1e-9

    def test_original_not_mutated(self, encoded_df):
        original_max = encoded_df["X1"].max()
        clip_outliers(encoded_df, ["X1"])
        assert encoded_df["X1"].max() == original_max


class TestSplitData:
    def test_correct_sizes(self, clean_df):
        X = clean_df[RAW_FEATURES]
        y = clean_df[TARGET_BIN]
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, val_size=0.15, test_size=0.15)
        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == len(X)

    def test_stratification(self, clean_df):
        X = clean_df[RAW_FEATURES]
        y = clean_df[TARGET_BIN]
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, stratify=True)
        # Доли банкротов должны быть примерно одинаковыми во всех выборках
        rates = [y_tr.mean(), y_v.mean(), y_te.mean()]
        assert max(rates) - min(rates) < 0.05

    def test_no_index_overlap_train_test(self, clean_df):
        X = clean_df[RAW_FEATURES]
        y = clean_df[TARGET_BIN]
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
        assert len(set(X_tr.index) & set(X_te.index)) == 0

    def test_reproducible_with_seed(self, clean_df):
        X = clean_df[RAW_FEATURES]
        y = clean_df[TARGET_BIN]
        splits1 = split_data(X, y, seed=SEED)
        splits2 = split_data(X, y, seed=SEED)
        pd.testing.assert_index_equal(splits1[0].index, splits2[0].index)


# Тесты моделирования (Modeling)
class TestClfMetrics:
    def test_keys_present(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.85])
        m = clf_metrics(y_true, y_pred, y_prob, prefix="test_")
        for key in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]:
            assert key in m

    def test_values_in_range(self):
        rng = np.random.default_rng(SEED)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.random(200)
        y_pred = (y_prob > 0.5).astype(int)
        m = clf_metrics(y_true, y_pred, y_prob)
        for v in m.values():
            assert 0.0 <= v <= 1.0


class TestFindBestThreshold:
    def test_returns_float(self):
        rng = np.random.default_rng(SEED)
        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        t = find_best_threshold(y, p)
        assert isinstance(t, float)

    def test_threshold_in_valid_range(self):
        rng = np.random.default_rng(SEED)
        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        t = find_best_threshold(y, p)
        assert 0.0 < t < 1.0

    def test_perfect_model_threshold_low(self):
        y = np.array([0] * 90 + [1] * 10)
        p = np.array([0.05] * 90 + [0.95] * 10)
        t = find_best_threshold(y, p)
        assert t < 0.7


class TestTrainLogisticBaseline:
    def test_pipeline_returned(self, clean_df):
        X = clean_df[RAW_FEATURES]
        y = clean_df[TARGET_BIN]
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y)
        
        res = train_logistic_baseline(X_tr, y_tr, X_v, y_v, X_te, y_te)
        
        assert "model" in res
        assert "test_roc_auc" in res
        assert "coef" in res
        assert hasattr(res["model"], "predict")