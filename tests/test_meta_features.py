"""Тесты для src/meta_features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import meta_features as mf


# --------------------------------------------------------------------------- #
# Собственные признаки дисбаланса                                             #
# --------------------------------------------------------------------------- #
def test_custom_imbalance_features_values():
    y = pd.Series([0] * 80 + [1] * 20)
    feats = mf.custom_imbalance_features(y)
    assert feats["imbalance_ratio"] == pytest.approx(4.0)
    assert feats["minority_fraction"] == pytest.approx(0.2)
    assert 0.0 < feats["class_entropy_norm"] < 1.0


def test_custom_imbalance_features_balanced_has_max_entropy():
    y = pd.Series([0] * 50 + [1] * 50)
    feats = mf.custom_imbalance_features(y)
    assert feats["imbalance_ratio"] == pytest.approx(1.0)
    assert feats["class_entropy_norm"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Ограничение размера данных перед расчётом                                   #
# --------------------------------------------------------------------------- #
def test_prepare_subsamples_rows_and_keeps_ir():
    rng = np.random.default_rng(0)
    n = config.MF_MAX_ROWS * 3
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series([0] * int(n * 0.9) + [1] * (n - int(n * 0.9)))
    X2, y2 = mf._prepare_for_extraction(X, y)
    assert len(X2) == config.MF_MAX_ROWS
    # стратификация сохранила оба класса
    assert set(y2.unique()) == {0, 1}


def test_prepare_caps_wide_features_by_variance():
    n = 100
    cols = {f"f{i}": np.full(n, 0.0) for i in range(config.MF_MAX_FEATURES + 20)}
    # первые MF_MAX_FEATURES столбцов делаем «дисперсными», остальные — константы
    rng = np.random.default_rng(1)
    for i in range(config.MF_MAX_FEATURES):
        cols[f"f{i}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    y = pd.Series([0] * 60 + [1] * 40)
    X2, _ = mf._prepare_for_extraction(X, y)
    assert X2.shape[1] == config.MF_MAX_FEATURES
    # отобраны именно дисперсные признаки (константные выброшены)
    assert all(X2[c].var() > 0 for c in X2.columns)


# --------------------------------------------------------------------------- #
# Извлечение вектора мета-признаков                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_imbalanced():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pd.Series([0] * 160 + [1] * 40)
    return X, y


def test_extract_meta_features_fixed_and_contains_custom(small_imbalanced):
    X, y = small_imbalanced
    s = mf.extract_meta_features(X, y)
    assert isinstance(s, pd.Series)
    assert len(s) > 50  # pymfe даёт десятки признаков
    for key in ("imbalance_ratio", "minority_fraction", "class_entropy_norm"):
        assert key in s.index
    assert s["imbalance_ratio"] == pytest.approx(4.0)


def test_extract_is_deterministic(small_imbalanced):
    X, y = small_imbalanced
    s1 = mf.extract_meta_features(X, y)
    s2 = mf.extract_meta_features(X, y)
    pd.testing.assert_series_equal(s1, s2)


# --------------------------------------------------------------------------- #
# Чистка матрицы: фиксированная размерность, без NaN                          #
# --------------------------------------------------------------------------- #
def test_clean_matrix_drops_high_nan_and_imputes():
    raw = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, 4.0],
            "one_nan": [1.0, np.nan, 3.0, 4.0],       # 25% NaN — остаётся, импутируется
            "mostly_nan": [1.0, np.nan, np.nan, np.nan],  # 75% NaN — выбрасывается
            "has_inf": [1.0, np.inf, 2.0, 3.0],       # inf → NaN (25%) — остаётся
        }
    )
    clean = mf._clean_matrix(raw)
    assert "mostly_nan" not in clean.columns
    assert {"good", "one_nan", "has_inf"} <= set(clean.columns)
    assert int(clean.isna().sum().sum()) == 0
    # импутация медианой: пропуск в one_nan → median(1,3,4)=3.0
    assert clean.loc[1, "one_nan"] == pytest.approx(3.0)


def test_build_meta_feature_matrix_uses_cache(monkeypatch, tmp_path):
    cached = pd.DataFrame(
        {"did": [1, 2], "f1": [0.1, 0.2], "f2": [1.0, 2.0]}
    ).set_index("did")
    path = tmp_path / "meta_features.csv"
    cached.to_csv(path)
    monkeypatch.setattr(config, "META_FEATURES_PATH", path)

    def _fail(*a, **k):
        raise AssertionError("не должно считаться при наличии кэша")

    monkeypatch.setattr(mf, "extract_meta_features", _fail)
    out = mf.build_meta_feature_matrix(force=False)
    assert list(out.columns) == ["f1", "f2"]
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# График корреляций                                                           #
# --------------------------------------------------------------------------- #
def test_build_meta_feature_matrix_end_to_end(monkeypatch, tmp_path):
    """Оркестрация: манифест → извлечение (мок) → чистка → матрица + корр. карта."""
    manifest = pd.DataFrame({"did": [10, 20], "name": ["a", "b"]})
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.csv")
    manifest.to_csv(config.MANIFEST_PATH, index=False)
    for did in (10, 20):
        pd.DataFrame({"f": [0.0, 1.0, 2.0], "target": [0, 0, 1]}).to_csv(
            config.PROCESSED_DIR / f"{did}.csv", index=False
        )
    monkeypatch.setattr(config, "META_FEATURES_PATH", tmp_path / "meta_features.csv")

    def _fake_extract(X, y):
        return pd.Series({"m1": 1.0, "m2": 2.0, "m3": 3.0})

    monkeypatch.setattr(mf, "extract_meta_features", _fake_extract)
    matrix = mf.build_meta_feature_matrix(force=True)

    assert matrix.shape == (2, 3)
    assert int(matrix.isna().sum().sum()) == 0
    assert config.META_FEATURES_PATH.exists()
    assert (config.FIGURES_DIR / "meta_features_correlation.png").exists()


def test_plot_correlation_saves_file():
    rng = np.random.default_rng(4)
    matrix = pd.DataFrame(rng.normal(size=(10, 6)),
                          columns=[f"m{i}" for i in range(6)])
    mf.plot_meta_feature_correlation(matrix)
    assert (config.FIGURES_DIR / "meta_features_correlation.png").exists()
