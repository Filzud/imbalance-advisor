"""Тесты для src/recommender.py.

Мета-модель и матрица признаков подменяются лёгкой синтетикой, чтобы тесты не
зависели от тяжёлых артефактов корпуса и считались быстро.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

import config
from src import recommender as rc


# --------------------------------------------------------------------------- #
# Загрузка датасета из CSV                                                     #
# --------------------------------------------------------------------------- #
def test_load_from_csv_default_last_column(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]}).to_csv(p, index=False)
    X, y, name = rc.load_dataset(str(p))
    assert list(X.columns) == ["a", "b"]
    assert y.tolist() == [0, 1]
    assert name == "d"


def test_load_from_csv_named_target(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"class": [0, 1], "a": [3, 4]}).to_csv(p, index=False)
    X, y, _ = rc.load_dataset(str(p))
    assert "class" not in X.columns and y.tolist() == [0, 1]


def test_load_missing_file_raises_friendly():
    with pytest.raises(rc.RecommenderError, match="не найден"):
        rc.load_dataset("no_such_file.csv")


def test_load_missing_target_column(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(rc.RecommenderError, match="не найден"):
        rc.load_dataset(str(p), target_column="nope")


# --------------------------------------------------------------------------- #
# Валидация краевых случаев                                                   #
# --------------------------------------------------------------------------- #
def test_validate_rejects_multiclass():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0, 1, 2, 3] * 25)
    with pytest.raises(rc.RecommenderError, match="не бинарный"):
        rc._validate(X, y)


def test_validate_rejects_single_class():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([1] * 100)
    with pytest.raises(rc.RecommenderError, match="нужна бинарная"):
        rc._validate(X, y)


def test_validate_rejects_too_small():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series([0] * 5 + [1] * 5)
    with pytest.raises(rc.RecommenderError, match="мало строк"):
        rc._validate(X, y)


def test_validate_accepts_good_binary():
    X = pd.DataFrame({"a": range(80)})
    y = pd.Series([0] * 60 + [1] * 20)
    rc._validate(X, y)  # не бросает


# --------------------------------------------------------------------------- #
# Фикстура лёгкой модели + подмена артефактов                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_model(tmp_path, monkeypatch):
    feature_names = ["imbalance_ratio", "n2", "l2"]
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for cls, c in zip(["smote", "baseline", "adasyn"], [0.0, 4.0, 8.0], strict=False):
        rows.append(pd.DataFrame({f: rng.normal(c, 0.4, 20) for f in feature_names}))
        labels += [cls] * 20
    Xtr = pd.concat(rows, ignore_index=True)
    ytr = pd.Series(labels)
    model = GradientBoostingClassifier(random_state=0).fit(Xtr.to_numpy(), ytr.to_numpy())

    bundle = {"model": model, "model_name": "gradient_boosting",
              "classes": list(np.unique(ytr)), "feature_names": feature_names}

    meta_path = tmp_path / "meta_features.csv"
    corpus = Xtr.copy()
    corpus.index.name = "did"
    corpus.to_csv(meta_path)
    monkeypatch.setattr(config, "META_FEATURES_PATH", meta_path)
    monkeypatch.setattr(config, "SHAP_VALUES_PATH", tmp_path / "shap.npz")
    return bundle, feature_names


# --------------------------------------------------------------------------- #
# Выравнивание вектора мета-признаков                                         #
# --------------------------------------------------------------------------- #
def test_compute_feature_vector_aligns_and_imputes(monkeypatch, toy_model):
    bundle, feature_names = toy_model
    # extract вернул лишние колонки и один NaN — должно выровняться к feature_names без NaN
    monkeypatch.setattr(
        rc, "extract_meta_features",
        lambda X, y: pd.Series({"imbalance_ratio": 5.0, "n2": np.nan,
                                "l2": 1.0, "extra_feat": 9.0}),
    )
    monkeypatch.setattr(rc, "preprocess_dataset", lambda X, y: (X, y))
    X = pd.DataFrame({"f": range(60)})
    y = pd.Series([0] * 40 + [1] * 20)
    vec = rc.compute_feature_vector(X, y, bundle)
    assert list(vec.index) == feature_names   # ровно 132-аналог: колонки модели
    assert not vec.isna().any()               # NaN импутирован медианой корпуса
    assert vec["imbalance_ratio"] == 5.0


# --------------------------------------------------------------------------- #
# Полный recommend на синтетике                                              #
# --------------------------------------------------------------------------- #
def test_recommend_end_to_end(monkeypatch, toy_model, tmp_path):
    bundle, feature_names = toy_model
    # датасет-«adasyn»: мета-признаки близко к центру 8.0
    monkeypatch.setattr(
        rc, "extract_meta_features",
        lambda X, y: pd.Series({f: 8.0 for f in feature_names}),
    )
    monkeypatch.setattr(rc, "preprocess_dataset", lambda X, y: (X, y))

    p = tmp_path / "ds.csv"
    df = pd.DataFrame({"a": range(60), "target": [0] * 40 + [1] * 20})
    df.to_csv(p, index=False)

    result = rc.recommend(str(p), bundle=bundle, explain=False)
    assert result["recommended_method"] == "adasyn"
    assert set(result["probabilities"]) == set(bundle["classes"])
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    assert result["explanation"] is None


def test_recommend_with_explanation_produces_text(monkeypatch, toy_model, tmp_path):
    bundle, feature_names = toy_model
    monkeypatch.setattr(
        rc, "extract_meta_features",
        lambda X, y: pd.Series({f: 0.0 for f in feature_names}),
    )
    monkeypatch.setattr(rc, "preprocess_dataset", lambda X, y: (X, y))
    p = tmp_path / "ds.csv"
    pd.DataFrame({"a": range(60), "target": [0] * 40 + [1] * 20}).to_csv(p, index=False)

    result = rc.recommend(str(p), bundle=bundle, explain=True)
    assert result["explanation"] is not None
    assert f"«{result['recommended_method']}»" in result["explanation"]["text"]


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def test_cli_reports_recommendation(monkeypatch, toy_model, tmp_path, capsys):
    bundle, feature_names = toy_model
    monkeypatch.setattr(rc, "load_model_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        rc, "extract_meta_features",
        lambda X, y: pd.Series({f: 8.0 for f in feature_names}),
    )
    monkeypatch.setattr(rc, "preprocess_dataset", lambda X, y: (X, y))
    p = tmp_path / "ds.csv"
    pd.DataFrame({"a": range(60), "target": [0] * 40 + [1] * 20}).to_csv(p, index=False)

    code = rc.main(["--csv", str(p), "--no-explain"])
    out = capsys.readouterr().out
    assert code == 0
    assert "РЕКОМЕНДОВАННЫЙ МЕТОД: adasyn" in out


def test_cli_friendly_error_on_multiclass(monkeypatch, tmp_path, capsys):
    p = tmp_path / "ds.csv"
    pd.DataFrame({"a": range(100), "target": [0, 1, 2, 3] * 25}).to_csv(p, index=False)
    code = rc.main(["--csv", str(p)])
    err = capsys.readouterr().err
    assert code == 2
    assert "не бинарный" in err
