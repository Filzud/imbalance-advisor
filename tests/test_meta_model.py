"""Тесты для src/meta_model.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import config
from src import meta_model as mm


# --------------------------------------------------------------------------- #
# Фабрика мета-моделей                                                        #
# --------------------------------------------------------------------------- #
def test_build_meta_model_types():
    assert isinstance(mm.build_meta_model("random_forest"), RandomForestClassifier)
    assert isinstance(mm.build_meta_model("gradient_boosting"), GradientBoostingClassifier)


def test_build_meta_model_unknown_raises():
    with pytest.raises(ValueError):
        mm.build_meta_model("xgboost")


def test_build_meta_model_passes_params():
    m = mm.build_meta_model("random_forest", {"n_estimators": 123})
    assert m.n_estimators == 123
    assert m.random_state == config.RANDOM_SEED


# --------------------------------------------------------------------------- #
# top-k accuracy                                                              #
# --------------------------------------------------------------------------- #
def test_top_k_accuracy_top1_and_top2():
    classes = np.array(["a", "b", "c"])
    # объект 1: истинный 'a', proba макс у 'a' → top-1 hit
    # объект 2: истинный 'c', proba: b>c>a → top-1 miss, top-2 hit
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    y_true = np.array(["a", "c"])
    assert mm.top_k_accuracy(y_true, proba, classes, k=1) == pytest.approx(0.5)
    assert mm.top_k_accuracy(y_true, proba, classes, k=2) == pytest.approx(1.0)


def test_top_k_accuracy_perfect():
    classes = np.array(["x", "y"])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array(["x", "y"])
    assert mm.top_k_accuracy(y_true, proba, classes, k=1) == 1.0


# --------------------------------------------------------------------------- #
# LODO-оценка на синтетике                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_meta_data():
    """Разделимая мета-выборка: 3 класса, признаки явно кодируют класс."""
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for cls, center in zip(["a", "b", "c"], [0.0, 5.0, 10.0], strict=False):
        n = 15
        frames.append(pd.DataFrame({
            "f1": rng.normal(center, 0.5, n),
            "f2": rng.normal(center, 0.5, n),
        }))
        labels += [cls] * n
    X = pd.concat(frames, ignore_index=True)
    X.index.name = "did"
    y = pd.Series(labels, name="best_method")
    return X, y


def test_evaluate_lodo_structure_and_high_accuracy(toy_meta_data):
    X, y = toy_meta_data
    res = mm.evaluate_lodo("random_forest", {"n_estimators": 50}, X, y)
    assert 0.0 <= res["top1"] <= 1.0
    assert res["topk"] >= res["top1"]
    # на разделимых данных top-1 должен быть высоким
    assert res["top1"] > 0.8
    # OOF: строка на объект, вероятности по всем классам + true/pred
    assert len(res["oof"]) == len(X)
    assert {"true_method", "pred_method"} <= set(res["oof"].columns)
    assert sum(c.startswith("proba_") for c in res["oof"].columns) == y.nunique()


def test_inner_cv_respects_rare_class():
    y = pd.Series(["a"] * 20 + ["b"] * 2)  # редкий класс = 2
    cv = mm._inner_cv(y)
    assert cv.get_n_splits() == 2  # не больше размера редкого класса


# --------------------------------------------------------------------------- #
# График важности                                                            #
# --------------------------------------------------------------------------- #
def test_plot_feature_importance_saves_file(toy_meta_data):
    X, y = toy_meta_data
    model = mm.build_meta_model("random_forest", {"n_estimators": 30})
    model.fit(X.to_numpy(), y.to_numpy())
    mm.plot_feature_importance(model, list(X.columns), top_n=2)
    assert (config.FIGURES_DIR / "meta_model_feature_importance.png").exists()


# --------------------------------------------------------------------------- #
# Кэш обученной модели                                                        #
# --------------------------------------------------------------------------- #
def test_train_meta_model_end_to_end(monkeypatch, tmp_path, toy_meta_data):
    """Оркестрация: HPO(1 trial) → LODO RF и GB → выбор → сохранение артефактов."""
    X, y = toy_meta_data
    monkeypatch.setattr(mm, "load_training_data", lambda: (X, y))
    monkeypatch.setattr(config, "META_MODEL_PATH", tmp_path / "meta_model.joblib")
    monkeypatch.setattr(config, "META_OOF_PATH", tmp_path / "oof.csv")
    monkeypatch.setattr(config, "META_MODEL_COMPARISON_PATH", tmp_path / "cmp.csv")

    out = mm.train_meta_model(n_trials=1, force=True)

    assert out["winner"] in config.META_MODELS
    assert out["cached"] is False
    assert set(out["results"]) == set(config.META_MODELS)
    # артефакты на месте
    assert config.META_MODEL_PATH.exists()
    assert config.META_OOF_PATH.exists()
    assert config.META_MODEL_COMPARISON_PATH.exists()
    assert (config.FIGURES_DIR / "meta_model_feature_importance.png").exists()
    # сохранённая модель — рабочий bundle
    import joblib
    bundle = joblib.load(config.META_MODEL_PATH)
    assert bundle["model_name"] == out["winner"]
    assert len(bundle["feature_names"]) == X.shape[1]


def test_train_meta_model_uses_cache(monkeypatch, tmp_path):
    import joblib
    path = tmp_path / "meta_model.joblib"
    joblib.dump({"model_name": "random_forest", "top1": 0.4, "topk": 0.7}, path)
    monkeypatch.setattr(config, "META_MODEL_PATH", path)

    def _fail(*a, **k):
        raise AssertionError("не должно обучаться при наличии кэша")

    monkeypatch.setattr(mm, "load_training_data", _fail)
    out = mm.train_meta_model(force=False)
    assert out["winner"] == "random_forest" and out["cached"] is True
