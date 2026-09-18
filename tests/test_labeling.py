"""Тесты для src/labeling.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import config
from src import labeling as lb


# --------------------------------------------------------------------------- #
# Фабрика классификатора                                                      #
# --------------------------------------------------------------------------- #
def test_build_base_classifier_default_is_decision_tree():
    clf = lb.build_base_classifier()
    assert isinstance(clf, DecisionTreeClassifier)
    assert clf.random_state == config.RANDOM_SEED


def test_build_base_classifier_class_weight_passed():
    clf = lb.build_base_classifier(class_weight="balanced")
    assert clf.class_weight == "balanced"


def test_build_base_classifier_switch_to_logreg():
    clf = lb.build_base_classifier("logistic_regression")
    assert isinstance(clf, LogisticRegression)


def test_build_base_classifier_unknown_raises():
    with pytest.raises(ValueError):
        lb.build_base_classifier("svm")


# --------------------------------------------------------------------------- #
# Сборка пайплайнов методов                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def y_imbalanced():
    return pd.Series([0] * 160 + [1] * 40)


def test_baseline_pipeline_is_bare_classifier(y_imbalanced):
    est = lb.build_pipeline("baseline", y_imbalanced, n_folds=5)
    assert isinstance(est, DecisionTreeClassifier)
    assert est.class_weight is None


def test_class_weight_pipeline_sets_balanced(y_imbalanced):
    est = lb.build_pipeline("class_weight", y_imbalanced, n_folds=5)
    assert est.class_weight == "balanced"


@pytest.mark.parametrize("method", ["random_undersample", "smote", "adasyn", "smote_enn"])
def test_sampler_pipelines_have_sampler_and_clf(method, y_imbalanced):
    est = lb.build_pipeline(method, y_imbalanced, n_folds=5)
    assert [name for name, _ in est.steps] == ["sampler", "clf"]


def test_build_pipeline_unknown_method_raises(y_imbalanced):
    with pytest.raises(ValueError):
        lb.build_pipeline("tomek", y_imbalanced, n_folds=5)


def test_safe_k_neighbors_shrinks_for_tiny_minority():
    # миноритарный класс = 10, 5 фолдов → train_minority≈8 → k = min(5, 7) = 5
    assert lb._safe_k_neighbors(pd.Series([0] * 90 + [1] * 10), n_folds=5) == 5
    # миноритарный = 5 → train≈4 → k = 3
    assert lb._safe_k_neighbors(pd.Series([0] * 95 + [1] * 5), n_folds=5) == 3
    # k не опускается ниже 1
    assert lb._safe_k_neighbors(pd.Series([0] * 98 + [1] * 2), n_folds=2) == 1


# --------------------------------------------------------------------------- #
# Оценка датасета и выбор победителя                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def separable_imbalanced():
    """Небольшой разделимый несбалансированный датасет — CV отрабатывает быстро."""
    rng = np.random.default_rng(0)
    n_maj, n_min = 160, 40
    X = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, n_maj), rng.normal(3, 1, n_min)]),
            "b": np.concatenate([rng.normal(0, 1, n_maj), rng.normal(3, 1, n_min)]),
        }
    )
    y = pd.Series([0] * n_maj + [1] * n_min)
    return X, y


def test_evaluate_dataset_all_methods_and_columns(separable_imbalanced):
    X, y = separable_imbalanced
    out = lb.evaluate_dataset(X, y)
    assert list(out["method"]) == list(config.BALANCING_METHODS)
    assert {"pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std"} <= set(out.columns)
    # на разделимых данных PR-AUC должен быть высоким хотя бы у одного метода
    assert out["pr_auc_mean"].max() > 0.7


def test_evaluate_dataset_raises_on_tiny_minority():
    X = pd.DataFrame({"a": np.arange(100.0)})
    y = pd.Series([0] * 99 + [1])  # миноритарный класс = 1 → CV невозможна
    with pytest.raises(ValueError):
        lb.evaluate_dataset(X, y)


def test_pick_best_method_prefers_higher_pr_auc():
    scores = pd.DataFrame(
        {
            "method": ["baseline", "smote", "adasyn"],
            "pr_auc_mean": [0.5, 0.8, 0.7],
            "f1_mean": [0.4, 0.6, 0.9],
        }
    )
    assert lb._pick_best_method(scores) == "smote"


def test_pick_best_method_tiebreak_by_f1_then_priority():
    scores = pd.DataFrame(
        {
            "method": ["baseline", "smote"],
            "pr_auc_mean": [0.8, 0.8],   # равны по PR-AUC
            "f1_mean": [0.6, 0.5],       # baseline лучше по F1
        }
    )
    assert lb._pick_best_method(scores) == "baseline"


# --------------------------------------------------------------------------- #
# Кэширование и график                                                        #
# --------------------------------------------------------------------------- #
def test_run_labeling_uses_cache(monkeypatch, tmp_path):
    scores_path = tmp_path / "labeling_scores.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"did": [1], "method": ["smote"], "pr_auc_mean": [0.9]}).to_csv(
        scores_path, index=False
    )
    pd.DataFrame({"did": [1], "name": ["x"], "best_method": ["smote"]}).to_csv(
        labels_path, index=False
    )
    monkeypatch.setattr(config, "LABELING_SCORES_PATH", scores_path)
    monkeypatch.setattr(config, "LABELS_PATH", labels_path)

    def _fail(*a, **k):
        raise AssertionError("не должно пересчитываться при наличии кэша")

    monkeypatch.setattr(lb, "evaluate_dataset", _fail)
    scores, labels = lb.run_labeling(force=False)
    assert labels.iloc[0]["best_method"] == "smote"


def test_run_labeling_end_to_end(monkeypatch, tmp_path):
    """Оркестрация: манифест → оценка (мок) → таблицы в data/ + график."""
    # манифест и обработанные датасеты в перенаправленных путях (см. conftest)
    manifest = pd.DataFrame({"did": [10, 20], "name": ["a", "b"]})
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.csv")
    manifest.to_csv(config.MANIFEST_PATH, index=False)
    for did in (10, 20):
        pd.DataFrame({"f": [0.0, 1.0, 2.0], "target": [0, 0, 1]}).to_csv(
            config.PROCESSED_DIR / f"{did}.csv", index=False
        )
    monkeypatch.setattr(config, "LABELING_SCORES_PATH", tmp_path / "scores.csv")
    monkeypatch.setattr(config, "LABELS_PATH", tmp_path / "labels.csv")

    def _fake_eval(X, y):
        return pd.DataFrame(
            {
                "method": list(config.BALANCING_METHODS),
                "pr_auc_mean": [0.5, 0.6, 0.4, 0.9, 0.3, 0.7],  # smote лучший
                "pr_auc_std": [0.0] * 6,
                "f1_mean": [0.5] * 6,
                "f1_std": [0.0] * 6,
            }
        )

    monkeypatch.setattr(lb, "evaluate_dataset", _fake_eval)
    scores, labels = lb.run_labeling(force=True)

    assert len(labels) == 2
    assert set(labels["best_method"]) == {"smote"}
    assert len(scores) == 2 * len(config.BALANCING_METHODS)
    assert config.LABELING_SCORES_PATH.exists() and config.LABELS_PATH.exists()
    assert (config.FIGURES_DIR / "labeling_best_method_distribution.png").exists()


def test_plot_best_method_distribution_saves_file():
    labels = pd.DataFrame(
        {"best_method": ["smote", "baseline", "smote", "smote_enn"]}
    )
    lb.plot_best_method_distribution(labels)
    assert (config.FIGURES_DIR / "labeling_best_method_distribution.png").exists()
