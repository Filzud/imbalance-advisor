"""Тесты для src/interpretation.py.

Используется маленькая синтетическая мультиклассовая модель, чтобы SHAP считался
быстро и без зависимости от реальных артефактов корпуса.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

import config
from src import interpretation as ip


# --------------------------------------------------------------------------- #
# Человекочитаемые описания                                                   #
# --------------------------------------------------------------------------- #
def test_humanize_known_and_suffix():
    assert ip._humanize("n2") == FEATURE_N2
    assert "разброс" in ip._humanize("n2.sd")
    assert ip._humanize("imbalance_ratio") == "степень дисбаланса классов (IR)"


FEATURE_N2 = "отношение внутри/меж-классовых расстояний (перекрытие, N2)"


def test_humanize_unknown_falls_back_to_name():
    assert ip._humanize("some_unknown_feat") == "some_unknown_feat"


# --------------------------------------------------------------------------- #
# Фикстура: маленькая обученная мультиклассовая модель + артефакты            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_bundle(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for cls, center in zip(["smote", "baseline", "adasyn"], [0.0, 4.0, 8.0], strict=False):
        n = 20
        frames.append(pd.DataFrame({
            "n2": rng.normal(center, 0.4, n),
            "l2": rng.normal(center, 0.4, n),
            "imbalance_ratio": rng.normal(center, 0.4, n),
        }))
        labels += [cls] * n
    X = pd.concat(frames, ignore_index=True)
    X.index = range(100, 100 + len(X))
    X.index.name = "did"
    y = pd.Series(labels)

    model = GradientBoostingClassifier(random_state=0).fit(X.to_numpy(), y.to_numpy())
    bundle = {
        "model": model, "model_name": "gradient_boosting",
        "classes": list(np.unique(y)), "feature_names": list(X.columns),
    }
    # перенаправляем пути и подменяем чтение матрицы признаков
    meta_path = tmp_path / "meta_features.csv"
    X.to_csv(meta_path)
    monkeypatch.setattr(config, "META_FEATURES_PATH", meta_path)
    monkeypatch.setattr(config, "SHAP_VALUES_PATH", tmp_path / "shap.npz")
    labels_df = pd.DataFrame({"did": X.index, "best_method": y.to_numpy()})
    labels_path = tmp_path / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    monkeypatch.setattr(config, "LABELS_PATH", labels_path)
    joblib.dump(bundle, tmp_path / "meta_model.joblib")
    monkeypatch.setattr(config, "META_MODEL_PATH", tmp_path / "meta_model.joblib")
    return bundle, X, y


# --------------------------------------------------------------------------- #
# Расчёт SHAP и кэш                                                           #
# --------------------------------------------------------------------------- #
def test_compute_shap_values_shape_and_cache(toy_bundle):
    bundle, X, y = toy_bundle
    expl = ip.compute_shap_values(bundle, force=True)
    assert expl.values.shape == (len(X), X.shape[1], y.nunique())
    assert config.SHAP_VALUES_PATH.exists()
    # повторный вызов читает кэш (форма сохраняется)
    expl2 = ip.compute_shap_values(bundle, force=False)
    assert expl2.values.shape == expl.values.shape


# --------------------------------------------------------------------------- #
# Глобальные и локальные графики                                             #
# --------------------------------------------------------------------------- #
def test_plot_global_importance_saves_file(toy_bundle):
    bundle, X, _ = toy_bundle
    expl = ip.compute_shap_values(bundle, force=True)
    ip.plot_global_importance(expl, X, list(bundle["classes"]))
    assert (config.FIGURES_DIR / "shap_global_importance.png").exists()


def test_plot_local_explanation_saves_file_and_returns_method(toy_bundle):
    bundle, X, _ = toy_bundle
    expl = ip.compute_shap_values(bundle, force=True)
    did = int(X.index[0])
    method = ip.plot_local_explanation(did, bundle, expl)
    assert method in bundle["classes"]
    figs = list(config.FIGURES_DIR.glob(f"shap_local_did{did}_*.png"))
    assert len(figs) == 1


# --------------------------------------------------------------------------- #
# Текстовое объяснение                                                        #
# --------------------------------------------------------------------------- #
def test_explain_recommendation_structure(toy_bundle):
    bundle, X, _ = toy_bundle
    expl = ip.compute_shap_values(bundle, force=True)
    did = int(X.index[0])
    info = ip.explain_recommendation(did, bundle, expl, top_n=2)
    assert info["did"] == did
    assert info["method"] in bundle["classes"]
    assert len(info["factors"]) == 2
    for f in info["factors"]:
        assert set(f) == {"feature", "human", "shap", "direction"}
        assert f["direction"] in ("повышает", "снижает")
    assert f"«{info['method']}»" in info["text"]


def test_build_interpretation_end_to_end(toy_bundle):
    bundle, X, _ = toy_bundle
    out = ip.build_interpretation(example_dids=[int(X.index[0])], force=True)
    assert out["n_datasets"] == len(X)
    assert len(out["examples"]) == 1
    assert (config.FIGURES_DIR / "shap_global_importance.png").exists()
