"""Тесты для src/experiments.py."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src import experiments as ex


# --------------------------------------------------------------------------- #
# Эвристика по IR                                                             #
# --------------------------------------------------------------------------- #
def test_ir_heuristic_choice_bands():
    assert ex.ir_heuristic_choice(2.0) == "baseline"
    assert ex.ir_heuristic_choice(5.0) == "smote"
    assert ex.ir_heuristic_choice(20.0) == "smote_enn"


def test_rank_with_tail_covers_all_methods():
    rank = ex._rank_with_tail(["smote"])
    assert rank[0] == "smote"
    assert set(rank) == set(config.BALANCING_METHODS)
    assert len(rank) == len(set(rank))  # без повторов


# --------------------------------------------------------------------------- #
# Синтетические артефакты                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_artifacts():
    methods = list(config.BALANCING_METHODS)
    dids = [1, 2, 3, 4]
    # oof: мета-модель уверенно предсказывает proba
    oof_rows = []
    true = {1: "smote", 2: "baseline", 3: "smote_enn", 4: "adasyn"}
    pred = {1: "smote", 2: "baseline", 3: "smote", 4: "smote_enn"}  # 2 верных
    for did in dids:
        row = {"did": did, "true_method": true[did], "pred_method": pred[did]}
        for m in methods:
            row[f"proba_{m}"] = 0.9 if m == pred[did] else 0.02
        oof_rows.append(row)
    oof = pd.DataFrame(oof_rows)

    # pr_auc: у истинно лучшего метода максимум
    pr_rows = []
    for did in dids:
        for m in methods:
            val = 0.9 if m == true[did] else 0.6
            pr_rows.append({"did": did, "name": f"d{did}", "method": m,
                            "pr_auc_mean": val, "pr_auc_std": 0.0,
                            "f1_mean": val, "f1_std": 0.0})
    scores = pd.DataFrame(pr_rows)
    pr_auc = scores.pivot_table(index="did", columns="method", values="pr_auc_mean")[methods]

    manifest = pd.DataFrame({
        "did": dids, "imbalance_ratio": [2.0, 5.0, 20.0, 12.0],
        "ir_band": ["light", "medium", "heavy", "heavy"],
        "n_rows": [300, 500, 800, 400],
    }).set_index("did")
    return oof, pr_auc, manifest


# --------------------------------------------------------------------------- #
# Стратегии и метрики                                                         #
# --------------------------------------------------------------------------- #
def test_strategy_rankings_have_all_strategies(toy_artifacts):
    oof, _, manifest = toy_artifacts
    ranks = ex.strategy_rankings(oof, manifest)
    assert set(ranks) == {"meta_model", "random", "always_smote",
                          "always_baseline", "ir_heuristic"}
    # always_smote всегда ставит smote первым
    assert all(r[0] == "smote" for r in ranks["always_smote"])


def test_regret_zero_when_optimal_chosen(toy_artifacts):
    _, pr_auc, _ = toy_artifacts
    # did=1 оптимум smote (0.9); выбрать smote → regret 0
    assert ex._regret_for_method(1, "smote", pr_auc) == pytest.approx(0.0)
    # выбрать субоптимальный → regret 0.9-0.6=0.3
    assert ex._regret_for_method(1, "baseline", pr_auc) == pytest.approx(0.3)


def test_evaluate_strategies_meta_beats_random(toy_artifacts):
    oof, pr_auc, manifest = toy_artifacts
    summary = ex.evaluate_strategies(oof, pr_auc, manifest)
    assert set(summary["strategy"]) == {"meta_model", "random", "always_smote",
                                        "always_baseline", "ir_heuristic"}
    meta = summary.set_index("strategy").loc["meta_model"]
    rnd = summary.set_index("strategy").loc["random"]
    # мета-модель угадала 2/4 → top1=0.5
    assert meta["top1"] == pytest.approx(0.5)
    # у мета-модели regret не больше, чем у случайного выбора
    assert meta["mean_regret"] <= rnd["mean_regret"]


# --------------------------------------------------------------------------- #
# Графики                                                                     #
# --------------------------------------------------------------------------- #
def test_plots_save_files(toy_artifacts):
    oof, pr_auc, manifest = toy_artifacts
    summary = ex.evaluate_strategies(oof, pr_auc, manifest)
    ex.plot_topk_comparison(summary)
    ex.plot_regret_comparison(summary)
    ex.plot_confusion_matrix(oof)
    ex.plot_error_analysis_by_ir(oof, manifest)
    for fname in ["experiments_topk_comparison", "experiments_regret_comparison",
                  "confusion_matrix_meta", "experiments_error_analysis_by_ir"]:
        assert (config.FIGURES_DIR / f"{fname}.png").exists()


def test_confusion_matrix_is_square_over_methods(toy_artifacts):
    oof, _, _ = toy_artifacts
    cm = ex.plot_confusion_matrix(oof)
    assert list(cm.index) == list(config.BALANCING_METHODS)
    assert list(cm.columns) == list(config.BALANCING_METHODS)


# --------------------------------------------------------------------------- #
# Оркестрация (без ablation — дорого)                                         #
# --------------------------------------------------------------------------- #
def test_run_ablation_base_classifier(monkeypatch, tmp_path):
    """Ablation DT vs LR: считает согласованность выбора метода (сеть/labeling замоканы)."""
    from src import labeling as lb

    manifest = pd.DataFrame({"did": [1, 2]}).set_index("did")
    for did in (1, 2):
        pd.DataFrame({"f": [0.0, 1.0, 2.0], "target": [0, 0, 1]}).to_csv(
            config.PROCESSED_DIR / f"{did}.csv", index=False
        )
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"did": [1, 2], "best_method": ["smote", "baseline"]}).to_csv(
        labels_path, index=False
    )
    monkeypatch.setattr(config, "LABELS_PATH", labels_path)
    monkeypatch.setattr(config, "ABLATION_PATH", tmp_path / "ablation.csv")

    # LR-разметка: did=1 совпадает (smote), did=2 расходится (smote вместо baseline)
    calls = {"i": 0}

    def _fake_eval(X, y):
        return pd.DataFrame({"method": list(config.BALANCING_METHODS)})

    def _fake_pick(scores):
        calls["i"] += 1
        return "smote"

    monkeypatch.setattr(lb, "evaluate_dataset", _fake_eval)
    monkeypatch.setattr(lb, "_pick_best_method", _fake_pick)

    ablation = ex.run_ablation_base_classifier(manifest)
    assert list(ablation["did"]) == [1, 2]
    assert ablation.set_index("did").loc[1, "agree"]      # smote == smote
    assert not ablation.set_index("did").loc[2, "agree"]  # smote != baseline
    assert config.ABLATION_PATH.exists()
    # базовый классификатор восстановлен после ablation
    assert config.BASE_CLASSIFIER == "decision_tree"


def test_run_experiments_without_ablation(monkeypatch, tmp_path, toy_artifacts):
    oof, pr_auc, manifest = toy_artifacts
    # подменяем загрузку артефактов на синтетику
    monkeypatch.setattr(ex, "_load_artifacts", lambda: (oof, pr_auc, manifest))
    monkeypatch.setattr(config, "EXPERIMENTS_SUMMARY_PATH", tmp_path / "summary.csv")

    result = ex.run_experiments(with_ablation=False)
    assert result["ablation_agreement"] is None
    assert "meta_model" in set(result["summary"]["strategy"])
    assert config.EXPERIMENTS_SUMMARY_PATH.exists()
