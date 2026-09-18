"""
Модуль 7. Экспериментальная оценка для Главы 3 ВКР.

Сравнивает рекомендации мета-модели со стратегиями-базлайнами и количественно
показывает, насколько выбор близок к оптимуму. Использует честные LODO
out-of-fold прогнозы (meta_model_oof.csv) и полную таблицу качества методов
(labeling_scores.csv), поэтому все числа воспроизводимы и без утечки.

Что считается:
    1. Стратегии выбора метода: мета-модель и базлайны — случайный выбор,
       «всегда SMOTE», «всегда baseline», эвристика по IR.
    2. Метрики выбора: top-1 и top-3 accuracy.
    3. Главный результат — regret (разрыв до оптимума): PR-AUC рекомендованного
       метода минус PR-AUC реально лучшего метода на этом датасете. Малый regret
       при неидеальном top-1 означает, что рекомендованный метод почти так же
       хорош, как оптимум.
    4. Confusion matrix мета-модели (какие методы путаются).
    5. Анализ ошибок по диапазонам IR.
    6. Ablation: decision_tree vs logistic_regression как базовый классификатор.

Все графики сохраняются отдельными файлами в docs/figures/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import config
from src.utils import get_logger, save_figure, set_global_seed

logger = get_logger(__name__)

METHODS: tuple[str, ...] = config.BALANCING_METHODS


# --------------------------------------------------------------------------- #
# Загрузка артефактов предыдущих модулей                                       #
# --------------------------------------------------------------------------- #
def _load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Вернуть (oof, матрица PR-AUC [did×method], manifest с IR/размерами)."""
    oof = pd.read_csv(config.META_OOF_PATH)
    scores = pd.read_csv(config.LABELING_SCORES_PATH)
    manifest = pd.read_csv(config.MANIFEST_PATH).set_index("did")

    pr_auc = scores.pivot_table(index="did", columns="method", values="pr_auc_mean")
    pr_auc = pr_auc.reindex(columns=list(METHODS))
    return oof, pr_auc, manifest


# --------------------------------------------------------------------------- #
# Стратегии выбора метода: каждая возвращает РАНЖИРОВАНИЕ методов на датасет    #
# --------------------------------------------------------------------------- #
def _rank_with_tail(head: list[str]) -> list[str]:
    """Дополнить список методов-приоритетов остальными в фиксированном порядке."""
    tail = [m for m in METHODS if m not in head]
    return head + tail


def ir_heuristic_choice(ir: float) -> str:
    """
    Эвристика выбора метода по Imbalance Ratio.

    Правило: лёгкий дисбаланс — ничего не менять (baseline); умеренный — SMOTE;
    сильный — гибрид SMOTE+ENN.
    """
    if ir < 3.0:
        return "baseline"
    if ir < 9.0:
        return "smote"
    return "smote_enn"


def strategy_rankings(oof: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Построить для каждой стратегии ранжирование методов по каждому датасету.

    Returns
    -------
    dict: имя стратегии → Series(did → list[str] ранжированных методов).
    """
    dids = oof["did"].to_numpy()
    proba_cols = [f"proba_{m}" for m in METHODS]

    # мета-модель: ранжирование по убыванию вероятности
    meta_rank = {}
    for _, row in oof.iterrows():
        probs = row[proba_cols].to_numpy(dtype=float)
        order = np.argsort(probs)[::-1]
        meta_rank[int(row["did"])] = [METHODS[i] for i in order]

    rankings: dict[str, dict[int, list[str]]] = {
        "meta_model": meta_rank,
        "random": {},
        "always_smote": {},
        "always_baseline": {},
        "ir_heuristic": {},
    }
    for did in dids:
        did = int(did)
        rng = np.random.default_rng(config.RANDOM_SEED + did)  # детерминированно
        rankings["random"][did] = list(rng.permutation(METHODS))
        rankings["always_smote"][did] = _rank_with_tail(["smote"])
        rankings["always_baseline"][did] = _rank_with_tail(["baseline"])
        ir = float(manifest.loc[did, "imbalance_ratio"])
        rankings["ir_heuristic"][did] = _rank_with_tail([ir_heuristic_choice(ir)])

    return {name: pd.Series(rank) for name, rank in rankings.items()}


# --------------------------------------------------------------------------- #
# Метрики выбора: top-1, top-3, regret                                         #
# --------------------------------------------------------------------------- #
def _regret_for_method(did: int, method: str, pr_auc: pd.DataFrame) -> float:
    """Разрыв до оптимума на датасете: max PR-AUC − PR-AUC выбранного метода."""
    row = pr_auc.loc[did]
    optimum = float(row.max())
    chosen = row.get(method, np.nan)
    if pd.isna(chosen):
        chosen = float(row.min())  # неприменимый метод — берём худший как консервативную оценку
    return optimum - float(chosen)


def evaluate_strategies(
    oof: pd.DataFrame, pr_auc: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """
    Посчитать top-1, top-3 и средний regret для каждой стратегии.

    Returns
    -------
    DataFrame: строка на стратегию, колонки top1, top3, mean_regret,
    mean_pr_auc (достигнутое), optimum_pr_auc.
    """
    true_best = dict(zip(oof["did"].astype(int), oof["true_method"], strict=False))
    rankings = strategy_rankings(oof, manifest)
    optimum_mean = float(pr_auc.max(axis=1).mean())

    rows = []
    for name, rank_series in rankings.items():
        top1 = top3 = 0
        regrets, achieved = [], []
        for did, ranked in rank_series.items():
            best = true_best[did]
            if ranked[0] == best:
                top1 += 1
            if best in ranked[:config.METRIC_TOP_K]:
                top3 += 1
            regrets.append(_regret_for_method(did, ranked[0], pr_auc))
            chosen = pr_auc.loc[did].get(ranked[0], np.nan)
            achieved.append(float(chosen) if not pd.isna(chosen) else float(pr_auc.loc[did].min()))
        n = len(rank_series)
        rows.append({
            "strategy": name,
            "top1": round(top1 / n, 4),
            f"top{config.METRIC_TOP_K}": round(top3 / n, 4),
            "mean_regret": round(float(np.mean(regrets)), 4),
            "mean_pr_auc": round(float(np.mean(achieved)), 4),
            "optimum_pr_auc": round(optimum_mean, 4),
        })

    df = pd.DataFrame(rows).sort_values("mean_regret").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Графики сравнения                                                            #
# --------------------------------------------------------------------------- #
_STRATEGY_LABELS = {
    "meta_model": "Мета-модель",
    "ir_heuristic": "Эвристика по IR",
    "always_smote": "Всегда SMOTE",
    "always_baseline": "Всегда baseline",
    "random": "Случайный выбор",
}


def plot_topk_comparison(summary: pd.DataFrame) -> None:
    """Групповой barchart top-1 и top-3 для всех стратегий → PNG."""
    order = summary.sort_values("top1", ascending=False)
    labels = [_STRATEGY_LABELS.get(s, s) for s in order["strategy"]]
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, order["top1"], w, label="top-1", color="#3b6ea5", edgecolor="black")
    ax.bar(x + w / 2, order[f"top{config.METRIC_TOP_K}"], w,
           label=f"top-{config.METRIC_TOP_K}", color="#78c2ad", edgecolor="black")
    ax.axhline(1 / len(METHODS), ls="--", c="gray", lw=1, label="случайный top-1 (1/6)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Точность выбора метода: мета-модель vs базлайны")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "experiments_topk_comparison")


def plot_regret_comparison(summary: pd.DataFrame) -> None:
    """Barchart среднего regret (разрыв до оптимума) → PNG. Меньше — лучше."""
    order = summary.sort_values("mean_regret")
    labels = [_STRATEGY_LABELS.get(s, s) for s in order["strategy"]]
    colors = ["#2e7d32" if s == "meta_model" else "#3b6ea5" for s in order["strategy"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, order["mean_regret"], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Средний regret (PR-AUC до оптимума)")
    ax.set_title("Разрыв до оптимума: мета-модель vs базлайны (меньше — лучше)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "experiments_regret_comparison")


def plot_confusion_matrix(oof: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix мета-модели (истинный vs предсказанный метод) → PNG."""
    cm = pd.crosstab(oof["true_method"], oof["pred_method"]).reindex(
        index=list(METHODS), columns=list(METHODS), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right")
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.set_xlabel("Предсказанный метод")
    ax.set_ylabel("Истинный (лучший) метод")
    ax.set_title("Confusion matrix мета-модели (LODO)")
    thr = cm.to_numpy().max() / 2
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            v = int(cm.iat[i, j])
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > thr else "black")
    fig.colorbar(im, ax=ax, label="Число датасетов")
    save_figure(fig, "confusion_matrix_meta")
    return cm


def plot_error_analysis_by_ir(oof: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Точность и средний regret мета-модели по диапазонам IR → PNG."""
    df = oof[["did", "true_method", "pred_method"]].copy()
    df["correct"] = df["true_method"] == df["pred_method"]
    df["ir_band"] = df["did"].map(manifest["ir_band"])

    order = [b[0] for b in config.IR_BANDS]
    acc = df.groupby("ir_band")["correct"].mean().reindex(order)
    counts = df.groupby("ir_band")["correct"].count().reindex(order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(order, acc.to_numpy(), color=["#78c2ad", "#f3969a", "#c0392b"],
                  edgecolor="black")
    ax.bar_label(bars, labels=[f"{a:.0%}\n(n={int(c)})" for a, c in zip(acc, counts, strict=False)],
                 padding=3)
    ax.axhline(1 / len(METHODS), ls="--", c="gray", lw=1, label="случайный top-1")
    ax.set_ylim(0, 1)
    ax.set_ylabel("top-1 accuracy мета-модели")
    ax.set_xlabel("Диапазон Imbalance Ratio")
    ax.set_title("Анализ ошибок: точность мета-модели по характеру дисбаланса")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "experiments_error_analysis_by_ir")
    return acc.to_frame("top1_accuracy").assign(n=counts)


# --------------------------------------------------------------------------- #
# Ablation: базовый классификатор decision_tree vs logistic_regression         #
# --------------------------------------------------------------------------- #
def run_ablation_base_classifier(manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Пересчитать разметку с logistic_regression и сравнить с decision_tree.

    Оценивает устойчивость выбора метода к смене базового классификатора:
    какая доля датасетов сохраняет тот же лучший метод. Опирается на
    labeling.evaluate_dataset, временно переключая config.BASE_CLASSIFIER.

    Returns
    -------
    DataFrame: did, best_dt, best_lr, agree.
    """
    from src import labeling as lb

    if manifest is None:
        manifest = pd.read_csv(config.MANIFEST_PATH).set_index("did")
    dt_labels = pd.read_csv(config.LABELS_PATH).set_index("did")["best_method"]

    original = config.BASE_CLASSIFIER
    rows = []
    try:
        config.BASE_CLASSIFIER = "logistic_regression"
        for did in manifest.index:
            path = config.PROCESSED_DIR / f"{did}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            try:
                scores = lb.evaluate_dataset(df.drop(columns="target"), df["target"])
                best_lr = lb._pick_best_method(scores)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ablation did=%s: пропуск (%s)", did, exc)
                continue
            best_dt = dt_labels.get(did)
            rows.append({"did": int(did), "best_dt": best_dt, "best_lr": best_lr,
                         "agree": best_dt == best_lr})
    finally:
        config.BASE_CLASSIFIER = original

    ablation = pd.DataFrame(rows)
    ablation.to_csv(config.ABLATION_PATH, index=False)
    agreement = float(ablation["agree"].mean()) if len(ablation) else float("nan")
    logger.info("Ablation DT vs LR: согласованность выбора метода = %.1f%% (%d датасетов)",
                agreement * 100, len(ablation))
    return ablation


# --------------------------------------------------------------------------- #
# Оркестрация                                                                  #
# --------------------------------------------------------------------------- #
def run_experiments(*, with_ablation: bool = True) -> dict:
    """
    Посчитать всю фактуру Главы 3: таблицу сравнения, графики, ablation.

    Returns
    -------
    dict: summary (DataFrame), confusion (DataFrame), ablation_agreement (float|None).
    """
    set_global_seed()
    oof, pr_auc, manifest = _load_artifacts()

    summary = evaluate_strategies(oof, pr_auc, manifest)
    summary.to_csv(config.EXPERIMENTS_SUMMARY_PATH, index=False)
    logger.info("Сводка экспериментов сохранена: %s", config.EXPERIMENTS_SUMMARY_PATH)

    plot_topk_comparison(summary)
    plot_regret_comparison(summary)
    cm = plot_confusion_matrix(oof)
    plot_error_analysis_by_ir(oof, manifest)

    ablation_agreement = None
    if with_ablation:
        ablation = run_ablation_base_classifier(manifest)
        ablation_agreement = float(ablation["agree"].mean()) if len(ablation) else None

    return {"summary": summary, "confusion": cm, "ablation_agreement": ablation_agreement}


if __name__ == "__main__":
    result = run_experiments(with_ablation=True)
    print(result["summary"].to_string(index=False))
