"""
Модуль 3. Разметка корпуса: определение лучшего метода борьбы с дисбалансом.

Для каждого датасета корпуса:
    * прогоняются ВСЕ методы балансировки из config.BALANCING_METHODS
      (baseline, class_weight, random_undersample, smote, adasyn, smote_enn);
    * качество каждого метода оценивается на стратифицированной k-fold CV
      честно, без утечки — ресемплинг применяется только к обучающим фолдам
      (через imblearn.Pipeline);
    * основная метрика — PR-AUC (average_precision), дополнительная — F1
      (при дисбалансе PR-AUC информативнее ROC-AUC, см. Главу 1 ВКР);
    * метка y датасета = метод с лучшим средним PR-AUC по фолдам.

Сохраняются два артефакта в data/:
    * labeling_scores.csv — полная таблица «датасет × метод × метрика»
      (нужна для Главы 3: анализ близости выбора к оптимуму, графики);
    * labels.csv — победитель на каждый датасет (целевая переменная мета-модели).

Этап дорогой, поэтому результат кэшируется: при повторном запуске без force
таблицы читаются из data/ без пересчёта.

Базовый классификатор берётся из config (BASE_CLASSIFIER) через фабрику, что
позволяет переключиться на logistic_regression для ablation Главы 3, не трогая код.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

import config
from src.utils import get_logger, save_figure, set_global_seed

logger = get_logger(__name__)

# Имена метрик sklearn → короткие имена колонок в таблице результатов.
_SCORING = {"pr_auc": config.SCORING_PRIMARY, "f1": config.SCORING_SECONDARY}


# --------------------------------------------------------------------------- #
# Фабрика базового классификатора (реестр из config)                          #
# --------------------------------------------------------------------------- #
def build_base_classifier(
    name: str | None = None, *, class_weight: str | None = None
) -> BaseEstimator:
    """
    Собрать базовый классификатор по имени из config.BASE_CLASSIFIERS.

    Реестр, а не хардкод: BASE_CLASSIFIER="logistic_regression" в config
    переключает весь этап разметки на логрегрессию (ablation Главы 3) без правок кода.
    Имя резолвится в момент ВЫЗОВА (не через дефолт-аргумент), чтобы временная
    смена config.BASE_CLASSIFIER во время ablation реально действовала.
    """
    name = name or config.BASE_CLASSIFIER
    if name not in config.BASE_CLASSIFIERS:
        raise ValueError(f"Неизвестный классификатор: {name}")
    params = dict(config.BASE_CLASSIFIERS[name])
    if name == "decision_tree":
        return DecisionTreeClassifier(
            **params, random_state=config.RANDOM_SEED, class_weight=class_weight
        )
    if name == "logistic_regression":
        return LogisticRegression(
            **params, random_state=config.RANDOM_SEED, class_weight=class_weight
        )
    raise ValueError(f"Неизвестный классификатор: {name}")


# --------------------------------------------------------------------------- #
# Сборка пайплайна «метод балансировки + классификатор»                        #
# --------------------------------------------------------------------------- #
def _safe_k_neighbors(y: pd.Series | np.ndarray, n_folds: int) -> int:
    """
    Консервативное число соседей для SMOTE/ADASYN.

    В обучающем фолде миноритарного класса примерно minority*(n_folds-1)/n_folds;
    соседей должно быть строго меньше этого числа. Возвращает k в диапазоне [1, 5].
    """
    minority = int(pd.Series(y).value_counts().min())
    train_minority = int(minority * (n_folds - 1) / n_folds)
    return max(1, min(5, train_minority - 1))


def build_pipeline(method: str, y: pd.Series | np.ndarray, n_folds: int) -> BaseEstimator:
    """
    Построить оценочный пайплайн для метода балансировки.

    baseline / class_weight обходятся без ресемплинга; остальные оборачивают
    сэмплер и классификатор в imblearn.Pipeline, где сэмплер применяется только
    к обучающей части каждого фолда (без утечки в валидацию).
    """
    seed = config.RANDOM_SEED
    if method == "baseline":
        return build_base_classifier()
    if method == "class_weight":
        return build_base_classifier(class_weight="balanced")

    k = _safe_k_neighbors(y, n_folds)
    clf = build_base_classifier()
    if method == "random_undersample":
        sampler = RandomUnderSampler(random_state=seed)
    elif method == "smote":
        sampler = SMOTE(k_neighbors=k, random_state=seed)
    elif method == "adasyn":
        sampler = ADASYN(n_neighbors=k, random_state=seed)
    elif method == "smote_enn":
        sampler = SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)
    else:
        raise ValueError(f"Неизвестный метод балансировки: {method}")

    return ImbPipeline([("sampler", sampler), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Оценка одного датасета по всем методам                                       #
# --------------------------------------------------------------------------- #
def evaluate_dataset(X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
    """
    Оценить все методы балансировки на одном датасете (стратифицированная CV).

    Returns
    -------
    DataFrame со строкой на метод и колонками:
        method, pr_auc_mean, pr_auc_std, f1_mean, f1_std.
    Метрика метода, упавшего на данных (например, SMOTE при слишком малом
    миноритарном классе), заполняется NaN — такой метод просто не выигрывает.
    """
    y = pd.Series(y).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)

    minority = int(y.value_counts().min())
    n_folds = min(config.CV_FOLDS, minority)
    if n_folds < 2:
        raise ValueError(f"Слишком малый миноритарный класс ({minority}) для CV.")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    rows: list[dict[str, float | str]] = []
    for method in config.BALANCING_METHODS:
        # Изоляция метода: некоторые сэмплеры (например, ADASYN на почти
        # разделимых данных) бросают жёсткую ошибку. Тогда метод получает NaN,
        # но датасет всё равно размечается остальными методами.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = build_pipeline(method, y, n_folds)
                scores = cross_validate(
                    estimator, X.to_numpy(), y.to_numpy(),
                    scoring=_SCORING, cv=cv, error_score=np.nan, n_jobs=-1,
                )
            metrics = {
                "pr_auc_mean": float(np.nanmean(scores["test_pr_auc"])),
                "pr_auc_std": float(np.nanstd(scores["test_pr_auc"])),
                "f1_mean": float(np.nanmean(scores["test_f1"])),
                "f1_std": float(np.nanstd(scores["test_f1"])),
            }
        except Exception as exc:  # noqa: BLE001 — метод неприменим к этим данным
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            logger.info("Метод %s неприменим к датасету: %s", method, reason)
            metrics = {k: float("nan") for k in
                       ("pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std")}
        rows.append({"method": method, **metrics})
    return pd.DataFrame(rows)


def _pick_best_method(scores: pd.DataFrame) -> str:
    """
    Выбрать метод-победитель: максимум среднего PR-AUC, тай-брейк — F1, затем
    порядок в config.BALANCING_METHODS (при равенстве предпочитаем более простой).
    """
    priority = {m: i for i, m in enumerate(config.BALANCING_METHODS)}
    ranked = scores.assign(_prio=scores["method"].map(priority)).sort_values(
        by=["pr_auc_mean", "f1_mean", "_prio"],
        ascending=[False, False, True],
    )
    return str(ranked.iloc[0]["method"])


# --------------------------------------------------------------------------- #
# Оркестрация разметки корпуса                                                 #
# --------------------------------------------------------------------------- #
def run_labeling(
    manifest: pd.DataFrame | None = None,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разметить весь корпус и сохранить таблицы результатов в data/.

    Returns
    -------
    (scores, labels)
        scores — полная таблица «датасет × метод × метрика»;
        labels — победитель на каждый датасет (did, name, best_method, ...).
    """
    set_global_seed()

    if config.LABELING_SCORES_PATH.exists() and config.LABELS_PATH.exists() and not force:
        logger.info("Разметка найдена в кэше: %s", config.LABELS_PATH)
        scores = pd.read_csv(config.LABELING_SCORES_PATH)
        labels = pd.read_csv(config.LABELS_PATH)
        _render_labeling_figure(labels)
        return scores, labels

    if manifest is None:
        manifest = pd.read_csv(config.MANIFEST_PATH)

    all_scores: list[pd.DataFrame] = []
    label_rows: list[dict] = []
    for _, row in manifest.iterrows():
        did, name = int(row["did"]), str(row.get("name", "?"))
        path = config.PROCESSED_DIR / f"{did}.csv"
        if not path.exists():
            logger.warning("Пропуск did=%s: нет файла %s", did, path)
            continue

        df = pd.read_csv(path)
        X, y = df.drop(columns="target"), df["target"]
        try:
            scores = evaluate_dataset(X, y)
        except Exception as exc:  # noqa: BLE001 — не роняем весь этап из-за одного датасета
            logger.warning("Пропуск did=%s (%s): ошибка оценки: %s", did, name, exc)
            continue

        scores.insert(0, "did", did)
        scores.insert(1, "name", name)
        all_scores.append(scores)

        best = _pick_best_method(scores)
        best_row = scores.loc[scores["method"] == best].iloc[0]
        label_rows.append(
            {
                "did": did, "name": name, "best_method": best,
                "best_pr_auc": round(float(best_row["pr_auc_mean"]), 4),
                "best_f1": round(float(best_row["f1_mean"]), 4),
            }
        )
        logger.info("Размечен did=%s (%s): лучший метод=%s PR-AUC=%.3f",
                    did, name, best, best_row["pr_auc_mean"])

    if not all_scores:
        raise RuntimeError("Не удалось разметить ни одного датасета.")

    scores_table = pd.concat(all_scores, ignore_index=True)
    labels = pd.DataFrame(label_rows)

    scores_table.to_csv(config.LABELING_SCORES_PATH, index=False)
    labels.to_csv(config.LABELS_PATH, index=False)
    logger.info("Таблица разметки сохранена: %s (%d строк)",
                config.LABELING_SCORES_PATH, len(scores_table))
    logger.info("Метки сохранены: %s (%d датасетов)", config.LABELS_PATH, len(labels))

    _render_labeling_figure(labels)
    return scores_table, labels


# --------------------------------------------------------------------------- #
# График распределения лучших методов (отдельный PNG)                          #
# --------------------------------------------------------------------------- #
def plot_best_method_distribution(labels: pd.DataFrame) -> None:
    """Сколько раз каждый метод оказался лучшим по корпусу → отдельный PNG."""
    order = list(config.BALANCING_METHODS)
    counts = labels["best_method"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(order, counts.to_numpy(), color="#3b6ea5", edgecolor="black")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Метод борьбы с дисбалансом")
    ax.set_ylabel("Число датасетов, где метод лучший")
    ax.set_title(f"Распределение лучших методов по корпусу (N={len(labels)})")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_figure(fig, "labeling_best_method_distribution")


def _render_labeling_figure(labels: pd.DataFrame) -> None:
    """Построить график распределения меток, если они есть."""
    if labels.empty:
        logger.warning("Меток нет — график распределения не строится.")
        return
    plot_best_method_distribution(labels)


if __name__ == "__main__":
    run_labeling(force=True)
