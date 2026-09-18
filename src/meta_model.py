"""
Модуль 4. Обучение мета-классификатора.

По парам (мета-признаки датасета → лучший метод борьбы с дисбалансом) обучаются
две модели-кандидата — RandomForest и GradientBoosting, — сравниваются по честной
схеме leave-one-dataset-out (LODO) и выбирается лучшая.

Схема оценки:
    * LODO = LeaveOneOut по датасетам: обучаемся на 89 датасетах, предсказываем
      1 отложенный — честная схема для мета-обучения (нет утечки между датасетами);
    * метрики: top-1 accuracy и top-3 accuracy (часто 2–3 метода дают близкое
      качество, попадание в тройку уже полезно);
    * подбор гиперпараметров — Optuna; чтобы отбор гиперпараметров не «подглядывал»
      в LODO-оценку, целевая функция HPO считается на отдельной внутренней
      стратифицированной CV, а итоговые числа и OOF-прогнозы — строго по LODO.

Артефакты:
    * models/meta_model.joblib — обученная на всём корпусе модель-победитель;
    * data/meta_model_oof.csv — LODO out-of-fold прогнозы и вероятности по каждому
      датасету (нужны Главе 3: разрыв «рекомендация vs оптимум vs базлайны»);
    * data/meta_model_comparison.csv — сравнение RF и GB по LODO;
    * docs/figures/meta_model_feature_importance.png — важность мета-признаков.
"""

from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict

import config
from src.utils import get_logger, save_figure, set_global_seed

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Данные для обучения мета-модели                                             #
# --------------------------------------------------------------------------- #
def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Загрузить и выровнять матрицу мета-признаков (модуль 2) и метки (модуль 3).

    Returns
    -------
    (X, y)
        X — DataFrame (датасеты × мета-признаки), индекс — did;
        y — Series имён лучших методов, индекс — did.
    """
    X = pd.read_csv(config.META_FEATURES_PATH, index_col="did")
    labels = pd.read_csv(config.LABELS_PATH).set_index("did")
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common, "best_method"]
    logger.info("Обучающая выборка мета-модели: %d датасетов × %d мета-признаков, классов=%d",
                X.shape[0], X.shape[1], y.nunique())
    return X, y


# --------------------------------------------------------------------------- #
# Фабрика мета-моделей                                                        #
# --------------------------------------------------------------------------- #
def build_meta_model(name: str, params: dict | None = None) -> BaseEstimator:
    """Собрать мета-модель по имени (random_forest / gradient_boosting)."""
    params = dict(params or {})
    if name == "random_forest":
        return RandomForestClassifier(random_state=config.RANDOM_SEED, **params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=config.RANDOM_SEED, **params)
    raise ValueError(f"Неизвестная мета-модель: {name}")


# --------------------------------------------------------------------------- #
# Метрики top-k и LODO out-of-fold прогнозы                                    #
# --------------------------------------------------------------------------- #
def top_k_accuracy(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int
) -> float:
    """Доля объектов, у которых истинный класс попал в top-k по вероятности."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    top_idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    hits = [class_to_idx[t] in top_idx[i] for i, t in enumerate(y_true)]
    return float(np.mean(hits))


def _oof_proba(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv) -> np.ndarray:
    """Out-of-fold вероятности классов по заданной схеме CV (колонки — sorted classes)."""
    return cross_val_predict(
        model, X.to_numpy(), y.to_numpy(),
        cv=cv, method="predict_proba", n_jobs=-1,
    )


def _inner_cv(y: pd.Series) -> StratifiedKFold:
    """Внутренняя стратифицированная CV для HPO (число фолдов ≤ размера редкого класса)."""
    n_splits = min(3, int(y.value_counts().min()))
    return StratifiedKFold(n_splits=max(2, n_splits), shuffle=True,
                           random_state=config.RANDOM_SEED)


# --------------------------------------------------------------------------- #
# Подбор гиперпараметров (Optuna) — objective на внутренней CV                  #
# --------------------------------------------------------------------------- #
def _suggest_params(trial: optuna.Trial, name: str) -> dict:
    """Пространство гиперпараметров для RF / GB (умеренное, под 90 объектов)."""
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    # gradient_boosting
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int) -> dict:
    """
    Подобрать гиперпараметры модели по top-1 accuracy на внутренней CV (Optuna).

    Returns
    -------
    dict лучших гиперпараметров.
    """
    classes = np.unique(y.to_numpy())
    cv = _inner_cv(y)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, name)
        model = build_meta_model(name, params)
        proba = _oof_proba(model, X, y, cv)
        return top_k_accuracy(y.to_numpy(), proba, classes, k=1)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("HPO %s: лучший top-1 (внутр. CV)=%.3f, params=%s",
                name, study.best_value, study.best_params)
    return study.best_params


# --------------------------------------------------------------------------- #
# Честная оценка модели по LODO                                                #
# --------------------------------------------------------------------------- #
def evaluate_lodo(name: str, params: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Оценить модель строго по leave-one-dataset-out и собрать OOF-прогнозы.

    Returns
    -------
    dict: name, params, top1, top3, classes, oof (DataFrame с вероятностями,
    предсказанным и истинным методом на каждый датасет).
    """
    classes = np.unique(y.to_numpy())
    model = build_meta_model(name, params)
    proba = _oof_proba(model, X, y, LeaveOneOut())

    top1 = top_k_accuracy(y.to_numpy(), proba, classes, k=1)
    topk = top_k_accuracy(y.to_numpy(), proba, classes, k=config.METRIC_TOP_K)

    pred = classes[np.argmax(proba, axis=1)]
    oof = pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes], index=X.index)
    oof.insert(0, "true_method", y.to_numpy())
    oof.insert(1, "pred_method", pred)
    oof.index.name = "did"

    logger.info("LODO %s: top-1=%.3f, top-%d=%.3f", name, top1, config.METRIC_TOP_K, topk)
    return {"name": name, "params": params, "top1": top1, "topk": topk,
            "classes": classes, "oof": oof}


# --------------------------------------------------------------------------- #
# График важности мета-признаков                                              #
# --------------------------------------------------------------------------- #
def plot_feature_importance(
    model: BaseEstimator, feature_names: list[str], top_n: int = 20
) -> None:
    """Топ-N мета-признаков по важности модели-победителя → отдельный PNG."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(top))))
    ax.barh(top.index, top.to_numpy(), color="#3b6ea5", edgecolor="black")
    ax.set_xlabel("Важность признака (снижение неопределённости)")
    ax.set_title(f"Топ-{top_n} важных мета-признаков мета-модели")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, "meta_model_feature_importance")


# --------------------------------------------------------------------------- #
# Оркестрация: обучить, сравнить, выбрать, сохранить                           #
# --------------------------------------------------------------------------- #
def train_meta_model(
    *, n_trials: int = config.HPO_N_TRIALS, force: bool = False
) -> dict:
    """
    Обучить и сравнить RF и GB по LODO, выбрать лучшую, сохранить артефакты.

    Returns
    -------
    dict сводки: победитель, top-1/top-3 обеих моделей, путь к модели.
    """
    set_global_seed()

    if config.META_MODEL_PATH.exists() and not force:
        logger.info("Мета-модель найдена в кэше: %s", config.META_MODEL_PATH)
        bundle = joblib.load(config.META_MODEL_PATH)
        return {"winner": bundle["model_name"], "cached": True,
                "top1": bundle.get("top1"), "topk": bundle.get("topk")}

    X, y = load_training_data()

    results: list[dict] = []
    for name in config.META_MODELS:
        best_params = tune_model(name, X, y, n_trials=n_trials)
        results.append(evaluate_lodo(name, best_params, X, y))

    # победитель: по top-1, тай-брейк — top-k
    winner = max(results, key=lambda r: (r["top1"], r["topk"]))
    logger.info("Победила мета-модель: %s (top-1=%.3f, top-%d=%.3f)",
                winner["name"], winner["top1"], config.METRIC_TOP_K, winner["topk"])

    # сравнительная таблица RF vs GB
    comparison = pd.DataFrame(
        [{"model": r["name"], "top1": round(r["top1"], 4),
          f"top{config.METRIC_TOP_K}": round(r["topk"], 4),
          "is_winner": r["name"] == winner["name"]} for r in results]
    )
    comparison.to_csv(config.META_MODEL_COMPARISON_PATH, index=False)

    # OOF-прогнозы победителя (для Главы 3)
    winner["oof"].to_csv(config.META_OOF_PATH)
    logger.info("LODO OOF-прогнозы сохранены: %s", config.META_OOF_PATH)

    # финальная модель — переобучение на всём корпусе
    final_model = build_meta_model(winner["name"], winner["params"])
    final_model.fit(X.to_numpy(), y.to_numpy())

    joblib.dump(
        {
            "model": final_model,
            "model_name": winner["name"],
            "params": winner["params"],
            "classes": list(winner["classes"]),
            "feature_names": list(X.columns),
            "top1": winner["top1"],
            "topk": winner["topk"],
            "top_k": config.METRIC_TOP_K,
        },
        config.META_MODEL_PATH,
    )
    logger.info("Мета-модель сохранена: %s", config.META_MODEL_PATH)

    plot_feature_importance(final_model, list(X.columns))

    return {
        "winner": winner["name"],
        "results": {r["name"]: {"top1": r["top1"], "topk": r["topk"]} for r in results},
        "model_path": str(config.META_MODEL_PATH),
        "cached": False,
    }


if __name__ == "__main__":
    logging.getLogger("optuna").setLevel(logging.WARNING)
    summary = train_meta_model(force=True)
    print(summary)
