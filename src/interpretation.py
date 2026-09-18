"""
Модуль 5. Интерпретирующий слой SHAP.

Объясняет рекомендации мета-модели-победителя (models/meta_model.joblib):
    * глобальная важность мета-признаков (summary/beeswarm) → docs/figures/;
    * локальное объяснение конкретной рекомендации (waterfall для выбранного
      датасета) → docs/figures/;
    * функция explain_recommendation — топ-факторы выбора в человекочитаемом виде.

Победитель — мультиклассовый GradientBoosting, а shap.TreeExplainer поддерживает
GB только для бинарной задачи. Поэтому используется модель-агностик shap.Explainer
поверх predict_proba (Permutation): он корректно даёт вклад каждого мета-признака
ОТДЕЛЬНО по каждому из 6 классов-методов. Тензор значений имеет форму
(датасеты × признаки × классы); объяснение конкретной рекомендации берётся по
срезу предсказанного класса.
"""

from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from src.utils import get_logger, save_figure, set_global_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Человекочитаемые описания мета-признаков                                     #
# --------------------------------------------------------------------------- #
# Базовые имена pymfe/собственные → короткое пояснение на русском. Суффиксы
# .mean/.sd обрабатываются отдельно (см. _humanize).
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # собственные признаки дисбаланса
    "imbalance_ratio": "степень дисбаланса классов (IR)",
    "minority_fraction": "доля миноритарного класса",
    "class_entropy_norm": "сбалансированность классов (энтропия)",
    # меры сложности Ho-Basu — перекрытие/разделимость классов
    "n1": "доля пограничных объектов (перекрытие классов, N1)",
    "n2": "отношение внутри/меж-классовых расстояний (перекрытие, N2)",
    "n3": "ошибка ближайшего соседа (N3)",
    "n4": "нелинейность границы ближайшего соседа (N4)",
    "f1": "информативность лучшего признака (F1)",
    "f1v": "направленная разделимость по признакам (F1v)",
    "f2": "перекрытие областей значений признаков (F2)",
    "f3": "перекрытие по отдельному признаку (F3)",
    "f4": "совместное перекрытие признаков (F4)",
    "t1": "покрытие гиперсферами (T1)",
    "t2": "плотность объектов на признак (T2)",
    "t3": "размерность относительно объёма данных (T3)",
    "t4": "эффективная размерность (T4)",
    "l1": "ошибка линейного классификатора (L1)",
    "l2": "качество линейной разделимости (L2)",
    "l3": "нелинейность линейного классификатора (L3)",
    "c1": "энтропия распределения классов (C1)",
    "c2": "мера дисбаланса классов (C2)",
    "cls_coef": "кластерность классов (граф-связность)",
    "hubs": "хабовость объектов (сетевая мера)",
    "density": "плотность графа сходства",
    # landmarking — качество быстрых моделей-зондов
    "best_node": "качество лучшего одиночного разбиения (landmark)",
    "worst_node": "качество худшего одиночного разбиения (landmark)",
    "random_node": "качество случайного разбиения (landmark)",
    "elite_nn": "качество зонда «элитный ближайший сосед»",
    "naive_bayes": "качество наивного байесовского зонда",
    "linear_discr": "качество линейного зонда (LDA)",
    "one_nn": "качество зонда 1-NN",
    # информационные
    "class_ent": "энтропия классов",
    "attr_ent": "средняя энтропия признаков",
    "mut_inf": "взаимная информация признак–класс",
    "eq_num_attr": "эквивалентное число информативных признаков",
    "ns_ratio": "доля неинформативного шума в признаках",
    # model-based (по дереву решений)
    "nodes": "число узлов решающего дерева",
    "leaves": "число листьев решающего дерева",
    "tree_depth": "глубина решающего дерева",
    "nodes_per_inst": "узлов дерева на объект",
    "nodes_per_level": "узлов на уровень дерева",
    "leaves_per_class": "листьев на класс",
    "leaves_branch": "длина ветвей до листьев",
    "leaves_corrob": "подтверждаемость листьев (доля объектов)",
    "var_importance": "разброс важности признаков в дереве",
    # general
    "nr_inst": "число объектов",
    "nr_attr": "число признаков",
    "attr_to_inst": "отношение признаков к объектам",
    "inst_to_attr": "отношение объектов к признакам",
    "attr_conc": "сопряжённость пар признаков",
    "class_conc": "сопряжённость признаков с классом",
    # статистические
    "skewness": "асимметрия распределений признаков",
    "kurtosis": "эксцесс (тяжесть хвостов) признаков",
    "cor": "коррелированность признаков",
    "cov": "ковариация признаков",
    "sparsity": "разрежённость данных",
    "gravity": "расстояние между центрами классов",
    "sd_ratio": "отношение внутриклассовых разбросов",
    "g_mean": "геометрическое среднее признаков",
    "h_mean": "гармоническое среднее признаков",
    "iq_range": "межквартильный размах признаков",
    "mad": "медианное абсолютное отклонение",
    "var": "дисперсия признаков",
    "range": "размах значений признаков",
    "nr_cor_attr": "доля сильно коррелированных пар признаков",
    "eigenvalues": "спектр ковариационной матрицы",
    "w_lambda": "лямбда Уилкса (разделимость классов)",
    "can_cor": "канонические корреляции признак–класс",
    # model-based (дополнительно)
    "leaves_homo": "однородность листьев дерева",
}


def _humanize(feature: str) -> str:
    """Превратить имя мета-признака в человекочитаемое описание."""
    base, _, agg = feature.partition(".")
    desc = FEATURE_DESCRIPTIONS.get(base, base)
    if agg == "sd":
        return f"{desc} (разброс)"
    return desc


# --------------------------------------------------------------------------- #
# Загрузка модели и данных                                                    #
# --------------------------------------------------------------------------- #
def load_model_bundle(path=None) -> dict:
    """Загрузить bundle мета-модели (модель, классы, имена признаков, метрики)."""
    return joblib.load(path or config.META_MODEL_PATH)


def _load_features(bundle: dict) -> pd.DataFrame:
    """Матрица мета-признаков в порядке колонок, ожидаемом моделью."""
    X = pd.read_csv(config.META_FEATURES_PATH, index_col="did")
    return X[bundle["feature_names"]]


# --------------------------------------------------------------------------- #
# Расчёт SHAP-значений (модель-агностик, мультикласс) с кэшем                   #
# --------------------------------------------------------------------------- #
def build_explainer(model, X_background: pd.DataFrame) -> shap.Explainer:
    """SHAP-объяснитель поверх predict_proba (Permutation) с фоновой выборкой."""
    masker = shap.maskers.Independent(X_background, max_samples=config.SHAP_BACKGROUND_MAX)
    return shap.Explainer(
        model.predict_proba, masker, feature_names=list(X_background.columns)
    )


def compute_shap_values(
    bundle: dict | None = None, *, force: bool = False
) -> shap.Explanation:
    """
    Посчитать (или загрузить из кэша) SHAP-значения по всему корпусу.

    Returns
    -------
    shap.Explanation с values формы (датасеты × признаки × классы) и
    base_values формы (датасеты × классы).
    """
    set_global_seed()
    bundle = bundle or load_model_bundle()
    X = _load_features(bundle)

    if config.SHAP_VALUES_PATH.exists() and not force:
        logger.info("SHAP-значения найдены в кэше: %s", config.SHAP_VALUES_PATH)
        cached = np.load(config.SHAP_VALUES_PATH, allow_pickle=True)
        return shap.Explanation(
            values=cached["values"],
            base_values=cached["base_values"],
            data=X.to_numpy(),
            feature_names=list(X.columns),
        )

    logger.info("Расчёт SHAP-значений (модель-агностик, %d датасетов)…", len(X))
    explainer = build_explainer(bundle["model"], X)
    explanation = explainer(X)

    np.savez_compressed(
        config.SHAP_VALUES_PATH,
        values=explanation.values,
        base_values=explanation.base_values,
    )
    logger.info("SHAP-значения сохранены: %s (форма %s)",
                config.SHAP_VALUES_PATH, explanation.values.shape)
    # прикрепим индекс датасетов для локальных объяснений
    explanation.did_index = list(X.index)  # type: ignore[attr-defined]
    return explanation


# --------------------------------------------------------------------------- #
# Глобальная важность                                                          #
# --------------------------------------------------------------------------- #
def _values_per_class(explanation: shap.Explanation) -> list[np.ndarray]:
    """Разложить тензор (n, features, classes) в список массивов по классам."""
    values = explanation.values
    return [values[:, :, k] for k in range(values.shape[2])]


def plot_global_importance(
    explanation: shap.Explanation, X: pd.DataFrame, classes: list[str]
) -> None:
    """Глобальная важность мета-признаков по всем классам-методам → PNG (bar)."""
    shap.summary_plot(
        _values_per_class(explanation), X,
        plot_type="bar", class_names=classes,
        max_display=config.SHAP_MAX_DISPLAY, show=False,
    )
    fig = plt.gcf()
    fig.suptitle("Глобальная важность мета-признаков (SHAP)", y=1.02)
    save_figure(fig, "shap_global_importance")


def plot_global_beeswarm(
    explanation: shap.Explanation, X: pd.DataFrame, classes: list[str], method: str
) -> None:
    """Beeswarm вкладов мета-признаков для одного метода-класса → PNG."""
    cls_idx = classes.index(method)
    shap.summary_plot(
        explanation.values[:, :, cls_idx], X,
        max_display=config.SHAP_MAX_DISPLAY, show=False,
    )
    fig = plt.gcf()
    fig.suptitle(f"Вклад мета-признаков в выбор метода «{method}» (SHAP)", y=1.02)
    save_figure(fig, f"shap_global_beeswarm_{method}")


# --------------------------------------------------------------------------- #
# Локальное объяснение конкретной рекомендации                                 #
# --------------------------------------------------------------------------- #
def _predicted_class(bundle: dict, x_row: pd.Series) -> str:
    """Метод, рекомендованный моделью для одного датасета."""
    return str(bundle["model"].predict(x_row.to_numpy().reshape(1, -1))[0])

def plot_local_explanation(
    did: int, bundle: dict | None = None, explanation: shap.Explanation | None = None
) -> str:
    """
    Waterfall-объяснение рекомендации для одного датасета → PNG.

    Returns
    -------
    str — рекомендованный метод (для подписи/лога).
    """
    bundle = bundle or load_model_bundle()
    X = _load_features(bundle)
    explanation = explanation if explanation is not None else compute_shap_values(bundle)
    classes = list(bundle["classes"])

    pos = list(X.index).index(did)
    method = _predicted_class(bundle, X.iloc[pos])
    cls_idx = classes.index(method)

    single = shap.Explanation(
        values=explanation.values[pos, :, cls_idx],
        base_values=float(np.asarray(explanation.base_values)[pos, cls_idx]),
        data=X.iloc[pos].to_numpy(),
        feature_names=list(X.columns),
    )
    shap.plots.waterfall(single, max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Почему для датасета did={did} рекомендован «{method}»", y=1.02)
    save_figure(fig, f"shap_local_did{did}_{method}")
    return method


# --------------------------------------------------------------------------- #
# Человекочитаемые топ-факторы выбора                                          #
# --------------------------------------------------------------------------- #
def _top_factors(
    contrib: np.ndarray, feature_names: list[str], top_n: int
) -> list[dict]:
    """Топ-|вклад| мета-признаков: список {feature, human, shap, direction}."""
    order = np.argsort(np.abs(contrib))[::-1][:top_n]
    return [
        {
            "feature": feature_names[j],
            "human": _humanize(feature_names[j]),
            "shap": float(contrib[j]),
            "direction": "повышает" if contrib[j] > 0 else "снижает",
        }
        for j in order
    ]


def _reasons_phrase(factors: list[dict]) -> str:
    """Собрать перечисление факторов в человекочитаемую фразу."""
    return ", ".join(f"{f['human']} ({f['direction']} вероятность)" for f in factors)


def explain_recommendation(
    did: int,
    bundle: dict | None = None,
    explanation: shap.Explanation | None = None,
    top_n: int = config.SHAP_TOP_FACTORS,
) -> dict:
    """
    Топ-факторы рекомендации в человекочитаемом виде (для датасета корпуса).

    Returns
    -------
    dict:
        did, method — датасет и рекомендованный метод;
        factors — список {feature, human, shap, direction} по убыванию |вклад|;
        text — готовая фраза-объяснение для отчёта/ВКР.
    """
    bundle = bundle or load_model_bundle()
    X = _load_features(bundle)
    explanation = explanation if explanation is not None else compute_shap_values(bundle)
    classes = list(bundle["classes"])

    pos = list(X.index).index(did)
    method = _predicted_class(bundle, X.iloc[pos])
    contrib = explanation.values[pos, :, classes.index(method)]

    factors = _top_factors(contrib, list(X.columns), top_n)
    text = (f"Для датасета did={did} рекомендован метод «{method}»: "
            f"ключевые факторы — {_reasons_phrase(factors)}.")
    return {"did": did, "method": method, "factors": factors, "text": text}


def explain_instance(
    x_row: pd.Series,
    bundle: dict,
    background: pd.DataFrame,
    *,
    label: str = "instance",
    top_n: int = config.SHAP_TOP_FACTORS,
    save_plot: bool = True,
) -> dict:
    """
    Объяснить рекомендацию для ПРОИЗВОЛЬНОГО датасета (вне корпуса).

    Считает SHAP для одного вектора мета-признаков на фоне корпуса, строит
    waterfall предсказанного класса и человекочитаемую фразу.

    Parameters
    ----------
    x_row : pd.Series
        Вектор мета-признаков, уже выровненный к bundle["feature_names"].
    background : pd.DataFrame
        Фоновая выборка (мета-признаки корпуса) для masker.
    label : str
        Метка для имени файла графика (например, имя датасета/id).

    Returns
    -------
    dict: method, factors, text, waterfall_path (str | None).
    """
    classes = list(bundle["classes"])
    feature_names = list(bundle["feature_names"])
    x_row = x_row[feature_names]

    method = str(bundle["model"].predict(x_row.to_numpy().reshape(1, -1))[0])
    cls_idx = classes.index(method)

    explainer = build_explainer(bundle["model"], background)
    with warnings.catch_warnings():
        # модель обучена на numpy — глушим UserWarning про имена признаков от sklearn
        warnings.simplefilter("ignore")
        expl = explainer(x_row.to_frame().T)
    contrib = expl.values[0, :, cls_idx]

    factors = _top_factors(contrib, feature_names, top_n)
    text = (f"Рекомендован метод «{method}»: "
            f"ключевые факторы — {_reasons_phrase(factors)}.")

    waterfall_path = None
    if save_plot:
        single = shap.Explanation(
            values=contrib,
            base_values=float(np.asarray(expl.base_values)[0, cls_idx]),
            data=x_row.to_numpy(),
            feature_names=feature_names,
        )
        shap.plots.waterfall(single, max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle(f"Почему рекомендован «{method}» ({label})", y=1.02)
        waterfall_path = str(save_figure(fig, f"shap_recommend_{label}_{method}"))

    return {"method": method, "factors": factors, "text": text,
            "waterfall_path": waterfall_path}


# --------------------------------------------------------------------------- #
# Оркестрация: собрать все интерпретационные артефакты                         #
# --------------------------------------------------------------------------- #
def build_interpretation(
    example_dids: list[int] | None = None, *, force: bool = False
) -> dict:
    """
    Посчитать SHAP и собрать интерпретационные артефакты (графики + тексты).

    Parameters
    ----------
    example_dids : list[int] | None
        Датасеты для локальных объяснений (по умолчанию — первые два корпуса).
    """
    bundle = load_model_bundle()
    X = _load_features(bundle)
    classes = list(bundle["classes"])
    explanation = compute_shap_values(bundle, force=force)

    plot_global_importance(explanation, X, classes)
    # beeswarm для самого частого метода-победителя корпуса
    top_method = pd.read_csv(config.LABELS_PATH)["best_method"].value_counts().idxmax()
    plot_global_beeswarm(explanation, X, classes, top_method)

    if example_dids is None:
        example_dids = list(X.index[:2])

    explanations = []
    for did in example_dids:
        plot_local_explanation(int(did), bundle, explanation)
        info = explain_recommendation(int(did), bundle, explanation)
        explanations.append(info)
        logger.info(info["text"])

    return {"n_datasets": len(X), "global_method_beeswarm": top_method,
            "examples": explanations}


if __name__ == "__main__":
    summary = build_interpretation(force=True)
    for ex in summary["examples"]:
        print(ex["text"])
