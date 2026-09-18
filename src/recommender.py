"""
Модуль 6. Финальный инструмент рекомендации метода борьбы с дисбалансом.

Главная функция recommend(source) принимает ПРОИЗВОЛЬНЫЙ бинарный датасет
(CSV-файл или id датасета OpenML), считает по нему мета-признаки (модуль 2),
приводит их к тем же 132 колонкам, что видела мета-модель, предсказывает лучший
метод и объясняет выбор через SHAP (модуль 5).

Возвращает словарь:
    * recommended_method — рекомендованный метод;
    * probabilities — вероятности по всем 6 методам;
    * explanation — {text: человекочитаемая фраза, waterfall_path: путь к графику}.

Краевые случаи (не бинарный таргет, слишком мало строк, сплошные пропуски)
обрабатываются понятным сообщением RecommenderError, а не трассировкой стека.

Есть CLI (argparse): подать --csv или --openml-id и получить рекомендацию в консоль.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data_collection import preprocess_dataset
from src.interpretation import explain_instance, load_model_bundle
from src.meta_features import extract_meta_features
from src.utils import configure_network, get_logger

logger = get_logger(__name__)

# Минимум строк для осмысленного расчёта мета-признаков при инференсе.
MIN_INFERENCE_ROWS: int = 50


class RecommenderError(ValueError):
    """Понятная пользователю ошибка входных данных (не баг пайплайна)."""


# --------------------------------------------------------------------------- #
# Загрузка входного датасета                                                  #
# --------------------------------------------------------------------------- #
def load_dataset(
    source: str | int, target_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Загрузить датасет из CSV-файла или из OpenML по id.

    Returns
    -------
    (X, y, name) — признаки, целевая переменная, человекочитаемое имя источника.
    """
    # OpenML id — целое число или строка из цифр
    if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
        return _load_from_openml(int(source))
    return _load_from_csv(str(source), target_column)


def _load_from_csv(
    path: str, target_column: str | None
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Прочитать CSV; целевой столбец — заданный, либо 'target'/'class', либо последний."""
    file = Path(path)
    if not file.exists():
        raise RecommenderError(f"Файл не найден: {path}")
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001
        raise RecommenderError(f"Не удалось прочитать CSV «{path}»: {exc}") from exc

    if df.shape[1] < 2:
        raise RecommenderError("В датасете меньше двух столбцов — нужны признаки и таргет.")

    if target_column is None:
        for candidate in ("target", "class", "Class", "label"):
            if candidate in df.columns:
                target_column = candidate
                break
        else:
            target_column = df.columns[-1]  # по умолчанию — последний столбец
    if target_column not in df.columns:
        raise RecommenderError(
            f"Столбец-таргет «{target_column}» не найден. Доступны: {list(df.columns)}"
        )

    y = df[target_column]
    X = df.drop(columns=target_column)
    return X, y, file.stem


def _load_from_openml(did: int) -> tuple[pd.DataFrame, pd.Series, str]:
    """Скачать датасет OpenML по id (с обходом прокси)."""
    configure_network()
    try:
        import openml
        ds = openml.datasets.get_dataset(did, download_data=True)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    except Exception as exc:  # noqa: BLE001
        raise RecommenderError(f"Не удалось загрузить датасет OpenML id={did}: {exc}") from exc
    if X is None or y is None:
        raise RecommenderError(f"OpenML id={did}: пустые данные или не задан таргет.")
    return X, y, f"openml_{did}_{ds.name}"


# --------------------------------------------------------------------------- #
# Валидация и подготовка вектора мета-признаков                                #
# --------------------------------------------------------------------------- #
def _validate(X: pd.DataFrame, y: pd.Series) -> None:
    """Проверить пригодность датасета; при проблеме — понятное сообщение."""
    n_classes = int(pd.Series(y).dropna().nunique())
    if n_classes < 2:
        raise RecommenderError(
            f"Целевая переменная содержит {n_classes} класс(а) — нужна бинарная задача."
        )
    if n_classes > 2:
        raise RecommenderError(
            f"Датасет не бинарный: у таргета {n_classes} классов. "
            "Инструмент рассчитан на бинарную классификацию."
        )
    if len(X) < MIN_INFERENCE_ROWS:
        raise RecommenderError(
            f"Слишком мало строк ({len(X)}): нужно минимум {MIN_INFERENCE_ROWS} "
            "для устойчивого расчёта мета-признаков."
        )


def _corpus_matrix(bundle: dict) -> pd.DataFrame:
    """Матрица мета-признаков корпуса (фон для SHAP и медианы для импутации)."""
    return pd.read_csv(config.META_FEATURES_PATH, index_col="did")[bundle["feature_names"]]


def compute_feature_vector(X: pd.DataFrame, y: pd.Series, bundle: dict) -> pd.Series:
    """
    Посчитать мета-признаки нового датасета и выровнять к 132 колонкам модели.

    Пропуски/inf в ожидаемых признаках импутируются медианой по корпусу — так
    вектор согласован с тем, на чём обучалась мета-модель.
    """
    X_proc, y_proc = preprocess_dataset(X, y)
    try:
        raw = extract_meta_features(X_proc, y_proc)
    except Exception as exc:  # noqa: BLE001
        raise RecommenderError(f"Не удалось посчитать мета-признаки: {exc}") from exc

    feature_names = list(bundle["feature_names"])
    medians = _corpus_matrix(bundle).median(axis=0)

    vec = raw.reindex(feature_names)
    vec = vec.replace([np.inf, -np.inf], np.nan)
    vec = vec.fillna(medians).fillna(0.0)
    return vec.astype(float)


# --------------------------------------------------------------------------- #
# Главная функция рекомендации                                                #
# --------------------------------------------------------------------------- #
def recommend(
    source: str | int,
    *,
    target_column: str | None = None,
    explain: bool = True,
    bundle: dict | None = None,
) -> dict:
    """
    Рекомендовать метод борьбы с дисбалансом для произвольного датасета.

    Returns
    -------
    dict:
        dataset — имя источника;
        recommended_method — рекомендованный метод;
        probabilities — {метод: вероятность} по всем 6 методам (по убыванию);
        explanation — {text, waterfall_path} (если explain=True), иначе None.
    """
    bundle = bundle or load_model_bundle()
    classes = list(bundle["classes"])

    X, y, name = load_dataset(source, target_column)
    _validate(X, y)
    vec = compute_feature_vector(X, y, bundle)

    proba = bundle["model"].predict_proba(vec.to_numpy().reshape(1, -1))[0]
    pairs = zip(classes, (float(p) for p in proba), strict=False)
    probabilities = dict(sorted(pairs, key=lambda kv: kv[1], reverse=True))
    recommended = max(probabilities, key=probabilities.get)

    explanation = None
    if explain:
        background = _corpus_matrix(bundle)
        info = explain_instance(vec, bundle, background, label=name)
        explanation = {"text": info["text"], "waterfall_path": info["waterfall_path"],
                       "factors": info["factors"]}

    logger.info("Датасет «%s»: рекомендован метод %s", name, recommended)
    return {
        "dataset": name,
        "recommended_method": recommended,
        "probabilities": probabilities,
        "explanation": explanation,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _format_report(result: dict) -> str:
    """Красиво оформить результат для вывода в консоль."""
    lines = [
        "=" * 60,
        f"Датасет: {result['dataset']}",
        f"РЕКОМЕНДОВАННЫЙ МЕТОД: {result['recommended_method']}",
        "-" * 60,
        "Вероятности по методам:",
    ]
    for method, p in result["probabilities"].items():
        marker = " <-- рекомендация" if method == result["recommended_method"] else ""
        lines.append(f"  {method:<20} {p:6.1%}{marker}")
    if result["explanation"]:
        lines += ["-" * 60, "Объяснение (SHAP):", f"  {result['explanation']['text']}"]
        if result["explanation"]["waterfall_path"]:
            lines.append(f"  График: {result['explanation']['waterfall_path']}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: подать датасет и получить рекомендацию."""
    parser = argparse.ArgumentParser(
        description="Рекомендация метода борьбы с дисбалансом классов по мета-признакам."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="путь к CSV-файлу с датасетом")
    group.add_argument("--openml-id", type=int, help="id датасета в OpenML")
    parser.add_argument("--target-column", help="имя столбца-таргета (для CSV)")
    parser.add_argument("--no-explain", action="store_true", help="без SHAP-объяснения")
    args = parser.parse_args(argv)

    source: str | int = args.openml_id if args.openml_id is not None else args.csv
    try:
        result = recommend(
            source, target_column=args.target_column, explain=not args.no_explain
        )
    except RecommenderError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
