"""
Модуль 2. Извлечение мета-признаков датасетов.

Назначение:
    * для каждого датасета корпуса посчитать вектор мета-признаков фиксированной
      размерности: простые, статистические, информационные (в т.ч. mutual
      information), landmarking, model-based и геометрические меры сложности
      Ho-Basu — через pymfe; плюс собственные признаки дисбаланса (IR, доля
      миноритарного класса, нормированная энтропия классов);
    * собрать матрицу корпуса (N датасетов × m мета-признаков), привести её к
      единому набору колонок (выкинуть признаки с большой долей NaN по всему
      корпусу, оставшиеся пропуски импутировать медианой) и сохранить в data/;
    * построить и сохранить корреляционную матрицу мета-признаков в docs/figures/.

Дорогие меры сложности имеют сложность O(n²) по числу строк и признаков, поэтому
для расчёта (и только для расчёта) большие датасеты прореживаются по строкам, а
широкие — усекаются до top-k признаков с наибольшей дисперсией. Исходный корпус
в data/processed при этом не меняется.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.model_selection import train_test_split

import config
from src.utils import get_logger, save_figure, set_global_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Подготовка данных к расчёту мета-признаков                                   #
# --------------------------------------------------------------------------- #
def _prepare_for_extraction(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Ограничить размер данных для вычислимости дорогих мер сложности.

    * если строк больше MF_MAX_ROWS — стратифицированная подвыборка (сохраняет IR);
    * если признаков больше MF_MAX_FEATURES — берём top-k по дисперсии.

    Всё детерминировано (random_state = RANDOM_SEED).
    """
    if len(X) > config.MF_MAX_ROWS:
        X, _, y, _ = train_test_split(
            X, y,
            train_size=config.MF_MAX_ROWS,
            stratify=y,
            random_state=config.RANDOM_SEED,
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    if X.shape[1] > config.MF_MAX_FEATURES:
        top = X.var().sort_values(ascending=False).index[: config.MF_MAX_FEATURES]
        # стабильный порядок столбцов (как в исходном датасете)
        X = X[[c for c in X.columns if c in set(top)]]

    return X, y


# --------------------------------------------------------------------------- #
# Собственные признаки дисбаланса                                              #
# --------------------------------------------------------------------------- #
def custom_imbalance_features(y: pd.Series | np.ndarray) -> dict[str, float]:
    """
    Признаки, характеризующие сам дисбаланс классов (домен задачи).

    Returns
    -------
    dict с ключами:
        imbalance_ratio      — мажоритарный / миноритарный;
        minority_fraction    — доля миноритарного класса (0..0.5];
        class_entropy_norm   — энтропия распределения классов, нормированная
                               к [0, 1] (1 — идеальный баланс, 0 — вырождение).
    """
    counts = pd.Series(y).value_counts()
    n = int(counts.sum())
    maj, minr = int(counts.max()), int(counts.min())

    ir = float(maj / minr) if minr > 0 else float("inf")
    minority_fraction = float(minr / n) if n > 0 else 0.0

    p = counts.to_numpy(dtype=float) / n
    entropy = float(-(p * np.log2(p)).sum())
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    class_entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "imbalance_ratio": ir,
        "minority_fraction": minority_fraction,
        "class_entropy_norm": class_entropy_norm,
    }


# --------------------------------------------------------------------------- #
# Извлечение вектора мета-признаков для одного датасета                        #
# --------------------------------------------------------------------------- #
def extract_meta_features(X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.Series:
    """
    Посчитать вектор мета-признаков для одного датасета.

    Объединяет мета-признаки pymfe (группы из config.META_FEATURE_GROUPS,
    сводки config.META_FEATURE_SUMMARY) и собственные признаки дисбаланса.
    Вектор может содержать NaN/inf там, где мера не определена на данных, —
    консистентная чистка выполняется на уровне всего корпуса (см.
    build_meta_feature_matrix).

    Returns
    -------
    pd.Series, индексированная именами мета-признаков.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    X_prep, y_prep = _prepare_for_extraction(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pymfe шумит предупреждениями о неопр. мерах
        mfe = MFE(
            groups=list(config.META_FEATURE_GROUPS),
            summary=list(config.META_FEATURE_SUMMARY),
            random_state=config.RANDOM_SEED,
        )
        mfe.fit(X_prep.to_numpy(), y_prep.to_numpy())
        names, values = mfe.extract()

    series = pd.Series(
        np.asarray(values, dtype=float), index=list(names), dtype=float
    )
    # собственные признаки дисбаланса (перекрывают одноимённые, если вдруг есть)
    for key, val in custom_imbalance_features(y).items():
        series[key] = val
    return series


# --------------------------------------------------------------------------- #
# Матрица мета-признаков по всему корпусу                                       #
# --------------------------------------------------------------------------- #
def _clean_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Привести матрицу мета-признаков к единому набору колонок.

    * inf → NaN;
    * выбросить столбцы, где доля NaN по корпусу > MF_MAX_NAN_FRACTION;
    * оставшиеся NaN импутировать медианой столбца (если медиана сама NaN — 0.0).

    Итог: у всех датасетов один и тот же фиксированный набор мета-признаков без
    пропусков.
    """
    matrix = raw.replace([np.inf, -np.inf], np.nan)

    nan_fraction = matrix.isna().mean(axis=0)
    keep = nan_fraction[nan_fraction <= config.MF_MAX_NAN_FRACTION].index
    dropped = [c for c in matrix.columns if c not in set(keep)]
    if dropped:
        logger.info(
            "Выброшено мета-признаков с долей NaN > %.0f%%: %d (%s)",
            config.MF_MAX_NAN_FRACTION * 100, len(dropped), dropped,
        )
    matrix = matrix[keep]

    medians = matrix.median(axis=0, skipna=True)
    matrix = matrix.fillna(medians).fillna(0.0)
    return matrix


def build_meta_feature_matrix(
    manifest: pd.DataFrame | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Посчитать матрицу мета-признаков для всего корпуса и сохранить её в data/.

    Parameters
    ----------
    manifest : DataFrame | None
        Манифест корпуса (модуль 1). Если None — читается из config.MANIFEST_PATH.
    force : bool
        Если False и матрица уже посчитана — вернуть кэш без пересчёта.

    Returns
    -------
    DataFrame формы (N датасетов × m мета-признаков), индекс — did датасета.
    """
    set_global_seed()

    if config.META_FEATURES_PATH.exists() and not force:
        logger.info("Матрица мета-признаков найдена в кэше: %s", config.META_FEATURES_PATH)
        matrix = pd.read_csv(config.META_FEATURES_PATH, index_col="did")
        _render_correlation_figure(matrix)
        return matrix

    if manifest is None:
        manifest = pd.read_csv(config.MANIFEST_PATH)

    vectors: dict[int, pd.Series] = {}
    for _, row in manifest.iterrows():
        did = int(row["did"])
        path = config.PROCESSED_DIR / f"{did}.csv"
        if not path.exists():
            logger.warning("Пропуск did=%s: нет файла %s", did, path)
            continue
        df = pd.read_csv(path)
        X, y = df.drop(columns="target"), df["target"]
        vectors[did] = extract_meta_features(X, y)
        logger.info("Мета-признаки посчитаны: did=%s (%s), признаков=%d",
                    did, row.get("name", "?"), len(vectors[did]))

    if not vectors:
        raise RuntimeError("Не удалось посчитать мета-признаки ни для одного датасета.")

    # выравнивание по объединению имён признаков → единая матрица
    raw = pd.DataFrame(vectors).T
    raw.index.name = "did"
    matrix = _clean_matrix(raw)

    matrix.to_csv(config.META_FEATURES_PATH)
    logger.info("Матрица мета-признаков сохранена: %s (%d × %d)",
                config.META_FEATURES_PATH, matrix.shape[0], matrix.shape[1])

    _render_correlation_figure(matrix)
    return matrix


# --------------------------------------------------------------------------- #
# Корреляционная матрица мета-признаков (отдельный PNG)                         #
# --------------------------------------------------------------------------- #
def plot_meta_feature_correlation(matrix: pd.DataFrame) -> None:
    """Тепловая карта попарных корреляций мета-признаков → отдельный PNG."""
    corr = matrix.corr().to_numpy()
    m = corr.shape[0]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Корреляционная матрица мета-признаков (m={m})")
    ax.set_xlabel("Мета-признаки")
    ax.set_ylabel("Мета-признаки")
    # подписей признаков много (m≈100+) — оставляем разреженную сетку без имён
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="Коэффициент корреляции Пирсона")
    save_figure(fig, "meta_features_correlation")


def _render_correlation_figure(matrix: pd.DataFrame) -> None:
    """Построить корреляционную карту, если матрица непуста."""
    if matrix.empty:
        logger.warning("Матрица мета-признаков пуста — корреляция не строится.")
        return
    plot_meta_feature_correlation(matrix)


if __name__ == "__main__":
    build_meta_feature_matrix(force=True)
