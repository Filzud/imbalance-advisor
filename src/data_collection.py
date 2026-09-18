"""
Модуль 1. Сбор корпуса датасетов из OpenML.

Назначение:
    * через OpenML API получить список бинарных несбалансированных датасетов;
    * отфильтровать кандидатов по качеству (размер, доля пропусков, IR, дубликаты);
    * привести каждый датасет к единому числовому виду (кодировка категориальных,
      импутация пропусков);
    * сохранить обработанные датасеты в data/processed/ (кэш) и записать манифест
      data/manifest.csv (id, имя, размеры, IR, доля пропусков);
    * построить два отдельных графика — распределение IR и распределение размеров
      корпуса — и сохранить их в docs/figures/.

Логика разбита на чистые (легко тестируемые) функции и тонкий слой обращения к сети.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd

import config
from src.utils import configure_network, get_logger, save_figure, set_global_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Структура записи манифеста                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetRecord:
    """Одна строка манифеста корпуса."""

    did: int
    name: str
    n_rows: int
    n_features: int
    n_classes: int
    imbalance_ratio: float
    ir_band: str
    missing_fraction: float
    processed_path: str
    source: str = "openml"


# --------------------------------------------------------------------------- #
# Чистые функции: метрики и предобработка                                      #
# --------------------------------------------------------------------------- #
def compute_imbalance_ratio(y: pd.Series | np.ndarray) -> float:
    """Imbalance Ratio = размер мажоритарного класса / размер миноритарного."""
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() == 0:
        return float("inf")
    return float(counts.max() / counts.min())


def assign_ir_band(ir: float) -> str:
    """
    Отнести значение Imbalance Ratio к диапазону дисбаланса (light/medium/heavy).

    Диапазоны заданы в config.IR_BANDS как полуинтервалы [lo, hi). Значения
    ниже первого lo относятся к первому диапазону, выше последнего — к последнему.
    """
    for name, lo, hi in config.IR_BANDS:
        if lo <= ir < hi:
            return name
    # ir вне всех интервалов (например, ровно на верхней границе последнего):
    return config.IR_BANDS[-1][0] if ir >= config.IR_BANDS[-1][1] else config.IR_BANDS[0][0]


def missing_fraction(frame: pd.DataFrame) -> float:
    """Доля пропущенных ячеек во всём датафрейме признаков (0..1)."""
    if frame.size == 0:
        return 0.0
    return float(frame.isna().to_numpy().sum() / frame.size)


def _encode_target(y: pd.Series | np.ndarray) -> pd.Series:
    """
    Закодировать бинарную целевую в {0, 1}, где 1 — миноритарный (положительный) класс.

    Это согласуется с метриками F1 / PR-AUC, которые в проекте считаются по
    положительному (редкому) классу.
    """
    y = pd.Series(y).reset_index(drop=True)
    codes, _ = pd.factorize(y)
    s = pd.Series(codes)
    minority_code = s.value_counts().idxmin()
    return (s == minority_code).astype(int)


def preprocess_dataset(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    categorical_indicator: list[bool] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Привести датасет к единому числовому виду.

    Шаги:
        1. выровнять X и y, выбросить строки с пропущенным таргетом;
        2. удалить дубликаты строк;
        3. категориальные признаки заполнить модой и закодировать порядково (ordinal);
        4. числовые признаки заполнить медианой;
        5. закодировать таргет в {0, 1} (1 — миноритарный класс).

    Returns
    -------
    (X_processed, y_processed)
        X_processed — DataFrame только из числовых столбцов без NaN;
        y_processed — Series int со значениями {0, 1}.
    """
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True)

    # 0. OpenML нередко отдаёт разреженные (sparse) ARFF-колонки — приводим их
    #    к плотному виду, иначе статистики (median и т.п.) на Sparse-dtype падают.
    for col in X.columns:
        if isinstance(X[col].dtype, pd.SparseDtype):
            X[col] = X[col].sparse.to_dense()

    # 1. строки с пропущенным таргетом не нужны
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    # 2. дубликаты строк (вместе с таргетом)
    combined = pd.concat([X, y.rename("__target__")], axis=1).drop_duplicates()
    X = combined.drop(columns="__target__").reset_index(drop=True)
    y = combined["__target__"].reset_index(drop=True)

    # 3–4. определить типы столбцов
    if categorical_indicator is not None and len(categorical_indicator) == X.shape[1]:
        cat_cols = [
            c for c, is_cat in zip(X.columns, categorical_indicator, strict=False) if is_cat
        ]
    else:
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for col in cat_cols:
        mode = X[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "__missing__"
        # factorize: пропуски (-1) тоже получают свой код после заполнения
        X[col] = pd.factorize(X[col].fillna(fill))[0]

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median = X[col].median()
        X[col] = X[col].fillna(0.0 if pd.isna(median) else median)

    X = X.astype(float)
    y_processed = _encode_target(y)
    return X, y_processed


# --------------------------------------------------------------------------- #
# Отбор кандидатов по метаданным OpenML (без скачивания данных)                #
# --------------------------------------------------------------------------- #
def select_candidates(listing: pd.DataFrame) -> pd.DataFrame:
    """
    Отобрать кандидатов из таблицы метаданных OpenML (output_format="dataframe").

    Фильтры (по доступным колонкам метаданных):
        * ровно 2 класса;
        * число объектов в [MIN_ROWS, MAX_ROWS];
        * imbalance ratio >= MIN_IMBALANCE_RATIO (из размеров классов);
        * доля пропусков <= MAX_MISSING_FRACTION (если колонка доступна);
        * без дубликатов по имени (берётся первая встретившаяся версия).

    Returns
    -------
    DataFrame кандидатов c колонками imbalance_ratio и ir_band, стабильно
    отсортированный по did. Стратификация по ir_band выполняется на следующем
    шаге (stratified_candidate_order), поэтому здесь по IR не обрезаем.
    """
    df = listing.copy()

    required = {"did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"В листинге OpenML нет колонок: {sorted(missing_cols)}")

    df = df[df["NumberOfClasses"] == config.N_CLASSES_REQUIRED]
    df = df[df["NumberOfInstances"].between(config.MIN_ROWS, config.MAX_ROWS)]

    # IR из размеров классов, если они есть
    if {"MajorityClassSize", "MinorityClassSize"}.issubset(df.columns):
        minority = df["MinorityClassSize"].replace(0, np.nan)
        df = df.assign(imbalance_ratio=df["MajorityClassSize"] / minority)
        df = df[df["imbalance_ratio"] >= config.MIN_IMBALANCE_RATIO]
    else:
        df = df.assign(imbalance_ratio=np.nan)

    # доля пропусков, если есть нужные колонки
    if {"NumberOfMissingValues", "NumberOfInstances", "NumberOfFeatures"}.issubset(df.columns):
        cells = df["NumberOfInstances"] * df["NumberOfFeatures"]
        frac = df["NumberOfMissingValues"] / cells.replace(0, np.nan)
        df = df[frac.fillna(0.0) <= config.MAX_MISSING_FRACTION]

    # IR обязателен для стратификации: кандидатов без известного IR отбрасываем.
    df = df[df["imbalance_ratio"].notna()]
    df = df.assign(ir_band=df["imbalance_ratio"].apply(assign_ir_band))

    df = df.drop_duplicates(subset="name", keep="first")
    df = df.sort_values("did").reset_index(drop=True)  # стабильный порядок
    return df


def stratified_candidate_order(
    candidates: pd.DataFrame,
    n_datasets: int = config.N_DATASETS_TARGET,
    oversample: float = config.CANDIDATE_OVERSAMPLE,
) -> list[int]:
    """
    Построить очередь did для скачивания, стратифицированную по диапазонам IR.

    Логика:
        * внутри каждого диапазона (light/medium/heavy) берём до
          ceil(n_datasets / n_bands * oversample) кандидатов (запас на неудачные
          скачивания и на отсев по фактическим данным);
        * диапазоны чередуем по кругу (round-robin), чтобы очередь была
          перемешана по характеру дисбаланса, а не шла блоками.

    Итог — порядок did; квоты на итоговый корпус накладывает build_corpus.
    """
    bands = [b[0] for b in config.IR_BANDS]
    per_band = int(np.ceil(n_datasets / len(bands) * oversample))

    queues: list[list[int]] = []
    for band in bands:
        sub = candidates[candidates["ir_band"] == band]
        queues.append(sub["did"].astype(int).tolist()[:per_band])

    order: list[int] = []
    for i in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if i < len(q):
                order.append(q[i])
    return order


def _band_targets(n_datasets: int) -> dict[str, int]:
    """Равномерно распределить n_datasets по диапазонам IR (остаток — первым)."""
    bands = [b[0] for b in config.IR_BANDS]
    base, extra = divmod(n_datasets, len(bands))
    return {band: base + (1 if i < extra else 0) for i, band in enumerate(bands)}


# --------------------------------------------------------------------------- #
# Слой обращения к сети (тонкий, мокается в тестах)                            #
# --------------------------------------------------------------------------- #
def _fetch_listing() -> pd.DataFrame:
    """Получить таблицу метаданных всех активных датасетов OpenML."""
    return openml.datasets.list_datasets(output_format="dataframe")


def download_and_process(did: int) -> tuple[DatasetRecord, pd.DataFrame, pd.Series] | None:
    """
    Скачать датасет по did, предобработать и проверить фактическое качество данных.

    Возвращает None, если датасет не прошёл фильтры качества по реальным данным
    или произошла ошибка скачивания (этап устойчив к сбоям отдельных датасетов).
    """
    try:
        ds = openml.datasets.get_dataset(did, download_data=True)
        X, y, categorical_indicator, _ = ds.get_data(
            target=ds.default_target_attribute, dataset_format="dataframe"
        )
        if X is None or y is None or X.empty:
            logger.warning("Пропуск did=%s: пустые данные", did)
            return None
        raw_missing = missing_fraction(X)
        X_proc, y_proc = preprocess_dataset(X, y, categorical_indicator)
    except Exception as exc:  # noqa: BLE001 — пропускаем проблемный датасет, не роняя пайплайн
        logger.warning("Пропуск did=%s: ошибка скачивания/обработки: %s", did, exc)
        return None

    # фактическая проверка качества по реальным данным
    n_rows, n_features = X_proc.shape
    n_classes = int(pd.Series(y_proc).nunique())
    ir = compute_imbalance_ratio(y_proc)

    if (
        n_rows < config.MIN_ROWS
        or n_classes != config.N_CLASSES_REQUIRED
        or ir < config.MIN_IMBALANCE_RATIO
        or not np.isfinite(ir)
        or raw_missing > config.MAX_MISSING_FRACTION
    ):
        logger.info(
            "Пропуск did=%s (name=%s): rows=%s classes=%s IR=%.2f missing=%.2f",
            did, ds.name, n_rows, n_classes, ir, raw_missing,
        )
        return None

    processed_path = config.PROCESSED_DIR / f"{did}.csv"
    out = X_proc.copy()
    out["target"] = y_proc.to_numpy()
    out.to_csv(processed_path, index=False)

    # В манифесте храним путь относительно корня репозитория, если возможно
    # (в проде так и есть); иначе — абсолютный (например, при тестах в tmp).
    try:
        stored_path = processed_path.relative_to(config.ROOT_DIR)
    except ValueError:
        stored_path = processed_path

    record = DatasetRecord(
        did=int(did),
        name=str(ds.name),
        n_rows=int(n_rows),
        n_features=int(n_features),
        n_classes=n_classes,
        imbalance_ratio=round(ir, 4),
        ir_band=assign_ir_band(ir),
        missing_fraction=round(raw_missing, 4),
        processed_path=str(stored_path),
    )
    logger.info("Добавлен did=%s name=%s IR=%.2f rows=%s", did, ds.name, ir, n_rows)
    return record, X_proc, y_proc


# --------------------------------------------------------------------------- #
# Графики корпуса (каждый — отдельный файл)                                    #
# --------------------------------------------------------------------------- #
def plot_ir_distribution(manifest: pd.DataFrame) -> None:
    """Гистограмма распределения Imbalance Ratio по корпусу → отдельный PNG.

    IR несбалансированных корпусов сильно скошен, поэтому ось X — логарифмическая;
    вертикальными линиями отмечены границы диапазонов стратификации.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ir = manifest["imbalance_ratio"].to_numpy()
    bins = np.logspace(np.log10(max(ir.min(), 1.0)), np.log10(ir.max() + 1e-9), 20)
    ax.hist(ir, bins=bins, color="#3b6ea5", edgecolor="black")
    ax.set_xscale("log")
    for _, _lo, hi in config.IR_BANDS:
        if np.isfinite(hi):
            ax.axvline(hi, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Imbalance Ratio (мажоритарный / миноритарный), log-шкала")
    ax.set_ylabel("Число датасетов")
    ax.set_title(f"Распределение Imbalance Ratio по корпусу (N={len(manifest)})")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "corpus_ir_distribution")


def plot_ir_bands(manifest: pd.DataFrame) -> None:
    """Число датасетов в каждом диапазоне дисбаланса → отдельный PNG.

    Наглядно показывает, что корпус стратифицирован (а не смещён к экстремумам).
    """
    order = [b[0] for b in config.IR_BANDS]
    counts = manifest["ir_band"].value_counts().reindex(order, fill_value=0)
    labels = [f"{name}\n[{lo:g}, {'∞' if not np.isfinite(hi) else f'{hi:g}'})"
              for name, lo, hi in config.IR_BANDS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.to_numpy(), color=["#78c2ad", "#f3969a", "#c0392b"])
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Диапазон Imbalance Ratio")
    ax.set_ylabel("Число датасетов")
    ax.set_title("Стратификация корпуса по характеру дисбаланса")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "corpus_ir_bands")


def plot_size_distribution(manifest: pd.DataFrame) -> None:
    """Распределение размеров (объекты × признаки), цвет — IR → отдельный PNG."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter = ax.scatter(
        manifest["n_rows"],
        manifest["n_features"],
        c=manifest["imbalance_ratio"],
        cmap="viridis",
        s=40,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Число объектов (log)")
    ax.set_ylabel("Число признаков")
    ax.set_title("Размеры датасетов корпуса")
    fig.colorbar(scatter, ax=ax, label="Imbalance Ratio")
    ax.grid(alpha=0.3)
    save_figure(fig, "corpus_size_distribution")


# --------------------------------------------------------------------------- #
# Оркестрация сбора корпуса                                                    #
# --------------------------------------------------------------------------- #
def build_corpus(
    n_datasets: int = config.N_DATASETS_TARGET,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Собрать корпус датасетов и записать манифест.

    Parameters
    ----------
    n_datasets : int
        Целевое число датасетов в корпусе.
    force : bool
        Если False и манифест уже существует — вернуть кэш без повторного сбора.

    Returns
    -------
    DataFrame манифеста корпуса.
    """
    set_global_seed()
    configure_network()  # обойти системный SOCKS-прокси для OpenML
    openml.config.set_root_cache_directory(str(config.RAW_DIR))

    if config.MANIFEST_PATH.exists() and not force:
        logger.info("Манифест найден в кэше: %s (force=False)", config.MANIFEST_PATH)
        manifest = pd.read_csv(config.MANIFEST_PATH)
        _render_corpus_figures(manifest)
        return manifest

    logger.info("Запрос листинга датасетов OpenML…")
    candidates = select_candidates(_fetch_listing())
    band_counts = candidates["ir_band"].value_counts().to_dict()
    logger.info("Кандидатов после фильтрации: %s (по диапазонам IR: %s)",
                len(candidates), band_counts)

    order = stratified_candidate_order(candidates, n_datasets)
    band_by_did = dict(zip(candidates["did"].astype(int), candidates["ir_band"], strict=False))
    targets = _band_targets(n_datasets)
    logger.info("Целевые квоты по диапазонам IR: %s", targets)

    records: list[DatasetRecord] = []
    accepted: dict[str, int] = {b: 0 for b in targets}
    deferred: list[int] = []  # кандидаты, отложенные из-за заполненной квоты

    # Проход 1: наполняем каждый диапазон до его квоты.
    for did in order:
        if len(records) >= n_datasets:
            break
        band = band_by_did.get(did, config.IR_BANDS[0][0])
        if accepted[band] >= targets[band]:
            deferred.append(did)
            continue
        result = download_and_process(did)
        if result is not None:
            records.append(result[0])
            accepted[result[0].ir_band] += 1

    # Проход 2: если какие-то диапазоны не добрали (мало данных/сбои скачивания),
    # добираем корпус до цели из отложенных кандидатов, не глядя на квоты.
    for did in deferred:
        if len(records) >= n_datasets:
            break
        result = download_and_process(did)
        if result is not None:
            records.append(result[0])
            accepted[result[0].ir_band] += 1

    if not records:
        raise RuntimeError("Не удалось собрать ни одного датасета — проверь сеть/фильтры.")

    manifest = pd.DataFrame([asdict(r) for r in records])
    manifest = manifest.sort_values("imbalance_ratio").reset_index(drop=True)
    manifest.to_csv(config.MANIFEST_PATH, index=False)
    logger.info("Манифест сохранён: %s (%s датасетов, по диапазонам: %s)",
                config.MANIFEST_PATH, len(manifest), accepted)

    _render_corpus_figures(manifest)
    return manifest


def _render_corpus_figures(manifest: pd.DataFrame) -> None:
    """Построить и сохранить обзорные графики корпуса (если есть данные)."""
    if manifest.empty:
        logger.warning("Манифест пуст — графики корпуса не строятся.")
        return
    plot_ir_distribution(manifest)
    plot_ir_bands(manifest)
    plot_size_distribution(manifest)


if __name__ == "__main__":
    build_corpus()
