"""
Общие утилиты проекта: фиксация seed, логирование и — главное —
хелпер save_figure, через который КАЖДЫЙ модуль сохраняет графики
в docs/figures/ единообразно (PNG, dpi=200, bbox_inches="tight").
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import matplotlib
import numpy as np

# Headless-бэкенд: пайплайн только СОХРАНЯЕТ графики в файлы (docs/figures/),
# окна не открываются. Это делает генерацию картинок воспроизводимой и не
# зависящей от наличия GUI/Tk. Должно стоять до импорта pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  — после выбора бэкенда

import config

__all__ = ["set_global_seed", "configure_network", "get_logger", "save_figure"]


# --------------------------------------------------------------------------- #
# Воспроизводимость                                                            #
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Зафиксировать seed во всех источниках случайности (Python, NumPy, hash)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Сеть: обход системного SOCKS-прокси для доменов OpenML                        #
# --------------------------------------------------------------------------- #
def configure_network() -> None:
    """
    Обойти системный SOCKS-прокси для запросов к OpenML.

    В окружении разработки Windows прописан системный SOCKS-прокси (WinINET),
    который `requests` подхватывает автоматически и который искажает/обрезает
    ответы OpenML API (листинг датасетов приходит почти пустым). Прямой доступ
    в сеть при этом работает, поэтому мы добавляем домены OpenML в `no_proxy` —
    для них соединение идёт напрямую, минуя прокси. Остальной трафик не трогаем.

    Управляется флагом config.BYPASS_SYSTEM_PROXY. Идемпотентна.
    """
    if not config.BYPASS_SYSTEM_PROXY:
        return
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        hosts = {h.strip() for h in current.split(",") if h.strip()}
        hosts.update(h.strip() for h in config.NO_PROXY_HOSTS.split(","))
        os.environ[var] = ",".join(sorted(hosts))


# --------------------------------------------------------------------------- #
# Логирование                                                                  #
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Логгер, пишущий и в консоль, и в logs/pipeline.log (без дублей хендлеров)."""
    logger = logging.getLogger(name)
    if logger.handlers:                      # уже настроен — не плодим хендлеры
        return logger

    logger.setLevel(config.LOG_LEVEL)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Сохранение графиков — ЕДИНАЯ точка для всего проекта                          #
# --------------------------------------------------------------------------- #
def save_figure(
    fig: plt.Figure,
    name: str,
    *,
    subdir: str | None = None,
    close: bool = True,
) -> Path:
    """
    Сохранить matplotlib-фигуру в docs/figures/ с параметрами из config.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Фигура для сохранения.
    name : str
        Имя файла без расширения (например, "shap_global_importance").
    subdir : str | None
        Необязательный подкаталог внутри docs/figures/.
    close : bool
        Закрыть фигуру после сохранения (освобождает память; True по умолчанию).

    Returns
    -------
    pathlib.Path
        Полный путь к сохранённому файлу.
    """
    stem = Path(name).stem                       # защита от случайного расширения в name
    target_dir = config.FIGURES_DIR if subdir is None else config.FIGURES_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / f"{stem}.{config.FIG_FORMAT}"
    fig.savefig(
        out_path,
        dpi=config.FIG_DPI,
        bbox_inches=config.FIG_BBOX,
        format=config.FIG_FORMAT,
    )
    get_logger(__name__).info("Сохранён график: %s", out_path)

    if close:
        plt.close(fig)
    return out_path
