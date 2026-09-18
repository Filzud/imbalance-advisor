"""Тесты для src/utils.py."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

import config
from src import utils


def test_set_global_seed_reproducible():
    """После фиксации seed последовательность NumPy воспроизводится."""
    utils.set_global_seed(123)
    a = np.random.rand(5)
    utils.set_global_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_get_logger_no_duplicate_handlers():
    """Повторный вызов get_logger не добавляет дублирующие хендлеры."""
    name = "test_logger_unique"
    logging.getLogger(name).handlers.clear()
    logger1 = utils.get_logger(name)
    n_handlers = len(logger1.handlers)
    logger2 = utils.get_logger(name)
    assert logger1 is logger2
    assert len(logger2.handlers) == n_handlers
    assert n_handlers >= 2  # консоль + файл


def test_save_figure_creates_png():
    """save_figure создаёт PNG с правильным именем в FIGURES_DIR."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = utils.save_figure(fig, "unit_test_plot")
    assert path.exists()
    assert path.suffix == f".{config.FIG_FORMAT}"
    assert path.parent == config.FIGURES_DIR
    assert path.stem == "unit_test_plot"


def test_save_figure_strips_extension_and_uses_subdir():
    """Расширение в имени игнорируется; subdir создаётся."""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    path = utils.save_figure(fig, "name.jpg", subdir="sub")
    assert path.stem == "name"
    assert path.suffix == f".{config.FIG_FORMAT}"
    assert path.parent.name == "sub"
    assert path.exists()
