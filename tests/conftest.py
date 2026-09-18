"""Общие фикстуры тестов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_figures_dir(tmp_path, monkeypatch):
    """
    Перенаправить вывод графиков и артефактов в tmp_path на время теста,
    чтобы тесты не засоряли реальные docs/figures, data/ и не зависели друг от друга.
    """
    import config
    from src import utils

    figures = tmp_path / "figures"
    processed = tmp_path / "processed"
    figures.mkdir()
    processed.mkdir()

    monkeypatch.setattr(config, "FIGURES_DIR", figures)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.csv")
    # utils.save_figure читает config.FIGURES_DIR во время вызова — патч выше уже действует.
    monkeypatch.setattr(utils, "config", config)
    return tmp_path


@pytest.fixture
def imbalanced_df() -> tuple[pd.DataFrame, pd.Series]:
    """Небольшой синтетический бинарный несбалансированный датасет с пропусками и категорией."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
            "cat_c": rng.choice(["x", "y", "z"], size=n),
        }
    )
    # внести пропуски
    X.loc[:9, "num_a"] = np.nan
    X.loc[:4, "cat_c"] = None
    # несбалансированный таргет ~ 1:5
    y = pd.Series(["neg"] * 250 + ["pos"] * 50)
    return X, y
