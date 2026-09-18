"""Тесты для src/data_collection.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import data_collection as dc


# --------------------------------------------------------------------------- #
# Чистые функции                                                              #
# --------------------------------------------------------------------------- #
def test_compute_imbalance_ratio_basic():
    y = pd.Series([0] * 80 + [1] * 20)
    assert dc.compute_imbalance_ratio(y) == pytest.approx(4.0)


def test_compute_imbalance_ratio_single_class_is_inf():
    assert dc.compute_imbalance_ratio(pd.Series([1, 1, 1])) == float("inf")


def test_missing_fraction():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]})
    assert dc.missing_fraction(frame) == pytest.approx(3 / 4)


def test_missing_fraction_empty():
    assert dc.missing_fraction(pd.DataFrame()) == 0.0


def test_encode_target_minority_is_one():
    y = pd.Series(["neg"] * 90 + ["pos"] * 10)
    enc = dc._encode_target(y)
    assert set(enc.unique()) == {0, 1}
    # миноритарный класс ('pos') должен стать 1
    assert enc.sum() == 10


def test_preprocess_dataset_numeric_and_no_nan(imbalanced_df):
    X, y = imbalanced_df
    X_proc, y_proc = dc.preprocess_dataset(X, y)
    # все столбцы числовые, без NaN
    assert X_proc.isna().to_numpy().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(X_proc[c]) for c in X_proc.columns)
    # таргет бинарный {0,1}
    assert set(y_proc.unique()) <= {0, 1}
    # длины согласованы
    assert len(X_proc) == len(y_proc)


def test_preprocess_dataset_drops_duplicate_rows():
    X = pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 6]})
    y = pd.Series([0, 0, 1])
    X_proc, y_proc = dc.preprocess_dataset(X, y)
    assert len(X_proc) == 2  # одна пара дублей удалена


def test_preprocess_respects_categorical_indicator():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["u", "v", "u", "v"]})
    y = pd.Series([0, 1, 0, 1])
    X_proc, _ = dc.preprocess_dataset(X, y, categorical_indicator=[False, True])
    assert pd.api.types.is_numeric_dtype(X_proc["b"])


# --------------------------------------------------------------------------- #
# Отбор кандидатов                                                            #
# --------------------------------------------------------------------------- #
def _listing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "did": [1, 2, 3, 4, 5],
            "name": ["good", "too_small", "multiclass", "balanced", "good"],
            "NumberOfInstances": [1000, 50, 1000, 1000, 1000],
            "NumberOfFeatures": [10, 10, 10, 10, 10],
            "NumberOfClasses": [2, 2, 3, 2, 2],
            "MajorityClassSize": [900, 40, 800, 520, 900],
            "MinorityClassSize": [100, 10, 100, 480, 100],
            "NumberOfMissingValues": [0, 0, 0, 0, 0],
        }
    )


def test_select_candidates_filters():
    out = dc.select_candidates(_listing())
    # отсеяны: did=2 (мелкий), did=3 (многоклассовый), did=4 (IR≈1.08 < 1.5)
    # did=5 — дубликат имени 'good', остаётся первый (did=1)
    assert out["did"].tolist() == [1]
    assert out.loc[0, "imbalance_ratio"] == pytest.approx(9.0)
    # каждому кандидату проставлен диапазон дисбаланса
    assert out.loc[0, "ir_band"] == "heavy"  # IR=9.0 → [9, inf)


def test_select_candidates_missing_columns_raises():
    with pytest.raises(ValueError):
        dc.select_candidates(pd.DataFrame({"did": [1]}))


# --------------------------------------------------------------------------- #
# Стратификация по диапазонам IR                                              #
# --------------------------------------------------------------------------- #
def test_assign_ir_band_boundaries():
    assert dc.assign_ir_band(1.5) == "light"
    assert dc.assign_ir_band(2.9) == "light"
    assert dc.assign_ir_band(3.0) == "medium"
    assert dc.assign_ir_band(8.9) == "medium"
    assert dc.assign_ir_band(9.0) == "heavy"
    assert dc.assign_ir_band(600.0) == "heavy"


def test_band_targets_distributes_evenly():
    targets = dc._band_targets(90)
    assert sum(targets.values()) == 90
    assert set(targets) == {b[0] for b in config.IR_BANDS}
    # 90 / 3 диапазона → ровно по 30
    assert set(targets.values()) == {30}


def test_stratified_candidate_order_interleaves_bands():
    # по 4 кандидата в каждом из трёх диапазонов
    cand = pd.DataFrame(
        {
            "did": list(range(12)),
            "ir_band": (["light"] * 4) + (["medium"] * 4) + (["heavy"] * 4),
            "imbalance_ratio": [2.0] * 4 + [5.0] * 4 + [20.0] * 4,
        }
    )
    order = dc.stratified_candidate_order(cand, n_datasets=6, oversample=2.0)
    # первые три элемента — по одному из каждого диапазона (round-robin)
    first_bands = [cand.set_index("did").loc[d, "ir_band"] for d in order[:3]]
    assert set(first_bands) == {"light", "medium", "heavy"}


# --------------------------------------------------------------------------- #
# Графики                                                                     #
# --------------------------------------------------------------------------- #
def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "imbalance_ratio": [2.0, 5.0, 9.0, 3.0],
            "ir_band": ["light", "medium", "heavy", "medium"],
            "n_rows": [300, 1000, 5000, 800],
            "n_features": [5, 10, 20, 8],
        }
    )


def test_plot_ir_distribution_saves_file():
    dc.plot_ir_distribution(_manifest())
    assert (config.FIGURES_DIR / "corpus_ir_distribution.png").exists()


def test_plot_ir_bands_saves_file():
    dc.plot_ir_bands(_manifest())
    assert (config.FIGURES_DIR / "corpus_ir_bands.png").exists()


def test_plot_size_distribution_saves_file():
    dc.plot_size_distribution(_manifest())
    assert (config.FIGURES_DIR / "corpus_size_distribution.png").exists()


# --------------------------------------------------------------------------- #
# Слой сети — мокаем openml                                                   #
# --------------------------------------------------------------------------- #
class _FakeDataset:
    def __init__(self, name, X, y, cat):
        self.name = name
        self.default_target_attribute = "target"
        self._X, self._y, self._cat = X, y, cat

    def get_data(self, target, dataset_format):  # noqa: ARG002
        return self._X, self._y, self._cat, list(self._X.columns)


def test_download_and_process_accepts_good_dataset(monkeypatch):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series([0] * 320 + [1] * 80)
    fake = _FakeDataset("synthetic_good", X, y, [False, False])

    monkeypatch.setattr(dc.openml.datasets, "get_dataset", lambda *a, **k: fake)
    result = dc.download_and_process(did=42)

    assert result is not None
    record, X_proc, y_proc = result
    assert record.did == 42
    assert record.n_classes == 2
    assert record.imbalance_ratio == pytest.approx(4.0)
    assert record.ir_band == "medium"  # IR=4.0 → [3, 9)
    # файл обработанного датасета сохранён
    assert (config.PROCESSED_DIR / "42.csv").exists()


def test_download_and_process_rejects_balanced(monkeypatch):
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=400)})
    y = pd.Series([0] * 200 + [1] * 200)  # IR=1 < порога
    fake = _FakeDataset("balanced", X, y, [False])

    monkeypatch.setattr(dc.openml.datasets, "get_dataset", lambda *a, **k: fake)
    assert dc.download_and_process(did=7) is None


def test_download_and_process_handles_download_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(dc.openml.datasets, "get_dataset", _boom)
    assert dc.download_and_process(did=999) is None


def test_build_corpus_stratifies_by_ir_band(monkeypatch):
    """Полный сбор (сеть замокана): корпус набирается с квотами по диапазонам IR."""
    # 5 датасетов в каждом из трёх диапазонов
    ir_by_did = {}
    rows = []
    did = 0
    for ir in (2.0, 5.0, 20.0):  # light / medium / heavy
        for _ in range(5):
            rows.append(
                {
                    "did": did, "name": f"ds{did}",
                    "NumberOfInstances": 1000, "NumberOfFeatures": 10,
                    "NumberOfClasses": 2,
                    "MajorityClassSize": int(1000 * ir / (ir + 1)),
                    "MinorityClassSize": int(1000 / (ir + 1)),
                    "NumberOfMissingValues": 0,
                }
            )
            ir_by_did[did] = ir
            did += 1
    listing = pd.DataFrame(rows)
    monkeypatch.setattr(dc, "_fetch_listing", lambda: listing)

    def _fake_download(did):
        ir = ir_by_did[did]
        rec = dc.DatasetRecord(
            did=did, name=f"ds{did}", n_rows=1000, n_features=10, n_classes=2,
            imbalance_ratio=ir, ir_band=dc.assign_ir_band(ir),
            missing_fraction=0.0, processed_path=f"{did}.csv",
        )
        return rec, None, None

    monkeypatch.setattr(dc, "download_and_process", _fake_download)

    manifest = dc.build_corpus(n_datasets=6, force=True)
    assert len(manifest) == 6
    # ровно по 2 датасета из каждого диапазона (квоты 6/3)
    assert manifest["ir_band"].value_counts().to_dict() == {"light": 2, "medium": 2, "heavy": 2}


def test_build_corpus_uses_cache(monkeypatch):
    """Если манифест уже есть и force=False — возвращается кэш без обращения к сети."""
    cached = _manifest()
    cached.to_csv(config.MANIFEST_PATH, index=False)

    def _fail_listing():
        raise AssertionError("сеть не должна вызываться при наличии кэша")

    monkeypatch.setattr(dc, "_fetch_listing", _fail_listing)
    out = dc.build_corpus(force=False)
    assert len(out) == len(cached)
