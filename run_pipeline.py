"""
Единая точка воспроизводимого запуска всего пайплайна (модули 1–7).

Одной командой прогоняет весь путь: сбор корпуса → мета-признаки → разметка →
обучение мета-модели → интерпретация SHAP → эксперименты Главы 3. Каждый этап
кэширует результат, поэтому повторный запуск не пересчитывает готовое (снять
кэш можно флагом --force).

    python run_pipeline.py                      # весь пайплайн (с учётом кэша)
    python run_pipeline.py --force              # пересчитать всё заново
    python run_pipeline.py --stage labeling     # только один этап
    python run_pipeline.py --stage meta_model --force

После пайплайна инструмент используется через recommender:
    python -m src.recommender --csv path/to/dataset.csv
    streamlit run app.py
"""

from __future__ import annotations

import argparse

from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Этапы пайплайна (в порядке зависимостей)                                     #
# --------------------------------------------------------------------------- #
def run_collect(force: bool) -> None:
    """Этап 1: сбор корпуса датасетов из OpenML."""
    from src.data_collection import build_corpus

    manifest = build_corpus(force=force)
    logger.info("Этап collect завершён: %s датасетов в корпусе.", len(manifest))


def run_meta_features(force: bool) -> None:
    """Этап 2: извлечение матрицы мета-признаков."""
    from src.meta_features import build_meta_feature_matrix

    matrix = build_meta_feature_matrix(force=force)
    logger.info("Этап meta_features завершён: матрица %s.", matrix.shape)


def run_labeling(force: bool) -> None:
    """Этап 3: разметка корпуса лучшим методом балансировки."""
    from src.labeling import run_labeling as _run

    _, labels = _run(force=force)
    logger.info("Этап labeling завершён: размечено %s датасетов.", len(labels))


def run_meta_model(force: bool) -> None:
    """Этап 4: обучение и выбор мета-модели (RF vs GB, LODO)."""
    from src.meta_model import train_meta_model

    summary = train_meta_model(force=force)
    logger.info("Этап meta_model завершён: победитель — %s.", summary["winner"])


def run_interpretation(force: bool) -> None:
    """Этап 5: интерпретация SHAP (глобальная/локальная)."""
    from src.interpretation import build_interpretation

    build_interpretation(force=force)
    logger.info("Этап interpretation завершён.")


def run_experiments(force: bool) -> None:
    """Этап 6: эксперименты Главы 3 (сравнение с базлайнами, regret, ablation)."""
    from src.experiments import run_experiments as _run

    # ablation дорогой (пересчёт разметки логрегрессией) — включаем только при --force
    _run(with_ablation=force)
    logger.info("Этап experiments завершён.")


# Порядок важен: каждый этап опирается на артефакты предыдущих.
STAGES: dict[str, callable] = {
    "collect": run_collect,
    "meta_features": run_meta_features,
    "labeling": run_labeling,
    "meta_model": run_meta_model,
    "interpretation": run_interpretation,
    "experiments": run_experiments,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Воспроизводимый пайплайн выбора метода борьбы с дисбалансом (модули 1–7)."
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Какой этап выполнить (по умолчанию — все по порядку).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Игнорировать кэш и пересчитать этап(ы) заново (включает ablation в experiments).",
    )
    args = parser.parse_args()

    set_global_seed()

    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, fn in stages.items():
        logger.info("=== Запуск этапа: %s ===", name)
        fn(force=args.force)
    logger.info("Пайплайн завершён (этап(ы): %s).", ", ".join(stages))


if __name__ == "__main__":
    main()
