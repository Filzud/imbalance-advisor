"""
Streamlit-демо инструмента (модуль 6) — для показа прототипа на защите.

Запуск:
    streamlit run app.py

Экран: загрузка датасета (CSV или id OpenML) → рекомендованный метод борьбы с
дисбалансом → вероятности по всем 6 методам → SHAP-объяснение (текст + waterfall).
Вся логика — в src/recommender.py; здесь только тонкий UI-слой.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.interpretation import load_model_bundle
from src.recommender import RecommenderError, recommend

st.set_page_config(page_title="Выбор метода борьбы с дисбалансом", page_icon="⚖️", layout="centered")

st.title("⚖️ Интерпретируемый выбор метода борьбы с дисбалансом классов")
st.caption(
    "Загрузите бинарный несбалансированный датасет — система посчитает мета-признаки, "
    "порекомендует метод балансировки и объяснит выбор через SHAP."
)


@st.cache_resource
def _bundle() -> dict:
    """Загрузить мета-модель один раз на сессию."""
    return load_model_bundle()


def _run(source, target_column: str | None) -> None:
    """Выполнить рекомендацию и отрисовать результат."""
    try:
        with st.spinner("Считаю мета-признаки, предсказываю и объясняю…"):
            result = recommend(source, target_column=target_column, bundle=_bundle())
    except RecommenderError as exc:
        st.error(f"⚠️ {exc}")
        return

    st.success(f"### Рекомендованный метод: **{result['recommended_method']}**")

    st.subheader("Вероятности по методам")
    proba = pd.Series(result["probabilities"], name="вероятность").sort_values()
    st.bar_chart(proba)

    exp = result["explanation"]
    if exp:
        st.subheader("Объяснение выбора (SHAP)")
        st.write(exp["text"])
        if exp.get("waterfall_path") and Path(exp["waterfall_path"]).exists():
            st.image(exp["waterfall_path"],
                     caption="Вклад мета-признаков в рекомендацию (waterfall)")


# --------------------------------------------------------------------------- #
# Ввод данных                                                                 #
# --------------------------------------------------------------------------- #
mode = st.radio("Источник датасета", ["CSV-файл", "OpenML id"], horizontal=True)

if mode == "CSV-файл":
    uploaded = st.file_uploader("CSV с бинарным датасетом", type=["csv"])
    if uploaded is not None:
        df_head = pd.read_csv(uploaded, nrows=5)
        uploaded.seek(0)
        target_column = st.selectbox(
            "Столбец-таргет", list(df_head.columns),
            index=len(df_head.columns) - 1,
        )
        st.dataframe(df_head, use_container_width=True)
        if st.button("Получить рекомендацию", type="primary"):
            with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            _run(tmp_path, target_column)
else:
    did = st.number_input("id датасета OpenML", min_value=1, value=310, step=1)
    if st.button("Получить рекомендацию", type="primary"):
        _run(int(did), None)

st.divider()
st.caption("Магистерская ВКР, РУДН · мета-обучение + SHAP · модель: "
           f"{_bundle().get('model_name', '?')}")
