import streamlit as st
import requests
import pandas as pd
import plotly.express as px

from minio import Minio
from pathlib import Path
from datetime import datetime
from src.utils.minio_utils import (
    upload_file
)

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://localhost:8000/predict"

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Komentar Politik",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "show_correction" not in st.session_state:
    st.session_state.show_correction = False

# =====================================================
# SAVE FEEDBACK
# =====================================================

def save_feedback(
    text_value,
    sentiment_label
):

    Path("feedback").mkdir(
        exist_ok=True
    )

    dataset_path = (
        "feedback/feedback_dataset.csv"
    )

    row = pd.DataFrame([{
        "clean_text": text_value,
        "sentiment": sentiment_label
    }])

    if Path(dataset_path).exists():

        row.to_csv(
            dataset_path,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            dataset_path,
            index=False
        )

    upload_file(
        dataset_path,
        "feedback/feedback_dataset.csv"
    )

    return dataset_path

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ MLOps Dashboard")

    st.success("🟢 FastAPI Running")

    st.markdown("---")

    st.subheader("Model")
    st.code("Sentiment Classifier v1")

    st.subheader("Services")

    st.write("FastAPI")
    st.code("localhost:8000")

    st.write("MLflow")
    st.code("localhost:5000")

    st.write("MinIO")
    st.code("localhost:9001")

    st.markdown("---")

    st.info(
        """
        Feedback pengguna
        akan digunakan
        untuk retraining model.
        """
    )

# =====================================================
# HEADER
# =====================================================

st.title("📊 Analisis Sentimen Komentar Politik")

st.markdown(
    """
Dashboard analisis sentimen berbasis Machine Learning.

Kategori sentimen:

- 😊 Positive
- 😐 Neutral
- 😠 Negative

Feedback pengguna akan digunakan
untuk continuous improvement model.
"""
)

# =====================================================
# METRICS
# =====================================================

positive_count = sum(
    1 for h in st.session_state.history
    if h["sentiment"] == "positive"
)

neutral_count = sum(
    1 for h in st.session_state.history
    if h["sentiment"] == "neutral"
)

negative_count = sum(
    1 for h in st.session_state.history
    if h["sentiment"] == "negative"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("😊 Positive", positive_count)

with col2:
    st.metric("😐 Neutral", neutral_count)

with col3:
    st.metric("😠 Negative", negative_count)

# =====================================================
# FEEDBACK METRICS
# =====================================================

correct_count = sum(
    1 for f in st.session_state.feedbacks
    if f["is_correct"]
)

incorrect_count = sum(
    1 for f in st.session_state.feedbacks
    if not f["is_correct"]
)

col4, col5 = st.columns(2)

with col4:
    st.metric(
        "✅ Correct Feedback",
        correct_count
    )

with col5:
    st.metric(
        "❌ Corrected Feedback",
        incorrect_count
    )

# =====================================================
# CHART
# =====================================================

st.subheader("📈 Distribusi Sentimen")

chart_df = pd.DataFrame({
    "Sentiment": ["Positive", "Neutral", "Negative"],
    "Count": [
        positive_count,
        neutral_count,
        negative_count
    ]
})

fig = px.pie(
    chart_df,
    names="Sentiment",
    values="Count",
    hole=0.5
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================
# INPUT
# =====================================================

st.subheader("✍️ Input Komentar")

text = st.text_area(
    "Masukkan komentar",
    height=150,
    placeholder="Contoh: Program pemerintah ini sangat membantu masyarakat"
)

# =====================================================
# PREDICTION
# =====================================================

if st.button("🔍 Analisis Sentimen"):

    if not text.strip():
        st.warning("Masukkan teks terlebih dahulu.")

    else:

        try:

            response = requests.post(
                API_URL,
                json={"text": text},
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

            sentiment = result.get(
                "sentiment",
                "unknown"
            )

            prediction_record = {
                "text": text,
                "sentiment": sentiment
            }

            st.session_state.history.append(
                prediction_record
            )

            st.session_state.last_prediction = (
                prediction_record
            )

            st.session_state.show_correction = False

            st.rerun()

        except Exception as e:

            st.error(
                f"Gagal menghubungi API: {e}"
            )

# =====================================================
# LAST PREDICTION
# =====================================================

if st.session_state.last_prediction:

    prediction = st.session_state.last_prediction

    sentiment = prediction["sentiment"]
    text_value = prediction["text"]

    st.divider()

    st.subheader("🎯 Hasil Prediksi")

    if sentiment == "positive":
        st.success(
            f"😊 SENTIMEN : {sentiment.upper()}"
        )

    elif sentiment == "negative":
        st.error(
            f"😠 SENTIMEN : {sentiment.upper()}"
        )

    else:
        st.info(
            f"😐 SENTIMEN : {sentiment.upper()}"
        )

    # =====================================
    # FEEDBACK SECTION
    # =====================================

    st.divider()

    st.subheader("📝 Evaluasi Prediksi")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "👍 Prediksi Benar",
            use_container_width=True
        ):

            feedback_record = {
              "clean_text": text_value,
              "sentiment": sentiment,
              "is_correct": True
            }

            st.session_state.feedbacks.append(
                feedback_record
            )

            filename = save_feedback(
              text_value,
              sentiment
            )

            st.success(
                f"Feedback tersimpan:\n{filename}"
            )

    with col2:

        if st.button(
            "👎 Koreksi Prediksi",
            use_container_width=True
        ):
            st.session_state.show_correction = True

    # =====================================
    # CORRECTION FORM
    # =====================================

    if st.session_state.show_correction:

        st.warning(
            "Pilih label yang benar."
        )

        correct_label = st.selectbox(
            "Label yang benar",
            [
                "positive",
                "neutral",
                "negative"
            ]
        )

        confirm = st.checkbox(
            "Saya yakin label ini benar"
        )

        if st.button(
            "💾 Simpan Koreksi"
        ):

            if not confirm:

                st.error(
                    "Centang konfirmasi terlebih dahulu."
                )

            else:

                feedback_record = {
                  "clean_text": text_value,
                  "sentiment": correct_label,
                  "is_correct": False
                }

                st.session_state.feedbacks.append(
                    feedback_record
                )

                filename = save_feedback(
                    text_value,
                    correct_label
                )

                st.success(
                    f"Koreksi disimpan: {filename}"
                )

                st.session_state.show_correction = False

# =====================================================
# HISTORY
# =====================================================

st.divider()

st.subheader("📜 Riwayat Prediksi")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info(
        "Belum ada prediksi."
    )

# =====================================================
# FEEDBACK HISTORY
# =====================================================

st.divider()

st.subheader("🗂 Riwayat Feedback")

if st.session_state.feedbacks:

    feedback_df = pd.DataFrame(
        st.session_state.feedbacks
    )

    st.dataframe(
        feedback_df,
        use_container_width=True
    )

else:

    st.info(
        "Belum ada feedback."
    )
