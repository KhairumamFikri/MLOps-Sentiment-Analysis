import os
import pandas as pd
import joblib

# =========================
# LOAD MODEL & VECTORIZER
# =========================

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("Model berhasil dimuat.")

# =========================
# FEEDBACK FILE
# =========================

FEEDBACK_FILE = "data/feedback/feedback.csv"

os.makedirs("data/feedback", exist_ok=True)

# =========================
# LOOP INFERENCE
# =========================

while True:

    text = input(
        "\nMasukkan kalimat (ketik 'exit' untuk keluar): "
    )

    if text.lower() == "exit":
        break

    # =========================
    # PREDICT
    # =========================

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    print(f"Prediksi Sentiment: {prediction}")

    # =========================
    # FEEDBACK
    # =========================

    feedback = input(
        "Apakah prediksi benar? (y/n): "
    ).lower()

    if feedback == "n":

        correct_label = input(
            "Masukkan label benar (positive/neutral/negative): "
        ).lower()

        # Validasi label
        if correct_label not in [
            "positive",
            "neutral",
            "negative"
        ]:

            print("Label tidak valid!")
            continue

        # Simpan feedback
        feedback_data = pd.DataFrame([{
            "clean_text": text,
            "sentiment": correct_label
        }])

        # Jika file sudah ada → append
        if os.path.exists(FEEDBACK_FILE):

            feedback_data.to_csv(
                FEEDBACK_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            feedback_data.to_csv(
                FEEDBACK_FILE,
                index=False
            )

        print("Feedback berhasil disimpan.")

    else:

        print("Feedback diterima.")